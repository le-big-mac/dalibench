"""The attempt loop: what a contestant is sent, how an attempt ends, and what a failure or a stop leaves on disk."""

from __future__ import annotations

import threading

import pytest

from paintbench.adapters.base import FatalError, TransientError
from paintbench.adapters.fake import FakeAdapter
from paintbench.canvas import load_picture, reply_line_template
from paintbench.harness import attempt_wanted, run_attempt, run_attempts
from paintbench.layout import RunLayout, read_json
from paintbench.paintings import load_painting
from paintbench.prompts import system_prompt, task_message, write_rows_tool
from paintbench.types import STOP_END, STOP_MAX_TOKENS, STOP_REFUSAL, STOP_TOOL_CALLS, Image, Reply, ToolCall, Usage
from paintbench.workers import ask_stop

K32 = "k" * 32


def calls(arguments) -> Reply:
    """A scripted reply that makes one write_rows call."""
    return Reply(tool_calls=[ToolCall(id="c", name="write_rows", arguments=arguments)], stop_reason=STOP_TOOL_CALLS, usage=Usage(100, 50, 20))


def says(text: str) -> Reply:
    return Reply(text=text, stop_reason=STOP_END, usage=Usage(100, 10, 0))


# A model that types a whole 32 x 22 picture in two strips and then says it is done.
FULL_PICTURE = [
    calls({"palette": ["k 20 40 90"], "start_row": 1, "rows": [K32] * 16}),
    calls({"start_row": 17, "rows": [K32] * 6}),
    says("Sea everywhere. I was unsure about everything."),
]
# The platform's block, as the live API sends it.
BLOCKED = Reply(text="", stop_reason=STOP_REFUSAL, raw_stop_reason="refusal", refusal_text="This request was blocked. (category: reasoning_extraction)")
# A reply that is all thinking and ends at the output limit: no text, no tool call.
CUT_OFF = Reply(text="", stop_reason=STOP_MAX_TOKENS, raw_stop_reason="max_tokens", usage=Usage(100, 128000, 128000))


def run_one(cfg, script, sleep=lambda seconds: None):
    """One attempt at The Great Wave, 32 wide, by a model that answers from `script`. Returns stats, ref, adapter."""
    ref = RunLayout(cfg.run_dir).attempt("great-wave", 32, "typist", 1)
    adapter = FakeAdapter(script=list(script) if isinstance(script, list) else script)
    painting = load_painting("great-wave", cfg.paintings_path)
    stats = run_attempt(ref, painting, cfg.model("typist"), cfg, adapter, sleep=sleep)
    return stats, ref, adapter


def test_every_contestant_request_holds_no_image_exactly_one_tool_and_the_same_prompt_for_every_model(make_config):
    faulty = {"faults": [{"row": 3, "kind": "long"}], "stop_early": True}  # draws a rejection and a reminder in each attempt
    cfg = make_config(
        repeats=2,
        models=[
            {"name": "a", "provider": "fake", "model": "m1", "reasoning": {"effort": "max"}, "options": faulty},
            {"name": "b", "provider": "fake", "model": "m2"},
        ],
    )
    adapters = {}
    results = run_attempts(
        cfg, RunLayout(cfg.run_dir), adapter_factory=lambda s: adapters.setdefault(s.name, FakeAdapter()), progress=lambda line: None
    )
    assert [r["state"] for r in results] == ["complete"] * 4
    assert sum(r["rows_rejected"] for r in results) == sum(r["nudges"] for r in results) == 2
    painting = load_painting("great-wave", cfg.paintings_path)
    prompt = [("system", system_prompt()), ("user", task_message(painting.first_line, 32, 22, 16))]
    requests = [request for adapter in adapters.values() for request in adapter.requests]
    for request in requests:
        assert request["tools"] == [write_rows_tool()]
        assert [(m.role, m.text) for m in request["messages"][:2]] == prompt
        assert not any(isinstance(part, Image) for m in request["messages"] for part in m.content)
        feedback = [m for m in request["messages"][2:] if m.role != "assistant"]  # tool results and reminders
        assert all(reply_line_template(line) for m in feedback for line in m.text.splitlines())  # format only
    assert sum(len(request["messages"]) == 2 for request in requests) == 4  # each attempt starts from the prompt alone


def test_an_attempt_that_stores_every_row_and_then_stops_is_complete_and_leaves_its_files(make_config):
    priced = {"name": "typist", "provider": "fake", "model": "fake-1", "price": {"input_per_mtok": 2.0, "output_per_mtok": 10.0}}
    stats, ref, _ = run_one(make_config(models=[priced]), FULL_PICTURE)
    assert (stats["state"], stats["stop"]) == ("complete", "finished")
    assert (stats["model_calls"], stats["tool_calls"], stats["nudges"], stats["rows_rejected"]) == (3, 2, 0, 0)
    assert (stats["input_tokens"], stats["output_tokens"], stats["cost_usd"]) == (300, 110, (300 * 2.0 + 110 * 10.0) / 1e6)
    assert stats["final_text"].startswith("Sea everywhere")
    assert read_json(ref.stats_path) == stats
    picture = load_picture(ref.picture_txt)
    assert picture.rows == [K32] * 22 and picture.palette == {"k": (20, 40, 90)}
    assert ref.picture_png.is_file()
    transcript = read_json(ref.transcript_path)
    assert [m["role"] for m in transcript["messages"]] == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert transcript["messages"][2]["tool_calls"][0]["arguments"]["rows"][0] == K32  # what the model typed is logged


@pytest.mark.parametrize(
    "third_reply, stop",
    [
        (BLOCKED, "closing_call_refused"),
        (FatalError("400 bad request"), "closing_call_failed"),
        (calls({"start_row": 1, "rows": [K32]}), "max_steps"),
    ],
)
def test_an_attempt_with_every_row_stored_is_complete_however_its_conversation_ended(third_reply, stop, make_config):
    stats, ref, adapter = run_one(make_config(limits={"max_steps": 3}), FULL_PICTURE[:2] + [third_reply])
    assert (stats["state"], stats["stop"], stats["error"], stats["discards"]) == ("complete", stop, None, 0)
    assert len(adapter.requests) == 3  # nothing was typed again
    assert load_picture(ref.picture_txt).height == 22 and ref.picture_png.is_file()


def test_an_attempt_with_rows_missing_ends_incomplete_after_three_reminders_or_at_the_step_limit(make_config):
    stats, ref, adapter = run_one(make_config(), [FULL_PICTURE[0]] + [says("That is all.")] * 10)
    assert (stats["state"], stats["stop"], stats["nudges"]) == ("incomplete", "nudges_exhausted", 3)
    assert len(adapter.requests) == 5
    reminder = adapter.requests[2]["messages"][-1]
    assert (reminder.role, reminder.text) == ("user", "The picture is not finished. Missing rows: 17-22. Send them with write_rows.")
    assert not ref.picture_txt.exists() and not ref.picture_png.exists()  # only a complete attempt gets a picture
    assert read_json(ref.canvas_json)["missing"] == [17, 18, 19, 20, 21, 22]

    one_row_for_ever = lambda messages, tools, settings: calls({"palette": ["k 1 2 3"], "start_row": 1, "rows": [K32]})
    stats, ref, adapter = run_one(make_config(limits={"max_steps": 5}), one_row_for_ever)
    assert (stats["state"], stats["stop"], len(adapter.requests)) == ("incomplete", "max_steps", 5)


@pytest.mark.parametrize(
    "first_reply, state, detected_by",
    [
        (BLOCKED, "refused", "provider"),
        (says("I'm sorry, but I can't reproduce this painting."), "refused", "text"),
        (says("I can't help with that. Let me know if there's anything else I can help with."), "refused", "text"),
        (says("I can't see the painting, so this is from memory."), "complete", None),
        (says("I can't reproduce this painting, but I'll do my best from memory."), "complete", None),
        (says("I can't reproduce the painting without simplifying it a great deal."), "complete", None),
    ],
)
def test_a_refusal_is_recorded_as_the_providers_block_or_the_models_own_and_a_model_at_work_is_not_taken_for_one(
    first_reply, state, detected_by, make_config
):
    stats, ref, adapter = run_one(make_config(), [first_reply] + FULL_PICTURE)
    assert (stats["state"], stats["refusal_detected_by"]) == (state, detected_by)
    assert len(adapter.requests) == (1 if state == "refused" else 4)  # a refusal is not asked again; a model at work is reminded
    assert ref.picture_txt.exists() == (state == "complete")
    assert not attempt_wanted(ref)  # however it ended, it is not made again unasked
    assert attempt_wanted(ref, retry_refused=True) == (detected_by == "provider")  # the model's own refusal is its answer


def test_a_failed_call_is_retried_after_a_capped_wait_and_one_that_keeps_failing_ends_the_attempt_as_an_error(make_config):
    cfg = make_config(retries={"call_retries": 2, "restarts": 0, "backoff_max_seconds": 60.0})
    rate_limited = TransientError("rate limited", status=429, retry_after=900.0)
    waits = []
    stats, _, _ = run_one(cfg, [rate_limited] * 2 + FULL_PICTURE, sleep=waits.append)
    assert (stats["state"], stats["api_retries"]) == ("complete", 2)
    assert waits == [60.0, 60.0]  # the configured maximum, though the provider asked for fifteen minutes

    stats, ref, adapter = run_one(cfg, [rate_limited] * 3 + FULL_PICTURE)
    assert (stats["state"], stats["stop"], stats["error_retryable"]) == ("error", "error", True)
    assert "rate limited" in stats["error"] and len(adapter.requests) == 3
    assert ref.stats_path.is_file() and not ref.picture_txt.exists()
    assert not attempt_wanted(ref) and attempt_wanted(ref, retry_errors=True)  # run again only when that is asked for


def test_a_conversation_that_cannot_continue_is_discarded_and_the_attempt_starts_again_from_the_prompt_alone(make_config):
    cfg = make_config(retries={"call_retries": 0, "restarts": 2})
    stats, ref, adapter = run_one(cfg, [FULL_PICTURE[0], TransientError("down")] + FULL_PICTURE)
    assert (stats["state"], stats["discards"], stats["discarded_model_calls"]) == ("complete", 1, 1)
    assert [m.role for m in adapter.requests[2]["messages"]] == ["system", "user"]  # nothing is carried over
    assert (stats["model_calls"], stats["tool_calls"], stats["input_tokens"]) == (3, 2, 300)  # the kept conversation only
    assert stats["discarded_input_tokens"] == 100
    discarded = read_json(ref.directory / "discards" / "1" / "transcript.json")
    assert discarded["discarded"] is True and len(discarded["messages"]) == 4  # the prompt, one reply, one tool result
    assert len(read_json(ref.transcript_path)["calls"]) == 3


def test_a_reply_cut_off_by_the_output_limit_is_counted_and_answered_like_any_other(make_config):
    stats, _, adapter = run_one(make_config(), [CUT_OFF] + FULL_PICTURE)  # cut off before any tool call: reminded
    assert (stats["state"], stats["truncated_replies"], stats["nudges"], stats["nudges_after_cutoff"]) == ("complete", 1, 1, 1)
    assert adapter.requests[1]["messages"][-1].text == "The picture is not finished. Missing rows: 1-22. Send them with write_rows."

    half_a_call = Reply(tool_calls=[ToolCall("t", "write_rows", '{"start_row": 1, "rows": ["kkkk')], stop_reason=STOP_MAX_TOKENS)
    stats, _, _ = run_one(make_config(), [half_a_call] + FULL_PICTURE)  # cut off inside a tool call: the call is rejected
    assert (stats["state"], stats["truncated_replies"], stats["calls_rejected"], stats["nudges"]) == ("complete", 1, 1, 0)


def test_a_stop_lets_the_attempt_in_flight_finish_and_be_saved_and_starts_no_other(make_config):
    cfg = make_config(repeats=3, concurrency={"fake": 1})
    layout = RunLayout(cfg.run_dir)
    in_flight, stopping, lines, said_before_the_answer = threading.Event(), threading.Event(), [], []

    class Held(FakeAdapter):  # its first call is answered only once the run has said that it is stopping
        def complete(self, messages, tools, settings):
            if not in_flight.is_set():
                in_flight.set()
                said_before_the_answer.append(stopping.wait(10))
            return super().complete(messages, tools, settings)

    def stop_once_in_flight():
        in_flight.wait(10)
        ask_stop()  # what the command line does with Ctrl-C

    def progress(line):
        lines.append(line)
        if line.startswith("stopping"):
            stopping.set()

    signaller = threading.Thread(target=stop_once_in_flight)
    signaller.start()
    with pytest.raises(KeyboardInterrupt):
        run_attempts(cfg, layout, adapter_factory=lambda s: Held(), progress=progress)
    signaller.join()
    assert said_before_the_answer == [True]  # the stop was taken at once, not when the model answered
    assert [line.split()[0] for line in lines] == ["start", "stopping:"]
    assert [(a.n, a.is_complete()) for a in layout.attempts()] == [(1, True)]  # saved before the run returned; no other begun
    later = run_attempts(cfg, layout, progress=lambda line: None)
    assert sorted(r["attempt"] for r in later) == [2, 3]  # a later run makes the rest, and not the one that was saved
