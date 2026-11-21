"""The attempt loop: a plain conversation with one tool.

One attempt:

1. Start a conversation with the system prompt and the task message.
2. Call the model with the single tool available.
3. If the reply has tool calls, apply each to the canvas, return each result,
   and go back to 2.
4. If the reply has no tool call and every row is stored, stop: complete.
5. If the reply has no tool call and rows are missing, send one message that
   lists them and go back to 2. At most `max_nudges` of these.
6. Stop in any case after `max_steps` model calls or `max_minutes`.

The rows decide the state. An attempt with every row stored is complete
however its conversation ended: at a limit, or with the call for the closing
reply failing or being refused by the provider. The stop reason then says
that the closing reply was lost, and the error or the refusal is recorded.

A refusal is recorded and not re-prompted. The provider can signal one. In
text, one is recognised only before the first write_rows call, in a short
reply that declines the request itself and says nothing of carrying on. A
doubtful reply gets the ordinary nudge instead: a model wrongly marked as
refusing is never judged, and that is the costly mistake.

A nudge answers any reply that has no tool call while rows are missing,
whether the model stopped early or the reply was cut off by the output limit.
The second kind is counted apart, in `nudges_after_cutoff`, so the nudge
figure can be read as models stopping early.

A failed API call is retried with backoff. No wait is longer than
`backoff_max_seconds`, whatever the provider asks for, and none is begun that
would end after the conversation's time limit. If the conversation cannot
continue it is discarded, recorded, and the attempt starts again from a fresh
conversation with its own limits. Nothing from a discarded conversation is
carried over.

A call that fails when there is no time left to wait and try it again is the
provider's failure, not the model running out of time. The conversation is
discarded and the attempt ends as an error worth retrying, with the provider's
message. No fresh conversation is started for it: one begun after the time
limit has run out could spend up to another whole limit.

Running an attempt first empties its directory, so nothing from an earlier
run of it is left beside the new files. stats.json is removed first and
written last: its presence marks the attempt as done. The record a run
replaces is not lost: the new stats.json lists it under `earlier`, with the
records that one replaced, oldest first. So an attempt made again after an
error or a provider block, or with `--force`, still shows what came before.
While the new run is in flight the old record waits in earlier.json, so a
run that is killed does not lose it either.

A run makes the attempts that `attempt_wanted` says are still to do, several
at once: each provider up to its `concurrency` and each model up to its own.
An attempt on disk is made again only when asked: `retry_errors` for one that
ended in an error, `retry_refused` for one the provider blocked. A refusal in
the model's own words is its answer, and only `force` asks again.
An interrupt lets the attempts in flight finish and be saved, starts no other,
and then reaches the caller. A model whose provider turns its calls away
outright is stopped for the rest of the run, with nothing written for the
attempts not started, so a later run makes them.
"""

from __future__ import annotations

import os
import random
import re
import shutil
import time
import traceback
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from typing import Any, Callable

from . import __version__
from .adapters import Adapter, AdapterError, get_adapter
from .canvas import Canvas
from .config import RunConfig
from .layout import (
    STATE_COMPLETE,
    STATE_ERROR,
    STATE_INCOMPLETE,
    STATE_REFUSED,
    AttemptRef,
    RunLayout,
    read_json,
    write_json,
    write_text,
)
from .paintings import Painting, load_painting
from .prompts import TOOL_NAME, system_prompt, task_message, write_rows_tool
from .render import render_picture
from .types import STOP_MAX_TOKENS, STOP_REFUSAL, Message, ModelSettings, Reply, system, tool_result, user
from .workers import Watch

# A reply counts as a refusal in text only when the model has never called the
# tool, the reply is short, it declines the request itself, and it does not go
# on to say it will try. Provider-signalled refusals do not need this.
#
# "Declines the request itself" is kept narrow on purpose. "I can't" followed
# by help, comply, take part or do this is a refusal. "I can't" followed by
# see, look anything up, be sure, or draw the boats in detail is a model
# talking about its limits before it types, and so is an apology on its own.
_CANNOT = (
    r"\bi(?:\s+am|['’]m)?\s+(?:can(?:no|['’])t|can\s+not|won['’]t|will\s+not|unable\s+to|not\s+able\s+to|"
    r"not\s+going\s+to)\s+(?:be\s+able\s+to\s+)?"
)
_CLAUSE_END = r"\s*(?=[.!?,;:]|$)"
# Declining the request: "help with ...", "do this." - but not "help but simplify" or "do this without counting".
_THE_REQUEST = (
    r"(?:(?:help|assist)(?:\s+you)?(?:\s+with\b|" + _CLAUSE_END + r")|comply|fulfil+|take\s+part|participate"
    r"|(?:proceed|continue|do\s+(?:this|that|so|it))(?:\s+with\s+(?:this|that|it))?(?:\s+(?:task|request|test))?"
    + _CLAUSE_END
    + ")"
)
# Declining to make the picture as a whole: "reproduce this painting" - but not a part of it ("the boats"), and
# not when the sentence goes on to qualify it ("... exactly", "... in one call").
_THE_WORK = (
    r"(?:create|produce|reproduce|recreate|generate|provide)\s+"
    r"(?:(?:this|that)" + _CLAUSE_END + r"|(?:(?:this|that|the|such)\s+(?:[\w'’-]+\s+){0,3}?(?:painting|artwork|picture|image|print)"
    r"|(?:this|that|the|such|any|an?)\s+(?:[\w'’-]+\s+){0,3}?(?:reproduction|copy))\b)"
    r"(?![^.!?\n]{0,60}\b(?:exactly|perfectly|precisely|accurately|faithfully|pixel|every|detail|only|at\s+once|in\s+one|"
    r"without|unless|until|so)\b)"
)
_DECLINE = re.compile(
    _CANNOT + "(?:" + _THE_REQUEST + "|" + _THE_WORK + r")|\bi\s+(?:must|have\s+to|need\s+to)\s+decline\b|\bi\s+decline\b",
    re.IGNORECASE,
)
# Saying it will try, or already talking about the plan, the palette or the rows.
_CARRIES_ON = re.compile(
    r"\b(?:i['’]ll|i\s+will(?!\s+not\b)|let\s+me(?!\s+know\b)|i\s+can\s+(?:still|try|attempt)|do\s+my\s+best|here\s+is|"
    r"here['’]s(?!\s+what\s+i\s+can\b)|plan|palette|rows?)\b",
    re.IGNORECASE,
)
_TEXT_REFUSAL_MAX_CHARS = 1200


def refusal_phrase(text: str) -> str | None:
    """The declining phrase that makes a text-only reply a refusal, or None if the reply is not one."""
    text = (text or "").strip()
    if not text or len(text) > _TEXT_REFUSAL_MAX_CHARS:
        return None
    declined = _DECLINE.search(text)
    if declined is None or _CARRIES_ON.search(text):
        return None
    return declined.group(0)


def looks_like_refusal(text: str) -> bool:
    return refusal_phrase(text) is not None


class ConversationFailed(Exception):
    """The conversation cannot continue; the attempt restarts from a fresh one, unless `restart` is false."""

    def __init__(self, message: str, retryable: bool = False, restart: bool = True):
        super().__init__(message)
        self.retryable = retryable  # a failure of a kind worth retrying, which outlasted the retries or the time
        self.restart = restart  # false: the conversation's time is used up, so the attempt ends here


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Where the record a run replaces is held while the run is in flight.
_EARLIER_FILE = "earlier.json"


def _make_room(ref: AttemptRef) -> list[dict[str, Any]]:
    """Empty the attempt's directory for a new run of it. Returns the stats records of the runs this one
    replaces, oldest first, each without its own "earlier" list.

    stats.json goes first, so that its presence still means "done" if the new run is killed half way. It is
    renamed to earlier.json, not deleted, and that file stays until the new stats.json is written. A run
    killed half way therefore does not lose the record it was replacing: the next run finds the file.
    """
    held = ref.directory / _EARLIER_FILE
    if ref.stats_path.is_file():
        os.replace(ref.stats_path, held)
    earlier: list[dict[str, Any]] = []
    if held.is_file():
        try:
            replaced = read_json(held)
        except ValueError:  # not a record: there is nothing to carry, and that must not stop the attempt being made again
            replaced = None
        if isinstance(replaced, dict):
            before = replaced.pop("earlier", None)
            earlier = [*(before if isinstance(before, list) else []), replaced]
    if ref.directory.exists():
        for left in ref.directory.iterdir():
            if left == held:
                continue
            if left.is_dir() and not left.is_symlink():
                shutil.rmtree(left)
            else:
                left.unlink()
    ref.directory.mkdir(parents=True, exist_ok=True)
    return earlier


class _Conversation:
    """One fresh conversation: its canvas, its messages and its counters."""

    def __init__(self, painting: Painting, width: int, height: int, cfg: RunConfig):
        self.canvas = Canvas(width, height, cfg.encoding, cfg.strip)
        self.tool = write_rows_tool(cfg.encoding)
        task = task_message(painting.first_line, width, height, cfg.strip, cfg.encoding, cfg.limits.max_steps)
        self.messages: list[Message] = [system(system_prompt(cfg.encoding)), user(task)]
        self.calls: list[dict[str, Any]] = []
        self.model_calls = 0
        self.tool_calls = 0
        self.nudges = 0
        self.nudges_after_cutoff = 0  # of those, the ones that answered a reply cut off by the output limit
        self.truncated_replies = 0
        self.api_retries = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens: int | None = None
        self.stop = ""
        self.refusal_text: str | None = None
        self.refusal_detected_by: str | None = None
        self.refusal_matched: str | None = None
        self.final_text = ""

    def add_usage(self, reply: Reply) -> None:
        self.input_tokens += reply.usage.input_tokens or 0
        self.output_tokens += reply.usage.output_tokens or 0
        if reply.usage.reasoning_tokens is not None:
            self.reasoning_tokens = (self.reasoning_tokens or 0) + reply.usage.reasoning_tokens

    def transcript(self, extra: dict[str, Any]) -> dict[str, Any]:
        return {
            **extra,
            "tool": {"name": self.tool.name, "description": self.tool.description, "parameters": self.tool.parameters},
            "messages": [m.to_dict() for m in self.messages],
            "calls": self.calls,
        }


def _call_with_retries(
    adapter: Adapter,
    convo: _Conversation,
    settings: ModelSettings,
    cfg: RunConfig,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    deadline: float,
) -> Reply:
    """One model call, retried with backoff. `deadline` is the clock reading at which the conversation's time is up.

    A call that fails with no time left to wait and try again raises ConversationFailed with `restart` false.
    """
    retries = cfg.retries
    for i in range(retries.call_retries + 1):
        try:
            return adapter.complete(convo.messages, [convo.tool], settings)
        except AdapterError as e:
            if not e.retryable or i == retries.call_retries:
                raise ConversationFailed(f"{type(e).__name__}: {e}", retryable=e.retryable) from e
            wait = (e.retry_after or retries.backoff_seconds * 2**i) * (0.75 + random.random() / 2)
            # The cap comes last, so it also holds for a provider that asks for hours.
            wait = min(wait, retries.backoff_max_seconds)
            if clock() + wait > deadline:
                # The provider failed and the time is used up. Recording that as the model reaching its time
                # limit would hide the failure and count it against the model, so it is an error to retry.
                raise ConversationFailed(
                    f"{type(e).__name__}: {e} (not tried again: the wait before another try would end after the "
                    f"conversation's time limit of {cfg.limits.max_minutes:g} minutes)",
                    retryable=True,
                    restart=False,
                ) from e
            convo.api_retries += 1
            sleep(wait)
        except Exception as e:  # an adapter bug or an SDK error the adapter did not classify
            raise ConversationFailed(f"{type(e).__name__}: {e}\n{traceback.format_exc()}") from e
    raise AssertionError("unreachable")


def _converse(
    adapter: Adapter,
    convo: _Conversation,
    settings: ModelSettings,
    cfg: RunConfig,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    """Run the loop on a fresh conversation until it stops. Sets convo.stop."""
    canvas, limits = convo.canvas, cfg.limits
    deadline = clock() + limits.max_minutes * 60
    convo.stop = "max_steps"
    for _step in range(limits.max_steps):
        if clock() > deadline:
            convo.stop = "max_minutes"
            return
        t0 = clock()
        reply = _call_with_retries(adapter, convo, settings, cfg, clock, sleep, deadline)
        convo.model_calls += 1
        convo.add_usage(reply)
        convo.calls.append(
            {
                "step": convo.model_calls,
                "seconds": round(clock() - t0, 3),
                "stop_reason": reply.stop_reason,
                "raw_stop_reason": reply.raw_stop_reason,
                "usage": reply.usage.to_dict(),
                "tool_calls": len(reply.tool_calls),
                "raw": reply.raw,
            }
        )
        convo.messages.append(reply.as_message())
        convo.final_text = "" if reply.tool_calls else reply.text  # only a reply that calls no tool can be the closing one
        if reply.stop_reason == STOP_MAX_TOKENS:
            convo.truncated_replies += 1

        if reply.stop_reason == STOP_REFUSAL:
            # With every row stored, only the closing reply is lost: the picture stands.
            convo.stop = "closing_call_refused" if canvas.is_complete() else "refused"
            convo.refusal_text = reply.refusal_text or reply.text
            convo.refusal_detected_by = "provider"
            return

        if reply.tool_calls:
            for call in reply.tool_calls:
                convo.tool_calls += 1
                result = canvas.write(call.arguments) if call.name == TOOL_NAME else canvas.reject_unknown_tool()
                convo.messages.append(tool_result(call, result))
            continue

        if canvas.is_complete():
            convo.stop = "finished"
            return
        matched = refusal_phrase(reply.text) if canvas.write_calls == 0 else None
        if matched:
            convo.stop = "refused"
            convo.refusal_text = reply.text
            convo.refusal_detected_by = "text"
            convo.refusal_matched = matched
            return
        if convo.nudges == limits.max_nudges:
            convo.stop = "nudges_exhausted"
            return
        convo.nudges += 1
        if reply.stop_reason == STOP_MAX_TOKENS:  # cut off before it could call the tool: not a model stopping early
            convo.nudges_after_cutoff += 1
        convo.messages.append(user(canvas.missing_rows_message()))


def run_attempt(
    ref: AttemptRef,
    painting: Painting,
    settings: ModelSettings,
    cfg: RunConfig,
    adapter: Adapter,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run one attempt and write its files. Returns the stats record."""
    width, height = ref.width, painting.height_for(ref.width)
    started_at, t0 = _now(), clock()
    header = {
        "harness_version": __version__,
        "prompt_version": cfg.prompt_version,
        "painting": painting.slug,
        "width": width,
        "height": height,
        "encoding": cfg.encoding,
        "strip_rows": cfg.strip,
        "model": settings.name,
        "attempt": ref.n,
        "model_settings": settings.to_dict(),
    }
    earlier = _make_room(ref)  # nothing of an earlier run is left beside the new files, but its record is kept

    discards: list[dict[str, Any]] = []
    convo = _Conversation(painting, width, height, cfg)
    failure: ConversationFailed | None = None
    closing_error: str | None = None
    for _restart in range(cfg.retries.restarts + 1):
        convo = _Conversation(painting, width, height, cfg)
        try:
            _converse(adapter, convo, settings, cfg, clock, sleep)
            failure = None
            break
        except ConversationFailed as e:
            if convo.canvas.is_complete():  # every row is stored, so only the closing reply was lost
                convo.stop, closing_error, failure = "closing_call_failed", str(e), None
                break
            failure = e
            discards.append(
                {
                    "error": str(e),
                    "model_calls": convo.model_calls,
                    "api_retries": convo.api_retries,
                    "input_tokens": convo.input_tokens,
                    "output_tokens": convo.output_tokens,
                }
            )
            discard_dir = ref.directory / "discards" / str(len(discards))
            write_json(discard_dir / "transcript.json", convo.transcript({**header, "discarded": True, "error": str(e)}))
            write_json(discard_dir / "canvas.json", convo.canvas.state_dict())
            if not e.restart:  # its time is used up: a fresh conversation could spend up to another whole limit
                break

    canvas = convo.canvas
    error = str(failure) if failure is not None else None
    if error is not None:
        state, stop = STATE_ERROR, "error"
    elif convo.stop == "refused":
        state, stop = STATE_REFUSED, "refused"
    else:
        state, stop = (STATE_COMPLETE if canvas.is_complete() else STATE_INCOMPLETE), convo.stop

    kept_in = 0 if error is not None else convo.input_tokens
    kept_out = 0 if error is not None else convo.output_tokens
    all_in = sum(d["input_tokens"] for d in discards) + kept_in
    all_out = sum(d["output_tokens"] for d in discards) + kept_out
    stats = {
        **header,
        "state": state,
        "stop": stop,
        "protocol": "tools",
        "model_calls": convo.model_calls,
        "tool_calls": convo.tool_calls,
        "nudges": convo.nudges,
        "nudges_after_cutoff": convo.nudges_after_cutoff,  # of the nudges, those that answered a reply cut off by the output limit
        "truncated_replies": convo.truncated_replies,
        **canvas.stats(),
        "input_tokens": kept_in,  # the kept conversation only
        "output_tokens": kept_out,
        "reasoning_tokens": convo.reasoning_tokens if error is None else None,
        "discarded_input_tokens": all_in - kept_in,
        "discarded_output_tokens": all_out - kept_out,
        "discarded_model_calls": sum(d["model_calls"] for d in discards),
        "wall_seconds": round(clock() - t0, 3),
        "cost_usd": settings.cost_usd(all_in, all_out),  # includes discarded conversations
        "api_retries": sum(d["api_retries"] for d in discards) + (convo.api_retries if error is None else 0),
        "discards": len(discards),
        "discard_errors": [d["error"].splitlines()[0] for d in discards],
        "refusal_text": convo.refusal_text,
        "refusal_detected_by": convo.refusal_detected_by,
        "refusal_matched": convo.refusal_matched,  # the phrase that made a text reply a refusal
        "closing_error": closing_error,  # why the closing reply was lost, for stop "closing_call_failed"
        "error": error,
        "error_retryable": failure.retryable if failure is not None else None,  # false: turned away outright
        "final_text": convo.final_text,
        "started_at": started_at,
        "finished_at": _now(),
        "earlier": earlier,  # the records of the runs of this attempt that this one replaces, oldest first
    }

    if error is None:
        write_json(ref.transcript_path, convo.transcript(header))
        write_json(ref.canvas_json, canvas.state_dict())
    if state == STATE_COMPLETE:
        picture = canvas.picture(
            comments=[
                f"{painting.first_line} {width}x{height}, typed from memory.",
                f"model {settings.name} ({settings.model}), attempt {ref.n}, paintbench {__version__}, prompt {cfg.prompt_version}",
            ]
        )
        write_text(ref.picture_txt, picture.to_text())
        render_picture(picture).save(ref.picture_png, format="PNG")
    write_json(ref.stats_path, stats)  # written last: its presence marks the attempt as done
    (ref.directory / _EARLIER_FILE).unlink(missing_ok=True)  # what it held is in stats.json now
    return stats


def _blocked_by_provider(stats: dict[str, Any]) -> bool:
    """The attempt ended refused on the provider's signal: a platform block, not a model declining in words."""
    return stats.get("state") == STATE_REFUSED and stats.get("refusal_detected_by") == "provider"


def attempt_wanted(ref: AttemptRef, force: bool = False, retry_errors: bool = False, retry_refused: bool = False) -> bool:
    """The single rule for whether an attempt is run: it has no stats.json; or `force`; or `retry_errors`
    and it ended in an error; or `retry_refused` and it ended refused on the provider's signal.

    A refusal in the model's own words is the model's answer: only `force` runs that attempt again. A
    provider block is never run again unasked, since each one is a request the platform flagged."""
    if force or not ref.stats_path.is_file():
        return True
    stats = ref.stats() or {}
    if retry_errors and stats.get("state") == STATE_ERROR:
        return True
    return retry_refused and _blocked_by_provider(stats)


def _say(line: str) -> None:
    print(line, flush=True)  # flushed, so that a log file or a pipe shows each line as it happens


def run_attempts(
    cfg: RunConfig,
    layout: RunLayout,
    jobs: list[tuple[str, int, str, int]] | None = None,
    adapter_factory: Callable[[ModelSettings], Adapter] | None = None,
    force: bool = False,
    progress: Callable[[str], None] = _say,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    retry_errors: bool = False,
    retry_refused: bool = False,
) -> list[dict[str, Any]]:
    """Run every attempt the config asks for that `attempt_wanted` says is still to do.

    Attempts are independent, so they run concurrently: each provider up to
    its `concurrency`, and each model up to its own where it has one. One
    adapter is shared by all attempts of a model.

    An attempt that ends in an error stays as it is on a later run unless
    `retry_errors` is set, and one that the provider blocked unless
    `retry_refused` is set. A model whose attempt ends in an error with no call
    answered, the last failure being one that is not worth retrying (bad
    credentials, an unknown model, a bad request), is stopped: no further
    attempt of it is started and nothing is written for those.

    An interrupt (Ctrl-C) lets the attempts in flight finish and be saved,
    starts no other, and is then raised to the caller. An attempt that
    crashes outside its model calls is reported and the others carry on; it
    gets no stats.json, so a later run makes it again.
    """
    cfg.check_attempts_can_be_made()
    jobs = cfg.attempt_jobs() if jobs is None else jobs
    paintings = {slug: load_painting(slug, cfg.paintings_path) for slug in {j[0] for j in jobs}}
    for painting, width, _model, _n in jobs:
        paintings[painting].height_for(width)  # fail before any call if a height is missing
    factory = adapter_factory or (lambda s: get_adapter(s.provider))
    adapters: dict[str, Adapter] = {}
    waiting: list[tuple[AttemptRef, ModelSettings]] = []
    for painting, width, model_name, n in jobs:
        settings = cfg.model(model_name)
        ref = layout.attempt(painting, width, model_name, n)
        if not attempt_wanted(ref, force, retry_errors, retry_refused):
            on_disk = ref.stats() or {}
            hint = ""
            if on_disk.get("state") == STATE_ERROR:
                hint = " (--retry-errors runs it again)"
            elif _blocked_by_provider(on_disk):
                hint = " (blocked by the provider; --retry-refused runs it again)"
            progress(f"skip {ref.id}: already {on_disk.get('state')}{hint}")
            continue
        if model_name not in adapters:
            adapters[model_name] = factory(settings)
        waiting.append((ref, settings))

    results: list[dict[str, Any]] = []
    running: dict[Future, tuple[AttemptRef, ModelSettings]] = {}
    on_provider: Counter[str] = Counter()
    on_model: Counter[str] = Counter()
    stopped: set[str] = set()  # models that are started no more in this run

    def has_room(settings: ModelSettings) -> bool:
        if on_provider[settings.provider] >= cfg.concurrency_for(settings.provider):
            return False
        return settings.concurrency is None or on_model[settings.name] < max(1, settings.concurrency)

    with ExitStack() as stack, Watch() as watch:
        # One pool per provider, so each provider runs at its own limit whatever the others are doing.
        pools = {
            provider: stack.enter_context(ThreadPoolExecutor(max_workers=cfg.concurrency_for(provider)))
            for provider in sorted({settings.provider for _ref, settings in waiting})
        }

        def begin(*attempt: Any) -> dict[str, Any] | None:
            """Run in the worker. An attempt handed out in the same instant as a stop was asked for is not
            begun: nothing is sent, nothing is written, and None says so."""
            return None if watch.stop_seen else run_attempt(*attempt)

        try:
            while waiting or running:
                # An attempt goes to its pool only when its provider and its model both have a free place, so
                # none ever waits inside a pool. An interrupt therefore leaves nothing queued to start later,
                # and a model at its own limit does not sit on places that its provider's other models could use.
                for job in list(waiting):
                    watch.checkpoint()  # also between two attempts handed out: a stop starts none of the rest
                    ref, settings = job
                    if has_room(settings):
                        waiting.remove(job)
                        on_provider[settings.provider] += 1
                        on_model[settings.name] += 1
                        attempt = (ref, paintings[ref.painting], settings, cfg, adapters[settings.name], clock, sleep)
                        running[pools[settings.provider].submit(begin, *attempt)] = job
                        progress(f"start {ref.id}")
                finished = watch.first_finished(running)  # in short waits, and a stop asked for is taken there
                for future in finished:
                    ref, settings = running.pop(future)
                    on_provider[settings.provider] -= 1
                    on_model[settings.name] -= 1
                    try:
                        stats = future.result()
                    except Exception as e:  # not a failed model call (those end as "error"): a fault in writing the files, say
                        progress(
                            f"CRASH {ref.id}: {type(e).__name__}: {e}. It has no stats.json, so a later run makes it again.\n"
                            + "".join(traceback.format_exception(type(e), e, e.__traceback__)).rstrip()
                        )
                        continue
                    if stats is None:  # not begun: a stop was asked for as it was handed out
                        progress(f"not begun {ref.id}: handed out as the stop came. Nothing was sent or written for it.")
                        continue
                    results.append(stats)
                    progress(
                        f"done  {ref.id}: {stats['state']} ({stats['stop']}), {stats['model_calls']} calls, "
                        f"{stats['rows_rejected']} rows rejected, {stats['nudges']} nudges, {stats['wall_seconds']:.0f}s"
                    )
                    turned_away = (
                        stats["state"] == STATE_ERROR and stats["error_retryable"] is False and stats["discarded_model_calls"] == 0
                    )
                    if turned_away and settings.name not in stopped:
                        # The same would happen to every attempt of this model, each ending as an error within seconds.
                        stopped.add(settings.name)
                        others = [job for job in waiting if job[1].name != settings.name]
                        dropped, waiting = len(waiting) - len(others), others
                        progress(
                            f"STOP  model {settings.name}: no call was answered and the last failure is not one to retry "
                            f"({stats['error'].splitlines()[0]}). {dropped} of its attempts were not started and nothing "
                            "was written for them, so a later run makes them."
                        )
        except BaseException:  # Ctrl-C, above all: start nothing more; leaving the pools waits for the attempts in flight
            progress(
                f"stopping: {len(waiting)} attempts were not started; {len(running)} in flight will finish and be saved"
            )
            raise
        finally:
            # Leaving the pools waits for the attempts in flight too, but in one wait with no end, during
            # which a further signal might not be answered (see workers.py). After an orderly end none is left.
            watch.all_finished(running)
            not_begun = [
                job[0].id
                for future, job in running.items()
                if not future.cancelled() and future.exception() is None and future.result() is None
            ]
            if not_begun:
                progress(f"not begun after all: {', '.join(not_begun)} (handed out as the stop came; nothing was sent or written)")
    return results
