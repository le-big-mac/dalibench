"""The judge stage: what a judge is sent, how its reply is read, and when a saved judgement is made again."""

from __future__ import annotations

import shutil
import string
from collections import Counter
from itertools import combinations
from pathlib import Path

import pytest
from PIL import Image as PILImage

from paintbench import judge as judge_module
from paintbench.adapters import get_adapter
from paintbench.adapters.fake import FakeAdapter
from paintbench.importer import import_picture
from paintbench.judge import parse_judge_reply, run_judging
from paintbench.layout import RunLayout, read_json, write_json
from paintbench.paintings import load_painting
from paintbench.prompts import JUDGE_RETRY, judge_prompt, judge_prompt_peak
from paintbench.reference import sha256_file
from paintbench.types import Image, Reply, Text

from conftest import ROOT, picture_text

# Six made-up contestants, each with a picture of The Great Wave at 32 x 22. The scripted judge ranks them in this order, best first.
MODELS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
IDS = [f"great-wave/32/{name}/1" for name in MODELS]
THREE = ["gamma", "delta", "epsilon"]


def judge_entry(name: str = "judge-a") -> dict:
    """A scripted judge that ranks the pictures in the order of MODELS, whatever their labels."""
    return {"name": name, "provider": "fake", "model": "script-judge", "options": {"behaviour": "judge", "order": MODELS}}


def typed_run(cfg, models=MODELS) -> RunLayout:
    """A run directory holding a picture by each of these models as a complete attempt."""
    layout = RunLayout(cfg.run_dir)
    painting = load_painting("great-wave", cfg.paintings_path)
    cfg.run_dir.parent.mkdir(parents=True, exist_ok=True)
    for model in models:
        source = cfg.run_dir.parent / f"{model}.txt"  # beside the run, not in it
        source.write_text(picture_text(32, 22, model), encoding="utf-8")
        import_picture(layout, painting, 32, model, 1, source)
    return layout


def judging(cfg, layout, adapter=None, **kwargs) -> list[dict]:
    """Run the judge stage quietly and return its records. With `adapter`, every judge is that adapter."""
    factory = (lambda settings: adapter) if adapter else None
    return run_judging(cfg, layout, adapter_factory=factory, progress=lambda line: None, sleep=lambda seconds: None, **kwargs)


def shown(request: dict, layout: RunLayout) -> list[tuple[str, str]]:
    """(label, attempt id) of each picture in a request, in the order sent: read from places in the first message alone."""
    parts = request["messages"][0].content
    pairs = []
    for text, image in zip(parts[3::2], parts[4::2]):
        assert isinstance(text, Text) and isinstance(image, Image)
        pairs.append((text.text.removeprefix("Picture "), Path(image.path).parent.relative_to(layout.attempts_root).as_posix()))
    return pairs


def replying(*forms: str) -> FakeAdapter:
    """A judge that gives these replies, one each time it is asked, each filled in with the call's labels in the order shown."""
    forms = list(forms)

    def script(messages, tools, settings):
        labels = [part.text.removeprefix("Picture ") for part in messages[0].content[3::2]]
        return Reply(text=forms.pop(0).format(*labels))

    return FakeAdapter(script=script)


def pixel_size(path) -> tuple[int, int]:
    with PILImage.open(path) as image:
        return image.size


def test_a_judge_is_sent_one_message_the_reference_then_every_picture_under_a_fresh_random_letter(make_config):
    cfg = make_config(judges=[judge_entry()], judge={"repeats": 3, "peak_size": False}, concurrency={"fake": 1})
    layout, adapter = typed_run(cfg), FakeAdapter()
    write_json(layout.attempt("great-wave", 32, "unfinished", 1).stats_path, {"state": "incomplete"})  # not judged
    records = judging(cfg, layout, adapter)
    reference = load_painting("great-wave", cfg.paintings_path).reference_path
    views = [shown(request, layout) for request in adapter.requests]  # one call at a time, so in the order of the records

    for request, view, record in zip(adapter.requests, views, records, strict=True):
        [message] = request["messages"]  # no system prompt
        assert message.role == "user" and request["tools"] == []
        assert message.content[:3] == [Text(judge_prompt(6, 32, 22)), Text("REFERENCE"), Image(str(reference))]
        assert len(message.content) == 3 + 2 * 6 and sorted(attempt for _, attempt in view) == sorted(IDS)
        letters = [label for label, _ in view]
        assert len(set(letters)) == 6 and all(len(label) == 1 and label in string.ascii_uppercase for label in letters)
        # The saved key is what was sent, so the judge's letters come back as the right attempts.
        assert record["labels"] == dict(view) and record["ranking_attempts"] == IDS
    for a, b in combinations(views, 2):
        assert dict(a) != dict(b), "the same letters on the same pictures twice"
        assert [label for label, _ in a] != [label for label, _ in b], "the same sequence of letters twice"
        assert [attempt for _, attempt in a] != [attempt for _, attempt in b], "the same order of pictures twice"


GOOD_REPLY = """\
OVERALL: K, G, B
NOTES:
K: a curling wave on the left, three boats, a small mountain.
G: the wave is there but the mountain is missing.
B: sky and sea only.
CONFIDENCE: fairly confident; G and B were the hardest to separate."""


def test_a_reply_in_the_required_form_is_read_into_an_order_with_its_notes_and_confidence():
    parsed = parse_judge_reply(GOOD_REPLY, ["B", "G", "K"])
    assert parsed.ranking == ["K", "G", "B"]
    assert parsed.notes == {
        "K": "a curling wave on the left, three boats, a small mountain.",
        "G": "the wave is there but the mountain is missing.",
        "B": "sky and sea only.",
    }
    assert parsed.confidence == "fairly confident; G and B were the hardest to separate."
    # Markdown on the heading, which judges add unasked, is the same reply.
    assert parse_judge_reply(GOOD_REPLY.replace("OVERALL:", "**OVERALL:**"), ["B", "G", "K"]).ranking == ["K", "G", "B"]


@pytest.mark.parametrize(
    "malformed",
    [
        "OVERALL: {0}, {1}",  # a label left out
        "OVERALL: {0}, {1}, {1}, {2}",  # a label twice
        "OVERALL: {0}, {1} = {2}",  # a tie
        "OVERALL: {0} > {1} > {2}",  # not separated by commas
        "My overall ranking is {0}, {1}, {2}.",  # no OVERALL line
        "OVERALL: {0}, {1}, {2}\nCORRECTED OVERALL: {1}, {0}, {2}",  # a second ranking, which may take the first back
    ],
)
def test_a_malformed_reply_is_asked_for_once_more_then_recorded_as_failed_and_never_guessed(make_config, malformed):
    cfg = make_config(judges=[judge_entry()], judge={"repeats": 2, "peak_size": False}, concurrency={"fake": 1})
    layout = typed_run(cfg, THREE)
    adapter = replying(malformed, malformed, malformed, "OVERALL: {2}, {0}, {1}")  # two replies to each of the two calls
    failed, mended = judging(cfg, layout, adapter)

    assert [len(request["messages"]) for request in adapter.requests] == [1, 3, 1, 3]  # each call asked, then once more only
    opening, reply, again = adapter.requests[1]["messages"]
    assert opening == adapter.requests[0]["messages"][0]  # the same conversation, so the same pictures
    assert (reply.role, again.role, again.images) == ("assistant", "user", [])
    assert again.text == JUDGE_RETRY.format(labels=", ".join(failed["order_shown"]), n=3)
    assert (failed["status"], failed["ranking"], failed["ranking_attempts"]) == ("failed", None, None)
    assert read_json(layout.judgement_dir("great-wave", 32, "judge-a") / "1.json") == failed
    order = mended["order_shown"]
    assert mended["status"] == "ok" and mended["ranking"] == [order[2], order[0], order[1]]  # the second reply, read as it stands


@pytest.mark.parametrize("change", ["an attempt joins the set", "a picture is typed again", "the reference image is replaced"])
def test_a_saved_judgement_is_made_again_only_when_its_set_or_its_reference_has_changed(make_config, tmp_path, change):
    paintings = tmp_path / "paintings"  # the test's own copy of the painting, so that its reference can be replaced
    (paintings / "great-wave").mkdir(parents=True)
    for name in ("meta.yaml", "reference.png"):
        shutil.copy(ROOT / "paintings" / "great-wave" / name, paintings / "great-wave" / name)
    cfg = make_config(judges=[judge_entry()], judge={"repeats": 2, "peak_size": False}, paintings_dir=str(paintings))
    layout = typed_run(cfg, THREE)
    assert len(judging(cfg, layout)) == 2 and judging(cfg, layout) == []  # on disk and current: not made again

    if change == "an attempt joins the set":
        typed_run(cfg, ["beta"])
    elif change == "a picture is typed again":
        layout.attempt("great-wave", 32, "gamma", 1).picture_txt.write_text(picture_text(32, 22, "typed again"), encoding="utf-8")
    else:
        PILImage.new("RGB", (64, 44), "red").save(paintings / "great-wave" / "reference.png")

    records = judging(cfg, layout)
    assert len(records) == 2 and judging(cfg, layout) == []
    for record in records:  # each was made on the attempts and the reference as they are now
        assert record["status"] == "ok"
        assert record["pictures"] == {attempt.id: attempt.picture_sha256() for attempt in layout.complete_attempts()}
        assert record["reference"] == sha256_file(paintings / "great-wave" / "reference.png")


def test_judging_a_saved_run_never_calls_a_contestant_even_for_a_judge_added_later(make_config, monkeypatch):
    contestants = [{"name": name, "provider": "anthropic", "model": "some-model"} for name in MODELS]
    judge = {"repeats": 1, "peak_size": False}
    cfg = make_config(models=contestants, judges=[judge_entry()], judge=judge)
    layout, asked = typed_run(cfg), []

    def watched(provider, client=None):
        asked.append(provider)
        return get_adapter(provider, client)

    monkeypatch.setattr(judge_module, "get_adapter", watched)  # what the stage builds its adapters with when given no factory
    [record] = judging(cfg, layout)
    assert record["status"] == "ok" and asked == ["fake"]  # the judge's provider, and not the contestants'

    both = make_config(models=contestants, judges=[judge_entry(), judge_entry("judge-b")], judge=judge)
    assert [later["judge"] for later in judging(both, layout)] == ["judge-b"]  # the first judge is not asked again
    assert asked == ["fake", "fake"]


def test_a_set_too_large_for_one_call_is_judged_in_pairs_shown_both_ways_round(make_config):
    cfg = make_config(judges=[judge_entry()], judge={"repeats": 2, "max_per_call": 3, "peak_size": False})
    layout, adapter = typed_run(cfg, MODELS[:4]), FakeAdapter()  # four attempts, more than three
    records = judging(cfg, layout, adapter)

    pairs = [frozenset(pair) for pair in combinations(IDS[:4], 2)]
    assert Counter(frozenset(record["labels"].values()) for record in records) == Counter(pairs * 2)  # every pair, twice
    assert all(record["kind"] == "pairwise" and record["status"] == "ok" for record in records)
    assert len(list(layout.pairwise_dir("great-wave", 32, "judge-a").glob("*.json"))) == 12
    assert not list(layout.judgement_dir("great-wave", 32, "judge-a").glob("*.json"))  # and no call ranking all four
    for request in adapter.requests:
        [message] = request["messages"]
        assert message.content[0] == Text(judge_prompt(2, 32, 22)) and len(message.images) == 1 + 2
    for one, two in zip(records[::2], records[1::2]):  # the two calls on a pair
        assert [one["labels"][label] for label in one["order_shown"]] == [two["labels"][label] for label in two["order_shown"]][::-1]
        assert one["ranking_attempts"] == two["ranking_attempts"] == sorted(one["labels"].values(), key=IDS.index)


def test_a_peak_call_shows_one_models_widths_all_drawn_at_one_size(make_config, tmp_path):
    cfg = make_config(paintings=["mona-lisa"], widths=[16, 32, 64], judges=[judge_entry()], judge={"repeats": 1})
    layout, adapter, painting = RunLayout(cfg.run_dir), FakeAdapter(), load_painting("mona-lisa", cfg.paintings_path)
    for model, width in [("alpha", 16), ("alpha", 32), ("alpha", 64), ("beta", 32)]:  # beta has one width: nothing to compare
        source = tmp_path / f"{model}-{width}.txt"
        source.write_text(picture_text(width, painting.height_for(width), f"{model}{width}"), encoding="utf-8")
        import_picture(layout, painting, width, model, 1, source)
    alpha = [f"mona-lisa/{width}/alpha/1" for width in (16, 32, 64)]
    assert len({pixel_size(layout.attempt_from_id(attempt).picture_png) for attempt in alpha}) == 3  # each a size of its own

    [record] = judging(cfg, layout, adapter, kinds=["peak"])
    [request] = adapter.requests
    [message] = request["messages"]
    assert (record["kind"], record["model"], record["status"]) == ("peak", "alpha", "ok")
    assert message.content[:3] == [Text(judge_prompt_peak(3, 16, 64)), Text("REFERENCE"), Image(str(painting.reference_path))]
    assert sorted(attempt for _, attempt in shown(request, layout)) == alpha and len(message.images) == 1 + 3
    assert len({pixel_size(image.path) for image in message.images[1:]}) == 1  # as the prompt tells the judge they are
