"""The whole pipeline through the command line, with scripted models and a scripted judge. No network."""

from __future__ import annotations

import pytest
import yaml

from paintbench import cli
from paintbench.adapters import fake
from paintbench.layout import RunLayout, read_json

from conftest import ROOT, picture_text


def scripted_judge(name: str, order: list[str]) -> dict:
    return {"name": name, "provider": "fake", "model": "script-judge", "options": {"behaviour": "judge", "order": order}}


def config(tmp_path) -> dict:
    """Two paintings at two widths. One model types a picture it is given in a file, one makes mistakes, one is plain, one declines."""
    (tmp_path / "great-wave-32.txt").write_text(picture_text(32, 22, "strong"), encoding="utf-8")
    return {
        "run_id": "e2e",
        "paintings": ["great-wave", "the-scream"],
        "widths": [16, 32],
        "paintings_dir": str(ROOT / "paintings"),
        "runs_dir": str(tmp_path / "runs"),
        "models": [
            {"name": "strong", "provider": "fake", "model": "script-a",
             "options": {"painting": "great-wave", "pictures": {"great-wave/32": str(tmp_path / "great-wave-32.txt")}}},
            {"name": "middling", "provider": "fake", "model": "script-b",
             "options": {"faults": [{"row": 3, "kind": "long"}, {"row": 7, "kind": "unknown"}], "stop_early": True}},
            {"name": "weak", "provider": "fake", "model": "script-c"},
            {"name": "refuser", "provider": "fake", "model": "script-d", "options": {"refuse": True}},
        ],
        "judges": [scripted_judge("judge", ["strong", "middling", "weak"])],
        "judge": {"repeats": 2},
    }


def write(path, data: dict) -> str:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(path)


def rows(path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]


def models_ranked(record: dict) -> list[str]:
    return [attempt.split("/")[2] for attempt in record["ranking_attempts"]]


@pytest.fixture
def judged_run(tmp_path):
    """A run with its attempts made and judged. Returns (layout, run directory)."""
    rundir = str(tmp_path / "runs" / "e2e")
    assert cli.main(["attempt", "--config", write(tmp_path / "e2e.yaml", config(tmp_path))]) == 0
    assert cli.main(["judge", "--run", rundir]) == 0
    return RunLayout(rundir), rundir


def test_the_whole_pipeline_runs_offline_from_attempt_to_report(judged_run, tmp_path):
    layout, rundir = judged_run

    # attempt: every model at every painting and width; the one that declined is recorded as refused
    assert len(layout.attempts()) == 4 * 2 * 2
    assert {(a.model, a.state()) for a in layout.attempts()} == {
        ("strong", "complete"), ("middling", "complete"), ("weak", "complete"), ("refuser", "refused")}
    typed = layout.attempt("great-wave", 32, "strong", 1)
    assert rows(typed.picture_txt) == rows(tmp_path / "great-wave-32.txt")  # the picture on disk is the one the model typed
    assert typed.picture_png.is_file() and typed.transcript_path.is_file()
    typed.picture_png.unlink()
    assert cli.main(["render", "--run", rundir]) == 0 and typed.picture_png.is_file()  # drawn again from the text
    stats = layout.attempt("great-wave", 32, "middling", 1).stats()
    assert (stats["rows_rejected_first_submission"], stats["nudges"]) == (2, 1)  # two bad rows, and it stopped early once

    # judge: each painting and width ranked twice, in the scripted judge's order; a refusal is not shown to the judge
    records = layout.judgements()
    rankings = [r for r in records if r["kind"] == "ranking"]
    assert len(rankings) == 2 * 2 * 2 and all(r["status"] == "ok" for r in records)
    assert all(models_ranked(r) == ["strong", "middling", "weak"] for r in rankings)
    peaks = [r for r in records if r["kind"] == "peak"]  # each model's two sizes of a painting against each other
    assert len(peaks) == 2 * 3 * 2 and all(len(r["labels"]) == 2 for r in peaks)

    # score: the judge's order is the order of the rank scores, and a model with no picture has none
    assert cli.main(["score", "--run", rundir]) == 0
    scores = read_json(layout.scores_path)
    assert [(m["model"], m["rank_score"]) for m in scores["models"]] == [
        ("strong", 1.0), ("middling", 0.5), ("weak", 0.0), ("refuser", None)]

    # report
    assert cli.main(["report", "--run", rundir]) == 0
    page = (layout.report_dir / "index.html").read_text(encoding="utf-8")
    assert all(text in page for text in ("strong", "middling", "weak", "refuser", "The Great Wave", "The Scream"))


def test_a_saved_run_is_judged_again_by_a_new_judge_without_calling_a_contestant(judged_run, tmp_path, monkeypatch):
    layout, rundir = judged_run
    attempts = {a.id: a.stats_path.stat().st_mtime_ns for a in layout.attempts()}
    earlier = {r["_path"] for r in layout.judgements()}
    called, complete = [], fake.FakeAdapter.complete

    def counted(self, messages, tools, settings):
        called.append(settings.name)
        return complete(self, messages, tools, settings)

    monkeypatch.setattr(fake.FakeAdapter, "complete", counted)
    new_judge = {"run_id": "e2e", "judges": [scripted_judge("judge-new", ["weak", "middling", "strong"])]}  # names no contestant
    assert cli.main(["judge", "--run", rundir, "--config", write(tmp_path / "new-judge.yaml", new_judge)]) == 0

    assert set(called) == {"judge-new"}
    assert {a.id: a.stats_path.stat().st_mtime_ns for a in layout.attempts()} == attempts
    after = layout.judgements()
    new = [r for r in after if r["_path"] not in earlier]
    assert len(after) == 2 * len(earlier) and {r["judge"] for r in new} == {"judge-new"}  # the first judge's records stay
    assert all(models_ranked(r) == ["weak", "middling", "strong"] for r in new if r["kind"] == "ranking")
    # The run's saved config now lists both judges and still lists its contestants, so both judges are scored.
    saved = yaml.safe_load(layout.config_path.read_text(encoding="utf-8"))
    assert [j["name"] for j in saved["judges"]] == ["judge", "judge-new"] and len(saved["models"]) == 4
    assert cli.main(["score", "--run", rundir]) == 0
    assert {m["model"]: m["rank_score"] for m in read_json(layout.scores_path)["models"]} == {
        "strong": 0.5, "middling": 0.5, "weak": 0.5, "refuser": None}  # two judges of opposite opinions
