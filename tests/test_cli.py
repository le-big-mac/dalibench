"""The command line: what it shows before it spends, and what it refuses to do. Scripted models, no network."""

from __future__ import annotations

import io
import os
import re
import sys

import pytest
import yaml

from paintbench import cli
from paintbench.adapters import fake
from paintbench.layout import RunLayout, run_lock

from conftest import ROOT

REAL = {"name": "real", "provider": "openai_compat", "model": "some-model", "options": {"base_url": "http://127.0.0.1:9"}}


def scripted(name: str, **settings) -> dict:
    return {"name": name, "provider": "fake", "model": "script", **settings}


def config(tmp_path, name: str = "run.yaml", **changes) -> str:
    """A config file for run `cli`: two scripted models at two widths of one painting, and one scripted judge."""
    data = {
        "run_id": "cli",
        "paintings": ["great-wave"],
        "widths": [16, 32],
        "paintings_dir": str(ROOT / "paintings"),
        "runs_dir": str(tmp_path / "runs"),
        "models": [scripted("a"), scripted("b")],
        "judges": [scripted("jj", options={"behaviour": "judge", "order": ["a", "b"]})],
        "judge": {"repeats": 2},
        **changes,
    }
    (tmp_path / name).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(tmp_path / name)


def counts(out: str) -> tuple[int, int]:
    """The attempts and the judging calls that a printed estimate says will be made."""
    return sum(map(int, re.findall(r"(\d+) attempts,", out))), sum(map(int, re.findall(r"(\d+) calls \(", out)))


@pytest.fixture
def run(capsys):
    """Run one command line; returns (exit code, what it printed, what it said on stderr)."""

    def run(*argv: str) -> tuple[int, str, str]:
        code = cli.main(list(argv))
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return run


@pytest.fixture
def calls(monkeypatch) -> list[str]:
    """The name of the config entry of every model call made."""
    made, complete = [], fake.FakeAdapter.complete

    def counted(self, messages, tools, settings):
        made.append(settings.name)
        return complete(self, messages, tools, settings)

    monkeypatch.setattr(fake.FakeAdapter, "complete", counted)
    return made


@pytest.fixture
def layout(tmp_path) -> RunLayout:
    return RunLayout(tmp_path / "runs" / "cli")


def test_estimate_prints_the_counts_and_calls_nothing(tmp_path, run, calls):
    code, out, _ = run("estimate", "--config", config(tmp_path))
    # Two models at two widths. The judge ranks each width's pictures twice, and each model's two sizes twice.
    assert code == 0 and counts(out) == (4, 4 + 4)
    assert calls == [] and not (tmp_path / "runs").exists()


def test_attempt_shows_the_estimate_and_does_nothing_without_a_yes(tmp_path, run, calls, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO())  # not a terminal, so nobody can be asked
    with_a_real_model = config(tmp_path, models=[scripted("a"), REAL], retries={"call_retries": 0, "restarts": 0})
    code, out, _ = run("attempt", "--config", with_a_real_model)
    assert code == 1 and counts(out) == (4, 0)
    assert calls == [] and not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "only, made",
    [
        (["great-wave/16/a/1"], ["great-wave/16/a/1"]),
        (["*/*/b"], ["great-wave/16/b/1", "great-wave/32/b/1"]),
        (["great-wave/16/a", "great-wave/32"], ["great-wave/16/a/1", "great-wave/32/a/1", "great-wave/32/b/1"]),
    ],
)
def test_only_limits_what_is_run(tmp_path, run, layout, only, made):
    flags = [word for pattern in only for word in ("--only", pattern)]
    code, out, _ = run("attempt", "--config", config(tmp_path), *flags)
    assert code == 0 and counts(out) == (len(made), 0)
    assert [attempt.id for attempt in layout.attempts()] == made


def test_a_second_attempt_on_a_run_directory_is_refused_by_the_lock(tmp_path, run, calls, layout):
    path = config(tmp_path)
    with run_lock(layout.root):  # as another `attempt` or `judge` holds the run while it works
        code, _, err = run("attempt", "--config", path)
        assert code == 2 and f"pid {os.getpid()}" in err  # it names the holder
        assert calls == [] and layout.attempts() == [] and not layout.config_path.exists()
    assert run("attempt", "--config", path)[0] == 0 and len(layout.attempts()) == 4  # the run is free again


@pytest.mark.parametrize(
    "change, named",
    [
        ({"prompt_version": "A3"}, "prompt_version"),
        ({"limits": {"max_steps": 10}}, "limits.max_steps"),
        ({"models": [scripted("a", model="script-v2"), scripted("b")]}, "model a: model"),
    ],
)
def test_a_changed_condition_of_a_saved_run_is_refused(tmp_path, run, calls, layout, change, named):
    assert run("attempt", "--config", config(tmp_path))[0] == 0
    saved, made = layout.config_path.read_bytes(), len(calls)
    # More repeats, so that there is something to run: under other conditions it is refused, and nothing is touched.
    code, _, err = run("attempt", "--config", config(tmp_path, "changed.yaml", repeats=2, **change))
    assert code == 2 and named in err
    assert len(calls) == made and len(layout.attempts()) == 4 and layout.config_path.read_bytes() == saved
    # Under the same conditions the run is carried on.
    assert run("attempt", "--config", config(tmp_path, "more.yaml", repeats=2))[0] == 0 and len(layout.attempts()) == 8
