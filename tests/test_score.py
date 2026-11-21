"""The score stage, the estimate, and the lock on a run directory.

A run is written record by record into tmp_path, in the shapes paintbench/layout.py fixes. No model is called.
A model is named by one letter and a ranking by its letters, best first: "bac" puts b above a above c.
"""

from __future__ import annotations

from collections import Counter

import pytest

from paintbench.estimate import estimate, to_do
from paintbench.harness import run_attempts
from paintbench.judge import run_judging
from paintbench.layout import RunLayout, RunLocked, run_lock, write_json
from paintbench.score import compute_scores, format_scores

P, Q = "great-wave", "the-scream"


@pytest.fixture
def layout(tmp_path) -> RunLayout:
    return RunLayout(tmp_path / "run")


def attempt(layout: RunLayout, model: str, painting: str = P, width: int = 32, state: str = "complete") -> str:
    """Write the record of a model's one attempt at a painting and width, and return its id."""
    ref = layout.attempt(painting, width, model, 1)
    write_json(ref.stats_path, {"state": state, "painting": painting, "width": width, "model": model, "attempt": 1})
    return ref.id


def rank(layout: RunLayout, order: str, judge: str = "judge", repeat: int = 1, painting: str = P, width: int = 32, status: str = "ok") -> None:
    """Write one ranking call by `judge`: a complete attempt for each model in `order`, ranked in that order."""
    ids = [attempt(layout, model, painting, width) for model in order]
    record = {"kind": "ranking", "painting": painting, "width": width, "judge": judge, "repeat": repeat, "status": status, "ranking_attempts": ids}
    write_json(layout.judgement_dir(painting, width, judge) / f"{repeat}.json", record)


def figures(rows: list[dict]) -> list[tuple]:
    """(model, rank score) for each row, in the order the rows are listed."""
    return [(row["model"], row["rank_score"]) for row in rows]


def test_a_rank_score_is_the_share_of_comparisons_won(layout):
    for repeat, order in enumerate(["abc", "bac", "abc"], 1):
        rank(layout, order, repeat=repeat)
    (scored,) = compute_scores(layout)["combined"]
    # A ranking of three gives each picture two comparisons. a wins 2 + 1 + 2 of its six, b 1 + 2 + 1, c none.
    assert [(m["model"], m["wins"], m["comparisons"]) for m in scored["models"]] == [("a", 5, 6), ("b", 4, 6), ("c", 0, 6)]
    assert figures(scored["models"]) == [("a", pytest.approx(5 / 6)), ("b", pytest.approx(4 / 6)), ("c", 0.0)]


def test_every_call_of_every_judge_counts_once_and_a_failed_call_not_at_all(layout):
    for repeat, order in enumerate(["abc", "abc", "acb"], 1):
        rank(layout, order, judge="one", repeat=repeat)
    rank(layout, "bac", judge="two")
    rank(layout, "cba", judge="two", repeat=2, status="failed")
    scores = compute_scores(layout)
    (pooled,) = scores["combined"]
    # a wins 2 + 2 + 2 with judge one and 1 with judge two: 7 of 8. The mean of the two judges' figures, 1.0 and 0.5, would be 0.75.
    assert (pooled["calls"], figures(pooled["models"])) == (4, [("a", 7 / 8), ("b", 4 / 8), ("c", 1 / 8)])
    assert (scores["counts"]["judgements"], scores["counts"]["judgements_failed"]) == (4, 1)


def test_each_width_has_its_own_standings_and_the_largest_leads(layout):
    for (painting, width), order in {(P, 16): "abc", (Q, 16): "abc", (P, 32): "cba", (Q, 32): "bca"}.items():
        rank(layout, order, painting=painting, width=width)
    scores = compute_scores(layout)
    narrow, wide = scores["standings_by_width"]
    # Wins over comparisons, pooled over the two paintings. At 16 a has 4 of 4. At 32 it has none, and b and c have 3 of 4 each.
    assert (narrow["width"], figures(narrow["models"])) == (16, [("a", 1.0), ("b", 0.5), ("c", 0.0)])
    assert (wide["width"], figures(wide["models"])) == (32, [("b", 0.75), ("c", 0.75), ("a", 0.0)])
    assert scores["default_width"] == 32
    text = format_scores(scores)
    assert text.index("Standings at 32 wide") < text.index("Standings at 16 wide") < text.index("All sizes together")


def test_a_missing_picture_counts_as_nothing_and_not_as_a_loss(layout):
    rank(layout, "cab")
    rank(layout, "ab", painting=Q)
    attempt(layout, "c", painting=Q, state="refused")
    scores = compute_scores(layout)
    # c is first of three on the one painting it has a picture of: 1.0, where a last place on the other would make it 0.5.
    # The set without c still counts for a (0.5 and 1.0) and for b.
    assert figures(scores["models"]) == [("c", 1.0), ("a", 0.75), ("b", 0.0)]
    (at_32,) = scores["standings_by_width"]
    assert [(m["model"], m["wins"], m["comparisons"], m["sets_scored"]) for m in at_32["models"]] == [("c", 2, 2, 1), ("a", 2, 3, 2), ("b", 0, 3, 2)]


def test_an_entry_kept_aside_is_taken_out_of_every_ranking(layout, make_config):
    rank(layout, "xab")
    models = [{"name": name, "provider": "fake", "model": name} for name in "abx"]
    assert figures(compute_scores(layout, make_config(models=models))["models"]) == [("x", 1.0), ("a", 0.5), ("b", 0.0)]

    scores = compute_scores(layout, make_config(models=models, aside=["x"]))
    # Without x the ranking reads a, b.
    assert figures(scores["models"]) == [("a", 1.0), ("b", 0.0), ("x", None)]
    assert all(m["model"] != "x" for listed in scores["combined"] + scores["standings_by_width"] for m in listed["models"])


@pytest.mark.parametrize(
    "rankings, separated, places",
    [
        ({P: "abc", Q: "bac"}, [False, True], {"a": "1-2", "b": "1-2", "c": "3"}),  # a and b change places between the paintings
        ({P: "abc", Q: "abc"}, [True, True], {"a": "1", "b": "2", "c": "3"}),  # the same order on both
        ({P: "abc"}, [None, None], {"a": "1-3", "b": "1-3", "c": "1-3"}),  # one painting: nothing to resample, no order established
    ],
)
def test_neighbours_whose_order_is_not_separated_share_a_place(layout, rankings, separated, places):
    for painting, order in rankings.items():
        rank(layout, order, painting=painting)
    (standings,) = compute_scores(layout)["standings_by_width"]
    assert [pair["separated"] for pair in standings["pairs"]] == separated
    assert {m["model"]: m["place"] for m in standings["models"]} == places


def test_agreement_with_a_person_and_between_judges_on_a_known_case(layout):
    for judge, order in {"same": "abcd", "swap": "bacd", "reverse": "dcba"}.items():
        rank(layout, order, judge=judge)
    a, b, c, d = (attempt(layout, model) for model in "abcd")
    person = {"kind": "human", "painting": P, "width": 32, "person": "person", "ranking_attempts": [a, b, c, d]}
    write_json(layout.human_path(P, 32, "person"), person)
    scores = compute_scores(layout)

    with_person = {row["judge"]: row["repeats"][0] for row in scores["judge_vs_person"]}
    # Four pictures are six pairs. With one pair the other way round, tau is (5 - 1) / 6.
    assert {judge: call["tau"] for judge, call in with_person.items()} == {"same": 1.0, "swap": pytest.approx(2 / 3), "reverse": -1.0}
    assert with_person["swap"]["swapped_pairs"] == [[a, b]] and with_person["swap"]["within_one_adjacent_swap"]
    assert not with_person["reverse"]["within_one_adjacent_swap"]
    between = {(row["judge_a"], row["judge_b"]): row["tau_b"] for row in scores["judge_vs_judge"]}
    assert between == {("reverse", "same"): -1.0, ("reverse", "swap"): pytest.approx(-2 / 3), ("same", "swap"): pytest.approx(2 / 3)}


def test_the_estimate_counts_the_attempts_and_judging_calls_that_are_then_made(make_config):
    models = [{"name": name, "provider": "fake", "model": "script"} for name in "abc"]
    judge = {"name": "judge", "provider": "fake", "model": "script", "options": {"behaviour": "judge", "order": ["a", "b", "c"]}}
    cfg = make_config(widths=[16, 32], models=models, judges=[judge], judge={"repeats": 2})
    layout = RunLayout(cfg.run_dir)

    before = estimate(cfg, layout)
    assert {name: row["attempts"] for name, row in before["models"].items()} == {"a": 2, "b": 2, "c": 2}
    # Each of the two widths is ranked twice, and each model's two sizes are ranked against each other twice.
    assert (before["judges"]["judge"]["ranking"], before["judges"]["judge"]["peak"], to_do(before)) == (4, 6, 16)

    assert len(run_attempts(cfg, layout, progress=lambda line: None)) == 6
    made = run_judging(cfg, layout, seed=1, progress=lambda line: None)
    assert Counter(record["kind"] for record in made) == {"ranking": 4, "peak": 6}
    assert to_do(estimate(cfg, layout)) == 0  # everything is on disk, so nothing is left to count


def test_a_run_directory_is_held_by_one_holder_at_a_time(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RunLocked):
            with run_lock(tmp_path):
                pass
    with run_lock(tmp_path):  # given up when the block ends
        pass
