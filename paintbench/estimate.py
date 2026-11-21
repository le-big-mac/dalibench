"""Count the attempts and judging calls a command would make and put an approximate cost on them.

Rules:

- An estimate is of what the command is about to do. It takes the jobs and
  the flags the command will use and decides with the predicates the stages
  decide with: harness.attempt_wanted for an attempt, judge.call_wanted for a
  judging call. So the counts printed before a confirmation are the counts
  then made, whatever is on disk and whichever flags are given.
- Judging is planned on one of two bases, and the estimate says which.
  Before the attempts exist (`estimate`): the complete attempts on disk
  plus every attempt still to run, as if each completes. At judge time
  (`from_disk`): the complete attempts on disk and nothing else, which is how
  the judge stage plans, and no attempt is estimated.
- An estimate with nothing left to make says so, and `to_do` is then 0.
- The attempts on disk that the command leaves alone are counted, and of
  those the ones that did not end complete are named by how they ended: in
  error, blocked by the provider, refused in the model's own words,
  incomplete. Where a flag runs a kind again the line gives the flag, and
  which attempts that is comes from attempt_wanted too.
- The counts of attempts and judging calls are exact. The token figures are
  rough, and say so: what a model types is small and known (at most W x H
  characters), but reasoning dominates the cost and nobody knows in advance
  how much a model will do. The high figure is the expected one times
  HIGH_FACTOR; it is not a ceiling.

The assumptions below can be overridden per model under `estimate:` in the
config; a key that is not one of these three is refused, not ignored.

    estimate:
      reasoning_tokens_first_call: 15000   # thinking before the first strip
      reasoning_tokens_per_call: 4000      # thinking before each later strip
      tokens_per_typed_char: 0.6

For scale, six pictures of The Great Wave at 32 x 22 that models typed before
the harness existed took 3 to 9 minutes each at the highest effort setting,
which is very roughly 10,000 to 40,000 output tokens.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from .canvas import VALUE
from .config import RunConfig
from .harness import attempt_wanted
from .judge import KINDS, call_wanted, plan_judging
from .layout import STATE_INCOMPLETE, STATE_REFUSED, RunLayout, read_json
from .paintings import load_painting
from .types import ModelSettings

DEFAULTS = {
    "reasoning_tokens_first_call": 15_000,
    "reasoning_tokens_per_call": 4_000,
    "tokens_per_typed_char": 0.6,
}
PROMPT_TOKENS = 450  # system prompt + task message + tool definition
TOOL_RESULT_TOKENS = 40
HIGH_FACTOR = 3.0  # the "high" figure is this many times the expected one

# Judge calls: rough, provider-independent figures.
JUDGE_PROMPT_TOKENS = 350
JUDGE_REFERENCE_IMAGE_TOKENS = 1_600
JUDGE_ATTEMPT_IMAGE_TOKENS = 600
JUDGE_OUTPUT_TOKENS_PER_PICTURE = 60
JUDGE_REASONING_TOKENS = 3_000

BASIS_CONFIG = "assuming every attempt still to run completes"
BASIS_DISK = "the complete attempts on disk"


def _assumption(model: ModelSettings, key: str) -> float:
    return float(model.estimate.get(key, DEFAULTS[key]))


def attempt_estimate(model: ModelSettings, width: int, height: int, encoding: str, strip: int, max_calls: int) -> dict[str, Any]:
    """Calls and tokens for one attempt by one model at one size. `max_calls` is the most the limits allow."""
    typed_chars = width * height * (7 if encoding == VALUE else 1)
    typed_tokens = typed_chars * _assumption(model, "tokens_per_typed_char")
    min_calls = math.ceil(height / strip) + 1  # one per strip, then the closing reply
    reasoning = _assumption(model, "reasoning_tokens_first_call") + (min_calls - 1) * _assumption(model, "reasoning_tokens_per_call")
    output = typed_tokens + reasoning + 150
    per_call_output = output / min_calls
    # Each call re-sends the conversation so far.
    input_tokens = sum(PROMPT_TOKENS + i * (per_call_output + TOOL_RESULT_TOKENS) for i in range(min_calls))
    return {
        "min_calls": min_calls,
        "max_calls": max_calls,
        "typed_chars": typed_chars,
        "input_tokens": round(input_tokens),
        "output_tokens": round(output),
        "cost_usd": model.cost_usd(round(input_tokens), round(output)),
    }


def estimate(
    cfg: RunConfig,
    layout: RunLayout | None = None,
    judge_names: Iterable[str] | None = None,
    jobs: list[tuple[str, int, str, int]] | None = None,
    force_attempts: bool = False,
    retry_errors: bool = False,
    kinds: Iterable[str] | None = None,
    force_judging: bool = False,
    retry_failed: bool = False,
    from_disk: bool = False,
    retry_refused: bool = False,
) -> dict[str, Any]:
    """What `attempt` and `judge` would make, given the jobs and flags the command will use.

    `jobs`, `force_attempts`, `retry_errors` and `retry_refused` are what
    run_attempts is given; `judge_names`, `kinds`, `force_judging` and
    `retry_failed` are what run_judging is given. `jobs` defaults to every
    attempt the config asks for, `kinds` to every kind. With `from_disk` the
    judging is planned from the complete attempts on disk, as the judge stage
    plans it, and no attempt is estimated.
    """
    layout = layout or RunLayout(cfg.run_dir)
    for model in [] if from_disk else cfg.models:
        unknown = sorted(set(model.estimate) - set(DEFAULTS))
        if unknown:  # a misspelt key would leave the default in force, silently
            raise ValueError(f"config: model {model.name!r}: estimate has unknown keys {unknown}; the keys are {list(DEFAULTS)}")
    jobs = [] if from_disk else cfg.attempt_jobs() if jobs is None else jobs
    paintings = {slug: load_painting(slug, cfg.paintings_path) for slug in ([] if from_disk else cfg.paintings)}
    # The most model calls an attempt can make: max_steps in each conversation, and a discarded one is started again.
    max_calls = cfg.limits.max_steps * (cfg.retries.restarts + 1)

    models: dict[str, dict[str, Any]] = {}
    to_run: set[str] = set()
    for painting, width, model_name, n in jobs:
        model = cfg.model(model_name)
        row = models.setdefault(
            model_name,
            {"model": model.model, "provider": model.provider, "reasoning": model.reasoning, "attempts": 0, "already_done": 0,
             "already_error": 0, "already_blocked": 0, "already_refused": 0, "already_incomplete": 0,
             "min_calls": 0, "max_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "priced": bool(model.price)},
        )
        ref = layout.attempt(painting, width, model_name, n)
        if not attempt_wanted(ref, force_attempts, retry_errors, retry_refused):
            row["already_done"] += 1
            # What each flag would run again is asked of the rule the stage runs by, so the hint is what the flag then does.
            error, blocked = attempt_wanted(ref, retry_errors=True), attempt_wanted(ref, retry_refused=True)
            state = ref.state()
            row["already_error"] += error
            row["already_blocked"] += blocked
            row["already_refused"] += state == STATE_REFUSED and not blocked  # the model's own answer: no flag but --force
            row["already_incomplete"] += state == STATE_INCOMPLETE
            continue
        to_run.add(ref.id)
        one = attempt_estimate(model, width, paintings[painting].height_for(width), cfg.encoding, cfg.strip, max_calls)
        row["attempts"] += 1
        for key in ("min_calls", "max_calls", "input_tokens", "output_tokens"):
            row[key] += one[key]
        if one["cost_usd"] is not None:
            row["cost_usd"] += one["cost_usd"]

    names = [j.name for j in cfg.judges] if judge_names is None else [judge_names] if isinstance(judge_names, str) else list(judge_names)
    expected = None if from_disk else sorted(to_run | {a.id for a in layout.complete_attempts()})
    plan = plan_judging(cfg, layout, judge_names=names, kinds=KINDS if kinds is None else kinds, assume_complete=expected)
    judges: dict[str, dict[str, Any]] = {}
    for judge in map(cfg.judge_model, names):  # every judge asked for has a line, with or without a call to make
        judges[judge.name] = {
            "model": judge.model, "provider": judge.provider, "reasoning": judge.reasoning, "ranking": 0, "pairwise": 0, "peak": 0,
            "already_done": 0, "already_failed": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "priced": bool(judge.price),
        }
    for call in plan:
        judge, row = cfg.judge_model(call.judge), judges[call.judge]
        # A picture about to be typed again will not be the one a saved ranking of it showed.
        if not (call_wanted(call, layout, force_judging, retry_failed) or to_run.intersection(call.attempt_ids)):
            row["already_done"] += 1
            row["already_failed"] += read_json(call.path).get("status") != "ok"
            continue
        n = len(call.attempt_ids)
        row[call.kind] += 1
        tokens_in = JUDGE_PROMPT_TOKENS + JUDGE_REFERENCE_IMAGE_TOKENS + n * JUDGE_ATTEMPT_IMAGE_TOKENS
        tokens_out = n * JUDGE_OUTPUT_TOKENS_PER_PICTURE + 100 + JUDGE_REASONING_TOKENS
        row["input_tokens"] += tokens_in
        row["output_tokens"] += tokens_out
        cost = judge.cost_usd(tokens_in, tokens_out)
        if cost is not None:
            row["cost_usd"] += cost

    return {
        "run_id": cfg.run_id,
        "encoding": cfg.encoding,
        "models": models,
        "judges": judges,
        "judging_basis": BASIS_DISK if from_disk else BASIS_CONFIG,
        "high_factor": HIGH_FACTOR,
    }


def to_do(est: dict[str, Any], attempts: bool = True, judging: bool = True) -> int:
    """How many attempts and judging calls an estimate says are still to be made."""
    count = sum(row["attempts"] for row in est["models"].values()) if attempts else 0
    if judging:
        count += sum(row[kind] for row in est["judges"].values() for kind in KINDS)
    return count


def _money(row: dict[str, Any]) -> str:
    if not row["priced"]:
        return "no price in config"
    return f"${row['cost_usd']:,.2f} (high figure ${row['cost_usd'] * HIGH_FACTOR:,.2f})"


# The work left alone that did not come to a picture or a ranking, by how it ended: the row key that counts it,
# what to call it, and the flag (with its verb) that makes that kind again, where there is one.
_ENDED = {
    "attempts": (
        ("already_error", "ended in error", "--retry-errors runs"),
        ("already_blocked", "blocked by the provider", "--retry-refused runs"),
        ("already_refused", "refused in the model's own words", None),
        ("already_incomplete", "incomplete", None),
    ),
    "judging": (("already_failed", "failed", "--retry-failed makes"),),
}


def _ended(rows: Iterable[dict[str, Any]], part: str) -> str:
    """"2 ended in error; --retry-errors runs them again; 1 incomplete": each kind, and after it its flag. "" if all ended well."""
    rows, kinds = list(rows), []
    for key, what, flag in _ENDED[part]:
        count = sum(row[key] for row in rows)
        if count:
            kinds.append(f"{count} {what}" + (f"; {flag} {'it' if count == 1 else 'them'} again" if flag else ""))
    return "; ".join(kinds)


def _on_disk(row: dict[str, Any], part: str) -> str:
    """The note at the end of a line: how much is left alone, and how that ended where it did not end well."""
    if not row["already_done"]:
        return ""
    ended = _ended([row], part)
    return f"  [{row['already_done']} already on disk" + (f", of which {ended}" if ended else "") + "]"


def _nothing_to_do(rows: Iterable[dict[str, Any]], part: str, things: str, none: str) -> str:
    rows = list(rows)
    on_disk, ended = sum(row["already_done"] for row in rows), _ended(rows, part)
    if not on_disk:
        return none
    return f"all of the {things} are already on disk: {on_disk}" + (f" ({ended})" if ended else "")


def format_estimate(est: dict[str, Any], attempts: bool = True, judging: bool = True) -> str:
    lines = [f"Estimate for run '{est['run_id']}' ({est['encoding']} encoding)", ""]
    total = 0.0
    unpriced = False
    nothing: list[str] = []
    if attempts:
        lines.append("Attempts")
        if not est["models"]:
            lines.append("  none asked for")
        for name, row in est["models"].items():
            total += row["cost_usd"]
            unpriced = unpriced or (not row["priced"] and row["attempts"] > 0)
            lines.append(
                f"  {name:<24} {row['attempts']:>4} attempts, {row['min_calls']:>5} to {row['max_calls']:>5} model calls, "
                f"about {row['input_tokens'] / 1000:,.0f}k in / {row['output_tokens'] / 1000:,.0f}k out tokens, {_money(row)}"
                + _on_disk(row, "attempts")
            )
        lines.append("")
        nothing.append(_nothing_to_do(est["models"].values(), "attempts", "attempts asked for", "no attempt is asked for"))
    if judging:
        lines.append(f"Judging ({est['judging_basis']})")
        if not est["judges"]:
            lines.append("  no judges configured")
        for name, row in est["judges"].items():
            total += row["cost_usd"]
            calls = row["ranking"] + row["pairwise"] + row["peak"]
            unpriced = unpriced or (not row["priced"] and calls > 0)
            lines.append(
                f"  {name:<24} {calls:>4} calls ({row['ranking']} ranking, {row['pairwise']} pairwise, {row['peak']} peak-size), "
                f"about {row['input_tokens'] / 1000:,.0f}k in / {row['output_tokens'] / 1000:,.0f}k out tokens, {_money(row)}"
                + _on_disk(row, "judging")
            )
        lines.append("")
        nothing.append(_nothing_to_do(est["judges"].values(), "judging", "judging calls planned", "no judging call is planned"))
    if not to_do(est, attempts, judging):
        lines.append(f"Nothing to do: {'; '.join(nothing)}.")
        return "\n".join(lines)
    lines.append(
        f"Total: about ${total:,.2f}, high figure about ${total * HIGH_FACTOR:,.2f}"
        + (" (models with no price are not counted)" if unpriced else "")
    )
    lines.append(
        "The numbers of attempts and judging calls are exact. Model calls per attempt run from the fewest a whole picture "
        "takes to the most the limits allow. Token and cost figures are rough, and the high figure is the expected one "
        f"times {HIGH_FACTOR:g}, not a ceiling. Reasoning dominates and is assumed, not known: "
        f"{DEFAULTS['reasoning_tokens_first_call']:,} tokens before the first strip and {DEFAULTS['reasoning_tokens_per_call']:,} before each "
        "later one, unless the config says otherwise under a model's `estimate:`."
    )
    return "\n".join(lines)
