"""The judge stage: a strong vision model ranks typed attempts against the real painting.

The protocol is a fixed point of the benchmark (docs/method.md describes it).
The rules this module implements:

- One judging call is one user message: the instructions, the text
  "REFERENCE", the real painting, then for each attempt the text
  "Picture <label>" followed by its image. No system prompt and no tools.
- Labels are random single capital letters in a random order. They are never
  drawn on an image and nothing depends on a file name: the position of a
  picture in the message is the only thing that ties it to its label.
- Two calls on the same set of attempts never share a label key or an order
  (see assign_labels). The seed of every call is saved with it.
- Like with like: only complete typed attempts are shown, each as its
  picture.png (solid cells, longest side about 640 pixels, no smoothing).
  Reduced or blurred copies of the reference are never mixed in.
- Three kinds of call, one record shape (see layout.py):
    ranking   one painting at one grid size, every complete attempt, when
              there are at most `judge.max_per_call` of them
    pairwise  the same set when it is larger, two attempts per call
    peak      one contestant's attempts at its different widths, one per
              width, ranked against each other at the same display size
- The pictures of a peak call are grids of different sizes, so picture.png
  would show each at a size of its own. They are drawn again for the call,
  all to one size (render.render_same_size), as the peak prompt says they are.
- The reply is parsed strictly (see parse_judge_reply). If it does not hold
  one ranking that lists every label exactly once, the judge is asked once
  more in the same conversation, so with the same images, and is told the
  required form, not what was wrong. If that fails too the call is recorded
  as failed. A ranking is never guessed or repaired.
- Every call is a file, and a call already on disk is not made again while
  the attempts it judged are unchanged. So judging can be resumed, and a
  saved run can be judged by a new judge without calling any contestant.
- A record holds the SHA-256 of each picture.txt it showed. When the set of
  attempts a call would show has changed (an attempt or a width was added or
  removed), or one of its pictures was typed again, the record is stale and
  the call is made again in its place (see still_current and call_wanted).
- A record also holds the SHA-256 of the reference image it showed
  ("reference"). One reference per work, never swapped mid-study: a record
  made against a file that is not the painting's reference now is not
  current either, and its call is made again.
- A call is handed to a worker only when its provider and its judge both
  have a free place, and the worker asks the judge only if the stage has not
  been stopped by the time it has drawn the pictures. Stopping the stage
  (Ctrl-C) therefore starts nothing new: the calls already with a judge
  finish and are saved, the rest are left for a later run.
"""

from __future__ import annotations

import hashlib
import math
import mimetypes
import os
import random
import re
import string
import threading
import time
import traceback
import uuid
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from . import __version__
from .adapters import Adapter, AdapterError, TransientError, get_adapter
from .canvas import load_picture
from .config import Retries, RunConfig
from .layout import AttemptRef, RunLayout, read_json, write_json
from .paintings import Painting, load_painting
from .prompts import (
    JUDGE_PROMPT_VERSION,
    JUDGE_RETRY,
    PEAK_PROMPT_VERSION,
    PICTURE_LABEL,
    REFERENCE_LABEL,
    judge_prompt,
    judge_prompt_peak,
)
from .reference import sha256_file
from .render import render_file, render_same_size
from .types import STOP_MAX_TOKENS, STOP_REFUSAL, Image, Message, ModelSettings, Reply, Text, user
from .workers import Watch

KINDS = ("ranking", "pairwise", "peak")
LETTERS = string.ascii_uppercase

_Shown = tuple[Mapping[str, str], Sequence[str]]  # what one call showed: ({label: attempt id}, labels in order)


# ---------------------------------------------------------------- the plan


@dataclass(frozen=True)
class JudgeCall:
    """One judging call that a run would make. Nothing has been asked of any model."""

    kind: str  # "ranking" | "pairwise" | "peak"
    painting: str
    judge: str
    repeat: int
    attempt_ids: tuple[str, ...]  # the attempts shown
    path: Path  # where the record goes
    width: int | None = None  # the grid width (ranking and pairwise)
    model: str | None = None  # the contestant whose sizes are ranked (peak)
    reference: str | None = None  # SHA-256 of the painting's reference image as it is now; None if it is not there


def reference_sha256(painting: str, cfg: RunConfig) -> str | None:
    """SHA-256 of a painting's reference image as the file is now: what a saved judgement's "reference" is held against.

    None when it cannot be told: no such painting in the paintings directory,
    metadata that will not load, or no image file. A plan is made all the
    same; a run with calls to make loads the painting itself and says what is
    wrong.
    """
    try:
        return sha256_file(load_painting(painting, cfg.paintings_path).reference_path)
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError):
        return None


def plan_judging(
    cfg: RunConfig,
    layout: RunLayout,
    judge_names: Iterable[str] | None = None,
    kinds: Iterable[str] = KINDS,
    assume_complete: Iterable[str] | None = None,
) -> list[JudgeCall]:
    """Every call a judging run would make, in a fixed order. Makes no call and writes nothing.

    The attempts judged are the complete ones on disk, or, when
    `assume_complete` is given, exactly the attempt ids in it: pass the ids
    of cfg.attempt_jobs() to plan from the config alone, before any attempt
    exists. `judge_names` defaults to every judge in the config. Calls already
    on disk are included; call_wanted() says which of them a run would make.
    """
    return _plan(cfg, layout, judge_names, kinds, assume_complete)[0]


def _plan(
    cfg: RunConfig,
    layout: RunLayout,
    judge_names: Iterable[str] | None,
    kinds: Iterable[str],
    assume_complete: Iterable[str] | None,
) -> tuple[list[JudgeCall], list[str]]:
    """The calls, and a line for each set that cannot be judged."""
    kinds = set(kinds)
    if kinds - set(KINDS):
        raise ValueError(f"unknown kind of judging call {sorted(kinds - set(KINDS))}; known: {list(KINDS)}")
    jc = cfg.judge
    if jc.max_per_call > len(LETTERS):
        raise ValueError(f"judge.max_per_call is {jc.max_per_call}, but a label is one of {len(LETTERS)} capital letters")
    if isinstance(judge_names, str):
        judge_names = [judge_names]
    judges = [j.name for j in cfg.judges]
    if judge_names is not None:
        judges = list(dict.fromkeys(cfg.judge_model(name).name for name in judge_names))

    if assume_complete is None:
        on_disk = layout.attempts()
        refs = [a for a in on_disk if a.is_complete()]
        sets = sorted({(a.painting, a.width) for a in on_disk})
    else:
        refs = sorted((layout.attempt_from_id(i) for i in set(assume_complete)), key=lambda a: (a.painting, a.width, a.model, a.n))
        sets = sorted({(a.painting, a.width) for a in refs})
    references = {painting: reference_sha256(painting, cfg) for painting in sorted({a.painting for a in refs})} if judges else {}

    calls: list[JudgeCall] = []
    skipped = [] if judges else ["no judge in the config: nothing to judge"]
    for painting, width in sets if kinds & {"ranking", "pairwise"} else []:
        group = [a for a in refs if (a.painting, a.width) == (painting, width)]
        n = len(group)
        if n < 2:
            skipped.append(f"skip {painting}/{width}: {n} complete attempt{'' if n == 1 else 's'}, a ranking needs at least two")
        elif n <= jc.max_per_call:
            if "ranking" in kinds:
                ids = tuple(a.id for a in group)
                for judge, k in product(judges, range(1, jc.repeats + 1)):
                    path = layout.judgement_dir(painting, width, judge) / f"{k}.json"
                    calls.append(JudgeCall("ranking", painting, judge, k, ids, path, width=width, reference=references[painting]))
        elif "pairwise" in kinds:
            repeats = range(1, jc.effective_pairwise_repeats + 1)
            for judge, (a, b), k in product(judges, _pairs(group, jc.pairs_per_attempt), repeats):
                path = layout.pairwise_dir(painting, width, judge) / f"{a.model}-{a.n}__{b.model}-{b.n}-{k}.json"
                calls.append(JudgeCall("pairwise", painting, judge, k, (a.id, b.id), path, width=width, reference=references[painting]))
        else:
            skipped.append(f"skip {painting}/{width}: {n} complete attempts are more than {jc.max_per_call} per call and need pairwise judging")

    if "peak" in kinds and jc.peak_size:
        too_few: dict[str, list[str]] = {}
        for painting, model in sorted({(a.painting, a.model) for a in refs}):
            by_width: dict[int, list[AttemptRef]] = {}
            for a in refs:
                if (a.painting, a.model) == (painting, model):
                    by_width.setdefault(a.width, []).append(a)
            if len(by_width) < 2:
                too_few.setdefault(painting, []).append(model)
                continue
            for judge, k in product(judges, range(1, jc.repeats + 1)):
                # Repeat k takes the k-th complete attempt at each width, wrapping round, so repeats use different attempts.
                ids = tuple(by_width[w][(k - 1) % len(by_width[w])].id for w in sorted(by_width))
                path = layout.peak_dir(painting, judge, model) / f"{k}.json"
                calls.append(JudgeCall("peak", painting, judge, k, ids, path, model=model, reference=references[painting]))
        for painting, models in sorted(too_few.items()):
            skipped.append(f"skip peak size for {painting}: fewer than two widths complete for {', '.join(models)}")

    if len({c.path for c in calls}) != len(calls):
        raise ValueError("two judging calls would be saved under the same file name; rename the models involved")
    return calls, skipped


def _pairs(group: list[AttemptRef], per_attempt: int | None) -> list[tuple[AttemptRef, AttemptRef]]:
    """The pairs to compare: every pair, or a balanced subset.

    With `per_attempt` = N the attempts are placed on a ring, in an order fixed
    by the SHA-256 of their ids, and each is paired with its nearest N/2
    (rounded up) neighbours on either side. Every attempt then meets N
    opponents (N + 1 when N is odd) and the comparisons form one connected
    graph, which a Bradley-Terry fit needs. The subset depends only on the
    set of attempts, so a resumed run plans the same files.
    """
    n = len(group)
    if not per_attempt or per_attempt >= n - 1:
        return list(combinations(group, 2))
    ring = sorted(group, key=lambda a: hashlib.sha256(a.id.encode()).hexdigest())
    reach = (per_attempt + 1) // 2  # at most (n - 1) / 2 here, so no pair is reached from both sides
    chosen = {frozenset((ring[i].id, ring[(i + step) % n].id)) for i in range(n) for step in range(1, reach + 1)}
    return [(a, b) for a, b in combinations(group, 2) if frozenset((a.id, b.id)) in chosen]


# ---------------------------------------------------------------- what is already on disk


def still_current(record: dict[str, Any], attempt_ids: Iterable[str], layout: RunLayout, reference: str | None = None) -> bool:
    """Whether a saved judgement ranks exactly these attempts, as they are on disk now, against the reference as it is now.

    The attempts the record ranked must be `attempt_ids`, no more and no
    fewer, and each must still have the picture that was shown: the record's
    "pictures" holds the SHA-256 of every picture.txt at the time of the
    call. A record with no "pictures" (an imported ranking) is judged on the
    ids alone.

    `reference` is the SHA-256 of the painting's reference image as the file
    is now (reference_sha256). A record that says which reference it showed
    (its "reference") must have shown that one. A record without the key,
    made before records held it or imported, is judged on its pictures
    alone; so is every record when `reference` is None, the file not being
    there to compare.
    """
    labels = record.get("labels") if isinstance(record, dict) else None
    if not isinstance(labels, dict) or set(labels.values()) != set(attempt_ids):
        return False
    if not _same_reference(record, reference):
        return False
    pictures = record.get("pictures")
    if pictures is None:
        return True
    return all(pictures.get(attempt) == layout.attempt_from_id(attempt).picture_sha256() for attempt in labels.values())


def _same_reference(record: dict[str, Any], reference: str | None) -> bool:
    """False only if the record says which reference image it showed and `reference`, the painting's now, is another."""
    shown = record.get("reference")
    return shown is None or reference is None or shown == reference


def _saved_state(call: JudgeCall, layout: RunLayout) -> str:
    """What is on disk for a planned call: "missing", "failed", "ok", or one of the two ways of not being still_current.

    "stale": the attempts it showed are not the call's attempts as they are
    now. "reference": those are unchanged, but the reference image it showed
    is not the painting's now.
    """
    if not call.path.is_file():
        return "missing"
    record = read_json(call.path)
    if not still_current(record, call.attempt_ids, layout):
        return "stale"
    if not _same_reference(record, call.reference):
        return "reference"
    return "ok" if record.get("status") == "ok" else "failed"


def _wanted(state: str, force: bool, retry_failed: bool) -> bool:
    return force or state in ("missing", "stale", "reference") or (retry_failed and state == "failed")


def call_wanted(call: JudgeCall, layout: RunLayout, force: bool = False, retry_failed: bool = False) -> bool:
    """Whether a judging run would make this planned call. The one rule, for run_judging and for the estimate.

    It is made if nothing is saved for it; if what is saved is not current,
    because the attempts it would show have changed since or the painting's
    reference image is not the one it showed; if `force` is set; or if
    `retry_failed` is set and what is saved is not an "ok" record.
    """
    return _wanted(_saved_state(call, layout), force, retry_failed)


# ---------------------------------------------------------------- labels and order


def assign_labels(attempt_ids: Iterable[str], rng: random.Random, previous: Iterable[_Shown] = ()) -> tuple[dict[str, str], list[str]]:
    """Random labels and a random order for one call: ({label: attempt id}, labels in the order shown).

    `previous` holds the same pair for earlier calls. Those made on the same
    set of attempts constrain this one, so that fresh labels and a fresh order
    are certain and not just likely:

    - The pictures are shown in an order no earlier call used. When every
      order has been used (two pictures have only two), it is the order used
      least often and never the latest one, so two pictures alternate.
    - The letters are redrawn until the label key differs from every earlier
      key and the sequence of letters differs from every earlier sequence.

    The result depends only on the set of ids, the rng and `previous`, so a
    saved call is reproduced by random.Random(record["seed"]) and the
    (labels, order_shown) of the earlier repeats of the same set.
    """
    ids = sorted(attempt_ids)
    if len(set(ids)) != len(ids) or not 1 <= len(ids) <= len(LETTERS):
        raise ValueError(f"a call shows between 1 and {len(LETTERS)} different attempts, not {ids}")
    earlier = [(dict(key), list(shown)) for key, shown in previous if sorted(key.values()) == ids and sorted(shown) == sorted(key)]
    order = _fresh_order(ids, rng, [tuple(key[label] for label in shown) for key, shown in earlier])
    keys = [key for key, _ in earlier]
    sequences = [shown for _, shown in earlier]
    for _ in range(1000):  # a clash is rare and a second one rarer; the bound only stops an endless loop
        letters = rng.sample(LETTERS, len(ids))
        key = dict(zip(letters, order))
        if key not in keys and letters not in sequences:
            break
    return key, letters


def _fresh_order(ids: list[str], rng: random.Random, used: list[tuple[str, ...]]) -> list[str]:
    counts = Counter(used)
    if len(counts) < math.factorial(len(ids)):  # some order is still unused: redraw until one comes up
        while True:
            order = tuple(rng.sample(ids, len(ids)))
            if order not in counts:
                return list(order)
    candidates = [o for o in counts if o != used[-1]] or list(counts)
    fewest = min(counts[o] for o in candidates)
    return list(rng.choice(sorted(o for o in candidates if counts[o] == fewest)))


def _on_disk(directory: Path, leave_out: set[Path]) -> list[dict[str, Any]]:
    """The judgement records saved in a directory, in repeat order, without the files about to be rewritten."""
    records = [read_json(path) for path in directory.glob("*.json") if path not in leave_out]
    usable = [r for r in records if isinstance(r, dict) and isinstance(r.get("labels"), dict) and r.get("order_shown")]
    return sorted(usable, key=lambda r: r.get("repeat") or 0)


def _draw(todo: list[JudgeCall], run_seed: int, layout: RunLayout) -> list[tuple[JudgeCall, dict[str, str], list[str], int]]:
    """Labels, order and seed for each call about to be made: [(call, labels, order shown, seed), ...].

    They are drawn here, one call after another and before any call is made,
    because each repeat has to differ from the earlier repeats of its set:
    those saved by an earlier invocation and those drawn a moment ago. A
    saved record counts only while it is still current for the set and the
    painting's reference, so a call made again because its attempts changed
    is held to the other repeats of the new set and not to rankings of the
    old one. A call's seed is fixed by the run's seed and the call's file,
    whatever else is run with it.
    """
    rewriting = {call.path for call in todo}
    saved: dict[Path, list[dict[str, Any]]] = {}
    earlier: dict[tuple[Path, frozenset[str]], list[_Shown]] = {}
    jobs = []
    for call in todo:
        directory, ids = call.path.parent, frozenset(call.attempt_ids)
        if directory not in saved:
            saved[directory] = _on_disk(directory, rewriting)
        if (directory, ids) not in earlier:
            current = [r for r in saved[directory] if still_current(r, ids, layout, call.reference)]  # one directory, one painting
            earlier[directory, ids] = [(r["labels"], r["order_shown"]) for r in current]
        name = f"{run_seed}/{call.path.relative_to(layout.root).as_posix()}"
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:6], "big")
        labels, order = assign_labels(call.attempt_ids, random.Random(seed), earlier[directory, ids])
        earlier[directory, ids].append((labels, order))
        jobs.append((call, labels, order, seed))
    return jobs


# ---------------------------------------------------------------- the message


def _image(path: Path | str) -> Image:
    return Image(str(path), mimetypes.guess_type(str(path))[0] or "image/png")


def build_judge_messages(prompt_text: str, reference_path: Path | str, labelled: Sequence[tuple[str, Path | str]]) -> list[Message]:
    """The one user message of a judging call.

    `labelled` is [(label, png path), ...] in the order shown. Each label is
    sent as text straight before its image; nothing else ties the two.
    """
    parts: list[Text | Image] = [Text(prompt_text), Text(REFERENCE_LABEL), _image(reference_path)]
    for label, png in labelled:
        parts += [Text(PICTURE_LABEL.format(label=label)), _image(png)]
    return [Message(role="user", content=parts)]


# ---------------------------------------------------------------- the reply


class JudgeParseError(ValueError):
    """The reply has no usable OVERALL line. The message says why."""


@dataclass(frozen=True)
class ParsedReply:
    ranking: list[str]  # labels, best first
    notes: dict[str, str]  # label -> the judge's line about that picture; a label may be absent
    confidence: str | None


_EMPHASIS = r"[*_]{0,3}"
_OVERALL = re.compile(rf"(?:#{{1,6}}\s*)?{_EMPHASIS}OVERALL{_EMPHASIS}\s*:{_EMPHASIS}\s*(.*)")
_CONFIDENCE = re.compile(rf"(?:#{{1,6}}\s*)?{_EMPHASIS}CONFIDENCE{_EMPHASIS}\s*:{_EMPHASIS}\s*(.*)")
_LIST_END = re.compile(r"\s*(?:\(best first(?:, all \d+ labels)?\))?\s*\.?\s*[*_]*$")
_OVERALL_WORD = re.compile(r"(?<![A-Za-z])OVERALL(?![A-Za-z])")
_LONE_CAPITAL = re.compile(r"(?<![A-Za-z])[A-Z](?![A-Za-z])")
_ENTRY = re.compile(r"(?:Picture\s+)?([A-Z])")
_NOTE = re.compile(rf"(?:[-*]\s+)?{_EMPHASIS}(?:Picture\s+)?([A-Z]){_EMPHASIS}\s*:{_EMPHASIS}\s*(\S.*)")


def parse_judge_reply(text: str, labels: Iterable[str]) -> ParsedReply:
    """Read a judge's reply. Raises JudgeParseError unless the ranking is beyond doubt.

    The reply must hold exactly one OVERALL line, and that line must list
    every label exactly once, separated by commas, and nothing else. A label
    that is missing, repeated or unknown, or any other text on the line, is
    an error. Text before and after the line is allowed.

    Tolerated on the OVERALL line, and nothing beyond this:
    - whitespace around the line, the colon, the commas and the entries;
    - "Picture K" as well as "K" (one or more spaces between the words);
    - Markdown on the heading: "#" marks before it, and up to three "*" or
      "_" around the word and after the colon ("**OVERALL:** K, G");
    - "*" or "_" at either end of the list ("**OVERALL: K, G**"), never
      between entries;
    - one full stop after the last entry;
    - the prompt's own remark copied after the list, "(best first)" or
      "(best first, all 6 labels)".
    Not tolerated: "Overall" or "overall", a lower-case label, a separator
    other than a comma, brackets or quotes around entries, ties, a list that
    starts on the next line, and a second OVERALL line.

    A second ranking under any other heading is not tolerated either, since
    the judge may be correcting the first ("CORRECTED OVERALL: G, K, B",
    "OVERALL (revised): G, K, B"). It is taken to be one when another line
    holds the word OVERALL in capitals and, after it, two or more of the
    labels as capital letters standing alone. A sentence that only mentions
    the OVERALL line is allowed.

    Notes and confidence are recorded when present and never cause an error.
    A note is a line between the OVERALL line and the CONFIDENCE line of the
    form "K: text" ("Picture K:", a list bullet and emphasis are allowed);
    the first such line for a label counts. Confidence is the text after
    "CONFIDENCE:", or the next line that is not blank if that line is empty.
    """
    labels = list(labels)
    lines = [line.strip() for line in (text or "").splitlines()]
    found = [(i, m.group(1)) for i, line in enumerate(lines) if (m := _OVERALL.fullmatch(line))]
    if len(found) != 1:
        raise JudgeParseError(f"{len(found)} OVERALL lines, need exactly one" if found else "no OVERALL line")
    at, listing = found[0]
    for i, line in enumerate(lines):
        if i != at and (word := _OVERALL_WORD.search(line)) and len(set(_LONE_CAPITAL.findall(line[word.end() :])) & set(labels)) >= 2:
            raise JudgeParseError(f"a second ranking after the word OVERALL on another line: {line[:60]!r}")
    listing = _LIST_END.sub("", listing.lstrip("*_"), count=1).strip()

    ranking: list[str] = []
    problems: list[str] = []
    for entry in [e.strip() for e in listing.split(",")] if listing else []:
        m = _ENTRY.fullmatch(entry)
        if m:
            ranking.append(m.group(1))
        else:
            problems.append(f"unreadable entry {entry[:40]!r}")
    faults = {
        "missing": [label for label in labels if label not in ranking],
        "repeated": sorted(label for label, count in Counter(ranking).items() if count > 1),
        "unknown": sorted(set(ranking) - set(labels)),
    }
    problems += [f"{fault} {', '.join(which)}" for fault, which in faults.items() if which]
    if problems:
        raise JudgeParseError("OVERALL line: " + "; ".join(problems))

    end = next((i for i in range(at + 1, len(lines)) if _CONFIDENCE.fullmatch(lines[i])), len(lines))
    notes: dict[str, str] = {}
    for line in lines[at + 1 : end]:
        m = _NOTE.fullmatch(line)
        if m and m.group(1) in labels:
            notes.setdefault(m.group(1), m.group(2))
    said = [_CONFIDENCE.fullmatch(lines[end]).group(1), *lines[end + 1 :]] if end < len(lines) else []
    return ParsedReply(ranking=ranking, notes=notes, confidence=next((s for s in said if s), None))


# ---------------------------------------------------------------- one call


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_render_lock = threading.Lock()


def _picture_png(ref: AttemptRef) -> Path:
    """The attempt's PNG, rendered from picture.txt first if it is missing."""
    with _render_lock:  # two calls may show the same attempt
        if not ref.picture_png.is_file():
            render_file(ref.picture_txt, ref.picture_png)
    return ref.picture_png


def _peak_pngs(refs: Sequence[AttemptRef]) -> list[Path]:
    """The pictures of one peak call, drawn for the call at one display size (see render_same_size).

    Each is saved beside its picture.png under a name that gives its size in
    pixels. It is drawn from picture.txt every time, so it is never an older
    typing, and the file is put in place whole, because another call may be
    reading the one it replaces. It is written first under a name that no
    other call has, in this process or in another one judging the same run,
    so two of them never write into one unfinished file.
    """
    images = render_same_size([load_picture(ref.picture_txt) for ref in refs])
    paths = [ref.directory / f"picture-{image.width}x{image.height}.png" for ref, image in zip(refs, images)]
    with _render_lock:
        for path, image in zip(paths, images):
            # The process id says whose a leftover is; the random part keeps the name apart where process ids
            # repeat (two containers on one run directory, a later process after a crash).
            unfinished = path.with_name(f"{path.name}.{os.getpid()}-{uuid.uuid4().hex}.tmp")
            try:
                image.save(unfinished, format="PNG")
                os.replace(unfinished, path)
            except BaseException:
                unfinished.unlink(missing_ok=True)
                raise
    return paths


def _complete(
    adapter: Adapter,
    messages: list[Message],
    settings: ModelSettings,
    retries: Retries,
    sleep: Callable[[float], None],
    usage: dict[str, Any],
) -> Reply:
    """One model call with no tools. A TransientError is retried with backoff; the last one, like any other error, is raised."""
    for i in range(retries.call_retries + 1):
        try:
            reply = adapter.complete(messages, [], settings)
        except TransientError as e:
            if i == retries.call_retries:
                raise
            usage["api_retries"] += 1
            wait = e.retry_after if e.retry_after else min(retries.backoff_max_seconds, retries.backoff_seconds * 2**i)
            sleep(wait * (0.75 + random.random() / 2))
            continue
        usage["model_calls"] += 1
        usage["input_tokens"] += reply.usage.input_tokens or 0
        usage["output_tokens"] += reply.usage.output_tokens or 0
        if reply.usage.reasoning_tokens is not None:
            usage["reasoning_tokens"] = (usage["reasoning_tokens"] or 0) + reply.usage.reasoning_tokens
        return reply
    raise AssertionError("unreachable")


def _ask(
    adapter: Adapter,
    messages: list[Message],
    order: list[str],
    settings: ModelSettings,
    retries: Retries,
    sleep: Callable[[float], None],
    record: dict[str, Any],
) -> None:
    """Ask for the ranking, and once more in the same conversation if the reply does not parse. Fills in the record."""
    for asking in (1, 2):
        reply = _complete(adapter, messages, settings, retries, sleep, record["usage"])
        record["raw_replies"].append(reply.text)
        if reply.stop_reason == STOP_REFUSAL:
            record["error"] = "refused: " + (reply.refusal_text or reply.text or "(no text)")
            return
        try:
            parsed = parse_judge_reply(reply.text, order)
        except JudgeParseError as e:
            cut_off = " (the reply was cut off by the output limit)" if reply.stop_reason == STOP_MAX_TOKENS else ""
            record["parse_errors"].append(f"{e}{cut_off}")
            if asking == 2:
                record["error"] = "no valid ranking in either reply"
                return
            record["asked_again"] = True
            # The images stay in the first message, so the judge is asked about the same pictures.
            messages = messages + [reply.as_message(), user(JUDGE_RETRY.format(labels=", ".join(order), n=len(order)))]
        else:
            record.update(
                status="ok",
                ranking=parsed.ranking,
                ranking_attempts=[record["labels"][label] for label in parsed.ranking],
                notes=parsed.notes,
                confidence=parsed.confidence,
            )
            return


def _run_call(
    call: JudgeCall,
    labels: dict[str, str],
    order: list[str],
    seed: int,
    painting: Painting,
    settings: ModelSettings,
    cfg: RunConfig,
    layout: RunLayout,
    adapter: Adapter,
    sleep: Callable[[float], None],
    stopping: Any,  # is_set(): a threading.Event, or the stage's _Stopping
) -> dict[str, Any] | None:
    """Make one judging call and write its record. Never raises for a failed call: the record says what went wrong.

    If `stopping` is set by the time the pictures are drawn, the stage has
    been stopped: the judge is not asked, nothing is written and None is
    returned, so a later run makes the call.
    """
    started = time.monotonic()
    n = len(order)
    if call.kind == "peak":
        widths = [layout.attempt_from_id(a).width for a in call.attempt_ids]
        prompt, version, height = judge_prompt_peak(n, min(widths), max(widths)), PEAK_PROMPT_VERSION, None
    else:
        height = painting.height_for(call.width)
        prompt, version = judge_prompt(n, call.width, height), JUDGE_PROMPT_VERSION
    record: dict[str, Any] = {
        "kind": call.kind,
        "painting": call.painting,
        "judge": call.judge,
        "repeat": call.repeat,
        "width": call.width,
        "height": height,
        "model": call.model,
        "labels": labels,
        "order_shown": order,
        "pictures": {attempt: layout.attempt_from_id(attempt).picture_sha256() for attempt in sorted(labels.values())},
        "reference": None,  # the SHA-256 of the reference image shown; filled in below, just before the judge is asked
        "status": "failed",
        "ranking": None,
        "ranking_attempts": None,
        "notes": {},
        "confidence": None,
        "raw_replies": [],
        "parse_errors": [],
        "asked_again": False,
        "error": None,
        "judge_settings": settings.to_dict(),
        "prompt_version": version,
        "seed": seed,
        "usage": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": None, "model_calls": 0, "api_retries": 0},
        "cost_usd": None,
        "wall_seconds": None,
        "created_at": _now(),
        "harness_version": __version__,
    }
    try:
        refs = [layout.attempt_from_id(labels[label]) for label in order]
        pngs = _peak_pngs(refs) if call.kind == "peak" else [_picture_png(ref) for ref in refs]
        shown = list(zip(order, pngs))
        record["reference"] = sha256_file(painting.reference_path)  # the file as it is at the call, not as it was when the plan was made
        if stopping.is_set():  # drawing can take a while when many calls begin together; a stop in that time starts no call
            return None
        _ask(adapter, build_judge_messages(prompt, painting.reference_path, shown), order, settings, cfg.retries, sleep, record)
    except AdapterError as e:
        record["error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # a picture that will not render, an adapter bug: this call fails and the others carry on
        record["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    record["cost_usd"] = settings.cost_usd(record["usage"]["input_tokens"], record["usage"]["output_tokens"])
    record["wall_seconds"] = round(time.monotonic() - started, 3)
    write_json(call.path, record)
    return record


# ---------------------------------------------------------------- a judging run


class _Stopping:
    """Whether the stage is being stopped, as its workers ask it.

    True once the stage has taken the stop (`set`), and already from the moment a stop was asked of it
    (workers.ask_stop): a call handed out in that instant, or still drawing its pictures, is not made.
    """

    def __init__(self) -> None:
        self._set = False
        self.watch: Watch | None = None

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set or (self.watch is not None and self.watch.stop_seen)


def run_judging(
    cfg: RunConfig,
    layout: RunLayout,
    judge_names: Iterable[str] | None = None,
    kinds: Iterable[str] = KINDS,
    adapter_factory: Callable[[ModelSettings], Adapter] | None = None,
    force: bool = False,
    progress: Callable[[str], None] = partial(print, flush=True),  # flushed, so a redirected log is live
    seed: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    retry_failed: bool = False,
) -> list[dict[str, Any]]:
    """Make every planned judging call that call_wanted() asks for. Returns the records written, in plan order.

    Only the judges are called: one adapter per judge, calls concurrent,
    each provider up to its `concurrency` and each judge up to its own where
    it has one. A call that fails (the API, a refusal, two replies that do
    not parse) is saved as a "failed" record and the others carry on. A
    record on disk, failed or not, is left alone unless `force` is set, or
    `retry_failed` and it is a failed one, or it is no longer current: the
    attempts its call would show have changed since it was made, or the
    reference image it showed is not the painting's now. Such a call is made
    again and its record replaced. `progress` is told how many calls are
    made again for each of the two reasons, how many are left alone, and how
    many of those are failed records.

    An interrupt (Ctrl-C) lets the calls already with a judge finish and be
    saved, starts no other, and is then raised to the caller. No call waits
    inside a worker for its judge's place, and one that is still drawing its
    pictures when the stop comes does not go on to ask its judge.

    `seed` fixes the labels and orders of the run; without it they come from
    the system's randomness. Either way each record holds its own seed.
    """
    plan, skipped = _plan(cfg, layout, judge_names, kinds, None)
    for line in skipped:
        progress(line)

    states = [_saved_state(call, layout) for call in plan]
    todo = [call for call, state in zip(plan, states) if _wanted(state, force, retry_failed)]
    if "stale" in states:
        progress(f"redo {states.count('stale')} judging calls: the attempts they judged have changed")
    if "reference" in states:
        progress(f"redo {states.count('reference')} judging calls: the reference image they were shown is not the painting's now")
    if len(todo) < len(plan):  # so `force` is off, and a failed record is left alone unless `retry_failed` is on
        failed = 0 if retry_failed else states.count("failed")
        note = f" ({failed} of them failed; --retry-failed makes failed calls again)" if failed else ""
        progress(f"skip {len(plan) - len(todo)} of {len(plan)} judging calls: already on disk{note}")
    if not todo:
        return []

    paintings = {slug: load_painting(slug, cfg.paintings_path) for slug in sorted({c.painting for c in todo})}
    for call in todo:  # fail before any call if the reference or a grid height is missing
        painting = paintings[call.painting]
        if not painting.reference_path.is_file():
            raise FileNotFoundError(f"no reference image at {painting.reference_path}; a judge must always see the real painting. `python scripts/fetch_originals.py {painting.slug}` fetches and rebuilds it")
        if call.width is not None:
            painting.height_for(call.width)

    jobs = _draw(todo, random.SystemRandom().getrandbits(48) if seed is None else seed, layout)
    factory = adapter_factory or (lambda s: get_adapter(s.provider))
    settings: dict[str, ModelSettings] = {}
    for name in dict.fromkeys(call.judge for call in todo):
        judge = cfg.judge_model(name)
        settings[name] = replace(judge, max_output_tokens=cfg.judge.max_tokens) if cfg.judge.max_tokens else judge
    adapters = {name: factory(s) for name, s in settings.items()}

    # The calls not yet started, as places in `jobs`: each judge's own queue, in plan order.
    waiting: dict[str, deque[int]] = {name: deque() for name in settings}
    for i, job in enumerate(jobs):
        waiting[job[0].judge].append(i)
    running: dict[Future, int] = {}
    stopping = _Stopping()  # set when the stage is stopped: a call still drawing its pictures then is not made
    on_provider: Counter[str] = Counter()
    on_judge: Counter[str] = Counter()

    def has_room(name: str) -> bool:
        judge = settings[name]
        if on_provider[judge.provider] >= cfg.concurrency_for(judge.provider):
            return False
        # A judge entry may carry its own `concurrency`: the most calls of that judge in flight at once.
        return not judge.concurrency or on_judge[name] < judge.concurrency

    records: dict[int, dict[str, Any]] = {}
    with ExitStack() as stack, Watch() as watch:
        stopping.watch = watch
        # One pool per provider, so each provider runs at its own limit whatever the others are doing.
        pools = {
            provider: stack.enter_context(ThreadPoolExecutor(max_workers=cfg.concurrency_for(provider)))
            for provider in sorted({s.provider for s in settings.values()})
        }
        try:
            while running or any(waiting.values()):
                # A call goes to its pool only when its provider and its judge both have a free place, so none
                # ever waits inside a pool. An interrupt therefore leaves nothing queued to start later, and a
                # judge at its own limit does not sit on places that its provider's other judges could use.
                # Of the calls that have a place to go to, the earliest in the plan starts first.
                while ready := [queue for name, queue in waiting.items() if queue and has_room(name)]:
                    watch.checkpoint()  # a stop that was asked for is taken before each call handed out, and starts none
                    i = min(ready, key=lambda queue: queue[0]).popleft()
                    call = jobs[i][0]
                    judge = settings[call.judge]
                    on_provider[judge.provider] += 1
                    on_judge[call.judge] += 1
                    rest = (paintings[call.painting], judge, cfg, layout, adapters[call.judge], sleep, stopping)
                    running[pools[judge.provider].submit(_run_call, *jobs[i], *rest)] = i
                finished = watch.first_finished(running)  # in short waits, and a stop asked for is taken there
                for future in finished:
                    i = running.pop(future)
                    call = jobs[i][0]
                    on_provider[settings[call.judge].provider] -= 1
                    on_judge[call.judge] -= 1
                    record = records[i] = future.result()
                    outcome = "ok" + (" after asking again" if record["asked_again"] else "")
                    if record["status"] != "ok":
                        outcome = "FAILED: " + str(record["error"]).splitlines()[0]
                    progress(f"judged {call.path.relative_to(layout.judgements_root).as_posix()}: {outcome}")
        except BaseException:  # Ctrl-C, above all: start nothing more; leaving the pools waits for the calls in flight
            stopping.set()
            left = sum(len(queue) for queue in waiting.values())
            progress(
                f"stopping: {left} judging calls were not started; of {len(running)} in hand, those already with "
                "their judge finish and are saved, the rest are dropped"
            )
            raise
        finally:
            # Leaving the pools waits for the calls in flight too, but in one wait with no end, during
            # which a further signal might not be answered (see workers.py). After an orderly end none is left.
            watch.all_finished(running)
    return [records[i] for i in sorted(records)]
