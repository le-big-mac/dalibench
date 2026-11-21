"""Where a run keeps its files, and the shape of each record.

Everything a run produces is a file, so results can be judged, scored and
reported again without calling any contestant model.

    runs/<run-id>/config.yaml
    runs/<run-id>/.lock   held by the one `attempt` or `judge` at work in the run (see run_lock); holds its pid
    runs/<run-id>/attempts/<painting>/<width>/<model>/<n>/
        transcript.json   every message, tool call and tool result
        picture.txt       palette lines and rows (complete attempts only)
        picture.png       rendered, solid cells, no smoothing (complete attempts only)
        picture-<W>x<H>.png  the picture as drawn for a peak-size call, every picture of the call at one size
        canvas.json       everything stored, complete or not
        stats.json        state, calls, rejected rows, nudges, tokens, time, cost, and the records of the
                          earlier runs of the attempt that this one replaced
        earlier.json      only while the attempt is being run again: the stats.json that run replaces
        discards/<i>/     transcript and error of each discarded conversation
    runs/<run-id>/judgements/<painting>/<width>/<judge>/<k>.json            one ranking call
    runs/<run-id>/judgements/<painting>/<width>/<judge>/pairwise/<pair>-<k>.json
    runs/<run-id>/judgements/<painting>/peak/<judge>/<model>/<k>.json       one model's sizes ranked
    runs/<run-id>/human/<painting>/<width>/<person>.json
    runs/<run-id>/human/<painting>/<width>/<person>.html                    the blind sheet
    runs/<run-id>/scores.json
    runs/<run-id>/report/index.html
    runs/<run-id>/report/attempts/<painting>/<width>/<model>/<n>.html       one page per attempt

Record shapes (JSON). Optional keys may be absent or null.

stats.json
    state                 "complete" | "incomplete" | "refused" | "error"
    stop                  why the loop ended: "finished" | "max_steps" | "max_minutes" |
                          "nudges_exhausted" | "refused" | "error" | "imported" |
                          "closing_call_failed" | "closing_call_refused" (both complete: every row was
                          stored and only the closing reply was lost; closing_error holds why)
    painting, width, height, encoding, model, attempt
    model_settings        the ModelSettings dict the attempt ran with
    model_calls, tool_calls, nudges, truncated_replies
    nudges_after_cutoff   of the nudges, those that followed a reply cut off by the output limit. Absent on
                          records made before the key existed: unknown, unless transcript.json tells
    rows_rejected, rows_rejected_first_submission, calls_rejected, palette_entries_rejected
    colours_defined, rows_stored
    input_tokens, output_tokens, reasoning_tokens (null if never reported)
    wall_seconds, cost_usd (null if no price)
    api_retries, discards, discarded_model_calls
    refusal_text, refusal_detected_by ("provider" | "text"), refusal_matched (the phrase, for "text")
    error, error_retryable (false: the provider turned the request away outright)
    final_text            the model's closing reply
    protocol              "tools" (or "plain_text" if ever added)
    harness_version, prompt_version, started_at, finished_at
    earlier               the stats records of the earlier runs of this attempt that were replaced by a
                          retry, oldest first, each without an "earlier" of its own. Absent on older records

judgement (ranking, pairwise and peak files share one shape)
    kind                  "ranking" | "pairwise" | "peak"
    painting, judge, repeat
    width, height         the grid size (null for kind "peak")
    model                 the contestant whose sizes are ranked (kind "peak" only)
    labels                {label: attempt id} - the key
    order_shown           labels in the order the pictures were sent
    pictures              {attempt id: SHA-256 of the picture.txt that was shown}. The judge stage makes a call
                          again when the set of attempts it would show has changed or a picture differs. The
                          score stage leaves a record out (as "stale") when a picture it ranked has changed or
                          is gone. A record made on a set that has since grown is used for the pictures it
                          shows until the judge stage, if it still plans the call, makes it again; a ranking
                          or a pair of a kind its set is no longer judged by is left out as "superseded".
                          Absent on imported records.
    reference             SHA-256 of the reference image the judge was shown; null if the call failed before
                          the file was read. A record made against a file that is not the painting's
                          reference now is stale. Absent on older and imported records, which are held to
                          their pictures alone.
    status                "ok" | "failed"
    ranking               labels best first (null if failed)
    ranking_attempts      the same ranking as attempt ids (null if failed)
    notes                 {label: the judge's line about that picture}
    confidence            the judge's confidence sentence
    raw_replies           the judge's reply text, one entry per call (two if asked again)
    parse_errors          why each rejected reply was rejected
    asked_again           true if the first reply failed to parse
    error                 null when ok; otherwise why the call failed (API error, refusal, no valid ranking)
    usage                 {input_tokens, output_tokens, reasoning_tokens, model_calls, api_retries}, summed
                          over the one or two model calls
    judge_settings, prompt_version, seed, cost_usd, wall_seconds, created_at, harness_version

human ranking
    kind "human", painting, width, height, person
    labels, order_shown, ranking, ranking_attempts   as for a judgement; ranking is null until recorded
    notes, created_at, ranked_at
    sheet                 the id of the blind sheet the ranking belongs to
    pictures              {attempt id: SHA-256 of picture.txt} as on the sheet; a ranking of pictures that have
                          since changed is stale and is left out of the agreement figures
    earlier               rankings this person recorded before the current one

An attempt id is "<painting>/<width>/<model>/<n>".

A run directory is known by its config.yaml. What a command is given as
`--run` is a path to one, or else a run id under ./runs (`resolve_run_dir`);
a directory that is not a run directory is never taken for one.

Rules for reading and writing:

- A record is written to a temporary file and renamed into place, so a
  killed process never leaves half a record under its name. The file is not
  forced to the disk first: a machine that goes down can leave one cut short,
  and so can a copy of a run that is read while it is being made.
- A JSON file that does not parse is an error whose message names the file
  and says what to do with it (`read_json`).
- The listings (attempts, judgements, human rankings) leave an unreadable
  record out with one warning line that names it, so that scores and a
  report can be built from what can be read. A layout made with `strict`
  raises instead: `attempt`, `judge` and `human-sheet` act on what is on
  disk, and must not take a record that cannot be read for one that is not
  there.
- One process at a time runs attempts or judging in a run directory
  (`run_lock`). Two would each plan from what was on disk at their start,
  and then empty and rewrite each other's attempt directories.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # not a POSIX system: run_lock then takes no lock
    fcntl = None  # type: ignore[assignment]

STATE_COMPLETE = "complete"
STATE_INCOMPLETE = "incomplete"
STATE_REFUSED = "refused"
STATE_ERROR = "error"
STATES = (STATE_COMPLETE, STATE_INCOMPLETE, STATE_REFUSED, STATE_ERROR)

LOCK_FILE = ".lock"


def _parse(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_json(path: Path | str) -> Any:
    """The JSON a file holds.

    A file that does not parse is a json.JSONDecodeError, as it always was, but one whose message
    starts with the file's path and says what to do: on its own the parser's message names no file,
    and a command that stops on it could not say which of a run's thousand records to look at.
    """
    try:
        return _parse(path)
    except ValueError as e:  # not JSON, or not UTF-8 text
        said, document, at = (e.msg, e.doc, e.pos) if isinstance(e, json.JSONDecodeError) else (str(e), "", 0)
        raise json.JSONDecodeError(
            f"{path} cannot be read as JSON. It was probably cut short while it was written or copied. Restore it, or move "
            "it away: an attempt or a judging call whose record is moved away is made again by the same command. "
            f"The parser stopped with: {said}",
            document, at,
        ) from None


def write_json(path: Path | str, data: Any) -> None:
    """Write atomically, so a killed run never leaves a half-written record.

    A model can type a lone surrogate, which UTF-8 cannot hold; it is written as its JSON escape and reads back unchanged.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", errors="backslashreplace") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class RunLocked(RuntimeError):
    """Another process holds the run directory. The message names its pid and the directory."""


@contextmanager
def run_lock(root: Path | str) -> Iterator[None]:
    """Hold a run directory for the process, for as long as the block lasts: one `attempt` or `judge` at a time.

    An exclusive lock on <run>/.lock, taken without waiting. The holder's pid
    is written into the file. If another process holds the lock, RunLocked is
    raised and nothing else in the directory is touched.

    The kernel drops the lock when the process ends for any reason, `kill -9`
    included, so a lock never outlives its holder. The file stays where it
    is: only the lock on it counts, and a file left by a process that has
    gone locks nobody out.

    On a system without `fcntl` (not POSIX) no lock is taken.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:  # held by another; any other failure (a file system with no locks) is raised as it is
                holder = os.read(fd, 32).decode("ascii", "replace").strip() or "not yet written"
                raise RunLocked(
                    f"another paintbench process (pid {holder}) is at work in {root.resolve()}, and one at a time runs "
                    "attempts or judging in a run directory. Wait for it to end. To end it, send it one signal and wait: "
                    "it finishes what is in flight and then stops. Nothing was run and no file was changed."
                ) from None
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(fd)  # which gives the lock up


def resolve_run_dir(value: Path | str, new: bool = False) -> Path:
    """The run directory that `--run VALUE` means.

    VALUE is taken as a path if a run directory (a directory holding config.yaml) is there.
    Otherwise a bare name is a run id: ./runs/VALUE. A directory that is not a run directory is
    never used, so in the repository `--run docs` is runs/docs and not the docs/ directory.

    When nothing is found, FileNotFoundError lists every path that was tried and what is there.
    With `new` (a command that may start the run, as `import` does) the directory need not exist
    yet: a run id gives ./runs/VALUE, and a longer path is used if nothing is in the way.
    """
    given = Path(value).expanduser()
    is_id = not given.is_absolute() and len(given.parts) == 1 and given.name not in ("", ".", "..")
    candidates = [given, Path("runs") / given] if is_id else [given]
    for path in candidates:
        if RunLayout(path).config_path.is_file():
            return path
    if new:
        if is_id:
            return candidates[-1]
        if not given.exists() or (given.is_dir() and not any(given.iterdir())):
            return given
        raise FileExistsError(
            f"{os.path.abspath(given)} is there already and is not a run directory (it holds no config.yaml), so no run is "
            "started in it. Give another path, or a run id to use ./runs/<id>."
        )

    def found(path: Path) -> str:
        if not path.exists():
            return "nothing is there"
        return "a directory with no config.yaml in it" if path.is_dir() else "not a directory"

    tried = "".join(f"\n  {os.path.abspath(path)}: {found(path)}" for path in candidates)
    raise FileNotFoundError(f"no run directory for {str(value)!r} (a run directory holds config.yaml). Tried:{tried}")


def attempt_id(painting: str, width: int, model: str, n: int) -> str:
    return f"{painting}/{width}/{model}/{n}"


def parse_attempt_id(value: str) -> tuple[str, int, str, int]:
    painting, width, model, n = value.split("/")
    return painting, int(width), model, int(n)


@dataclass(frozen=True)
class AttemptRef:
    painting: str
    width: int
    model: str
    n: int
    directory: Path

    @property
    def id(self) -> str:
        return attempt_id(self.painting, self.width, self.model, self.n)

    @property
    def transcript_path(self) -> Path:
        return self.directory / "transcript.json"

    @property
    def picture_txt(self) -> Path:
        return self.directory / "picture.txt"

    @property
    def picture_png(self) -> Path:
        return self.directory / "picture.png"

    @property
    def canvas_json(self) -> Path:
        return self.directory / "canvas.json"

    @property
    def stats_path(self) -> Path:
        return self.directory / "stats.json"

    def stats(self) -> dict[str, Any] | None:
        return read_json(self.stats_path) if self.stats_path.is_file() else None

    def state(self) -> str | None:
        s = self.stats()
        return s.get("state") if s else None

    def is_complete(self) -> bool:
        return self.state() == STATE_COMPLETE and self.picture_txt.is_file()

    def picture_sha256(self) -> str | None:
        """SHA-256 of picture.txt, which identifies exactly what was typed; None if there is no picture."""
        if not self.picture_txt.is_file():
            return None
        return hashlib.sha256(self.picture_txt.read_bytes()).hexdigest()


class RunLayout:
    """The files of one run directory.

    `strict` decides what the listings do with a record that cannot be read as JSON: False (the
    default) leaves it out with one warning line, True raises read_json's error. See the module
    docstring. `unreadable` holds the paths left out so far, in the order they were met.
    """

    def __init__(self, root: Path | str, strict: bool = False):
        self.root = Path(root)
        self.strict = strict
        self.unreadable: list[Path] = []

    def _listed(self, path: Path) -> Any:
        """A record for a listing. None for a file that has gone since it was found, and for one that
        cannot be read unless the layout is `strict`: that one is left out, with one warning line."""
        try:
            return read_json(path) if self.strict else _parse(path)
        except FileNotFoundError:  # an attempt being made again, or a copy still under way
            return None
        except ValueError as e:
            if self.strict:
                raise
            if path not in self.unreadable:  # named once, however often the run is listed
                self.unreadable.append(path)
                print(f"warning: left out, as it cannot be read as JSON ({e}): {path}", file=sys.stderr, flush=True)
            return None

    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"

    @property
    def attempts_root(self) -> Path:
        return self.root / "attempts"

    @property
    def judgements_root(self) -> Path:
        return self.root / "judgements"

    @property
    def human_root(self) -> Path:
        return self.root / "human"

    @property
    def scores_path(self) -> Path:
        return self.root / "scores.json"

    @property
    def report_dir(self) -> Path:
        return self.root / "report"

    # ---- attempts ----

    def attempt(self, painting: str, width: int, model: str, n: int) -> AttemptRef:
        return AttemptRef(painting, int(width), model, int(n), self.attempts_root / painting / str(width) / model / str(n))

    def attempt_from_id(self, value: str) -> AttemptRef:
        return self.attempt(*parse_attempt_id(value))

    def attempts(self) -> list[AttemptRef]:
        """Every attempt directory on disk that has a stats.json that can be read, in a stable order."""
        found: list[AttemptRef] = []
        if not self.attempts_root.is_dir():
            return found
        for stats in self.attempts_root.glob("*/*/*/*/stats.json"):
            d = stats.parent
            try:
                ref = AttemptRef(d.parents[2].name, int(d.parents[1].name), d.parent.name, int(d.name), d)
            except ValueError:
                continue
            if isinstance(self._listed(stats), dict):
                found.append(ref)
        return sorted(found, key=lambda a: (a.painting, a.width, a.model, a.n))

    def complete_attempts(self, painting: str | None = None, width: int | None = None) -> list[AttemptRef]:
        return [
            a
            for a in self.attempts()
            if (painting is None or a.painting == painting) and (width is None or a.width == width) and a.is_complete()
        ]

    # ---- judgements ----

    def judgement_dir(self, painting: str, width: int, judge: str) -> Path:
        return self.judgements_root / painting / str(width) / judge

    def pairwise_dir(self, painting: str, width: int, judge: str) -> Path:
        return self.judgement_dir(painting, width, judge) / "pairwise"

    def peak_dir(self, painting: str, judge: str, model: str) -> Path:
        return self.judgements_root / painting / "peak" / judge / model

    def judgements(self) -> list[dict[str, Any]]:
        """Every judgement record on disk (ranking, pairwise and peak), each with its path under "_path"."""
        records: list[dict[str, Any]] = []
        if not self.judgements_root.is_dir():
            return records
        for path in sorted(self.judgements_root.rglob("*.json")):
            rec = self._listed(path)
            if isinstance(rec, dict) and rec.get("kind") in ("ranking", "pairwise", "peak"):
                rec["_path"] = str(path)
                records.append(rec)
        return records

    # ---- human rankings ----

    def human_path(self, painting: str, width: int, person: str) -> Path:
        return self.human_root / painting / str(width) / f"{person}.json"

    def human_sheet_path(self, painting: str, width: int, person: str) -> Path:
        return self.human_root / painting / str(width) / f"{person}.html"

    def human_rankings(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not self.human_root.is_dir():
            return records
        for path in sorted(self.human_root.rglob("*.json")):
            rec = self._listed(path)
            if isinstance(rec, dict) and rec.get("kind") == "human":
                rec["_path"] = str(path)
                records.append(rec)
        return records
