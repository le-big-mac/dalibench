"""Bring pictures typed outside the harness into a run directory.

For pictures that models typed without the harness and that were saved in the
picture.txt format (canvas.parse_picture), with the rankings a person or a
judge gave them, if any. An imported attempt has a picture and a stats record
but no transcript, and its stats say so (`stop: imported`).

A manifest is a YAML file that names one painting at one width:

    run_id: typed-elsewhere               # the run is runs/<run_id>; not needed when the import is given a run
    painting: great-wave                  # a directory under paintings/
    width: 32                             # a grid width of the painting; every picture must be that grid
    attempts:
      - model: model-a                    # directory name for the attempt
        file: pictures/model-a.txt        # relative to the manifest
        n: 1                              # optional, default 1
        stats: {wall_seconds: 523}        # optional extra fields for stats.json
      - model: model-b
        file: pictures/model-b.txt
    rankings:                             # optional; each `order` is models of `attempts`, best first
      - {kind: human, name: alice, order: [model-b, model-a], source: ranked on paper}
      - {kind: judge, name: judge-a, repeat: 1, order: [model-a, model-b]}

A ranking's `source` (a note on where it came from) is optional, and so are a
judge ranking's `repeat` (default 1) and `judge_settings` (what is known of
the judge, kept in the record: `{family: anthropic}`, say, which scoring
reads).

Rules:

- An import never replaces anything silently. If an attempt directory or a
  ranking file it would write is already in the run, whoever wrote it, the
  import stops with FileExistsError naming it, unless `overwrite` is given.
  With `overwrite`, an attempt directory is emptied first, so no transcript
  or canvas of the attempt it replaces is left beside the imported picture.
- A manifest is checked whole before anything is written: every name must be
  safe as a directory name, every picture must load at the painting's grid
  size, and no two entries may write the same place. A manifest that fails
  leaves the run directory as it was.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .canvas import Picture, load_picture
from .config import check_name
from .layout import STATE_COMPLETE, AttemptRef, RunLayout, write_json, write_text
from .paintings import Painting, load_painting
from .render import render_picture

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _number(where: str, value: Any) -> int:
    """A whole number of 1 or more: attempt numbers and repeats become directory and file names."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where} must be a whole number of 1 or more, not {value!r}")
    return value


def _load(painting: Painting, width: int, source: Path) -> Picture:
    try:
        picture = load_picture(source)
    except ValueError as e:  # the parser says which line; say which file
        raise ValueError(f"{source}: {e}") from e
    height = painting.height_for(width)
    if (picture.width, picture.height) != (width, height):
        raise ValueError(f"{source}: picture is {picture.width} x {picture.height}, the grid for {painting.slug} is {width} x {height}")
    return picture


def _refusal(what: str, clashes: list[Path]) -> FileExistsError:
    return FileExistsError(
        f"{what} would replace {len(clashes)} of the attempts and rankings already in the run:\n"
        + "".join(f"  {path}\n" for path in clashes)
        + "Nothing was imported. Import into another run directory, or ask for the overwrite (--overwrite) to replace them."
    )


def import_picture(
    layout: RunLayout,
    painting: Painting,
    width: int,
    model: str,
    n: int,
    source: Path | str,
    stats_extra: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> AttemptRef:
    """Copy one picture.txt into the run as a complete attempt, and render it."""
    source = Path(source)
    picture = _load(painting, width, source)
    ref = layout.attempt(painting.slug, width, check_name("model", model), _number("the attempt number", n))
    if ref.directory.exists():
        if not overwrite:
            raise _refusal(f"importing {source.name} as {ref.id}", [ref.directory])
        shutil.rmtree(ref.directory)
    write_text(ref.picture_txt, source.read_text(encoding="utf-8"))  # unchanged
    render_picture(picture).save(ref.picture_png, format="PNG")
    stats: dict[str, Any] = {
        "state": STATE_COMPLETE,
        "stop": "imported",
        "protocol": "imported",
        "painting": painting.slug,
        "width": width,
        "height": picture.height,
        "encoding": picture.encoding,
        "model": model,
        "attempt": n,
        "model_settings": {"name": model, "provider": "imported", "model": model},
        "model_calls": None,
        "tool_calls": None,
        "nudges": None,
        "truncated_replies": None,
        "rows_rejected": None,
        "rows_rejected_first_submission": None,
        "calls_rejected": None,
        "palette_entries_rejected": None,
        "colours_defined": len(picture.palette) or picture.colours_used(),
        "rows_stored": picture.height,
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "wall_seconds": None,
        "cost_usd": None,
        "api_retries": None,
        "discards": 0,
        "refusal_text": None,
        "refusal_detected_by": None,
        "error": None,
        "final_text": "",
        "imported_from": str(source),
        "harness_version": __version__,
        "prompt_version": "imported",
        "started_at": None,
        "finished_at": _now(),
    }
    stats.update(stats_extra or {})
    write_json(ref.stats_path, stats)  # written last: its presence marks the attempt as done
    return ref


def _ranking_record(kind: str, painting: Painting, width: int, name: str, order_ids: list[str], extra: dict[str, Any]) -> dict[str, Any]:
    labels = {LETTERS[i]: attempt for i, attempt in enumerate(sorted(order_ids))}
    by_attempt = {attempt: label for label, attempt in labels.items()}
    ranking = [by_attempt[a] for a in order_ids]
    record: dict[str, Any] = {
        "kind": kind,
        "painting": painting.slug,
        "width": width,
        "height": painting.height_for(width),
        "labels": labels,
        "order_shown": sorted(labels),
        "ranking": ranking,
        "ranking_attempts": list(order_ids),
        "notes": {} if kind != "human" else "",
        "imported": True,
        "created_at": _now(),
        "harness_version": __version__,
    }
    record.update(extra)
    if kind == "human":
        record["person"] = name
        record["ranked_at"] = record["created_at"]
    else:
        record.update({"judge": name, "status": "ok", "confidence": "", "raw_replies": [], "parse_errors": [], "asked_again": False})
    return record


def _entries(manifest: dict[str, Any], key: str, where: str) -> list[dict[str, Any]]:
    entries = manifest.get(key) or []
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise ValueError(f"{where}: `{key}` must be a list of mappings")
    return entries


def import_manifest(
    manifest_path: Path | str,
    run_dir: Path | str | None = None,
    paintings_dir: Path | str | None = None,
    overwrite: bool = False,
) -> RunLayout:
    """Import every attempt and ranking named in a manifest. Returns the run layout."""
    manifest_path = Path(manifest_path).resolve()
    where = manifest_path.name
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{where}: the manifest must be a mapping")
    for key in ("painting", "width") + (() if run_dir else ("run_id",)):
        if manifest.get(key) is None:
            raise ValueError(f"{where}: `{key}` is required")
    painting = load_painting(check_name("painting", str(manifest["painting"])), paintings_dir)
    width = _number(f"{where}: width", manifest["width"])
    layout = RunLayout(Path(run_dir) if run_dir else Path("runs") / check_name("run", str(manifest["run_id"])))

    # First pass: read and check everything, and work out every place that would be written.
    pictures: list[tuple[AttemptRef, Path, Any]] = []
    ids: dict[str, list[str]] = {}
    for entry in _entries(manifest, "attempts", where):
        for key in ("model", "file"):
            if not entry.get(key):
                raise ValueError(f"{where}: an attempt has no `{key}`")
        model = check_name("model", str(entry["model"]))
        ref = layout.attempt(painting.slug, width, model, _number(f"{where}: n of {model}", entry.get("n", 1)))
        if ref.id in ids.get(model, []):
            raise ValueError(f"{where}: {ref.id} is listed twice")
        if not isinstance(entry.get("stats"), (dict, type(None))):
            raise ValueError(f"{where}: `stats` of {model} must be a mapping of extra fields for stats.json")
        source = manifest_path.parent / str(entry["file"])
        _load(painting, width, source)
        pictures.append((ref, source, entry.get("stats")))
        ids.setdefault(model, []).append(ref.id)
    rankings: list[tuple[Path, dict[str, Any]]] = []
    for ranking in _entries(manifest, "rankings", where):
        kind, order = ranking.get("kind"), ranking.get("order")
        if kind not in ("human", "judge"):
            raise ValueError(f"{where}: a ranking's `kind` must be human or judge, not {kind!r}")
        name = check_name("person" if kind == "human" else "judge", str(ranking.get("name") or ""))
        if not isinstance(order, list) or not 0 < len(order) <= len(LETTERS) or len(set(map(str, order))) != len(order):
            raise ValueError(f"{where}: the `order` of {name} must be a list of up to {len(LETTERS)} different model names, best first")
        for model in order:
            if len(ids.get(str(model), [])) != 1:  # none, or several attempts and no way to say which was ranked
                raise ValueError(f"{where}: {name} ranks {model!r}, which must be the model of exactly one attempt in the manifest")
        order_ids = [ids[str(model)][0] for model in order]
        if kind == "human":
            path = layout.human_path(painting.slug, width, name)
            record = _ranking_record("human", painting, width, name, order_ids, {"source": ranking.get("source", "")})
        else:
            repeat = _number(f"{where}: repeat of {name}", ranking.get("repeat", 1))
            path = layout.judgement_dir(painting.slug, width, name) / f"{repeat}.json"
            record = _ranking_record(
                "ranking", painting, width, name, order_ids,
                {"repeat": repeat, "source": ranking.get("source", ""), "judge_settings": ranking.get("judge_settings") or {}},
            )
        if path in [p for p, _ in rankings]:
            raise ValueError(f"{where}: two rankings would be written to {path.relative_to(layout.root)}")
        rankings.append((path, record))
    clashes = [ref.directory for ref, _, _ in pictures if ref.directory.exists()] + [path for path, _ in rankings if path.exists()]
    if clashes and not overwrite:
        raise _refusal(f"importing {where} into {layout.root}", clashes)

    # Second pass: write.
    for ref, source, stats in pictures:
        import_picture(layout, painting, width, ref.model, ref.n, source, stats, overwrite=overwrite)
    for path, record in rankings:
        if record["kind"] == "human":
            # A blind sheet made for the ranking this replaces has another key.
            layout.human_sheet_path(painting.slug, width, record["person"]).unlink(missing_ok=True)
        write_json(path, record)
    return layout
