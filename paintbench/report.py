"""The report stage: one HTML page for a run, and one page for each attempt.

    runs/<run-id>/report/index.html
    runs/<run-id>/report/attempts/<painting>/<width>/<model>/<n>.html

The rules this module implements:

- The index leads with one screen: what was run, the models by rank score at
  the largest width, completion, and how far the judges agree with people.
  Everything below it is detail, and each line of that screen links to its
  detail.
- Each width has standings of its own, and nothing on the page is an overall
  rating: larger sizes tell the models apart more, and a model's standing can
  change with the size. The standings at the largest width lead, with place,
  figure, interval, and the sets each figure rests on. The other widths follow
  under one heading, largest first, in the same form. What holds of every such
  table is said once, under the first. The mean over all sizes is a secondary
  table lower down, named "All sizes together", and gives no place; the other
  tables list the models in the order of the leading standings. A scores file
  from before version 7 has no standings, and is laid out as it always was:
  one table of the mean over all sizes at the top, with the places.
- For each painting and width the page shows the real painting, the real
  painting reduced to the same grid (for scale only: it is never judged and
  never ranked), and then the attempts, best first, each with its model name,
  its rank and one judge's line about it. The order is the judges' pooled
  order from the scores. A set no judge has ranked is shown in a person's
  order if there is one, and otherwise by model name. The page says which.
- Every score comes from score.compute_scores, and this module lays the scores
  out. Its only arithmetic is numbering the places of an order it is given
  (ties share a place), adding up or dividing counts the scores give (the row
  for all models, a share of the planned attempts, the sets that scored some
  model and the shared sets among them, the sets that only attempts never
  started keep out of the shared sets) and, for a scores file from before the
  scores did it, averaging tau over sets.
- Which judgements count is score's rule too (score.is_used): a call that
  failed, is superseded because its set is now judged the other way, or is
  stale because a picture it ranked or the reference image has changed, is
  listed with the reason and shapes no order, note or table here.
- How an attempt ended is worded from what its record says. A refusal the
  provider signalled is a platform block and is called one, with the
  provider's category where the saved response has it; it is never worded as
  the model declining. An attempt whose every reply the output limit cut off
  is said to have ended that way, not to have stopped early. An attempt that
  replaced an earlier run of itself is listed with how that run ended.
- The mean over all sizes is a mean over every set that scored the model, so a
  set in which some model has no complete picture still counts for the others.
  Where the sets differ from model to model the page says so beside the
  figures: how many sets each one rests on, a mark on a model with no score in
  some of the sets, a mark on each set that lacks a model, and beside the
  overall figure the same mean over the shared sets only (those in which every
  scored model has a score), where all models are compared like for like. A
  scores file from before version 5 holds overall figures over the shared sets
  only, and the page words them as that.
- Other places where figures are not like for like are said beside them too:
  complete attempts that no judging call in use ranks (and that running the
  judge stage again will rank them), and a set judged on a partial schedule.
- What a figure cannot carry is said beside it too: planned attempts that were
  never started, the interval on each overall rank score and the neighbouring
  models whose order it leaves open, the neighbours whose difference has no
  interval and whose order is therefore neither separated nor level, how many
  sets a figure from one judge alone rests on, a "best size" that is only the
  best of the sizes a model finished, judges that are themselves contestants,
  and that a rank score is a standing inside one run. The figures are the
  scores'; a scores file from before a figure existed simply leaves its part
  out.
- What the records cannot hold at all (why a stage was stopped, judging calls
  that failed and were made again) reaches the page only as the operator's
  notes: a plain-text file notes.txt in the run directory, which the index
  prints as written, a blank line starting a new paragraph, under a line
  saying that it is not derived from the records. With no such file there is
  no such block. A config that plans no attempt (an import) is no plan, and
  nothing is said about planned attempts.
- A wide table keeps the names of its rows in view while it scrolls, and on a
  narrow screen the leading table comes first and fits the screen.
- Each page is self-contained: images are data URIs, styles are inline, and
  there is no script. A Content-Security-Policy in every page has the browser
  enforce that. The only links that leave the report are the source and
  licence pages of the reference images, which their credit lines must carry.
- Everything that reaches a page is escaped, unless this module built it as
  HTML (the _Html type). Notes, closing texts, refusals, errors and
  transcripts are text a model wrote or caused, and a model's name is a
  directory name, which is no safer.
- A missing piece (no judgement, no painting metadata, a picture that will not
  load, no transcript) becomes a short line saying so. The report builds from
  whatever is on disk.
- A typed picture is embedded as the PNG a judge is shown (solid cells, longest
  side about 640 pixels) and displayed without smoothing. The real painting is
  embedded as a JPEG with its longest side at 640 pixels, to keep pages light.
"""

from __future__ import annotations

import base64
import html
import io
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable
from urllib.parse import quote

from PIL import Image

from . import __version__, score
from .canvas import load_picture
from .layout import (
    STATE_COMPLETE,
    STATE_INCOMPLETE,
    STATE_REFUSED,
    STATES,
    AttemptRef,
    RunLayout,
    parse_attempt_id,
    read_json,
    write_text,
)
from .paintings import Painting, default_paintings_dir, load_painting
from .render import render_canvas_state, render_picture, render_reference_grid

if TYPE_CHECKING:
    from .config import RunConfig

REFERENCE_LONGEST_SIDE = 640  # pixels, for the embedded copy of the real painting
REFERENCE_JPEG_QUALITY = 85
CARD_REM = 15.0  # the narrowest a picture on the index may get, along its longer side
CELL_PX = 2  # and the fewest screen pixels a cell of it may get, so that a 128-wide grid is not shown at a pixel a cell
DASH = "–"  # an unknown figure
NOTES_FILE = "notes.txt"  # in the run directory: the operator's notes, which the index prints as written

_CSP = "default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"
_KINDS = ("ranking", "pairwise", "peak", "human")
_Img = tuple[str, int, int]  # data URI, pixel width, pixel height
_Entry = tuple[str, str, Any]  # attempt id, place ("1", "2=", or "" if unranked), rank information

_STOPS = {
    "max_steps": "reached the limit on model calls",
    "max_minutes": "reached the time limit",
    "nudges_exhausted": "stopped with rows missing after the last nudge",
    "refused": "declined the task",
    "error": "the conversation could not continue",
    "closing_call_failed": "every row was stored; the call for the closing reply failed",
    "closing_call_refused": "every row was stored; the provider refused the call for the closing reply",
}

# stats.json keys and their labels, in the order an attempt's page shows them. Any other key follows under its own name.
_FIGURES = {
    "state": "State",
    "stop": "Why the loop ended",
    "protocol": "Protocol",
    "encoding": "Encoding",
    "strip_rows": "Rows per call, at most",
    "model_calls": "Model calls",
    "tool_calls": "Tool calls",
    "write_calls": "write_rows calls",
    "nudges": "Nudges",
    "nudges_after_cutoff": "Nudges that followed a cut-off reply",
    "truncated_replies": "Replies cut off by the output limit",
    "rows_stored": "Rows stored",
    "colours_defined": "Colours defined",
    "rows_rejected_first_submission": "Rows rejected on first submission",
    "rows_rejected": "Rows rejected, in all",
    "calls_rejected": "Calls rejected",
    "palette_entries_rejected": "Palette entries rejected",
    "input_tokens": "Input tokens",
    "output_tokens": "Output tokens",
    "reasoning_tokens": "Reasoning tokens",
    "discarded_input_tokens": "Input tokens in discarded conversations",
    "discarded_output_tokens": "Output tokens in discarded conversations",
    "wall_seconds": "Wall time, seconds",
    "cost_usd": "Cost, US dollars",
    "api_retries": "API retries",
    "discards": "Conversations discarded",
    "discard_errors": "Why they were discarded",
    "refusal_detected_by": "Refusal detected by",
    "harness_version": "Harness version",
    "prompt_version": "Prompt version",
    "started_at": "Started",
    "finished_at": "Finished",
}
_SHOWN_ELSEWHERE = {"painting", "width", "height", "model", "attempt", "model_settings", "final_text", "refusal_text", "error", "earlier"}
_SETTINGS = {
    "provider": "Provider",
    "model": "Model id",
    "family": "Family",
    "reasoning": "Reasoning",
    "max_output_tokens": "Max output tokens",
    "price": "Price per million tokens",
    "options": "Options",
}
_ROLES = {"system": "System prompt", "user": "Harness", "assistant": "Model", "tool": "Tool reply"}


# ---------------------------------------------------------------- escaping


class _Html(str):
    """Text that is already HTML. Any other value is escaped on its way into a page."""


def _e(value: Any) -> _Html:
    return value if isinstance(value, _Html) else _Html(html.escape(str(value), quote=True))


def _h(template: str, *values: Any) -> _Html:
    """Fill a template written in this module. Every value is escaped unless it is already _Html."""
    return _Html(template.format(*(_e(v) for v in values)))


def _join(parts: Iterable[Any], separator: Any = "") -> _Html:
    return _Html(_e(separator).join(_e(p) for p in parts))


def _attrs(**values: Any) -> _Html:
    """' class="..." id="..."', leaving out the attributes with no value."""
    return _join(_h(' {}="{}"', name.rstrip("_"), value) for name, value in values.items() if value not in (None, ""))


_BREAK = _Html("<br>")


# ---------------------------------------------------------------- figures as text


def _num(value: float | None, places: int = 2) -> str:
    return DASH if value is None else f"{value:.{places}f}"


def _count(value: float | None) -> str:
    """A count or a number of tokens: in full below ten thousand, then to about three figures."""
    if value is None:
        return DASH
    if abs(value) < 10_000:
        return f"{value:,.0f}"
    return f"{value / 1e3:,.1f}k" if abs(value) < 1e6 else f"{value / 1e6:,.2f}M"


def _money(value: float | None) -> str:
    if value is None:
        return DASH
    return f"${value:,.2f}" if value == 0 or abs(value) >= 1 else f"${value:.2g}"


def _starred(entry: dict[str, Any]) -> bool:
    """True for an overall figure (a model's, or one of its by_judge) whose sets are not those of the other figures.

    Since version 5 of the scores that is a model with no score in some of the sets. Before, it was a model with
    scores in sets that its figure, a mean over the shared sets, left out.
    """
    return entry["rank_score"] is not None and entry["coverage_differs"]


def _overall(entry: dict[str, Any] | None) -> str:
    """An overall rank score, marked when it does not rest on the same sets as the other figures (see `_starred`)."""
    return DASH if entry is None else _num(entry["rank_score"]) + ("*" if _starred(entry) else "")


def _signed(value: float | None, places: int = 2) -> str:
    """A difference with its sign, a true minus sign for a negative. One that rounds to nothing carries no sign."""
    if value is None:
        return DASH
    text = f"{value:+.{places}f}"
    return text[1:] if float(text) == 0 else text.replace("-", "−")


def _range(interval: Any, signed: bool = False, places: int = 2) -> str:
    """'0.81 to 0.90' for an interval, and a dash where there is none."""
    if not interval:
        return DASH
    return " to ".join(_signed(end, places) if signed else _num(end, places) for end in interval)


def _percent(share: float | None) -> str:
    """A whole percentage; one that is not all or nothing but would round to it is given a decimal (99.5%)."""
    return score.percent_text(share, DASH)


def _plain(value: Any) -> str:
    """Any stats or settings value as text: a dash for unknown, figures rounded, anything nested as compact JSON."""
    if value is None or value == "" or value == [] or value == {}:
        return DASH
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.1f}" if abs(value) >= 100 else f"{value:.3g}"
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _pairs(value: Any) -> str:
    """{'effort': 'high'} -> 'effort: high'."""
    if not isinstance(value, dict):
        return _plain(value)
    return ", ".join(f"{key}: {_plain(v)}" for key, v in value.items()) or DASH


def _setting(settings: dict[str, Any], key: str) -> str:
    """A reasoning setting or an output limit: left unset it is the provider's default, and absent it was not recorded."""
    if key not in settings:
        return DASH
    return _pairs(settings[key]) if settings[key] else "provider default"


def _distinct(records: Iterable[dict[str, Any]], key: str) -> str:
    """Every value the records hold under one key, such as the versions a run was made with."""
    return ", ".join(sorted({str(r[key]) for r in records if r.get(key)})) or DASH


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _listing(items: Iterable[Any]) -> _Html:
    """'a', 'a and b', 'a, b and c'."""
    items = [_e(i) for i in items]
    return _join(items) if len(items) < 2 else _h("{} and {}", _join(items[:-1], ", "), items[-1])


def _and(items: list[str]) -> str:
    """The same as plain text, for a sentence that is escaped whole: titles and names are not escaped twice."""
    return "".join(items) if len(items) < 2 else f"{', '.join(items[:-1])} and {items[-1]}"


def _places(values: list[float]) -> list[str]:
    """The place of each value in a best-first list (lower is better), ties marked: '1', '2=', '2=', '4'."""
    return [f"{1 + sum(v < value for v in values)}{'=' if values.count(value) > 1 else ''}" for value in values]


# ---------------------------------------------------------------- small pieces of HTML


def _th(text: Any, cls: str = "", span: int = 1, rows: int = 1) -> _Html:
    return _h("<th{}>{}</th>", _attrs(class_=cls, colspan=span if span > 1 else None, rowspan=rows if rows > 1 else None), text)


def _name(text: Any) -> _Html:
    """The cell that names a row."""
    return _h('<th scope="row">{}</th>', text)


def _td(text: Any, cls: str = "") -> _Html:
    return _h("<td{}>{}</td>", _attrs(class_=cls), text)


def _k(text: Any) -> _Html:
    """A cell holding a name or a short key: kept on one line where there is room, so that only prose wraps."""
    return _td(text, "k")


def _who(text: Any) -> _Html:
    """A cell holding one name, of a model or an attempt: kept whole in a narrower window than the other keys are."""
    return _td(text, "k name")


def _p(text: Any) -> _Html:
    """A cell holding prose: it wraps, and is never squeezed narrower than a readable line."""
    return _td(text, "p")


def _sub(text: Any) -> _Html:
    """A smaller line beneath the figure of a cell."""
    return _h('<span class="sub">{}</span>', text)


def _n(text: Any) -> _Html:
    """A cell holding a figure: set to the right, and muted when it is a zero or a dash."""
    return _td(text, "n quiet" if str(text) in ("0", DASH) else "n")


def _tr(cells: Iterable[Any], cls: str = "", anchor: str = "") -> _Html:
    return _h("<tr{}>{}</tr>", _attrs(class_=cls, id=anchor), _join(cells))


def _table(head: list[_Html], body: list[_Html], cls: str = "") -> _Html:
    """`head` and `body` are rows from _tr. The wrapper lets a wide table scroll sideways inside a narrow window.

    `cls` names how the table behaves: "pin" keeps the names of its rows in view while it scrolls sideways, "stack"
    sets each row out as a block on a narrow screen, "fit" wraps the names of its rows sooner than it scrolls.
    """
    return _h(
        '<div class="{}"><table>\n<thead>\n{}\n</thead>\n<tbody>\n{}\n</tbody>\n</table></div>',
        f"scroll {cls}".strip(), _join(head, "\n"), _join(body, "\n"),
    )


def _facts(pairs: Iterable[tuple[Any, Any]], cls: str = "facts") -> _Html:
    """Labels and their values."""
    return _h('<dl class="{}">{}</dl>', cls, _join(_h("<div><dt>{}</dt><dd>{}</dd></div>", label, value) for label, value in pairs))


def _note_p(template: str, *values: Any) -> _Html:
    """A muted paragraph: an explanation, or a line standing in for something that is missing."""
    return _h('<p class="muted">{}</p>', _h(template, *values))


def _state(state: Any) -> _Html:
    """The state as a word with a status dot beside it. The word carries the meaning."""
    return _h('<span class="state {}"><i></i>{}</span>', state if state in STATES else "unknown", state or "no state")


def _said(text: Any) -> _Html:
    """Text somebody wrote, kept as written. A long or many-lined text folds under its first line."""
    text = str(text).strip()
    if len(text) <= 400 and "\n" not in text:
        return _h('<span class="said">{}</span>', text)
    return _h('<details class="long"><summary>{}</summary><pre class="said">{}</pre></details>', text.splitlines()[0][:200], text)


def _img(image: _Img, alt: str, cls: str = "") -> _Html:
    uri, width, height = image
    return _h('<img{} src="{}" width="{}" height="{}" alt="{}">', _attrs(class_=cls), uri, width, height, alt)


def _figure(image: _Img, alt: str, caption: Any, cls: str = "") -> _Html:
    """An image whose displayed size follows its shape (--ar), with its caption."""
    ratio = f"{image[1] / image[2]:.4f}"
    return _h('<figure style="--ar:{}">{}<figcaption>{}</figcaption></figure>', ratio, _img(image, alt, cls), caption)


def _data_uri(image: Image.Image, fmt: str, **options: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **options)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _external(url: Any, text: str) -> Any:
    """A link out of the report, for a reference image's source or licence page. Anything but a web address is left out."""
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        return ""
    return _h('<a href="{}" rel="noopener noreferrer">{}</a>', url, text)


def _page(title: str, body: _Html) -> str:
    return _h(
        '<!doctype html>\n<html lang="en-GB">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta http-equiv="Content-Security-Policy" content="{}">\n<link rel="icon" href="data:,">\n'
        '<title>{}</title>\n<style>{}</style>\n</head>\n<body>\n<div class="page">\n{}\n</div>\n</body>\n</html>\n',
        _Html(_CSP), title, _Html(_CSS), body,
    )


# ---------------------------------------------------------------- reading the run


def _call_key(record: dict[str, Any]) -> tuple:
    kind = record.get("kind")
    return (_KINDS.index(kind) if kind in _KINDS else len(_KINDS), str(record.get("judge")), record.get("repeat") or 0)


def _same_size(record: dict[str, Any]) -> bool:
    return record.get("kind") in ("ranking", "pairwise")


def _where(record: dict[str, Any]) -> tuple[Any, Any]:
    return record.get("painting"), record.get("width")


def _label(record: dict[str, Any], attempt: str) -> str | None:
    return next((label for label, shown in (record.get("labels") or {}).items() if shown == attempt), None)


def _note(record: dict[str, Any], attempt: str) -> str | None:
    """The judge's line about one attempt in one call, if it wrote one."""
    notes = record.get("notes")
    line = notes.get(_label(record, attempt)) if isinstance(notes, dict) else None
    return str(line) if line else None


class _Report:
    """One build: the run's files read once, each image drawn once, and every section of the two kinds of page."""

    def __init__(self, layout: RunLayout, cfg: "RunConfig | None", scores: dict[str, Any], paintings_dir: Path | str | None):
        self.layout, self.cfg, self.scores = layout, cfg, scores
        self.paintings_dir = paintings_dir if paintings_dir is not None else (cfg.paintings_path if cfg else None)
        self.index = layout.report_dir / "index.html"
        self.built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        self.refs = {a.id: a for a in layout.attempts()}
        self.stats = {i: a.stats() or {} for i, a in self.refs.items()}
        self.records = layout.judgements()
        current = score.Current(layout, cfg, self.paintings_dir)
        self.calls = sorted((r for r in self.records if score.is_used(r, current)), key=_call_key)
        self.people = sorted((h for h in layout.human_rankings() if score.is_recorded(h)), key=lambda h: str(h.get("person")))
        self.numbered = any(a.n != 1 for a in self.refs.values())
        self.grids = scores["grids"]
        self.slugs = list(dict.fromkeys(g["painting"] for g in self.grids))
        # Since version 7 of the scores each width has standings of its own, and the page leads with those of the largest
        # width; the mean over all sizes is a secondary table. A file from before has none, and is laid out as it was.
        self.sized = scores.get("scores_version", 0) >= 7
        self.standings = sorted(scores.get("standings_by_width") or [], key=lambda at: -at["width"]) if self.sized else []
        # The order of the models wherever a table lists them: that of the leading standings, then the models with no
        # figure there, as the scores list them (which, with no standings, is best rank score over all sizes first).
        leading = [m["model"] for m in self.standings[0]["models"]] if self.standings else []
        self.models = sorted(scores["models"], key=lambda m: leading.index(m["model"]) if m["model"] in leading else len(leading))
        self.position = {m["model"]: i for i, m in enumerate(self.models)}
        self.anchors = {(g["painting"], g["width"]): f"set-{self.slugs.index(g['painting']) + 1}-{g['width']}" for g in self.grids}
        self.unfinished = {a["attempt"]: a for a in scores["not_complete"]}
        self.retried = {a["attempt"]: a for a in scores["retried"]}
        # Figures added to the scores later than the report: a scores file from before one existed leaves its part out.
        planned = scores.get("planned") or {}
        self.planned = planned if planned.get("attempts") else {}  # a config that plans no attempt (an import) is no plan
        self.unstarted = self.planned.get("not_started", [])
        self.headline = scores.get("headline") or {}
        # Since version 5 of the scores an overall figure is a mean over every set that scored the model. Before, it was
        # a mean over the shared sets only (those in which every scored model has a score), and the page says which.
        self.every_set = scores.get("scores_version", 0) >= 5
        # Since version 6 a judge is held against the others on its own model over every set in which both scored it.
        # Before, over the shared sets only.
        self.own_model_every_set = scores.get("scores_version", 0) >= 6
        self.scored_sets = [s for s in scores["combined"] if any(m["rank_score"] is not None for m in s["models"])]
        self.shared_sets = [s for s in scores["combined"] if s.get("common")]
        # Where some set lacks a model, the figure over the shared sets differs from the overall one and is shown beside it.
        self.like_for_like = self.every_set and len(self.shared_sets) != len(self.scored_sets)
        self.places = {m: place.replace("-", "–") for m, place in (score.tier_places(self.headline) if self.headline.get("tiers") else {}).items()}
        self.judges = scores.get("judges", [])
        self.rows = {picture["attempt"]: picture for picture in scores.get("pictures", [])}
        self.settings = self._settings_on_record()
        self.notes = self._run_notes()
        self._images: dict[Any, _Img | None] = {}
        self._paintings: dict[str, Painting | None] = {}

        # What each call, and each person, made of each attempt: (record, place), judges first, in judge then repeat order.
        self.said: dict[str, list[tuple[dict[str, Any], int]]] = defaultdict(list)
        for record in self.calls + self.people:
            for place, attempt in enumerate(record["ranking_attempts"], 1):
                self.said[attempt].append((record, place))

        # The attempts of each painting and width in the order the page shows them, and what that order is.
        self.orders = {(g["painting"], g["width"]): self._order(g) for g in self.grids}
        self.standing: dict[str, tuple[str, Any, int]] = {}  # attempt -> (place, rank information, how many were ranked)
        self.neighbours: dict[str, tuple[str | None, str | None]] = {}  # attempt -> the one shown before it, and after
        for entries, _what in self.orders.values():
            ranked = sum(1 for _attempt, place, _info in entries if place)
            on_disk = [attempt for attempt, _place, _info in entries if attempt in self.refs]
            for attempt, place, info in entries:
                self.standing[attempt] = (place, info, ranked)
            for before, attempt, after in zip([None] + on_disk, on_disk, on_disk[1:] + [None]):
                self.neighbours[attempt] = (before, after)

    def _run_notes(self) -> list[str] | None:
        """The paragraphs of the operator's notes file. [] with no file or an empty one, None for one that cannot be read."""
        path = self.layout.root / NOTES_FILE
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, ValueError):  # a directory, no permission, or not text
            return None
        paragraphs: list[list[str]] = [[]]
        for line in lines:
            if line.strip():
                paragraphs[-1].append(line.strip())
            elif paragraphs[-1]:  # a blank line ends a paragraph
                paragraphs.append([])
        return [" ".join(paragraph) for paragraph in paragraphs if paragraph]

    # ---- names, paths and links ----

    def ref(self, attempt: str) -> AttemptRef:
        return self.refs.get(attempt) or self.layout.attempt_from_id(attempt)

    def name(self, attempt: str) -> str:
        """The model, and the attempt number once the run has more than one attempt of anything."""
        _painting, _width, model, n = parse_attempt_id(attempt)
        return f"{model} #{n}" if self.numbered else model

    def detail(self, attempt: str) -> Path:
        painting, width, model, n = parse_attempt_id(attempt)
        return self.layout.report_dir / "attempts" / painting / str(width) / model / f"{n}.html"

    @staticmethod
    def href(target: Path, page: Path, anchor: str = "") -> str:
        """A link from one page of the report to another, relative so that the report can be moved or served anywhere."""
        return quote(os.path.relpath(target, page.parent).replace(os.sep, "/")) + anchor

    def link(self, attempt: str, page: Path, text: Any = None) -> _Html:
        text = self.name(attempt) if text is None else text
        if attempt not in self.refs:  # named by a ranking, but no longer on disk: there is no page to link to
            return _e(text)
        return _h('<a href="{}">{}</a>', self.href(self.detail(attempt), page), text)

    def set_link(self, painting: str, width: int) -> _Html:
        """A link, within the index, to the pictures of one painting and width."""
        text = f"{self.title(painting)}, {width} wide"
        anchor = self.anchors.get((painting, width))
        return _h('<a href="#{}">{}</a>', anchor, text) if anchor else _e(text)

    def pair_list(self, pairs: list[list[str]], between: str = " ↔ ") -> str:
        return ", ".join(f"{self.name(a)}{between}{self.name(b)}" for a, b in pairs) or "none"

    # ---- paintings and images ----

    def painting(self, slug: str) -> Painting | None:
        if slug not in self._paintings:
            try:
                self._paintings[slug] = load_painting(slug, self.paintings_dir)
            except (OSError, ValueError, TypeError, AttributeError):  # no metadata, or metadata that cannot be read
                self._paintings[slug] = None
        return self._paintings[slug]

    def title(self, slug: str) -> str:
        painting = self.painting(slug)
        return painting.display_name if painting else slug

    def height(self, slug: str, width: int, known: int | None = None) -> int | None:
        painting = self.painting(slug)
        return known or (painting.grid_heights.get(width) if painting else None)

    def _image(self, key: Any, draw: Callable[[], Image.Image | None], fmt: str, **options: Any) -> _Img | None:
        """Draw an image once. None if it cannot be drawn: the caller says so on the page."""
        if key not in self._images:
            try:
                image = draw()
                self._images[key] = (_data_uri(image, fmt, **options), image.width, image.height) if image else None
            except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):  # missing, or not in the format
                self._images[key] = None
        return self._images[key]

    def real(self, slug: str) -> _Img | None:
        """The real painting, small enough to keep a page light."""

        def draw() -> Image.Image:
            with Image.open(self.painting(slug).reference_path) as reference:
                image = reference.convert("RGB")
            image.thumbnail((REFERENCE_LONGEST_SIDE, REFERENCE_LONGEST_SIDE), Image.LANCZOS)
            return image

        return self._image(("real", slug), draw, "JPEG", quality=REFERENCE_JPEG_QUALITY, optimize=True)

    def reduced(self, slug: str, width: int, height: int | None) -> _Img | None:
        """The real painting reduced to the grid. For scale only: it is never shown to a judge."""
        if not height:
            return None
        return self._image(
            ("reduced", slug, width, height),
            lambda: render_reference_grid(self.painting(slug).reference_path, width, height),
            "PNG", optimize=True,
        )

    def picture(self, attempt: str) -> _Img | None:
        """The typed picture as a judge is shown it, or, for an unfinished attempt, what its canvas holds."""
        ref = self.ref(attempt)

        def draw() -> Image.Image | None:
            if ref.picture_txt.is_file():
                return render_picture(load_picture(ref.picture_txt))
            try:  # a canvas cut short while the run directory was being copied is drawn as no picture, not a failed build
                state = read_json(ref.canvas_json) if ref.canvas_json.is_file() else {}
            except (OSError, ValueError):
                return None
            return render_canvas_state(state) if isinstance(state, dict) and state.get("rows") else None

        return self._image(("picture", attempt), draw, "PNG", optimize=True)

    # ---- the order of a set ----

    def _order(self, grid: dict[str, Any]) -> tuple[list[_Entry], Any]:
        """The attempts of one painting and width, best first, and a sentence saying what that order is."""
        key = (grid["painting"], grid["width"])
        judged = next((s for s in self.scores["combined"] if (s["painting"], s["width"]) == key), None)
        people = [h for h in self.people if _where(h) == key]
        if judged and judged["mode"] == "pairwise":
            places = _places([-e["strength"] for e in judged["order"]])
            entries = [(e["attempt"], place, f"strength {e['strength']:+.2f}") for e, place in zip(judged["order"], places)]
            what = _h(
                "Best first, by Bradley-Terry strength over the {} of {}. This set was judged in pairs.",
                _plural(judged["calls"], "judging call"), _listing(judged["judges"]),
            )
        elif judged:
            places = _places([e["mean_rank"] for e in judged["order"]])
            entries = [(e["attempt"], place, self._rank_info(e)) for e, place in zip(judged["order"], places)]
            what = _h("Best first, by mean rank over the {} of {}.", _plural(judged["calls"], "ranking call"), _listing(judged["judges"]))
        elif people:
            entries = [(attempt, str(place), "") for place, attempt in enumerate(people[0]["ranking_attempts"], 1)]
            what = _h("No judge has ranked this set. Best first, in the order {} ranked them.", people[0].get("person"))
        elif len(grid["complete"]) == 1:  # nothing is waiting here
            entries = []
            what = "Not ranked: it is the only complete attempt at this size, and a ranking needs two."
        else:
            entries = []
            what = "Not ranked yet: no judge and no person has ranked this set. Listed by model name."
        ranked = {attempt for attempt, _place, _info in entries}
        left_out = "" if not entries else "not judged" if judged else "not on the sheet"
        entries += [(attempt, "", left_out) for attempt in grid["complete"] if attempt not in ranked]
        return entries, what

    @staticmethod
    def _rank_info(entry: dict[str, Any]) -> str:
        ranks = entry["ranks"]
        each = f" · over {len(ranks)} calls" if len(ranks) > 9 else f" · ranks {', '.join(map(str, ranks))}" if len(ranks) > 1 else ""
        return f"mean rank {_num(entry['mean_rank'])}{each}"

    # ================================================================ the index

    def index_page(self) -> str:
        sections = [self.summary(), self.pictures(), self.scores_section(), self.agreement(), self.not_complete(), self.facts()]
        return _page(f"{self.scores['run_id']}: DaliBench: The Persistence of Memory", _join(sections, "\n"))

    # ---- the first screen ----

    def summary(self) -> _Html:
        nav = _h(
            '<a href="#pictures">Pictures</a> <a href="#scores">Scores per model</a> <a href="#agreement">Judges and people</a> '
            '<a href="#not-complete">Not complete ({}{})</a> <a href="#run">Run facts and credits</a>',
            len(self.scores["not_complete"]), f", and {len(self.unstarted)} never started" if self.unstarted else "",
        )
        jumps = []
        for slug in self.slugs:
            widths = [g["width"] for g in self.grids if g["painting"] == slug]
            links = _join((_h('<a href="#{}">{}</a>', self.anchors[(slug, width)], width) for width in widths), " · ")
            jumps.append(_h("{} at {}", self.title(slug), links))
        return _h(
            '<header class="top">\n<h1>DaliBench: The Persistence of Memory <span class="by">run {}</span></h1>\n'
            '<p class="muted">Built {} by paintbench {}.</p>\n<nav>{}</nav>\n</header>\n'
            '<section class="summary">\n<div class="what">\n<h2 class="quiet">What was run</h2>\n{}\n{}'
            '<h2 class="quiet">Do the judges agree with people?</h2>\n{}\n{}</div>\n'
            '<div class="models">\n<h2 class="quiet">Models by rank score{}</h2>\n{}\n</div>\n</section>',
            self.scores["run_id"], self.built_at, __version__, nav,
            _join((_h("<p>{}</p>", sentence) for sentence in self._what_was_run()), "\n"), self._notes_block(),
            _join((_h("<p>{}</p>", sentence) for sentence in self._agreement_sentences()), "\n"),
            _note_p("Jump to the pictures: {}.", _join(jumps, "; ")) if jumps else "",
            f" at {self.standings[0]['width']} wide" if self.standings else "",
            self._leading_standings() if self.standings else self._headline_table(),
        )

    def _notes_block(self) -> Any:
        """The operator's notes as written, set apart from what the page works out from the records. Nothing without a file."""
        if self.notes == []:
            return ""
        if self.notes is None:
            body: Any = _note_p("The notes file {} in the run directory could not be read.", NOTES_FILE)
        else:
            body = _join((_h("<p>{}</p>", paragraph) for paragraph in self.notes), "\n")
        return _h(
            '<h2 class="quiet" id="notes">Notes on this run</h2>\n<div class="runnotes">\n<p class="muted small">The operator\'s notes, '
            "as written in {} in the run directory. They are not derived from the run's records, and no figure on this page "
            "depends on them.</p>\n{}\n</div>\n",
            NOTES_FILE, body,
        )

    def _what_was_run(self) -> list[Any]:
        counts, total = self.scores["counts"], self.scores["completion"]["total"]
        if not counts["attempts"]:
            return [f"Nothing yet: there is no attempt under {self.layout.attempts_root}."]
        widths = sorted({g["width"] for g in self.grids})
        encodings = sorted({str(s["encoding"]) for s in self.stats.values() if s.get("encoding")})
        names = [c["model"] for c in self.scores["completion"]["by_model"]]
        ids = {str(self.settings[0].get(name, {}).get("model") or "") for name in names}
        # Two entries may be one model under two settings: they are configurations, and are counted as such.
        who = _plural(len(names), "model")
        if "" not in ids and len(ids) < len(names):
            who = f"{_plural(len(names), 'configuration')} of {_plural(len(ids), 'model')}"
        typed = _h(
            "{} typed {} from memory, on {} {} cells wide{}.",
            who,
            _listing(self.title(slug) for slug in self.slugs) if len(self.slugs) <= 4 else f"{len(self.slugs)} paintings",
            "a grid" if len(widths) == 1 else "grids", _listing(widths),
            f", in {' and '.join(encodings)} encoding" if encodings else "",
        )
        blocked = total["refused_by_provider"]  # a platform block is not the model refusing, and is not called that
        after = total.get("refused_by_provider_after_cut_off", 0)
        cut_first = f", {after} of them after the output limit had already cut off earlier replies" if after else ""
        short = (("incomplete", total["incomplete"]), (f"blocked by the provider{cut_first}", blocked), ("refused", total["refused"] - blocked))
        short += (("ended in an error", total["error"]),)
        short = ", ".join(f"{n} {word}" for word, n in short if n)
        done = _h(
            "{}: {}{}.",
            f"Of the {_plural(total['attempts'], 'attempt')} run" if self.unstarted else _plural(total["attempts"], "attempt"),
            "all complete" if total["complete"] == total["attempts"] else f"{total['complete']} complete",
            _h(', {} (<a href="#not-complete">the list, with reasons</a>)', short) if short else "",
        )
        left_out = self.scores["failed_judgements"]
        if not counts["judgements"]:
            judged = _h(
                "{}, so nothing below is in a judge's order. To judge them: <code>paintbench judge --run {}</code>",
                "No judging call in use ranks them" if counts["judgements_stale"] or counts["judgements_superseded"]
                else "No judge has ranked them yet",
                self.layout.root,
            )
        else:
            judges = sorted({str(r.get("judge")) for r in self.calls})
            failed = counts["judgements_failed"]
            judged = _h(
                "{} ({}) ranked the complete attempts blind against the real painting, in {}.{}",
                _plural(len(judges), "judge"), _listing(judges), _plural(counts["judgements"], "judging call"),
                _h(' {} failed and {} not scored (<a href="#failed">why</a>).', _plural(failed, "more call"), "is" if failed == 1 else "are")
                if failed else "",
            )
            alone = sum(1 for g in self.grids if len(g["complete"]) == 1 and g["not_judged"])
            if alone:  # nothing is waiting for these: a ranking needs two pictures
                judged = _h(
                    "{} {} alone at {} painting and width, so no ranking call holds {}.",
                    judged, f"{_plural(alone, 'complete attempt')} {'stands' if alone == 1 else 'each stand'}",
                    "its" if alone == 1 else "their", "it" if alone == 1 else "them",
                )
        sentences = [sentence for sentence in (typed, self._reasoning_sentence(names), self._planned_sentence(), done) if sentence]
        if self.retried:
            sentences.append(
                _h(
                    '{} {} run again and replaced an earlier run (<a href="#retried">how that run ended</a>).',
                    _plural(len(self.retried), "attempt"), "was" if len(self.retried) == 1 else "were",
                )
            )
        sentences.append(judged)
        why_not = (
            ("stale", "picture", "ranked a picture that has since been replaced or is no longer complete, and {is} not scored"),
            ("stale", "reference", "{was} made against a reference image that has since been replaced, and {is} not scored"),
            ("superseded", None, "{is} of a kind the set is no longer judged by (whole, or in pairs), and {is} not scored"),
        )
        for status, reason, what in why_not:
            n = sum(1 for f in left_out if (f["status"], f["stale_reason"]) == (status, reason))
            if n:
                what = what.format(**{"is": "is", "was": "was"} if n == 1 else {"is": "are", "was": "were"})
                sentences.append(_h('{} {} (<a href="#failed">which</a>).', _plural(n, "judging call"), what))
        unchecked = self.scores["reference_unchecked"]
        if unchecked:
            sentences.append(
                _h(
                    "The reference image of {} was not found, so the judging calls made on {} could not be held against the "
                    "image they were made with. <code>python scripts/fetch_originals.py &lt;painting&gt;</code> fetches and rebuilds one.",
                    _listing(self.title(slug) for slug in unchecked), "it" if len(unchecked) == 1 else "them",
                )
            )
        waiting = sum(len(g["not_judged"]) for g in self.grids if len(g["complete"]) > 1)
        if waiting and counts["judgements"]:
            sentences.append(
                _h(
                    "{} {} not judged yet: no judging call in use ranks {}. Running the judge stage again will rank {}: "
                    "<code>paintbench judge --run {}</code>",
                    _plural(waiting, "complete attempt"), "is" if waiting == 1 else "are", "it" if waiting == 1 else "them",
                    "it" if waiting == 1 else "them", self.layout.root,
                )
            )
        return sentences

    def _reasoning_sentence(self, names: list[str]) -> Any:
        """How each model was set to reason, in one sentence: the condition of a run that a reader most needs beside its scores."""
        groups: dict[str, list[str]] = defaultdict(list)
        for name in sorted(names, key=lambda name: (self.position.get(name, len(self.position)), name)):
            settings = self.settings[0].get(name) or {}
            if "reasoning" in settings:
                groups[_setting(settings, "reasoning")].append(name)
        if not groups:
            return ""
        if len(groups) == 1 and len(names) > 1:
            return _h("Reasoning setting, the same for every one: {}.", next(iter(groups)))
        return _h("Reasoning setting: {}.", _join((_h("{} ({})", setting, _listing(whose)) for setting, whose in groups.items()), "; "))

    def _planned_sentence(self) -> Any:
        """The planned grid against what was run, when some of it was never started."""
        if not self.unstarted:
            return ""
        where: dict[str, list[int]] = defaultdict(list)
        for row in self.planned.get("by_model_width", []):
            if row["not_started"]:
                where[row["model"]].append(row["width"])
        whose = _join((_h("{} at {} wide", model, _listing(widths)) for model, widths in where.items()), "; ")
        # The records hold no reason. What the operator wrote down is not a record, and is pointed to, not vouched for.
        noted = _Html(' The operator\'s notes are <a href="#notes">below</a>.') if self.notes else ""
        return _h(
            '{} of {} planned attempts were run; {} {} never started: {} (<a href="#never-started">the list</a>). '
            "No reason is recorded for them.{}",
            self.planned["run"], self.planned["attempts"], len(self.unstarted), "was" if len(self.unstarted) == 1 else "were", whose, noted,
        )

    def _agreement_sentences(self) -> list[Any]:
        counts = self.scores["counts"]
        by_judge: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.scores["judge_vs_person"]:
            by_judge[str(row["judge"])].append(row)
        if not counts["human_rankings"]:
            waiting = counts["human_rankings_pending"]
            sentences = [
                _h(
                    "No person has ranked a blind sheet yet, so the judges are unchecked.{} To make a sheet: "
                    "<code>paintbench human-sheet --run {} --painting … --width … --person …</code>",
                    f" {_plural(waiting, 'sheet')} made and not yet ranked." if waiting else "", self.layout.root,
                )
            ]
        elif not by_judge:
            people = sorted({str(h.get("person")) for h in self.people})
            sentences = [_h("{} ranked {}, but no judge has ranked the same pictures yet.", _listing(people), _plural(len(self.people), "blind sheet"))]
        else:
            sentences = [self._judge_sentence(judge, rows) for judge, rows in sorted(by_judge.items())]
        between = [sentence for sentence in (self._contestant_judges(), self._between_judges()) if sentence]
        if between or by_judge:
            sentences += between
            sentences[-1] = _h('{} <a href="#agreement">Every comparison</a>.', sentences[-1])
        return sentences

    def _contestant_judges(self) -> Any:
        """A sentence naming the judges that are themselves contestants: the same model id judges and is judged."""
        both = [j for j in self.judges if j["contestants"]]
        if not both:
            return ""
        if len(both) < len(self.judges):
            lead = f"{len(both)} of the {len(self.judges)} judges {'is' if len(both) == 1 else 'are'}"
        else:
            lead = {1: "The judge is", 2: "Both judges are"}.get(len(both), f"All {len(both)} judges are")
        each = (_h("{} is {}, entered as {}", j["judge"], j["model"], _listing(j["contestants"])) for j in both)
        return _h("{} {} in this run: {}.", lead, "a contestant" if len(both) == 1 else "themselves contestants", _join(each, "; "))

    def _between_judges(self) -> Any:
        """How far the judges agree with each other, and each with itself, in a sentence. It is no check against a person."""
        agreement = self.scores.get("judge_agreement")
        if agreement is None:  # scores from before the three levels: the judges' orders only, averaged here over the sets
            taus = [r["tau_b"] for r in self.scores["judge_vs_judge"] if r["tau_b"] is not None]
            agreement = {"between_orders": {"tau_b": _mean(taus) if taus else None, "sets": len(taus), "lowest": min(taus) if taus else None}}
        orders = agreement["between_orders"]
        if not orders["sets"]:
            return ""
        calls, within = agreement.get("between_calls") or {}, agreement.get("within") or []
        single: Any = ""
        if calls.get("pairs"):
            repeats = _h(", and one judge's own repeats at {}", _listing(f"{_num(row['tau'])} ({row['judge']})" for row in within)) if within else ""
            single = _h(" Single calls of different judges agree at {}{}.", _num(calls["tau"]), repeats)
        return _h(
            "Between the judges, Kendall's tau is {} on average over {} (lowest {}), each judge's order of a set being "
            "averaged over its repeats first.{} This is agreement between the judges, not a check of either against a person.",
            _num(orders["tau_b"]), _plural(orders["sets"], "set"), _num(orders["lowest"]), single,
        )

    @staticmethod
    def _judge_sentence(judge: str, rows: list[dict[str, Any]]) -> _Html:
        """One judge against the people who ranked the same pictures, in a sentence."""
        compared = sum(r["repeats_compared"] for r in rows)
        same = sum(r["repeats_identical"] for r in rows)
        near = sum(r["repeats_within_one_adjacent_swap"] for r in rows)
        taus = [r["combined"]["tau_b"] for r in rows if r["combined"]["tau_b"] is not None]
        if not compared:  # judged in pairs only, so there is no single ranking to hold against the person's
            found = f"{_plural(sum(len(r['combined']['swapped_pairs']) for r in rows), 'pair')} in the opposite order"
        elif same == compared:
            found = "its one ranking" if compared == 1 else f"all {compared} of its rankings"
            found += " gave the person's order exactly"
        else:
            found = (
                f"{same} of its {compared} rankings gave the person's order exactly, "
                f"and {near} of {compared} were within one swap of neighbours"
            )
        if taus:
            found += f"; Kendall's tau {_num(_mean(taus))} for its pooled order"
            found += f", averaged over {len(taus)} sheets" if len(taus) > 1 else ""
        return _h("<b>{}</b> against {}: {}.", judge, _listing(sorted({str(r["person"]) for r in rows})), found)

    def _standings_table(self, at: dict[str, Any]) -> _Html:
        """The standings at one width: place, model, figure and its bar, interval, and the sets the figure rests on.

        The models with a record or a planned attempt at that width and no figure there follow the others, with why.
        """
        done = {c["model"]: c for c in self.scores["completion"].get("by_model_width", []) if c["width"] == at["width"]}
        never = Counter(a["model"] for a in self.unstarted if a["width"] == at["width"])
        intervals = any(m["interval"] for m in at["models"])
        last = {name: tier[-1] for tier in at["tiers"] for name in tier}

        def name(model: str) -> _Html:
            return _name(_h('<a href="#model-{}">{}</a>', self.position[model] + 1, model))

        rows = []
        for m in at["models"]:
            place, value, interval = m["place"].replace("-", "–"), m["rank_score"], m["interval"]
            whisker = ""
            if interval:  # the interval, drawn over the bar on the same scale
                low, high = (round(100 * min(max(end, 0.0), 1.0), 1) for end in interval)
                whisker = _h('<i style="left:{}%;width:{}%"></i>', low, round(high - low, 1))
            meter = _h(
                '<span class="meter" title="Rank score {} of 1"><span style="width:{}%"></span>{}</span>',
                _num(value), round(100 * value, 1), whisker,
            )
            cells = [_n(place), name(m["model"]), _td(_num(value), "n"), _td(meter, "bar")]
            cells += [_n(_range(interval).replace(" to ", "–"))] if intervals else []
            # Neighbours whose order is not separated share a place, and the rule between them is dotted.
            cls = ("tied" if "–" in place else "") + (" joined" if last[m["model"]] != m["model"] else "")
            rows.append(_tr(cells + [_td(m["sets_scored"], "n sets")], cls.strip()))
        scored = {m["model"] for m in at["models"]}
        for m in self.models:
            c, missing = done.get(m["model"]), never[m["model"]]
            if m["model"] in scored or not (c or missing):
                continue
            why = ([f"{c['complete']} of {c['attempts']} complete"] if c else []) + ([f"{missing} never started"] if missing else [])
            cells = [_n(DASH), name(m["model"]), _n(DASH), _td("", "bar")] + ([_n(DASH)] if intervals else [])
            rows.append(_tr(cells + [_td(_h("{}{}", DASH, _sub(", ".join(why))), "n sets quiet")]))
        head = [_th("", "n"), _th("Model"), _th("Rank score", "n"), _th("", "bar")] + ([_th("95% interval", "n")] if intervals else [])
        head += [_th(_h("Sets{}", _sub(f"of {at['sets']}")), "n sets")]
        return _table([_tr(head)], rows, "fit headline standings")

    def _partial_at(self, width: int) -> str:
        """A sentence for the standings at one width, where some of its sets had a partial schedule."""
        partial = sum(1 for s in self.scored_sets if s["width"] == width and s["partial_schedule"])
        if not partial:
            return ""
        return (
            f" {_plural(partial, 'set')} at this width had a partial schedule, where a score also depends on which opponents "
            "each picture met: see the caution with the pictures."
        )

    def _leading_standings(self) -> _Html:
        """The standings at the largest width, which the page leads with, and what a reader needs to read any width's standings."""
        at = self.standings[0]
        # Said once, for every width: what the Sets column means for setting two figures side by side, where it means anything.
        fewer = any(m["sets_scored"] < each["sets"] for each in self.standings for m in each["models"])
        key = _h(
            "Rank score: the share of comparisons with other models' pictures at {} wide that a model's pictures won, pooled "
            "over judges, repeats and the paintings that scored the model (Sets). A painting a model has no complete picture "
            "of is no loss to it.{} 1 is always best, 0 always worst. "
            "It is a share of comparisons won inside this run, against this run's own pictures, so figures from different "
            "runs cannot be set against each other.{}",
            at["width"], " A figure on fewer sets is not over the same paintings as the others." if fewer else "", self._partial_at(at["width"]),
        )
        if len(self.standings) > 1:  # with one width there is nothing more to say of the widths, and one table to point to
            links = _Html(
                'Each width has its own standings, the largest first: a larger grid asks more of a model, and a standing can '
                'fall as the grid grows. Below: <a href="#standings">the other widths</a>, <a href="#by-width">the widths side '
                'by side</a>, <a href="#by-set">each painting and width</a>, <a href="#all-sizes">all sizes together</a>.'
            )
        else:
            links = _Html('<a href="#by-set">Per painting and width</a>.')
        intervals = any(m["interval"] for m in at["models"])
        without = [m["model"] for m in at["models"] if not m["interval"]]
        places = {m["model"]: m["place"] for m in at["models"]}
        # The paragraph on the interval speaks for every table below as well: it names a number of paintings only where all draw as many.
        apart = len({each["paintings"] for each in self.standings} | {self.headline.get("paintings", at["paintings"])}) > 1
        notes = self._order_notes(at["pairs"], places, at["paintings"], intervals, without, None, "This width has", apart)
        return _h(
            '{}\n<p class="muted small">{} {}</p>\n{}',
            self._standings_table(at), key, links, _join((_h('<p class="muted small">{}</p>', note) for note in notes), "\n"),
        )

    def _open_pairs(self, pairs: list[dict[str, Any]], without: list[str], named: bool) -> list[Any]:
        """The neighbours an order does not separate, and those it has too little to tell apart, named and not explained again."""
        level = [pair for pair in pairs if pair["separated"] is False]
        unknown = [pair for pair in pairs if pair["separated"] is None] if named else []
        found = [_h("Not separated: {}.", _join((self._level_pair(pair, None, draws=False) for pair in level), "; "))] if level else []
        return found + ([_h("Too little to tell: {}.", _join((self._too_few(pair, without) for pair in unknown), "; "))] if unknown else [])

    def _other_standings(self) -> Any:
        """The standings at the widths the page does not lead with, largest first, under one heading."""
        if len(self.standings) < 2:
            return ""
        parts = []
        for at in self.standings[1:]:
            intervals = any(m["interval"] for m in at["models"])
            found = self._open_pairs(at["pairs"], [m["model"] for m in at["models"] if not m["interval"]], intervals)
            if at["pairs"] and not intervals:  # one sentence for all of its neighbours
                found.append("No interval at this width: no model has scores at two paintings there, so its neighbours share a place.")
            partial = self._partial_at(at["width"]).strip()
            found += [partial] if partial else []
            small = _h('\n<p class="muted small">{}</p>', _join(found, " ")) if found else ""
            parts.append(_h('<h4 id="standings-{}">{} wide</h4>\n{}{}', at["width"], at["width"], self._standings_table(at), small))
        return _h(
            '<h3 id="standings">Standings at the other widths</h3>\n<div class="widths">\n<p class="muted small">Read as the '
            'standings at {} wide, at the top of the page.</p>\n{}\n</div>',
            self.standings[0]["width"], _join(parts, "\n"),
        )

    def _all_sizes(self) -> Any:
        """The mean over every set, whatever its width: a secondary table, which gives no place."""
        models = [m for m in self.scores["models"] if m["rank_score"] is not None]
        # In a run of one width the standings are over the same sets. The table is then kept only for what it adds: the
        # mean over the shared sets where some set lacks a model, or means that are not the figures of the standings
        # (a mean counts each set once, a pooled figure each comparison, and sets can hold different numbers of those).
        if not models or not (self.like_for_like or not score.means_repeat_standings(self.scores)):
            return ""
        starred = any(_starred(m) for m in models)
        intervals = any(m.get("rank_score_interval") for m in models)
        beside = self.like_for_like and bool(self.shared_sets)  # a column for the same mean over the shared sets only
        rows = []
        for m in models:
            figure: Any = _overall(m)
            if starred and not _starred(m):
                figure = _h('{}<span class="ghost">*</span>', figure)  # the room a star takes, so that the figures stay in line
            cells = [_name(m["model"]), _td(figure, "n")] + ([_n(_range(m.get("rank_score_interval")).replace(" to ", "–"))] if intervals else [])
            rows.append(_tr(cells + [_n(m["sets_scored"])] + ([_n(_num(m.get("rank_score_common_sets")))] if beside else [])))
        head = [_th("Model"), _th("Mean rank score", "n")] + ([_th("95% interval", "n")] if intervals else []) + [_th("Sets scored", "n")]
        head += [_th(_h("Over shared sets{}", _sub(_plural(len(self.shared_sets), "set"))), "n")] if beside else []
        key = (
            f"The mean of a model's rank scores over every painting and width that scored it ({self._sets_by_width(self.scored_sets)}), "
            f"each set counting once.{' It is not a rating: the widths are rated apart.' if len(self.standings) > 1 else ''}"
        )
        # With no interval at all, the top of the page has said so of the largest width, which then has none either.
        found = self._open_pairs(self.headline["pairs"], [m["model"] for m in models if not m.get("rank_score_interval")], intervals)
        return _h(
            '<h3 id="all-sizes">All sizes together</h3>\n{}\n<p class="muted small">{}{}{}</p>',
            _table([_tr(head)], rows), key, self._coverage_key(), _join((_h(" {}", sentence) for sentence in found)),
        )

    def _headline_table(self) -> _Html:
        models = self.scores["models"]
        if not models:
            return _note_p("No model has an attempt yet.")
        done = {c["model"]: c for c in self.scores["completion"]["by_model"]}
        plan = {row["model"]: row for row in self.planned.get("by_model", [])}
        judges = sorted({j["judge"] for m in models for j in m["by_judge"]})
        starred = any(_starred(m) for m in models)
        intervals = any(m.get("rank_score_interval") for m in models)
        beside = self.like_for_like and bool(self.shared_sets)  # a column for the same mean over the shared sets only
        rows = []
        for i, m in enumerate(models, 1):
            c, value, interval = done.get(m["model"]), m["rank_score"], m.get("rank_score_interval")
            place = self.places.get(m["model"], DASH if value is None else str(i))
            figure: Any = _overall(m)
            if starred and not _starred(m) and value is not None:
                figure = _h('{}<span class="ghost">*</span>', figure)  # the room a star takes, so that the figures stay in line
            theirs = {j["judge"]: j for j in m["by_judge"]}
            if len(judges) > 1 and value is not None:  # the same figure from each judge alone, in the order the note names
                figure = _h("{}{}", figure, _sub(" · ".join(_num(theirs[j]["rank_score"]) if j in theirs else DASH for j in judges)))
            meter: Any = ""
            if value is not None:
                whisker = ""
                if interval:  # the interval, drawn over the bar on the same scale
                    low, high = (round(100 * min(max(end, 0.0), 1.0), 1) for end in interval)
                    whisker = _h('<i style="left:{}%;width:{}%"></i>', low, round(high - low, 1))
                meter = _h(
                    '<span class="meter" title="Rank score {} of 1"><span style="width:{}%"></span>{}</span>',
                    _num(value), round(100 * value, 1), whisker,
                )
            complete = DASH
            if c:
                planned = plan.get(m["model"])
                complete = f"{c['complete']} of {c['attempts']}"
                if planned and planned["not_started"]:
                    complete += f" run ({planned['planned']} planned)"
            cells = [_n(place), _name(_h('<a href="#model-{}">{}</a>', i, m["model"])), _td(figure, "n"), _td(meter, "bar")]
            cells += [_n(_range(interval).replace(" to ", "–"))] if intervals else []
            # Neighbours whose order is not separated share a place, and the rule between them is dotted.
            last = next((tier[-1] for tier in self.headline.get("tiers", []) if m["model"] in tier), m["model"])
            cls = ("tied" if "–" in place else "") + (" joined" if last != m["model"] else "")
            cells += [_td(m["sets_scored"], "n sets")] + ([_td(_num(m.get("rank_score_common_sets")), "n sets")] if beside else [])
            rows.append(_tr(cells + [_td(complete, "n wrap")], cls.strip()))
        head = [_th("", "n"), _th("Model"), _th("Rank score", "n"), _th("", "bar")] + ([_th("95% interval", "n")] if intervals else [])
        head += [_th("Sets scored", "n sets")]
        head += [_th(_h("Over shared sets{}", _sub(_plural(len(self.shared_sets), "set"))), "n sets")] if beside else []
        head = _tr(head + [_th("Complete", "n")])
        notes = _join((_h('<p class="muted small">{}</p>', note) for note in self._headline_notes(judges)), "\n")
        return _h(
            '{}\n<p class="muted small">{} <a href="#scores">Per painting and width</a>.</p>\n{}',
            _table([head], rows, "fit headline"), self._headline_key(), notes,
        )

    @staticmethod
    def _sets_by_width(sets: list[dict[str, Any]]) -> str:
        """'31 sets: 15 at 16 wide, 15 at 32 and 1 at 64': how many sets a list holds, by width."""
        widths = sorted({s["width"] for s in sets})
        counts = [sum(1 for s in sets if s["width"] == width) for width in widths]
        if len(widths) == 1:
            return f"{_plural(len(sets), 'set')}, at {widths[0]} wide"
        each = [f"{n} at {width}" + (" wide" if i == 0 else "") for i, (n, width) in enumerate(zip(counts, widths))]
        return f"{_plural(len(sets), 'set')}: {_listing(each)}"

    def _nearly_shared(self) -> str:
        """The sets that are not shared only because of attempts never started: how far the count depends on them.

        A set is one of them when every scored model without a complete picture there was planned there and has no
        record there. An attempt that was run and did not complete keeps a set out on its own account.
        """
        scored = {m["model"] for m in self.scores["models"] if m["sets_scored"]}
        never = {(a["painting"], a["width"], a["model"]) for a in self.unstarted}
        run = {(c["painting"], c["width"], c["model"]) for c in self.scores["completion"]["by_cell"]}
        shared = {_where(s) for s in self.shared_sets}
        titles: dict[int, list[str]] = defaultdict(list)
        attempts = 0
        for grid in self.grids:
            key = (grid["painting"], grid["width"])
            lacking = scored - {parse_attempt_id(attempt)[2] for attempt in grid["complete"]}
            if key not in shared and lacking and all(key + (model,) in never and key + (model,) not in run for model in lacking):
                titles[grid["width"]].append(self.title(grid["painting"]))
                attempts += sum(1 for a in self.unstarted if (a["painting"], a["width"]) == key and a["model"] in lacking)
        if not titles:
            return ""
        sets = sum(len(names) for names in titles.values())
        where = "; ".join(f"{_and(names)} at {width} wide" for width, names in sorted(titles.items()))
        return (
            f" {sets} more {'set' if sets == 1 else 'sets'} would be shared had the {'attempt' if attempts == 1 else 'attempts'} "
            f"never started there been run and completed: {where}."
        )

    def _headline_key(self) -> str:
        """What the overall figures are a mean over, and what a reader must not take them for."""
        models = self.scores["models"]
        if not any(m["sets_scored"] for m in models):
            return "No rank scores yet: they come from the judging calls."
        if not self.every_set:
            return self._headline_key_shared_only()
        key = (
            "Rank score: the share of comparisons with other models' pictures that a model's pictures won, pooled over judges "
            "and repeats, and averaged over every painting and width that scored the model "
            f"({self._sets_by_width(self.scored_sets)}). A set in which a model has no complete picture still counts for the "
            "other models; for that model it counts as nothing, and is not a loss. "
            "1 is always best, 0 always worst. "
            "It is a share of comparisons won inside this run, against this run's own pictures, so figures from different "
            "runs cannot be set against each other."
        )
        return key + self._coverage_key()

    def _coverage_key(self) -> str:
        """Beside the means over every set: the mark on a model some set did not score, the figure over the shared sets, and partial schedules."""
        models = self.scores["models"]
        key = ""
        if any(_starred(m) for m in models):
            key += (
                " * No score in some of the sets: the figure is an average over fewer sets than were scored (Sets scored "
                "says how many), and not over the same sets as another model's."
            )
        if self.like_for_like and self.shared_sets:
            key += (
                " Over shared sets: the same average over only the paintings and widths at which every scored model has a score "
                f"({self._sets_by_width(self.shared_sets)}), so that all models are compared on the same sets.{self._nearly_shared()}"
            )
        elif self.like_for_like:
            key += (
                " No painting and width has a score for every scored model, so there is no figure that compares them all "
                f"on the same sets.{self._nearly_shared()}"
            )
        partial = sum(1 for s in self.scored_sets if s["partial_schedule"])
        if partial:
            key += (
                f" {_plural(partial, 'set')} among those scored had a partial schedule, where a score also depends on which "
                "opponents each picture met: see the caution with the pictures."
            )
        return key

    def _headline_key_shared_only(self) -> str:
        """The same for a scores file from before version 5, whose overall figures are a mean over the shared sets only."""
        models = self.scores["models"]
        common = self.shared_sets
        if not common:
            return (
                "No overall rank score: no painting and width has a score for every scored model, so there is no set to compare "
                "them all on. Each set has its own scores."
            )
        key = (
            "Rank score: the share of comparisons with other models' pictures that a model's pictures won, pooled over judges "
            "and repeats, and averaged over the paintings and widths at which every scored model has a score "
            f"({self._sets_by_width(common)}), so that all models are compared on the same sets.{self._nearly_shared()} "
            "1 is always best, 0 always worst. "
            "It is a share of comparisons won inside this run, against this run's own pictures, so figures from different "
            "runs cannot be set against each other."
        )
        if any(_starred(m) for m in models):
            key += " * Also scored in other sets, which this figure leaves out."
        partial = sum(1 for s in common if s["partial_schedule"])
        if partial:
            key += (
                f" {_plural(partial, 'set')} among them had a partial schedule, where a score also depends on which opponents "
                "each picture met: see the caution with the pictures."
            )
        return key

    @staticmethod
    def _share_a_place(pair: dict[str, Any], places: dict[str, str]) -> bool:
        """True where a table gives the two models of a pair of neighbours the same place."""
        first, second = places.get(pair["higher"]), places.get(pair["lower"])
        return first is not None and first == second

    @staticmethod
    def _level_pair(pair: dict[str, Any], sets: dict[str, int] | None, draws: bool = True) -> _Html:
        """Two neighbours whose order is not separated, with their difference and its interval.

        With `draws`, how often the draws reversed the order; with `sets`, the sets each figure is a mean over where those differ.
        """
        first, second = (sets or {}).get(pair["higher"]), (sets or {}).get(pair["lower"])
        rests_on = f"; over {first} and {second} sets" if sets is not None and first != second else ""
        how_often = f", the order reversed in {_percent(pair['share_reversed'])} of the draws" if draws else ""
        return _h("{} and {} (difference {}, interval {}{}{})", pair["higher"], pair["lower"],
                  _signed(pair["difference"]), _range(pair["interval"], True), how_often, rests_on)

    @staticmethod
    def _too_few(pair: dict[str, Any], without: list[str]) -> _Html:
        """Two neighbours whose difference has no interval, and which of them has scores at fewer than two paintings."""
        few = [name for name in (pair["higher"], pair["lower"]) if name in without]
        why = _h("{} {} scored on fewer than two paintings", _listing(few), "was" if len(few) == 1 else "were each") if few else "no interval"
        return _h("{} and {} ({})", pair["higher"], pair["lower"], why)

    def _headline_notes(self, judges: list[str]) -> list[Any]:
        """What the interval is and is not, which neighbours it leaves open or cannot tell apart, and what one judge alone changes."""
        headline, models = self.headline, self.scores["models"]
        if not headline.get("sets"):
            return []
        intervals = any(m.get("rank_score_interval") for m in models)
        without = [m["model"] for m in models if not m.get("rank_score_interval")]
        sets = {m["model"]: m["sets_scored"] for m in models} if self.every_set else None
        notes = self._order_notes(headline["pairs"], self.places, headline["paintings"], intervals, without, sets, "The run has")
        if len(judges) > 1:
            alone = self._alone()
            # A figure from one judge alone can rest on fewer sets than that judge scored: the table by judge says how many.
            fewer = self.every_set and any(_starred(j) for m in models for j in m["by_judge"])
            notes.append(
                _h(
                    "Beneath each figure, smaller: the same from each judge alone ({}).{} {}",
                    _join(judges, ", then "),
                    _Html(' How many sets each of those rests on is in the table of <a href="#by-judge">rank score by judge</a>.') if fewer else "",
                    _h("By one judge alone: {}.", _join(alone, "; ")) if alone
                    else "No order of two neighbours is reversed, or left open where the pooled figure separates it, by one judge alone.",
                )
            )
        return notes

    def _order_notes(
        self, pairs: list[dict[str, Any]], places: dict[str, str], paintings: int, intervals: bool, without: list[str],
        sets: dict[str, int] | None, has: str, apart: bool = False,
    ) -> list[Any]:
        """What the interval is and is not, and which neighbours of one order it leaves open or cannot tell apart.

        The order is that of one width's standings, or the one over all sizes of a scores file from before the standings.
        `places` are its places, `paintings` the paintings it resamples, `intervals` whether any of its figures has an
        interval and `without` the models whose figure has none. With `sets`, two neighbours whose figures are means over
        different numbers of sets say so. `has` words whose paintings they are. With `apart` the tables the notes speak
        for do not all resample as many paintings, and what an interval is is said without their number.
        """
        notes: list[Any] = []
        several = paintings >= 2  # paintings to resample: a model may still have scores at one of them only
        if intervals:
            once = "" if self.numbered else ", which this run cannot measure: each model made one attempt at each painting and width"
            notes.append(
                f"95% interval (and the line across each bar, where the bars are shown): where the figure falls when "
                f"{'the paintings of its table' if apart else f'the {paintings} paintings'} are drawn again at random "
                f"({self.headline['draws']:,} draws, each painting counted as often as "
                f"it is drawn). It says how far a figure "
                f"depends on which paintings were chosen. It does not include the variation between one attempt and another by the "
                f"same model{once}. So it is if anything too narrow."
            )
        elif several:
            notes.append(
                "No interval is given: it comes from drawing the paintings again at random, and that needs a model to have scores "
                f"at two paintings or more. {has} {paintings} paintings, but no model was scored on more than one of them."
            )
        else:
            none = "No interval is given: it comes from drawing the paintings again at random, and that needs at least two paintings."
            if pairs and all(self._share_a_place(pair, places) for pair in pairs):
                none += " Without one there is too little to tell the order of any two neighbours, so the models share a place."
            notes.append(none)
        shared = [pair for pair in pairs if pair["separated"] is False]
        # A difference with no interval, where there are two paintings or more: one of its two models has scores at fewer than two
        # of them. (With one painting no difference has an interval, and the sentence above says so of all of them.)
        unknown = [pair for pair in pairs if pair["separated"] is None] if intervals or several else []
        if shared:
            notes.append(
                _h(
                    "A shared place means the order of those neighbours is not separated{}: the interval of their difference, with "
                    "the same paintings drawn for both, includes zero, so it could go either way. Here: {}.",
                    ", or that there is too little to tell. Not separated" if unknown else "",
                    _join((self._level_pair(pair, sets) for pair in shared), "; "),
                )
            )
        elif pairs and all(pair["separated"] for pair in pairs):  # every one of them: a pair with no interval is not separated
            notes.append(
                "The order of every two neighbours is separated: no interval of a difference, with the same paintings drawn for both, "
                "includes zero."
            )
        if unknown:
            others = [pair for pair in pairs if pair["separated"] is not None]
            notes.append(
                _h(
                    "Too little to tell: {}. A difference has no interval when either model was scored on fewer than two "
                    "paintings, so the order of those neighbours is neither separated nor shown to be level{}.{}",
                    _join((self._too_few(pair, without) for pair in unknown), "; "),
                    ", and they share a place" if all(self._share_a_place(pair, places) for pair in unknown) else "",
                    " The order of every other two neighbours is separated: no interval of a difference, with the same paintings "
                    "drawn for both, includes zero." if others and all(pair["separated"] for pair in others) else "",
                )
            )
        return notes

    def _alone(self) -> list[Any]:
        """The neighbours of the order over all sizes that one judge alone has the other way round, or does not separate where the pooled figure does."""
        models = self.scores["models"]
        alone = []
        theirs = {m["model"]: {j["judge"]: j for j in m["by_judge"]} for m in models}

        def rests_on_alone(judge: str, first: str, second: str) -> str:
            """The sets two figures from one judge alone are means over, where either is fewer than that judge's sets."""
            counts = [theirs[name][judge] for name in (first, second) if judge in theirs[name]]
            if not self.every_set or len(counts) < 2 or not any(_starred(entry) for entry in counts):
                return ""
            return f"; over {counts[0]['sets_scored']} and {counts[1]['sets_scored']} of its {_plural(counts[0]['sets_all'], 'set')}"

        for pair in self.headline.get("pairs") or []:
            for j in pair["by_judge"]:
                if j["judge"] in pair["reversed_by"]:
                    higher, lower = theirs[pair["lower"]][j["judge"]]["rank_score"], theirs[pair["higher"]][j["judge"]]["rank_score"]
                    places = 2 if _num(higher) != _num(lower) else 3  # enough figures to see which is ahead
                    figures = (_num(higher, places), _num(lower, places))
                    alone.append(
                        _h("{} has {} above {} ({} against {}{})", j["judge"], pair["lower"], pair["higher"], *figures,
                           rests_on_alone(j["judge"], pair["lower"], pair["higher"]))
                    )
                elif pair["separated"] and j["separated"] is False:
                    alone.append(
                        _h("{} over {} is not separated by {} alone (difference {}, interval {}{})", pair["higher"], pair["lower"],
                           j["judge"], _signed(j["difference"]), _range(j["interval"], True),
                           rests_on_alone(j["judge"], pair["higher"], pair["lower"]))
                    )
        return alone

    # ---- the pictures ----

    def pictures(self) -> _Html:
        if not self.slugs:
            return _Html('<h2 id="pictures">Pictures</h2>\n<p class="muted">There are no attempts to show yet.</p>')
        return _h('<h2 id="pictures">Pictures</h2>\n{}', _join((self._painting_section(slug) for slug in self.slugs), "\n"))

    def _painting_section(self, slug: str) -> _Html:
        painting, real = self.painting(slug), self.real(slug)
        byline = f"{painting.artist}, {painting.date}" if painting else ""
        if painting is None:
            where = Path(self.paintings_dir or default_paintings_dir()) / slug / "meta.yaml"
            figure = _h('<p class="gap">The real painting cannot be shown: there is no painting metadata at {}.</p>', where)
        elif real is None:
            figure = _h('<p class="gap">The real painting cannot be shown: no reference image could be read at {}. If it is missing, `python scripts/fetch_originals.py {}` fetches and rebuilds it.</p>', painting.reference_path, slug)
        else:
            caption = _Html('The real painting. <a href="#credits">Credit</a>')
            figure = _figure(real, f"The real painting: {painting.display_name}", caption)
        sets = [self._set_section(g) for g in self.grids if g["painting"] == slug]
        return _h(
            '<section class="painting">\n<h3>{} <span class="by">{}</span></h3>\n<div class="split">\n'
            '<aside class="real">{}</aside>\n<div class="sets">\n{}\n</div>\n</div>\n</section>',
            self.title(slug), byline, figure, _join(sets, "\n"),
        )

    def _set_section(self, grid: dict[str, Any]) -> _Html:
        slug, width = grid["painting"], grid["width"]
        key = (slug, width)
        height = self.height(slug, width, grid["height"])
        entries, what = self.orders[key]
        cards = [self._card(attempt, place, info) for attempt, place, info in entries]
        reduced = self.reduced(slug, width, height)
        if reduced and cards:
            cards.insert(0, self._scale_card(reduced, width, height))
        if cards:
            # Wide enough for the shape of the picture, and for about two screen pixels to each cell of a large grid.
            card_min = round(max(CARD_REM * min(1.0, width / height if height else 1.0), (CELL_PX * width + 20) / 16), 2)
            lines = [_note_p("{}", what)] + self._cautions(grid)
            lines.append(_h('<ol class="cards" style="--card-min:{}rem">\n{}\n</ol>', card_min, _join(cards, "\n")))
        else:
            lines = [_note_p("No complete attempt at this size.")]
        unfinished = sum(1 for a in self.scores["not_complete"] if (a["painting"], a["width"]) == key)
        if unfinished:
            lines.append(
                _note_p(
                    '{} here did not complete and {} not judged: <a href="#not-complete">the list</a>.',
                    _plural(unfinished, "attempt"), "was" if unfinished == 1 else "were",
                )
            )
        unstarted = sorted({self.name(a["attempt"]) for a in self.unstarted if (a["painting"], a["width"]) == key})
        if unstarted:
            lines.append(
                _note_p(
                    '{} {} planned here and never started: <a href="#never-started">the list</a>.',
                    _listing(unstarted), "was" if len(unstarted) == 1 else "were",
                )
            )
        for human in self.people:
            if _where(human) == key and isinstance(human.get("notes"), str) and human["notes"].strip():
                template = '<div class="notes"><span class="muted">Notes from {}, who ranked this sheet:</span> {}</div>'
                lines.append(_h(template, human.get("person"), _said(human["notes"])))
        lines.append(self._calls_table(key))
        return _h(
            '<section class="set" id="{}">\n<h4>{} × {} cells</h4>\n{}\n</section>',
            self.anchors[key], width, height or "?", _join(lines, "\n"),
        )

    def _cautions(self, grid: dict[str, Any]) -> list[_Html]:
        """What the order of one set leaves out or cannot carry: attempts no call in use ranks, and a partial schedule."""
        lines = []
        waiting = len(grid["not_judged"])
        if waiting and len(grid["complete"]) > 1:
            lines.append(
                _h(
                    '<p class="caution">{} here {} not judged yet: no judging call in use ranks {}. Running the judge stage '
                    "again will rank {}.</p>",
                    _plural(waiting, "complete attempt"), "is" if waiting == 1 else "are", "it" if waiting == 1 else "them",
                    "it" if waiting == 1 else "them",
                )
            )
        elif waiting and any(place for _attempt, place, _info in self.orders[_where(grid)][0]):
            # Beside an order of pictures that are gone. Where it stands alone, the line above the pictures has said so.
            lines.append(_note_p("One complete attempt alone cannot be judged: a ranking needs at least two."))
        judged = next((s for s in self.scores["combined"] if _where(s) == _where(grid)), None)
        if judged and judged["partial_schedule"]:
            if judged["mode"] == "pairwise":
                advice = " Go by the order shown here: Bradley-Terry strength allows for how strong those opponents were."
            else:
                advice = ""
            lines.append(
                _h(
                    '<p class="caution">A partial schedule: not every pair of pictures of different models was compared here, so '
                    "a model's rank score in this set also depends on which opponents its pictures met.{}</p>",
                    advice,
                )
            )
        return lines

    @staticmethod
    def _scale_card(reduced: _Img, width: int, height: int) -> _Html:
        return _h(
            '<li class="card scale">{}<p class="who">The real painting at {} × {}</p>'
            '<p class="muted">Each cell is the average of its area. Shown for scale: it was never judged.</p></li>',
            _img(reduced, f"The real painting reduced to {width} by {height} cells", "typed"), width, height,
        )

    def _card(self, attempt: str, place: str, info: Any) -> _Html:
        """One attempt on the index: its picture, its place and model, its rank, and one judge's line about it."""
        image = self.picture(attempt)
        if image:
            picture = self.link(attempt, self.index, _img(image, f"Typed by {self.name(attempt)}", "typed"))
        else:
            picture = _Html('<p class="gap">No picture could be read for this attempt.</p>')
        lines = [_h('<p class="who"><span class="place">{}</span> {}</p>', place, self.link(attempt, self.index))]
        if info:
            lines.append(_note_p("{}", info))
        sheets = [(record, at) for record, at in self.said[attempt] if record.get("kind") == "human"]
        if sheets:
            lines.append(_note_p("{}", "; ".join(f"{r.get('person')}: {at} of {len(r['ranking_attempts'])}" for r, at in sheets)))
        notes = [(record, _note(record, attempt)) for record, _at in self.said[attempt] if _same_size(record)]
        notes = [(record, note) for record, note in notes if note]
        if notes:
            record, note = notes[0]
            lines.append(
                _h(
                    '<p class="note"><span class="muted">{}, repeat {}:</span> {} {}</p>',
                    record.get("judge"), record.get("repeat"), note,
                    self.link(attempt, self.index, f"all {len(notes)} notes") if len(notes) > 1 else "",
                )
            )
        return _h('<li class="card" data-attempt="{}">{}{}</li>', attempt, picture, _join(lines))

    def _calls_table(self, key: tuple[str, int]) -> Any:
        """Every usable judging call on one set: who ranked what in which order, and how sure they said they were."""
        calls = [r for r in self.calls if _same_size(r) and _where(r) == key]
        if not calls:
            return ""
        rows = [
            _tr(
                [
                    _k(r.get("judge")),
                    _k(f"{'pair' if r['kind'] == 'pairwise' else 'ranking'}, repeat {r.get('repeat')}"),
                    _p(", ".join(self.name(a) for a in r["ranking_attempts"])),
                    _p(_said(r["confidence"]) if r.get("confidence") else DASH),
                ]
            )
            for r in calls
        ]
        head = _tr([_th("Judge"), _th("Call"), _th("Order, best first"), _th("The judge's confidence")])
        return _h(
            '<details class="calls"><summary>The {} behind this order</summary>\n{}\n</details>',
            _plural(len(calls), "judging call"), _table([head], rows),
        )

    # ---- scores per model ----

    def scores_section(self) -> _Html:
        title = _Html('<h2 id="scores">Scores per model</h2>')
        if not self.scores["models"]:
            return _join([title, _note_p("No model has an attempt yet.")], "\n")
        if self.sized:  # each width's standings first, then the figures side by side, set by set, and last their mean over all sizes
            parts = [title, self._other_standings(), self._by_width_table(), self._rank_score_table(), self._all_sizes()]
        else:
            parts = [title, self._rank_score_table(), self._by_width_table()]
        return _join([part for part in parts + [self._peak_table(), self._completion_table()] if part], "\n")

    def _rank_score_table(self) -> _Html:
        sets = self.scores["combined"]
        scored = {(s["painting"], s["width"], m["model"]): m["rank_score"] for s in sets for m in s["models"]}
        keys = [(g["painting"], g["width"]) for g in self.grids]
        any_score = any(value is not None for value in scored.values())
        beside = self.like_for_like and bool(self.shared_sets)  # a column for the same mean over the shared sets only
        # The marks on a width: judged in pairs, a partial schedule, and a set that lacks a score for some scored model.
        if self.sized:  # no column of means here: the table of all sizes together has them
            lacking = "Not every scored model has a score here" + (", so it is not one of the shared sets." if beside else ".")
        elif not self.every_set:
            lacking = "Not part of Overall: not every scored model has a score here."
        elif beside:
            lacking = (
                "Not every scored model has a score here: the set counts in Overall for the models it scored, and is not "
                "one of the shared sets."
            )
        else:
            lacking = "Not every scored model has a score here: the set counts in Overall for the models it scored."
        notes = {
            "†": "Judged in pairs, because there were more attempts than one ranking call takes.",
            "‡": "A partial schedule: not every pair of pictures of different models was compared, so a score here also depends "
            "on which opponents each picture met. Go by the order of that set's pictures.",
            "*": lacking,
        }
        marked = {
            "†": {_where(s) for s in sets if s["mode"] == "pairwise"},
            "‡": {_where(s) for s in sets if s["partial_schedule"]},
            "*": set(keys) - {_where(s) for s in self.shared_sets} if any_score else set(),
        }
        done = {(c["painting"], c["width"], c["model"]): c for c in self.scores["completion"]["by_cell"]}
        unstarted = {(a["painting"], a["width"], a["model"]) for a in self.unstarted}
        top = [_th("Model", "pin", rows=2)]
        top += [] if self.sized else [_th("Overall", "n", rows=2)] + ([_th("Over shared sets", "n", rows=2)] if beside else [])
        top += [_th(self.title(slug), "group", span=sum(1 for painting, _width in keys if painting == slug)) for slug in self.slugs]
        widths = []
        for key in keys:
            marks = "".join(mark for mark, where in marked.items() if key in where)
            widths.append(_th(_h('<a href="#{}">{}{}</a>', self.anchors[key], key[1], f" {marks}" if marks else ""), "n"))
        rows = []
        for i, m in enumerate(self.models, 1):
            cells = [self._score_cell(scored.get(key + (m["model"],)), done.get(key + (m["model"],)), key + (m["model"],) in unstarted) for key in keys]
            figures = [] if self.sized else [_n(_num(m["rank_score"]))] + ([_n(_num(m.get("rank_score_common_sets")))] if beside else [])
            rows.append(_tr([_name(m["model"])] + figures + cells, anchor=f"model-{i}"))
        never = " “Never started” means the config planned it there and the run holds no record of it." if unstarted else ""
        marks = "".join(f" {mark} {note}" for mark, note in notes.items() if marked[mark])
        if self.sized:
            return _h(
                '<h3 id="by-set">Rank score by painting and width</h3>\n{}\n<p class="muted small">Pooled over judges and repeats. '
                "An empty cell means the model was not run at that size; {} means it has no score there.{}{}</p>",
                _table([_tr(top), _tr(widths)], rows, "pin"), DASH, never, marks,
            )
        if not self.every_set and (not marked["*"] or self.shared_sets):
            overall = "Overall is the mean over the sets in which every scored model has a score, so that all models are compared on the same sets."
        elif not self.every_set:
            overall = "There is no Overall figure: no set has a score for every scored model."
        else:
            overall = (
                "Overall is the mean over every set that scored the model. A set in which a model has no complete picture "
                "counts for the other models; for that model it counts as nothing, and is not a loss."
            )
            if beside:
                overall += (
                    f" Over shared sets is the same mean over only the {_plural(len(self.shared_sets), 'set')} in which every "
                    "scored model has a score, so that all models are compared on the same sets."
                )
            elif self.like_for_like:
                overall += " No set has a score for every scored model, so no figure compares them all on the same sets."
            overall += self._overall_counts()
        return _h(
            '<h3>Rank score</h3>\n{}\n<p class="muted small">Rank score by painting and grid width, pooled over judges and '
            "repeats: the share of comparisons with other models' pictures that the model's pictures won. It is a standing "
            "inside this run, against this run's own pictures: it cannot be set against a figure from another run. {} "
            "An empty cell means the model was not run at that size; {} means it has no score there.{}{}</p>",
            _table([_tr(top), _tr(widths)], rows, "pin"), overall, DASH, never, marks,
        )

    def _overall_counts(self) -> str:
        """A sentence for a table that repeats the Overall column bare, where its figures rest on different numbers of sets."""
        if self.sized or not self.every_set or not any(_starred(m) for m in self.scores["models"]):
            return ""
        return (
            " The Overall figures do not all rest on the same number of sets: the table at the top of the page says how many "
            "each rests on (Sets scored)."
        )

    @staticmethod
    def _score_cell(value: float | None, done: dict[str, Any] | None, unstarted: bool = False) -> _Html:
        """A model's rank score in one set, with a line beneath it when not all of its attempts there completed."""
        if value is None and not done:
            return _td(_h("{}{}", DASH, _sub("never started")), "n quiet") if unstarted else _td("", "n")
        short = ""
        if done and done["complete"] < done["attempts"]:
            short = _sub(f"{done['complete']} of {done['attempts']} complete")
        return _n(_h("{}{}", _num(value), short)) if value is not None else _td(_h("{}{}", DASH, short), "n quiet")

    def _by_width_table(self) -> Any:
        """Each model's rank score at each width, over every set there that scored it, and how many sets that is where it differs.

        In a scores file from before version 5 the figures are over the sets every scored model shares at that width.
        """
        widths = self.headline.get("sets_by_width") or []
        if not widths:
            return ""
        at = {(m["model"], w["width"]): w for m in self.scores["models"] for w in m.get("by_width", [])}
        fewer = False
        rows = []
        for m in self.models:
            cells = []
            for width in widths:
                here = at.get((m["model"], width["width"]))
                if here is None or here["rank_score"] is None:
                    cells.append(_n(DASH))
                    continue
                beneath = _sub(f"{here['wins']:,} of {here['comparisons']:,}")
                if here["sets"] != width["sets"]:  # some set at this width did not score the model
                    beneath = _h("{}{}", beneath, _sub(f"{here['sets']} of {_plural(width['sets'], 'set')}"))
                    fewer = True
                cells.append(_n(_h("{}{}", _num(here["rank_score"]), beneath)))
            rows.append(_tr([_name(m["model"])] + ([] if self.sized else [_n(_num(m["rank_score"]))]) + cells))
        head = [_th("Model", "pin")] + ([] if self.sized else [_th("Overall", "n")])
        head += [_th(_h("{} wide{}", w["width"], _sub(_plural(w["sets"], "set" if self.every_set else "shared set"))), "n") for w in widths]
        absent = sorted({g["width"] for g in self.grids} - {w["width"] for w in widths})
        lacking = "a score" if self.every_set else "a score for every scored model"
        none = f" No set at {_listing(absent)} wide has {lacking}, so there is no column for it." if absent else ""
        if self.sized and len(widths) < 2:  # one width: its figures are in the standings, and there is nothing to set them beside
            return ""
        if self.sized:  # the standings have said what a figure is, and which neighbours each width does not separate
            return _h(
                '<h3 id="by-width">Rank score by width</h3>\n{}\n<p class="muted small">The figures of the standings side by side, '
                "with the comparisons won over those made beneath each.{}{} Each is a standing against the models and paintings "
                "scored at that width, not a measure of the pictures themselves: a change across widths is not necessarily a "
                "change against the same field.</p>",
                _table([_tr(head)], rows, "pin"),
                " Beneath that, where some set at the width did not score the model: the sets its figure rests on." if fewer else "", none,
            )

        def figure(top: dict[str, Any], which: str) -> str:
            """One of the top two figures at a width, with the sets it rests on where the two rest on different numbers."""
            sets = [at.get((top[side], top["width"]), {}).get("sets") for side in ("higher", "lower")]
            mine = sets[0] if which == "higher" else sets[1]
            return _num(top[f"{which}_score"]) + (f" over {_plural(mine, 'set')}" if self.every_set and sets[0] != sets[1] else "")

        level = [
            _h(
                "<p class=\"small\">At {} wide the top two, {} and {}, are not separated ({} against {}): the interval of their "
                "difference there, with the same paintings drawn for both, runs from {} and includes zero, so their order at "
                "that width could go either way.</p>\n",
                top["width"], top["higher"], top["lower"], figure(top, "higher"), figure(top, "lower"), _range(top["interval"], True),
            )
            for top in self.headline.get("top_by_width") or []
            if top["separated"] is False
        ]
        if self.every_set:
            over = (
                "every set at that width that scored the model. A set in which a model has no complete picture counts for the "
                "other models; for that model it counts as nothing, and is not a loss."
            )
            if fewer:
                over += (
                    " Where a model has no score in some of the sets at a width, the number of sets its figure rests on is "
                    "given beneath it: the figure is not over the same sets as the others in its column."
                )
        else:
            over = "the sets every scored model shares at that width, so that each column compares all models on the same sets."
        return _h(
            '<h3 id="by-width">Rank score by width</h3>\n{}\n{}<p class="muted small">Comparisons won over comparisons made '
            "(beneath each figure), pooled over judges, repeats and {}{} Like every rank score this is a standing against the other "
            "models at that width, not a measure of the pictures themselves: a figure that falls with width is a model "
            "losing ground to the field there, and one that rises is a model pulling away from it.{}</p>",
            _table([_tr(head)], rows, "pin"), _join(level), over, none, self._overall_counts(),
        )

    def _peak_table(self) -> _Html:
        title = _Html('<h3 id="peak">Peak size</h3>')
        peaks = self.scores["peak_combined"]
        if not peaks:
            return _h(
                '{}\n<p class="muted">No peak-size judgement yet. It needs a model with complete attempts of one painting '
                "at two or more widths, and a judge to rank those against each other.</p>",
                title,
            )
        widths = sorted({w["width"] for p in peaks for w in p["widths"]})
        lacking = sum(1 for p in peaks if p.get("larger_not_complete"))
        rows = []
        for p in sorted(peaks, key=lambda p: (self.position.get(p["model"], len(self.position)), p["painting"])):
            ranks = {w["width"]: w["mean_rank"] for w in p["widths"]}
            best: Any = str(p["best_width"]) if p["best_width"] is not None else " and ".join(map(str, p["best_widths"])) + " tie"
            if p.get("larger_not_complete"):  # the best of what it finished: not a peak, and the cell says so
                best = _h("{} <span class=\"sub inline\">({} not complete)</span>", best, _listing(p["larger_not_complete"]))
            cells = [_name(p["model"]), _k(self.title(p["painting"])), _td(best, "n wrap" if p.get("larger_not_complete") else "n")]
            for width in widths:
                figure = _num(ranks[width]) if width in ranks else ""
                cells.append(_n(_h("<b>{}</b>", figure) if width in p["best_widths"] else figure))
            rows.append(_tr(cells + [_n(p["calls"])]))
        top = [_th("Model", "pin", rows=2), _th("Painting", rows=2), _th("Judged best at", "n", rows=2)]
        top += [_th("Mean rank among its own sizes, by width", "group", span=len(widths)), _th("Calls", "n", rows=2)]
        if lacking:
            reading = (
                f" In {lacking} of the {_plural(len(peaks), 'row')} a larger width has no complete picture (the attempt was cut off, "
                "blocked or never run), and the cell says which: there the figure is the best of the sizes the model finished, "
                "and cannot be read as the size at which its memory of the picture runs out."
            )
        elif not any(p.get("widths_not_complete", True) for p in peaks):
            reading = (
                " Every row has a complete picture at every width, so the figure can be read as a peak: the size at which the "
                "model's memory of the picture runs out. That reading is untested."
            )
        else:
            reading = ""
        return _h(
            '{}\n{}\n{}\n<p class="muted small">The width at which a model\'s own complete attempts are judged best when they '
            "are ranked against each other at one display size. Rank 1 is best. An empty cell means no complete attempt at "
            "that width.{}</p>\n{}",
            title, self._peak_tally(widths), _table([_tr(top), _tr(_th(width, "n") for width in widths)], rows, "pin"), reading,
            self._distinct_rows(),
        )

    def _peak_tally(self, widths: list[int]) -> Any:
        """Per model: on how many paintings each width was judged best, ties counted apart."""
        tally = self.scores.get("peak_tally")
        if not tally:
            return ""
        lacking = any(t["larger_not_complete"] for t in tally)
        rows = []
        for t in sorted(tally, key=lambda t: (self.position.get(t["model"], len(self.position)), t["model"])):
            outright = {o["width"]: o["paintings"] for o in t["outright"]}
            most = max(outright.values(), default=0)
            cells = [_name(t["model"]), _n(t["paintings"])]
            cells += [_n(_h("<b>{}</b>", outright[w]) if outright.get(w) and outright[w] == most else outright.get(w, 0)) for w in widths]
            ties = ", ".join(f"{tie['paintings']} ({' = '.join(map(str, tie['widths']))})" for tie in t["tied"])
            cells.append(_td(ties or "0", "n" if ties else "n quiet"))
            rows.append(_tr(cells + ([_n(t["larger_not_complete"])] if lacking else [])))
        top = [_th("Model", "pin", rows=2), _th("Paintings", "n", rows=2)]
        top += [_th("Paintings on which this width alone was judged best", "group", span=len(widths)), _th("Ties for best", "n", rows=2)]
        if lacking:
            top.append(_th("Of its paintings, those lacking a complete picture at a larger width", "n", rows=2))
        return _h(
            '{}\n<p class="muted small">The rows of the table below, counted by model. A tie is counted apart, with the widths '
            "that tied, and is in no width's column.{}</p>",
            _table([_tr(top), _tr(_th(width, "n") for width in widths)], rows, "pin"),
            " Where a larger width has no complete picture, a count is of the best size the model finished, not of a peak." if lacking else "",
        )

    def _distinct_rows(self) -> Any:
        """How many complete pictures repeat most of their rows, by model and width: what "complete" does not promise."""
        by_width = self.scores.get("pictures_by_width") or []
        if not by_width:
            return ""
        widths = sorted({row["width"] for row in by_width})
        cells_of = {(row["model"], row["width"]): row for row in by_width}
        models = sorted({row["model"] for row in by_width}, key=lambda name: (self.position.get(name, len(self.position)), name))
        rows = []
        for model in models:
            cells = [_name(model)]
            for width in widths:
                here = cells_of.get((model, width))
                if here is None:
                    cells.append(_td("", "n"))
                else:
                    text = _h("{} of {}{}", here["under_half"], here["pictures"], _sub(f"mean {_percent(here['mean_share_distinct'])} distinct"))
                    cells.append(_td(text, "n" if here["under_half"] else "n quiet"))
            rows.append(_tr(cells))
        totals = [(sum(r["under_half"] for r in by_width if r["width"] == w), sum(r["pictures"] for r in by_width if r["width"] == w)) for w in widths]
        if len(models) > 1:
            rows.append(_tr([_name("All models")] + [_td(f"{under} of {pictures}", "n" if under else "n quiet") for under, pictures in totals], "total"))
        found = [f"{under} of the {pictures} at {width} wide" for width, (under, pictures) in zip(widths, totals) if under]
        repeating = sorted((p for p in self.rows.values() if p["share_distinct"] < 0.5), key=lambda p: (p["share_distinct"], p["attempt"]))
        listed: Any = ""
        if repeating:
            items = (
                _h("<li>{}, {}: {} of {} rows distinct</li>", self.link(p["attempt"], self.index), self.set_link(p["painting"], p["width"]),
                   p["distinct_rows"], p["rows"])
                for p in repeating
            )
            listed = _h(
                '<details class="calls"><summary>The {} with fewer than half {} rows distinct, fewest first</summary>\n'
                '<ul class="plain">\n{}\n</ul></details>',
                _plural(len(repeating), "picture"), "its" if len(repeating) == 1 else "their", _join(items, "\n"),
            )
        return _h(
            '<h3 id="distinct-rows">Complete pictures that repeat their rows</h3>\n{}\n<p class="muted small">Complete pictures with '
            "fewer than half their rows distinct, of the complete pictures at that width. A picture is complete when every "
            "row was stored, whatever the rows hold: one finished by repeating a row, or typed as a coarse grid enlarged, is "
            "complete. Two rows are the same when they draw the same colours.{} For those pictures the larger sizes, in the "
            "peak sizes above and in the rank scores at those widths, partly measure how much of the grid was filled, not "
            "only what the model recalls of the painting.</p>\n{}",
            _table([_tr([_th("Model", "pin")] + [_th(f"{width} wide", "n") for width in widths])], rows, "pin"),
            f" Here: {_listing(found)}." if found else "", listed,
        )

    def _completion_table(self) -> _Html:
        completion = self.scores["completion"]
        by_model = sorted(completion["by_model"], key=lambda c: self.position.get(c["model"], len(self.position)))
        plan = {row["model"]: row for row in self.planned.get("by_model", [])}
        whole = {"planned": self.planned["attempts"], "not_started": len(self.unstarted)} if self.planned else None
        rows = [self._completion_row(c, c["model"], plan.get(c["model"]) if self.planned else None, bool(self.planned)) for c in by_model]
        use = [self._use_row(c, c["model"]) for c in by_model]
        if len(rows) > 1:
            rows.append(self._completion_row(completion["total"], "All models", whole, bool(self.planned), "total"))
            use.append(self._use_row(completion["total"], "All models", "total"))
        attempts = (["Planned"] if self.planned else []) + ["Run", "Complete", "Share of run", "Incomplete", "Refused", "Error"]
        top = [_th("Model", "pin", rows=2), _th("Attempts", "group", span=len(attempts)), _th("Care, mean per attempt", "group", span=5)]
        names = attempts + ["Rows rejected first time", "Rows rejected", "Nudges", "Replies cut off", "Attempts averaged"]
        total = completion["total"]
        idle = total.get("no_tool_call", 0)
        averaged = total["complete"] + total["incomplete"]
        care = (
            f" {idle} of the {_plural(averaged, 'attempt')} averaged made no tool call (every reply was cut off first, or the model "
            "never called the tool), so had no row to be rejected: they pull the two row figures down."
            if idle else ""
        )
        priced = total["known"]["cost_usd"]
        cost = (
            f" The cost total covers the {priced} of {_plural(total['attempts'], 'attempt')} whose model has a price; the others add nothing to it."
            if 0 < priced < total["attempts"] else ""
        )
        use_top = [_th("Model", "pin", rows=2), _th("Use, mean per attempt run", "group", span=5), _th("Cost", "group")]
        use_names = ["Model calls", "Tokens in", "Tokens out", "of which reasoning", "Seconds", "Total"]
        return _h(
            '<h3 id="completion">Completion and care</h3>\n{}\n<p class="muted small">{}Refused is the count of attempts in '
            "that state. It holds two different things, told apart beneath the figure: a request the provider blocked, which "
            "is not the model declining, and the model declining in words. Care covers the attempts that are complete or "
            "incomplete (their number is the last column): rows rejected on their first submission, rows rejected in all, "
            "nudges, and replies cut off by the output limit.{} A nudge is a reminder to finish, and only those sent after "
            "the model stopped of its own accord are counted: a nudge that followed a cut-off reply says nothing about care. "
            "{} means no attempt reported the figure; imported pictures report few.</p>\n{}\n"
            '<h3 id="use">Use and cost</h3>\n{}\n<p class="muted small">Means are over the attempts that were run and reported '
            "the figure, in any state.{}</p>",
            _table([_tr(top), _tr(_th(name, "n") for name in names)], rows, "pin dense"),
            "Run is the attempts with a record; Share of run is the share of those that are complete. " if self.planned else "",
            care, DASH, self._complete_by_width(),
            _table([_tr(use_top), _tr(_th(name, "n") for name in use_names)], use, "pin"), cost,
        )

    @staticmethod
    def _completion_row(c: dict[str, Any], name: str, plan: dict[str, Any] | None, planned: bool, cls: str = "") -> _Html:
        share: Any = _percent(c["share_complete"])
        if plan and plan["not_started"] and plan["planned"]:  # the same count against what was planned
            share = _h("{}{}", share, _sub(f"{_percent(c['complete'] / plan['planned'])} of planned"))
        after = c.get("refused_by_provider_after_cut_off", 0)
        who = (("blocked by the provider", c["refused_by_provider"]), ("of them after cut-off replies", after), ("declined in words", c["refused_in_words"]))
        refused = _h("{}{}", c["refused"], _join(_sub(f"{n} {what}") for what, n in who if n))
        cells = [_name(name)]
        if planned:
            if plan and plan["not_started"]:
                cells.append(_n(_h("{}{}", plan["planned"], _sub(f"{plan['not_started']} never started"))))
            else:
                cells.append(_n(plan["planned"] if plan else DASH))
        cells += [_n(c["attempts"]), _n(c["complete"]), _n(share)]
        cells += [_n(c["incomplete"]), _n(refused if c["refused"] else 0), _n(c["error"])]
        care = ("rows_rejected_first_submission", "rows_rejected", "nudges", "truncated_replies")
        cells += [_n(_num(c[f"mean_{field}"], 1)) for field in care]
        averaged, idle = c["complete"] + c["incomplete"], c.get("no_tool_call", 0)
        cells.append(_n(_h("{}{}", averaged, _sub(f"{idle} made no tool call")) if idle else averaged))
        return _tr(cells, cls)

    @staticmethod
    def _use_row(c: dict[str, Any], name: str, cls: str = "") -> _Html:
        cells = [_name(name), _n(_num(c["mean_model_calls"], 1))]
        cells += [_n(_count(c[f"mean_{field}"])) for field in ("input_tokens", "output_tokens", "reasoning_tokens")]
        priced = c["known"]["cost_usd"]
        cost: Any = _money(c["cost_usd"])
        if 0 < priced < c["attempts"]:  # a total that leaves attempts out says how many it covers
            cost = _h("{}{}", cost, _sub(f"{priced} of {c['attempts']} attempts priced"))
        elif not priced and c["attempts"]:
            cost = _h("{}{}", cost, _sub("no price"))
        return _tr(cells + [_n(_num(c["mean_wall_seconds"], 0)), _n(cost)], cls)

    def _complete_by_width(self) -> Any:
        """Complete attempts over attempts run, by model and width, with the planned number wherever some were never started."""
        cells_of = {(c["model"], c["width"]): c for c in self.scores["completion"].get("by_model_width", [])}
        if not cells_of:
            return ""
        plan = {(row["model"], row["width"]): row for row in self.planned.get("by_model_width", [])}
        plan_width = {row["width"]: row for row in self.planned.get("by_width", [])}
        widths = sorted({width for _model, width in cells_of} | {width for _model, width in plan})
        models = sorted({model for model, _width in cells_of}, key=lambda name: (self.position.get(name, len(self.position)), name))

        def cell(c: dict[str, Any] | None, planned: dict[str, Any] | None) -> _Html:
            if c is None and not (planned and planned["not_started"]):
                return _td("", "n")
            done, run = (c["complete"], c["attempts"]) if c else (0, 0)
            if planned and planned["not_started"]:
                return _td(_h("{} of {} run{}", done, run, _sub(f"{planned['planned']} planned, {planned['not_started']} never started")), "n wrap")
            return _n(f"{done} of {run}")

        rows = [_tr([_name(model)] + [cell(cells_of.get((model, width)), plan.get((model, width))) for width in widths]) for model in models]
        if len(models) > 1:
            totals = {c["width"]: c for c in self.scores["completion"].get("by_width", [])}
            rows.append(_tr([_name("All models")] + [cell(totals.get(width), plan_width.get(width)) for width in widths], "total"))
        return _h(
            '<h3 id="by-width-complete">Complete attempts by width</h3>\n{}\n<p class="muted small">Complete attempts over the '
            "attempts run at that width.{} An empty cell means the model was not run there.</p>",
            _table([_tr([_th("Model", "pin")] + [_th(f"{width} wide", "n") for width in widths])], rows, "pin"),
            " Where planned attempts were never started the cell gives both numbers: counts beside it are out of a different total."
            if self.unstarted else "",
        )

    # ---- judges and people ----

    def agreement(self) -> _Html:
        parts = [self._the_judges(), self._judge_vs_person(), self._agreement_levels(), self._judge_vs_judge(), self._by_judge()]
        parts += [self._own_model(), self._own_family(), self._failed_judgements()]
        return _join([_Html('<h2 id="agreement">Judges and people</h2>')] + [part for part in parts if part], "\n")

    def _the_judges(self) -> Any:
        """Who the judges are and how their calls went: what must sit beside any figure of how far they agree."""
        if not self.judges:
            return ""
        counts = self.scores["counts"]
        lines = [sentence for sentence in (self._contestant_judges(),) if sentence]
        if not counts["human_rankings"]:
            waiting = counts["human_rankings_pending"]
            lines.append(
                "No person has ranked a blind sheet in this run"
                + (f"; {_plural(waiting, 'sheet')} {'is' if waiting == 1 else 'are'} made and waiting to be ranked" if waiting else "")
                + ". Until one has, every figure of agreement below is between judges only, and the order of the models rests on "
                "judges that have not been checked."
            )
        lines.append(
            "Labels and the order in which the pictures are shown are drawn afresh at random for every call, so no model and "
            "no width is shown first more often than chance gives."
        )
        retried = any(j["calls_reporting_api_retries"] for j in self.judges)
        ordered = any(j["calls_with_order"] for j in self.judges)
        unnamed = any(j["calls_without_reference"] for j in self.judges)
        rows = []
        for j in self.judges:
            cells = [_name(j["judge"]), _k(j["model"] or DASH), _td(", ".join(j["contestants"]) or "no", "k" if j["contestants"] else "k quiet")]
            cells.append(_n(j["calls"]))
            if retried:
                known = j["calls_reporting_api_retries"]
                cells += [_n(f"{j['calls_with_api_retries']} of {known}" if known else DASH), _n(j["api_retries"] if known else DASH)]
            if ordered:
                known = j["calls_with_order"]
                cells.append(_n(_h("{} of {}{}", j["first_shown_best"], known, _sub(f"{j['first_shown_expected']:.1f} by chance")) if known else DASH))
            if unnamed:
                cells.append(_n(j["calls_without_reference"]))
            rows.append(_tr(cells))
        head = [_th("Judge", "pin"), _th("Model id"), _th("Also a contestant, as"), _th("Calls in use", "n")]
        head += [_th("Calls that needed API retries", "n"), _th("Retries in all", "n")] if retried else []
        head += [_th("Ranked the picture shown first as best", "n")] if ordered else []
        head += [_th("Records that name no reference image", "n")] if unnamed else []
        notes = []
        if retried:
            notes.append(
                "An API retry is a try the provider turned away (overloaded, or a rate limit) before the call was answered; the "
                "ranking is that of the one reply that came. A call that failed outright and was made again leaves no trace "
                "here: the judge stage keeps only the record that stands."
            )
        if ordered:
            notes.append(
                "A judge blind to the order would rank the picture shown first as best about as often as the number beneath "
                "the count. A count above it in one run or one judge is not an established lean."
            )
        if unnamed:
            notes.append(
                "A record that names no reference image was made before records saved one, or was imported: it is held to "
                "the pictures it ranked, and not to the image of the painting it was shown."
            )
        return _h(
            "<h3 id=\"judges\">The judges</h3>\n{}\n{}\n<p class=\"muted small\">{}</p>",
            _join((_h("<p>{}</p>", line) for line in lines), "\n"), _table([_tr(head)], rows, "pin"), " ".join(notes),
        )

    def _agreement_levels(self) -> Any:
        """Agreement between the judges' averaged orders, between single calls of different judges, and within each judge."""
        agreement = self.scores.get("judge_agreement")
        if not agreement or not (agreement["between_orders"]["sets"] or agreement["within"]):
            return ""
        orders, calls = agreement["between_orders"], agreement["between_calls"]
        rows = []
        if orders["sets"]:
            cells = [_name("Between the judges' orders of a set, each averaged over its repeats"), _n(_num(orders["tau_b"]))]
            rows.append(_tr(cells + [_td(f"{_plural(orders['sets'], 'set')}; lowest {_num(orders['lowest'])}")]))
        if calls["pairs"]:
            cells = [_name("Between one call of one judge and one call of another"), _n(_num(calls["tau"]))]
            rows.append(_tr(cells + [_td(f"{_plural(calls['pairs'], 'pair')} of calls, in {_plural(calls['sets'], 'set')}")]))
        for row in agreement["within"]:
            cells = [_name(_h("Between two calls of {} on the same set", row["judge"])), _n(_num(row["tau"]))]
            rows.append(_tr(cells + [_td(f"{_plural(row['pairs'], 'pair')} of calls, in {_plural(row['sets'], 'set')}")]))
        return _h(
            '<h3 id="levels">How far the judges agree, at three levels</h3>\n{}\n<p class="muted small">The first row is the '
            "figure usually quoted: tau-b between the two orders the report shows for a set, where averaging over repeats "
            "has already smoothed each judge. The other rows are Kendall's tau between single ranking calls, over the "
            "pictures both ranked, averaged over every such pair of calls. Where single calls of different judges agree "
            "about as well as one judge agrees with itself, the judges differ from each other little more than each "
            "differs from one of its own repeats. None of this is a check against a person.</p>",
            _table([_tr([_th("Level"), _th("Kendall's tau, mean", "n"), _th("Over")])], rows, "fit"),
        )

    def _judge_vs_person(self) -> _Html:
        title = _Html("<h3>Each judge against each person</h3>")
        calls = {(s["painting"], s["width"], s["judge"]): s["calls"] for s in self.scores["sets"]}
        rows = []
        for row in self.scores["judge_vs_person"]:
            pooled, key = row["combined"], (row["painting"], row["width"], row["judge"])
            if row["repeats"]:
                within = f"{row['repeats_within_one_adjacent_swap']} of {_plural(row['repeats_compared'], 'ranking')}"
            else:
                within = "yes" if pooled["within_one_adjacent_swap"] else "no"
            swapped = self.pair_list(pooled["swapped_pairs"])
            if pooled["tied_pairs"]:
                swapped += f"; tied in the judge's pooled order: {self.pair_list(pooled['tied_pairs'], ' = ')}"
            cells = [_k(self.set_link(row["painting"], row["width"])), _k(row["judge"]), _k(row["person"])]
            cells += [_k(f"all {_plural(calls.get(key, 0), 'call')} pooled"), _n(pooled["n_common"]), _n(_num(pooled["tau_b"]))]
            rows.append(_tr(cells + [_k(within), _p(swapped)], "lead"))
            for r in row["repeats"]:
                within = "the same order" if r["identical"] else "yes" if r["within_one_adjacent_swap"] else "no"
                cells = [_td(""), _td(""), _td(""), _k(f"ranking, repeat {r['repeat']}"), _n(r["n_common"]), _n(_num(r["tau"]))]
                rows.append(_tr(cells + [_k(within), _p(self.pair_list(r["swapped_pairs"]))], "repeat"))
        waiting = self.scores["counts"]["human_rankings_pending"]
        pending = _note_p("{} made and not yet ranked.", _plural(waiting, "blind sheet")) if waiting else ""
        if not rows:
            none = "No person has ranked a blind sheet yet." if not self.people else "No judge has ranked the pictures a person ranked."
            return _h(
                '{}\n<p class="muted">{} A judge is trusted only once it has been checked against a person.</p>\n{}',
                title, none, pending,
            )
        head = [_th("Set"), _th("Judge"), _th("Person"), _th("Judge's order"), _th("Pictures", "n"), _th("Kendall's tau", "n")]
        head += [_th("Within one swap of neighbours"), _th("Pairs in opposite order, the person's preference first")]
        return _h(
            '{}\n{}\n<p class="muted small">Tau is 1 for the same order and −1 for the reverse. A pooled order can hold ties, '
            "so its figure is tau-b and a tie is not counted as a swap. The test a judge is held to: within one swap of "
            "neighbours of the person's order in at least two of three rankings.</p>\n{}",
            title, _table([_tr(head)], rows), pending,
        )

    def _judge_vs_judge(self) -> Any:
        pairs = self.scores["judge_vs_judge"]
        if not pairs:
            return ""
        # With one pair of judges every row would name the same two: they are named once, beneath the table, and the
        # room goes to the pairs of pictures, so that the last column is not pushed out of a narrow window.
        judges = sorted({(r["judge_a"], r["judge_b"]) for r in pairs})
        named = len(judges) > 1
        rows = []
        for r in pairs:
            cells = [_k(self.set_link(r["painting"], r["width"]))] + ([_k(r["judge_a"]), _k(r["judge_b"])] if named else [])
            cells += [_n(r["n_common"]), _n(_num(r["tau_b"])), _p(self.pair_list(r["swapped_pairs"])), _td(self.pair_list(r["tied_pairs"], " = "))]
            rows.append(_tr(cells))
        head = [_th("Set")] + ([_th("Judge"), _th("Judge")] if named else []) + [_th("Pictures", "n"), _th("Kendall's tau‑b", "n")]
        head += [_th("Pairs in opposite order, the first judge's preference first"), _th("Pairs either judge ties")]
        return _h(
            '<h3 id="judge-against-judge">Judge against judge</h3>\n{}\n<p class="muted small">Each judge\'s pooled order of a set, compared.{}</p>',
            _table([_tr(head)], rows),
            "" if named else _h(" Every row holds {}, the first judge, against {}.", *judges[0]),
        )

    def _by_judge(self) -> Any:
        judges = sorted({j["judge"] for m in self.scores["models"] for j in m["by_judge"]})
        if len(judges) < 2:
            return ""

        def cell(entry: dict[str, Any] | None) -> _Html:
            """A figure of the table. One that rests on fewer sets than its column has says on how many, beneath it."""
            if entry is None or not self.every_set or not _starred(entry):
                return _n(_overall(entry))
            return _n(_h("{}{}", _overall(entry), _sub(f"{entry['sets_scored']} of {_plural(entry['sets_all'], 'set')}")))

        rows = []
        for m in self.scores["models"]:
            theirs = {j["judge"]: j for j in m["by_judge"]}
            rows.append(_tr([_name(m["model"]), cell(m)] + [cell(theirs.get(judge)) for judge in judges]))
        head = _tr([_th("Model"), _th("All judges pooled", "n")] + [_th(judge, "n") for judge in judges])
        starred = any(_starred(entry) for m in self.scores["models"] for entry in [m] + m["by_judge"])
        if not self.every_set:
            return _h(
                '<h3 id="by-judge">Rank score by judge</h3>\n{}\n<p class="muted small">Each figure is a mean over the sets in which every model '
                "scored in its column has a score, so a column compares its models on the same sets.{}</p>",
                _table([head], rows), " * Also scored in other sets, which the figure leaves out." if starred else "",
            )
        # With standings by width the page has no table of these figures above this one: what one judge alone changes is said here.
        alone = self._alone() if self.sized else []
        return _h(
            '<h3 id="by-judge">Rank score by judge{}</h3>\n{}\n<p class="muted small">Each figure is a mean over every set in which the '
            "judging calls of its column scored the model.{}{}</p>",
            ", all sizes together" if self.sized else "", _table([head], rows),
            " * No score in some of its column's sets: the figure is a mean over fewer sets than the column has, and not over "
            "the same sets as an unmarked figure. Beneath it is the number of sets it rests on, of those its column scored." if starred else "",
            _h(" By one judge alone: {}.", _join(alone, "; ")) if alone else "",
        )

    def _own_model(self) -> Any:
        """A judge that is itself a contestant, on that contestant: its score from this judge against the other judges'."""
        own = self.scores.get("own_model") or []
        if not own:
            return ""
        title = _Html('<h3 id="own-model">Does a judge score its own model higher?</h3>')
        measured = [r for r in own if r["difference"] is not None]
        if not measured:
            return _h(
                '{}\n<p class="muted">{} also {}, but there is no second judge that scored the same pictures to hold it against.</p>',
                title, _listing(sorted({r["judge"] for r in own})), "a contestant" if len({r["judge"] for r in own}) == 1 else "contestants",
            )
        rows = []
        for r in measured:  # the interval sits beneath its difference, so that the table fits a narrow window
            interval = _range(r["interval"], True, 3) if r["interval"] else "no interval"
            cells = [_name(r["judge"]), _k(r["contestant"]), _n(r["sets"]), _n(_num(r["own_score"])), _n(_num(r["others_score"]))]
            cells += [_n(_h("{}{}", _signed(r["difference"], 3), _sub(interval))), _n(f"{r['sets_higher']} / {r['sets_lower']} / {r['sets_equal']}")]
            rows.append(_tr(cells))
        # Since version 6 of the scores: every set in which this judge and another scored the model. Before: the shared sets only.
        head = [_th("Judge", "pin"), _th("Its own model"), _th("Sets compared" if self.own_model_every_set else "Shared sets", "n")]
        head += [_th("Score from this judge", "n"), _th("From the others", "n")]
        head += [_th("Difference, and its 95% interval", "n"), _th("Sets higher / lower / equal", "n")]
        against = _join((_h("{} with {}", r["judge"], _listing(r["other_judges"])) for r in measured), "; ")
        return _h(
            '{}\n{}\n<p class="muted small">For a judge whose model id is also a contestant\'s: that contestant\'s rank score '
            "from this judge alone against the same from the other judges, {} "
            "Compared here: {}. A "
            "positive difference means this judge put its own model's pictures higher than the others did; the last column "
            "counts the sets in which its score was higher than, lower than and equal to theirs. The interval is "
            "from drawing the paintings again at random, as for {}; one that includes zero does not "
            "establish a difference. A difference does not say which judge is off: the other judge marking that model down, or one "
            "judge being the noisier (which pulls both ends of the table towards the middle), would look the same, and "
            "where the other judge is a contestant too the two cannot be told apart. It is not on its own evidence of "
            "favour.</p>",
            title, _table([_tr(head)], rows, "pin"),
            "set by set, over every set in which this judge and at least one other judge scored it. A set in which some other "
            "model has no picture is one of them. The judges are matched on the painting and the width, not on what they ranked "
            "the model against: where they were not given the same pictures there, that is part of the difference too."
            if self.own_model_every_set
            else "over the sets every scored model shares.",
            against, "the standings" if self.sized else "the headline figures",
        )

    def _own_family(self) -> Any:
        if not self.scores["sets"]:
            return ""
        title = _Html("<h3>Do judges favour their own family?</h3>")
        measured = [r for r in self.scores["own_family"] if r["difference"] is not None]
        if not measured:
            narrower = any(r["difference"] is not None for r in self.scores.get("own_model") or [])
            return _h(
                '{}\n<p class="muted">{} It needs a judge whose model family is recorded, contestants '
                "of that family and of another in one set, and a judge of another family to compare with. Judges of one "
                "family are not held against each other: they may share the favour.{}</p>",
                title, "By family, not measurable in this run." if narrower else "Not measurable in this run.",
                _Html(' The narrower question, a judge on its own model, has figures: <a href="#own-model">above</a>.') if narrower else "",
            )
        rows = []
        for r in measured:
            cells = [_k(r["judge"]), _k(r["family"]), _p(", ".join(r["own_models"])), _p(", ".join(r["other_judges"]))]
            cells += [_td(", ".join(r["same_family_judges"]) or DASH), _n(r["scores_compared"]), _n(_num(r["own_mean"]))]
            rows.append(_tr(cells + [_n(_num(r["others_mean"])), _n(f"{r['difference']:+.2f}")]))
        head = [_th("Judge"), _th("Family"), _th("Contestants of its family"), _th("Compared with: judges of other families")]
        head += [_th("Left out: judges of its family"), _th("Scores", "n"), _th("Its mean rank score for them", "n")]
        head += [_th("The other families' judges'", "n"), _th("Difference", "n")]
        return _h(
            '{}\n{}\n<p class="muted small">A positive difference means the judge was kinder to its own family than the '
            "judges of other families were. A judge of its own family is left out of the comparison: it may share the "
            "favour.</p>",
            title, _table([_tr(head)], rows),
        )

    def _failed_judgements(self) -> Any:
        failed = self.scores["failed_judgements"]
        if not failed:
            return ""
        rows = []
        # One re-typed picture makes every call that showed it stale, and a set that changes kind supersedes all of its
        # calls, so these are counted by directory. A failed call is listed on its own, with what went wrong.
        together: dict[tuple, int] = defaultdict(int)
        reasons = {
            ("stale", "picture"): "stale: a picture ranked here has since been replaced, or its attempt is no longer complete.",
            ("stale", "reference"): "stale: the painting's reference image has been replaced since.",
            ("superseded", "ranking"): "superseded: this set now has more complete attempts than one call takes, so it is judged in pairs.",
            ("superseded", "pairwise"): "superseded: this set is now small enough to rank in one call, so it is not judged in pairs.",
        }
        for f in failed:
            what = f"{f['kind']}, {self.title(str(f['painting']))}"
            what += (f", {f['width']} wide" if f["width"] else "") + (f", sizes of {f['model']}" if f["model"] else "")
            reason = (f["status"], f["stale_reason"] if f["status"] == "stale" else f["kind"])
            if reason in reasons:
                together[(what, f["judge"], os.path.dirname(f["path"]), reason)] += 1
                continue
            why = [_said(text) for text in [f["error"]] + f["parse_errors"] if text]
            cells = [_k(what), _k(f["judge"]), _n(DASH if f["repeat"] is None else f["repeat"])]
            rows.append(_tr(cells + [_p(_join(why, _BREAK) if why else f"status {f['status']}"), _p(_h("<code>{}</code>", f["path"]))]))
        for (what, judge, where, reason), n in together.items():
            why = f"{_plural(n, 'call')} {reasons[reason]}"
            rows.append(_tr([_k(what), _k(judge), _n(DASH), _p(why), _p(_h("<code>{}</code>", where))]))
        statuses = {reason[0] for _what, _judge, _where, reason in together}
        advice = " Running the judge stage again remakes the stale calls it still plans." if "stale" in statuses else ""
        if "superseded" in statuses:
            advice += " The judge stage no longer makes a superseded call; its file is left on disk."
        head = _tr([_th("Call"), _th("Judge"), _th("Repeat", "n"), _th("What went wrong"), _th("Record, under the run directory")])
        return _h(
            '<h3 id="failed">Judging calls that are not scored</h3>\n<p class="muted">{} {} not scored. A ranking is never '
            "guessed or repaired.{}</p>\n{}",
            _plural(len(failed), "call"), "was" if len(failed) == 1 else "were", advice, _table([head], rows),
        )

    # ---- attempts that did not complete ----

    def not_complete(self) -> _Html:
        attempts, unstarted = self.scores["not_complete"], self.unstarted
        title = _h('<h2 id="not-complete">Not complete ({}){}</h2>', len(attempts), f", and never started ({len(unstarted)})" if unstarted else "")
        if not attempts and not unstarted:
            none = _note_p("Every attempt completed." if self.refs else "There are no attempts yet.")
            return _join([part for part in (title, none, self._retried()) if part], "\n")
        rows = []
        for a in attempts:
            image = self.picture(a["attempt"])
            canvas = self.link(a["attempt"], self.index, _img(image, "The unfinished canvas", "typed thumb")) if image else DASH
            cells = [_td(canvas, "" if image else "none"), _who(self.link(a["attempt"], self.index)), _k(self.set_link(a["painting"], a["width"]))]
            rows.append(_tr(cells + [_k(_state(a["state"])), _p(self._why(a))]))
        for i, a in enumerate(unstarted):  # planned, and no record of them: listed with the rest, and said to be so
            cells = [_td(DASH, "none"), _who(self.name(a["attempt"])), _k(self.set_link(a["painting"], a["width"]))]
            cells += [_k(_Html('<span class="state unknown"><i></i>never started</span>'))]
            why = "Never started: the config plans this attempt and the run holds no record of it. No reason is recorded."
            rows.append(_tr(cells + [_p(why)], anchor="never-started" if i == 0 else ""))
        head = _tr([_th("Canvas"), _th("Attempt"), _th("Set"), _th("State"), _th("Why")])
        total = self.scores["completion"]["total"]
        cut = total.get("cut_off") or {}
        failed_cut = sum(n for state, n in cut.items() if state != STATE_COMPLETE)
        overlap = ""
        if failed_cut or cut.get(STATE_COMPLETE):
            overlap = (
                f" The causes overlap: of the {len(attempts)} that did not complete, {failed_cut} had at least one reply cut off by "
                f"the output limit; {_plural(cut.get(STATE_COMPLETE, 0), 'more attempt')} had a reply cut off and still completed."
            )
        never: Any = ""
        if unstarted:
            never = _h(
                " {} {} never started: the run holds no record of {}, and no reason is recorded. {} listed last.{}",
                _plural(len(unstarted), "planned attempt"), "was" if len(unstarted) == 1 else "were", "it" if len(unstarted) == 1 else "them",
                "It is" if len(unstarted) == 1 else "They are",
                _Html(' The operator\'s notes on this run, which are not part of its records, are <a href="#notes">on the first screen</a>.')
                if self.notes else "",
            )
        return _h(
            '{}\n<p class="muted">Only complete attempts are judged. These are counted and kept. Rows that were never stored '
            "are drawn in magenta. The state “refused” covers a request the provider blocked as well as the model "
            "declining; the reason says which.{}{}</p>\n{}\n{}",
            title, overlap, never, _table([head], rows, "stack"), self._retried(),
        )

    def _retried(self) -> Any:
        """The attempts that replaced an earlier run of themselves, whatever their state now, with how each such run ended."""
        if not self.retried:
            return ""
        rows = []
        for a in self.retried.values():
            cells = [_who(self.link(a["attempt"], self.index)), _k(self.set_link(a["painting"], a["width"])), _k(_state(a["state"]))]
            rows.append(_tr(cells + [_p(_join((self._earlier_run(run) for run in a["earlier"]), _BREAK))]))
        head = _tr([_th("Attempt"), _th("Set"), _th("State now"), _th("Its earlier runs, oldest first")])
        return _h(
            '<h3 id="retried">Run again ({})</h3>\n<p class="muted">These attempts replaced an earlier run of themselves. The '
            "figures of this report count the run that stands; how the earlier ones ended is kept with it.</p>\n{}",
            len(self.retried), _table([head], rows),
        )

    @staticmethod
    def _ended(run: dict[str, Any]) -> str:
        """How one run of an attempt ended (an ENDING of the scores), in words. A provider's block is never the model declining."""
        stop, by = run.get("stop"), run.get("refusal_detected_by")
        if run.get("state") == STATE_REFUSED and by == "provider":
            category = f" (category {run['refusal_category']})" if run.get("refusal_category") else ""
            ended = f"blocked by the provider{category}: the platform stopped the request, the model did not decline"
            cut, calls = run.get("truncated_replies"), run.get("model_calls")
            if isinstance(cut, int) and isinstance(calls, int) and 0 < cut <= calls:  # both happened, and both are said
                ended += f"; before the block the output limit cut off {cut} of its {calls} replies"
            return ended
        if run.get("state") == STATE_REFUSED and by != "text":
            return "refused; the record does not say whether the model declined or the provider blocked the request"
        if stop == "nudges_exhausted" and run.get("every_reply_cut_off"):
            return "every reply was cut off by the output limit"
        return str(_STOPS.get(stop, stop or "no reason was recorded"))

    def _how_it_ended(self, record: dict[str, Any]) -> str:
        """A sentence for an attempt that did not complete: how the loop ended and how far it got."""
        stats = self.stats.get(record["attempt"], {})
        ended = self._ended({"model_calls": stats.get("model_calls"), "truncated_replies": stats.get("truncated_replies"), **record})
        if record.get("state") == STATE_INCOMPLETE:
            if stats.get("rows_stored") is not None and stats.get("height"):
                ended += f", with {stats['rows_stored']} of {stats['height']} rows stored"
            cut, calls = stats.get("truncated_replies"), stats.get("model_calls")
            if isinstance(cut, int) and isinstance(calls, int) and 0 < cut < calls:
                ended += f"; the output limit cut off {cut} of its {calls} replies"
        return ended[:1].upper() + ended[1:] + "."

    def _why(self, record: dict[str, Any]) -> _Html:
        """Why an attempt did not complete: how the loop ended, how far it got, and the refusal or the error."""
        lines = [self._how_it_ended(record)] + [_said(record[key]) for key in ("refusal_text", "error") if record.get(key)]
        return _join(lines, _BREAK)

    def _earlier_run(self, run: dict[str, Any]) -> _Html:
        """One replaced run of an attempt: when it finished, how it ended, and its refusal or error."""
        ended = self._ended(run) if run.get("state") != STATE_COMPLETE else "complete"
        when = f"Finished {run['finished_at']}: " if run.get("finished_at") else ""
        lines = [f"{when}{ended}." if when else ended[:1].upper() + ended[1:] + "."]
        return _join(lines + [_said(run[key]) for key in ("refusal_text", "error") if run.get(key)], _BREAK)

    # ---- run facts and credits ----

    def facts(self) -> _Html:
        head = [_th("Name"), _th("Provider"), _th("Model id"), _th("Family"), _th("Reasoning"), _th("Max output tokens", "n")]
        head = _tr(head + [_th("Price per million tokens, in and out", "n")])
        models, judges = self._settings_on_record()
        in_order = sorted(models, key=lambda name: (self.position.get(name, len(self.position)), name))
        model_rows = [self._settings_row(name, models[name]) for name in in_order]
        judge_rows = [self._settings_row(name, judges[name]) for name in sorted(judges)]
        return _h(
            '<h2 id="run">Run facts and credits</h2>\n{}\n<h3>Models</h3>\n{}\n<h3>Judges</h3>\n{}\n'
            '<p class="muted small">Settings are those saved with each attempt and each judgement. Temperature is always the '
            'provider\'s default. {} means the setting was not recorded.</p>\n<h3 id="credits">The real paintings</h3>\n{}',
            _facts(self._run_facts()),
            _table([head], model_rows) if model_rows else _note_p("No models yet."),
            _table([head], judge_rows) if judge_rows else _note_p("No judges yet."),
            DASH, _join((self._credit(slug) for slug in self.slugs), "\n") or _note_p("None yet."),
        )

    def _run_facts(self) -> list[tuple[str, Any]]:
        cfg, stats = self.cfg, list(self.stats.values())
        facts: list[tuple[str, Any]] = [("Run", self.scores["run_id"]), ("Run directory", self.layout.root.resolve())]
        facts.append(("Encoding", cfg.encoding if cfg else _distinct(stats, "encoding")))
        if cfg:
            limits, judge = cfg.limits, cfg.judge
            facts += [
                ("Paintings", ", ".join(cfg.paintings) or DASH),
                ("Widths", ", ".join(map(str, cfg.widths)) or DASH),
                ("Attempts per model, painting and width", cfg.repeats),
                *([("Attempts planned", self._planned_fact())] if self.planned else []),
                (
                    "Limits per attempt",
                    f"{limits.max_steps} model calls, {limits.max_minutes:g} minutes, {limits.max_nudges} nudges, {cfg.strip} rows per call",
                ),
                ("Judging", f"{judge.repeats} repeats per set; pairs when a set has more than {judge.max_per_call} attempts"),
            ]
        else:
            facts.append(("Config", "None was given, so only what the saved records hold is shown."))
        return facts + [
            ("Harness version", _distinct(stats, "harness_version")),
            ("Contestant prompt version", _distinct(stats, "prompt_version")),
            ("Judge prompt version", _distinct(self.records, "prompt_version")),
            ("People who ranked a sheet", ", ".join(sorted({str(h.get("person")) for h in self.people})) or DASH),
            ("Report built", f"{self.built_at}, by paintbench {__version__}"),
        ]

    def _planned_fact(self) -> str:
        text = f"{self.planned['attempts']:,}; {self.planned['run']:,} of them run"
        if self.unstarted:
            text += f", {len(self.unstarted):,} never started (no reason is recorded)"
        if self.planned.get("unplanned"):
            text += f"; {self.planned['unplanned']:,} more on disk that the config does not plan"
        return text

    def _settings_on_record(self) -> tuple[dict[str, dict], dict[str, dict]]:
        """{model: settings} and {judge: settings}: what the attempts and judgements saved, then what only the config names."""
        models: dict[str, dict] = {}
        judges: dict[str, dict] = {}
        for attempt, stats in self.stats.items():
            models.setdefault(self.refs[attempt].model, stats.get("model_settings") or {})
        for record in self.records:
            judges.setdefault(str(record.get("judge")), record.get("judge_settings") or {})
        for known, configured in ((models, self.cfg.models), (judges, self.cfg.judges)) if self.cfg else ():
            for entry in configured:
                known.setdefault(entry.name, entry.to_dict())
        return models, judges

    @staticmethod
    def _settings_row(name: str, settings: dict[str, Any]) -> _Html:
        price = settings.get("price")
        cost = f"${price.get('input_per_mtok', 0):g} and ${price.get('output_per_mtok', 0):g}" if isinstance(price, dict) else DASH
        cells = [_name(name)] + [_k(_plain(settings.get(key))) for key in ("provider", "model", "family")]
        return _tr(cells + [_td(_setting(settings, "reasoning")), _n(_setting(settings, "max_output_tokens")), _n(cost)])

    def _credit(self, slug: str) -> _Html:
        painting = self.painting(slug)
        if painting is None:
            return _h("<p><b>{}</b>: no painting metadata was found, so there is no credit to show.</p>", slug)
        reference = painting.meta.get("reference") or {}
        links = [_external(reference.get("source_page_url"), "Source page"), _external(reference.get("licence_url"), "Licence")]
        return _h(
            "<p><b>{}</b>, {}, {}. {} {} {}</p>",
            painting.display_name, painting.artist, painting.date,
            f"Held by {reference['holder']}." if reference.get("holder") else "",
            f"Image: {reference['credit']}" if reference.get("credit") else "No credit line is recorded for the image.",
            _join((link for link in links if link), " · "),
        )

    # ================================================================ one attempt

    def attempt_page(self, attempt: str) -> str:
        ref, stats, page = self.refs[attempt], self.stats[attempt], self.detail(attempt)
        height = self.height(ref.painting, ref.width, stats.get("height"))
        size = f"{ref.width} × {height or '?'} cells"
        back = self.href(self.index, page, "#" + self.anchors.get((ref.painting, ref.width), "pictures"))
        nav = [_h('<a href="{}">← {}, {}</a>', back, self.title(ref.painting), size)]
        for word, other in zip(("Previous:", "Next:"), self.neighbours.get(attempt, (None, None))):
            if other:
                nav.append(_h("{} {}", word, self.link(other, page)))
        body = _h(
            '<header class="top">\n<nav>{}</nav>\n<h1>{} <span class="by">attempt {}</span> {}</h1>\n'
            '<p class="muted">{}, {}. Run {}.</p>\n{}\n</header>\n{}\n'
            '<h2 id="judges">What the judges said</h2>\n{}\n<h2 id="closing">The model\'s closing text</h2>\n{}\n{}{}'
            '<h2 id="settings">Model and settings</h2>\n{}\n<h2 id="figures">Figures</h2>\n{}\n'
            '<h2 id="transcript">Transcript</h2>\n{}',
            _join(nav, " "), ref.model, ref.n, _state(stats.get("state")),
            self.title(ref.painting), size, self.scores["run_id"], self._standing(attempt), self._pair(attempt, height),
            self._judges_table(attempt), self._closing_text(stats), self._trouble(attempt), self._earlier(attempt),
            self._settings(stats), self._figures(ref, stats), self._transcript(ref, stats),
        )
        return _page(f"{ref.model}, {self.title(ref.painting)} {size}", body)

    def _standing(self, attempt: str) -> _Html:
        place, info, ranked = self.standing.get(attempt, ("", "", 0))
        if place:
            return _h("<p>Placed <b>{}</b> of {} in its set{}.</p>", place, ranked, f": {info}" if info else "")
        if self.stats[attempt].get("state") == STATE_COMPLETE:
            ref = self.refs[attempt]
            together = next((g["complete"] for g in self.grids if (g["painting"], g["width"]) == (ref.painting, ref.width)), [])
            if len(together) > 1 or not self.calls:
                return _note_p("Not ranked yet.")
            own = any(record.get("kind") == "peak" for record, _place in self.said[attempt])
            return _note_p(
                "Not ranked against other models: it is the only complete attempt at this size, and a ranking needs two.{}",
                " It was ranked among this model's own sizes: see below." if own else "",
            )
        ended = self._how_it_ended(self.unfinished[attempt]) + " " if attempt in self.unfinished else ""
        return _note_p("{}Not ranked: only complete attempts are judged.", ended)

    def _pair(self, attempt: str, height: int | None) -> _Html:
        """The picture large, beside the real painting."""
        ref, stats = self.refs[attempt], self.stats[attempt]
        image, real, reduced = self.picture(attempt), self.real(ref.painting), self.reduced(ref.painting, ref.width, height)
        if image is None:
            typed = _Html('<p class="gap">There is no picture: nothing was stored, or the picture file could not be read.</p>')
        elif stats.get("state") == STATE_COMPLETE:
            rows = self.rows.get(attempt)
            distinct = ""
            if rows:
                distinct = f" {rows['distinct_rows']} of its {_plural(rows['rows'], 'row')} are distinct ({_percent(rows['share_distinct'])})."
                if rows["share_distinct"] < 0.5:
                    distinct += " It is complete because every row was stored; more than half of its rows repeat another."
            caption = _h("Typed by {}. Every cell is a solid square, as a judge is shown it.{}", ref.model, distinct)
            typed = _figure(image, f"Typed by {ref.model}", caption, "typed")
        else:
            caption = _h(
                "The unfinished canvas of {}: {} of {} rows stored. Rows never stored are drawn in magenta.",
                ref.model, _plain(stats.get("rows_stored")), height or "?",
            )
            typed = _figure(image, f"The unfinished canvas of {ref.model}", caption, "typed")
        if real is None:
            beside = _Html(
                '<p class="gap">The real painting cannot be shown: its metadata or its reference image was not found. '
                "If the image is missing, <code>python scripts/fetch_originals.py &lt;painting&gt;</code> fetches and rebuilds it.</p>"
            )
        else:
            beside = _figure(real, "The real painting", "The real painting.")
        if reduced:
            caption = "Each cell is the average of its area. Shown for scale: it was never judged."
            beside = _h(
                '{}<details class="scale"><summary>The real painting at this grid size</summary>{}</details>',
                beside, _figure(reduced, "The real painting reduced to the grid", caption, "typed"),
            )
        return _h('<div class="pair">\n<div>{}</div>\n<div>{}</div>\n</div>', typed, beside)

    def _judges_table(self, attempt: str) -> _Html:
        """Every call that ranked this attempt, and every person: who, which call, its label, its place and their line."""
        rows = []
        for record, place in self.said[attempt]:
            person = record.get("kind") == "human"
            label, note = _label(record, attempt), _note(record, attempt)
            cells = [_k(f"{record.get('person')} (a person)" if person else record.get("judge")), _k(self._call_text(record, attempt))]
            cells += [_k(f"Picture {label}" if label else DASH), _n(f"{place} of {len(record['ranking_attempts'])}")]
            rows.append(_tr(cells + [_p(_said(note) if note else DASH)]))
        if not rows:
            return _note_p("No judge or person has ranked this attempt.")
        head = _tr([_th("Who"), _th("Call"), _th("Shown as"), _th("Place", "n"), _th("Their line about this picture")])
        return _table([head], rows, "stack")

    def _call_text(self, record: dict[str, Any], attempt: str) -> str:
        kind, repeat, shown = record.get("kind"), record.get("repeat"), record["ranking_attempts"]
        if kind == "human":
            return "blind sheet"
        if kind == "pairwise":
            return f"pair with {', '.join(self.name(a) for a in shown if a != attempt)}, repeat {repeat}"
        if kind == "peak":
            return f"its own sizes ({', '.join(map(str, sorted(parse_attempt_id(a)[1] for a in shown)))} wide), repeat {repeat}"
        return f"ranking, repeat {repeat}"

    @staticmethod
    def _closing_text(stats: dict[str, Any]) -> _Html:
        text = stats.get("final_text")
        if isinstance(text, str) and text.strip():
            return _h('<blockquote class="said">{}</blockquote>', text.strip())
        if stats.get("stop") == "imported":
            return _note_p("This picture was typed outside the harness and imported, so no closing text was recorded.")
        return _note_p("The model wrote no closing text.")

    def _trouble(self, attempt: str) -> _Html:
        """The refusal and the error, each under its own heading, when the attempt has one.

        What a provider sent with its block is the provider's text, not the model's, and is headed so.
        """
        stats = self.stats[attempt]
        blocked = stats.get("refusal_detected_by") == "provider"
        titles = (("refusal_text", "The provider's message with its block" if blocked else "The refusal"), ("error", "The error"))
        return _join(_h('<h2>{}</h2>\n<pre class="said">{}</pre>\n', title, stats[key]) for key, title in titles if stats.get(key))

    def _earlier(self, attempt: str) -> Any:
        """How the runs this attempt replaced ended, oldest first, when it was run again."""
        if attempt not in self.retried:
            return ""
        runs = _join((_h("<li>{}</li>", self._earlier_run(run)) for run in self.retried[attempt]["earlier"]), "\n")
        return _h(
            '<h2 id="earlier">Earlier runs of this attempt</h2>\n<p class="muted">This attempt was run again. Everything '
            "else on this page is the run that stands; the earlier ones, oldest first, ended like this.</p>\n<ol>\n{}\n</ol>\n",
            runs,
        )

    @staticmethod
    def _settings(stats: dict[str, Any]) -> _Html:
        settings = stats.get("model_settings")
        if not isinstance(settings, dict) or not settings:
            return _note_p("No settings were saved with this attempt.")
        shown = []
        for key in [key for key in _SETTINGS if key in settings] + sorted(key for key in settings if key not in _SETTINGS):
            unset = key in ("reasoning", "max_output_tokens")
            if key != "name" and (key in _SETTINGS or settings[key] not in (None, {}, [])):
                shown.append((_SETTINGS.get(key, key), _setting(settings, key) if unset else _pairs(settings[key])))
        return _facts(shown)

    @staticmethod
    def _figures(ref: AttemptRef, stats: dict[str, Any]) -> _Html:
        if not stats:
            return _note_p("This attempt has no readable stats.json.")
        after = score.nudges_after_cutoff(ref, stats)
        if stats.get("nudges") is not None and after is not None:
            stats = {**stats, "nudges_after_cutoff": after}  # worked out from the transcript where the attempt did not save it
        keys = [key for key in _FIGURES if key in stats]
        keys += sorted(key for key in stats if key not in _FIGURES and key not in _SHOWN_ELSEWHERE)
        figures = _facts(((_FIGURES.get(key, key.replace("_", " ")), _pairs(stats[key])) for key in keys), "figures")
        return _h('{}\n<p class="muted small">{} means the figure was not recorded.</p>', figures, DASH)

    # ---- the transcript ----

    def _transcript(self, ref: AttemptRef, stats: dict[str, Any]) -> _Html:
        typed = self._picture_file(ref)
        if not ref.transcript_path.is_file():
            if stats.get("stop") == "imported":
                why = "None: this picture was typed outside the harness and imported."
            elif stats.get("discards"):
                kept = _plural(stats["discards"], "discarded conversation")
                why = f"None was kept: the attempt ended in an error. {kept} can be read under {ref.directory / 'discards'}."
            else:
                why = "None was saved with this attempt."
            return _h("{}\n{}", _note_p("{}", why), typed)
        try:
            data = read_json(ref.transcript_path)
            messages = [m for m in data["messages"] if isinstance(m, dict)]
            calls = [c for c in data.get("calls") or [] if isinstance(c, dict)]
            replies = sum(1 for m in messages if m.get("role") == "assistant")
            timings = iter(calls if len(calls) == replies else [])  # one entry per model reply, when the two line up
            blocks = [_message(m, next(timings, None) if m.get("role") == "assistant" else None) for m in messages]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):  # not JSON, or not the shape the harness writes
            return _h("{}\n{}", _note_p("The transcript at {} could not be read.", ref.transcript_path), typed)
        return _h(
            '<p class="muted">{}, oldest first. Rows are shown as typed.</p>\n<div class="transcript">\n{}\n</div>\n{}',
            _plural(len(messages), "message"), _join(blocks, "\n"), typed,
        )

    @staticmethod
    def _picture_file(ref: AttemptRef) -> Any:
        try:
            text = ref.picture_txt.read_text(encoding="utf-8")
        except (OSError, ValueError):
            return ""
        return _h('<details class="long"><summary>The picture file, picture.txt</summary><pre class="rows">{}</pre></details>', text)


def _message(message: dict[str, Any], call: dict[str, Any] | None) -> _Html:
    """One message of a transcript: who spoke, its text, and any tool calls with their rows as typed."""
    role, content, tool_calls = message.get("role"), message.get("content"), message.get("tool_calls")
    parts = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict) or part.get("type") != "text":
            parts.append(_note_p("(a part that is not text)"))
        elif str(part.get("text") or "").strip():
            parts.append(_h('<pre class="{}">{}</pre>', "rows" if role == "tool" else "said", part["text"]))
    parts += [_tool_call(c) for c in (tool_calls if isinstance(tool_calls, list) else []) if isinstance(c, dict)]
    meta = ""
    if call:
        usage = call.get("usage") or {}
        meta = (
            f"call {call.get('step')}: {_plain(call.get('seconds'))} s, {_plain(usage.get('input_tokens'))} tokens in, "
            f"{_plain(usage.get('output_tokens'))} out, stopped for {call.get('stop_reason')}"
        )
    return _h(
        '<details class="message {}"{}><summary><b>{}</b> <span class="muted">{}</span></summary>{}</details>',
        role if role in _ROLES else "other", _Html("" if role == "system" else " open"), _ROLES.get(role, role or "message"),
        meta, _join(parts) or _note_p("(empty)"),
    )


def _tool_call(call: dict[str, Any]) -> _Html:
    """A tool call as the model sent it: the palette entries and the rows, one per line, exactly as typed."""
    name, arguments = call.get("name"), call.get("arguments")
    if not isinstance(arguments, dict):
        return _h(
            '<div class="call"><p><b>{}</b> <span class="muted">with arguments that are not an object, as sent</span></p>'
            '<pre class="rows">{}</pre></div>',
            name, arguments if isinstance(arguments, str) else _plain(arguments),
        )
    palette, rows, start = arguments.get("palette"), arguments.get("rows"), arguments.get("start_row")
    head, parts = [], []
    if "start_row" in arguments:
        head.append(f"start_row {_plain(start)}")
    if isinstance(rows, list):
        head.append(_plural(len(rows), "row"))
    if isinstance(palette, list):
        head.append(_plural(len(palette), "palette colour"))
        entries = (_h("{}{}", _swatch(entry), entry if isinstance(entry, str) else _plain(entry)) for entry in palette)
        parts.append(_h('<pre class="rows">{}</pre>', _join(entries, "\n")))
    if isinstance(rows, list):
        numbered = isinstance(start, int) and not isinstance(start, bool)
        lines = (
            _h('<span class="ln">{}</span>{}', start + i if numbered else "", row if isinstance(row, str) else _plain(row))
            for i, row in enumerate(rows)
        )
        parts.append(_h('<pre class="rows">{}</pre>', _join(lines, "\n")))
    other = {key: value for key, value in arguments.items() if key not in ("palette", "rows", "start_row")}
    if other:
        parts.append(_h('<pre class="rows">{}</pre>', _plain(other)))
    return _h('<div class="call"><p><b>{}</b> <span class="muted">{}</span></p>{}</div>', name, ", ".join(head), _join(parts))


def _swatch(entry: Any) -> _Html:
    """A square of the colour a palette entry defines, when the entry is well formed."""
    parts = entry.split() if isinstance(entry, str) else []
    if len(parts) == 4 and all(p.isascii() and p.isdigit() and len(p) <= 3 and int(p) <= 255 for p in parts[1:]):
        return _h('<i class="swatch" style="background:rgb({},{},{})"></i>', *(int(p) for p in parts[1:]))
    return _Html('<i class="swatch none"></i>')


# ---------------------------------------------------------------- building


def build_report(
    layout: RunLayout,
    cfg: "RunConfig | None" = None,
    scores: dict[str, Any] | None = None,
    paintings_dir: Path | str | None = None,
) -> Path:
    """Write report/index.html and one page per attempt under report/attempts/. Returns the index path.

    `scores` is what score.compute_scores returns; it is computed here when
    not given. Paintings are looked for in `paintings_dir`, else where the
    config says, else in the default place.
    """
    if scores is None:
        scores = score.compute_scores(layout, cfg, paintings_dir)
    report = _Report(layout, cfg, scores, paintings_dir)
    written = set()
    for attempt in report.refs:
        page = report.detail(attempt)
        write_text(page, report.attempt_page(attempt))
        written.add(page)
    # A page left by an earlier build, for an attempt that is gone, would show a picture the run no longer holds.
    for old in (layout.report_dir / "attempts").glob("*/*/*/*.html"):
        if old not in written:
            old.unlink()
    write_text(report.index, report.index_page())  # last: its presence marks a finished build
    return report.index


# ---------------------------------------------------------------- the style

# Neutral greys around the pictures, so that the surround does not tint how their colours read. The one hue is the
# blue of links and of the rank-score meter. State dots use the reserved status colours and always sit beside the word.
_CSS = r"""
:root {
  color-scheme: light dark;
  --bg: #f3f3f2; --surface: #ffffff; --text: #1c1c1b; --muted: #61615d; --line: #d6d6d2; --link: #1f5fbf;
  --meter: #2a78d6; --track: #cde2fb;
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #151515; --surface: #212121; --text: #ebebe9; --muted: #a3a39e; --line: #3a3a3a; --link: #8fb8ff;
    --meter: #3987e5; --track: #0d366b;
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.page { max-width: 112rem; margin: 0 auto; padding: 1.5rem clamp(1rem, 3vw, 2.5rem) 4rem; }
h1 { font-size: 1.5rem; font-weight: 600; line-height: 1.25; margin: 0; }
h2 { font-size: 1.2rem; font-weight: 600; margin: 3rem 0 0.75rem; padding-top: 1.1rem; border-top: 1px solid var(--line); }
h2.quiet { font-size: 1rem; margin: 1.1rem 0 0.3rem; padding: 0; border: 0; }
h3 { font-size: 1.05rem; font-weight: 600; margin: 2rem 0 0.5rem; }
h4 { font-size: 1rem; font-weight: 600; margin: 0 0 0.2rem; }
p { margin: 0.35rem 0; max-width: 88ch; }
a { color: var(--link); text-decoration: none; }
a:hover, a:focus-visible { text-decoration: underline; }
:focus-visible { outline: 2px solid var(--link); outline-offset: 2px; }
:target { scroll-margin-top: 1rem; }
.muted, figcaption, .by, .sub { color: var(--muted); }
.by { font-weight: 400; font-size: 0.9em; }
.small { font-size: 0.87rem; }
figure { margin: 0; }
figcaption { font-size: 0.87rem; margin-top: 0.35rem; }
img { display: block; max-width: 100%; height: auto; }
img.typed { image-rendering: crisp-edges; image-rendering: pixelated; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.87em; }
code { overflow-wrap: anywhere; }
pre { margin: 0.4rem 0; }
.said { white-space: pre-wrap; overflow-wrap: anywhere; }
pre.said, blockquote.said { font: inherit; }
blockquote.said { margin: 0.4rem 0; padding: 0.1rem 0 0.1rem 1rem; border-left: 3px solid var(--line); max-width: 88ch; }
.gap { color: var(--muted); padding: 0.75rem; border: 1px dashed var(--line); border-radius: 0.4rem; }
.caution { padding-left: 0.7rem; border-left: 3px solid var(--warning); }
nav { display: flex; flex-wrap: wrap; gap: 0.2rem 1.4rem; margin-top: 0.6rem; }
.top nav:first-child { margin: 0 0 0.9rem; }
summary { cursor: pointer; }
details.long > summary, details.calls > summary, details.scale > summary { color: var(--link); }

/* Tables: names to the left, figures to the right in tabular digits. A wide table scrolls inside its own box. */
.scroll { overflow-x: auto; max-width: 100%; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { padding: 0.35rem 0.8rem; text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { padding-left: 0; }
thead th { font-size: 0.83rem; font-weight: 600; color: var(--muted); vertical-align: bottom; max-width: 13rem; }
thead th.group { text-align: center; border-bottom-color: var(--muted); }
tbody th { font-weight: 600; white-space: nowrap; }
tbody td { max-width: 46rem; overflow-wrap: break-word; }
/* Prose wraps, but a cell of it is never squeezed to a few letters a line: the table scrolls sideways first. */
td.p { min-width: min(14rem, 55vw); }
.n { text-align: right; }
td.n, td.k { white-space: nowrap; }
td.wrap { white-space: normal; }
td.quiet { color: var(--muted); }
.sub { display: block; font-size: 0.8rem; font-weight: 400; }
.sub.inline { display: inline-block; white-space: nowrap; }
.ghost { visibility: hidden; }
tr.total th, tr.total td { border-top: 1px solid var(--muted); }
ul.plain { margin: 0.4rem 0; padding-left: 1.2rem; font-size: 0.87rem; }
/* A table wider than the window keeps the names of its rows in view while the rest scrolls under them. */
.dense th, .dense td { padding-left: 0.5rem; padding-right: 0.5rem; }
.dense th:first-child, .dense td:first-child { padding-left: 0; }
.dense .sub { font-size: 0.75rem; }
.pin th[scope="row"], .pin th.pin { position: sticky; left: 0; z-index: 1; background: var(--bg); box-shadow: 1px 0 0 var(--line); }
/* A table that wraps the names of its rows sooner than it scrolls. */
.fit tbody th { white-space: normal; overflow-wrap: break-word; }
/* A judge's pooled order leads a group of rows, and its rankings follow it under dotted rules. */
tr.lead td, tr.repeat td { border-bottom: 0; }
tr.repeat td { border-top: 1px dotted var(--line); }
tr.repeat + tr.lead td, tr.lead + tr.lead td { border-top: 1px solid var(--line); }
tbody tr:last-child td { border-bottom: 1px solid var(--line); }

/* The leading table, and the standings at the other widths. The rank-score meter is one hue, the track a lighter step of
   it, with the figure printed beside it. The meter is as wide as its cell, which gives way before the table would scroll,
   so a bar is never cut short. */
.headline table { width: 100%; }
.headline th, .headline td { padding-left: 0.5rem; padding-right: 0.5rem; }
.headline td.bar, .headline th.bar { width: 6rem; min-width: 2.5rem; padding-left: 0; vertical-align: middle; }
.meter { display: block; position: relative; width: 100%; height: 10px; border-radius: 5px; background: var(--track); overflow: hidden; }
.meter > span { display: block; height: 100%; background: var(--meter); border-radius: 0 5px 5px 0; }
/* The interval, as a line across the bar on the same scale. */
.meter > i { position: absolute; top: 4px; height: 2px; background: var(--text); }
tr.tied td:first-child { font-weight: 600; }
tr.joined th, tr.joined td { border-bottom-style: dotted; }
/* The standings at the other widths keep to the measure of the leading table on a wide page. */
.widths { max-width: 46rem; }
.widths h4 { margin-top: 1.4rem; }

.state { white-space: nowrap; font-size: 0.87rem; font-weight: 400; }
.state i { display: inline-block; width: 0.6rem; height: 0.6rem; margin-right: 0.35rem; border-radius: 50%; background: var(--muted); }
.state.complete i { background: var(--good); }
.state.incomplete i { background: var(--warning); }
.state.refused i { background: var(--serious); }
.state.error i { background: var(--critical); }
h1 .state { font-size: 0.95rem; margin-left: 0.5rem; vertical-align: 0.15em; }

.summary { display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 6fr); gap: 0.5rem 3.5rem; align-items: start; margin-top: 0.6rem; }

/* One painting: the real painting stays in view on the left while its attempts scroll past on the right. */
.painting { margin-top: 1.5rem; }
.painting > h3 { margin-top: 0; }
.split { display: grid; grid-template-columns: minmax(15rem, 30%) minmax(0, 1fr); column-gap: 2rem; align-items: start; }
.real { position: sticky; top: 1rem; }
.real img { width: min(100%, calc(72vh * var(--ar))); }
.set + .set { margin-top: 2.25rem; }
.cards {
  list-style: none; margin: 0.75rem 0; padding: 0; display: grid; gap: 1rem;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, var(--card-min)), 1fr));
}
.card { padding: 0.6rem; background: var(--surface); border: 1px solid var(--line); border-radius: 0.5rem; }
.card.scale { background: none; border-style: dashed; }
.card img { width: 100%; }
.card p { margin: 0.25rem 0 0; font-size: 0.87rem; overflow-wrap: anywhere; }
.card .who { margin-top: 0.5rem; font-size: 1rem; font-weight: 600; }
.place { display: inline-block; min-width: 1.2rem; margin-right: 0.15rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.place:empty { display: none; }
.calls, .notes { margin-top: 0.5rem; }
/* The operator's notes: text from outside the records, ruled off from what the page works out. */
.runnotes { padding-left: 0.8rem; border-left: 3px solid var(--line); }
.thumb { width: 7rem; }

/* One attempt: the picture large, beside the real painting. */
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; align-items: start; margin-top: 1.25rem; }
.pair img { width: min(100%, calc(74vh * var(--ar))); }
.pair .scale { margin-top: 0.75rem; }
.pair .scale img { margin-top: 0.5rem; width: min(100%, calc(40vh * var(--ar))); }
dl { margin: 0.5rem 0; }
dt { color: var(--muted); }
dd { margin: 0; overflow-wrap: anywhere; }
.facts { display: grid; grid-template-columns: minmax(8rem, max-content) minmax(0, 1fr); max-width: 76rem; }
.facts > div { display: contents; }
.facts dt, .facts dd { padding: 0.35rem 0; border-bottom: 1px solid var(--line); }
.facts dt { padding-right: 2rem; }
.figures { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 13.5rem), 1fr)); gap: 0 2rem; }
.figures > div { padding: 0.4rem 0; border-bottom: 1px solid var(--line); }
.figures dt { font-size: 0.83rem; }
.figures dd { font-variant-numeric: tabular-nums; }
.transcript { max-width: 76rem; }
.message { margin: 0.5rem 0; padding: 0.5rem 0.8rem; background: var(--surface); border: 1px solid var(--line); border-radius: 0.5rem; }
.message.assistant { border-left: 3px solid var(--muted); }
.call { margin-top: 0.4rem; }
pre.rows {
  overflow-x: auto; max-height: 34rem; padding: 0.5rem 0.7rem; background: var(--bg); border-radius: 0.35rem;
  white-space: pre; line-height: 1.35;
}
.ln { display: inline-block; min-width: 3ch; margin-right: 1.5ch; text-align: right; color: var(--muted); user-select: none; -webkit-user-select: none; }
.swatch { display: inline-block; width: 0.85em; height: 0.85em; margin-right: 1ch; border: 1px solid var(--line); vertical-align: -0.08em; }
.swatch.none { border-style: dashed; }

/* Below this the headline table has the whole width, and comes first: the scores are the first thing on a small screen. */
@media (max-width: 80rem) {
  .summary { grid-template-columns: minmax(0, 1fr); }
  .summary .models { order: -1; }
  /* A key gives way before a cell of prose is pushed out of the window; one name stays whole a while longer. */
  td.k { white-space: normal; }
  td.k.name { white-space: nowrap; }
}
@media (max-width: 60rem) {
  td.k.name { white-space: normal; }
  .thumb { width: 4.5rem; }
  .facts { grid-template-columns: minmax(0, 1fr); }
  .facts dt { padding-bottom: 0; border-bottom: 0; }
  .facts dd { padding-top: 0; }
  .split { display: block; }
  /* A narrow window has no room beside the attempts, so the real painting rides along the top edge. */
  .real { top: 0; z-index: 1; padding: 0.5rem 0; margin-bottom: 0.75rem; background: var(--bg); border-bottom: 1px solid var(--line); }
  .real img { width: min(100%, calc(26vh * var(--ar))); }
  .set { scroll-margin-top: calc(26vh + 4rem); }
}
@media (max-aspect-ratio: 1/1) {
  .pair { grid-template-columns: minmax(0, 1fr); }
  .pair img { width: min(100%, calc(40vh * var(--ar))); }
}
/* A phone: the leading table drops its bars and keeps its figures, and a table of prose sets each row out as a block.
   Standings keep the sets each figure rests on, which say how far two figures can be compared; the table of means from
   before the standings, which has more columns, drops them as it always did. */
@media (max-width: 40rem) {
  .headline td.bar, .headline th.bar, .headline:not(.standings) .sets { display: none; }
  .standings td.sets { white-space: normal; }
  .headline th, .headline td { padding-left: 0.3rem; padding-right: 0.3rem; }
  .headline table { font-size: 0.93rem; }
  /* A pinned column of names gives way to the figures it names: the names wrap. */
  .pin tbody th[scope="row"] { white-space: normal; overflow-wrap: break-word; min-width: 6.5rem; max-width: 8rem; }
  .stack thead { display: none; }
  .stack table, .stack tbody, .stack tr, .stack tbody tr td, .stack tbody tr th { display: block; }
  .stack tr { padding: 0.6rem 0; border-bottom: 1px solid var(--line); }
  .stack tbody tr td, .stack tbody tr th { border: 0; padding: 0.1rem 0; min-width: 0; max-width: none; text-align: left; white-space: normal; }
  /* The short cells of a row run on one line, and its prose follows beneath. */
  .stack tbody tr td.k, .stack tbody tr td.n { display: inline; }
  .stack td.k + td.k::before, .stack td.k + td.n::before, .stack td.n + td.k::before { content: " · "; color: var(--muted); }
  .stack tbody tr td.none { display: none; }
}
"""
