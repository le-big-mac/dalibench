"""The canvas: the only place the picture format rules live.

A Canvas holds what one contestant has typed so far. `Canvas.write` takes the
arguments of one `write_rows` tool call, validates them, stores what is valid
and returns the reply text. Replies are built from the fixed templates below
and say nothing about what is in the picture or how good it is.

Two encodings:

palette  The model defines colours, each a one-character code with red, green
         and blue values. One character is one cell. Spaces inside a row are
         ignored, so a model can type in blocks to keep count.
value    No palette. Every cell is six hex digits, cells separated by spaces.

A call that breaks a rule for the call as a whole stores nothing: more rows
than the strip limit, or rows outside the grid. Otherwise a bad palette entry
or a bad row is rejected alone and the rest of the call is stored.

Palette entries are applied in the order sent, however many there are, so a
code defined twice in one call ends with its later colour. A reply stays
bounded all the same: the first 92 rejected entries of a call get a line
each, and one more line gives the count when there were others.

The same module reads and writes the on-disk picture format (`picture.txt`).
A picture typed outside the harness in that format loads unchanged, which is
how `paintbench import` brings one into a run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

PALETTE = "palette"
VALUE = "value"
ENCODINGS = (PALETTE, VALUE)

# Rows per tool call. At 128 x 128 that keeps one call near 2,000 characters in
# palette encoding and near 7,200 in value encoding.
DEFAULT_STRIP_ROWS = {PALETTE: 16, VALUE: 8}

RGB = tuple[int, int, int]

_HEX6 = re.compile(r"[0-9A-Fa-f]{6}")


def is_code(ch: Any) -> bool:
    """A palette code is one printable ASCII character other than space, # or =."""
    return isinstance(ch, str) and len(ch) == 1 and "!" <= ch <= "~" and ch not in "#="


MAX_COLOURS = sum(1 for o in range(0x21, 0x7F) if is_code(chr(o)))  # 92

# Rejected palette entries that one reply lists line by line. A call that defines each code once has at
# most 92 entries, so for such a call every rejected entry is still named.
MAX_REJECTIONS_LISTED = MAX_COLOURS


class PictureFormatError(ValueError):
    """A picture file does not follow the on-disk format."""


# --------------------------------------------------------------------------
# Reply templates. Every line a contestant ever receives from the tool, and
# the nudge, is one of these with its fields filled in. The fields carry row
# numbers, counts, limits and at most one echoed character; never picture
# content. tests/test_canvas.py checks both properties.
# --------------------------------------------------------------------------

TEMPLATES: dict[str, str] = {
    "palette_defined": "palette: {n} {colours} defined.",
    # The whole rule for an entry, so that the line is right whatever was wrong with the entry.
    "palette_entry_rejected": (
        "palette entry {i} rejected: write it as text: a one-character code (an ASCII letter, digit or symbol; "
        "not a space, # or =), then red, green and blue as whole numbers from 0 to 255, with spaces between, "
        "for example 'k 20 40 90'."
    ),
    # Only after {limit} lines of the kind above, so {n} is always more than one.
    "palette_rejections_not_listed": "palette: {n} entries rejected in all; only the first {limit} are listed above.",
    "rows_summary": "rows {first}-{last}: stored {stored}, rejected {rejected}.",
    "row_summary": "row {first}: stored {stored}, rejected {rejected}.",
    "row_bad_length": "row {row} rejected: {got} characters, need {need}.",
    "row_unknown_char": "row {row} rejected: unknown character {char}.",
    "row_bad_cells": "row {row} rejected: {got} cells, need {need}.",
    "row_bad_value": "row {row} rejected: cell {cell} is not six hex digits.",
    "row_not_text": "row {row} rejected: a row must be text.",
    "call_too_many_rows": "call rejected: {got} rows sent, at most {limit} per call. Nothing from this call was stored.",
    "call_out_of_range": "call rejected: rows {lo} to {hi} are outside 1 to {height}. Nothing from this call was stored.",
    "call_row_out_of_range": "call rejected: row {lo} is outside 1 to {height}. Nothing from this call was stored.",
    "call_bad_start_row": "call rejected: start_row must be a whole number from 1 to {height}. Nothing from this call was stored.",
    "call_bad_field": "call rejected: {field} must be a list of text entries. Nothing from this call was stored.",
    "call_bad_arguments": "call rejected: the arguments must be an object with start_row and rows. Nothing from this call was stored.",
    "call_empty": "call rejected: no rows were sent.",
    "call_unknown_tool": "call rejected: the only tool is write_rows. Nothing from this call was stored.",
    "progress": "stored so far: {stored} of {height}. missing: {missing}.",
    "complete": "all {height} rows stored. Reply without calling the tool to finish.",
    "nudge": "The picture is not finished. Missing rows: {missing}. Send them with write_rows.",
}

# What each template field may hold, as a regular expression. This is the
# whole vocabulary of a reply.
FIELD_PATTERNS: dict[str, str] = {
    "n": r"\d+",
    "i": r"\d+",
    "first": r"\d+",
    "last": r"\d+",
    "stored": r"\d+",
    "rejected": r"\d+",
    "row": r"\d+",
    "got": r"\d+",
    "need": r"\d+",
    "cell": r"\d+",
    "limit": r"\d+",
    "height": r"\d+",
    "lo": r"-?\d+",  # a row number as sent, which may be zero or negative
    "hi": r"-?\d+",
    "missing": r"\d+(?:-\d+)?(?:, \d+(?:-\d+)?)*",
    "char": r"(?:'[!-~]'|U\+[0-9A-F]{4,6})",
    "colours": r"colours?",
    "field": r"(?:palette|rows)",
}


def _template_regex(template: str) -> re.Pattern[str]:
    out = []
    pos = 0
    for m in re.finditer(r"\{(\w+)\}", template):
        out.append(re.escape(template[pos : m.start()]))
        out.append(FIELD_PATTERNS[m.group(1)])
        pos = m.end()
    out.append(re.escape(template[pos:]))
    return re.compile("".join(out))


REPLY_LINE_PATTERNS: dict[str, re.Pattern[str]] = {name: _template_regex(t) for name, t in TEMPLATES.items()}


def reply_line_template(line: str) -> str | None:
    """Name of the template a reply line was built from, or None if it matches none."""
    for name, pattern in REPLY_LINE_PATTERNS.items():
        if pattern.fullmatch(line):
            return name
    return None


def format_ranges(numbers: list[int]) -> str:
    """[7, 12, 17, 18, 19] -> '7, 12, 17-19'."""
    nums = sorted(set(numbers))
    out: list[str] = []
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


def _show_char(ch: str) -> str:
    if "!" <= ch <= "~":
        return f"'{ch}'"
    return f"U+{ord(ch):04X}"


def _parse_palette_entry(entry: Any) -> tuple[str, RGB] | None:
    """'k 20 40 90' -> ('k', (20, 40, 90)); None if malformed."""
    if not isinstance(entry, str):
        return None
    parts = entry.split()
    if len(parts) != 4 or not is_code(parts[0]):
        return None
    values = []
    for p in parts[1:]:
        if not (p.isascii() and p.isdigit()) or len(p) > 3:
            return None
        v = int(p)
        if v > 255:
            return None
        values.append(v)
    return parts[0], (values[0], values[1], values[2])


def _as_row_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        digits = value.strip()
        # Nine digits is far beyond any row number; int() raises on a digit string thousands long.
        if digits.isascii() and digits.isdigit() and len(digits) <= 9:
            return int(digits)
    return None


# --------------------------------------------------------------------------
# A finished picture, and the on-disk format
# --------------------------------------------------------------------------


@dataclass
class Picture:
    """A complete typed picture.

    palette encoding: `rows` holds H strings of W codes and `palette` maps
        each code to its colour.
    value encoding: `rows` holds H tuples of W upper-case 'RRGGBB' strings
        and `palette` is empty.
    """

    width: int
    height: int
    encoding: str
    rows: list[Any]
    palette: dict[str, RGB] = field(default_factory=dict)
    comments: list[str] = field(default_factory=list)

    def pixels(self) -> list[list[RGB]]:
        if self.encoding == PALETTE:
            return [[self.palette[ch] for ch in row] for row in self.rows]
        return [[(int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)) for v in row] for row in self.rows]

    def colours_used(self) -> int:
        if self.encoding == PALETTE:
            return len({ch for row in self.rows for ch in row})
        return len({v for row in self.rows for v in row})

    def to_text(self) -> str:
        """The picture.txt form: comments, palette lines, then one row per line."""
        lines = [f"# {c}" if c else "#" for c in self.comments]
        if self.encoding == PALETTE:
            lines += [f"= {code} {r} {g} {b}" for code, (r, g, b) in self.palette.items()]
            lines += list(self.rows)
        else:
            lines += [" ".join(row) for row in self.rows]
        return "\n".join(lines) + "\n"


def parse_picture(text: str) -> Picture:
    """Read a picture.txt.

    Lines starting with '#' are comments. Lines starting with '=' define a
    palette colour ('= k 20 40 90'). Every other non-blank line is a row. A
    file with palette lines is palette-encoded (spaces inside rows are
    ignored); a file with none is value-encoded (rows of six-digit hex values
    separated by spaces). Width is the row length and height the row count.
    """
    palette: dict[str, RGB] = {}
    comments: list[str] = []
    raw_rows: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        if line.startswith("#"):
            comments.append(line[1:].strip())
        elif line.startswith("="):
            parsed = _parse_palette_entry(line[1:])
            if parsed is None:
                raise PictureFormatError(f"line {lineno}: malformed palette line {line!r}")
            palette[parsed[0]] = parsed[1]
        else:
            raw_rows.append((lineno, line))
    if not raw_rows:
        raise PictureFormatError("no rows found")

    if palette:
        rows: list[Any] = []
        for lineno, line in raw_rows:
            row = line.replace(" ", "")
            for ch in row:
                if ch not in palette:
                    raise PictureFormatError(f"line {lineno}: character {_show_char(ch)} is not in the palette")
            rows.append(row)
        encoding = PALETTE
    else:
        rows = []
        for lineno, line in raw_rows:
            cells = line.split()
            for i, cell in enumerate(cells, 1):
                if not _HEX6.fullmatch(cell):
                    raise PictureFormatError(f"line {lineno}: cell {i} is not six hex digits (and the file has no palette lines)")
            rows.append(tuple(c.upper() for c in cells))
        encoding = VALUE

    width = len(rows[0])
    for (lineno, _), row in zip(raw_rows, rows):
        if len(row) != width:
            raise PictureFormatError(f"line {lineno}: row has {len(row)} cells, the first row has {width}")
    if width == 0:
        raise PictureFormatError("rows are empty")
    return Picture(width=width, height=len(rows), encoding=encoding, rows=rows, palette=palette, comments=comments)


def load_picture(path: Any) -> Picture:
    with open(path, encoding="utf-8") as f:
        return parse_picture(f.read())


# --------------------------------------------------------------------------
# The canvas
# --------------------------------------------------------------------------


class Canvas:
    """Holds the palette and the stored rows of one attempt."""

    def __init__(self, width: int, height: int, encoding: str = PALETTE, strip_rows: int | None = None):
        if encoding not in ENCODINGS:
            raise ValueError(f"unknown encoding {encoding!r}; use one of {ENCODINGS}")
        if width < 1 or height < 1:
            raise ValueError("width and height must be at least 1")
        self.width = width
        self.height = height
        self.encoding = encoding
        self.strip_rows = strip_rows if strip_rows else DEFAULT_STRIP_ROWS[encoding]
        self.palette: dict[str, RGB] = {}
        self.rows: dict[int, Any] = {}
        # Counters for the "care" figures.
        self.write_calls = 0
        self.calls_rejected = 0
        self.rows_rejected = 0
        self.palette_entries_rejected = 0
        self._first_seen: set[int] = set()
        self._first_rejected: set[int] = set()

    # ---- state ----

    def is_complete(self) -> bool:
        return len(self.rows) == self.height

    def missing_rows(self) -> list[int]:
        return [n for n in range(1, self.height + 1) if n not in self.rows]

    def missing_rows_message(self) -> str:
        """The nudge sent when a model stops with rows missing."""
        return TEMPLATES["nudge"].format(missing=format_ranges(self.missing_rows()))

    @property
    def rows_rejected_first_submission(self) -> int:
        """Rows whose first submission was rejected for a format fault."""
        return len(self._first_rejected)

    def stats(self) -> dict[str, int]:
        return {
            "write_calls": self.write_calls,
            "calls_rejected": self.calls_rejected,
            "rows_rejected": self.rows_rejected,
            "rows_rejected_first_submission": self.rows_rejected_first_submission,
            "palette_entries_rejected": self.palette_entries_rejected,
            "rows_stored": len(self.rows),
            "colours_defined": len(self.palette),
        }

    def state_dict(self) -> dict[str, Any]:
        """Everything stored so far, complete or not, as JSON-serialisable data."""
        return {
            "width": self.width,
            "height": self.height,
            "encoding": self.encoding,
            "palette": {code: list(rgb) for code, rgb in self.palette.items()},
            "rows": {str(n): (row if isinstance(row, str) else list(row)) for n, row in sorted(self.rows.items())},
            "missing": self.missing_rows(),
        }

    def picture(self, comments: list[str] | None = None) -> Picture:
        if not self.is_complete():
            raise ValueError(f"picture is not complete: missing rows {format_ranges(self.missing_rows())}")
        rows = [self.rows[n] for n in range(1, self.height + 1)]
        palette = dict(self.palette) if self.encoding == PALETTE else {}
        return Picture(self.width, self.height, self.encoding, rows, palette, list(comments or []))

    # ---- the tool ----

    def write(self, arguments: Any) -> str:
        """Apply one write_rows call and return the reply text."""
        self.write_calls += 1
        lines = self._apply(arguments)
        lines.append(self._state_line())
        return "\n".join(lines)

    def reject_unknown_tool(self) -> str:
        """The reply to a call naming a tool that does not exist."""
        self.calls_rejected += 1
        return "\n".join([TEMPLATES["call_unknown_tool"], self._state_line()])

    def _state_line(self) -> str:
        if self.is_complete():
            return TEMPLATES["complete"].format(height=self.height)
        return TEMPLATES["progress"].format(
            stored=len(self.rows), height=self.height, missing=format_ranges(self.missing_rows())
        )

    def _reject_call(self, template: str, **fields: Any) -> list[str]:
        self.calls_rejected += 1
        return [TEMPLATES[template].format(**fields)]

    def _apply(self, arguments: Any) -> list[str]:
        if not isinstance(arguments, dict):
            return self._reject_call("call_bad_arguments")

        uses_palette = self.encoding == PALETTE
        palette = arguments.get("palette") if uses_palette else None
        rows = arguments.get("rows")

        # Check the shape of the whole call before storing anything, so that a
        # rejected call stores nothing.
        if palette is not None and not isinstance(palette, list):
            return self._reject_call("call_bad_field", field="palette")
        if rows is not None and not isinstance(rows, list):
            return self._reject_call("call_bad_field", field="rows")
        rows = rows or []
        if not rows and not palette:
            return self._reject_call("call_empty")
        start = 0
        if rows:
            parsed_start = _as_row_number(arguments.get("start_row"))
            if parsed_start is None:
                return self._reject_call("call_bad_start_row", height=self.height)
            start = parsed_start
            if len(rows) > self.strip_rows:
                return self._reject_call("call_too_many_rows", got=len(rows), limit=self.strip_rows)
            last = start + len(rows) - 1
            if start < 1 or last > self.height:
                if len(rows) == 1:
                    return self._reject_call("call_row_out_of_range", lo=start, height=self.height)
                return self._reject_call("call_out_of_range", lo=start, hi=last, height=self.height)

        lines: list[str] = []
        if palette is not None:
            rejected: list[int] = []
            for i, entry in enumerate(palette, 1):
                parsed = _parse_palette_entry(entry)
                if parsed is None:
                    rejected.append(i)
                else:
                    self.palette[parsed[0]] = parsed[1]
            self.palette_entries_rejected += len(rejected)
            n = len(self.palette)
            lines.append(TEMPLATES["palette_defined"].format(n=n, colours="colour" if n == 1 else "colours"))
            # A line per rejected entry would let one call draw a reply of any size, so the lines stop at
            # MAX_REJECTIONS_LISTED and one more gives the count. The entries themselves are never capped: a
            # code may be defined again, so a list longer than the 92 codes can be well formed throughout.
            lines += [TEMPLATES["palette_entry_rejected"].format(i=i) for i in rejected[:MAX_REJECTIONS_LISTED]]
            if len(rejected) > MAX_REJECTIONS_LISTED:
                lines.append(TEMPLATES["palette_rejections_not_listed"].format(n=len(rejected), limit=MAX_REJECTIONS_LISTED))

        if rows:
            stored = 0
            faults: list[str] = []
            for offset, raw in enumerate(rows):
                number = start + offset
                value, fault = self._check_row(number, raw)
                first_time = number not in self._first_seen
                self._first_seen.add(number)
                if fault is None:
                    self.rows[number] = value
                    stored += 1
                else:
                    faults.append(fault)
                    self.rows_rejected += 1
                    if first_time:
                        self._first_rejected.add(number)
            summary = "row_summary" if len(rows) == 1 else "rows_summary"
            lines.append(
                TEMPLATES[summary].format(first=start, last=start + len(rows) - 1, stored=stored, rejected=len(faults))
            )
            lines += faults
        return lines

    def _check_row(self, number: int, raw: Any) -> tuple[Any, str | None]:
        """Return (value to store, None) or (None, the rejection line)."""
        if not isinstance(raw, str):
            return None, TEMPLATES["row_not_text"].format(row=number)
        if self.encoding == PALETTE:
            row = raw.replace(" ", "")
            if len(row) != self.width:
                return None, TEMPLATES["row_bad_length"].format(row=number, got=len(row), need=self.width)
            for ch in row:
                if ch not in self.palette:
                    return None, TEMPLATES["row_unknown_char"].format(row=number, char=_show_char(ch))
            return row, None
        cells = raw.split()
        if len(cells) != self.width:
            return None, TEMPLATES["row_bad_cells"].format(row=number, got=len(cells), need=self.width)
        for i, cell in enumerate(cells, 1):
            if not _HEX6.fullmatch(cell):
                return None, TEMPLATES["row_bad_value"].format(row=number, cell=i)
        return tuple(c.upper() for c in cells), None
