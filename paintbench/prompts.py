"""The fixed wording: contestant prompt, tool definition, and judge prompt.

These texts are fixed points of the benchmark. Changing any of them changes
the task, so each has a version string that is saved with every attempt and
every judgement, and a run never mixes pictures made under two wordings.
docs/method.md shows the texts and describes the judging protocol.
"""

from __future__ import annotations

from .canvas import PALETTE, VALUE
from .types import ToolSpec

PROMPT_VERSION = "A5"  # the contestant's system prompt, task message, and tool definition, as they stand below
# The judge versions cover every word a judge can be sent: the judge prompt and the ask-again message.
# C2: the ask-again message states the required form. The prompt itself is as in C1.
JUDGE_PROMPT_VERSION = "C2"  # judge prompt (ranking and pairwise)
PEAK_PROMPT_VERSION = "C2-peak1"  # judge prompt variant for ranking one model's sizes

TOOL_NAME = "write_rows"

# ---------------------------------------------------------------- the contestant prompt

SYSTEM_PROMPT_PALETTE = """\
You are taking part in a memory test. You will be asked to type a small picture of a well-known painting from memory, one character per cell.

You have one tool, write_rows. It stores the palette and the rows you type, and it checks only their format. You have no other tools. You cannot look anything up, run code, or see your picture.

Type every row yourself, cell by cell. Do not describe the picture in place of typing it."""

# Value encoding: the words "the palette and" are dropped. Nothing else changes.
SYSTEM_PROMPT_VALUE = SYSTEM_PROMPT_PALETTE.replace("It stores the palette and the rows you type", "It stores the rows you type")
assert SYSTEM_PROMPT_VALUE != SYSTEM_PROMPT_PALETTE

_TASK_HEAD = """\
Painting: {first_line}

Type it as a grid {W} cells wide and {H} cells tall. Row 1 is the top of the picture. The first character of a row is the left edge.

How to write it
"""

_PALETTE_CHOOSE = "- Choose your own palette, with as many colours as you want. Each colour has a one-character code (any character except a space, # or =) and red, green and blue values from 0 to 255.\n"
_PALETTE_ENTRY = "- Write each palette entry as the code, then red, green and blue, separated by spaces, for example: k 20 40 90. Put the entries in the palette field of write_rows, with your first rows or before them. You can add colours in a later call. If you send a code that is already defined, it is redefined: every cell typed with that code takes the new colour, in rows already stored too.\n"
_PALETTE_ROWS = """\
- Send the palette and the rows with write_rows. Each row must be exactly {W} characters long and use only your palette codes.
- You may put spaces inside a row to help you count, for example between blocks of eight. Spaces are ignored.
"""

_BULLETS_VALUE = """\
- Type every cell as its own colour: six hex digits for red, green and blue, for example 1B3A6E. Put one space between cells.
- Send the rows with write_rows. Each row must have exactly {W} cells.
"""

_SEND_ROWS = "- Send at most {strip} rows per call. Keep calling write_rows until all {H} rows are stored.\n"
_REJECTED_ROW = "- If a row is rejected, correct it and send it again.\n\n"

_HOW_THE_WORK_GOES = """\
How the work goes
- You do not have to finish in one turn. After each write_rows call you get its result back and carry on from there, for up to {turns} turns in all.
- Work a strip at a time: type a strip of rows, send it, then go on to the next. Do not work out every row of the picture before you send the first strip.
"""
_AIM = "Aim\nMake the picture as close as you can to the real painting: what is in it, where each thing is, its shape, its size and its colours."
DEFAULT_TURNS = 40  # the default limit on model calls per attempt (config: limits.max_steps)


def system_prompt(encoding: str = PALETTE) -> str:
    return SYSTEM_PROMPT_VALUE if encoding == VALUE else SYSTEM_PROMPT_PALETTE


def painting_line(title: str, artist: str, date: str, version_note: str = "") -> str:
    """'{title}, by {artist}, {date}. {version_note}' with no trailing space."""
    line = f"{title}, by {artist}, {date}."
    note = (version_note or "").strip()
    return f"{line} {note}" if note else line


def task_message(first_line: str, width: int, height: int, strip: int, encoding: str = PALETTE, turns: int = DEFAULT_TURNS) -> str:
    """The user message that starts every attempt. `first_line` comes from painting_line().

    `turns` is the run's limit on model calls per attempt. The message states it, so that a model
    working in strips knows how many turns it has.
    """
    if turns < 2:
        raise ValueError(f"the task message says that the model is called again after each tool call; that is false with {turns} turn")
    bullets = _BULLETS_VALUE if encoding == VALUE else _PALETTE_CHOOSE + _PALETTE_ENTRY + _PALETTE_ROWS
    template = _TASK_HEAD + bullets + _SEND_ROWS + _REJECTED_ROW + _HOW_THE_WORK_GOES + "\n" + _AIM
    return template.format(first_line=first_line, W=width, H=height, strip=strip, turns=turns)


# ---------------------------------------------------------------- the tool


def write_rows_tool(encoding: str = PALETTE) -> ToolSpec:
    """The one tool. Adapters translate this without changing the name or the descriptions."""
    if encoding == VALUE:
        # Value encoding has no palette field. The two descriptions that
        # mention the palette or the counting spaces are reworded to match.
        return ToolSpec(
            name=TOOL_NAME,
            description="Store picture rows. It checks format only. A stored row can be replaced by sending it again.",
            parameters={
                "type": "object",
                "properties": {
                    "start_row": {
                        "type": "integer",
                        "description": "The row number of the first row in 'rows'. Row 1 is the top.",
                    },
                    "rows": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Consecutive rows, starting at start_row. Each row is its cells as six hex digits, separated by spaces.",
                    },
                },
            },
        )
    return ToolSpec(
        name=TOOL_NAME,
        description="Store palette colours and picture rows. It checks format only. A stored row can be replaced by sending it again.",
        parameters={
            "type": "object",
            "properties": {
                "palette": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional. Colours to define, each written as a one-character code then red, green and blue from 0 to 255, for example 'k 20 40 90'.",
                },
                "start_row": {
                    "type": "integer",
                    "description": "The row number of the first row in 'rows'. Row 1 is the top.",
                },
                "rows": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Consecutive rows, starting at start_row. Spaces are ignored.",
                },
            },
        },
    )


# ---------------------------------------------------------------- the judge prompt

REFERENCE_LABEL = "REFERENCE"
PICTURE_LABEL = "Picture {label}"

_JUDGE_HEAD = "You are judging attempts to depict a well-known painting from memory on a small grid.\n\n"

_JUDGE_SAME_SIZE = (
    'First you will see the real painting, labelled REFERENCE. Then you will see {n} attempts, each introduced by a '
    'label such as "Picture K". Each attempt is a grid {W} cells wide and {H} cells tall, shown enlarged as solid '
    "squares. The attempts were made by different sources. You are not told which is which, and the labels are random.\n\n"
)

# Peak-size variant: one model's attempts at several grid sizes, all shown at
# the same display size. Only the sentence describing the grids differs.
_JUDGE_MIXED_SIZE = (
    'First you will see the real painting, labelled REFERENCE. Then you will see {n} attempts, each introduced by a '
    'label such as "Picture K". The attempts are grids of different sizes, from {Wmin} to {Wmax} cells wide, each '
    "shown enlarged to the same display size as solid squares. You are not told which is which, and the labels are random.\n\n"
)

_JUDGE_TAIL = """\
Rank all {n} attempts from best to worst as a depiction of this painting, the way a person who knows the painting well would judge them. Take into account what is in the picture, whether each thing is in the right place, whether its shape is right, and the colours. No ties.

Reply in exactly this form:
OVERALL: K, G, ... (best first, all {n} labels)
NOTES:
K: one line saying what in it does or does not match the painting. Name specific things, and count them where the number matters.
(one line per label)
CONFIDENCE: one sentence saying how confident you are, and which neighbouring pairs were hardest to separate"""


def judge_prompt(n: int, width: int, height: int) -> str:
    """Instructions for ranking n attempts of one painting at one grid size."""
    return (_JUDGE_HEAD + _JUDGE_SAME_SIZE + _JUDGE_TAIL).format(n=n, W=width, H=height)


def judge_prompt_peak(n: int, min_width: int, max_width: int) -> str:
    """Instructions for ranking one model's attempts at different grid sizes."""
    return (_JUDGE_HEAD + _JUDGE_MIXED_SIZE + _JUDGE_TAIL).format(n=n, Wmin=min_width, Wmax=max_width)


# Sent once when a reply does not parse. A reply is rejected for many reasons (no OVERALL line, two of them, a
# separator that is not a comma, a second ranking further down), and the judge gets this one text whichever it
# was. So it states what is required and says nothing about what the reply did wrong.
JUDGE_RETRY = (
    "That reply could not be read as a ranking. Give the ranking once, on one line that begins OVERALL: and then "
    "lists every label exactly once, best first, as capital letters separated by commas, with nothing else on that "
    "line. Do not write the word OVERALL anywhere else in the reply. The labels are: {labels}. "
    "Reply again in exactly the required form, ranking all {n}."
)
