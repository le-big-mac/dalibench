"""The fixed wording: every text a contestant or a judge is sent, and the definition of the one tool.

Changing any of it changes the benchmark, so each text is held here word for word, beside its version."""

from __future__ import annotations

from PIL import Image

from paintbench.canvas import PALETTE, VALUE
from paintbench.prompts import (
    JUDGE_PROMPT_VERSION,
    JUDGE_RETRY,
    PEAK_PROMPT_VERSION,
    PICTURE_LABEL,
    PROMPT_VERSION,
    REFERENCE_LABEL,
    judge_prompt,
    judge_prompt_peak,
    painting_line,
    system_prompt,
    task_message,
    write_rows_tool,
)
from paintbench.render import render_file

from conftest import picture_text

SYSTEM = """\
You are taking part in a memory test. You will be asked to type a small picture of a well-known painting from memory, one character per cell.

You have one tool, write_rows. It stores the palette and the rows you type, and it checks only their format. You have no other tools. You cannot look anything up, run code, or see your picture.

Type every row yourself, cell by cell. Do not describe the picture in place of typing it."""

# 32 x 22, 16 rows a call, 40 turns.
TASK = """\
Painting: Under the Wave off Kanagawa (often called The Great Wave), by Katsushika Hokusai, about 1831.

Type it as a grid 32 cells wide and 22 cells tall. Row 1 is the top of the picture. The first character of a row is the left edge.

How to write it
- Choose your own palette, with as many colours as you want. Each colour has a one-character code (any character except a space, # or =) and red, green and blue values from 0 to 255.
- Write each palette entry as the code, then red, green and blue, separated by spaces, for example: k 20 40 90. Put the entries in the palette field of write_rows, with your first rows or before them. You can add colours in a later call. If you send a code that is already defined, it is redefined: every cell typed with that code takes the new colour, in rows already stored too.
- Send the palette and the rows with write_rows. Each row must be exactly 32 characters long and use only your palette codes.
- You may put spaces inside a row to help you count, for example between blocks of eight. Spaces are ignored.
- Send at most 16 rows per call. Keep calling write_rows until all 22 rows are stored.
- If a row is rejected, correct it and send it again.

How the work goes
- You do not have to finish in one turn. After each write_rows call you get its result back and carry on from there, for up to 40 turns in all.
- Work a strip at a time: type a strip of rows, send it, then go on to the next. Do not work out every row of the picture before you send the first strip.

Aim
Make the picture as close as you can to the real painting: what is in it, where each thing is, its shape, its size and its colours."""

JUDGE = """\
You are judging attempts to depict a well-known painting from memory on a small grid.

First you will see the real painting, labelled REFERENCE. Then you will see 6 attempts, each introduced by a label such as "Picture K". Each attempt is a grid 32 cells wide and 22 cells tall, shown enlarged as solid squares. The attempts were made by different sources. You are not told which is which, and the labels are random.

Rank all 6 attempts from best to worst as a depiction of this painting, the way a person who knows the painting well would judge them. Take into account what is in the picture, whether each thing is in the right place, whether its shape is right, and the colours. No ties.

Reply in exactly this form:
OVERALL: K, G, ... (best first, all 6 labels)
NOTES:
K: one line saying what in it does or does not match the painting. Name specific things, and count them where the number matters.
(one line per label)
CONFIDENCE: one sentence saying how confident you are, and which neighbouring pairs were hardest to separate"""

# Ranking one model's pictures of several sizes: the same prompt but for its second paragraph.
JUDGE_PEAK_GRIDS = 'First you will see the real painting, labelled REFERENCE. Then you will see 6 attempts, each introduced by a label such as "Picture K". The attempts are grids of different sizes, from 16 to 128 cells wide, each shown enlarged to the same display size as solid squares. You are not told which is which, and the labels are random.'

JUDGE_ASKED_AGAIN = "That reply could not be read as a ranking. Give the ranking once, on one line that begins OVERALL: and then lists every label exactly once, best first, as capital letters separated by commas, with nothing else on that line. Do not write the word OVERALL anywhere else in the reply. The labels are: Q, M, Z. Reply again in exactly the required form, ranking all 3."


def test_system_prompt_is_the_fixed_text():
    assert system_prompt(PALETTE) == SYSTEM
    assert system_prompt(VALUE) == SYSTEM.replace("stores the palette and the rows", "stores the rows")


def test_task_message_is_the_fixed_text():
    line = painting_line("Under the Wave off Kanagawa (often called The Great Wave)", "Katsushika Hokusai", "about 1831")
    assert task_message(line, 32, 22, 16) == task_message(line, 32, 22, 16, PALETTE, 40) == TASK
    assert PROMPT_VERSION == "A5"  # the name this wording is saved under: a change to the text needs a new one
    assert painting_line("The Scream", "Edvard Munch", "1893", "The 1893 version.") == "The Scream, by Edvard Munch, 1893. The 1893 version."


def test_the_one_tool_is_the_fixed_definition():
    start_row = {"type": "integer", "description": "The row number of the first row in 'rows'. Row 1 is the top."}
    tool = write_rows_tool(PALETTE)
    assert tool.name == "write_rows"
    assert tool.description == "Store palette colours and picture rows. It checks format only. A stored row can be replaced by sending it again."
    assert tool.parameters == {
        "type": "object",
        "properties": {
            "palette": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional. Colours to define, each written as a one-character code then red, green and blue from 0 to 255, for example 'k 20 40 90'.",
            },
            "start_row": start_row,
            "rows": {"type": "array", "items": {"type": "string"}, "description": "Consecutive rows, starting at start_row. Spaces are ignored."},
        },
    }
    value = write_rows_tool(VALUE)  # the same tool with no palette
    assert value.name == "write_rows"
    assert value.description == "Store picture rows. It checks format only. A stored row can be replaced by sending it again."
    rows = {"type": "array", "items": {"type": "string"}, "description": "Consecutive rows, starting at start_row. Each row is its cells as six hex digits, separated by spaces."}
    assert value.parameters == {"type": "object", "properties": {"start_row": start_row, "rows": rows}}


def test_judge_prompt_and_the_ask_again_message_are_the_fixed_text():
    assert judge_prompt(6, 32, 22) == JUDGE
    same = JUDGE.split("\n\n")
    assert judge_prompt_peak(6, 16, 128).split("\n\n") == [same[0], JUDGE_PEAK_GRIDS, *same[2:]]
    assert JUDGE_RETRY.format(labels="Q, M, Z", n=3) == JUDGE_ASKED_AGAIN
    assert REFERENCE_LABEL in JUDGE and PICTURE_LABEL.format(label="K") in JUDGE  # the labels the images are sent under
    assert (JUDGE_PROMPT_VERSION, PEAK_PROMPT_VERSION) == ("C2", "C2-peak1")


def test_a_typed_picture_renders_as_solid_squares_with_no_smoothing(tmp_path):
    (tmp_path / "picture.txt").write_text(picture_text(32, 22, "typed"), encoding="utf-8")
    picture = render_file(tmp_path / "picture.txt", tmp_path / "picture.png")
    assert (picture.width, picture.height, picture.encoding) == (32, 22, PALETTE)
    with Image.open(tmp_path / "picture.png") as png:
        image = png.convert("RGB")
    assert image.size == (640, 440)  # cells of 20 pixels
    for y, row in enumerate(picture.pixels()):
        for x, colour in enumerate(row):
            cell = image.crop((x * 20, y * 20, x * 20 + 20, y * 20 + 20))
            assert cell.getcolors() == [(400, colour)], f"cell ({x}, {y}) is not one solid colour"
