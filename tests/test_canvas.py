"""The canvas: what the write_rows tool stores, what it rejects, and what its replies may say."""

from __future__ import annotations

import re

import pytest

from paintbench.canvas import FIELD_PATTERNS, PALETTE, TEMPLATES, VALUE, Canvas, parse_picture, reply_line_template

K32 = "k" * 32


def canvas(width=32, height=22) -> Canvas:
    """A palette canvas with two colours defined, k and w."""
    c = Canvas(width, height)
    c.write({"palette": ["k 20 40 90", "w 240 240 230"]})
    return c


def test_a_well_formed_call_stores_its_palette_and_its_rows():
    c = Canvas(8, 3)
    reply = c.write({"palette": ["k 20 40 90", "K 240 240 230"], "start_row": 1, "rows": ["kkkkKKKK", " kkkk KKKK "]})
    assert reply.splitlines() == ["palette: 2 colours defined.", "rows 1-2: stored 2, rejected 0.", "stored so far: 2 of 3. missing: 3."]
    assert c.palette == {"k": (20, 40, 90), "K": (240, 240, 230)}  # a code is case-sensitive
    assert c.rows == {1: "kkkkKKKK", 2: "kkkkKKKK"}  # spaces inside a row are ignored

    codes = [chr(o) for o in range(0x21, 0x7F) if chr(o) not in "#="]  # printable ASCII but for space, # and =
    wide = Canvas(len(codes), 1)
    wide.write({"palette": [f"{code} {i} {i} {i}" for i, code in enumerate(codes)], "start_row": 1, "rows": ["".join(codes)]})
    assert len(wide.palette) == 92 and wide.is_complete()


def test_a_bad_row_is_rejected_alone_and_the_reply_names_it():
    c = canvas()
    rows = [K32] * 16
    rows[6] = "k" * 33  # row 7: too long
    rows[11] = "k" * 31 + "q"  # row 12: a code that is not in the palette
    reply = c.write({"start_row": 1, "rows": rows})
    assert reply.splitlines() == [
        "rows 1-16: stored 14, rejected 2.",
        "row 7 rejected: 33 characters, need 32.",
        "row 12 rejected: unknown character 'q'.",
        "stored so far: 14 of 22. missing: 7, 12, 17-22.",
    ]
    assert sorted(c.rows) == [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16]
    assert c.missing_rows_message() == "The picture is not finished. Missing rows: 7, 12, 17-22. Send them with write_rows."
    c.write({"start_row": 7, "rows": ["k" * 31]})  # wrong again: counted, but not as a first submission
    assert (c.rows_rejected, c.rows_rejected_first_submission) == (3, 2)


@pytest.mark.parametrize("entry", ["k 1 2", "# 1 2 3", "= 1 2 3", "k 256 0 0", "k: 1, 2, 3", ["k", 1, 2, 3]])
def test_a_malformed_palette_entry_is_rejected_alone_by_a_line_that_repeats_none_of_it(entry):
    c = Canvas(4, 2)
    lines = c.write({"palette": ["a 1 2 3", entry]}).splitlines()
    assert lines[:2] == ["palette: 1 colour defined.", TEMPLATES["palette_entry_rejected"].format(i=2)]
    assert c.palette == {"a": (1, 2, 3)}


@pytest.mark.parametrize(
    "fault",
    [
        {"start_row": 1, "rows": [K32] * 17},  # more than 16 rows in one call
        {"start_row": 22, "rows": [K32, K32]},  # runs past the bottom of the grid
        {"start_row": 0, "rows": [K32]},  # there is no row 0
        {"rows": [K32]},  # no start_row
        {"start_row": 1, "rows": K32},  # rows that are not a list
    ],
)
def test_a_call_that_breaks_a_rule_for_the_whole_call_stores_nothing(fault):
    c = Canvas(32, 22)
    reply = c.write({"palette": ["k 20 40 90"], **fault})
    assert reply.startswith("call rejected: ") and reply.endswith("stored so far: 0 of 22. missing: 1-22.")
    assert c.palette == {} and c.rows == {}  # not even its palette


def test_defining_a_code_again_recolours_the_rows_already_stored():
    c = Canvas(4, 2)
    c.write({"palette": ["k 0 0 0"], "start_row": 1, "rows": ["kkkk"]})
    c.write({"palette": ["k 10 10 60", "w 255 255 255"], "start_row": 2, "rows": ["kkww"]})
    assert c.picture().pixels() == [[(10, 10, 60)] * 4, [(10, 10, 60)] * 2 + [(255, 255, 255)] * 2]


def test_value_encoding_stores_rows_of_six_digit_hex_cells_and_has_no_palette():
    c = Canvas(3, 3, VALUE)
    reply = c.write({"palette": ["k 1 2 3"], "start_row": 1, "rows": ["1b3a6e FFFFFF 000000", "1B3A6E FFFFFF", "1B3A6E #FFFFF 000000"]})
    assert reply.splitlines() == [
        "rows 1-3: stored 1, rejected 2.",
        "row 2 rejected: 2 cells, need 3.",
        "row 3 rejected: cell 2 is not six hex digits.",
        "stored so far: 1 of 3. missing: 2-3.",
    ]
    assert c.palette == {}
    c.write({"start_row": 2, "rows": ["010203 040506 070809", "FFFFFF FFFFFF FFFFFF"]})
    text = c.picture().to_text()
    assert text == "1B3A6E FFFFFF 000000\n010203 040506 070809\nFFFFFF FFFFFF FFFFFF\n"
    assert parse_picture(text).pixels()[:2] == [[(0x1B, 0x3A, 0x6E), (255, 255, 255), (0, 0, 0)], [(1, 2, 3), (4, 5, 6), (7, 8, 9)]]


def test_a_picture_is_complete_once_every_row_is_stored_and_is_saved_as_it_was_typed():
    c = canvas(4, 3)
    c.write({"start_row": 1, "rows": ["kkkk", "wwww"]})
    assert not c.is_complete()
    with pytest.raises(ValueError, match="missing rows 3"):
        c.picture()
    reply = c.write({"start_row": 3, "rows": ["kwkw"]})
    assert reply.splitlines() == ["row 3: stored 1, rejected 0.", "all 3 rows stored. Reply without calling the tool to finish."]
    assert c.is_complete()

    c.write({"start_row": 1, "rows": ["wwww"]})  # a stored row can be replaced
    c.write({"start_row": 2, "rows": ["kkk"]})  # a rejected resend leaves the stored row in place
    text = c.picture().to_text()
    assert text == "= k 20 40 90\n= w 240 240 230\nwwww\nwwww\nkwkw\n"
    assert parse_picture(text).pixels() == c.picture().pixels()


# ---------------------------------------------------------------- replies say nothing about the picture

PICTURE_CONTENT = ["sswwbbdd", "kk", "ssss wwww", "236 226 200", "k 20 40 90", "1B3A6E", "1B3A6E FFFFFF", "wave", "blue"]


def calls_of_every_kind(row: str) -> list:
    """Calls that between them draw every line the tool can say, for a canvas 8 wide and 5 tall whose rows read `row`."""
    return [
        {"palette": ["s 236 226 200", "w 244 240 230", "b 30 60 120", "d 12 24 66", "ss 9 9 9"], "start_row": 1, "rows": [row, row]},
        {"start_row": 3, "rows": [row[:-1]]},  # the last cell cut short
        {"start_row": 3, "rows": [row[:-7], "q" + row[1:], 7]},  # too few cells, a character that is no code, not text
        {"palette": ["ss 9 9 9"] * 93},  # more rejected entries than a reply lists one by one
        {"start_row": 1, "rows": [row] * 17},
        {"start_row": 5, "rows": [row, row]},
        {"start_row": 0, "rows": [row]},
        {"start_row": "top", "rows": [row]},
        {"start_row": 1, "rows": row},
        row,
        {},
        {"start_row": 3, "rows": [row, row, row]},  # the last rows
    ]


def typed(call) -> list[str]:
    """The rows and palette entries a call typed as text."""
    if not isinstance(call, dict):
        return []
    return [text for key in ("palette", "rows") if isinstance(call.get(key), list) for text in call[key] if isinstance(text, str)]


def test_no_field_of_a_reply_template_can_hold_picture_content():
    for name, template in TEMPLATES.items():
        for field in re.findall(r"\{(\w+)\}", template):
            for content in PICTURE_CONTENT:
                assert not re.fullmatch(FIELD_PATTERNS[field], content), f"{field!r} in template {name!r} could carry {content!r}"


def test_every_line_the_tool_says_is_a_template_and_repeats_nothing_that_was_typed():
    drawn = set()
    for encoding, row in [(PALETTE, "sswwbbdd"), (VALUE, "1B3A6E " * 7 + "FFFFFF")]:
        c = Canvas(8, 5, encoding)
        replies = [(None, c.reject_unknown_tool()), (None, c.missing_rows_message())]
        replies += [(call, c.write(call)) for call in calls_of_every_kind(row)]
        assert c.is_complete()
        for call, reply in replies:
            names = [reply_line_template(line) for line in reply.splitlines()]
            assert None not in names, f"a line that is no template: {reply!r}"
            drawn.update(names)
            for text in typed(call):
                assert not any(text[i : i + 4] in reply for i in range(len(text) - 3)), f"the reply repeats part of {text!r}: {reply!r}"
    assert drawn == set(TEMPLATES)  # every template was drawn, so every one was checked
