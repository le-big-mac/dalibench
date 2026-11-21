"""Turn typed pictures into PNGs: every cell a solid square, no smoothing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .canvas import Picture, load_picture

LONGEST_SIDE = 640  # pixels, approximately; the cell size is a whole number

MISSING_ROW_COLOUR = (255, 0, 255)  # only used when drawing an unfinished canvas for the report


def cell_size(width: int, height: int, longest_side: int = LONGEST_SIDE) -> int:
    return max(1, round(longest_side / max(width, height)))


def _grid_image(pixels: list[list[tuple[int, int, int]]]) -> Image.Image:
    height, width = len(pixels), len(pixels[0])
    img = Image.new("RGB", (width, height))
    img.putdata([rgb for row in pixels for rgb in row])
    return img


def _enlarge(img: Image.Image, longest_side: int) -> Image.Image:
    cell = cell_size(img.width, img.height, longest_side)
    return img.resize((img.width * cell, img.height * cell), Image.NEAREST)


def render_picture(picture: Picture, longest_side: int = LONGEST_SIDE) -> Image.Image:
    return _enlarge(_grid_image(picture.pixels()), longest_side)


def render_same_size(pictures: Sequence[Picture], longest_side: int = LONGEST_SIDE) -> list[Image.Image]:
    """Pictures of different grid sizes, all drawn to one display size (a peak-size set).

    Every picture comes out at the same pixel width: the width render_picture
    gives the finest grid, where that grid has whole cells and a longest side
    of about `longest_side`. Each height follows from the picture's own
    proportions, so the heights agree as nearly as the grids do (16 x 25 and
    128 x 196 are not quite the same shape).

    Cells may differ by a pixel here, which they never do in render_picture.
    Giving every picture whole cells of its own is what made the pictures of
    one call differ in size, by up to a sixth for a tall painting. A shared
    width is not a whole number of cells for every grid: 640 pixels across 48
    cells is 13.3 pixels a cell, so some cells are drawn 13 pixels wide and
    others 14. Nothing is smoothed: the scaling is nearest-neighbour, so every
    pixel has the colour of exactly one cell. With the standard widths (16,
    32, 64, 128), each of which divides the widest, every cell is whole.
    """
    finest = max(pictures, key=lambda p: (p.width, p.height))
    width = finest.width * cell_size(finest.width, finest.height, longest_side)
    return [
        _grid_image(p.pixels()).resize((width, max(1, round(p.height * width / p.width))), Image.NEAREST) for p in pictures
    ]


def render_file(picture_txt: Path | str, picture_png: Path | str, longest_side: int = LONGEST_SIDE) -> Picture:
    """picture.txt -> picture.png. Returns the picture that was read."""
    picture = load_picture(picture_txt)
    out = Path(picture_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    render_picture(picture, longest_side).save(out, format="PNG")
    return picture


def render_reference_grid(reference_path: Path | str, width: int, height: int, longest_side: int = LONGEST_SIDE) -> Image.Image:
    """The real painting reduced to the grid (each cell the average of its area), drawn as solid cells.

    For the report only. It is never shown to a judge: reduced copies of the
    reference are not ranked against typed attempts.
    """
    with Image.open(reference_path) as ref:
        small = ref.convert("RGB").resize((width, height), Image.BOX)
    return _enlarge(small, longest_side)


def render_canvas_state(state: dict[str, Any], longest_side: int = LONGEST_SIDE) -> Image.Image:
    """Draw an unfinished canvas (canvas.json), with missing rows in a flat marker colour. Report only."""
    width, height = int(state["width"]), int(state["height"])
    palette = {code: tuple(rgb) for code, rgb in (state.get("palette") or {}).items()}
    rows = state.get("rows") or {}
    pixels: list[list[tuple[int, int, int]]] = []
    for n in range(1, height + 1):
        row = rows.get(str(n))
        if row is None:
            pixels.append([MISSING_ROW_COLOUR] * width)
        elif isinstance(row, str):
            pixels.append([palette.get(ch, MISSING_ROW_COLOUR) for ch in row])
        else:
            pixels.append([(int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)) for v in row])
    return _enlarge(_grid_image(pixels), longest_side)
