"""Painting metadata: the wording given to the model, the grid heights, the reference.

Each painting is a directory `paintings/<slug>/` holding `meta.yaml` and
`reference.png`. Adding a painting means adding such a directory and naming
the slug in a run config; no code changes.

The public repository leaves out the file each reference was built from
(`original.jpg`) and the reference image of a work in copyright.
`python scripts/fetch_originals.py <slug>` downloads the one and rebuilds
the other.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .prompts import painting_line

STANDARD_WIDTHS = (16, 32, 64, 128)


def default_paintings_dir() -> Path:
    """`$PAINTBENCH_PAINTINGS`, else ./paintings, else the paintings directory beside the package."""
    env = os.environ.get("PAINTBENCH_PAINTINGS")
    if env:
        return Path(env)
    cwd = Path.cwd() / "paintings"
    if cwd.is_dir():
        return cwd
    return Path(__file__).resolve().parent.parent / "paintings"


def grid_heights_for_ratio(height_over_width: float, widths: tuple[int, ...] = STANDARD_WIDTHS) -> dict[int, int]:
    """Height = width x ratio, rounded half up. Fixed in meta.yaml once computed."""
    return {w: max(1, int(w * height_over_width + 0.5)) for w in widths}


@dataclass
class Painting:
    slug: str
    title: str
    artist: str
    date: str
    version_note: str
    display_name: str
    grid_heights: dict[int, int]
    directory: Path
    reference_file: str = "reference.png"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def reference_path(self) -> Path:
        return self.directory / self.reference_file

    @property
    def first_line(self) -> str:
        """The text after 'Painting: ' in the task message."""
        return painting_line(self.title, self.artist, self.date, self.version_note)

    def height_for(self, width: int) -> int:
        try:
            return self.grid_heights[width]
        except KeyError:
            raise KeyError(
                f"painting {self.slug!r} has no grid height for width {width}; "
                f"add it under grid_heights in {self.directory / 'meta.yaml'}"
            ) from None


def load_painting(slug: str, paintings_dir: Path | str | None = None) -> Painting:
    base = Path(paintings_dir) if paintings_dir else default_paintings_dir()
    directory = base / slug
    meta_path = directory / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"no painting metadata at {meta_path}")
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    prompt = meta.get("prompt") or {}
    for key in ("title", "artist", "date"):
        if not prompt.get(key):
            raise ValueError(f"{meta_path}: prompt.{key} is required")
    heights = meta.get("grid_heights") or {}
    if not heights:
        raise ValueError(f"{meta_path}: grid_heights is required (a height for each width)")
    reference = meta.get("reference") or {}
    return Painting(
        slug=slug,
        title=str(prompt["title"]),
        artist=str(prompt["artist"]),
        date=str(prompt["date"]),
        version_note=str(prompt.get("version_note") or ""),
        display_name=str(meta.get("display_name") or prompt["title"]),
        grid_heights={int(w): int(h) for w, h in heights.items()},
        directory=directory,
        reference_file=str(reference.get("file") or "reference.png"),
        meta=meta,
    )
