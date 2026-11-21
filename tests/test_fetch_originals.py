"""scripts/fetch_originals.py: an original is downloaded, checked against its recorded hash, and the reference image built from it.

The network is a stand-in and the painting is made up, in a temporary directory.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from urllib.error import HTTPError

import pytest
import yaml
from PIL import Image

from paintbench.reference import prepare_reference, sha256_file

from conftest import ROOT

spec = importlib.util.spec_from_file_location("fetch_originals", ROOT / "scripts" / "fetch_originals.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)

URL = "https://museum.example/images/alpha.jpg"


class Painting:
    """A made-up painting whose record names an original at URL: its directory, and what the two files should hold."""

    def __init__(self, tmp_path: Path):
        self.directory = tmp_path / "paintings" / "alpha"
        self.directory.mkdir(parents=True)
        Image.new("RGB", (48, 32), (200, 30, 10)).save(tmp_path / "original.jpg")
        facts = prepare_reference(tmp_path / "original.jpg", tmp_path / "reference.png")
        self.original, self.reference_sha256 = (tmp_path / "original.jpg").read_bytes(), facts["reference_sha256"]
        meta = {
            "slug": "alpha", "prompt": {"title": "Alpha", "artist": "A Painter", "date": "1800"}, "grid_heights": {16: 11},
            "reference": {"file": "reference.png", "original_file": "original.jpg", "download_url": URL,
                          "original_sha256": facts["original_sha256"], "reference_sha256": facts["reference_sha256"]},
        }
        (self.directory / "meta.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")

    def fetch(self) -> int:
        return fetch.main(["--paintings-dir", str(self.directory.parent), "alpha"])

    def files(self) -> list[str]:
        return sorted(path.name for path in self.directory.iterdir())


@pytest.fixture
def painting(tmp_path) -> Painting:
    return Painting(tmp_path)


def serve(monkeypatch, answer: bytes | Exception) -> None:
    """Stand in for the network: URL answers with these bytes, or asking for it raises this exception."""

    def urlopen(request, timeout=None):
        assert request.full_url == URL
        if isinstance(answer, Exception):
            raise answer
        served = io.BytesIO(answer)
        served.headers = {"Content-Type": "image/jpeg"}
        return served

    monkeypatch.setattr(fetch, "urlopen", urlopen)


def test_a_good_download_is_checked_and_saved_and_the_reference_image_is_built(painting, monkeypatch):
    serve(monkeypatch, painting.original)
    assert painting.fetch() == 0
    assert painting.files() == ["meta.yaml", "original.jpg", "reference.png"]
    assert (painting.directory / "original.jpg").read_bytes() == painting.original
    assert sha256_file(painting.directory / "reference.png") == painting.reference_sha256


def test_a_download_with_the_wrong_hash_is_refused_and_the_file_is_not_kept(painting, monkeypatch):
    serve(monkeypatch, painting.original + b"changed since")
    assert painting.fetch() == 1
    assert painting.files() == ["meta.yaml"]  # no original, no part of one, and no reference image built from it


def test_a_failed_download_says_how_to_fetch_the_file_by_hand_and_the_exit_code_is_not_zero(painting, monkeypatch, capsys):
    serve(monkeypatch, HTTPError(URL, 403, "Forbidden", None, None))
    assert painting.fetch() == 1
    by_hand = painting.directory / "original.jpg"
    assert f"open {URL} in a browser, save the image as {by_hand}, and run this script again" in capsys.readouterr().err

    by_hand.write_bytes(painting.original)  # done as it says, with the address still refusing
    assert painting.fetch() == 0
    assert sha256_file(painting.directory / "reference.png") == painting.reference_sha256
