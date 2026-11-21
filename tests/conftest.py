"""Shared fixtures and helpers.

Every test runs offline: any attempt to reach a host other than the loopback
address fails the test.

The repository leaves out the museum photographs and the reference image of
a work in copyright; .gitattributes lists them (`export-ignore`), and
scripts/fetch_originals.py fetches them. A test that reads such a file is
skipped when the file is absent (`left_out_of_export` says which those are).
"""

from __future__ import annotations

import random
import re
import socket
from pathlib import Path

import pytest

from paintbench.config import RunConfig

ROOT = Path(__file__).resolve().parent.parent
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def picture_text(width: int, height: int, seed: str) -> str:
    """A typed picture as picture.txt holds it: three palette lines, then the rows. Each seed gives another picture."""
    rng = random.Random(seed)
    lines = [f"= {code} {rng.randrange(256)} {rng.randrange(256)} {rng.randrange(256)}" for code in "abc"]
    lines += ["".join(rng.choice("abc") for _ in range(width)) for _ in range(height)]
    return "\n".join(lines) + "\n"


def _export_ignored() -> list[re.Pattern[str]]:
    """The paths .gitattributes marks `export-ignore`, as patterns over paths relative to the repository root."""
    attributes = ROOT / ".gitattributes"
    if not attributes.is_file():
        return []
    patterns = []
    for line in attributes.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if len(words) < 2 or words[0].startswith("#") or "export-ignore" not in words[1:]:
            continue
        glob = re.escape(words[0].strip("/")).replace(r"\*", "[^/]*").replace(r"\?", "[^/]")
        anywhere = "" if "/" in words[0].rstrip("/") else "(?:.*/)?"  # a pattern with no slash is a name, in any directory
        patterns.append(re.compile(f"{anywhere}{glob}(?:/.*)?"))  # a directory that is left out takes its files with it
    return patterns


def left_out_of_export(path: Path | str) -> bool:
    """Whether the public repository leaves this file out: .gitattributes marks it `export-ignore`."""
    try:
        relative = Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:  # not a file of this repository
        return False
    return any(pattern.fullmatch(relative) for pattern in _export_ignored())


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Tests are offline: no model API, no network. Loopback is allowed (the human-sheet server test)."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def check(address):
        host = address[0] if isinstance(address, tuple) else None
        if host is not None and host not in _LOOPBACK:
            raise RuntimeError(f"a test tried to open a network connection to {address!r}")

    def connect(self, address, *args, **kwargs):
        check(address)
        return real_connect(self, address, *args, **kwargs)

    def connect_ex(self, address, *args, **kwargs):
        check(address)
        return real_connect_ex(self, address, *args, **kwargs)

    def getaddrinfo(host, *args, **kwargs):
        if host is not None and host not in _LOOPBACK and host != "":
            raise RuntimeError(f"a test tried to resolve the host name {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@pytest.fixture
def paintings_dir() -> Path:
    return ROOT / "paintings"


@pytest.fixture
def make_config(tmp_path):
    """A RunConfig over the real paintings, writing runs under tmp_path. Keyword arguments override keys."""

    def make(**overrides) -> RunConfig:
        data = {
            "run_id": "test",
            "paintings": ["great-wave"],
            "widths": [32],
            "repeats": 1,
            "paintings_dir": str(ROOT / "paintings"),
            "runs_dir": str(tmp_path / "runs"),
            "retries": {"backoff_seconds": 0.001, "backoff_max_seconds": 0.002},
            "models": [{"name": "typist", "provider": "fake", "model": "fake-1"}],
        }
        data.update(overrides)
        return RunConfig.from_dict(data)

    return make
