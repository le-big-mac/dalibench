"""The human check: a person ranks a blind sheet, so the judge can be checked.

A sheet is the real painting and every complete attempt for one painting and
width, each under a random letter, in a random order. The person arranges the
attempts best to worst, and the ranking goes into the human-ranking record
(shape in layout.py), where scoring compares it with each judge. The page asks
the person what the judge prompt asks the judge.

A sheet takes 2 to 24 pictures. Each picture is known by one letter, there are
24 letters (I and O are left out), and one person ranks the whole sheet at a
sitting. A painting and width with more complete attempts than that cannot be
put on a sheet, in whole or in part, and make_sheet says so.

Blindness is the point:

- The key (letter -> attempt id) lives only in the JSON record on disk. The
  sheet, its source and the server's replies carry letters and pictures and
  nothing else: no model name, attempt id, attempt path or hash of one. The
  person's own ranking and notes are not secret, and the server shows them
  back to the person.
- Letters are drawn at random from the alphabet and the order shown is
  shuffled separately, so neither follows the attempts. Both come from a
  128-bit secret that stays in the record, so the key cannot be worked out
  from the page. A sheet made from a seed the caller chose is for tests:
  whoever guesses the seed has the key.
- Pictures are redrawn from picture.txt, so no file metadata comes along.

Beside the key, the record holds what the sheet was made from and what has
been said about it:

    sheet      the sheet's id, which its page sends with a ranking
    pictures   {attempt id: SHA-256 of the picture.txt that was shown}
    earlier    the rankings that later ones replaced, oldest first, each
               {ranking, ranking_attempts, notes, ranked_at}

A sheet shows the attempts as they were when it was made, and they can change:

- It is stale once a picture on it has been typed again or is no longer a
  complete attempt. A stale sheet is not served, and no ranking of it is
  recorded: it would be a ranking of pictures the run no longer holds.
- It is short once there are complete attempts that are not on it. A ranking
  of a short sheet is still a true ranking of the pictures it shows, so it is
  recorded, but the sheet is not given out again: check_sheet refuses it, and
  so does the server when it starts.

Either way the answer is a fresh sheet (force; --new on the command line). A
record with no "pictures" (a ranking brought in by `paintbench import`) is
compared on attempt ids alone.

A ranking is recorded only if it lists every letter on the sheet exactly once.
A person may save again: the last ranking stands, and the one it replaced is
kept under "earlier". Making a sheet again draws new letters, so it refuses to
replace a record that holds a ranking unless forced. A ranking typed from the
sheet it replaced is nearly always rejected, because its letters are not the
new sheet's. With many pictures the letters can be the same, so the page also
sends its sheet's id, and a ranking sent from the page of an earlier sheet is
refused instead of being read against the new key.

The sheet is one self-contained HTML file: images are data URIs, styles and
script are inline, and it loads nothing from anywhere. Served, its Save button
posts the ranking, and once a ranking is recorded the server sends the page
with that ranking and its notes in place, marked as saved, so that the sheet's
first order is never shown as the person's. Opened as a file, the page shows
the command line that records the ranking, and says that it cannot show one
saved earlier.

The server serves that one sheet on the one address it is given. A page is
never served on every interface, so the host has no default and "0.0.0.0" and
"::" are refused. An error inside the server is answered with a 500 whose text
is fixed.
"""

from __future__ import annotations

import base64
import collections
import hashlib
import html
import io
import ipaddress
import json
import random
import re
import secrets
import shlex
import socket
import socketserver
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from PIL import Image

from . import __version__
from .canvas import load_picture
from .config import check_name
from .layout import RunLayout, read_json, write_json, write_text
from .paintings import Painting
from .render import cell_size, render_picture

LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # no I or O: typed by hand they are read as 1, l and 0
MAX_BODY_BYTES = 64 * 1024  # a ranking and a page of notes are far smaller
_DRAIN_BYTES = 1024 * 1024  # how much of a refused body is read and thrown away before the connection closes
_SAVED_SLOT = b"const SAVED = null;"  # in the page's script: where the server puts a recorded ranking


class RankingError(ValueError):
    """The ranking cannot be recorded. The message says why, names no attempt, and may be shown to the person."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _record_path(layout: RunLayout, painting_slug: str, width: int, person: str) -> Path:
    """Both names become part of a file path, so neither may climb out of the run directory."""
    return layout.human_path(check_name("painting", painting_slug), int(width), check_name("person", person))


# --------------------------------------------------------------------------
# Making the sheet
# --------------------------------------------------------------------------


def make_sheet(
    layout: RunLayout, painting: Painting, width: int, person: str, seed: int | None = None, force: bool = False
) -> Path:
    """Write the blind sheet and its record (ranking null). Returns the sheet's path.

    Leave `seed` out: a 128-bit secret is then drawn and kept in the record,
    and the letters and order cannot be worked out from the page. A seed given
    here is for tests. It gives the same letters, order and page every time,
    and anyone who can guess it can work out the key.
    """
    width = int(width)
    record_path = _record_path(layout, painting.slug, width, person)
    if not force and record_path.is_file() and read_json(record_path).get("ranking"):
        raise FileExistsError(
            f"{record_path} already holds a ranking by {person}. A new sheet draws new letters and would discard it; "
            "make it again with --new (force=True) to replace it."
        )
    attempts = layout.complete_attempts(painting.slug, width)
    if not 2 <= len(attempts) <= len(LETTERS):
        problem = (
            f"a blind sheet takes 2 to {len(LETTERS)} pictures; {painting.slug} at width {width} has "
            f"{len(attempts)} complete attempts in {layout.root}"
        )
        if len(attempts) > len(LETTERS):
            problem += (
                f". The limit is {len(LETTERS)} because each picture is known by one letter (the alphabet without I and O) "
                "and one person ranks the whole sheet at a sitting. A sheet cannot hold part of a grid, "
                "so no human check can be made of this one"
            )
        raise ValueError(problem)
    if seed is None:
        seed = secrets.randbits(128)
    rng = random.Random(seed)
    letters = rng.sample(LETTERS, len(attempts))  # letters[i] is the label of attempts[i]
    order_shown = rng.sample(letters, len(letters))
    # The id goes on the page, so it is a one-way function of the seed: it names the sheet and leads back to nothing.
    sheet_id = hashlib.sha256(f"paintbench sheet {seed}".encode()).hexdigest()[:16]
    by_label = dict(zip(letters, attempts))
    height = painting.height_for(width)

    page = _sheet_html(
        painting,
        width,
        height,
        person,
        pictures=[(label, _picture_uri(by_label[label].picture_txt)) for label in order_shown],
        command=record_command(layout, painting.slug, width, person, sheet_id),
        sheet_id=sheet_id,
    )
    record = {
        "kind": "human",
        "painting": painting.slug,
        "width": width,
        "height": height,
        "person": person,
        "sheet": sheet_id,
        "labels": {label: by_label[label].id for label in sorted(letters)},
        "order_shown": order_shown,
        "pictures": {a.id: a.picture_sha256() for a in attempts},
        "ranking": None,
        "ranking_attempts": None,
        "notes": "",
        "earlier": [],
        "created_at": _now(),
        "ranked_at": None,
        "seed": seed,
        "harness_version": __version__,
    }
    write_json(record_path, record)
    sheet_path = layout.human_sheet_path(painting.slug, width, person)
    write_text(sheet_path, page)
    return sheet_path


def record_command(layout: RunLayout, painting_slug: str, width: int, person: str, sheet_id: str | None = None) -> str:
    """The command line that records a ranking for this sheet. The ranking ("C,A,F,B") is typed after it.

    It names the sheet, so a ranking copied from a sheet that has since been replaced is refused.
    """
    words = ["paintbench", "human-sheet", "--run", str(layout.root.resolve()), "--painting", painting_slug]
    words += ["--width", str(width), "--person", person]
    if sheet_id:
        words += ["--sheet", sheet_id]
    words.append("--record")
    return " ".join(shlex.quote(word) for word in words)


def _data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _picture_uri(picture_txt: Path) -> str:
    """Redrawn from the typed rows: the image holds its cells and nothing about where it came from."""
    buffer = io.BytesIO()
    render_picture(load_picture(picture_txt)).save(buffer, format="PNG")
    return _data_uri(buffer.getvalue(), "image/png")


# --------------------------------------------------------------------------
# A sheet and the attempts it was made from
# --------------------------------------------------------------------------


def _out_of_date(layout: RunLayout, record: dict[str, Any], short_too: bool = True) -> str:
    """How the attempts on disk differ from those the sheet was made from, or "" if they do not.

    With `short_too` false only what makes a sheet stale is looked for. The
    text holds counts and no names, because its reader may be the person ranking.
    """
    shown = record.get("pictures") or {}
    on_sheet = set(record["labels"].values())
    now = {a.id: a for a in layout.complete_attempts(record["painting"], record["width"])}
    counts = {
        "pictures typed again since it was made": sum(
            1 for i in on_sheet & now.keys() if i in shown and now[i].picture_sha256() != shown[i]
        ),
        "pictures that are no longer complete attempts": len(on_sheet - now.keys()),
    }
    if short_too:
        counts["complete attempts that are not on it"] = len(now.keys() - on_sheet)
    return "; ".join(f"{what}: {n}" for what, n in counts.items() if n)


def check_sheet(layout: RunLayout, painting_slug: str, width: int, person: str) -> None:
    """Raise unless the person's sheet is there and still shows the attempts on disk.

    FileNotFoundError if it was never made. ValueError if it is stale or short
    (see the module docstring), with a message that says to make a fresh one.
    """
    record_path = _record_path(layout, painting_slug, width, person)
    sheet_path = layout.human_sheet_path(painting_slug, int(width), person)
    if not (record_path.is_file() and sheet_path.is_file()):
        raise FileNotFoundError(f"no sheet at {sheet_path}; make the sheet first")
    changed = _out_of_date(layout, read_json(record_path))
    if changed:
        raise ValueError(
            f"the blind sheet for {person} ({painting_slug}, width {width}) is out of date ({changed}). "
            "Make a fresh one with --new (force=True). It draws new letters and discards any ranking recorded on this one; "
            "a sheet made under another person name leaves that ranking in place."
        )


# --------------------------------------------------------------------------
# Recording a ranking
# --------------------------------------------------------------------------


def parse_ranking(text: str) -> list[str]:
    """Letters from text as the sheet shows it ("C, A, F, B") or as typed on a command line ("c,a,f,b")."""
    return [part.upper() for part in re.split(r"[,\s]+", text) if part]


def _checked_ranking(given: Any, on_sheet: list[str]) -> list[str]:
    if isinstance(given, str) or not isinstance(given, (list, tuple)) or not all(isinstance(g, str) for g in given):
        raise RankingError("the ranking must be a list of picture letters, best first")
    ranking = [g.strip().upper() for g in given]
    counts = collections.Counter(ranking)  # counted once: the list comes from a request and may be long
    problems = []
    missing = [label for label in on_sheet if label not in counts]
    if missing:
        problems.append("missing " + ", ".join(missing))
    repeated = [label for label in on_sheet if counts[label] > 1]
    if repeated:
        problems.append("repeated " + ", ".join(repeated))
    unknown = sorted(label for label in counts if label not in on_sheet)
    if unknown:
        problems.append("not on this sheet " + ", ".join(repr(label[:20]) for label in unknown[:10]))
    if problems:
        raise RankingError(
            f"the ranking must list every picture on the sheet exactly once: {'; '.join(problems)}. "
            f"The pictures on this sheet are {', '.join(on_sheet)}."
        )
    return ranking


def record_ranking(
    layout: RunLayout,
    painting_slug: str,
    width: int,
    person: str,
    ranking_labels: list[str],
    notes: str = "",
    sheet_id: str | None = None,
) -> dict[str, Any]:
    """Store a person's ranking (letters, best first) in the sheet's record. Returns the record.

    Raises RankingError unless every letter on the sheet appears exactly once,
    the sheet is not stale, and `sheet_id` is this sheet's. The page sends its
    id; a ranking typed by hand has none and is taken to be of this sheet. A
    ranking this one replaces is kept under "earlier".
    The record holds the key, so it must not be shown to the person.
    """
    path = _record_path(layout, painting_slug, width, person)
    if not path.is_file():
        raise FileNotFoundError(f"no sheet record at {path}; make the sheet first")
    record = read_json(path)
    if sheet_id is not None and record.get("sheet") and sheet_id != record["sheet"]:
        raise RankingError(
            "this ranking was made on an earlier sheet, and the sheet has since been made again with new letters. "
            "Reload the page and rank the pictures it shows now."
        )
    stale = _out_of_date(layout, record, short_too=False)
    if stale:
        raise RankingError(
            f"this sheet is out of date ({stale}), so the ranking was not recorded. "
            "Ask for a fresh sheet (paintbench human-sheet --new) and rank that one."
        )
    ranking = _checked_ranking(ranking_labels, sorted(record["labels"]))
    # Half of a surrogate pair cannot be written to the record. It becomes "?", so the ranking is not lost over it.
    notes = str(notes).encode("utf-8", "replace").decode("utf-8")
    if record.get("ranking") and (record["ranking"], record.get("notes")) != (ranking, notes):
        replaced = {key: record.get(key) for key in ("ranking", "ranking_attempts", "notes", "ranked_at")}
        record.setdefault("earlier", []).append(replaced)
    record["ranking"] = ranking
    record["ranking_attempts"] = [record["labels"][label] for label in ranking]
    record["notes"] = notes
    record["ranked_at"] = _now()
    write_json(path, record)
    return record


# --------------------------------------------------------------------------
# Serving the sheet
# --------------------------------------------------------------------------


def _binds_everything(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
        mapped = getattr(address, "ipv4_mapped", None)  # ::ffff:0.0.0.0 is every IPv4 interface
        return address.is_unspecified or (mapped is not None and mapped.is_unspecified)
    except ValueError:
        pass
    try:
        return socket.inet_aton(host) == bytes(4)  # older spellings of 0.0.0.0, such as "0"
    except OSError:
        return False


def check_host(host: str) -> str:
    """Refuse an address that means every interface."""
    if not isinstance(host, str) or not host.strip() or _binds_everything(host.strip()):
        raise ValueError(
            f"refusing to serve the page on {host!r}: that address listens on every network interface. "
            "Pages are served on one named address only: give one address of this machine, for example 127.0.0.1 or its address on a private network."
        )
    return host.strip()


class _SheetServer(ThreadingHTTPServer):
    def __init__(self, layout: RunLayout, painting_slug: str, width: int, person: str, host: str, port: int):
        self.layout, self.painting_slug, self.width, self.person = layout, painting_slug, int(width), person
        self.sheet_path = layout.human_sheet_path(painting_slug, int(width), person)
        self.record_path = _record_path(layout, painting_slug, width, person)
        self.saving = threading.Lock()
        host = check_host(host)
        self.address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
        super().__init__((host, int(port)), _SheetHandler)

    def server_bind(self) -> None:
        # Not HTTPServer.server_bind: it looks the host's name up in DNS, and nothing here needs that name.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]
        check_host(self.server_name)  # a host name can resolve to every interface too


class _SheetHandler(BaseHTTPRequestHandler):
    """GET / is the sheet, POST /rank records a ranking, everything else is 404."""

    server: _SheetServer
    server_version = "paintbench"
    sys_version = ""
    timeout = 30  # seconds a silent connection may hold a thread

    def do_GET(self) -> None:
        self._answer(self._get)

    def do_POST(self) -> None:
        self._answer(self._post)

    def _answer(self, respond: Callable[[], None]) -> None:
        """Run a handler so that an error inside it is answered with a 500, not with a closed connection."""
        try:
            respond()
        except (ConnectionError, TimeoutError):
            raise  # the client has gone or stopped sending: there is no one to answer
        except Exception as e:
            self.log_error("internal error on %s %s: %s: %s", self.command, self.path, type(e).__name__, e)
            # The reply's text is fixed: the exception's own can hold paths, and paths name attempts.
            self._send_json(500, {"ok": False, "error": "the server could not read or write this sheet's files"})

    def _get(self) -> None:
        if urlsplit(self.path).path != "/":
            return self._not_found()
        server = self.server
        record = read_json(server.record_path)
        stale = _out_of_date(server.layout, record, short_too=False)
        if stale:
            text = f"This sheet is out of date ({stale}). Ask for a fresh one (paintbench human-sheet --new).\n"
            return self._send(409, "text/plain; charset=utf-8", text.encode("utf-8"))
        page = server.sheet_path.read_bytes()
        head, slot, tail = page.rpartition(_SAVED_SLOT)
        if slot and record.get("ranking"):
            # The person's own ranking and notes, and never the key. Without them the page would show
            # the sheet's first order, and one press of Save would record that order as the person's.
            saved = {"ranking": record["ranking"], "notes": str(record.get("notes") or ""), "ranked_at": record.get("ranked_at")}
            page = head + b"const SAVED = " + _in_script(saved).encode("ascii") + b";" + tail
        self._send(200, "text/html; charset=utf-8", page)

    def _post(self) -> None:
        length = self.headers.get("Content-Length", "0")
        if not (length.isascii() and length.isdigit()):  # isdigit alone lets "²" through to int(), which refuses it
            return self._send_json(400, {"ok": False, "error": "the Content-Length header is not a number"})
        # Twelve digits are past any length this server reads, and int() refuses a number thousands of digits long.
        if len(length) > 12 or int(length) > MAX_BODY_BYTES:
            return self._refuse_body(int(length[:12]))
        body = self.rfile.read(int(length))
        if urlsplit(self.path).path != "/rank":
            return self._not_found()
        # A page on another site cannot send this content type without asking first, and the asking gets a 404.
        if self.headers.get_content_type() != "application/json":
            return self._send_json(415, {"ok": False, "error": "the request must be application/json"})
        try:
            data = json.loads(body)
            ranking, notes, sheet_id = data["ranking"], data.get("notes", ""), data.get("sheet")
            if not isinstance(notes, str) or not isinstance(sheet_id, (str, type(None))):
                raise TypeError("notes or sheet")
        except (ValueError, KeyError, TypeError, AttributeError, RecursionError):
            return self._send_json(400, {"ok": False, "error": 'send JSON such as {"ranking": ["C", "A"], "notes": ""}'})
        server = self.server
        try:
            with server.saving:
                record_ranking(server.layout, server.painting_slug, server.width, server.person, ranking, notes, sheet_id)
        except RankingError as e:
            return self._send_json(400, {"ok": False, "error": str(e)})
        self._send_json(200, {"ok": True})  # never the record: it holds the key

    def _not_found(self) -> None:
        self._send(404, "text/plain; charset=utf-8", b"not found\n")

    do_HEAD = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _not_found

    def _refuse_body(self, length: int) -> None:
        self._send_json(413, {"ok": False, "error": f"the request is larger than {MAX_BODY_BYTES} bytes"})
        # Take in, briefly and without keeping it, what the client is still sending: closing the
        # connection on unread data can lose the reply just sent.
        self.connection.settimeout(2)
        remaining = min(length, _DRAIN_BYTES)
        try:
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    return
                remaining -= len(chunk)
        except OSError:
            pass

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        self._send(status, "application/json", json.dumps(data).encode("utf-8"))

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")  # a sheet made again must not be shown from a cache
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


def make_server(layout: RunLayout, painting_slug: str, width: int, person: str, host: str, port: int) -> ThreadingHTTPServer:
    """The server for one sheet, bound to `host` and not yet serving. Port 0 picks a free port.

    Call serve_forever() on it, then shutdown() and server_close(). The port
    in use is server.server_address[1]. A sheet that is stale or short is not
    served: see check_sheet.
    """
    host = check_host(host)
    check_sheet(layout, painting_slug, width, person)
    return _SheetServer(layout, painting_slug, width, person, host, port)


def serve_sheet(layout: RunLayout, painting_slug: str, width: int, person: str, host: str, port: int) -> None:
    """Serve one sheet on `host` until interrupted. `host` has no default: see check_host.

    Raises what make_server raises, except that an address that cannot be
    listened on is a ValueError that says so in one line.
    """
    try:
        server = make_server(layout, painting_slug, width, person, host, port)
    except FileNotFoundError:
        raise
    except (OSError, OverflowError) as e:  # OverflowError: a port number that is not one
        # Most likely the port is in use. Raised as a ValueError because the command line prints
        # that in one line, where it would show an OSError as a traceback.
        raise ValueError(
            f"cannot listen on {host} port {port}: {getattr(e, 'strerror', None) or e}. "
            "If another server holds the port, stop it or give another with --port."
        ) from None
    address, bound_port = server.server_address[:2]
    shown = f"[{address}]" if ":" in address else address
    print(f"Blind sheet for {person}: http://{shown}:{bound_port}/  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

# Neutral greys only, so the surround does not tint how the pictures' colours read.
_CSS = r"""
:root {
  color-scheme: light dark;
  --bg: #f3f3f2; --surface: #ffffff; --text: #1c1c1b; --muted: #61615d; --line: #d6d6d2;
  --focus: #1f5fbf; --good: #1b6a3b; --bad: #a12327;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #151515; --surface: #212121; --text: #ebebe9; --muted: #a3a39e; --line: #3a3a3a;
    --focus: #8fb8ff; --good: #7fd29a; --bad: #ff9c9c;
  }
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font: 16px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.page { max-width: 120rem; margin: 0 auto; padding: 1.75rem clamp(1rem, 3vw, 2.5rem) 0; }
h1 { font-size: 1.5rem; font-weight: 600; line-height: 1.25; margin: 0; }
h2 { font-size: 1.05rem; font-weight: 600; margin: 0; }
p { margin: 0; }
.muted, figcaption { color: var(--muted); }
header p { max-width: 72ch; margin-top: 0.6rem; }
header .muted { margin-top: 0.2rem; }
figure { margin: 0; }
figcaption { font-size: 0.9rem; margin-bottom: 0.4rem; }
img { display: block; height: auto; margin: 0 auto; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }

/* The side panel stays in view. Nothing sits below the two columns: it would push the panel, and the
   reference with it, off the top of the screen at the end of the page. */
.layout { display: grid; grid-template-columns: minmax(18rem, 34%) minmax(0, 1fr); column-gap: 2rem; align-items: start; margin-top: 1.75rem; }
.side { grid-row: 1 / span 2; position: sticky; top: 1rem; max-height: calc(100vh - 2rem); overflow-y: auto; padding: 2px; }
.work { padding-bottom: 2.5rem; }
.reference img { width: min(100%, calc(48vh * var(--ref-ar))); }
.order { margin-top: 1rem; }
.order output { display: block; font-size: 1.15rem; font-weight: 600; letter-spacing: 0.02em; user-select: all; }
.finish { margin-top: 1rem; }
label { display: block; margin-bottom: 0.3rem; }
textarea {
  display: block; width: 100%; min-height: 4.5rem; resize: vertical; padding: 0.5rem 0.6rem; font: inherit;
  color: var(--text); background: var(--surface); border: 1px solid var(--line); border-radius: 0.4rem;
}
button {
  font: inherit; color: var(--text); background: var(--surface); border: 1px solid var(--line);
  border-radius: 0.4rem; padding: 0.45rem 0.9rem; cursor: pointer;
}
button:hover:not(:disabled) { border-color: var(--muted); }
button:disabled { opacity: 0.35; cursor: default; }
#save { margin-top: 0.9rem; padding: 0.6rem 1.3rem; font-weight: 600; color: var(--bg); background: var(--text); border-color: var(--text); }
#status { margin-top: 0.6rem; min-height: 1.5rem; }
#status.good { color: var(--good); }
#status.bad { color: var(--bad); }
#fallback { margin-top: 0.4rem; }
#command {
  margin: 0.5rem 0; padding: 0.6rem 0.7rem; white-space: pre-wrap; overflow-wrap: anywhere; user-select: all;
  font: 0.85rem/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: var(--surface); border: 1px solid var(--line); border-radius: 0.4rem;
}

.cards {
  list-style: none; margin: 0; padding: 0; counter-reset: place; display: grid; gap: 1.5rem 1.25rem;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, var(--card-min)), 1fr));
}
.card {
  counter-increment: place; padding: 0.75rem; background: var(--surface); border: 1px solid var(--line);
  border-radius: 0.5rem; cursor: grab; user-select: none; -webkit-user-select: none; scroll-margin: 1rem;
}
.card.dragging { opacity: 0.35; }
.card-head { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.6rem; }
.card-head h2 { margin-right: auto; white-space: nowrap; }
.place { color: var(--muted); font-variant-numeric: tabular-nums; }
.place::before { content: counter(place); }
.end { font-size: 0.9rem; margin: 0 0 0.5rem; }
.cards + .end { margin: 0.5rem 0 0; }
.card-head button { width: 2.5rem; height: 2.5rem; padding: 0; line-height: 1; }
.card img, #zoom-picture { image-rendering: crisp-edges; image-rendering: pixelated; }
.card img { width: min(100%, calc(62vh * var(--ar))); cursor: zoom-in; }
footer { grid-column: 2; padding-bottom: 1.5rem; font-size: 0.85rem; }

dialog {
  width: min(96vw, 112rem); max-width: none; max-height: 96vh; padding: 0; border: 0; border-radius: 0.6rem;
  background: var(--bg); color: var(--text);
}
dialog::backdrop { background: rgb(0 0 0 / 0.7); }
.zoom-box { padding: 1rem 1.25rem 1.25rem; }
.zoom-head { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem; }
.zoom-head p { margin-right: auto; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; align-items: start; }
.pair figcaption { text-align: center; }
#zoom-reference { width: min(100%, calc((96vh - 8.5rem) * var(--ref-ar))); }
#zoom-picture { width: min(100%, calc((96vh - 8.5rem) * var(--ar))); }

@media (max-width: 56rem) {
  .layout { display: flex; flex-direction: column; gap: 1.25rem; padding-bottom: 2rem; }
  .side { display: contents; }
  .reference { order: 1; position: sticky; top: 0; z-index: 1; padding: 0.6rem 0; background: var(--bg); border-bottom: 1px solid var(--line); }
  .reference img { width: min(100%, calc(30vh * var(--ref-ar))); }
  .order { margin-top: 0.5rem; }
  .order output { font-size: 1rem; }
  .work { order: 2; padding-bottom: 0; }
  .card { scroll-margin-top: 48vh; }
  .finish { order: 3; margin-top: 0; }
  footer { order: 4; }
}
@media (max-aspect-ratio: 1/1) {
  .pair { grid-template-columns: 1fr; }
  #zoom-reference { width: min(100%, calc((48vh - 6rem) * var(--ref-ar))); }
  #zoom-picture { width: min(100%, calc((48vh - 6rem) * var(--ar))); }
}
"""

_JS = r"""
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const COMMAND = __COMMAND__;
  const SHEET = __SHEET__;
  const SAVED = null;  // the server puts the recorded ranking here: {ranking, notes, ranked_at}
  const asFile = location.protocol === "file:";
  const list = $("cards"), notes = $("notes"), status = $("status"), zoom = $("zoom");
  const cards = () => Array.from(list.children);
  const labels = () => cards().map((card) => card.dataset.label);
  const snapshot = () => labels().join(", ") + "\n" + notes.value;
  const canChange = " You can still change the order and save again.";
  let savedMessage = "";  // set while there is a saved ranking to compare the page with

  if (SAVED) {
    // Moved to the end one by one in the saved order, the cards finish in that order.
    SAVED.ranking.forEach((label) => {
      const card = cards().find((c) => c.dataset.label === label);
      if (card) list.appendChild(card);
    });
    notes.value = SAVED.notes;  // the box has autocomplete off, so a reload does not put back text typed before the save
    const time = new Date(SAVED.ranked_at || NaN);
    const day = isNaN(time) ? "earlier" : "on " + time.toLocaleString([], {day: "numeric", month: "short", hour: "2-digit", minute: "2-digit"});
    savedMessage = "Saved " + day + ": " + labels().join(", ") + "." + canChange;
  }
  let kept = snapshot();  // what leaving the page would not lose: the sheet as loaded, or as last saved

  function say(text, kind) {
    status.textContent = text;
    status.className = kind || "";
  }

  function shellQuote(text) {
    return "'" + text.replace(/'/g, "'\\''") + "'";
  }

  function refresh() {
    const order = labels();
    $("order").textContent = order.join(", ");
    cards().forEach((card, i) => {
      card.querySelector(".up").disabled = i === 0;
      card.querySelector(".down").disabled = i === order.length - 1;
    });
    const typed = notes.value.trim();
    $("command").textContent = COMMAND + " " + order.join(",") + (typed ? " --notes=" + shellQuote(typed) : "");
    $("copied").textContent = "";
    if (savedMessage) {
      if (snapshot() === kept) say(savedMessage, "good");
      else say("Changed since it was saved. Save again to keep the change.");
    }
  }

  // ---- moving a picture: its buttons, or drag and drop ----

  function move(card, step) {
    const other = step < 0 ? card.previousElementSibling : card.nextElementSibling;
    if (!other) return;
    list.insertBefore(card, step < 0 ? other : other.nextElementSibling);
    refresh();
  }

  list.addEventListener("click", (event) => {
    const card = event.target.closest(".card");
    const button = event.target.closest("button");
    if (button) {
      const up = button.classList.contains("up");
      move(card, up ? -1 : 1);
      // Moving a node can drop the focus, and at either end the button just pressed is now disabled.
      (button.disabled ? card.querySelector(up ? ".down" : ".up") : button).focus();
      card.scrollIntoView({block: "nearest"});
    } else if (event.target.tagName === "IMG") {
      showZoom(card);
    }
  });

  let dragged = null;
  list.addEventListener("dragstart", (event) => {
    dragged = event.target.closest ? event.target.closest(".card") : null;
    if (!dragged) return;
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("application/x-picture", dragged.dataset.label);  // Firefox starts no drag without data
    // Dim it on the next frame, after the browser has taken the image that follows the pointer.
    requestAnimationFrame(() => dragged && dragged.classList.add("dragging"));
  });
  list.addEventListener("dragover", (event) => {
    if (!dragged) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    const over = event.target.closest(".card");
    if (!over || over === dragged) return;
    // Take the place of the card under the pointer. The cards are one size, so it then sits under the pointer itself.
    const overIsEarlier = dragged.compareDocumentPosition(over) & Node.DOCUMENT_POSITION_PRECEDING;
    list.insertBefore(dragged, overIsEarlier ? over : over.nextElementSibling);
    refresh();
  });
  list.addEventListener("drop", (event) => event.preventDefault());
  list.addEventListener("dragend", () => {
    if (dragged) dragged.classList.remove("dragging");
    dragged = null;
  });

  // ---- one picture large, beside the reference ----

  let zoomed = null;
  function showZoom(card) {
    const all = cards(), name = "Picture " + card.dataset.label;
    zoomed = card;
    $("zoom-reference").src = $("reference").src;
    $("zoom-picture").src = card.querySelector("img").src;
    $("zoom-picture").alt = name;
    $("zoom-name").textContent = name;
    $("zoom-place").textContent = "Place " + (all.indexOf(card) + 1) + " of " + all.length;
    if (!zoom.open) zoom.showModal();
  }
  function stepZoom(step) {
    const all = cards();
    showZoom(all[(all.indexOf(zoomed) + step + all.length) % all.length]);
  }
  $("previous").addEventListener("click", () => stepZoom(-1));
  $("next").addEventListener("click", () => stepZoom(1));
  $("close").addEventListener("click", () => zoom.close());
  zoom.addEventListener("click", (event) => {
    if (event.target === zoom) zoom.close();  // a click outside the box lands on the dialog itself
  });
  document.addEventListener("keydown", (event) => {
    if (zoom.open && event.key === "ArrowLeft") stepZoom(-1);
    if (zoom.open && event.key === "ArrowRight") stepZoom(1);
  });
  list.addEventListener("keydown", (event) => {
    if (event.target.tagName === "IMG" && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      showZoom(event.target.closest(".card"));
    }
  });

  // ---- saving ----

  function offerCommand(why) {
    $("why").textContent = why + " To record it, run this command where the run is kept:";
    $("fallback").hidden = false;
    kept = snapshot();
    savedMessage = "";
  }

  $("save").addEventListener("click", async () => {
    const ranking = labels(), typed = notes.value;
    if (asFile) {
      say("");
      return offerCommand("This page was opened as a file, so it cannot save the ranking itself.");
    }
    say("Saving…");
    let response, reply;
    try {
      response = await fetch("rank", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ranking: ranking, notes: typed, sheet: SHEET}),
      });
      reply = await response.json();
    } catch (error) {
      say("Not saved.", "bad");
      return offerCommand("The ranking could not be saved from this page.");
    }
    if (response.ok && reply.ok === true) {
      const time = new Date().toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
      kept = ranking.join(", ") + "\n" + typed;
      savedMessage = "Saved at " + time + ": " + ranking.join(", ") + "." + canChange;
      $("fallback").hidden = true;
      refresh();
    } else {
      say("Not saved: " + (typeof reply.error === "string" ? reply.error : "the server answered " + response.status), "bad");
    }
  });

  $("copy").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("command").textContent);
      $("copied").textContent = "Copied.";
    } catch (error) {
      getSelection().selectAllChildren($("command"));
      $("copied").textContent = "Selected. Copy it with Ctrl+C or ⌘C.";
    }
  });

  notes.addEventListener("input", refresh);
  window.addEventListener("beforeunload", (event) => {
    if (snapshot() !== kept) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  // What the order on the page is, before anything is moved: no one should take the sheet's first order for a saved one.
  $("as-file").hidden = !asFile;
  if (!asFile && !SAVED) say("Nothing is saved yet.");
  refresh();
})();
"""

# The browser enforces what the page promises: it can load nothing from anywhere, and can only post back to its own server.
_CSP = (
    "default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'"
)


def _card_min_rem(count: int, width: int, height: int) -> float:
    """The narrowest a card may get: large for a few pictures, smaller so that many still fit on a laptop screen."""
    long_side = 18 if count <= 6 else 16 if count <= 12 else 13.5
    return round(long_side * min(1.0, width / height), 2)


def _in_script(value: Any) -> str:
    """A value as a JavaScript literal, in ASCII, that cannot close the script element it sits in."""
    return json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _sheet_html(
    painting: Painting, width: int, height: int, person: str, pictures: list[tuple[str, str]], command: str, sheet_id: str
) -> str:
    """The page. `pictures` is (letter, data URI) in the order shown; nothing else about an attempt comes in."""
    esc = html.escape
    if not painting.reference_path.is_file():
        raise FileNotFoundError(
            f"no reference image at {painting.reference_path}; a sheet always shows the real painting. "
            f"`python scripts/fetch_originals.py {painting.slug}` fetches and rebuilds it"
        )
    with Image.open(painting.reference_path) as ref:
        ref_width, ref_height = ref.size
        ref_uri = _data_uri(painting.reference_path.read_bytes(), ref.get_format_mimetype())
    pic_width, pic_height = (side * cell_size(width, height) for side in (width, height))
    cards = "\n".join(
        f'<li class="card" draggable="true" data-label="{label}">\n'
        f'<div class="card-head"><span class="place"></span><h2>Picture {label}</h2>'
        f'<button type="button" class="up" aria-label="Move picture {label} up, towards best">&uarr;</button>'
        f'<button type="button" class="down" aria-label="Move picture {label} down, towards worst">&darr;</button></div>\n'
        f'<img src="{uri}" width="{pic_width}" height="{pic_height}" alt="Picture {label}" draggable="false" '
        f'tabindex="0" role="button" aria-label="Enlarge picture {label} beside the reference">\n</li>'
        for label, uri in pictures
    )
    credit = str((painting.meta.get("reference") or {}).get("credit") or "").strip()
    footer = f'<footer><p class="muted">Reference image: {esc(credit)}</p></footer>\n' if credit else ""
    variables = (
        f":root {{ --ar: {width / height:.4f}; --ref-ar: {ref_width / ref_height:.4f}; "
        f"--card-min: {_card_min_rem(len(pictures), width, height)}rem; }}"
    )
    return f"""<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{_CSP}">
<link rel="icon" href="data:,">
<title>Blind sheet: {esc(painting.display_name)}, {width} x {height}</title>
<style>
{variables}{_CSS}</style>
</head>
<body>
<div class="page">
<header>
<h1>{esc(painting.display_name)}</h1>
<p class="muted">{esc(painting.artist)}, {esc(painting.date)}. Blind sheet for {esc(person)}.</p>
<p>These {len(pictures)} pictures are attempts to depict this painting from memory on a small grid,
{width} cells wide and {height} cells tall. The letters are random, and you are not told who made which.</p>
<p>Put them in order from best to worst as a depiction of this painting, the way someone who knows the
painting well would judge them. Take into account what is in the picture, whether each thing is in the
right place, whether its shape is right, and the colours. No ties.</p>
<p>Drag a picture to move it, or use its &uarr; and &darr; buttons. Click a picture to see it large
beside the reference.</p>
<p id="as-file" hidden>This page was opened as a file, so it cannot show a ranking that was saved earlier. The
pictures are in the order the sheet was made with, whether or not a ranking has been recorded.</p>
</header>
<div class="layout">
<aside class="side">
<div class="reference">
<figure>
<figcaption>Reference: the real painting</figcaption>
<img id="reference" src="{ref_uri}" width="{ref_width}" height="{ref_height}" alt="The real painting">
</figure>
<p class="order"><span class="muted">Your order, best first</span> <output id="order">{", ".join(label for label, _ in pictures)}</output></p>
</div>
<div class="finish">
<label for="notes">Notes <span class="muted">(optional)</span></label>
<textarea id="notes" rows="3" autocomplete="off"></textarea>
<button type="button" id="save">Save ranking</button>
<p id="status" role="status"></p>
<div id="fallback" hidden>
<p id="why"></p>
<pre id="command"></pre>
<button type="button" id="copy">Copy command</button> <span id="copied" class="muted" role="status"></span>
</div>
</div>
</aside>
<div class="work">
<p class="end muted">Best first</p>
<ol id="cards" class="cards" aria-label="The pictures, best first">
{cards}
</ol>
<p class="end muted">Worst last</p>
</div>
{footer}</div>
</div>
<dialog id="zoom" aria-labelledby="zoom-name">
<div class="zoom-box">
<div class="zoom-head">
<p id="zoom-place" class="muted"></p>
<button type="button" id="previous">Previous</button>
<button type="button" id="next">Next</button>
<button type="button" id="close" autofocus>Close</button>
</div>
<div class="pair">
<figure><figcaption>Reference</figcaption><img id="zoom-reference" width="{ref_width}" height="{ref_height}" alt="The real painting"></figure>
<figure><figcaption id="zoom-name"></figcaption><img id="zoom-picture" width="{pic_width}" height="{pic_height}" alt=""></figure>
</div>
</div>
</dialog>
<script>{_JS.replace("__SHEET__", _in_script(sheet_id)).replace("__COMMAND__", _in_script(command))}</script>
</body>
</html>
"""
