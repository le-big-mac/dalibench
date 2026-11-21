"""A fake adapter that plays back a scripted model. No network, ever.

It exists so the whole pipeline can run offline: in the tests, and for anyone
who wants to see the files a run produces before spending anything.

Two ways to use it.

1. Pass a script to the constructor (tests):

       FakeAdapter(script=[reply1, reply2, TransientError("boom"), reply3])
       FakeAdapter(script=lambda messages, tools, settings: Reply(...))

   A list is consumed one item per call; an exception in the list is raised.

2. Name a behaviour in the model's config entry (provider: fake):

       options: {behaviour: typist}     a contestant that types a picture
       options: {behaviour: judge}      a judge that ranks by a fixed order

   typist options
     pictures     {"<painting>/<width>" or "<width>": path to a picture.txt}. The file for the
                  size in the task message is typed; without one a synthetic picture is typed.
     painting     the slug used to look up `pictures` (the harness does not tell a model the slug)
     faults       [{"row": 3, "kind": "long" | "short" | "unknown"}] spoil the first sending of a row
     stop_early   true: reply with text after the first strip, to draw a nudge
     refuse       true: decline in text.  "provider": signal a provider refusal
     never_call   true: reply with text and never call the tool
     fail_first   N: raise a retryable error on the first N calls made through this adapter
     fail_always  true: every call raises a non-retryable error
   judge options
     order        contestant model names, best first. Pictures are ranked by where their model
                  comes in this list (the fake reads the model name from the image path).
     bad_replies  N: the first N replies of each judging conversation leave out a label

Every request passed to `complete()` is kept in `.requests` for inspection.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Callable

from ..canvas import PALETTE, VALUE, Picture, load_picture
from ..prompts import TOOL_NAME
from ..types import (
    STOP_END,
    STOP_REFUSAL,
    STOP_TOOL_CALLS,
    Image,
    Message,
    ModelSettings,
    Reply,
    Text,
    ToolCall,
    ToolSpec,
    Usage,
)
from .base import Adapter, FatalError, TransientError

_GRID = re.compile(r"grid (\d+) cells wide and (\d+) cells tall")
_STRIP = re.compile(r"Send at most (\d+) rows per call")
_REJECTED = re.compile(r"^row (\d+) rejected", re.MULTILINE)
_LABEL = re.compile(r"^Picture (\S+)$")


def synthetic_picture(width: int, height: int, encoding: str, seed: str = "") -> Picture:
    """A deterministic striped picture, different for each seed."""
    shift = sum(ord(c) for c in seed) % 5
    codes = "abcde"
    colours = [(30, 60, 110), (90, 130, 170), (230, 220, 190), (250, 248, 240), (150, 140, 120)]
    cells = [[(x // 3 + y // 2 + shift) % 5 for x in range(width)] for y in range(height)]
    if encoding == VALUE:
        rows: list[Any] = [tuple("%02X%02X%02X" % colours[c] for c in row) for row in cells]
        return Picture(width, height, VALUE, rows)
    return Picture(width, height, PALETTE, ["".join(codes[c] for c in row) for row in cells], dict(zip(codes, colours)))


class FakeAdapter(Adapter):
    provider = "fake"

    def __init__(self, client: Any = None, script: Any = None):
        super().__init__(client)
        self._script = script
        self._lock = threading.Lock()
        self._calls = 0
        self.requests: list[dict[str, Any]] = []

    # ---- the Adapter interface ----

    def build_request(self, messages: list[Message], tools: list[ToolSpec], settings: ModelSettings) -> dict[str, Any]:
        return {
            "model": settings.model,
            "messages": [m.to_dict() for m in messages],
            "tools": [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools],
            "reasoning": dict(settings.reasoning),
            "max_output_tokens": settings.max_output_tokens,
        }

    def parse_response(self, response: Any) -> Reply:
        if isinstance(response, Reply):
            return response
        raise TypeError("the fake adapter's script must produce Reply objects")

    def complete(self, messages: list[Message], tools: list[ToolSpec], settings: ModelSettings) -> Reply:
        with self._lock:
            self._calls += 1
            call_number = self._calls
            self.requests.append(
                {"messages": list(messages), "tools": list(tools), "settings": settings, "request": self.build_request(messages, tools, settings)}
            )
            item = self._next_scripted(messages, tools, settings)
        if item is None:
            item = self._behave(messages, tools, settings, call_number)
        if isinstance(item, BaseException):
            raise item
        reply = self.parse_response(item)
        reply.provider = self.provider
        if reply.provider_payload is None:
            reply.provider_payload = {"fake": True}
        return reply

    # ---- scripts ----

    def _next_scripted(self, messages: list[Message], tools: list[ToolSpec], settings: ModelSettings) -> Any:
        if self._script is None:
            return None
        if callable(self._script):
            return self._script(messages, tools, settings)
        if not self._script:
            raise FatalError("the fake adapter's script ran out of replies")
        return self._script.pop(0)

    # ---- behaviours ----

    def _behave(self, messages: list[Message], tools: list[ToolSpec], settings: ModelSettings, call_number: int) -> Any:
        options = settings.options or {}
        if options.get("fail_always"):
            return FatalError("scripted failure")
        if call_number <= int(options.get("fail_first", 0)):
            return TransientError("scripted transient failure")
        behaviour = options.get("behaviour", "typist")
        if behaviour == "typist":
            return self._typist(messages, settings)
        if behaviour == "judge":
            return self._judge(messages, settings)
        return FatalError(f"fake adapter: unknown behaviour {behaviour!r}")

    @staticmethod
    def _usage(messages: list[Message], text: str) -> Usage:
        chars_in = sum(len(m.text) for m in messages)
        return Usage(input_tokens=chars_in // 4 + 1, output_tokens=len(text) // 4 + 1, reasoning_tokens=0)

    def _typist(self, messages: list[Message], settings: ModelSettings) -> Reply:
        options = settings.options or {}
        task = next(m.text for m in messages if m.role == "user")
        width, height = (int(v) for v in _GRID.search(task).groups())
        strip = int(_STRIP.search(task).group(1))
        encoding = VALUE if "six hex digits" in task else PALETTE
        turns = [m for m in messages if m.role == "assistant"]

        if options.get("refuse") == "provider":
            return Reply(text="", stop_reason=STOP_REFUSAL, raw_stop_reason="refusal", refusal_text="Declined by the provider.")
        if options.get("refuse"):
            text = "I'm sorry, but I can't help with reproducing this painting."
            return Reply(text=text, stop_reason=STOP_END, raw_stop_reason="end", usage=self._usage(messages, text))
        if options.get("never_call"):
            text = "The painting shows a large wave on the left and a small mountain on the right."
            return Reply(text=text, stop_reason=STOP_END, raw_stop_reason="end", usage=self._usage(messages, text))

        picture = self._picture_for(options, width, height, encoding, settings.name)
        faults = {int(f["row"]): f.get("kind", "long") for f in options.get("faults", [])}
        rows_as_text = [row if isinstance(row, str) else " ".join(row) for row in picture.rows]

        # Rows the tool rejected last time are sent again, correctly this time.
        last_results = []
        for m in reversed(messages):
            if m.role != "tool":
                break
            last_results.append(m.text)
        to_fix = sorted({int(n) for text in last_results for n in _REJECTED.findall(text)})
        sent = {
            call.arguments["start_row"] + i
            for m in turns
            for call in m.tool_calls
            for i in range(len(call.arguments.get("rows", [])))
        }

        def call_for(start: int, count: int, with_palette: bool, spoil: bool) -> ToolCall:
            rows = []
            for n in range(start, start + count):
                row = rows_as_text[n - 1]
                if spoil and n in faults:
                    kind = faults[n]
                    filler = row[0] if encoding == PALETTE else " " + row.split()[0]
                    if kind == "long":
                        row = row + filler
                    elif kind == "short":
                        row = row[:-1] if encoding == PALETTE else " ".join(row.split()[:-1])
                    else:
                        row = ("?" + row[1:]) if encoding == PALETTE else " ".join(["XYZ"] + row.split()[1:])
                elif encoding == PALETTE and n % 2 == 0:
                    row = " ".join(row[i : i + 8] for i in range(0, len(row), 8))  # type in blocks of eight
                rows.append(row)
            arguments: dict[str, Any] = {"start_row": start, "rows": rows}
            if with_palette and encoding == PALETTE:
                arguments = {"palette": [f"{c} {r} {g} {b}" for c, (r, g, b) in picture.palette.items()], **arguments}
            return ToolCall(id=f"call_{len(turns) + 1}_{start}", name=TOOL_NAME, arguments=arguments)

        calls: list[ToolCall] = []
        if to_fix:
            calls = [call_for(n, 1, False, False) for n in to_fix]
        else:
            next_row = next((n for n in range(1, height + 1) if n not in sent), None)
            if next_row is not None:
                nudged = any(m.role == "user" and m.text.startswith("The picture is not finished") for m in messages)
                if options.get("stop_early") and sent and not nudged:
                    text = "That is the top of the picture."
                    return Reply(text=text, stop_reason=STOP_END, raw_stop_reason="end", usage=self._usage(messages, text))
                calls = [call_for(next_row, min(strip, height - next_row + 1), not sent, True)]
        if calls:
            typed = " ".join(str(c.arguments) for c in calls)
            return Reply(tool_calls=calls, stop_reason=STOP_TOOL_CALLS, raw_stop_reason="tool_calls", usage=self._usage(messages, typed))
        text = "I typed the picture row by row as scripted. I was unsure about nothing, being a script."
        return Reply(text=text, stop_reason=STOP_END, raw_stop_reason="end", usage=self._usage(messages, text))

    @staticmethod
    def _picture_for(options: dict[str, Any], width: int, height: int, encoding: str, seed: str) -> Picture:
        pictures = options.get("pictures") or {}
        slug = options.get("painting")
        path = pictures.get(f"{slug}/{width}") if slug else None
        path = path or pictures.get(str(width)) or pictures.get(width)
        if path:
            picture = load_picture(path)
            if (picture.width, picture.height, picture.encoding) == (width, height, encoding):
                return picture
        return synthetic_picture(width, height, encoding, seed)

    def _judge(self, messages: list[Message], settings: ModelSettings) -> Reply:
        options = settings.options or {}
        order = list(options.get("order") or [])
        first = next(m for m in messages if m.role == "user")
        labelled: list[tuple[str, str]] = []  # (label, image path)
        label = None
        for part in first.content:
            if isinstance(part, Text):
                m = _LABEL.match(part.text.strip())
                label = m.group(1) if m else None
            elif isinstance(part, Image) and label is not None:
                labelled.append((label, part.path))
                label = None

        def rank_key(item: tuple[str, str]) -> tuple[int, str]:
            _label, path = item
            pieces = path.replace("\\", "/").split("/")
            position = min((order.index(name) for name in pieces if name in order), default=len(order))
            return (position, path)

        ranked = [lab for lab, _path in sorted(labelled, key=rank_key)]
        asked = sum(1 for m in messages if m.role == "assistant")
        shown = ranked[:-1] if asked < int(options.get("bad_replies", 0)) else ranked
        lines = ["OVERALL: " + ", ".join(shown), "NOTES:"]
        lines += [f"{lab}: scripted note for this picture." for lab in ranked]
        lines.append("CONFIDENCE: scripted, so entirely confident; no pair was hard.")
        text = "\n".join(lines)
        return Reply(text=text, stop_reason=STOP_END, raw_stop_reason="end", usage=self._usage(messages, text))
