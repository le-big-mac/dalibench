"""Neutral message types shared by the harness, the judge and every adapter.

The harness and the judge only ever build and read these types. Each adapter
translates them to and from one provider's wire format. Nothing in here knows
about any provider.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

# Normalised stop reasons. Adapters map the provider's value onto one of these
# and keep the original in Reply.raw_stop_reason.
STOP_END = "end"  # the model finished its turn with no tool call
STOP_TOOL_CALLS = "tool_calls"  # the model stopped to have tools run
STOP_MAX_TOKENS = "max_tokens"  # cut off by the output limit
STOP_REFUSAL = "refusal"  # the provider says the model declined or was blocked
STOP_OTHER = "other"
STOP_REASONS = (STOP_END, STOP_TOOL_CALLS, STOP_MAX_TOKENS, STOP_REFUSAL, STOP_OTHER)

PRICE_KEYS = ("input_per_mtok", "output_per_mtok")  # the only keys of a model's `price`


@dataclass(frozen=True)
class ToolSpec:
    """A function tool: a name, a description and a JSON Schema for the arguments."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Text:
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass(frozen=True)
class Image:
    """An image file sent to a judge. Contestant conversations never hold one."""

    path: str
    media_type: str = "image/png"

    def read_bytes(self) -> bytes:
        return Path(self.path).read_bytes()

    def base64(self) -> str:
        return base64.standard_b64encode(self.read_bytes()).decode("ascii")

    def sha256(self) -> str:
        return hashlib.sha256(self.read_bytes()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        # Transcripts record which file was shown, not its bytes.
        return {"type": "image", "path": self.path, "media_type": self.media_type, "sha256": self.sha256()}


Part = Union[Text, Image]


@dataclass
class ToolCall:
    """One tool call made by the model.

    `arguments` is the parsed JSON value, normally a dict. If the provider
    delivered arguments that are not valid JSON (a truncated reply, a sloppy
    local server), the adapter passes the raw string through unchanged and the
    canvas rejects the call with a format message.
    """

    id: str
    name: str
    arguments: Any

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class Message:
    """One turn of a conversation.

    role "system" / "user": `content` holds the text (and, for a judge, images).
    role "assistant": `content` holds the visible text, `tool_calls` the calls,
        and `provider_payload` the provider's own form of the turn. An adapter
        must replay `provider_payload` verbatim when `provider` is its own, so
        that reasoning state (thinking blocks, reasoning items, thought
        signatures) survives the tool-result turn.
    role "tool": `content` holds the result text for `tool_call_id`.
    """

    role: str
    content: list[Part] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    provider: str | None = None
    provider_payload: Any = None

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.content if isinstance(p, Text))

    @property
    def images(self) -> list[Image]:
        return [p for p in self.content if isinstance(p, Image)]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": [p.to_dict() for p in self.content]}
        if self.tool_calls:
            d["tool_calls"] = [c.to_dict() for c in self.tool_calls]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_name is not None:
            d["tool_name"] = self.tool_name
        if self.provider is not None:
            d["provider"] = self.provider
        if self.provider_payload is not None:
            d["provider_payload"] = self.provider_payload
        return d


def system(text: str) -> Message:
    return Message(role="system", content=[Text(text)])


def user(*parts: Part | str) -> Message:
    return Message(role="user", content=[Text(p) if isinstance(p, str) else p for p in parts])


def tool_result(call: ToolCall, text: str) -> Message:
    return Message(role="tool", content=[Text(text)], tool_call_id=call.id, tool_name=call.name)


@dataclass
class Usage:
    """Token counts for one model call.

    `output_tokens` is everything billed as output, reasoning included.
    `reasoning_tokens` is the reasoning share of that when the provider reports
    it, else None. Adapters whose provider reports reasoning separately from
    visible output add the two.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_input_tokens": self.cached_input_tokens,
        }


@dataclass
class Reply:
    """What an adapter returns from one model call."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = STOP_END
    raw_stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    refusal_text: str | None = None  # set when the provider signals a refusal or a safety block
    provider: str | None = None
    provider_payload: Any = None  # JSON-serialisable; replayed verbatim on the next call
    raw: Any = None  # the whole provider response as JSON-serialisable data, for the transcript

    def as_message(self) -> Message:
        return Message(
            role="assistant",
            content=[Text(self.text)] if self.text else [],
            tool_calls=list(self.tool_calls),
            provider=self.provider,
            provider_payload=self.provider_payload,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "stop_reason": self.stop_reason,
            "raw_stop_reason": self.raw_stop_reason,
            "usage": self.usage.to_dict(),
            "refusal_text": self.refusal_text,
            "provider": self.provider,
            "raw": self.raw,
        }


@dataclass
class ModelSettings:
    """One config entry: how to reach a model and how to run it.

    `reasoning` and `options` are passed to the adapter untouched; each adapter
    documents the keys it understands. Temperature is never sent: it stays at
    the provider default.

    `price` holds exactly two keys, `input_per_mtok` and `output_per_mtok`, in
    US dollars per million tokens. `concurrency` is the most attempts of this
    model running at once; the provider-wide limit in the run config applies
    as well.
    """

    name: str  # label used in directory names and in the report
    provider: str  # "anthropic" | "openai" | "google" | "openai_compat" | "fake"
    model: str  # the provider's model id
    reasoning: dict[str, Any] = field(default_factory=dict)
    max_output_tokens: int | None = None
    price: dict[str, float] | None = None  # {"input_per_mtok": .., "output_per_mtok": ..} in USD
    options: dict[str, Any] = field(default_factory=dict)
    widths: list[int] | None = None  # optional per-model override of the run's widths
    family: str | None = None  # model family, used to look for judges favouring their own
    estimate: dict[str, Any] = field(default_factory=dict)  # optional overrides for `estimate`
    concurrency: int | None = None  # None: only the provider-wide limit applies

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "reasoning": self.reasoning,
            "max_output_tokens": self.max_output_tokens,
            "price": self.price,
            "options": self.options,
            "widths": self.widths,
            "family": self.family,
            "estimate": self.estimate,
            "concurrency": self.concurrency,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSettings":
        known = {
            "name", "provider", "model", "reasoning", "max_output_tokens", "price", "concurrency", "options", "widths",
            "family", "estimate",
        }
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"model entry {d.get('name', '?')!r}: unknown keys {sorted(unknown)}")
        for key in ("name", "provider", "model"):
            if not d.get(key):
                raise ValueError(f"model entry {d.get('name', '?')!r}: missing {key!r}")
        # A mistyped price key would otherwise cost nothing, silently, in every estimate and record.
        price = d.get("price") or None
        if price is not None:
            if not isinstance(price, dict) or set(price) != set(PRICE_KEYS):
                raise ValueError(f"model entry {d['name']!r}: price must have exactly the keys {list(PRICE_KEYS)} (US dollars per million tokens)")
            for key, value in price.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < float("inf"):
                    raise ValueError(f"model entry {d['name']!r}: price.{key} must be a number, zero or more, not {value!r}")
        concurrency = d.get("concurrency")
        if concurrency is not None and (isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1):
            raise ValueError(f"model entry {d['name']!r}: concurrency must be a whole number, one or more, not {concurrency!r}")
        return cls(
            name=str(d["name"]),
            provider=str(d["provider"]),
            model=str(d["model"]),
            reasoning=dict(d.get("reasoning") or {}),
            max_output_tokens=d.get("max_output_tokens"),
            price=dict(price) if price else None,
            options=dict(d.get("options") or {}),
            widths=list(d["widths"]) if d.get("widths") else None,
            family=d.get("family"),
            estimate=dict(d.get("estimate") or {}),
            concurrency=concurrency,
        )

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float | None:
        if not self.price:
            return None
        return (
            input_tokens * float(self.price.get("input_per_mtok", 0.0))
            + output_tokens * float(self.price.get("output_per_mtok", 0.0))
        ) / 1_000_000
