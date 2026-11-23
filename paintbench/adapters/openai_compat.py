"""The generic adapter: any endpoint that speaks /v1/chat/completions with tools.

OpenRouter first, also local servers (vLLM, Ollama, llama.cpp, LM Studio) and
other vendors' OpenAI-compatible endpoints. Written from
docs/provider-notes/openai_compat.md against the `openai` SDK 2.53.0; the
section numbers below are sections of those notes. Not for api.openai.com with
tools (11.1): use the `openai` adapter there. It was run live through
OpenRouter on 23 November 2026; the tests use a mocked transport.

Settings
--------
reasoning   Empty: nothing is sent and the server's default applies.
    effort   "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max",
             sent under the name options.effort_param chooses. Which of these a
             model takes is the server's business (5.1, 5.3). Reasoning controls
             are not standardised on this API: every other one (the other keys
             of OpenRouter's `reasoning` object, `thinking`,
             `chat_template_kwargs`) goes in options.extra_body.

max_output_tokens
    Sent under the name options.max_tokens_param chooses. None: 64000, which
    leaves room for reasoning plus one strip of rows, is within the output
    limit of every model in 6.1, and stays below the 65,536 at which Gemini 4
    on OpenRouter starts answers that span several requests (6.3; this adapter
    does not continue those, it reports the stop as max_tokens). On most
    servers reasoning counts towards this limit (5.4). On OpenRouter a higher
    value narrows routing to providers that allow a reply of that length and
    raises the credit held while the call runs (6.2).

options
    base_url          Required. For example https://openrouter.ai/api/v1 or
                      http://localhost:8000/v1.
    api_key_env       Name of the environment variable that holds the key.
                      Left out or null: the placeholder key "EMPTY" is sent,
                      which servers that need no key ignore.
    extra_body        Mapping merged into the request body last and verbatim,
                      for vendor fields: {"provider": {"require_parameters":
                      true}}, {"reasoning": {...}}. It may not hold a key the
                      adapter sets itself, nor one the benchmark never sends
                      (sampling parameters, tool_choice, parallel_tool_calls),
                      nor an effort when reasoning.effort is set as well.
    extra_headers     Mapping of header name to value, sent with every request.
    max_tokens_param  "max_tokens" (default) | "max_completion_tokens" | null
                      to send no limit at all (2.6).
    effort_param      "reasoning_effort" (default) | "reasoning". How
                      reasoning.effort is sent: as the standard top-level
                      `reasoning_effort`, which every server of 5.1 with a
                      request setting reads, or as {"reasoning": {"effort": ..}},
                      OpenRouter's own object (5.2), together with any other
                      keys of that object given in extra_body.reasoning. Set
                      "reasoning" for OpenRouter: its models declare support
                      under that name, which is what
                      provider.require_parameters goes by (2.8).
    stream            true (default) | false. See "What bounds one call".
    stream_usage      true (default) | false: whether a streamed request asks
                      for token counts with stream_options.include_usage. Set
                      false for OpenRouter, which always sends them (1.6).
    timeout_seconds   The longest silence allowed on the connection. Default
                      900 when streaming, 3600 when not.
    sdk_max_retries   Retries inside the SDK. Default 0: the harness does its
                      own backoff, and an SDK retry would resend a long call.
    replace_bad_tool_arguments
                      false (default) | true. See "What is replayed".

An unknown key, or a value outside these lists, is a FatalError: a typo must
never run a model at its default effort unnoticed.

What is replayed
----------------
The assistant message object exactly as the server sent it, every key, so that
whichever field carries the reasoning state on that server goes back to it:
`reasoning_details` (OpenRouter), `reasoning_content` (DeepSeek, llama.cpp,
Fireworks), `reasoning` (vLLM, Ollama), a list-valued `content` (Mistral).
Only this is changed before it is stored as the payload (4.3): keys whose value
is null are dropped, except `content`; a null `content` without tool calls
becomes ""; a tool call with a missing or repeated id gets an id of its own;
arguments sent as an object become the JSON string the API specifies. After it
comes one {"role": "tool", "tool_call_id", "content"} message per call, in
call order. The system prompt, the tool and every earlier message are resent
unchanged on every call: reasoning blocks are bound to that prefix (4.6).

A streamed reply is put together from the keys the notes document for a chunk
(1.6): content, refusal, reasoning, reasoning_content, reasoning_details and
tool_calls with their id, name and arguments. A server that keeps state under
any other key of a chunk, or whose streamed reasoning does not go back together
as it was sent, needs `stream: false`, where the message is replayed byte for
byte (5.6).

A reply cut off inside a tool call is replayed as received, with its
incomplete argument string. Ollama is known to reject such a replay; what the
other servers do is not documented. `replace_bad_tool_arguments: true` replays
"{}" in place of an argument string that is not valid JSON.

What bounds one call
--------------------
With `stream: true` the reply is read as it is generated: read_events() below
splits the body into its events and accumulate() assembles them, never the
SDK's stream helper, which corrupts these streams (1.7). The connection may
then be silent for timeout_seconds at most; OpenRouter and DeepSeek send
keep-alive comments while a model reasons. Nothing else limits the length of a
streamed call: it ends when the server finishes or the output limit is reached.
With `stream: false` timeout_seconds has to cover the whole call on servers
that send nothing until the reply is complete.

A streamed reply counts as whole when it carried a finish_reason and either
that value is one no server sends before the end, or the stream closed with
`[DONE]`. xAI documents "end_turn" for chunks that are not the last (3.4), so a
stream that stops after that value alone, with no `[DONE]`, was cut off.

Errors
------
TransientError: rate limits, overload, timeouts, connection failures, 408, 409,
429, 5xx, a stream that breaks off, an empty or non-JSON body, and a provider
failure reported inside an HTTP 200. FatalError: 400, 401, 403, 404, 413, 422,
402 (except OpenRouter's in-flight budget hold). A refusal or policy block that
arrives as an error (OpenRouter 403: 3.5) is returned as a refusal Reply. An
error object inside an HTTP 200 is classified by the same rules wherever it
sits: the whole body, one choice of it, a chunk of a stream or one choice of a
chunk (3.7).
"""

from __future__ import annotations

import copy
import json
import os
import threading
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from ..types import (
    STOP_END,
    STOP_MAX_TOKENS,
    STOP_OTHER,
    STOP_REFUSAL,
    STOP_TOOL_CALLS,
    Image,
    Message,
    ModelSettings,
    Reply,
    ToolCall,
    ToolSpec,
    Usage,
)
from .base import Adapter, AdapterError, FatalError, TransientError

PROVIDER = "openai_compat"
DEFAULT_MAX_OUTPUT_TOKENS = 64000
NO_KEY = "EMPTY"  # the placeholder vLLM's documentation uses; keyless servers ignore it
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
MAX_TOKENS_PARAMS = ("max_tokens", "max_completion_tokens", None)
EFFORT_PARAMS = ("reasoning_effort", "reasoning")
OPTION_DEFAULTS: dict[str, Any] = {
    "base_url": None,
    "api_key_env": None,
    "extra_body": {},
    "extra_headers": {},
    "max_tokens_param": "max_tokens",
    "effort_param": "reasoning_effort",
    "stream": True,
    "stream_usage": True,
    "timeout_seconds": None,  # 900 when streaming, 3600 when not
    "sdk_max_retries": 0,
    "replace_bad_tool_arguments": False,
}

# Keys options.extra_body may not hold. The adapter sets the first group itself. The benchmark never
# sends the second: sampling, tool choice and parallel tool calls stay at the provider default (2.7).
_OWN_KEYS = ("model", "messages", "tools", "stream", "stream_options", "max_tokens", "max_completion_tokens")
_NEVER_SENT = (
    "temperature", "top_p", "top_k", "min_p", "top_a", "presence_penalty", "frequency_penalty", "repetition_penalty",
    "seed", "stop", "n", "logprobs", "top_logprobs", "logit_bias", "response_format", "user", "tool_choice",
    "parallel_tool_calls",
)  # fmt: skip

# Headers the SDK adds unasked. They describe the client machine or carry OpenAI account ids read from
# the environment, and no compatible server needs them (1.2). The last two can only be left out per request.
_SDK_HEADERS = (
    "X-Stainless-Lang", "X-Stainless-Package-Version", "X-Stainless-OS", "X-Stainless-Arch", "X-Stainless-Runtime",
    "X-Stainless-Runtime-Version", "X-Stainless-Async", "OpenAI-Organization", "OpenAI-Project",
)  # fmt: skip
_SDK_REQUEST_HEADERS = ("X-Stainless-Retry-Count", "X-Stainless-Read-Timeout")

# OpenRouter's error.metadata.error_type values (9.3).
_REFUSAL_TYPES = ("refusal", "content_policy_violation")
_TRANSIENT_TYPES = ("provider_overloaded", "provider_unavailable", "rate_limit_exceeded", "timeout", "server", "unmapped")
# finish_reason values that mean the provider failed part-way (3.4): OpenRouter, then two of DeepSeek's.
_FAILED_FINISH = ("error", "insufficient_system_resource", "aborted")
# finish_reason values a server also puts on chunks that are not the last (3.4: xAI). Seeing one does not
# show that a stream ran to its end.
_NOT_FINAL_FINISH = ("end_turn",)
DONE = "[DONE]"  # the data of the event that closes a stream


# ---------------------------------------------------------------- settings


def _check(settings: ModelSettings) -> tuple[dict[str, Any], str | None]:
    """The options with defaults filled in, and the effort. FatalError for anything unknown or invalid."""
    reasoning, given = settings.reasoning or {}, settings.options or {}
    for where, mapping, known in (("reasoning", reasoning, ("effort",)), ("options", given, tuple(OPTION_DEFAULTS))):
        unknown = sorted((k for k in mapping if k not in known), key=str)
        if unknown:
            raise FatalError(f"openai_compat: unknown {where} key {unknown}; allowed: {list(known)}")
    o = {**OPTION_DEFAULTS, **given}

    def bad(key: str, allowed: str) -> FatalError:
        return FatalError(f"openai_compat: options.{key} is {o[key]!r}; allowed: {allowed}")

    def is_number(value: Any, whole: bool = False) -> bool:
        return isinstance(value, int if whole else (int, float)) and not isinstance(value, bool)

    if not (isinstance(o["base_url"], str) and o["base_url"].startswith(("http://", "https://"))):
        raise bad("base_url", "required: the endpoint's http(s) URL, for example https://openrouter.ai/api/v1")
    if o["api_key_env"] is not None and not (isinstance(o["api_key_env"], str) and o["api_key_env"]):
        raise bad("api_key_env", "the name of an environment variable, or null for a server that needs no key")
    if o["max_tokens_param"] not in MAX_TOKENS_PARAMS:
        raise bad("max_tokens_param", '"max_tokens", "max_completion_tokens" or null')
    if o["effort_param"] not in EFFORT_PARAMS:
        raise bad("effort_param", '"reasoning_effort" or "reasoning"')
    for key in ("stream", "stream_usage", "replace_bad_tool_arguments"):
        if not isinstance(o[key], bool):
            raise bad(key, "true or false")
    if o["timeout_seconds"] is None:
        o["timeout_seconds"] = 900.0 if o["stream"] else 3600.0
    if not is_number(o["timeout_seconds"]) or o["timeout_seconds"] <= 0:
        raise bad("timeout_seconds", "a number of seconds above 0")
    if not is_number(o["sdk_max_retries"], whole=True) or o["sdk_max_retries"] < 0:
        raise bad("sdk_max_retries", "a whole number, 0 or more")
    headers = o["extra_headers"]
    if not isinstance(headers, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
        raise bad("extra_headers", "a mapping of header name to value, both strings")
    extra = o["extra_body"]
    if not isinstance(extra, Mapping) or not all(isinstance(k, str) for k in extra):
        raise bad("extra_body", "a mapping of body field to value")
    clash = sorted(k for k in extra if k in _OWN_KEYS + _NEVER_SENT)
    if clash:
        raise FatalError(
            f"openai_compat: options.extra_body may not hold {clash}: the adapter sets {list(_OWN_KEYS)} itself "
            f"and never sends {list(_NEVER_SENT)}"
        )

    effort = reasoning.get("effort")
    if "effort" in reasoning:
        if not isinstance(effort, str) or effort not in EFFORTS:
            raise FatalError(f"openai_compat: reasoning.effort is {effort!r}; allowed: {list(EFFORTS)}")
        nested = extra.get("reasoning")
        if "reasoning_effort" in extra or (isinstance(nested, Mapping) and "effort" in nested):
            raise FatalError("openai_compat: the effort is set in both reasoning.effort and options.extra_body; allowed: one of them")
        if o["effort_param"] == "reasoning" and "reasoning" in extra and not isinstance(nested, Mapping):
            raise FatalError(
                f"openai_compat: options.extra_body.reasoning is {nested!r}; allowed: a mapping, "
                'whose keys are sent next to the effort when options.effort_param is "reasoning"'
            )

    limit = settings.max_output_tokens
    if limit is not None and (not is_number(limit, whole=True) or limit < 1):
        raise FatalError(f"openai_compat: max_output_tokens is {limit!r}; allowed: a whole number above 0, or null for {DEFAULT_MAX_OUTPUT_TOKENS}")
    if limit is not None and o["max_tokens_param"] is None:
        raise FatalError("openai_compat: max_output_tokens is set but options.max_tokens_param is null, which sends no limit; allowed: one of them")
    return o, effort


def _sdk() -> tuple[Any, Any]:
    try:
        import httpx
        import openai
    except ImportError as e:
        raise FatalError('the openai SDK is not installed: pip install "paintbench[openai]"') from e
    return openai, httpx


def build_client(options: Mapping[str, Any], http_client: Any = None) -> Any:
    """The SDK client for one endpoint, from checked options. `http_client` replaces the transport (tests)."""
    openai, httpx = _sdk()
    key = NO_KEY
    if options["api_key_env"]:
        key = os.environ.get(options["api_key_env"], "").strip()
        if not key:
            raise FatalError(
                f"openai_compat: options.api_key_env names {options['api_key_env']!r}, "
                "but that environment variable is not set or is empty"
            )
    headers: dict[str, Any] = {name: openai.Omit() for name in _SDK_HEADERS}
    # The SDK would also send every header in OPENAI_CUSTOM_HEADERS to this endpoint, whoever runs it. Each
    # is left out under the name the variable gives it. A name spelt exactly as one of the four headers the
    # request needs cannot be left out: that would take the SDK's own header with it (with no Authorization
    # the SDK raises a TypeError before it sends anything). It is given the value the SDK sends instead.
    needed = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": f"OpenAI/Python {openai.__version__}",
    }
    for line in os.environ.get("OPENAI_CUSTOM_HEADERS", "").split("\n"):
        if ":" in line:
            name = line.split(":", 1)[0].strip()
            headers[name] = needed.get(name, openai.Omit())
    return openai.OpenAI(
        base_url=options["base_url"],  # always passed, with the key: the SDK would fall back to OPENAI_* variables
        api_key=key,
        timeout=httpx.Timeout(connect=15.0, read=float(options["timeout_seconds"]), write=60.0, pool=60.0),
        max_retries=options["sdk_max_retries"],
        default_headers=headers,
        http_client=http_client,
    )


# ---------------------------------------------------------------- the request


def _is_json(text: Any) -> bool:
    try:
        json.loads(text)
        return True
    except (TypeError, ValueError):
        return False


def _assistant(m: Message, replace_bad_arguments: bool) -> dict[str, Any]:
    if m.provider == PROVIDER and isinstance(m.provider_payload, dict):
        out = copy.deepcopy(m.provider_payload)  # verbatim: whatever carries the reasoning state goes back
    else:  # from a test or another adapter: rebuilt from its text and tool calls
        out = {"role": "assistant", "content": m.text or (None if m.tool_calls else "")}
        if m.tool_calls:
            out["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments if isinstance(c.arguments, str) else json.dumps(c.arguments)},
                }
                for c in m.tool_calls
            ]
    if replace_bad_arguments:
        for call in out.get("tool_calls") or []:
            function = call.get("function")
            if isinstance(function, dict) and not _is_json(function.get("arguments")):
                function["arguments"] = "{}"
    return out


def _message(m: Message, replace_bad_arguments: bool) -> dict[str, Any]:
    if m.role == "assistant":
        return _assistant(m, replace_bad_arguments)
    if m.role == "tool":  # a plain string, and no `name`: Groq rejects it (4.1)
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.text}
    if m.role == "system" or (m.role == "user" and not m.images):
        return {"role": m.role, "content": m.text}
    if m.role == "user":  # a judge's message: text and images in the order given, images as data URLs
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{p.media_type};base64,{p.base64()}"}}
                if isinstance(p, Image)
                else {"type": "text", "text": p.text}
                for p in m.content
            ],
        }
    raise FatalError(f"openai_compat: cannot send a message with role {m.role!r}")


# ---------------------------------------------------------------- streamed replies


def _merge_reasoning_details(merged: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> None:
    """Append streamed `reasoning_details` fragments, merging by type and never by `index` (1.6).

    The rule is the one in OpenRouter's own SDK: text fragments join the last text block until that
    block is signed, summary fragments join the last summary, anything else is a block of its own.
    """
    for delta in deltas:
        delta = dict(delta)
        last = merged[-1] if merged else None
        kind = delta.get("type")
        if kind == "reasoning.text" and last and last.get("type") == "reasoning.text" and not last.get("signature"):
            last["text"] = (last.get("text") or "") + (delta.get("text") or "")
            if delta.get("signature"):
                last["signature"] = delta["signature"]
            if not last.get("format") and delta.get("format"):
                last["format"] = delta["format"]
        elif kind == "reasoning.summary" and last and last.get("type") == "reasoning.summary":
            last["summary"] = (last.get("summary") or "") + (delta.get("summary") or "")
            if not last.get("format") and delta.get("format"):
                last["format"] = delta["format"]
        else:
            merged.append(delta)


def read_events(body: Iterable[bytes]) -> Iterator[Any]:
    """The events of a streamed reply, from the bytes of its body: each chunk as plain JSON data, then DONE.

    DONE is yielded only if the server sent `[DONE]`, and nothing is read after it. The adapter reads the
    events itself, not through the SDK's iterator, because that iterator stops at `[DONE]` without saying
    that it saw it, and `[DONE]` is the one sign of a stream's end that no chunk can be mistaken for.

    The body is split by the rule of the SDK's own decoder: a line ends at \\r, \\n or \\r\\n and is cut out
    before it is decoded (a U+2028 inside a JSON string is not a line end); a blank line ends an event; a
    line that starts with a colon is a comment, which is what the keep-alives of 1.5 are; only `data` lines
    count. An event the body breaks off in the middle of is dropped, as the SDK drops it. A ValueError
    (bytes that are not UTF-8, data that is not JSON) is left to the caller.
    """
    data: list[str] = []  # the data lines of the event being read

    def events(block: bytearray) -> Iterator[Any]:
        for raw in block.splitlines():
            line = raw.decode("utf-8")
            if not line:
                if data:
                    text = "\n".join(data)
                    data.clear()
                    yield DONE if text.startswith(DONE) else json.loads(text)
            elif not line.startswith(":"):
                name, _, value = line.partition(":")
                if name == "data":
                    data.append(value[1:] if value.startswith(" ") else value)

    block = bytearray()  # the bytes read since the last blank line
    for piece in body:
        for line in piece.splitlines(keepends=True):
            block += line
            if block.endswith((b"\r\r", b"\n\n", b"\r\n\r\n")):
                for event in events(block):
                    yield event
                    if event == DONE:
                        return
                block = bytearray()
    for event in events(block):  # what is left when the body ends: complete lines, but no blank line after them
        yield event
        if event == DONE:
            return
    if "\n".join(data).startswith(DONE):  # `[DONE]` as the last line of the body
        yield DONE


def accumulate(chunks: Iterable[Any]) -> dict[str, Any]:
    """Assemble streamed chunks (as dicts) into a dict shaped like a non-streaming chat.completion.

    `chunks` is what read_events() yields: the chunks in order, then DONE if the stream closed with it.
    This is the helper that returns the final response; the code and the reason for each rule are in
    notes 1.6. Tool-call fragments are matched to their call by `index`, with the documented exceptions:
    a different id at a known index starts a new call, and a fragment without an index continues the
    latest call unless it carries a new id, or a name once the latest call's arguments are complete.

    An error object, on a chunk or on a choice of one, is kept where the non-streaming body has it (3.7),
    for parse_response() to classify. Without one the reply has to be whole: see "What bounds one call".
    """
    message: dict[str, Any] = {"role": "assistant", "content": None}
    slots: dict[Any, dict[str, Any]] = {}  # stream index, or a key of our own, -> tool call
    calls: list[dict[str, Any]] = []  # the same tool calls in order of arrival
    last_key: Any = None
    out: dict[str, Any] = {"object": "chat.completion", "choices": [], "usage": None}
    finish = native = choice_error = None
    final = done = False  # a finish_reason that only a last chunk carries was seen; `[DONE]` was seen
    count = 0
    for chunk in chunks:
        if chunk == DONE:
            done = True
            break
        if not isinstance(chunk, Mapping):
            continue  # JSON, but not a chunk: there is nothing in it to assemble
        count += 1
        for key in ("id", "model", "created", "provider", "system_fingerprint"):
            if chunk.get(key) is not None and key not in out:
                out[key] = chunk[key]
        if chunk.get("usage"):
            out["usage"] = chunk["usage"]  # the last one wins; it may ride on a chunk of its own or on the final one
        if chunk.get("error"):
            out["error"] = chunk["error"]  # the provider failed part-way: this is the last thing it sends (10.6)
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, Mapping) or choice.get("index", 0) != 0:
                continue
            reason = choice.get("finish_reason")
            if reason:
                finish = reason
                final = final or reason not in _NOT_FINAL_FINISH
            native = choice.get("native_finish_reason") or native
            choice_error = choice.get("error") or choice_error
            delta = choice.get("delta") if isinstance(choice.get("delta"), Mapping) else {}
            for key in ("content", "refusal", "reasoning", "reasoning_content"):
                value = delta.get(key)
                if isinstance(value, str) and value:
                    message[key] = (message.get(key) or "") + value
                elif isinstance(value, list):  # Mistral's native thinking chunks
                    raise FatalError(f"openai_compat: the stream sends {key} as a list; set options.stream to false for this server")
            if delta.get("reasoning_details"):
                _merge_reasoning_details(message.setdefault("reasoning_details", []), delta["reasoning_details"])
            for part in delta.get("tool_calls") or []:
                function = part.get("function") or {}
                index = part.get("index")
                if index is not None:
                    key = index
                else:  # out of spec, but seen: does this fragment continue the latest call?
                    latest = slots.get(last_key) if last_key is not None else None
                    if latest is None:
                        key = None
                    elif part.get("id"):
                        key = last_key if part["id"] == latest["id"] else None
                    elif function.get("name") and (
                        function["name"] != latest["function"]["name"] or _is_json(latest["function"]["arguments"])
                    ):
                        key = None
                    else:
                        key = last_key
                slot = slots.get(key) if key is not None else None
                if slot is not None and part.get("id") and slot["id"] and part["id"] != slot["id"]:
                    slot = None  # the same index with another id: a new call
                if slot is None:
                    key = ("call", len(calls))
                    slot = {"id": None, "type": "function", "function": {"name": None, "arguments": ""}}
                    slots[key] = slot
                    calls.append(slot)
                    if index is not None:
                        slots[index] = slot  # later fragments at this index continue this call
                last_key = key
                if part.get("id") and not slot["id"]:
                    slot["id"] = part["id"]  # set once: some servers repeat it on every fragment
                if function.get("name") and not slot["function"]["name"]:
                    slot["function"]["name"] = function["name"]
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    slot["function"]["arguments"] += arguments
                elif arguments is not None:  # an object where a string fragment belongs
                    slot["function"]["arguments"] = json.dumps(arguments)
        if "error" in out:
            break
    if count == 0:
        raise TransientError("openai_compat: the stream held no chunks (if this server cannot stream, set options.stream to false)")
    if "error" not in out and not choice_error:  # no failure reported, so the reply must have arrived whole
        if finish is None and done:
            raise TransientError(
                "openai_compat: the stream closed with [DONE] but no chunk carried a finish_reason, so the reply cannot be "
                "taken as whole (if this server never sends one, set options.stream to false)"
            )
        if finish is None:
            raise TransientError("openai_compat: the stream ended without a finish_reason and without [DONE], so the reply was cut off")
        if not (final or done):
            raise TransientError(
                f"openai_compat: the stream ended after finish_reason {finish!r} and without [DONE]. That value is also sent "
                "on chunks that are not the last, so the reply was cut off (if this server ends its streams this way, "
                "set options.stream to false)"
            )
    if calls:
        message["tool_calls"] = calls
    choice = {"index": 0, "message": message, "finish_reason": finish}
    if native:
        choice["native_finish_reason"] = native
    if choice_error:
        choice["error"] = choice_error
    out["choices"] = [choice]
    return out


# ---------------------------------------------------------------- the response


def _payload(message: Mapping[str, Any]) -> dict[str, Any]:
    """The assistant message as it is replayed: as received, changed only as far as notes 4.3 allow."""
    out = {k: copy.deepcopy(v) for k, v in message.items() if v is not None or k == "content"}
    out.setdefault("role", "assistant")
    calls = out.get("tool_calls") or []
    if not calls and out.get("content") is None:
        out["content"] = ""  # Ollama rejects a null content unless tool calls are present
    taken = {c.get("id") for c in calls if c.get("id")}
    seen: set[Any] = set()
    for i, call in enumerate(calls):
        if not call.get("id") or call["id"] in seen:  # missing or repeated: an id no other call in the reply has
            new_id = f"call_{i}"
            while new_id in taken:
                new_id += "_"
            call["id"] = new_id
            taken.add(new_id)
        seen.add(call["id"])
        function = call.get("function")
        if isinstance(function, dict) and not isinstance(function.get("arguments"), str):
            # An object, or nothing, where the API specifies a JSON string: replay the string.
            function["arguments"] = json.dumps({} if function.get("arguments") is None else function["arguments"])
    return out


def _arguments(raw: Any) -> Any:
    """Tool-call arguments parsed; a string that is not valid JSON is passed through unchanged (3.3)."""
    if raw is None or raw == "":
        return {}
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts; reasoning parts are not text (3.2)
        return "".join(p.get("text") or "" for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _usage(usage: Any) -> Usage:
    if not isinstance(usage, dict):
        return Usage()  # an error body, or a stream without the usage chunk

    def count(value: Any) -> int | None:
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def detail(key: str, field: str) -> int | None:
        return count(usage[key].get(field)) if isinstance(usage.get(key), dict) else None

    prompt, completion, total = (count(usage.get(k)) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
    prompt, output = prompt or 0, completion or 0
    reasoning = detail("completion_tokens_details", "reasoning_tokens")
    cached = detail("prompt_tokens_details", "cached_tokens")
    if cached is None:
        cached = count(usage.get("prompt_cache_hit_tokens"))  # DeepSeek's name for it
    # Reasoning is inside completion_tokens on OpenAI and OpenRouter, and on top of it on xAI. The total tells which (3.6).
    if reasoning and total is not None and prompt + output != total and prompt + output + reasoning == total:
        output += reasoning
    return Usage(input_tokens=prompt, output_tokens=output, reasoning_tokens=reasoning, cached_input_tokens=cached)


def _metadata(error: Any) -> dict[str, Any]:
    metadata = error.get("metadata") if isinstance(error, dict) else None
    return metadata if isinstance(metadata, dict) else {}


def _is_refusal(error: Any, status: int | None) -> bool:
    """A refusal or policy block delivered as an error (3.5): retrying the same input gives the same block."""
    metadata = _metadata(error)
    code = status if status is not None else error.get("code") if isinstance(error, dict) else None
    return metadata.get("error_type") in _REFUSAL_TYPES or (code == 403 and bool(metadata.get("reasons")))


def _refusal_reply(error: dict[str, Any], raw: Any, raw_stop_reason: str | None = None) -> Reply:
    text = str(error.get("message") or _metadata(error).get("error_type") or "blocked by the provider")
    return Reply(stop_reason=STOP_REFUSAL, raw_stop_reason=raw_stop_reason, refusal_text=text, provider=PROVIDER, raw=raw)


def _failure(error: Any, status: int | None, retry_after: float | None = None) -> AdapterError:
    """The exception for a provider error: an HTTP error body, or an error object inside a 200 (3.7, 9.3)."""
    if not isinstance(error, dict):
        error = {"message": str(error) if error else "no error body"}
    kind = _metadata(error).get("error_type")
    code = status if status is not None else error.get("code")
    if not isinstance(code, int) or isinstance(code, bool):
        code = None
    if kind in _TRANSIENT_TYPES or code in (408, 409, 429) or (code or 0) >= 500:
        transient = True
    elif code == 402:  # out of credit is final; OpenRouter's hold for requests in flight clears by itself
        transient = _metadata(error).get("limit_source") == "openrouter_in_flight_budget"
    else:
        # An error with neither a code nor a type, sent after a 200: the request itself was accepted, so
        # this is taken for a failure during generation.
        transient = status is None and code is None and kind is None
    where = f"HTTP {status}" if status is not None else f"error inside an HTTP 200 (code {error.get('code')!r})"
    text = f"openai_compat: {where}{f' [{kind}]' if kind else ''}: {error.get('message') or 'no message'}"
    if transient:
        return TransientError(text, status=code, retry_after=retry_after)
    return FatalError(text, status=code)


def _retry_after(headers: Any) -> float | None:
    """Seconds to wait, from `retry-after-ms` or `retry-after` when the server sent a number."""
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            seconds = float(headers.get(name)) * scale
        except (TypeError, ValueError):
            continue
        if seconds >= 0:
            return seconds
    return None


# ---------------------------------------------------------------- the adapter


class OpenAICompatAdapter(Adapter):
    """One instance serves one endpoint: its client is built once, from the options of the first call."""

    provider = PROVIDER

    def __init__(self, client: Any = None):
        super().__init__(client)
        self._lock = threading.Lock()

    def build_request(self, messages: list[Message], tools: list[ToolSpec], settings: ModelSettings) -> dict[str, Any]:
        """The JSON body of the request, exactly as complete() sends it."""
        options, effort = _check(settings)
        body: dict[str, Any] = {
            "model": settings.model,
            "messages": [_message(m, options["replace_bad_tool_arguments"]) for m in messages],
        }
        if tools:  # no `strict`, no tool_choice, no parallel_tool_calls: the schema as given, the rest at the default
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": copy.deepcopy(t.parameters)}}
                for t in tools
            ]
        if options["max_tokens_param"]:
            body[options["max_tokens_param"]] = settings.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS
        if effort and options["effort_param"] == "reasoning":
            body["reasoning"] = {"effort": effort}
        elif effort:
            body["reasoning_effort"] = effort
        if options["stream"]:
            body["stream"] = True
            if options["stream_usage"]:
                body["stream_options"] = {"include_usage": True}
        extra = copy.deepcopy(dict(options["extra_body"]))
        if "reasoning" in body and "reasoning" in extra:
            body["reasoning"].update(extra.pop("reasoning"))  # the other keys of OpenRouter's object, beside the effort
        body.update(extra)
        return body

    def parse_response(self, response: Any) -> Reply:
        """A Reply from a chat.completion as plain JSON data. An error inside the body is raised (3.7)."""
        if not isinstance(response, dict):
            raise TransientError("openai_compat: the response is not a JSON object")
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        finish = choice.get("finish_reason")
        finish = finish if isinstance(finish, str) else None
        error = response.get("error") or choice.get("error")
        if error:
            if _is_refusal(error, None):
                return _refusal_reply(error, response, finish)
            raise _failure(error, None)
        if not choice:
            raise TransientError("openai_compat: the response holds no choices and no error")
        if finish in _FAILED_FINISH:
            raise TransientError(f"openai_compat: the provider stopped the reply with finish_reason {finish!r}")

        message = choice.get("message")
        payload = _payload(message if isinstance(message, dict) else {})
        text = _text(payload.get("content"))
        tool_calls = [
            ToolCall(id=c["id"], name=(c.get("function") or {}).get("name") or "", arguments=_arguments((c.get("function") or {}).get("arguments")))
            for c in payload.get("tool_calls") or []
        ]
        refusal = payload.get("refusal") if isinstance(payload.get("refusal"), str) else ""
        refusal_text = None
        if refusal or finish == "content_filter":
            stop, refusal_text = STOP_REFUSAL, refusal or text or None
        elif finish == "length":
            stop = STOP_MAX_TOKENS
        elif tool_calls:  # the calls decide, not finish_reason: some servers say "stop" with calls present (3.4)
            stop = STOP_TOOL_CALLS
        elif finish in ("stop", "end_turn"):
            stop = STOP_END
        else:
            stop = STOP_OTHER
        return Reply(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop,
            raw_stop_reason=finish,
            usage=_usage(response.get("usage")),
            refusal_text=refusal_text,
            provider=PROVIDER,
            provider_payload=payload,
            raw=response,
        )

    def complete(self, messages: list[Message], tools: list[ToolSpec], settings: ModelSettings) -> Reply:
        body = self.build_request(messages, tools, settings)
        options, _effort = _check(settings)
        openai, httpx = _sdk()
        client = self._client_for(options)
        # The SDK types these four; every other field of the body goes through extra_body untouched.
        named = {k: body[k] for k in ("model", "messages", "tools", "stream") if k in body}
        extra = {k: v for k, v in body.items() if k not in named}
        headers = {**{name: openai.Omit() for name in _SDK_REQUEST_HEADERS}, **options["extra_headers"]}
        try:
            if body.get("stream"):
                # The SDK opens the stream and raises for an error status. The body is then read from the open
                # response, as bytes, and not through the SDK's iterator: see read_events().
                with client.chat.completions.create(**named, extra_body=extra, extra_headers=headers) as stream:
                    response = accumulate(read_events(stream.response.iter_bytes()))
            else:
                # The raw text, not the parsed object: the SDK does not raise for a 200 that holds only an error (1.6).
                raw = client.chat.completions.with_raw_response.create(**named, extra_body=extra, extra_headers=headers)
                response = json.loads(raw.text)
        except openai.APIStatusError as e:
            if _is_refusal(e.body, e.status_code):
                return _refusal_reply(e.body, {"error": e.body})
            raise _failure(e.body, e.status_code, _retry_after(e.response.headers)) from e
        except openai.APIConnectionError as e:  # includes APITimeoutError
            raise TransientError(f"openai_compat: {type(e).__name__}: {e}") from e
        except openai.APIError as e:  # any other error the SDK reports for a response
            if _is_refusal(e.body, None):
                return _refusal_reply(e.body, {"error": e.body})
            raise _failure(e.body or str(e), None) from e
        except httpx.HTTPError as e:  # the connection failed while the stream was being read
            raise TransientError(f"openai_compat: the stream broke off: {type(e).__name__}: {e}") from e
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise TransientError(f"openai_compat: the response was not JSON: {e}") from e
        return self.parse_response(response)

    def _client_for(self, options: Mapping[str, Any]) -> Any:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = build_client(options)
        return self._client
