"""The Google adapter: Gemini models through `generateContent` on the Gemini
Developer API, called with the google-genai SDK (written against 2.17.0; see
docs/provider-notes/google.md). Run live on 20 November 2026 against
gemini-4-carbon (thinking level high); the tests use a mocked transport.

One call is one streamed HTTP request. The SDK has no helper that returns the
final message of a stream, so `assemble_stream` joins the chunks: the parts in
the order they arrived and never merged, the last finish reason, the last
usage. A stream that ends without a finish reason is an incomplete reply and is
retried.

`build_request` returns {"model", "contents", "config"}: `contents` holds the
turns in the API's own JSON form (role and parts, as in the API reference) and
`config` holds the keyword arguments of the SDK's GenerateContentConfig.
`complete()` turns both into SDK objects and sends exactly that. Bytes (images,
thought signatures) are base64 text in the request and in every payload.

A leading system message becomes `system_instruction`. The tool goes out as
one function declaration with its schema, untouched, under
`parameters_json_schema`, which takes plain JSON Schema. No tool choice is sent
(the default lets the model answer in text or call the tool), the API has no
switch for parallel calls, and no sampling parameter is ever sent.

Settings
--------
reasoning  (an empty mapping sends no thinking setting: the model's default)
  effort            "minimal" | "low" | "medium" | "high" | "xhigh", sent as
                    thinking_config.thinking_level. A model takes only some of
                    them: MODEL_EFFORTS lists the ones the documentation gives
                    for each current model (highest: "high" on gemini-4-carbon
                    and -argon, "xhigh" on gemini-4-neon and -hydrogen), and a
                    level a listed model does not take is refused here. A
                    model that is not listed may be given any of the five.
                    Thinking cannot be turned off on Gemini 4. SDK 2.17.0 does
                    not know "xhigh": it warns (UserWarning) and sends it all
                    the same, so do not run with warnings turned into errors.
  include_thoughts  true | false, sent as thinking_config.include_thoughts.
                    With true the reply carries thought summaries. They are
                    never part of Reply.text; they are kept in the payload and
                    replayed, and they keep bytes moving on the stream while
                    the model thinks.
max_output_tokens   Counts thinking as well as the visible reply. None means
                    65,536, the most one request can return and the value
                    Google recommends for evaluations. A larger value is sent
                    as 65,536: more would need continuation requests, which
                    this adapter does not make.
options
  api_key_env       Name of the environment variable that holds the API key.
                    Default GEMINI_API_KEY. The key is passed to the SDK
                    explicitly; the SDK's own lookup (GOOGLE_API_KEY first,
                    then GEMINI_API_KEY) is not relied on.
  base_url          Default https://generativelanguage.googleapis.com/ (the
                    SDK appends /v1beta/models/<model>:streamGenerateContent).
  timeout_seconds   Default 1200. The longest wait for a connection or for the
                    next bytes of the stream, not a limit on the whole call.
  sdk_max_retries   Default 0: the harness does the backing off. N lets the
                    SDK repeat a request that failed before any bytes arrived
                    up to N times.
An unknown key, or a value known to be invalid, is a FatalError.

What bounds one call: the server closes a streamed request after 20 minutes
and returns at most 65,536 tokens of thinking and output in it; the client
gives up when no bytes arrive for `timeout_seconds`. A call is exactly one
request: SDK retries are off by default, and where the SDK can continue a
reply over several requests by itself (google-genai 2.30.1 and later) that is
switched off, because a resumed stream can repeat parts already received.

A stream that is not read to its end is closed before `complete()` raises.
SDK 2.17.0 stops reading at an error in mid-stream (an error event, an event
it cannot read) without closing the HTTP response; what then frees the
connection is only Python discarding the abandoned iterators. So the client
`make_client` builds notes every response it opens (an httpx response hook,
passed through the SDK's `client_args`), and `complete()` closes them on every
way out. A client handed to the constructor is used as it is, without the hook.

What is replayed
----------------
Reply.provider_payload is the model's turn exactly as the API returned it:
{"role": "model", "parts": [...]} with every part in place - text, thought
summaries, function calls, and the `thoughtSignature` each part came with (an
empty text part that only carries a signature included). An assistant message
from this provider is sent back as that payload, unchanged; the API rejects a
function call whose signature is missing. An assistant message from anywhere
else is rebuilt as a text part and function-call parts. Those carry no
signature, so the live API would refuse to continue from such a call; the
harness never mixes providers in one conversation.

The tool results that follow a turn become ONE user turn of `functionResponse`
parts and nothing else, one per call and in the order of the calls, each
echoing the call's id and name with the result text under "output" (stored or
rejected alike: the harness gives one text per call).

A reply that came back with no parts at all (possible when the output limit is
reached while thinking) has nothing to replay, and an empty model turn cannot
be sent, so it sends no turn. The user message after it joins the user message
before it as further parts, so that roles still alternate.

After a turn of tool results that is not possible: that turn takes no text, so
a user message after it would be a second user turn in a row, and the
documentation does not say whether the API accepts one. `complete()` therefore
does not return a reply that was cut off by the output limit with no parts
when the request ended in tool results. It raises TransientError, the harness
sends the same request again, and the empty reply never enters the
conversation (its tokens are billed but reach no count: an error carries no
usage). One case is left in which a user message follows tool results
directly, as a turn of its own: the model ended its turn (STOP) with no parts
while rows were missing, and the harness asks for them. That reply has to be
returned, because when every row is stored the same empty reply is how a
finished attempt ends.

Tool call ids: ToolCall.id is the id the API gave the call. If any call of a
reply has none, or two share one, the calls of that reply are numbered by
position instead (call_1, call_2, ...). Either way the ids are worked out from
the payload alone, the same way when parsing and when replaying.

Stop reasons: STOP is "end", or "tool_calls" when the reply holds function
calls (the API has no finish reason for those; the parts decide). MAX_TOKENS
and CONTINUATION are "max_tokens"; a function call always arrives whole, so
any calls in such a reply are complete and are returned. A blocked prompt
(raw_stop_reason "blockReason:<value>") and the blocking finish reasons
(SAFETY, RECITATION and the others in BLOCKED) are "refusal". Replies the API
marks unusable (MALFORMED_FUNCTION_CALL and the others in UNUSABLE) raise
TransientError, so the same request is sent again; so does a stream that ends
without a finish reason, and the empty cut-off reply to tool results described
above.

Errors: HTTP 408, 429 and 5xx, timeouts, connection errors and broken streams
are TransientError; every other API error (400, 401, 402, 403, 404, ...) is
FatalError. A 402 has the same status text as a 429, so the code decides.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import threading
from collections.abc import Iterable, Mapping
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
from .base import Adapter, FatalError, TransientError

PROVIDER = "google"
API_VERSION = "v1beta"  # explicit, because google-genai 3.x would default to v1
DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/"
DEFAULT_TIMEOUT_SECONDS = 1200  # Google: "set your HTTP read timeout to 20 minutes"
HOP_LIMIT = 65_536  # thinking plus output that one request can return

REASONING_KEYS = ("effort", "include_thoughts")
OPTION_KEYS = ("api_key_env", "base_url", "timeout_seconds", "sdk_max_retries")
EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
# The thinking levels each model accepts (provider notes, section 5.1).
MODEL_EFFORTS: dict[str, tuple[str, ...]] = {
    "gemini-4-carbon": ("low", "medium", "high"),
    "gemini-4-argon": ("low", "medium", "high"),
    "gemini-4-neon": ("low", "medium", "high", "xhigh"),
    "gemini-4-hydrogen": ("low", "medium", "high", "xhigh"),
    "gemini-3.8-flash": ("low", "medium", "high"),
    "gemini-3.6-flash": ("minimal", "low", "medium", "high"),
    "gemini-3.5-flash-lite": ("minimal", "low", "medium", "high"),
    "gemini-3.1-pro-preview": ("low", "medium", "high"),
}

# Finish reasons, compared as strings: the SDK's enums lag behind the API.
TRUNCATED = {"MAX_TOKENS", "CONTINUATION"}
BLOCKED = {"SAFETY", "RECITATION", "LANGUAGE", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "ESCALATION"}
UNUSABLE = {"MALFORMED_FUNCTION_CALL", "MALFORMED_RESPONSE", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS"}
NEVER_RETRY = {"MISSING_THOUGHT_SIGNATURE", "PUP_LIMITED_DISABLED"}

NO_SDK = 'the google-genai SDK is not installed; install it with: pip install "paintbench[google]"'


# ---------------------------------------------------------------- settings


def check_settings(settings: ModelSettings) -> None:
    """Raise FatalError for a reasoning or options key, or a value, this adapter does not accept."""

    def bad(what: str) -> FatalError:
        return FatalError(f"model {settings.name!r} (google): {what}")

    reasoning, options = settings.reasoning or {}, settings.options or {}
    for where, given, known in (("reasoning", reasoning, REASONING_KEYS), ("options", options, OPTION_KEYS)):
        unknown = sorted(str(key) for key in given if key not in known)
        if unknown:
            raise bad(f"unknown {where} key {', '.join(unknown)}; allowed: {', '.join(known)}")
    if "effort" in reasoning:
        allowed = MODEL_EFFORTS.get(settings.model.removeprefix("models/"), EFFORTS)
        if reasoning["effort"] not in allowed:
            raise bad(f"reasoning.effort {reasoning['effort']!r} is not valid for {settings.model}; allowed: {', '.join(allowed)}")
    if not isinstance(reasoning.get("include_thoughts", False), bool):
        raise bad(f"reasoning.include_thoughts must be true or false, not {reasoning['include_thoughts']!r}")
    limit = settings.max_output_tokens
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise bad(f"max_output_tokens must be a whole number of at least 1, not {limit!r}")
    for key in ("api_key_env", "base_url"):
        if key in options and not (isinstance(options[key], str) and options[key].strip()):
            raise bad(f"options.{key} must be a non-empty string, not {options[key]!r}")
    timeout = options.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < math.inf:
        raise bad(f"options.timeout_seconds must be a number of seconds above 0, not {timeout!r}")
    retries = options.get("sdk_max_retries", 0)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise bad(f"options.sdk_max_retries must be a whole number of at least 0, not {retries!r}")


# The HTTP responses opened by the call that is running in this thread. An SDK call runs in the
# thread that makes it, so each thread sees its own.
_in_flight = threading.local()


def _note_response(response: Any) -> None:
    """The response hook of the HTTP client: httpx calls it with each response before its body is read."""
    opened = getattr(_in_flight, "opened", None)
    if opened is not None:
        opened.append(response)


def _close(*things: Any) -> None:
    """Close each of these that can be closed. Closing a second time does nothing."""
    for thing in things:
        close = getattr(thing, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):  # the call's own outcome is the one to report, not a failure to tidy up
                close()


def make_client(options: Mapping[str, Any], transport: Any = None) -> Any:
    """The SDK client for these options. `transport` (an httpx transport) stands in for the network in tests."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise FatalError(NO_SDK) from e
    env = options.get("api_key_env", DEFAULT_API_KEY_ENV)
    api_key = (os.environ.get(env) or "").strip()
    if not api_key:
        raise FatalError(f"no Gemini API key: the environment variable {env} is not set (options.api_key_env names the variable)")
    # Arguments for the HTTP client the SDK builds for itself; everything else about that client stays the SDK's.
    client_args: dict[str, Any] = {"event_hooks": {"response": [_note_response]}}
    if transport is not None:
        client_args["transport"] = transport
    return genai.Client(
        api_key=api_key,
        vertexai=False,  # so that GOOGLE_GENAI_USE_VERTEXAI in the environment cannot switch API
        http_options=types.HttpOptions(
            api_version=API_VERSION,
            base_url=options.get("base_url", DEFAULT_BASE_URL),
            timeout=max(1, int(options.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS) * 1000)),  # milliseconds; 0 would mean none
            retry_options=types.HttpRetryOptions(attempts=1 + options.get("sdk_max_retries", 0)),
            client_args=client_args,
        ),
    )


# ---------------------------------------------------------------- the request


def _user_part(part: Any) -> dict[str, Any]:
    if isinstance(part, Image):
        return {"inlineData": {"mimeType": part.media_type, "data": part.base64()}}
    return {"text": part.text}


def _model_turn(message: Message) -> dict[str, Any]:
    payload = message.provider_payload
    if message.provider == PROVIDER and isinstance(payload, dict) and payload.get("parts"):
        return copy.deepcopy(payload)  # verbatim: every part, with its thought signature
    parts: list[dict[str, Any]] = [{"text": message.text}] if message.text else []
    for call in message.tool_calls:
        # `args` must be a JSON object here; arguments that are not one cannot be expressed.
        args = call.arguments if isinstance(call.arguments, dict) else {}
        parts.append({"functionCall": {"id": call.id, "name": call.name, "args": args}})
    return {"role": "model", "parts": parts}


def _calls(parts: Iterable[Any]) -> list[tuple[str, dict[str, Any]]]:
    """The function calls of a model turn, each with the id the harness knows it by: [(id, functionCall), ...]."""
    calls = [p["functionCall"] for p in parts if isinstance(p, dict) and isinstance(p.get("functionCall"), dict)]
    ids = [call.get("id") for call in calls]
    if not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        ids = [f"call_{n}" for n in range(1, len(calls) + 1)]
    return list(zip(ids, calls))


def _tool_turn(calls: list[tuple[str, dict[str, Any]]], results: list[Message]) -> dict[str, Any]:
    """One user turn answering every call of the model turn before it, in the order of the calls."""
    by_id = {m.tool_call_id: m for m in results}
    if len(by_id) != len(results) or set(by_id) != {call_id for call_id, _ in calls}:
        raise FatalError(
            "tool results do not match the calls of the reply they follow: "
            f"calls {[call_id for call_id, _ in calls]}, results {[m.tool_call_id for m in results]}"
        )
    parts = []
    for call_id, call in calls:
        answer = {"name": call.get("name"), "response": {"output": by_id[call_id].text}}
        if call.get("id"):
            answer = {"id": call["id"], **answer}  # echo the API's own id; a call that had none gets none
        parts.append({"functionResponse": answer})
    return {"role": "user", "parts": parts}


def _answers_calls(contents: list[dict[str, Any]]) -> bool:
    """True when the last turn is one of tool results (see _tool_turn)."""
    return bool(contents) and any("functionResponse" in part for part in contents[-1]["parts"])


def _contents(messages: list[Message]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    joinable = False  # the last turn is a user turn of text and images, which a further user message joins
    i = 0
    while i < len(messages):
        message = messages[i]
        i += 1
        if message.role == "user":
            parts = [_user_part(p) for p in message.content]
            if joinable:
                contents[-1]["parts"] += parts
            else:
                contents.append({"role": "user", "parts": parts})
            joinable = True
        elif message.role == "assistant":
            turn = _model_turn(message)
            results = []
            while i < len(messages) and messages[i].role == "tool":
                results.append(messages[i])
                i += 1
            calls = _calls(turn["parts"])
            if turn["parts"]:  # a turn without parts cannot be sent
                contents.append(turn)
                joinable = False
            if calls or results:
                contents.append(_tool_turn(calls, results))  # function responses only: no text joins this turn
                joinable = False
        else:
            raise FatalError(
                f"the google adapter has no place for a {message.role!r} message there: "
                "'system' may only come first and 'tool' only after an assistant message"
            )
    return contents


# ---------------------------------------------------------------- the response


def assemble_stream(chunks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Join the chunks of one streamed reply into one response of the documented non-streaming shape.

    Each chunk holds only new parts. They are concatenated in arrival order
    and never merged, because a thought signature belongs to the exact part
    it came with. Usage is cumulative, so the last one stands.
    """
    response: dict[str, Any] = {}
    candidate: dict[str, Any] = {}
    parts: list[Any] = []
    role = None
    for chunk in chunks:
        response.update({k: v for k, v in chunk.items() if k != "candidates"})
        for piece in (chunk.get("candidates") or [])[:1]:
            content = piece.get("content") or {}
            role = role or content.get("role")
            parts.extend(p for p in content.get("parts") or [] if p)  # an empty part is one the SDK could not read
            candidate.update({k: v for k, v in piece.items() if k != "content"})
    if parts:
        candidate["content"] = {"role": role or "model", "parts": parts}
    if candidate:
        response["candidates"] = [candidate]
    return response


def _arguments(args: Any) -> Any:
    """The API delivers arguments as a JSON object. A string (a sloppy proxy) is parsed, or passed on as it is."""
    if args is None:
        return {}
    if isinstance(args, str):
        try:
            return json.loads(args)
        except ValueError:
            return args
    return args


def _usage(meta: Mapping[str, Any] | None) -> Usage:
    meta = meta or {}
    thoughts = meta.get("thoughtsTokenCount")
    return Usage(
        input_tokens=(meta.get("promptTokenCount") or 0) + (meta.get("toolUsePromptTokenCount") or 0),
        output_tokens=(meta.get("candidatesTokenCount") or 0) + (thoughts or 0),  # thinking is billed as output, on top
        reasoning_tokens=thoughts,
        cached_input_tokens=meta.get("cachedContentTokenCount"),
    )


# ---------------------------------------------------------------- errors


def _seconds(value: Any) -> float | None:
    """'12', 12 or '1.5s' -> seconds; None for anything else."""
    try:
        seconds = float(str(value).strip().rstrip("s"))
    except ValueError:
        return None
    return seconds if 0 < seconds < math.inf else None


def _retry_after(error: Any) -> float | None:
    """The wait the API asked for: a Retry-After header, else a google.rpc.RetryInfo detail. Neither is documented."""
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    waits = [value for name, value in dict(headers).items() if str(name).lower() == "retry-after"]
    body = error.details.get("error") if isinstance(error.details, dict) else None
    details = body.get("details") if isinstance(body, dict) else None
    for detail in details if isinstance(details, list) else []:
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
            waits.append(detail.get("retryDelay"))
    return next((s for s in map(_seconds, waits) if s), None)


def _api_error(error: Any) -> Exception:
    code = error.code if isinstance(error.code, int) else None
    text = f"Gemini API error {code} {error.status}: {error.message}"
    if code is not None and (code in (408, 429) or code >= 500):
        return TransientError(text, status=code, retry_after=_retry_after(error))
    return FatalError(text, status=code)


# ---------------------------------------------------------------- the adapter


class GoogleAdapter(Adapter):
    provider = PROVIDER

    def __init__(self, client: Any = None):
        super().__init__(client)
        self._lock = threading.Lock()

    def build_request(self, messages: list[Message], tools: list[ToolSpec], settings: ModelSettings) -> dict[str, Any]:
        check_settings(settings)
        messages = list(messages)
        config: dict[str, Any] = {}
        if messages and messages[0].role == "system":
            config["system_instruction"] = messages.pop(0).text
        if tools:
            # `parameters_json_schema` takes plain JSON Schema and passes it on untouched.
            declarations = [
                {"name": t.name, "description": t.description, "parameters_json_schema": copy.deepcopy(t.parameters)} for t in tools
            ]
            config["tools"] = [{"function_declarations": declarations}]
        # Never run tools inside the SDK. This setting is the SDK's own and is not sent to the API.
        config["automatic_function_calling"] = {"disable": True}
        reasoning = settings.reasoning or {}
        thinking: dict[str, Any] = {}
        if "effort" in reasoning:
            thinking["thinking_level"] = reasoning["effort"].upper()  # the API's enum spelling
        if "include_thoughts" in reasoning:
            thinking["include_thoughts"] = reasoning["include_thoughts"]
        if thinking:
            config["thinking_config"] = thinking
        config["max_output_tokens"] = min(settings.max_output_tokens or HOP_LIMIT, HOP_LIMIT)
        return {"model": settings.model, "contents": _contents(messages), "config": config}

    def parse_response(self, response: Any) -> Reply:
        candidate = (response.get("candidates") or [{}])[0] or {}
        content = candidate.get("content") or {}
        parts = [p for p in content.get("parts") or [] if p]
        text = "".join(p["text"] for p in parts if isinstance(p.get("text"), str) and not p.get("thought"))
        tool_calls = [
            ToolCall(id=call_id, name=str(call.get("name") or ""), arguments=_arguments(call.get("args")))
            for call_id, call in _calls(parts)
        ]
        reason = candidate.get("finishReason")
        feedback = response.get("promptFeedback") or {}
        block = feedback.get("blockReason")  # set when the prompt was blocked: then there is no candidate
        refusal_text = None
        if block or reason in BLOCKED:
            stop = STOP_REFUSAL
            what = f"prompt blocked by the provider (blockReason {block})" if block else f"reply blocked by the provider (finishReason {reason})"
            said = candidate.get("finishMessage") or feedback.get("blockReasonMessage") or text
            refusal_text = f"{what}: {said}" if said else what
        elif reason in TRUNCATED:
            stop = STOP_MAX_TOKENS
        elif tool_calls:
            stop = STOP_TOOL_CALLS
        elif reason == "STOP":
            stop = STOP_END
        else:
            stop = STOP_OTHER
        return Reply(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop,
            raw_stop_reason=reason if reason is not None else (f"blockReason:{block}" if block else None),
            usage=_usage(response.get("usageMetadata")),
            refusal_text=refusal_text,
            provider=self.provider,
            provider_payload={**content, "parts": parts} if parts else None,
            raw=response,
        )

    def complete(self, messages: list[Message], tools: list[ToolSpec], settings: ModelSettings) -> Reply:
        request = self.build_request(messages, tools, settings)
        try:
            import httpx
            from google.genai import errors, types
        except ImportError as e:
            raise FatalError(NO_SDK) from e
        client = self._client_for(settings.options or {})
        config = dict(request["config"])
        if "automatic_continuation" in types.GenerateContentConfig.model_fields:
            config["automatic_continuation"] = False  # google-genai 2.30.1 and later: keep a call to one request
        try:
            contents = [types.Content.model_validate(turn) for turn in request["contents"]]
            config = types.GenerateContentConfig(**config)
        except ValueError as e:  # pydantic's ValidationError is a ValueError
            raise FatalError(f"the request does not fit the installed google-genai types: {e}") from e

        chunks: list[dict[str, Any]] = []
        stream = None
        _in_flight.opened = opened = []  # filled by the response hook of a client that make_client built
        try:
            # The request is sent on the first step of the loop, so HTTP errors are raised inside it.
            stream = client.models.generate_content_stream(model=request["model"], contents=contents, config=config)
            for chunk in stream:
                if not isinstance(chunk, dict):
                    chunk = chunk.model_dump(mode="json", by_alias=True, exclude_none=True, exclude={"sdk_http_response"})
                chunks.append(chunk)
        except errors.APIError as e:
            raise _api_error(e) from e
        except httpx.TimeoutException as e:
            raise TransientError(f"the request timed out ({type(e).__name__}): {e}") from e
        except httpx.TransportError as e:
            raise TransientError(f"connection error ({type(e).__name__}): {e}") from e
        except errors.UnknownApiResponseError as e:
            raise TransientError(f"the stream held something that is not a response: {e}") from e
        except ValueError as e:  # a model name the SDK refuses, a response that does not fit its types
            raise FatalError(f"{type(e).__name__}: {e}") from e
        finally:
            # The SDK does not close a stream it stops reading (2.17.0), so the connection would be held
            # until Python discards the abandoned iterators. A response read to its end is closed already.
            _in_flight.opened = None
            _close(stream, *opened)

        reply = self.parse_response(assemble_stream(chunks))
        reason = reply.raw_stop_reason
        if reason is None:
            raise TransientError("the stream ended without a finish reason, so the reply is incomplete")
        if reason in UNUSABLE:
            raise TransientError(f"the API marked the reply unusable (finishReason {reason})")
        if reason in NEVER_RETRY:
            raise FatalError(f"the API ended the reply with finishReason {reason}")
        if reply.stop_reason == STOP_MAX_TOKENS and reply.provider_payload is None and _answers_calls(request["contents"]):
            # The harness would answer with a user message, which cannot follow tool results directly.
            raise TransientError(
                f"the reply to the tool results was cut off by the output limit before it held anything (finishReason {reason}); "
                "it leaves no turn to continue from, so the same request is sent again"
            )
        return reply

    def _client_for(self, options: Mapping[str, Any]) -> Any:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = make_client(options)
        return self._client
