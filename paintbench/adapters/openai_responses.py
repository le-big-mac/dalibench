"""The "openai" adapter: OpenAI's own reasoning models through the Responses API.

Written from docs/provider-notes/openai.md (section numbers below are that
file's) against openai 2.53.0. Run live on 20 November 2026 against
gpt-6.2-astra (effort max, background mode); the tests use a mocked transport.

The request (POST /v1/responses)

    model               settings.model
    input               the conversation as a flat list of items, see "Messages"
    tools               each tool as {"type": "function", name, description, parameters,
                        "strict": false}; no key at all when there is no tool. "strict": false
                        is sent on purpose: when it is left out the API tries to rewrite the
                        schema into strict mode (2.4).
    reasoning           settings.reasoning; no key at all when that is empty
    max_output_tokens   settings.max_output_tokens, or 128000
    store               true                              (options.store)
    background          true                              (options.background)
    include             not sent; ["reasoning.encrypted_content"] only with
                        options.include_encrypted_reasoning true, which needs options.store false:
                        the API refuses encrypted reasoning for a stored response (HTTP 400
                        "Encrypted content cannot be requested for persisted responses", the first
                        live request, 20 November 2026). With the response stored, the reasoning
                        items that are replayed are found by their ids.

Nothing else is sent: no temperature or other sampling parameter, no
tool_choice or parallel_tool_calls (provider defaults), no instructions, no
metadata, no user or safety identifier. The request is the same from call to
call of one conversation except that "input" grows at its end, which the newer
models need in order to keep their earlier reasoning (4.5).

Settings

    reasoning (empty mapping: nothing is sent and the model runs at its own default)
        effort    "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"
                  Every value the API knows. Not every model accepts every one and the adapter
                  keeps no per-model list: the value is sent as written and never replaced. If
                  the API rejects it the call fails with a FatalError that names it.
        summary   "auto" | "concise" | "detailed". Asks for reasoning summaries, which then
                  appear in the reasoning items of the transcript. May need a verified
                  organisation (5.2).
        mode      "standard" | "pro" (5.2). Independent of effort.

    max_output_tokens
        A whole number, 16 or more. None means 128000, the documented maximum of every GPT-6.x
        model. Reasoning tokens count against it (5.3), so a low value at a high effort ends
        replies during reasoning. It may also count in full against the tokens-per-minute rate
        limit (6): lower it if calls draw 429s while real usage is small.

    options (a key left out or set to null takes its default)
        api_key_env                  name of the environment variable holding the API key.
                                     Default "OPENAI_API_KEY".
        base_url                     default "https://api.openai.com/v1". Always passed to the
                                     SDK, so OPENAI_BASE_URL in the environment has no effect.
        timeout_seconds              limit on one HTTP request, as the longest silence on the
                                     connection. Default 600. Connecting is limited to 10 s.
        sdk_max_retries              the SDK's own retries of a failed HTTP request. Default 0:
                                     the harness does the backoff, and a retried create could
                                     start a second generation (9.2).
        background                   true (default): background mode, see "Long calls".
                                     false: one long HTTP request; raise timeout_seconds to the
                                     longest call you will accept.
        store                        true (default) | false. Only decides how long a finished
                                     response can be fetched by id; the conversation never
                                     depends on stored state. With false, a background result
                                     is kept for only about ten minutes (1.4).
        include_encrypted_reasoning  false (default) | true. false leaves "include" out of the
                                     request; keep store true with it, so that replayed
                                     reasoning is found by its item id. true needs store false:
                                     the API refuses the two together (see "include" above).
        poll_seconds                 pause between polls of a background response. Default 2.
        max_wait_seconds             longest wait for one background response, from the moment
                                     the API accepts it. Default 3600. See "Long calls".
        image_detail                 "low" | "high" | "original" | "auto"; sent with every
                                     image (judge calls). Default "high" (7.2).

    An unknown key, or a value outside the lists above, is a FatalError that names the key and
    what is allowed.

Messages

    system      {"role": "developer", "content": text} where it stands, which for the leading
                system message is the first input item. The top-level "instructions" field is
                not used: newer models append it after the input and bill it uncached on every
                request (2.2).
    user        {"role": "user", "content": text}; with images, content is a list of
                {"type": "input_text"} and {"type": "input_image"} parts in the order given.
                An image is sent as a base64 data URL of the file's bytes. No path or file name
                is sent.
    assistant   see "What is replayed".
    tool        {"type": "function_call_output", "call_id": id, "output": text}, in the order
                given, which is the order of the calls.

What is replayed

    Conversation state is carried by the request alone (4.1); previous_response_id is not
    used. Reply.provider_payload is the response's "output" list: every item, in order, exactly
    as the API sent it (reasoning items with their id, summary and encrypted_content, function
    calls with their call_id and their argument string untouched, messages with their phase).
    An assistant message whose provider is "openai" is replayed by putting those items back
    into "input" unchanged. Any other assistant message is rebuilt: its text as
    {"role": "assistant", "content": text}, then one {"type": "function_call", call_id, name,
    arguments} per tool call.

    A reply that did not finish (status "incomplete": cut off by the output limit) is replayed
    in the same way. The harness goes on after one: it runs the calls that arrived whole and
    answers every call, or sends its reminder; a judge asks again. The turn it goes on from is
    then the turn the API sent, reasoning included, which is what 4.1 and 4.2 ask for every
    response. No item is rebuilt, repaired or left out. A call that was cut off (3.3: never run
    a call marked incomplete) is returned with its argument string unparsed, so the canvas
    rejects it and the replayed call has an answer.

    The documentation does not say whether the API takes such a turn back (notes, "Unverified":
    a reasoning item with nothing after it, a call marked incomplete). If it does not, the
    request fails with HTTP 400. That is a FatalError which says a cut-off reply was replayed;
    the harness discards the conversation and starts a fresh one. The adapter never answers a
    400 by sending the turn again in another form or without its reasoning.

The reply

    text          the output_text parts of every message item, in order. Separate messages (an
                  interim note, then the answer) are joined by a blank line.
    tool_calls    one per function_call item, in order; the id is the item's call_id, which is
                  what a result must name, not the item's own id.
    stop_reason   The API has no such field (3.5). It is read from the status and the output,
                  the first of these that applies:
                      incomplete, reason content_filter        refusal
                      a refusal part and no function call      refusal
                      incomplete, reason max_output_tokens     max_tokens
                      incomplete, any other reason             other
                      completed, with a function call          tool_calls
                      completed, without one                   end
                  raw_stop_reason is the status, then ":" and the reason when there is one.
    usage         output_tokens already holds the reasoning tokens and input_tokens the cached
                  ones (3.7); nothing is added twice.
    raw           the whole response. Its "input_transformations" list, when not empty, names
                  reasoning items the API dropped from the request (4.5).

Long calls

    A call at the highest effort can run for many minutes, and a plain request sends nothing
    until it is finished. So the default is what the documentation recommends (1.4): background
    mode. The create call returns at once with status "queued" and the adapter polls
    GET /v1/responses/{id} until the status is final. Every HTTP request is then short and is
    limited by timeout_seconds. Streaming is not used: the SDK's helper that returns the final
    response raises on every reply that is cut off (1.4, 11.9).

    The wait for a reply is limited by max_wait_seconds, counted from the moment the API accepts
    the job. A pause never runs past that deadline, and a poll is given no longer than the time
    left (but at least 10 s, so that the poll made at the deadline can still fetch a reply that
    is ready). A reply a poll returns is always kept: it is paid for.

    Once a job is accepted, the adapter only ever polls that job. A poll that cannot be used is
    repeated for the same id: an error worth retrying, a body that is not a JSON object, a
    status the API does not document. The pause doubles after each (up to 60 s), or is the
    API's Retry-After when that is longer.

    The job is given up at the deadline, after six such polls in a row, and when a poll fails
    for good. The caller will then send the request again, which starts a second, paid reply.
    So the adapter first asks for the job to be cancelled (once, limited to 30 s) and only then
    raises; the error names the response and says whether the cancel worked. A cancel that did
    not work is also logged as a warning, since the harness does not keep the text of an error
    it retries. (A create call that fails before its answer is read leaves no id to cancel:
    9.2.)

    Responses are read as the JSON the API sent (the SDK's with_raw_response), not through the
    SDK's models, so items of a kind this SDK version does not know are replayed unchanged.

Errors

    TransientError   429 rate limits, 5xx, 408, 409, timeouts, connection errors (with
                     retry_after when the API sends Retry-After); an answer that is not a JSON
                     object; a response that ended "failed" with code server_error or
                     rate_limit_exceeded, or "cancelled"; a background reply given up at the
                     deadline or after too many failed polls.
    FatalError       400, 401, 403, 404, 422; a 429 for billing, spend or quota; any other
                     failed response; invalid settings; no API key; no SDK.
    A 400 that the notes could not rule out says which setting to change: an effort the model
    does not take, "include" (options.include_encrypted_reasoning), the image detail
    (options.image_detail), a cut-off reply that was replayed. The request is never changed
    and sent again by the adapter itself.
    A safety block that arrives as an error (code invalid_prompt, bio_policy, cyber_policy,
    image_content_policy_violation or misalignment_policy_violation, as an HTTP error or as a
    failed response) is returned as a refusal, not raised: the documentation says never to
    retry it (3.6), and the harness would restart the conversation after a FatalError.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable

from ..types import (
    STOP_END,
    STOP_MAX_TOKENS,
    STOP_OTHER,
    STOP_REFUSAL,
    STOP_TOOL_CALLS,
    Image,
    Message,
    ModelSettings,
    Part,
    Reply,
    ToolCall,
    ToolSpec,
    Usage,
)
from .base import Adapter, AdapterError, FatalError, TransientError

log = logging.getLogger(__name__)

PROVIDER = "openai"
DEFAULT_MAX_OUTPUT_TOKENS = 128_000
MIN_MAX_OUTPUT_TOKENS = 16
ENCRYPTED_REASONING = "reasoning.encrypted_content"

REASONING_VALUES: dict[str, tuple[str, ...]] = {
    "effort": ("none", "minimal", "low", "medium", "high", "xhigh", "max"),
    "summary": ("auto", "concise", "detailed"),
    "mode": ("standard", "pro"),
}
IMAGE_DETAILS = ("low", "high", "original", "auto")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _is_env_name(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value))


def _is_flag(value: Any) -> bool:
    return isinstance(value, bool)


def _is_positive(value: Any) -> bool:
    return _is_number(value) and value > 0


# option -> (default, test of a value, what is allowed)
OPTIONS: dict[str, tuple[Any, Callable[[Any], bool], str]] = {
    "api_key_env": ("OPENAI_API_KEY", _is_env_name, "the name of an environment variable, never the key itself"),
    "base_url": ("https://api.openai.com/v1", _is_text, "a URL"),
    "timeout_seconds": (600.0, _is_positive, "a number of seconds above 0"),
    "sdk_max_retries": (0, lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0, "a whole number, 0 or more"),
    "background": (True, _is_flag, "true or false"),
    "store": (True, _is_flag, "true or false"),
    "include_encrypted_reasoning": (False, _is_flag, "true or false"),
    "poll_seconds": (2.0, _is_positive, "a number of seconds above 0"),
    "max_wait_seconds": (3600.0, _is_positive, "a number of seconds above 0"),
    "image_detail": ("high", lambda v: v in IMAGE_DETAILS, "one of " + ", ".join(IMAGE_DETAILS)),
}

IN_FLIGHT = ("queued", "in_progress")  # a background response that is not finished
FINISHED = ("completed", "incomplete", "failed", "cancelled")  # one that is (3.5)
POLL_FAILURES = 6  # polls in a row that could not be used, before the reply is given up
POLL_PAUSE_MAX = 60.0  # the longest pause the adapter chooses itself; a Retry-After may be longer
LAST_POLL_SECONDS = 10.0  # the least time a poll is given, so the one made at the deadline can still answer
CANCEL_SECONDS = 30.0  # the most time the cancel request is given
CONNECT_SECONDS = 10.0  # the most time connecting is given, in any request

# Safety blocks that arrive as errors (3.6). Never retried.
POLICY_CODES = frozenset(
    {"invalid_prompt", "bio_policy", "cyber_policy", "image_content_policy_violation", "misalignment_policy_violation"}
)
# 429s that no retry can lift (9.3).
QUOTA_CODES = frozenset(
    {
        "credit_balance_exhausted",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
        "insufficient_quota",
        "billing_hard_limit_reached",
    }
)
# Codes of a response that ended "failed" for which sending the request again is reasonable (9.3).
RETRYABLE_FAILURES = frozenset({"server_error", "rate_limit_exceeded"})


# ---------------------------------------------------------------- settings


def _check_keys(settings: ModelSettings, where: str, given: dict[str, Any], known: Any) -> None:
    unknown = sorted(str(key) for key in given if key not in known)
    if unknown:
        raise FatalError(f"model {settings.name!r}: unknown {where} key {', '.join(unknown)}; allowed: {', '.join(known)}")


def read_options(settings: ModelSettings) -> dict[str, Any]:
    """settings.options with the defaults filled in. An unknown key or an invalid value is a FatalError."""
    given = dict(settings.options or {})
    _check_keys(settings, "options", given, OPTIONS)
    options: dict[str, Any] = {}
    for key, (default, is_valid, allowed) in OPTIONS.items():
        value = given.get(key)
        if value is None:
            value = default
        elif not is_valid(value):
            shown = "not valid" if key == "api_key_env" else repr(value)  # a key pasted here must not reach a log
            raise FatalError(f"model {settings.name!r}: options.{key} is {shown}; allowed: {allowed}")
        options[key] = value
    if options["include_encrypted_reasoning"] and options["store"]:
        raise FatalError(
            f"model {settings.name!r}: options.include_encrypted_reasoning and options.store are both true, and the API "
            "refuses encrypted reasoning for a stored response (HTTP 400, seen on 20 November 2026). Leave "
            "include_encrypted_reasoning out, so that replayed reasoning is found by its item id, or set options.store to false."
        )
    return options


def read_reasoning(settings: ModelSettings) -> dict[str, Any]:
    """settings.reasoning as the request's "reasoning" object. An unknown key or value is a FatalError."""
    given = dict(settings.reasoning or {})
    _check_keys(settings, "reasoning", given, REASONING_VALUES)
    for key, value in given.items():
        if not isinstance(value, str) or value not in REASONING_VALUES[key]:
            allowed = ", ".join(REASONING_VALUES[key])
            raise FatalError(f"model {settings.name!r}: reasoning.{key} is {value!r}; allowed: {allowed}")
    return {key: given[key] for key in REASONING_VALUES if key in given}


def read_max_output_tokens(settings: ModelSettings) -> int:
    value = settings.max_output_tokens
    if value is None:
        return DEFAULT_MAX_OUTPUT_TOKENS
    if isinstance(value, bool) or not isinstance(value, int) or value < MIN_MAX_OUTPUT_TOKENS:
        raise FatalError(
            f"model {settings.name!r}: max_output_tokens is {value!r}; allowed: a whole number, "
            f"{MIN_MAX_OUTPUT_TOKENS} or more, or nothing for the default of {DEFAULT_MAX_OUTPUT_TOKENS}"
        )
    return value


# ---------------------------------------------------------------- the request


def _tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": copy.deepcopy(tool.parameters),
        "strict": False,
    }


def _user_content(parts: list[Part], detail: str) -> Any:
    """Plain text for a message that is one piece of text; otherwise the parts, in the order given."""
    if not any(isinstance(part, Image) for part in parts) and len(parts) <= 1:
        return "".join(part.text for part in parts)
    content: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, Image):
            url = f"data:{part.media_type};base64,{part.base64()}"
            content.append({"type": "input_image", "image_url": url, "detail": detail})
        else:
            content.append({"type": "input_text", "text": part.text})
    return content


def _plain_turn(text: str, calls: list[tuple[Any, Any, Any]]) -> list[dict[str, Any]]:
    """An assistant turn as fresh input items: its text, then its calls as (call_id, name, argument string)."""
    items: list[dict[str, Any]] = [{"role": "assistant", "content": text}] if text else []
    for call_id, name, arguments in calls:
        items.append({"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments})
    return items


def _arguments_text(arguments: Any) -> str:
    return arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)


def _input_items(messages: list[Message], detail: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            items.append({"role": "developer", "content": m.text})
        elif m.role == "user":
            items.append({"role": "user", "content": _user_content(m.content, detail)})
        elif m.role == "assistant" and m.provider == PROVIDER and m.provider_payload is not None:
            if not isinstance(m.provider_payload, list):
                raise FatalError("an assistant message from this adapter must carry its output items as a list")
            items.extend(copy.deepcopy(m.provider_payload))
        elif m.role == "assistant":
            items.extend(_plain_turn(m.text, [(c.id, c.name, _arguments_text(c.arguments)) for c in m.tool_calls]))
        elif m.role == "tool" and m.tool_call_id:
            items.append({"type": "function_call_output", "call_id": m.tool_call_id, "output": m.text})
        else:
            raise FatalError(f"cannot send a {m.role!r} message" + (" without a tool_call_id" if m.role == "tool" else ""))
    return items


# ---------------------------------------------------------------- the response


def _parsed_arguments(item: dict[str, Any]) -> Any:
    """The arguments as a JSON value, or the string as it came: not valid JSON, or the call was cut off."""
    raw = item.get("arguments")
    if not isinstance(raw, str) or item.get("status") not in (None, "completed"):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _usage(response: dict[str, Any]) -> Usage:
    """Reasoning tokens are part of output_tokens, and cached tokens part of input_tokens (3.7)."""
    usage = response.get("usage") or {}
    return Usage(
        input_tokens=usage.get("input_tokens") or 0,
        output_tokens=usage.get("output_tokens") or 0,
        reasoning_tokens=(usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
        cached_input_tokens=(usage.get("input_tokens_details") or {}).get("cached_tokens"),
    )


def _blocked(code: str, message: Any, raw_stop_reason: str, raw: Any, usage: Usage | None = None) -> Reply:
    """A safety block that arrived as an error: a refusal as far as the harness is concerned."""
    return Reply(
        stop_reason=STOP_REFUSAL,
        raw_stop_reason=raw_stop_reason,
        usage=usage or Usage(),
        refusal_text=f"{code}: {message}" if message else code,
        provider=PROVIDER,
        provider_payload=[],
        raw=raw,
    )


# ---------------------------------------------------------------- the call


def _sdk() -> Any:
    try:
        import openai
    except ImportError as e:
        raise FatalError('the openai package is not installed; install it with: pip install "paintbench[openai]"') from e
    return openai


def _make_client(options: dict[str, Any]) -> Any:
    openai = _sdk()
    key = os.environ.get(options["api_key_env"])
    if not key:
        raise FatalError(
            f"no OpenAI API key: the environment variable {options['api_key_env']} is not set "
            "(options.api_key_env names the variable to read)"
        )
    return openai.OpenAI(
        api_key=key,
        base_url=options["base_url"],
        timeout=_timeout(openai, options["timeout_seconds"]),
        max_retries=options["sdk_max_retries"],
    )


def _timeout(openai: Any, seconds: float) -> Any:
    """The limit on one HTTP request: `seconds` of silence on the connection, and less for connecting."""
    return openai.Timeout(seconds, connect=min(CONNECT_SECONDS, seconds))


def _json(raw: Any) -> dict[str, Any]:
    """The body of an SDK raw response, as the JSON the API sent."""
    http = raw.http_response
    try:
        data = json.loads(http.content)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise TransientError(f"OpenAI answered HTTP {http.status_code} with a body that is not a JSON object: {http.text[:200]!r}")
    return data


def _retry_after(headers: Any) -> float | None:
    """Seconds to wait, from Retry-After-Ms or Retry-After (9.4)."""
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            seconds = float(headers.get(name)) * scale
        except (TypeError, ValueError):
            continue
        if seconds > 0:
            return seconds
    return None


def _adapter_error(e: Any, request: dict[str, Any]) -> AdapterError:
    """The harness's error for an SDK exception, chosen by the SDK's exception class (9.1, 9.3)."""
    openai = _sdk()
    model = request.get("model")
    if isinstance(e, openai.APITimeoutError):
        return TransientError(f"OpenAI request for model {model!r} timed out (options.timeout_seconds)")
    if isinstance(e, openai.APIConnectionError):
        return TransientError(f"OpenAI connection error for model {model!r}: {e.__cause__ or e}")
    status = e.status_code
    body = e.body if isinstance(e.body, dict) else {}
    message = (
        f"OpenAI HTTP {status} ({type(e).__name__}) for model {model!r}: {body.get('message') or e.message} "
        f"[code={e.code}, type={e.type}, param={e.param}, request_id={e.request_id}]"
    )
    if isinstance(e, openai.RateLimitError) and (e.code in QUOTA_CODES or e.type == "insufficient_quota"):
        return FatalError(message + ". This is a billing, spend or quota limit; retrying cannot lift it.", status=status)
    if isinstance(e, (openai.RateLimitError, openai.InternalServerError, openai.ConflictError)) or status == 408:
        return TransientError(message, status=status, retry_after=_retry_after(e.response.headers))
    if isinstance(e, openai.BadRequestError):
        message += _advice(f"{e.param} {message}".lower(), str(e.param).lower(), request)
    return FatalError(message, status=status)


def _newest_turn_was_cut_off(items: list[dict[str, Any]]) -> bool:
    """Whether the newest model turn in the input is one the output limit cut off: it holds an item marked
    incomplete, or it ends in a reasoning item. Earlier turns are not looked at: the API has taken those before."""
    turn: list[dict[str, Any]] = []  # the model's own items of its newest turn, last first
    for item in reversed(items):
        if item.get("type") not in (None, "function_call_output"):  # not a message with a role, not a tool result
            turn.append(item)
        elif turn:
            break
    return bool(turn) and (turn[0].get("type") == "reasoning" or any(item.get("status") == "incomplete" for item in turn))


def _advice(said: str, param: str, request: dict[str, Any]) -> str:
    """What to change after an HTTP 400 that the notes could not rule out ("Still unverified").

    The shape of these errors is not documented, so each is recognised loosely, from the parameter the API names
    and the words of its message, and only when the request holds the thing in question. The adapter says what
    to change; it never changes the request and sends it again.
    """
    advice = ""
    effort = (request.get("reasoning") or {}).get("effort")
    if effort and "effort" in said:
        advice += (
            f". The API did not accept reasoning.effort {effort!r} for this model. Set reasoning.effort in the "
            f"config to a value the model's page lists (the API knows {', '.join(REASONING_VALUES['effort'])}; "
            "not every model takes every one). The adapter never replaces the value itself."
        )
    if request.get("include") and (param.startswith("include") or "encrypted_content" in said):
        advice += (
            f'. If the API is refusing "include": ["{ENCRYPTED_REASONING}"], set options.include_encrypted_reasoning '
            "to false and keep options.store true: replayed reasoning is then found by its item id."
        )
    items = request.get("input") or []
    images = [part for item in items if isinstance(item.get("content"), list) for part in item["content"] if part.get("type") == "input_image"]
    if images and "detail" in said:
        advice += (
            f". If the API is refusing the image detail {images[0].get('detail')!r}, set options.image_detail to "
            '"auto", which the documentation\'s own examples for the newest models use.'
        )
    if _newest_turn_was_cut_off(items):
        advice += (
            ". The newest turn this request replays is a reply that was cut off by the output limit (it ends in a "
            "reasoning item, or holds an item marked incomplete), sent back as it came. The documentation does not "
            "say whether the API takes such a turn back. If that is what it is refusing, this conversation cannot go "
            "on, and the way to avoid it is to keep replies from being cut off: raise max_output_tokens or lower "
            "reasoning.effort."
        )
    return advice


def _final_response(client: Any, request: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    """Send the request and return the finished response as JSON data.

    Raises the SDK's own exceptions for the create call. Once the API has accepted a background job, whatever ends
    the wait for it without a reply (the deadline, polls that keep failing, a poll that fails for good) is raised
    only after the job has been asked to stop: the caller reacts to an error by sending the request again, and that
    must not leave two replies running.
    """
    responses = client.responses.with_raw_response
    response = _json(responses.create(**request))
    if not request.get("background") or response.get("status") in FINISHED:
        return response
    response_id = response.get("id")
    if not _is_text(response_id):
        raise TransientError(f"OpenAI accepted a background request but its answer has no id to poll: {json.dumps(response)[:200]}")
    log.debug("OpenAI: background response %s accepted for model %s", response_id, request.get("model"))  # 1.4: note the id at once
    try:
        return _poll(responses, response_id, response.get("status"), request, options)
    except Exception as e:
        outcome = _cancel(responses, response_id, options)
        if isinstance(e, AdapterError):
            e.args = (f"{str(e).rstrip('.')}. {outcome}",)  # the message is the first argument
        raise  # an SDK error (a safety block) is turned into a refusal by complete()


def _poll(responses: Any, response_id: str, status: Any, request: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    """Poll an accepted background response until it is finished, and return it. Creates nothing and cancels nothing.

    Raises a TransientError at the deadline and after too many polls in a row that could not be used, a FatalError
    for a poll that failed for good, and the SDK's own exception for a safety block.
    """
    openai = _sdk()
    include = {"include": request["include"]} if request.get("include") else {}
    every, longest = options["poll_seconds"], options["max_wait_seconds"]
    deadline = time.monotonic() + longest
    pause, failures, trouble = every, 0, ""
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TransientError(
                f"OpenAI background response {response_id} was still {status} after {longest:g} s (options.max_wait_seconds){trouble}."
            )
        time.sleep(min(pause, left))
        limit = min(options["timeout_seconds"], max(deadline - time.monotonic(), LAST_POLL_SECONDS))
        try:
            polled = _json(responses.retrieve(response_id, timeout=_timeout(openai, limit), **include))
            if polled.get("status") not in IN_FLIGHT + FINISHED:
                raise TransientError(
                    f"OpenAI answered a poll of background response {response_id} with status {polled.get('status')!r}, "
                    f"which is not one of {', '.join(IN_FLIGHT + FINISHED)}"
                )
        except (openai.APIStatusError, openai.APIConnectionError, TransientError) as e:
            if isinstance(e, openai.APIStatusError) and e.code in POLICY_CODES:
                raise
            if isinstance(e, openai.NotFoundError):
                raise FatalError(
                    f"OpenAI background response {response_id} can no longer be retrieved (HTTP 404). With "
                    "options.store false a finished response is kept for only about ten minutes: set it to true.",
                    status=404,
                ) from e
            error = e if isinstance(e, TransientError) else _adapter_error(e, request)
            failures += 1
            if error.retryable and failures <= POLL_FAILURES:  # ask again, for the same response
                pause = max(error.retry_after or 0.0, min(POLL_PAUSE_MAX, every * 2**failures))
                trouble = f"; the last poll failed: {error}"
                continue
            if error is e:
                raise
            raise error from e
        if polled["status"] in FINISHED:
            return polled
        status, pause, failures, trouble = polled["status"], every, 0, ""


def _cancel(responses: Any, response_id: str, options: dict[str, Any]) -> str:
    """Ask once, briefly, for a background response nobody will read to be cancelled. Returns how that went, as a sentence."""
    openai = _sdk()
    try:
        responses.cancel(response_id, timeout=_timeout(openai, min(options["timeout_seconds"], CANCEL_SECONDS)))
    except openai.NotFoundError:
        return f"Background response {response_id} no longer exists, so there was nothing to cancel."
    except (openai.APIStatusError, openai.APIConnectionError) as e:
        why = f"HTTP {e.status_code}" if isinstance(e, openai.APIStatusError) else type(e).__name__
        outcome = (
            f"Background response {response_id} could NOT be cancelled ({why}) and may still be running and billed; "
            f"POST /v1/responses/{response_id}/cancel stops it."
        )
        log.warning("OpenAI: %s", outcome)  # the harness drops the message of an error it retries
        return outcome
    return f"Background response {response_id} was cancelled."


class OpenAIAdapter(Adapter):
    provider = PROVIDER

    def __init__(self, client: Any = None):
        super().__init__(client)
        self._lock = threading.Lock()

    def _client_for(self, options: dict[str, Any]) -> Any:
        """The client passed to the constructor, or one built once from the options of the first call."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = _make_client(options)
        return self._client

    def build_request(self, messages: list[Message], tools: list[ToolSpec], settings: ModelSettings) -> dict[str, Any]:
        """The keyword arguments of client.responses.create(), which are the JSON body sent."""
        options = read_options(settings)
        reasoning = read_reasoning(settings)
        max_output_tokens = read_max_output_tokens(settings)
        request: dict[str, Any] = {"model": settings.model, "input": _input_items(messages, options["image_detail"])}
        if tools:
            request["tools"] = [_tool(tool) for tool in tools]
        if reasoning:
            request["reasoning"] = reasoning
        request["max_output_tokens"] = max_output_tokens
        if options["include_encrypted_reasoning"]:
            request["include"] = [ENCRYPTED_REASONING]
        request["store"] = options["store"]
        if options["background"]:
            request["background"] = True
        return request

    def parse_response(self, response: Any) -> Reply:
        """A finished response (status "completed", "incomplete" or "failed") as a Reply.

        Raises for a response that failed for a reason other than a safety block, and for one
        that is not finished.
        """
        status = response.get("status")
        response_id = response.get("id")
        if status == "failed":
            error = response.get("error") or {}
            code, message = error.get("code"), error.get("message")
            if code in POLICY_CODES:
                return _blocked(code, message, f"failed:{code}", response, _usage(response))
            kind = TransientError if code in RETRYABLE_FAILURES else FatalError
            raise kind(f"OpenAI response {response_id} failed: {code}: {message}")
        if status not in ("completed", "incomplete"):
            kind = TransientError if status == "cancelled" else FatalError
            raise kind(f"OpenAI response {response_id} has status {status!r}, which is not a finished reply")

        # Find things by item type, never by position: reasoning, messages and calls come in any number and order (3.1).
        output = response.get("output") or []
        texts: list[str] = []
        refusals: list[str] = []
        calls: list[dict[str, Any]] = []
        for item in output:
            if item.get("type") == "message":
                parts = item.get("content") or []
                texts.append("".join(p.get("text") or "" for p in parts if p.get("type") == "output_text"))
                refusals += [p.get("refusal") or "" for p in parts if p.get("type") == "refusal"]
            elif item.get("type") == "function_call" and item.get("call_id"):
                calls.append(item)
        text = "\n\n".join(t for t in texts if t)  # an interim note and the answer are separate messages
        tool_calls = [ToolCall(id=c["call_id"], name=c.get("name"), arguments=_parsed_arguments(c)) for c in calls]

        # There is no stop reason field: it is the status, the reason a reply is incomplete, and what the output holds (3.5).
        reason = (response.get("incomplete_details") or {}).get("reason")
        refusal_text = None
        if reason == "content_filter":
            stop, refusal_text = STOP_REFUSAL, "\n".join(refusals) or "The provider's content filter stopped the reply."
        elif refusals and not tool_calls:
            stop, refusal_text = STOP_REFUSAL, "\n".join(refusals)
        elif status == "incomplete":
            stop = STOP_MAX_TOKENS if reason == "max_output_tokens" else STOP_OTHER
        else:
            stop = STOP_TOOL_CALLS if tool_calls else STOP_END

        return Reply(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop,
            raw_stop_reason=f"{status}:{reason}" if reason else status,
            usage=_usage(response),
            refusal_text=refusal_text,
            provider=PROVIDER,
            provider_payload=output,  # every item, exactly as it came, whether or not the reply finished
            raw=response,
        )

    def complete(self, messages: list[Message], tools: list[ToolSpec], settings: ModelSettings) -> Reply:
        options = read_options(settings)
        openai = _sdk()
        client = self._client_for(options)
        request = self.build_request(messages, tools, settings)
        try:
            response = _final_response(client, request, options)
        except (openai.APIStatusError, openai.APIConnectionError) as e:
            if isinstance(e, openai.APIStatusError) and e.code in POLICY_CODES:
                body = e.body if isinstance(e.body, dict) else {}
                raw = {"http_status": e.status_code, "error": e.body}
                return _blocked(e.code, body.get("message"), f"http_{e.status_code}:{e.code}", raw)
            raise _adapter_error(e, request) from e
        return self.parse_response(response)
