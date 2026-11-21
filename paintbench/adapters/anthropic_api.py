"""The Anthropic Messages API (Claude), through the `anthropic` SDK.

Written from docs/provider-notes/anthropic.md for SDK 0.121 (the `anthropic>=0.121,<1` line). Run
live on 20 November 2026 against the Claude models of the published run; the tests use mocked
transports only. Where a rule rests on what the live API did, the text says so and gives the date
("What is replayed").

Settings of a model entry
-------------------------
reasoning          An empty mapping sends neither field: the model runs at the provider's default.
  effort           low | medium | high | xhigh | max. Sent as output_config.effort.
  thinking         adaptive | enabled | disabled | between_tools. Sent as thinking.type.
                   Fable 5.2, Opus 5.5, Sonnet 5.5 and Haiku 5.5 think adaptively when the field is
                   absent; their highest setting is `thinking: adaptive` with `effort: max`. All
                   four reject `enabled`. Of the four only Haiku 5.5 accepts `disabled` and only
                   Sonnet 5.5 `between_tools`, and there neither goes with an effort above high.
  budget_tokens    An integer of at least 1024, sent as thinking.budget_tokens. Required with
                   `thinking: enabled` (older models: the 4.5 line) and not allowed otherwise.
  display          summarized | omitted. Sent as thinking.display; needs `thinking: adaptive` or
                   `enabled`. The provider's default is omitted: thinking blocks arrive empty.
                   `summarized` puts a readable summary in the transcript and keeps bytes flowing
                   while the model thinks. It changes neither the reasoning nor the bill.
max_output_tokens  Sent as max_tokens. Default 128000, the ceiling of the four current models:
                   thinking counts against it, every call has its own, and an unused allowance
                   costs nothing and does not count towards rate limits. Opus 4.5, Sonnet 4.5 and
                   Haiku 4.5 stop at 64000 and need it set.
options
  auth             Required. There is no default and no falling back from one form to the other.
                     {type: api_key, env: NAME}
                         The key is read from the environment variable NAME. It never goes in
                         the config, because the config is saved with every attempt.
                     {type: wif, org: NAME}
                         Workload Identity Federation from a Google Cloud machine: no key is
                         stored anywhere. NAME is a short name of your choosing, for example
                         `research`: lowercase letters, digits, '-' and '_', starting with a
                         letter or digit, at most 32 characters. The ids are read from the file
                         ~/.anthropic/wif-NAME.env, one KEY=VALUE per line, and handed to the SDK:
                             ANTHROPIC_FEDERATION_RULE_ID_GCP    the federation rule
                             ANTHROPIC_ORGANIZATION_ID           the organization
                             ANTHROPIC_SERVICE_ACCOUNT_ID        the service account
                             ANTHROPIC_WORKSPACE_ID              the workspace
                         The identity token is the Google one of the machine the run is on.
                         Refused while a variable named ANTHROPIC_API_KEY is present in the
                         environment, even an empty one. Needs google-auth and requests, which
                         the `anthropic` extra installs.
  base_url         Default https://api.anthropic.com. Always passed to the SDK, so an exported
                   ANTHROPIC_BASE_URL redirects neither the calls nor the token exchange.
                   With auth type wif nothing else is accepted: the SDK sends the token exchange,
                   and with it the machine's identity token, to whatever base_url names.
  timeout_seconds  Default 1800: the longest silence allowed on the connection. Connecting has 10 s.
  sdk_max_retries  Default 1. The harness owns backoff. 1 and not 0, because the SDK mints a fresh
                   federation token after a 401 only while a retry is left.
  one_call_per_reply
                   Default false. With true, a reply may hold at most one tool call
                   (tool_choice {type: auto, disable_parallel_tool_use: true}): the model sends
                   one strip, gets its result, and goes on, and cannot send every strip of a
                   picture in one reply. It changes what the model is asked, so unlike the four
                   options above it is a condition of a run. The tool itself is unchanged.
A key that is not listed, or a value that is not allowed, is a FatalError before any request.

One call
--------
Every call is streamed with client.messages.stream() and the reply is the final message the SDK
assembles: the SDK refuses a non-streaming call above 21,333 max_tokens, and idle connections get
dropped. Nothing caps the length of a call. It ends when the model stops, when max_tokens is
reached, when the server gives up, or when no byte arrives for timeout_seconds; the harness looks
at its own time limit only between calls. A reply counts only when its stream ran to the
message_stop event: one that stops sooner, even after the stop reason arrived, is a failed call
and no tool call in it is returned. A stream that breaks cannot be resumed: the call fails with a
TransientError and the harness makes it again.

One client is built per distinct `options` on first use and shared by every thread after that.

What is replayed
----------------
Reply.provider_payload is the assistant turn as the API wants it back, {"role": "assistant",
"content": [...]}: every content block of the response in order, which means thinking blocks with
their signatures (also when their text is empty), redacted_thinking, text and tool_use blocks. An
assistant message of this provider is sent as exactly that payload. Two changes are made when the
payload is built and none later: keys the SDK filled with null are dropped, and blocks the API
does not take back are left out. Those are a text block that is empty or only whitespace, and
the reasoning at the end of a turn (the thinking and redacted_thinking blocks after its last
text or tool_use block) when one of its thinking blocks has no signature, or an empty one: the
reply was cut off inside that block. Such a block can never be verified; the API allows the
newest thinking to be left out, and never a thinking block between two that are kept. So that
reasoning goes back whole or not at all, never in part. A thinking block without a signature
that a text or tool_use block does follow goes back as it came, and it is the API's to refuse.

Thinking that carries its signature goes back wherever it sits, and so does redacted_thinking:
between calls, as the last block of a turn, and as the only block of one. A reply cut off by the
output limit while the model was still thinking is one signed thinking block and nothing else.
That turn is sent as it came and the user message after it continues from it, which is the
documented way to continue a truncated turn: pass it back unchanged and append a user message.

Leaving such a turn out costs an attempt all it has reasoned. A live run on 20 November 2026 did
leave it out: 55 attempts spent all 128,000 output tokens of each of their four calls thinking
and were cut off, every cut-off block had its signature, and each next call was given the task
and the reminder only. A live check on the same day then sent such a turn back (one signed
thinking block, cut off by max_tokens, followed by the reminder as a user message) on
claude-haiku-5-5, claude-fable-5-2, and claude-sonnet-5. The API accepted it on all three and
counted the earlier thinking in the input: about 2,535 tokens more, for 2,500 tokens of thinking
and the reminder. That shape is the one that was observed. A turn that ends in signed thinking
after a text or tool_use block, and a turn of redacted_thinking only, go back on the documented
rule alone. Replayed thinking is billed as input on every later call: about 128,000 tokens for
each reply that was cut off at the default limit.

A turn in which nothing can be sent back (no text, no tool_use, no signed thinking, no
redacted_thinking: an empty reply, or one cut off before a signature arrived) is not sent at
all, in any request, and the user message after it then follows the one before it; the API reads
two user messages in a row as one turn. A conversation whose last message is an assistant turn
that would be sent is refused with a FatalError before any request: current models reject it as
a prefill. The harness and the judge never make one; they put the tool results or a user message
after every reply they go on from, and stop at a refusal, so a refused reply is never replayed.

An assistant turn from anywhere else is rebuilt as a text block and one tool_use block per call.
The tool results that follow a turn go into one user message, in the order of its tool_use
blocks. The system prompt, the tool and all earlier messages come out the same on every call of
a conversation; the API checks that they do.

Reply.raw is the final message as JSON plus `_stream`: the request id, and for each tool call
whose streamed arguments are not valid JSON (a reply cut off by the output limit) the raw argument
string. That string becomes the call's arguments, so the canvas rejects it, while the payload keeps
the object the SDK made of it, because the API takes only an object back. Whether the API accepts
such a cut-off turn again is not documented; if it does not, the harness discards the conversation.

Errors
------
TransientError: rate limits (with retry_after when the API names one), overload, 5xx, 408 and 409,
timeouts, connection errors, a stream that breaks, ends before message_stop or cannot be read, and
a token exchange or Google token fetch that failed for a reason that may pass. FatalError: a bad
request, authentication, permission, an unknown model, a request that is too large, a spend cap,
a token exchange that was refused, and whatever the SDK raises before a reply has started.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import re
import threading
from collections.abc import Mapping
from pathlib import Path
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

PROVIDER = "anthropic"
INSTALL_HINT = 'pip install "paintbench[anthropic]"'

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_MAX_TOKENS = 128_000
DEFAULT_TIMEOUT_SECONDS = 1800.0
CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_SDK_MAX_RETRIES = 1

REASONING_KEYS = ("effort", "thinking", "budget_tokens", "display")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
THINKING_TYPES = ("adaptive", "enabled", "disabled", "between_tools")
DISPLAYS = ("summarized", "omitted")
OPTION_KEYS = ("auth", "base_url", "timeout_seconds", "sdk_max_retries", "one_call_per_reply")
AUTH_FORMS = (
    "{type: api_key, env: NAME_OF_AN_ENVIRONMENT_VARIABLE}, or {type: wif, org: NAME} for Workload Identity "
    "Federation from a Google Cloud machine, with the ids in ~/.anthropic/wif-NAME.env"
)

WIF_DIR = "~/.anthropic"  # holds wif-<org>.env, one KEY=VALUE per line
# What options.auth.org may be. It becomes part of a file name, so it holds nothing that could name another
# directory, and it is shorter than an API key, so a key pasted here by mistake is refused and never shown.
WIF_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
WIF_NAME_RULE = "lowercase letters, digits, '-' and '_', starting with a letter or digit, at most 32 characters"
WIF_AUDIENCE = "https://api.anthropic.com"  # what the federation rule expects in the Google token
WIF_ID_KEYS = (
    "ANTHROPIC_FEDERATION_RULE_ID_GCP",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
    "ANTHROPIC_WORKSPACE_ID",
)
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_STOP = {
    "end_turn": STOP_END,
    "tool_use": STOP_TOOL_CALLS,
    "max_tokens": STOP_MAX_TOKENS,
    "model_context_window_exceeded": STOP_MAX_TOKENS,  # generation filled the context window: also cut off
    "refusal": STOP_REFUSAL,
}
# An error that arrives inside a stream has no HTTP status of its own, only an `error.type`. These
# types say the request itself is at fault; every other one (overloaded_error, api_error,
# timeout_error, rate_limit_error, or none that can be read) is worth another try.
_REQUEST_FAULTS = (
    "invalid_request_error",
    "authentication_error",
    "billing_error",
    "permission_error",
    "not_found_error",
    "request_too_large",
)


# ---------------------------------------------------------------- settings


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_keys(section: str, mapping: Any, allowed: tuple[str, ...]) -> None:
    if not isinstance(mapping, Mapping):
        raise FatalError(f"anthropic: {section} must be a mapping, not {type(mapping).__name__}")
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        raise FatalError(f"anthropic: unknown key in {section}: {', '.join(unknown)}. Allowed: {', '.join(allowed)}")


def _choice(mapping: Mapping[str, Any], key: str, allowed: tuple[str, ...]) -> str | None:
    if key not in mapping:
        return None
    if mapping[key] not in allowed:
        raise FatalError(f"anthropic: reasoning.{key} is {mapping[key]!r}. Allowed: {', '.join(allowed)}")
    return mapping[key]


def reasoning_fields(reasoning: Mapping[str, Any] | None) -> dict[str, Any]:
    """The request fields for a `reasoning` mapping: `thinking`, `output_config`, both or neither."""
    reasoning = {} if reasoning is None else reasoning
    _check_keys("reasoning", reasoning, REASONING_KEYS)
    effort = _choice(reasoning, "effort", EFFORTS)
    thinking = _choice(reasoning, "thinking", THINKING_TYPES)
    display = _choice(reasoning, "display", DISPLAYS)
    budget = reasoning.get("budget_tokens")
    if "budget_tokens" in reasoning:
        if not _is_int(budget) or budget < 1024:
            raise FatalError(f"anthropic: reasoning.budget_tokens is {budget!r}. Allowed: an integer of at least 1024")
        if thinking != "enabled":
            raise FatalError("anthropic: reasoning.budget_tokens is allowed only with reasoning.thinking: enabled")
    elif thinking == "enabled":
        raise FatalError("anthropic: reasoning.thinking: enabled needs reasoning.budget_tokens (an integer of at least 1024)")
    if display is not None and thinking not in ("adaptive", "enabled"):
        raise FatalError("anthropic: reasoning.display needs reasoning.thinking set to adaptive or enabled")

    fields: dict[str, Any] = {}
    if thinking is not None:
        fields["thinking"] = {"type": thinking}
        if budget is not None:
            fields["thinking"]["budget_tokens"] = budget
        if display is not None:
            fields["thinking"]["display"] = display
    if effort is not None:
        fields["output_config"] = {"effort": effort}
    return fields


def _official_endpoint(auth_type: Any, base_url: Any) -> str:
    """base_url as it goes to the SDK. With wif it has to be the API itself.

    The SDK posts the token exchange to `<base_url>/v1/oauth/token` and checks only that the URL is
    https (or a local address). The exchange carries the machine's Google identity token, which
    anyone who receives it can trade for an access token of the organization. So nothing but the
    exact address is accepted: no other host, port or path, and no reading of the URL that a
    parser could be talked out of.
    """
    if auth_type != "wif":
        return base_url
    if base_url not in (DEFAULT_BASE_URL, DEFAULT_BASE_URL + "/"):
        raise FatalError(
            f"anthropic: options.base_url must be {DEFAULT_BASE_URL} with options.auth.type wif (leave it out); it is "
            f"{base_url!r}. The token exchange goes to base_url and carries the machine's identity token, so "
            "federation is never pointed at a gateway or any other host"
        )
    return DEFAULT_BASE_URL


def _is_wif_name(name: Any) -> bool:
    return isinstance(name, str) and WIF_NAME.fullmatch(name) is not None


def parse_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Check `options` and fill in the defaults. Reads no file and no environment variable."""
    options = {} if options is None else options
    _check_keys("options", options, OPTION_KEYS)

    auth = options.get("auth")
    if not isinstance(auth, Mapping) or auth.get("type") not in ("wif", "api_key"):
        raise FatalError(f"anthropic: options.auth is required and has no default. Use {AUTH_FORMS}")
    if auth["type"] == "wif":
        _check_keys("options.auth", auth, ("type", "org"))
        # Values under auth are never shown: someone may have pasted the key itself here.
        if not _is_wif_name(auth.get("org")):
            raise FatalError(
                f"anthropic: options.auth.org must be a short name of your choosing ({WIF_NAME_RULE}). "
                f"The federation ids are read from {WIF_DIR}/wif-<org>.env"
            )
    else:
        _check_keys("options.auth", auth, ("type", "env"))
        if not isinstance(auth.get("env"), str) or not _ENV_NAME.fullmatch(auth["env"]):
            raise FatalError("anthropic: options.auth.env must be the name of the environment variable that holds the key")

    base_url = options.get("base_url", DEFAULT_BASE_URL)
    if not isinstance(base_url, str) or not base_url:
        raise FatalError(f"anthropic: options.base_url is {base_url!r}. Allowed: a URL such as {DEFAULT_BASE_URL}")
    base_url = _official_endpoint(auth["type"], base_url)
    timeout = options.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if not (_is_int(timeout) or isinstance(timeout, float)) or not timeout > 0:
        raise FatalError(f"anthropic: options.timeout_seconds is {timeout!r}. Allowed: a number of seconds above 0")
    retries = options.get("sdk_max_retries", DEFAULT_SDK_MAX_RETRIES)
    if not _is_int(retries) or retries < 0:
        raise FatalError(f"anthropic: options.sdk_max_retries is {retries!r}. Allowed: an integer of 0 or more")
    one_call = options.get("one_call_per_reply", False)
    if not isinstance(one_call, bool):
        raise FatalError(f"anthropic: options.one_call_per_reply is {one_call!r}. Allowed: true or false")
    # one_call_per_reply is checked here and read by build_request: it shapes a request, and is not one of the
    # settings a client is built from, so it is not returned.
    return {"auth": dict(auth), "base_url": base_url, "timeout_seconds": float(timeout), "sdk_max_retries": retries}


# ---------------------------------------------------------------- the client


def _sdk() -> tuple[Any, Any]:
    try:
        import anthropic
        import httpx
    except ImportError as e:
        raise FatalError(f"anthropic: the anthropic SDK is not installed. Run: {INSTALL_HINT}") from e
    return anthropic, httpx


def wif_ids(org: str) -> dict[str, str]:
    """The federation ids kept under the name `org` (options.auth.org), read from ~/.anthropic/wif-<org>.env."""
    if not _is_wif_name(org):  # checked again: the name is part of the path, and no other directory is ever read
        raise FatalError(f"anthropic: options.auth.org must be a short name ({WIF_NAME_RULE})")
    path = Path(WIF_DIR).expanduser() / f"wif-{org}.env"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise FatalError(f"anthropic: cannot read the federation ids named {org!r} (options.auth.org) from {path}: {e.strerror or e}") from e
    ids: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            ids[key.strip()] = value.strip()
    missing = [key for key in WIF_ID_KEYS if not ids.get(key)]
    if missing:
        raise FatalError(f"anthropic: {path} has no value for {', '.join(missing)}")
    return ids


def google_identity_token() -> str:
    """The Google identity token of the machine this runs on, for the audience the federation rule expects."""
    import google.auth.transport.requests
    import google.oauth2.id_token

    return google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), WIF_AUDIENCE)


def make_client(options: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> Any:
    """Build the SDK client from parse_options() output. Opens no connection.

    Credentials and base_url are always passed explicitly, so the SDK reads neither
    ANTHROPIC_API_KEY nor ANTHROPIC_BASE_URL. Nothing is written to the environment.
    One variable the SDK does still act on is ANTHROPIC_CUSTOM_HEADERS: when it is set, its
    headers go out with every request, and an Authorization or X-Api-Key header among them
    takes the place of the federation token.
    """
    anthropic, httpx = _sdk()
    environ = os.environ if environ is None else environ
    auth = options["auth"]
    common = {
        "base_url": _official_endpoint(auth["type"], options["base_url"]),  # checked again: before any token is fetched
        "timeout": httpx.Timeout(options["timeout_seconds"], connect=CONNECT_TIMEOUT_SECONDS),
        "max_retries": options["sdk_max_retries"],
    }
    try:
        if auth["type"] == "api_key":
            key = environ.get(auth["env"])
            if not key:
                raise FatalError(f"anthropic: the environment variable {auth['env']} (options.auth.env) is not set or is empty")
            return anthropic.Anthropic(api_key=key, **common)

        if "ANTHROPIC_API_KEY" in environ:  # presence only: its value is never read
            raise FatalError(
                "anthropic: options.auth.type is wif, but a variable named ANTHROPIC_API_KEY is present in the "
                "environment. It overrides federation even when empty: unset it and run again"
            )
        try:
            import google.auth.transport.requests
            import google.oauth2.id_token  # noqa: F401
        except ImportError as e:
            raise FatalError(f"anthropic: options.auth.type wif needs google-auth and requests. Run: {INSTALL_HINT}") from e
        ids = wif_ids(auth["org"])

        def identity_token() -> str:
            # Raised as the SDK's own error so that the SDK neither retries it nor reports it as
            # "Connection error", and keeps serving a token that is still valid.
            try:
                return google_identity_token()
            except Exception as e:
                raise anthropic.WorkloadIdentityError(
                    f"could not get the machine's Google identity token: {type(e).__name__}: {e}"
                ) from e

        credentials = anthropic.WorkloadIdentityCredentials(
            identity_token_provider=identity_token,
            federation_rule_id=ids["ANTHROPIC_FEDERATION_RULE_ID_GCP"],
            organization_id=ids["ANTHROPIC_ORGANIZATION_ID"],
            service_account_id=ids["ANTHROPIC_SERVICE_ACCOUNT_ID"],
            workspace_id=ids["ANTHROPIC_WORKSPACE_ID"],
        )
        return anthropic.Anthropic(credentials=credentials, **common)
    except AdapterError:
        raise
    except Exception as e:  # e.g. a base_url the SDK will not use
        raise FatalError(f"anthropic: could not build the client: {type(e).__name__}: {e}") from e


# ---------------------------------------------------------------- request


def _max_tokens(value: Any) -> int:
    if value is None:
        return DEFAULT_MAX_TOKENS
    if not _is_int(value) or value < 1:
        raise FatalError(f"anthropic: max_output_tokens is {value!r}. Allowed: a positive integer, or nothing for {DEFAULT_MAX_TOKENS}")
    return value


def _has_text(text: Any) -> bool:
    """Whether an assistant text block can be sent back: not when it is empty or only whitespace."""
    return isinstance(text, str) and bool(text.strip())


def _is_signed(block: Mapping[str, Any]) -> bool:
    """Whether a thinking block carries its signature. One that does not can never be verified."""
    signature = block.get("signature")
    return isinstance(signature, str) and bool(signature)


def _can_go_back(block: Mapping[str, Any]) -> bool:
    """Whether a block gives the API something to take back, so that a turn holding it is sent."""
    kind = block.get("type")
    return kind in ("text", "tool_use", "redacted_thinking") or (kind == "thinking" and _is_signed(block))


def _user_content(message: Message) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, Image):
            source = {"type": "base64", "media_type": part.media_type, "data": part.base64()}
            blocks.append({"type": "image", "source": source})
        elif part.text:  # the API rejects an empty text block
            blocks.append({"type": "text", "text": part.text})
    if not blocks:
        raise FatalError("anthropic: a user message with no text and no image cannot be sent")
    return blocks


def _assistant_turn(message: Message) -> dict[str, Any]:
    """The assistant message for the request: our own payload verbatim, or rebuilt from text and calls."""
    if message.provider == PROVIDER and message.provider_payload is not None:
        payload = message.provider_payload
        if not isinstance(payload, dict) or payload.get("role") != "assistant" or not isinstance(payload.get("content"), list):
            raise FatalError("anthropic: the provider_payload of an assistant message is not {role: assistant, content: [...]}")
        return copy.deepcopy(payload)
    content: list[dict[str, Any]] = []
    if _has_text(message.text):
        content.append({"type": "text", "text": message.text})
    for call in message.tool_calls:
        # `input` must be an object; arguments that never parsed are sent as an empty one.
        arguments = copy.deepcopy(call.arguments) if isinstance(call.arguments, dict) else {}
        content.append({"type": "tool_use", "id": call.id, "name": call.name, "input": arguments})
    return {"role": "assistant", "content": content}


def _tool_results(run: list[Message], call_ids: list[str]) -> list[dict[str, Any]]:
    """One tool_result block per tool message, in the order of the turn's tool_use blocks."""
    position = {call_id: n for n, call_id in enumerate(call_ids)}
    blocks = []
    for message in sorted(run, key=lambda m: position.get(m.tool_call_id, len(position))):
        block: dict[str, Any] = {"type": "tool_result", "tool_use_id": message.tool_call_id}
        if message.text:  # an empty result is expressed by leaving `content` out
            block["content"] = message.text
        blocks.append(block)
    return blocks


def _wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    call_ids: list[str] = []  # the tool_use ids of the latest assistant turn
    for role, group in itertools.groupby(messages, key=lambda m: m.role):
        run = list(group)
        if role == "user":
            wire += [{"role": "user", "content": _user_content(m)} for m in run]
        elif role == "tool":  # every result of one turn goes into a single user message
            wire.append({"role": "user", "content": _tool_results(run, call_ids)})
        elif role == "assistant":
            for message in run:
                turn = _assistant_turn(message)
                call_ids = [block.get("id") for block in turn["content"] if block.get("type") == "tool_use"]
                # A turn with nothing to take back is left out, here and in every later request. One that
                # holds signed thinking and nothing else is sent: the user message after it continues from it.
                if any(_can_go_back(block) for block in turn["content"]):
                    wire.append(turn)
        elif role == "system":
            raise FatalError("anthropic: only a leading system message is supported")
        else:
            raise FatalError(f"anthropic: unknown message role {role!r}")
    if wire and wire[-1]["role"] == "assistant":
        # Whether a turn is sent never depends on what follows it, so it is not dropped here: a turn that
        # is sent in one request and not in another changes the earlier messages, which the API checks.
        raise FatalError(
            "anthropic: the conversation ends with an assistant turn, which current models reject as a prefill. "
            "The tool results or a user message must follow it"
        )
    return wire


# ---------------------------------------------------------------- response


def _parses(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _replayable(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The response's content blocks in the form the API takes back."""
    kept = []
    for block in blocks:
        block = {key: copy.deepcopy(value) for key, value in block.items() if value is not None}  # top level only
        if block.get("type") == "text" and not _has_text(block.get("text")):
            continue
        kept.append(block)
    # The reasoning at the end of a turn (the thinking and redacted_thinking blocks after its last text or
    # tool_use block) goes back whole or not at all. Whole when every thinking block in it has its
    # signature: that is the turn a model cut off while thinking has to get back (see "What is
    # replayed"). Not at all when one has none: the reply was cut off inside that block, it can never be
    # verified, the newest thinking is the only thinking that may be left out, and a part of a run of
    # reasoning has never been sent.
    end = len(kept)
    while end and kept[end - 1].get("type") in ("thinking", "redacted_thinking"):
        end -= 1
    if any(block.get("type") == "thinking" and not _is_signed(block) for block in kept[end:]):
        del kept[end:]
    return kept


def _usage(usage: Mapping[str, Any]) -> Usage:
    details = usage.get("output_tokens_details") or {}
    return Usage(
        # input_tokens leaves out what was written to or read from the cache
        input_tokens=sum(usage.get(key) or 0 for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
        output_tokens=usage.get("output_tokens") or 0,  # thinking included
        reasoning_tokens=details.get("thinking_tokens"),
        cached_input_tokens=usage.get("cache_read_input_tokens"),
    )


def _refusal_text(details: Any) -> str:
    details = details if isinstance(details, dict) else {}
    text = details.get("explanation") or "The provider declined this request."
    return f"{text} (category: {details['category']})" if details.get("category") else text


# ---------------------------------------------------------------- errors


def _retry_after(headers: Any) -> float | None:
    try:
        seconds = float(headers.get("retry-after"))
    except (TypeError, ValueError):  # absent, or not a number of seconds
        return None
    return seconds if seconds > 0 else None


def _from_status_error(e: Any) -> AdapterError:
    """Classify an anthropic.APIStatusError: an HTTP error status, or an `error` event inside a stream."""
    status, kind = e.status_code, e.type
    error = e.body.get("error") if isinstance(e.body, dict) else None
    details = error.get("details") if isinstance(error, dict) else None
    spend_cap = isinstance(details, dict) and details.get("error_code") == "enforced_spend_limit_reached"
    if status >= 400:
        message, status_or_none = f"anthropic: {type(e).__name__}: {e.message}", status
        transient = status in (408, 409, 429) or status >= 500  # 529 (overloaded) included
    else:  # the stream had started: the HTTP status is 200 and the kind of error is in `type`
        message, status_or_none = f"anthropic: error event in the stream: {e.message}", None
        transient = kind not in _REQUEST_FAULTS
    if transient and not spend_cap:  # a spend cap is a 429 that no retry gets past
        return TransientError(message, status=status_or_none, retry_after=_retry_after(e.response.headers))
    return FatalError(message, status=status_or_none)


def _from_identity_error(e: Any) -> AdapterError:
    """Classify an anthropic.WorkloadIdentityError: the token exchange, or the Google token before it."""
    message = f"anthropic: workload identity federation: {e}"
    if e.status_code is None or e.status_code == 429 or e.status_code >= 500:
        return TransientError(message, status=e.status_code)
    return FatalError(message, status=e.status_code)


# ---------------------------------------------------------------- the adapter


class AnthropicAdapter(Adapter):
    provider = PROVIDER

    def __init__(self, client: Any = None):
        super().__init__(client)
        self._lock = threading.Lock()
        self._clients: dict[str, Any] = {}  # the clients built here, by the options they were built from

    def build_request(self, messages: list[Message], tools: list[ToolSpec], settings: ModelSettings) -> dict[str, Any]:
        """The keyword arguments for client.messages.stream(), which adds `"stream": true` itself."""
        request: dict[str, Any] = {"model": settings.model, "max_tokens": _max_tokens(settings.max_output_tokens)}
        if messages and messages[0].role == "system":
            if messages[0].text:
                request["system"] = messages[0].text
            messages = messages[1:]
        request.update(reasoning_fields(settings.reasoning))
        if tools:
            request["tools"] = [
                {"name": tool.name, "description": tool.description, "input_schema": copy.deepcopy(tool.parameters)}
                for tool in tools
            ]
            if (settings.options or {}).get("one_call_per_reply") is True:
                request["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        request["messages"] = _wire_messages(messages)
        return request

    def parse_response(self, response: Any) -> Reply:
        """`response` is the final message as JSON, as complete() makes it (see "Reply.raw" above)."""
        blocks = response.get("content") or []
        unparsed = (response.get("_stream") or {}).get("unparsed_tool_input") or {}
        raw_stop = response.get("stop_reason")
        stop = _STOP.get(raw_stop, STOP_OTHER)
        refused = stop == STOP_REFUSAL  # whatever was produced before a refusal is to be discarded
        calls = [
            ToolCall(id=block.get("id"), name=block.get("name"), arguments=unparsed.get(str(index), copy.deepcopy(block.get("input"))))
            for index, block in enumerate(blocks)
            if block.get("type") == "tool_use"
        ]
        return Reply(
            text="" if refused else "".join(block.get("text") or "" for block in blocks if block.get("type") == "text"),
            tool_calls=[] if refused else calls,
            stop_reason=stop,
            raw_stop_reason=raw_stop,
            usage=_usage(response.get("usage") or {}),
            refusal_text=_refusal_text(response.get("stop_details")) if refused else None,
            provider=self.provider,
            provider_payload={"role": "assistant", "content": _replayable(blocks)},
            raw=response,
        )

    def complete(self, messages: list[Message], tools: list[ToolSpec], settings: ModelSettings) -> Reply:
        options = parse_options(settings.options)
        request = self.build_request(messages, tools, settings)
        anthropic, httpx = _sdk()
        client = self._client_for(options)

        tool_json: dict[int, str] = {}  # content index -> the argument string as it was streamed
        thinking_tokens = None
        started = finished = False
        try:
            with client.messages.stream(**request) as stream:
                started = True
                for event in stream:
                    if event.type == "content_block_delta" and getattr(event.delta, "type", None) == "input_json_delta":
                        tool_json[event.index] = tool_json.get(event.index, "") + event.delta.partial_json
                    elif event.type == "message_delta":
                        # The thinking share is only on this last event, and the SDK's assembled message loses it.
                        details = getattr(event.usage, "output_tokens_details", None)
                        thinking_tokens = getattr(details, "thinking_tokens", thinking_tokens)
                    elif event.type == "message_stop":
                        finished = True
                message = stream.get_final_message()
                request_id = stream.request_id
            data = message.to_dict(mode="json")
        except anthropic.APIStatusError as e:
            raise _from_status_error(e) from e
        except (anthropic.APIConnectionError, httpx.HTTPError) as e:  # before the response, or while reading it
            raise TransientError(f"anthropic: {type(e).__name__}: {e}") from e
        except anthropic.WorkloadIdentityError as e:
            raise _from_identity_error(e) from e
        except Exception as e:
            if started:  # an empty or malformed stream: the SDK raises AssertionError, AttributeError and the like
                raise TransientError(f"anthropic: the stream could not be read: {type(e).__name__}: {e}") from e
            raise FatalError(f"anthropic: the SDK failed before any reply started: {type(e).__name__}: {e}") from e

        # The SDK raises nothing when a stream just ends, and assembles whatever had arrived.
        if data.get("stop_reason") is None:
            raise TransientError("anthropic: the stream ended before the reply was finished (no stop_reason)")
        if not finished:  # the stop reason arrived, the closing event did not: not a reply to act on
            raise TransientError("anthropic: the stream ended before the reply was finished (no message_stop event)")
        if thinking_tokens is not None:
            data.setdefault("usage", {})["output_tokens_details"] = {"thinking_tokens": thinking_tokens}
        unparsed = {str(index): text for index, text in tool_json.items() if text and not _parses(text)}
        data["_stream"] = {"request_id": request_id, "unparsed_tool_input": unparsed}
        return self.parse_response(data)

    def _client_for(self, options: Mapping[str, Any]) -> Any:
        """The client passed to the constructor, or else one per distinct options, built on first use.

        The credentials are part of the options, so a call is never made with the client of other
        settings (other federation ids, another key, another endpoint). Threads that arrive together build one.
        """
        if self._client is not None:
            return self._client
        key = json.dumps(options, sort_keys=True)
        client = self._clients.get(key)
        if client is None:
            with self._lock:
                client = self._clients.get(key)
                if client is None:  # a failed build stores nothing: the next call tries again
                    client = self._clients[key] = make_client(options)
        return client
