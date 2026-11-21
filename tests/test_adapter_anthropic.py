"""The Anthropic adapter, offline: every call goes through the real SDK, with mock transports for the network."""

from __future__ import annotations

import json

import httpx
import pytest

from paintbench.adapters import anthropic_api
from paintbench.adapters.anthropic_api import AnthropicAdapter, make_client, parse_options
from paintbench.adapters.base import FatalError, TransientError
from paintbench.prompts import system_prompt, task_message, write_rows_tool
from paintbench.types import STOP_END, STOP_MAX_TOKENS, STOP_REFUSAL, STOP_TOOL_CALLS, ModelSettings, system, tool_result, user

API = "https://api.anthropic.com"
GATEWAY = "http://127.0.0.1:9/a-gateway"
KEY = "sk-ant-api03-" + "Not-A-Real-Key_0" * 5
KEY_AUTH = {"type": "api_key", "env": "PAINTBENCH_TEST_KEY"}
FEDERATION = {"type": "wif", "org": "research"}
IDENTITY_TOKEN = "the-google-identity-token-of-this-machine"
TASK = task_message("Under the Wave off Kanagawa (often called The Great Wave), by Katsushika Hokusai, about 1831.", 16, 11, 16)
REMINDER = "The picture is not finished. Missing rows: 1-11. Send them with write_rows."

THINKING = {"type": "thinking", "thinking": "Sky first, then the crest.", "signature": "opaque-signature"}
ROWS_A = {"palette": ["k 20 40 90"], "start_row": 1, "rows": ["kkkk"]}
ROWS_B = {"start_row": 2, "rows": ["kkkk"]}
CALL_A = {"type": "tool_use", "id": "toolu_A", "name": "write_rows", "input": ROWS_A}
CALL_B = {"type": "tool_use", "id": "toolu_B", "name": "write_rows", "input": ROWS_B}


def response(stop_reason: str, *content: dict, **more) -> dict:
    usage = {"input_tokens": 1042, "output_tokens": 3100, "output_tokens_details": {"thinking_tokens": 2900}}
    message = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-fable-5-2", "content": list(content)}
    return {**message, "stop_reason": stop_reason, "usage": usage, **more}


TWO_CALLS = response("tool_use", THINKING, CALL_A, CALL_B)
DONE = response("end_turn", {"type": "text", "text": "All rows are written."})
BLOCKED = response("refusal", THINKING, CALL_A, stop_details={"type": "refusal", "category": "cyber", "explanation": "This request was blocked."})
CUT_OFF = response("max_tokens", THINKING)


def stream_of(message: dict) -> list[dict]:
    """A whole response as the stream events the API sends for it."""
    first = {**message, "content": [], "stop_reason": None, "usage": {"input_tokens": message["usage"]["input_tokens"], "output_tokens": 1}}
    first.pop("stop_details", None)
    events = [{"type": "message_start", "message": first}]
    for index, block in enumerate(message["content"]):
        if block["type"] == "thinking":
            empty = {**block, "thinking": "", "signature": ""}
            deltas = [{"type": "thinking_delta", "thinking": block["thinking"]}, {"type": "signature_delta", "signature": block["signature"]}]
        elif block["type"] == "text":
            empty, deltas = {**block, "text": ""}, [{"type": "text_delta", "text": block["text"]}]
        else:
            empty, deltas = {**block, "input": {}}, [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        events.append({"type": "content_block_start", "index": index, "content_block": empty})
        events += [{"type": "content_block_delta", "index": index, "delta": delta} for delta in deltas]
        events.append({"type": "content_block_stop", "index": index})
    ending = {key: message[key] for key in ("stop_reason", "stop_details") if key in message}
    last_usage = {key: value for key, value in message["usage"].items() if key != "input_tokens"}
    return events + [{"type": "message_delta", "delta": ending, "usage": last_usage}, {"type": "message_stop"}]


class Played:
    """A mock transport. Request N gets reply N: a response (sent as its stream), a list of stream events, or a
    ready httpx.Response. Every request is kept."""

    def __init__(self, *replies):
        self.replies, self.requests = replies, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies[len(self.requests) - 1]
        if isinstance(reply, httpx.Response):
            return reply
        events = reply if isinstance(reply, list) else stream_of(reply)
        body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]


def adapter_playing(*replies, auth: dict = KEY_AUTH) -> tuple[AnthropicAdapter, Played]:
    """The adapter over the client it builds for itself, with `replies` in place of the network."""
    played = Played(*replies)
    client = make_client(parse_options({"auth": auth}))
    client._client = httpx.Client(transport=httpx.MockTransport(played))
    if client.credentials is not None:  # federation: the token exchange has an HTTP client of its own
        client.credentials._http_client = httpx.Client(transport=httpx.MockTransport(played))
    return AnthropicAdapter(client=client), played


def settings(**overrides) -> ModelSettings:
    entry = {"name": "fable", "provider": "anthropic", "model": "claude-fable-5-2", "options": {"auth": KEY_AUTH}}
    return ModelSettings.from_dict({**entry, **overrides})


def start() -> list:
    return [system(system_prompt()), user(TASK)]


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    """The key under the name the config gives, and a gateway under the name the SDK reads by itself."""
    monkeypatch.setenv("PAINTBENCH_TEST_KEY", KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", GATEWAY)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def federation(monkeypatch, tmp_path):
    """A stand-in for ~/.anthropic and for Google: the ids under the name `research`, the same ids in a file one
    directory up, and an identity token."""
    ids = "".join(f"{key}=id-for-tests\n" for key in anthropic_api.WIF_ID_KEYS)
    home = tmp_path / "anthropic"
    (home / "wif-x").mkdir(parents=True)
    (home / "wif-research.env").write_text(ids)
    (tmp_path / "outside.env").write_text(ids)
    assert (home / "wif-x/../../outside.env").is_file()  # what the name x/../../outside would reach
    monkeypatch.setattr(anthropic_api, "WIF_DIR", str(home))
    monkeypatch.setattr(anthropic_api, "google_identity_token", lambda: IDENTITY_TOKEN)


def test_the_contestant_request_is_the_prompt_one_tool_and_the_reasoning_settings_with_no_image():
    adapter, played = adapter_playing(DONE)
    tool = write_rows_tool()
    adapter.complete(start(), [tool], settings(reasoning={"thinking": "adaptive", "effort": "max", "display": "summarized"}))
    assert str(played.requests[0].url) == f"{API}/v1/messages"  # not the gateway that ANTHROPIC_BASE_URL names
    assert played.bodies[0] == {
        "model": "claude-fable-5-2",
        "max_tokens": 128000,
        "system": system_prompt(),
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "max"},
        "tools": [{"name": "write_rows", "description": tool.description, "input_schema": tool.parameters}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": TASK}]}],
        "stream": True,
    }
    default = adapter.build_request(start(), [tool], settings())
    assert "thinking" not in default and "output_config" not in default  # no reasoning setting: the provider's default
    with pytest.raises(FatalError):  # a misspelt setting is refused, not run at the default
        adapter.build_request(start(), [tool], settings(reasoning={"efort": "max"}))


def test_a_reply_with_tool_calls_is_read_back_and_answered_in_the_next_request():
    adapter, played = adapter_playing(TWO_CALLS, DONE)
    tool = write_rows_tool()
    reply = adapter.complete(start(), [tool], settings())
    assert (reply.stop_reason, reply.text) == (STOP_TOOL_CALLS, "")
    assert [(call.id, call.name, call.arguments) for call in reply.tool_calls] == [("toolu_A", "write_rows", ROWS_A), ("toolu_B", "write_rows", ROWS_B)]
    assert reply.usage.to_dict() == {"input_tokens": 1042, "output_tokens": 3100, "reasoning_tokens": 2900, "cached_input_tokens": None}

    results = [tool_result(call, f"result of {call.id}") for call in reply.tool_calls]
    final = adapter.complete(start() + [reply.as_message(), *results], [tool], settings())
    assert played.bodies[1]["messages"][1:] == [
        {"role": "assistant", "content": [THINKING, CALL_A, CALL_B]},  # as it came: the thinking with its signature
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_A", "content": "result of toolu_A"},
                {"type": "tool_result", "tool_use_id": "toolu_B", "content": "result of toolu_B"},
            ],
        },
    ]
    assert (final.stop_reason, final.text) == (STOP_END, "All rows are written.")


def test_a_block_is_a_refusal_a_broken_stream_is_no_reply_and_cut_off_thinking_goes_back():
    adapter, played = adapter_playing(BLOCKED, stream_of(TWO_CALLS)[:-1], CUT_OFF, DONE)
    tool = write_rows_tool()

    blocked = adapter.complete(start(), [tool], settings())
    assert (blocked.stop_reason, blocked.tool_calls) == (STOP_REFUSAL, [])  # the call made before the block is discarded
    assert "This request was blocked." in blocked.refusal_text

    with pytest.raises(TransientError):  # the stream stopped before its last event: the calls in it are not acted on
        adapter.complete(start(), [tool], settings())

    cut = adapter.complete(start(), [tool], settings())
    assert (cut.stop_reason, cut.text, cut.tool_calls) == (STOP_MAX_TOKENS, "", [])
    adapter.complete(start() + [cut.as_message(), user(REMINDER)], [tool], settings())
    assert played.bodies[3]["messages"][1:] == [
        {"role": "assistant", "content": [THINKING]},  # the next call goes on from the thinking that was cut off
        {"role": "user", "content": [{"type": "text", "text": REMINDER}]},
    ]


def test_an_api_key_is_read_only_from_the_named_variable_and_is_never_shown():
    invalid = httpx.Response(401, json={"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}})
    adapter, played = adapter_playing(DONE, invalid)
    reply = adapter.complete(start(), [], settings())
    with pytest.raises(FatalError) as refused:
        adapter.complete(start(), [], settings())
    sent = played.requests[0]
    assert sent.headers["x-api-key"] == KEY  # in this header and nowhere else
    assert KEY not in str(sent.url) + sent.content.decode() + json.dumps(reply.to_dict()) + str(refused.value)

    for auth in ({"type": "api_key", "env": KEY}, {"type": "wif", "org": KEY}, {"type": KEY}):  # pasted where a name belongs
        with pytest.raises(FatalError) as caught:
            parse_options({"auth": auth})
        assert KEY not in str(caught.value)

    with pytest.raises(FatalError):  # ANTHROPIC_API_KEY, which the SDK reads by itself, is no substitute
        make_client(parse_options({"auth": KEY_AUTH}), environ={"ANTHROPIC_API_KEY": KEY})


def test_federation_sends_the_identity_token_to_the_official_api_and_nowhere_else(federation):
    minted = httpx.Response(200, json={"access_token": "minted", "token_type": "Bearer", "expires_in": 3600})
    adapter, played = adapter_playing(minted, DONE, auth=FEDERATION)
    adapter.complete(start(), [], settings(options={"auth": FEDERATION}))
    exchange, call = played.requests
    assert str(exchange.url) == f"{API}/v1/oauth/token"  # not the gateway that ANTHROPIC_BASE_URL names
    assert json.loads(exchange.content)["assertion"] == IDENTITY_TOKEN
    assert str(call.url) == f"{API}/v1/messages" and call.headers["authorization"] == "Bearer minted"
    assert IDENTITY_TOKEN not in call.content.decode() + json.dumps(dict(call.headers))


@pytest.mark.parametrize(
    "auth, base_url, environ",
    [
        pytest.param(FEDERATION, API, {"ANTHROPIC_API_KEY": ""}, id="while-ANTHROPIC_API_KEY-is-set-even-empty"),
        pytest.param(FEDERATION, "https://gateway.example/anthropic", {}, id="towards-a-gateway"),
        pytest.param(FEDERATION, "https://api.anthropic.com.gateway.example", {}, id="towards-a-host-that-starts-like-the-api"),
        pytest.param(FEDERATION, "http://localhost:8080", {}, id="towards-a-local-address"),
        pytest.param({"type": "wif", "org": "x/../../outside"}, API, {}, id="under-a-name-that-leaves-the-directory"),
    ],
)
def test_federation_is_refused(federation, auth, base_url, environ):
    assert make_client(parse_options({"auth": FEDERATION, "base_url": API}), environ={}).credentials  # nothing wrong: a client is built
    with pytest.raises(FatalError):
        make_client(parse_options({"auth": auth, "base_url": base_url}), environ=environ)
