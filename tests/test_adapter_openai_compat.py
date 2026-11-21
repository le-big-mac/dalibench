"""The "openai_compat" adapter (chat completions: OpenRouter, local servers), offline.

Every call goes through the real SDK over a mocked transport, so a request checked here is the one
the SDK would put on the wire. Nothing leaves the process and no real key is read.
"""

from __future__ import annotations

import json

import httpx
import pytest

from paintbench.adapters import openai_compat
from paintbench.adapters.base import FatalError, TransientError
from paintbench.adapters.openai_compat import OpenAICompatAdapter
from paintbench.prompts import system_prompt, task_message, write_rows_tool
from paintbench.types import STOP_END, STOP_MAX_TOKENS, STOP_REFUSAL, STOP_TOOL_CALLS, ModelSettings, system, tool_result, user

HOST = "example.invalid"
KEY = "test-key-not-a-real-one"
KEY_ENV = "PAINTBENCH_TEST_KEY"
MODEL = "x-ai/grok-4.7"
TOOL = write_rows_tool()
TASK = task_message("Under the Wave off Kanagawa, by Katsushika Hokusai, about 1831.", 32, 22, 16)
FIRST = [system(system_prompt()), user(TASK)]
CALL = {"id": "call_a", "type": "function", "function": {"name": "write_rows", "arguments": '{"start_row": 1, "rows": ["kkkk"]}'}}
CUT_CALL = {"id": "call_b", "type": "function", "function": {"name": "write_rows", "arguments": '{"start_row": 2, "rows": ["kk'}}


def settings(**options) -> ModelSettings:
    """The OpenRouter entry of configs/example.yaml, pointed at a host that does not exist."""
    return ModelSettings(name="grok", provider="openai_compat", model=MODEL, reasoning={"effort": "xhigh"}, max_output_tokens=128000,
                         options={"base_url": f"https://{HOST}/api/v1", "api_key_env": KEY_ENV, "stream_usage": False,
                                  "extra_body": {"provider": {"require_parameters": True}}, **options})


def stream(*deltas: dict, finish: str, usage: dict | None = None) -> bytes:
    """A streamed reply as the bytes a server sends: a chunk per delta, one more with the finish reason, then [DONE]."""
    choices = [{"index": 0, "delta": delta, "finish_reason": None} for delta in deltas] + [{"index": 0, "delta": {}, "finish_reason": finish}]
    chunks = [{"id": "gen-1", "object": "chat.completion.chunk", "model": MODEL, "choices": [choice]} for choice in choices]
    chunks[-1]["usage"] = usage
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode() + b"data: [DONE]\n\n"


def completion(finish: str, **message) -> dict:
    """A reply that is not streamed, with this finish reason and these keys of the assistant message."""
    return {"id": "gen-1", "object": "chat.completion", "model": MODEL,
            "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": None, **message}}],
            "usage": {"prompt_tokens": 800, "completion_tokens": 200, "total_tokens": 1000}}


def error(status: int, message: str, **metadata) -> tuple[int, dict]:
    return status, {"error": {"code": status, "message": message, "metadata": metadata}}


class Server:
    """An endpoint behind a mocked transport. It gives its answers in order and keeps every request.
    An answer is the bytes of an event stream, the JSON body of a 200, or (status, body) for an error."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, bytes):
            return httpx.Response(200, content=answer, headers={"content-type": "text/event-stream"})
        status, body = answer if isinstance(answer, tuple) else (200, answer)
        return httpx.Response(status, json=body)


@pytest.fixture
def connect(monkeypatch):
    """connect(server) -> an adapter whose own SDK client, built from the settings and the environment, reaches `server` alone."""
    monkeypatch.setenv(KEY_ENV, KEY)
    build_client = openai_compat.build_client

    def make(server: Server) -> OpenAICompatAdapter:
        http = httpx.Client(transport=httpx.MockTransport(server))
        monkeypatch.setattr(openai_compat, "build_client", lambda options: build_client(options, http_client=http))
        return OpenAICompatAdapter()

    return make


def test_contestant_request_holds_the_prompt_and_one_tool_and_no_image(connect):
    server = Server(stream({"content": "Done."}, finish="stop"))
    connect(server).complete(FIRST, [TOOL], settings())

    (request,) = server.requests
    assert (request.method, str(request.url)) == ("POST", f"https://{HOST}/api/v1/chat/completions")
    body = json.loads(request.content)
    assert body == {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_prompt()}, {"role": "user", "content": TASK}],
        "tools": [{"type": "function", "function": {"name": "write_rows", "description": TOOL.description, "parameters": TOOL.parameters}}],
        "max_tokens": 128000,
        "reasoning_effort": "xhigh",
        "stream": True,
        "provider": {"require_parameters": True},
    }  # and so no sampling parameter and no tool choice
    assert len(body["tools"]) == 1
    assert all(isinstance(message["content"], str) for message in body["messages"])  # plain text and nothing else: no image


def test_a_streamed_reply_is_read_back_whole_and_replayed_as_it_came(connect):
    signed = {"type": "reasoning.text", "text": "The wave is on the left.", "signature": "signed-by-the-provider", "index": 0}
    arguments = CALL["function"]["arguments"]
    body = stream(
        {"reasoning": signed["text"], "reasoning_details": [{**signed, "signature": None}]},
        {"reasoning_details": [{k: v for k, v in signed.items() if k != "text"}]},
        {"content": "Row 1."},
        {"tool_calls": [{"index": 0, "id": "call_a", "type": "function", "function": {"name": "write_rows", "arguments": ""}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": arguments[:12]}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": arguments[12:]}}]},
        finish="tool_calls",
        usage={"prompt_tokens": 812, "completion_tokens": 2304, "total_tokens": 3116, "completion_tokens_details": {"reasoning_tokens": 2101}},
    )
    adapter = connect(Server(body))
    reply = adapter.complete(FIRST, [TOOL], settings())

    assert (reply.text, reply.stop_reason) == ("Row 1.", STOP_TOOL_CALLS)
    assert [call.to_dict() for call in reply.tool_calls] == [{"id": "call_a", "name": "write_rows", "arguments": {"start_row": 1, "rows": ["kkkk"]}}]
    assert reply.usage.to_dict() == {"input_tokens": 812, "output_tokens": 2304, "reasoning_tokens": 2101, "cached_input_tokens": None}

    # The next request carries the assistant turn as the server sent it, signed reasoning included, then the result.
    request = adapter.build_request(FIRST + [reply.as_message(), tool_result(reply.tool_calls[0], "row 1: stored.")], [TOOL], settings())
    assert request["messages"][2:] == [
        {"role": "assistant", "content": "Row 1.", "reasoning": signed["text"], "reasoning_details": [signed], "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_a", "content": "row 1: stored."},
    ]

    # The same stream broken off before its end is not a reply: the call is to be made again.
    with pytest.raises(TransientError):
        connect(Server(body[: body.rindex(b"data: {")])).complete(FIRST, [TOOL], settings())


ENDINGS = {
    "it finishes with text": (completion("stop", content="Done."), STOP_END),
    "it calls the tool, though the server says stop": (completion("stop", tool_calls=[CALL]), STOP_TOOL_CALLS),
    "the output limit cuts it off inside a call": (completion("length", tool_calls=[CALL, CUT_CALL]), STOP_MAX_TOKENS),
    "the model refuses": (completion("stop", refusal="I can't help with that."), STOP_REFUSAL),
    "the content filter stops it": (completion("content_filter"), STOP_REFUSAL),
    "the provider blocks it with an HTTP 403": (error(403, "Flagged for possible risk.", error_type="refusal"), STOP_REFUSAL),
}


@pytest.mark.parametrize("name", list(ENDINGS))
def test_how_a_reply_ended_is_mapped(connect, name):
    answer, expected = ENDINGS[name]
    reply = connect(Server(answer)).complete(FIRST, [TOOL], settings(stream=False))
    assert reply.stop_reason == expected


def test_the_key_goes_to_the_configured_endpoint_alone_and_is_never_shown(connect, monkeypatch):
    for name, value in {"OPENAI_API_KEY": "leaked-key", "OPENAI_BASE_URL": "https://leaked.invalid/v1", "OPENAI_ORG_ID": "leaked-org",
                        "OPENAI_PROJECT_ID": "leaked-project", "OPENAI_CUSTOM_HEADERS": "X-Leaked: leaked-header"}.items():
        monkeypatch.setenv(name, value)  # what a machine set up for OpenAI itself holds: none of it is for this endpoint
    server = Server(error(401, "No auth credentials found"), completion("stop", content="Done."))
    adapter = connect(server)
    with pytest.raises(FatalError) as refused:
        adapter.complete(FIRST, [TOOL], settings(stream=False))
    reply = adapter.complete(FIRST, [TOOL], settings(stream=False))
    keyless = Server(completion("stop", content="Done."))  # a local server that needs no key gets a placeholder
    connect(keyless).complete(FIRST, [TOOL], settings(stream=False, api_key_env=None))

    sent = [(r.url.host, r.headers["authorization"]) for r in server.requests + keyless.requests]
    assert sent == [(HOST, f"Bearer {KEY}"), (HOST, f"Bearer {KEY}"), (HOST, "Bearer EMPTY")]
    for request in server.requests + keyless.requests:
        assert "leaked" not in json.dumps(dict(request.headers)).lower() + request.content.decode().lower()
    assert reply.text == "Done." and KEY not in str(refused.value) + json.dumps(reply.to_dict())
