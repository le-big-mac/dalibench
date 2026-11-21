"""The Google adapter, offline: every call goes through the real google-genai SDK, with a mock transport for the network."""

from __future__ import annotations

import json
import re

import httpx
import pytest

pytest.importorskip("google.genai")

from paintbench.adapters.base import FatalError, TransientError
from paintbench.adapters.google_genai import GoogleAdapter, make_client
from paintbench.prompts import system_prompt, task_message, write_rows_tool
from paintbench.types import STOP_END, STOP_MAX_TOKENS, STOP_REFUSAL, STOP_TOOL_CALLS, ModelSettings, system, tool_result, user

KEY_ENV = "PAINTBENCH_TEST_GEMINI_KEY"
KEY = "not-a-real-key-123"
TASK = task_message("Under the Wave off Kanagawa (often called The Great Wave), by Katsushika Hokusai, about 1831.", 16, 11, 16)

THOUGHT = {"text": "Sky first, then the crest.", "thought": True}
ROWS_A = {"palette": ["k 20 40 90"], "start_row": 1, "rows": ["kkkk"]}
ROWS_B = {"start_row": 2, "rows": ["kkkk"]}
CALL_A = {"functionCall": {"id": "fc_A", "name": "write_rows", "args": ROWS_A}, "thoughtSignature": "b3BhcXVl"}
CALL_B = {"functionCall": {"id": "fc_B", "name": "write_rows", "args": ROWS_B}}


def chunk(*parts: dict, finish: str | None = None) -> dict:
    """One chunk of a streamed reply: its new parts, and on the last chunk the finish reason."""
    candidate = {"index": 0, "content": {"role": "model", "parts": list(parts)}}
    if finish:
        candidate["finishReason"] = finish
    return {"candidates": [candidate], "usageMetadata": {"promptTokenCount": 412, "candidatesTokenCount": 60, "thoughtsTokenCount": 900}}


TWO_CALLS = [chunk(THOUGHT), chunk(CALL_A, CALL_B, finish="STOP")]
DONE = [chunk({"text": "All rows are written."}, finish="STOP")]
PROMPT_BLOCKED = [{"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}, "usageMetadata": {"promptTokenCount": 412}}]
REPLY_BLOCKED = [chunk(finish="SAFETY")]
CUT_OFF = [chunk({"text": "The painting is"}), chunk(finish="MAX_TOKENS")]


class Played:
    """A mock transport. Request N gets reply N: the chunks of a stream, or a ready httpx.Response. Every request is kept."""

    def __init__(self, *replies):
        self.replies, self.requests = replies, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies[len(self.requests) - 1]
        if isinstance(reply, httpx.Response):
            return reply
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in reply)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]


def adapter_playing(*replies) -> tuple[GoogleAdapter, Played]:
    """The adapter over the client it builds for itself, with `replies` in place of the network."""
    played = Played(*replies)
    return GoogleAdapter(client=make_client({"api_key_env": KEY_ENV}, transport=httpx.MockTransport(played))), played


def settings(**overrides) -> ModelSettings:
    entry = {"name": "gem", "provider": "google", "model": "gemini-4-carbon", "options": {"api_key_env": KEY_ENV}}
    return ModelSettings.from_dict({**entry, **overrides})


def start() -> list:
    return [system(system_prompt()), user(TASK)]


def camel(mapping: dict) -> dict:
    """A mapping with its keys as the API reference spells them. The SDK writes some in snake_case, which the API also takes."""
    return {re.sub(r"_(.)", lambda match: match[1].upper(), key): value for key, value in mapping.items()}


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    """The key under the name the config gives, and another key and another host under the names the SDK reads by itself."""
    monkeypatch.setenv(KEY_ENV, KEY)
    monkeypatch.setenv("GOOGLE_API_KEY", "the-key-the-sdk-would-pick")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "http://127.0.0.1:9/elsewhere")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")


def test_the_contestant_request_is_the_prompt_one_tool_and_the_thinking_level_with_no_image():
    adapter, played = adapter_playing(DONE)
    tool = write_rows_tool()
    adapter.complete(start(), [tool], settings(reasoning={"effort": "high"}))
    url = played.requests[0].url  # not the host that GOOGLE_GEMINI_BASE_URL names
    assert (url.host, url.path) == ("generativelanguage.googleapis.com", "/v1beta/models/gemini-4-carbon:streamGenerateContent")
    (body,) = played.bodies
    assert set(body) == {"systemInstruction", "contents", "tools", "generationConfig"}  # no tool choice, nothing of our own
    assert body["systemInstruction"]["parts"] == [{"text": system_prompt()}]
    assert body["contents"] == [{"role": "user", "parts": [{"text": TASK}]}]
    (group,) = body["tools"]
    (declaration,) = group["functionDeclarations"]
    assert camel(declaration) == {"name": "write_rows", "description": tool.description, "parametersJsonSchema": tool.parameters}
    generation = body["generationConfig"]
    assert set(generation) == {"maxOutputTokens", "thinkingConfig"} and generation["maxOutputTokens"] == 65_536  # no sampling parameter
    assert camel(generation["thinkingConfig"]) == {"thinkingLevel": "HIGH"}

    assert "thinking_config" not in adapter.build_request(start(), [tool], settings())["config"]  # no setting: the model's default
    with pytest.raises(FatalError):  # a misspelt setting is refused, not run at the default
        adapter.build_request(start(), [tool], settings(reasoning={"efort": "high"}))


def test_a_reply_with_tool_calls_is_read_back_and_answered_in_the_next_request():
    adapter, played = adapter_playing(TWO_CALLS, DONE)
    tool = write_rows_tool()
    reply = adapter.complete(start(), [tool], settings())
    assert (reply.stop_reason, reply.text) == (STOP_TOOL_CALLS, "")  # a thought summary is not the reply's text
    assert [(call.id, call.name, call.arguments) for call in reply.tool_calls] == [("fc_A", "write_rows", ROWS_A), ("fc_B", "write_rows", ROWS_B)]
    assert reply.usage.to_dict() == {"input_tokens": 412, "output_tokens": 960, "reasoning_tokens": 900, "cached_input_tokens": None}

    results = [tool_result(call, f"result of {call.id}") for call in reply.tool_calls]
    final = adapter.complete(start() + [reply.as_message(), *results], [tool], settings())
    assert played.bodies[1]["contents"][1:] == [
        {"role": "model", "parts": [THOUGHT, CALL_A, CALL_B]},  # as it came: the signature on the part it came with
        {
            "role": "user",
            "parts": [
                {"functionResponse": {"id": "fc_A", "name": "write_rows", "response": {"output": "result of fc_A"}}},
                {"functionResponse": {"id": "fc_B", "name": "write_rows", "response": {"output": "result of fc_B"}}},
            ],
        },
    ]
    assert (final.stop_reason, final.text) == (STOP_END, "All rows are written.")


def test_a_block_is_a_refusal_a_reply_at_the_output_limit_is_cut_off_and_a_broken_stream_is_no_reply():
    def reply_to(chunks):
        adapter, _played = adapter_playing(chunks)
        return adapter.complete(start(), [write_rows_tool()], settings())

    blocked = reply_to(PROMPT_BLOCKED)
    assert blocked.stop_reason == STOP_REFUSAL and "PROHIBITED_CONTENT" in blocked.refusal_text
    assert reply_to(REPLY_BLOCKED).stop_reason == STOP_REFUSAL
    cut = reply_to(CUT_OFF)
    assert (cut.stop_reason, cut.text, cut.tool_calls) == (STOP_MAX_TOKENS, "The painting is", [])
    with pytest.raises(TransientError):  # the stream stopped before a finish reason: the calls in it are not acted on
        reply_to([chunk(CALL_A, CALL_B)])


def test_the_key_is_read_only_from_the_named_variable_and_is_never_shown(monkeypatch):
    invalid = httpx.Response(400, json={"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "API key not valid."}})
    adapter, played = adapter_playing(DONE, invalid)
    reply = adapter.complete(start(), [], settings())
    with pytest.raises(FatalError) as refused:
        adapter.complete(start(), [], settings())
    sent = played.requests[0]
    assert sent.headers["x-goog-api-key"] == KEY  # in this header and nowhere else
    assert KEY not in str(sent.url) + sent.content.decode() + json.dumps(reply.to_dict()) + str(refused.value)

    monkeypatch.delenv(KEY_ENV)
    with pytest.raises(FatalError):  # GOOGLE_API_KEY, which the SDK reads by itself, is no substitute
        make_client({"api_key_env": KEY_ENV})
