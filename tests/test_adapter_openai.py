"""The "openai" adapter (OpenAI's Responses API), offline.

Every call goes through the real SDK over a mocked transport, so a request checked here is the one
the SDK would put on the wire. Nothing leaves the process and no real key is read.
"""

from __future__ import annotations

import json

import httpx
import openai
import pytest

from paintbench.adapters.base import FatalError
from paintbench.adapters.openai_responses import OpenAIAdapter
from paintbench.prompts import system_prompt, task_message, write_rows_tool
from paintbench.types import STOP_END, STOP_MAX_TOKENS, STOP_REFUSAL, STOP_TOOL_CALLS, ModelSettings, system, tool_result, user

KEY = "test-key-not-a-real-one"
KEY_ENV = "PAINTBENCH_TEST_KEY"
TOOL = write_rows_tool()
TASK = task_message("Under the Wave off Kanagawa, by Katsushika Hokusai, about 1831.", 32, 22, 16)
FIRST = [system(system_prompt()), user(TASK)]
REASONING = {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "encrypted-reasoning"}


def settings(**options) -> ModelSettings:
    """The model entry of configs/dalibench.yaml, with a short pause between polls."""
    return ModelSettings(name="astra-max", provider="openai", model="gpt-6.2-astra", reasoning={"effort": "max"},
                         max_output_tokens=128000, options={"api_key_env": KEY_ENV, "poll_seconds": 0.001, **options})


def text_item(text: str) -> dict:
    return {"id": "msg_1", "type": "message", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def refusal_item(text: str) -> dict:
    return {"id": "msg_1", "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "refusal", "refusal": text}]}


def call_item(n: int, arguments: dict) -> dict:
    return {"id": f"fc_{n}", "type": "function_call", "status": "completed", "call_id": f"call_{n}", "name": "write_rows",
            "arguments": json.dumps(arguments)}


def response(output: list, status: str = "completed", reason: str | None = None) -> dict:
    """A response as the API sends it: finished, unless `status` is "queued" or "in_progress"."""
    return {"id": "resp_1", "object": "response", "status": status, "model": "gpt-6.2-astra", "output": output, "error": None,
            "incomplete_details": {"reason": reason} if reason else None,
            "usage": {"input_tokens": 412, "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens": 9345, "output_tokens_details": {"reasoning_tokens": 9280}}}


def error(status: int, code: str | None, message: str) -> tuple[int, dict]:
    return status, {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": code}}


class Server:
    """api.openai.com behind a mocked transport. It gives its answers in order and keeps every request.
    An answer is the JSON body of a 200, or (status, body) for an error."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.pop(0)
        status, body = answer if isinstance(answer, tuple) else (200, answer)
        return httpx.Response(status, json=body)

    def http(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    def adapter(self) -> OpenAIAdapter:
        return OpenAIAdapter(client=openai.OpenAI(api_key=KEY, base_url="https://api.openai.com/v1", max_retries=0, http_client=self.http()))


def test_contestant_request_holds_the_prompt_and_one_tool_and_no_image():
    server = Server(response([text_item("Done.")]))
    server.adapter().complete(FIRST, [TOOL], settings())

    (request,) = server.requests
    assert (request.method, str(request.url)) == ("POST", "https://api.openai.com/v1/responses")
    body = json.loads(request.content)
    assert body == {
        "model": "gpt-6.2-astra",
        "input": [{"role": "developer", "content": system_prompt()}, {"role": "user", "content": TASK}],
        "tools": [{"type": "function", "name": "write_rows", "description": TOOL.description, "parameters": TOOL.parameters, "strict": False}],
        "reasoning": {"effort": "max"},
        "max_output_tokens": 128000,
        "store": True,
        "background": True,
    }  # and so no sampling parameter, no tool choice and no instructions of the adapter's own
    assert len(body["tools"]) == 1
    assert all(isinstance(item["content"], str) for item in body["input"])  # plain text and nothing else: no image


def test_a_reply_with_tool_calls_is_read_back_and_replayed_as_it_came():
    first, second = {"palette": ["k 20 40 90"], "start_row": 1, "rows": ["kkkk"]}, {"start_row": 2, "rows": ["kkkk"]}
    output = [REASONING, text_item("Rows 1 and 2."), call_item(1, first), call_item(2, second)]
    adapter = Server(response(output)).adapter()
    reply = adapter.complete(FIRST, [TOOL], settings())

    assert (reply.text, reply.stop_reason) == ("Rows 1 and 2.", STOP_TOOL_CALLS)
    assert [call.to_dict() for call in reply.tool_calls] == [
        {"id": "call_1", "name": "write_rows", "arguments": first},
        {"id": "call_2", "name": "write_rows", "arguments": second},
    ]
    # Reasoning is billed as output and is already inside output_tokens: it is reported, not added again.
    assert reply.usage.to_dict() == {"input_tokens": 412, "output_tokens": 9345, "reasoning_tokens": 9280, "cached_input_tokens": 0}

    # The next request carries every item of the reply unchanged, reasoning included, then one result per call.
    results = [tool_result(reply.tool_calls[0], "row 1: stored."), tool_result(reply.tool_calls[1], "row 2: stored.")]
    request = adapter.build_request(FIRST + [reply.as_message(), *results], [TOOL], settings())
    assert request["input"][2:] == output + [
        {"type": "function_call_output", "call_id": "call_1", "output": "row 1: stored."},
        {"type": "function_call_output", "call_id": "call_2", "output": "row 2: stored."},
    ]


def test_a_background_reply_is_created_once_and_polled_to_the_end():
    busy = error(503, "server_is_overloaded", "The server is overloaded.")
    server = Server(response([], "queued"), response([], "in_progress"), busy, response([text_item("All rows stored.")]))
    reply = server.adapter().complete(FIRST, [TOOL], settings())

    # One reply is paid for: a poll that fails is asked again for the same response, never answered by a second create.
    assert [(r.method, r.url.path) for r in server.requests] == [("POST", "/v1/responses")] + [("GET", "/v1/responses/resp_1")] * 3
    assert (reply.text, reply.stop_reason) == ("All rows stored.", STOP_END)


ENDINGS = {
    "the model refuses": (response([refusal_item("I cannot help with that.")]), STOP_REFUSAL),
    "the content filter stops it": (response([], "incomplete", "content_filter"), STOP_REFUSAL),
    "a safety block sent as an HTTP error": (error(403, "cyber_policy", "This request was blocked."), STOP_REFUSAL),
    "the output limit cuts it off": (response([REASONING], "incomplete", "max_output_tokens"), STOP_MAX_TOKENS),
    "it finishes with text": (response([text_item("Done.")]), STOP_END),
}


@pytest.mark.parametrize("name", list(ENDINGS))
def test_how_a_reply_ended_is_mapped(name):
    answer, expected = ENDINGS[name]
    reply = Server(answer).adapter().complete(FIRST, [TOOL], settings())
    assert reply.stop_reason == expected
    assert (reply.refusal_text is not None) == (expected == STOP_REFUSAL)


def test_the_key_goes_to_api_openai_com_alone_and_is_never_shown(monkeypatch):
    server = Server(error(401, "invalid_api_key", "Incorrect API key provided."), response([text_item("Done.")]))
    sdk_client = openai.OpenAI
    monkeypatch.setattr(openai, "OpenAI", lambda **given: sdk_client(**given, http_client=server.http()))
    monkeypatch.setenv(KEY_ENV, KEY)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://elsewhere.invalid/v1")  # must not redirect the adapter
    adapter = OpenAIAdapter()  # builds its own client, from the settings and the environment

    with pytest.raises(FatalError) as refused:
        adapter.complete(FIRST, [TOOL], settings())
    reply = adapter.complete(FIRST, [TOOL], settings())
    with pytest.raises(FatalError) as pasted:  # the key itself, written where the name of its variable belongs
        adapter.build_request(FIRST, [TOOL], settings(api_key_env=KEY))

    assert {(r.url.host, r.headers["authorization"]) for r in server.requests} == {("api.openai.com", f"Bearer {KEY}")}
    assert reply.text == "Done."
    assert KEY not in str(refused.value) + str(pasted.value) + json.dumps(reply.to_dict())
