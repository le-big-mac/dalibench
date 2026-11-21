# OpenAI adapter notes ("openai" adapter, first-party OpenAI reasoning models)

Researched 2026-11-19 and 2026-11-20 (UTC). No model/inference endpoint was called while these notes were written.
Independently re-verified 2026-11-20 against the installed SDK and freshly fetched documentation; see the last section,
"Verification (independent pass)", for what was checked, what was changed, and what was still unverified.

The adapter written from these notes has since been run against the live API (20 November 2026, model
`gpt-6.2-astra`). That run settled two of the open points, and each is marked where it stands under "UNVERIFIED (from
memory)": the API refuses `include=["reasoning.encrypted_content"]` for a stored response, so the request examples
below that send it beside `store: true` are not what the adapter sends; and `gpt-6.2-astra` accepts
`reasoning.effort: "max"`. The notes were not otherwise revised after that run.

Evidence tags used below:

- `[SDK: <path> :: <symbol>]` = read in the installed SDK, `openai 2.53.0`, under `site-packages/openai/`.
- `[DOC: <url>]` = official page fetched for these notes (all `developers.openai.com` pages were fetched as
  markdown by appending `.md`; `platform.openai.com/docs/*` 301-redirects to `developers.openai.com/api/docs/*`).
- `[OFFLINE]` = behaviour observed by running the installed SDK against `httpx.MockTransport`
  (no network; the request body the SDK builds and how it parses a canned reply).
- Anything not backed this way is in the section "UNVERIFIED (from memory)" at the end and nowhere else.

## 0. Read this first: decisions, and the SDK/API version skew

**Decisions**

| Question | Answer |
| --- | --- |
| Endpoint | Responses API, `POST /v1/responses`; SDK `client.responses.create(...)` (+ `client.responses.retrieve(id)` for polling). Not Chat Completions. |
| Why not Chat Completions | For the current models tool calling *requires* Responses: "GPT-6.2 Astra and GPT-6.2 Sol support Chat Completions, but tool calling requires Responses." `[DOC: https://developers.openai.com/api/docs/guides/latest-model/gpt-6.2-astra.md]` |
| Long calls (10-30 min) | Background mode: `background=True` on create, then poll `retrieve` until status is terminal. Returns a plain `Response`. |
| Reasoning across tool turns | Replay **every** output item of every response, verbatim and in order, in the next request's `input` (stateless replay). Do **not** use `previous_response_id`. Always send `include=["reasoning.encrypted_content"]` (but not beside `store=true`: the live API refuses that, see the top of this file). |
| Highest reasoning setting | `reasoning={"effort": "max"}` (see section 5 for the per-model table and the one caveat for `gpt-6.2-*`). |
| Tool strictness | Send `"strict": false` explicitly on the `write_rows` tool. |
| System prompt | First item of `input`, role `developer`. Not the top-level `instructions` parameter. |
| Temperature etc. | Never send `temperature`, `top_p`, `top_logprobs`. |

**Version skew (important).** The installed SDK is `openai 2.53.0`, released 2026-08-03; the newest SDK is `3.56.0`
(2026-11-19) and the newest models (GPT-6.2) shipped 2026-11-17.
`[DOC: https://raw.githubusercontent.com/openai/openai-python/main/CHANGELOG.md]`
`[DOC: https://developers.openai.com/api/docs/changelog.md]`
So the docs describe things the installed SDK has no types for. None of this blocks the adapter:

- `model` is typed `Union[str, ...]`, so any model id string is accepted. `[SDK: types/shared/responses_model.py :: ResponsesModel]`
  The installed `ChatModel` literal list stops at `gpt-5.6-*`; it has no `gpt-6*` entries. `[SDK: types/shared/chat_model.py :: ChatModel]`
- Unknown response fields are kept, not dropped: they are readable as attributes and via `response.model_extra`
  (observed for `store`, `input_transformations`, and an invented field). `[OFFLINE]`
- Unknown enum values in responses are not validated: e.g. `incomplete_details.reason == "max_messages"` parses fine
  although the installed type only lists two values. `[OFFLINE]`
- Request parameters the SDK does not know go in `extra_body={...}`, which is merged into the top-level JSON body
  (observed with `background_options`). `[OFFLINE]` `[SDK: resources/responses/responses.py :: Responses.create (extra_body)]`

Present in today's API reference but **not typed in SDK 2.53.0** (use `extra_body` / `model_extra` if ever needed):
request `background_options`, `image`, `access_programs`, `service_tier: "ultrafast"`, function-tool `async`,
input item `configuration_update`; response `input_transformations`, `prompt_cache_diagnostics`, `tool_usage`,
`usage.input_tokens_details.cache_write_tokens_details`, `incomplete_details.reason` values `max_messages` and
`steered`, error code `misalignment_policy_violation`.
`[DOC: https://developers.openai.com/api/reference/resources/responses/methods/create.md]`

Do not upgrade the SDK casually: `3.0.0` (2026-08-12) is a breaking change that makes HTTPX2 the default HTTP client
and stops installing `httpx`; tests that mock `httpx` transports would need rework.
`[DOC: https://raw.githubusercontent.com/openai/openai-python/main/CHANGELOG.md]`

---

## 1. Endpoint, client, auth, timeouts, long requests

### 1.1 Endpoint and method

- `POST {base_url}/responses`, default `base_url = https://api.openai.com/v1`. SDK: `client.responses.create(**params) -> Response`
  (or `Stream[ResponseStreamEvent]` when `stream=True`). `[SDK: resources/responses/responses.py :: Responses.create]`
- `GET {base_url}/responses/{response_id}` -> `client.responses.retrieve(response_id, include=[...]) -> Response`.
  `[SDK: resources/responses/responses.py :: Responses.retrieve]` With `include=["reasoning.encrypted_content"]` the SDK sends
  `?include[]=reasoning.encrypted_content`. `[OFFLINE]`
- `POST {base_url}/responses/{response_id}/cancel` -> `client.responses.cancel(response_id) -> Response`; "Only responses created
  with the `background` parameter set to `true` can be cancelled." `[SDK: resources/responses/responses.py :: Responses.cancel]`
- Why Responses: "Reasoning models work better with the Responses API" and "Use the Responses API for function calling. Chat
  Completions does not support function calling with GPT-6 Astra or GPT-6.1 Sol."
  `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]`
  "GPT-6 Sol and GPT-6 Luna support function calling in Chat Completions only with `reasoning_effort: "none"`. Use Responses for
  reasoning with tools." `[DOC: https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md]`

### 1.2 Client construction and authentication

```python
import httpx
from openai import OpenAI

client = OpenAI(
    # api_key defaults to os.environ["OPENAI_API_KEY"]
    # base_url defaults to os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    timeout=httpx.Timeout(600.0, connect=5.0),   # this is the SDK default, shown explicitly
    max_retries=2,                               # SDK default
)
```

- Environment variables read by the constructor: `OPENAI_API_KEY` (api key), `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`,
  `OPENAI_BASE_URL` (base URL override), `OPENAI_CUSTOM_HEADERS`, `OPENAI_ADMIN_KEY`, `OPENAI_WEBHOOK_SECRET`.
  `[SDK: _client.py :: OpenAI.__init__]`
- The key is sent as `Authorization: Bearer <key>`. `[OFFLINE]`
- With no key the constructor raises `OpenAIError("Missing credentials. Please pass an `api_key`, `workload_identity`,
  `admin_api_key`, or set the `OPENAI_API_KEY` or `OPENAI_ADMIN_KEY` environment variable.")`, so build the client
  lazily (not at import time) and pass `api_key="test"` (or an `http_client` with a mock transport) in tests.
  The check passes when only `OPENAI_ADMIN_KEY` is set, so do not use "constructor did not raise" as proof that an API key
  exists. `[SDK: _client.py :: OpenAI.__init__]`
- `OPENAI_ORG_ID` / `OPENAI_PROJECT_ID`, when set, are sent on every request as the headers `OpenAI-Organization` /
  `OpenAI-Project`. `[SDK: _client.py :: OpenAI.default_headers]`
- `OPENAI_BASE_URL`, if set in the environment, silently redirects every call. Pass `base_url="https://api.openai.com/v1"`
  explicitly if the adapter must be immune to that. `[SDK: _client.py :: OpenAI.__init__]`
- Per-call overrides: `client.with_options(timeout=..., max_retries=...)` or `timeout=` on the method.
  `[SDK: _client.py :: OpenAI.copy / with_options]` `[OFFLINE]`
- Every response object has `response._request_id` (from the `x-request-id` header); log it.
  `[SDK: openai-2.53.0.dist-info/METADATA :: "Request IDs"]`

### 1.3 Timeouts and built-in retries (what the SDK does by default)

- Default timeout: `httpx.Timeout(timeout=600, connect=5.0)` (10 minutes); default `max_retries = 2`.
  `[SDK: _constants.py :: DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES]`
- A timeout raises `APITimeoutError` **after** the retries: "requests that time out are retried twice by default".
  `[SDK: openai-2.53.0.dist-info/METADATA :: "Timeouts"]` `[SDK: _base_client.py :: SyncAPIClient.request]`
  For a foreground call that legitimately needs 20 minutes this means three 10-minute attempts, each restarting the generation
  from scratch (closing the connection is how a synchronous response is cancelled: "For a synchronous response, close the
  connection instead." `[DOC: https://developers.openai.com/api/docs/guides/background.md]`).
  A transport `httpx.ReadTimeout` surfaces as `openai.APITimeoutError` after 3 attempts with the default `max_retries=2`.
  `[OFFLINE]`
- The 600 s is applied per socket read, not to the request as a whole: the transport sets the socket timeout before each
  `recv`. A non-streaming response sends nothing until it is finished, so the read timeout fires after 600 s of silence.
  `[SDK: ../httpcore/_backends/sync.py :: SyncStream.read]` (`httpcore` is the transport installed with `httpx 0.28.1`)
- The SDK has **no** rule that forces streaming for long requests (nothing comparable exists in `Responses.create`); it just
  applies the timeout. `[SDK: resources/responses/responses.py :: Responses.create]`

### 1.4 What OpenAI recommends for long requests, and the recommended path

Documented guidance:

- "Use `background=True` for requests that may take a long time. Instead of keeping the client connection open, the API starts a job
  and returns an ID. Your app can poll that job until it finishes, fails, or is canceled."
  `[DOC: https://developers.openai.com/api/docs/guides/deployment-checklist.md]`
- "Use background mode to run long-running Responses API requests asynchronously, including multi-step agent runs."
  `[DOC: https://developers.openai.com/api/docs/guides/background.md]`
- For long-thinking models: "some requests may take several minutes to finish. To avoid timeouts, try using background mode."
  `[DOC: https://developers.openai.com/api/docs/models/gpt-5.5-pro.md]`
- The only timeout advice for foreground calls is in the Flex guide: raise the SDK `timeout` (their example uses 900 s); "The
  default timeout is 10 minutes when making API requests with an official OpenAI SDK."
  `[DOC: https://developers.openai.com/api/docs/guides/flex-processing.md]`

**Recommended (contestant calls and judge calls alike): background mode + polling.** Every HTTP request is then short, the default
timeout and retries are appropriate, and the result is one ordinary `Response` object.

```python
import time

IN_FLIGHT = ("queued", "in_progress")           # keep polling
# terminal: "completed", "incomplete", "failed", "cancelled"

def run_response(client, *, poll_seconds=5.0, deadline_seconds=3 * 3600, **params):
    """params = model, input, tools, reasoning, max_output_tokens, store, include, ..."""
    resp = client.responses.create(background=True, **params)       # returns quickly, status "queued"/"in_progress"
    started = time.monotonic()
    while resp.status in IN_FLIGHT:
        if time.monotonic() - started > deadline_seconds:
            client.responses.cancel(resp.id)
            raise TimeoutError(f"OpenAI response {resp.id} still {resp.status} after {deadline_seconds}s")
        time.sleep(poll_seconds)
        resp = client.responses.retrieve(resp.id, include=["reasoning.encrypted_content"])
    return resp                                                      # plain openai.types.responses.Response
```

This is the documented polling loop (`while resp.status in {"queued", "in_progress"}: sleep(2); resp = client.responses.retrieve(resp.id)`).
`[DOC: https://developers.openai.com/api/docs/guides/background.md]` Log `resp.id` as soon as create returns so an interrupted run
can resume polling. Mock surface for tests: `client.responses.create`, `client.responses.retrieve`,
`client.responses.cancel`.

`store` with background mode (read carefully):

- `store` defaults to `true`; "If set to true, response data will be stored for at least 30 days".
  `[DOC: https://developers.openai.com/api/reference/resources/responses/methods/create.md]`
- "Background mode temporarily stores response data even when you set `store` to `false` or use Zero Data Retention. The API uses
  this data to run the request and make the result available for retrieval."
  `[DOC: https://developers.openai.com/api/docs/guides/background.md]` The official example uses exactly
  `background=True, store=False` with `gpt-6.2-astra` and a 2-second poll.
  `[DOC: https://developers.openai.com/api/docs/guides/deployment-checklist.md]`
- But the temporary copy is short-lived: "Background mode with the default completion window stores response data to disk for
  roughly 10 minutes to run the request and make the response available for retrieval. ... With the default completion window,
  background responses follow the standard retention period only if you explicitly set `store=true`. If you omit `store` or set
  it to `false`, the response is deleted after the temporary retention period."
  `[DOC: https://developers.openai.com/api/docs/guides/your-data.md]`
- **Recommendation: send `store=True` explicitly on background calls.** Conversation continuity in this adapter never depends on
  stored state (section 4 replays everything), so `store` only decides how long a finished response can still be fetched by id.
  With `store=True` that is 30 days (and a crashed harness can recover an expensive result); with `store=False` it is "roughly
  10 minutes" and the docs do not say from when. An organisation with Zero Data Retention gets the `store=False` behaviour
  whatever is sent: "With ZDR enabled, the Responses API treats `store` as `false`."
  `[DOC: https://developers.openai.com/api/docs/guides/latest-model/gpt-6.2-astra.md]`
  Keep `store` as one config constant. If it is set to `False`, keep
  `poll_seconds <= 10`, and report a `retrieve` that raises `openai.NotFoundError` (the SDK class for HTTP 404) as "background
  result no longer retrievable; set store=True" (the exact error for an expired response is not documented).

Not recommended, documented for completeness:

- **Foreground, non-streaming, long timeout**: `client.with_options(timeout=httpx.Timeout(3600.0, connect=10.0), max_retries=0).responses.create(...)`.
  One call, no polling, works with `store=False`. Risks: a single idle HTTP connection for 10-30 minutes; if it drops the whole
  generation is lost and (with retries on) silently re-run. `[SDK: _base_client.py :: SyncAPIClient.request]`
- **Streaming** (`stream=True`): see section 10.6 for the event sequence. Two SDK traps make it a poor fit here:
  `client.responses.stream(...).get_final_response()` raises `RuntimeError("Didn't receive a `response.completed` event.")`
  whenever the stream ends with `response.incomplete` or `response.failed` (i.e. on every max-token truncation), and it returns a
  `ParsedResponse` whose items carry extra keys (section 11). `[SDK: lib/streaming/responses/_responses.py :: ResponseStream.get_final_response, ResponseStreamState.accumulate_event]` `[OFFLINE]`
- **Discounted background windows** (`extra_body={"background_options": {"completion_window": "2h" | "8h"}}`, 30% / 40% off, public
  beta, requires `service_tier="default"`, no streaming): the API only *targets* completion within the window, per request, so
  a multi-turn tool loop could take hours per turn. Not for this harness.
  `[DOC: https://developers.openai.com/api/docs/guides/background.md]`

---

## 2. Request shape for a contestant call

### 2.1 Full request (first turn)

```json
{
  "model": "gpt-6.2-astra",
  "input": [
    {"role": "developer", "content": "<system prompt text>"},
    {"role": "user", "content": "<task text, e.g. the painting name and instructions>"}
  ],
  "tools": [
    {
      "type": "function",
      "name": "write_rows",
      "description": "<what the tool does and how to use it>",
      "parameters": {
        "type": "object",
        "properties": {
          "palette": {"type": "array", "items": {"type": "string"}, "description": "<...>"},
          "start_row": {"type": "integer", "description": "<...>"},
          "rows": {"type": "array", "items": {"type": "string"}, "description": "<...>"}
        },
        "required": ["start_row", "rows"]
      },
      "strict": false
    }
  ],
  "reasoning": {"effort": "max"},
  "max_output_tokens": 128000,
  "include": ["reasoning.encrypted_content"],
  "store": true,
  "background": true
}
```

Python: `client.responses.create(model=..., input=[...], tools=[...], reasoning={"effort": "max"}, max_output_tokens=128000,
include=["reasoning.encrypted_content"], store=True, background=True)`. The SDK sends exactly these keys and nothing else
(omitted parameters are not serialised). `[OFFLINE]`

Keep `model`, `tools`, `reasoning`, and every earlier `input` item byte-identical across all requests of one conversation
(section 4.5 explains why).

### 2.2 System prompt

- Use a message with role `developer` as the **first** item of `input`: "To use the same guidance across turns, put it in the first
  `developer` message in `input`." `[DOC: https://developers.openai.com/api/docs/guides/text.md]`
- Do not use top-level `instructions` for it: "Models starting with GPT-6.2 append top-level `instructions` after `input` in the
  Responses API" and bill those tokens at the uncached input rate on every request.
  `[DOC: https://developers.openai.com/api/docs/guides/latest-model/gpt-6.2-astra.md]`
  `[DOC: https://developers.openai.com/api/docs/guides/prompt-caching.md]`
- Accepted roles for input messages: `user`, `assistant`, `system`, `developer`; "Instructions given with the `developer` or `system`
  role take precedence over instructions given with the `user` role." `[SDK: types/responses/easy_input_message_param.py :: EasyInputMessageParam]`
- Message `content` may be a plain string or a list of content parts (`input_text`, `input_image`, `input_file`).
  `[SDK: types/responses/response_input_message_content_list_param.py]`

### 2.3 The function tool format

- Responses uses the **flat** ("internally tagged") form `{"type": "function", "name", "description", "parameters", "strict"}`.
  The Chat Completions form with a nested `"function": {...}` object is wrong here.
  `[DOC: https://developers.openai.com/api/docs/guides/migrate-to-responses.md]` `[SDK: types/responses/function_tool_param.py :: FunctionToolParam]`
- Fields: `name` (required), `parameters` (JSON Schema object, required key, nullable), `strict` (required key in the SDK type,
  nullable), `type: "function"`, `description` (optional). Other optional keys exist (`defer_loading`, `allowed_callers`,
  `output_schema`); leave them out. `[SDK: types/responses/function_tool_param.py :: FunctionToolParam]`

### 2.4 `strict`: set it to `false`

- Strict mode requires: "`additionalProperties` must be set to `false` for each object in the `parameters`" and "All fields in
  `properties` must be marked as `required`" (optional fields are expressed by adding `null` to the type).
  `[DOC: https://developers.openai.com/api/docs/guides/function-calling.md]`
- `write_rows` has an optional property (`palette`; only `start_row` and `rows` are in `required`) and no
  `additionalProperties: false`, so it does not satisfy strict mode as written. (Making it strict would mean listing `palette`
  in `required` with type `["array", "null"]` and adding `additionalProperties: false`; the model would then send
  `"palette": null` instead of omitting it.) A `required` list without `additionalProperties` is accepted in a request: the
  API reference's function example sends exactly that (there with `strict` omitted, and the echoed tool shows
  `"strict": true`, i.e. the normalisation described in the next bullet).
  `[DOC: https://developers.openai.com/api/reference/resources/responses/methods/create.md]` (example "Functions")
- The Responses default when `strict` is **omitted** is not "off": "Responses requests will attempt to normalize your schema into
  strict mode when possible, and will fall back to non-strict, best-effort function calling if the schema cannot be made
  compatible with strict mode. When fallback happens, the response tool will show `strict: false`. ... To opt out of strict mode
  in Responses and keep non-strict, best-effort function calling, explicitly set `strict: false`."
  `[DOC: https://developers.openai.com/api/docs/guides/function-calling.md]`
- "If you send `strict: true` and your schema does not meet the requirements above, the request will be rejected".
  `[DOC: https://developers.openai.com/api/docs/guides/function-calling.md]`
- Therefore: send `"strict": false`. Consequence: arguments are best-effort, so the harness must `json.loads` defensively and
  validate types itself, returning a plain error string as the tool result when the call is malformed.

### 2.5 Tool choice and parallel calls (we leave defaults)

- `tool_choice` default is `"auto"` ("Call zero, one, or multiple functions"). Other values: `"none"`, `"required"`,
  `{"type": "function", "name": "write_rows"}`, and `{"type": "allowed_tools", "mode": "auto"|"required", "tools": [...]}`.
  `[DOC: https://developers.openai.com/api/docs/guides/function-calling.md]` `[SDK: types/responses/tool_choice_options.py :: ToolChoiceOptions]`
- `parallel_tool_calls` (boolean, default `true`): "The model may choose to call multiple functions in a single turn. You can prevent
  this by setting `parallel_tool_calls` to `false`, which ensures exactly zero or one tool is called."
  `[DOC: https://developers.openai.com/api/docs/guides/function-calling.md]`
  `[DOC: https://developers.openai.com/api/reference/resources/responses/methods/create.md]`
  With the default, one response can contain several `function_call` items; the adapter must handle that (section 4.3).

### 2.6 Other request parameters worth knowing (all left unset)

`[SDK: types/responses/response_create_params.py :: ResponseCreateParamsBase]`

| Parameter | Default | Note |
| --- | --- | --- |
| `truncation` | `"disabled"` | With the default, an input larger than the context window fails with HTTP 400 rather than being trimmed. |
| `service_tier` | `"auto"` | GPT-6.2 does not support `flex`. `[DOC: .../guides/latest-model/gpt-6.2-astra.md]` |
| `previous_response_id`, `conversation` | unset | Server-side state; not used. |
| `safety_identifier` | unset | If ever sent, send the identical value on every request of a conversation (section 4.5). |
| `text` | `{"format": {"type": "text"}}`, `verbosity: "medium"` | Plain text output; leave alone. `[DOC: .../responses/methods/create.md]` |
| `metadata` | unset | Up to 16 string pairs; optional run labels. |
| `temperature`, `top_p`, `top_logprobs` | - | Never send (section 5.4). |

---

## 3. Response shape

`[SDK: types/responses/response.py :: Response]` unless noted.

### 3.1 Top level

Abridged skeleton (placeholders, not valid JSON; complete fixtures are in section 10):

```text
{
  "id": "resp_...",
  "object": "response",
  "created_at": 1795078800,
  "completed_at": 1795079712,
  "status": "completed",
  "background": true,
  "error": null,
  "incomplete_details": null,
  "model": "gpt-6.2-astra",
  "output": [ ...items... ],
  "usage": { ... },
  "reasoning": {"effort": "max", "summary": null},
  "parallel_tool_calls": true,
  "tool_choice": "auto",
  "tools": [ ...echo of the tools, with the resolved "strict" value... ]
}
```

`output` is an ordered list of **items**, each with a `type`. For this adapter three types matter: `reasoning`, `function_call`,
`message`. "The `output` array often has more than one item in it! ... It is not safe to assume that the model's text output is
present at `output[0].content[0].text`." `[DOC: https://developers.openai.com/api/docs/guides/text.md]`

Always branch on the string `item.type`. Never use `isinstance`: an item or stream event whose `type` the installed SDK does not
know is still returned, parsed into an arbitrary class of the union, with `.type` and `.to_dict()` intact. `[OFFLINE]`

### 3.2 Text

- Text lives in items of type `message` (`role: "assistant"`), in `content` parts of type `output_text`, field `text`:

```json
{
  "id": "msg_...",
  "type": "message",
  "role": "assistant",
  "status": "completed",
  "phase": "final_answer",
  "content": [
    {"type": "output_text", "text": "Done. The painting is complete.", "annotations": [], "logprobs": []}
  ]
}
```

  `[SDK: types/responses/response_output_message.py :: ResponseOutputMessage]` `[SDK: types/responses/response_output_text.py :: ResponseOutputText]`
- `response.output_text` is an SDK convenience property: the concatenation (`"".join`) of every `output_text` part of every
  `message` item, `""` if there is none. `[SDK: types/responses/response.py :: Response.output_text]`
- `phase` is optional (`"commentary"` | `"final_answer"` | absent). A `commentary` message is an interim note (e.g. a preamble
  before tool calls), not the answer; `output_text` concatenates both kinds. Preserve `phase` when replaying.
  `[SDK: types/responses/response_output_message.py :: ResponseOutputMessage.phase]`
  `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]` (section "`phase` parameter")

### 3.3 Tool calls

```json
{
  "id": "fc_...",
  "type": "function_call",
  "status": "completed",
  "call_id": "call_...",
  "name": "write_rows",
  "arguments": "{\"start_row\":0,\"rows\":[\"0011\",\"0110\"]}"
}
```

`[SDK: types/responses/response_function_tool_call.py :: ResponseFunctionToolCall]`

- `arguments` is a **JSON string**, not an object ("A JSON string of the arguments to pass to the function"). `json.loads` it.
- Two different ids: `id` is the item's own id (optional); **`call_id`** is the one the tool result must reference.
  "Each entry with a `call_id` (used later to submit the function result), `name`, and JSON-encoded `arguments`."
  `[DOC: https://developers.openai.com/api/docs/guides/function-calling.md]`
- `status` is `in_progress` | `completed` | `incomplete` (optional). Do not execute a call whose status is `incomplete`.
- Optional fields `namespace`, `caller` may appear; ignore them but replay them.
- "Since model responses can include zero, one, or multiple calls, it is best practice to assume there are several."
  `[DOC: https://developers.openai.com/api/docs/guides/function-calling.md]`

### 3.4 Reasoning items

```json
{"id": "rs_...", "type": "reasoning", "summary": [], "encrypted_content": "gAAAAAB..."}
```

`[SDK: types/responses/response_reasoning_item.py :: ResponseReasoningItem]`

- Required fields in the SDK type: `id`, `summary`, `type`. Optional: `encrypted_content`, `content` (list of
  `{"type": "reasoning_text", "text": ...}`), `status`. The docs' own output examples show reasoning items with
  `"content": []` and without `encrypted_content`, so do not require any particular optional key; replay whatever arrived.
  `[DOC: https://developers.openai.com/api/docs/guides/function-calling.md]` (custom-tool output example)
- `summary` is a list of `{"type": "summary_text", "text": "..."}`; empty unless `reasoning.summary` was requested.
- `encrypted_content` is opaque; "This is populated by default for reasoning items returned by `POST /v1/responses`".
- Raw reasoning text is never returned ("we don't expose the raw reasoning tokens").
  `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]`
- In the docs' tool-call output example the reasoning item comes first and the call second (`[reasoning, custom_tool_call]`).
  Do not rely on a fixed position or count; scan the whole `output` list by `type`.
  `[DOC: https://developers.openai.com/api/docs/guides/function-calling.md]` (custom-tool output example)

### 3.5 Stop / finish reasons

There is **no** `finish_reason` / `stop_reason` field. Use `status` + `incomplete_details.reason` + whether `output` contains
`function_call` items.

`status` values: `completed`, `failed`, `in_progress`, `cancelled`, `queued`, `incomplete`. `[SDK: types/responses/response_status.py :: ResponseStatus]`

| `status` | `incomplete_details.reason` | `function_call` items | Meaning for the harness |
| --- | --- | --- | --- |
| `queued`, `in_progress` | - | - | Background job not finished; keep polling. |
| `completed` | `null` | one or more | Tool-use turn: run the calls, send results (section 4). |
| `completed` | `null` | none | Model ended its turn with text (or a refusal part). |
| `incomplete` | `max_output_tokens` | any | Hit `max_output_tokens` or the context window. May have **no visible output at all** (ran out while reasoning). |
| `incomplete` | `content_filter` | any | Output stopped by the content filter (safety block). |
| `incomplete` | `max_messages`, `steered` | - | Listed in the current API reference only; not expected for this request shape (`steered` is WebSocket-only). |
| `failed` | - | - | Generation failed; see `response.error.code` / `.message`. In background mode this is read from a successful `retrieve` (no exception is raised), so the adapter must check `status` itself. |
| `cancelled` | - | - | Background response was cancelled via the cancel endpoint. |

- Reason values: installed SDK types `max_output_tokens`, `content_filter` `[SDK: types/responses/response.py :: IncompleteDetails]`;
  the current reference adds `max_messages`, `steered`. `[DOC: https://developers.openai.com/api/reference/resources/responses/methods/create.md]`
- "If the generated tokens reach the context window limit or the `max_output_tokens` value you've set, you'll receive a response
  with a `status` of `incomplete` and `incomplete_details` with `reason` set to `max_output_tokens`. This might occur before any
  visible output tokens are produced, meaning you could incur costs for input and reasoning tokens without receiving a visible
  response." `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]`
- `response.error` (when `status == "failed"`) is `{"code": ..., "message": ...}`. Codes typed in the SDK: `server_error`,
  `rate_limit_exceeded`, `invalid_prompt`, `data_residency_mismatch`, `bio_policy`, `vector_store_timeout`, and a family of
  image errors (`invalid_image`, `invalid_image_format`, `invalid_base64_image`, `invalid_image_url`, `image_too_large`,
  `image_too_small`, `image_parse_error`, `image_content_policy_violation`, `invalid_image_mode`, `image_file_too_large`,
  `unsupported_image_media_type`, `empty_image_file`, `failed_to_download_image`, `image_file_not_found`).
  `[SDK: types/responses/response_error.py :: ResponseError]` The current reference adds the code
  `misalignment_policy_violation` and three optional fields on the error object: `param` (string or null), `type` (string
  or null, "The original error type") and `misalignment` (object). SDK 2.53.0 types only `code` and `message`; the extra
  fields, when present, are readable via `response.error.model_extra`. Compare `error.code` as a plain string.
  `[DOC: https://developers.openai.com/api/reference/resources/responses/methods/create.md]`

### 3.6 Refusals and safety blocks (four different signals)

1. **Refusal content part** inside a normal `completed` response: a `message` item whose `content` contains
   `{"type": "refusal", "refusal": "<explanation>"}` instead of (or beside) `output_text`. `response.output_text` ignores it, so
   it will be `""`. `[SDK: types/responses/response_output_refusal.py :: ResponseOutputRefusal]`
   `[DOC: https://developers.openai.com/api/docs/guides/structured-outputs.md]` (the docs describe this part only in the
   Structured Outputs context; how a refusal is delivered in plain-text mode is listed under UNVERIFIED).
2. **`status: "incomplete"` with `incomplete_details.reason: "content_filter"`**. `[SDK: types/responses/response.py :: IncompleteDetails]`
3. **`status: "failed"`** with `error.code` such as `invalid_prompt`, `bio_policy`, `image_content_policy_violation`,
   `misalignment_policy_violation`. `[SDK: types/responses/response_error.py]` `[DOC: .../responses/methods/create.md]`
4. **HTTP error** raised by the SDK: e.g. HTTP 403, error type `invalid_request_error`, code `misalignment_policy_violation` ->
   `openai.PermissionDeniedError`; "Do not automatically retry the blocked workflow." "Match the error code rather than the
   message text." This applies to this harness: "Responses API requests using persisted reasoning, WebSockets, or OpenAI
   compaction" are "Monitored. The system can identify continuations of a conversation and block further execution."
   `[DOC: https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring.md]`
   Cyber-classifier blocks use error code `cyber_policy`. `[DOC: https://developers.openai.com/api/docs/guides/safety-checks/cybersecurity.md]`

### 3.7 Usage

```json
{
  "usage": {
    "input_tokens": 12000,
    "input_tokens_details": {"cached_tokens": 4000, "cache_write_tokens": 7000},
    "output_tokens": 1186,
    "output_tokens_details": {"reasoning_tokens": 1024},
    "total_tokens": 13186
  }
}
```

`[SDK: types/responses/response_usage.py :: ResponseUsage, InputTokensDetails, OutputTokensDetails]`

- `response.usage` can be `None` (typed `Optional`; it is `null` while a background response is still running in the fixtures
  below). Guard for it.
- **Reasoning tokens are included in `output_tokens`, not additional to it.** The doc example has `output_tokens: 1186` with
  `reasoning_tokens: 1024` and `total_tokens = input_tokens + output_tokens`; reasoning tokens "are billed as output tokens".
  `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]` Visible output is roughly `output_tokens - reasoning_tokens`,
  minus uncounted formatting tokens: "the reported output or completion token count can be higher than the number of visible
  tokens ... even when the reported `reasoning_tokens` value is `0`."
  `[DOC: https://developers.openai.com/api/docs/guides/token-counting.md]`
- **`input_tokens` includes cached and cache-write tokens.** Ordinary input = `input_tokens - cached_tokens - cache_write_tokens`.
  `[DOC: https://developers.openai.com/api/docs/guides/prompt-caching.md]`
- Cost formula for GPT-5.6-and-later models (they bill cache writes):
  `ordinary * input_price + cached_tokens * cached_price + cache_write_tokens * 1.25 * input_price + output_tokens * output_price`
  (30-minute cache writes are 1.25x input; this adapter never requests 12-hour writes). Caching is automatic (implicit); no
  request parameter is needed. `[DOC: https://developers.openai.com/api/docs/guides/prompt-caching.md]`
- The current API also returns `input_tokens_details.cache_write_tokens_details: {ttl_30m_tokens, ttl_12h_tokens}` (not typed in
  2.53.0; readable via `model_extra`). `[DOC: .../responses/methods/create.md]`

---

## 4. The tool-result turn

### 4.1 Recommendation: stateless replay of all output items (not `previous_response_id`)

Both are documented:

- "You can carry reasoning items forward using `previous_response_id`. If you manage conversation history yourself, include the
  original input and all output items from each response in the input of the next request. Keep those items in their original
  order and append new items." `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]`
- "Note that for reasoning models like GPT-5 or o4-mini, any reasoning items returned in model responses with tool calls must also
  be passed back with tool call outputs." `[DOC: https://developers.openai.com/api/docs/guides/function-calling.md]`

**Use replay.** Reasons: the harness transcript stays the single source of truth; it works identically with `store` true, false or
Zero Data Retention; it survives response expiry; and it is the pattern the function-calling guide itself uses
(`input_list += response.output`, then append `function_call_output`). `previous_response_id` needs the previous response to still be stored server-side,
and once it expires the documented fallback is replay anyway ("If the prior response expires, omit `previous_response_id` and
pass the conversation history in `input`"). `[DOC: https://developers.openai.com/api/docs/guides/background.md]`

Always send `include=["reasoning.encrypted_content"]` (on create and on retrieve). In stateless mode the docs say the encrypted
content is now returned by default and the include value is "legacy ... for compatibility", so sending it is harmless and removes
any dependence on that default. `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]`
`[SDK: types/responses/response_includable.py :: ResponseIncludable]`

### 4.2 Exact procedure

```python
history = [
    {"role": "developer", "content": system_prompt},
    {"role": "user", "content": task_prompt},
]

resp = run_response(client, model=MODEL, input=history, tools=TOOLS, reasoning=REASONING,
                    max_output_tokens=MAX_OUT, include=["reasoning.encrypted_content"], store=STORE)

# 1. Replay the assistant turn: EVERY output item, in order, exactly as returned.
history.extend(item.to_dict() for item in resp.output)

# 2. One function_call_output per function_call, in the same order.
for item in resp.output:
    if item.type == "function_call":
        result_text = apply_tool(item.name, item.arguments)      # arguments is a JSON string
        history.append({"type": "function_call_output", "call_id": item.call_id, "output": result_text})

# 3. Next request: same model/tools/reasoning, input = the grown history.
```

- `item.to_dict()` is the documented way to turn output items into input items
  (`history.extend(item.to_dict() for item in first.output)`). `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]`
  It emits exactly the keys the API sent (unset fields are excluded; unknown future fields are preserved). `[OFFLINE]`
  Passing the SDK objects themselves (`history += resp.output`) produces the identical JSON. `[OFFLINE]`
- What must be passed back verbatim for reasoning to stay valid: **all** items - each `reasoning` item including its `id`,
  `summary` and `encrypted_content`; each `function_call` item including `call_id`, `name`, `arguments`; each `message` item
  including `phase`. "Your app does not need to understand that value. It just keeps each reasoning item exactly as returned and
  sends it back during the next turn". `[DOC: https://developers.openai.com/api/docs/guides/deployment-checklist.md]`
  Common migration errors: "Dropping reasoning, function call, or function call output Items when manually carrying context into
  the next response." / "Sending a function result without the matching `call_id`."
  `[DOC: https://developers.openai.com/api/docs/guides/migrate-to-responses.md]`
- Do not rewrite anything: do not re-serialise `arguments` (no `json.loads` -> `json.dumps` round trip), do not reorder, do not
  drop "empty" reasoning items, do not shorten earlier tool results later (section 4.5).

### 4.3 Tool result item format

```json
{"type": "function_call_output", "call_id": "call_...", "output": "rows 0-1 written; 30 rows remain"}
```

`[SDK: types/responses/response_input_item_param.py :: FunctionCallOutput]`

- Required: `type`, `call_id` (the `call_id` of the matching `function_call`, **not** its `id`), `output`.
- `output` is a string (max length 10,485,760) or a list of `input_text` / `input_image` / `input_file` content parts. A plain
  string is right for this harness. "The result you pass in the `function_call_output` message should typically be a string, where
  the format is up to you (JSON, error codes, plain text, etc.)." There is no `is_error` flag: report tool errors in the string.
  `[DOC: https://developers.openai.com/api/docs/guides/function-calling.md]` `[DOC: .../responses/methods/create.md]`
- It is a top-level input item, not a message: no `role`.
- **Several tool calls in one reply**: replay all output items first (reasoning, then every `function_call`), then append one
  `function_call_output` per call, in the order of the calls, each with its own `call_id`. This is the loop in the docs
  (`input_messages += response.output` then `for tool_call in response.output: ... input_messages.append({...})`).
  `[DOC: https://developers.openai.com/api/docs/guides/function-calling.md]`

### 4.4 Plain follow-up user message (the "nudge")

After a reply with no tool call, append that response's output items, then a user message:

```python
history.extend(item.to_dict() for item in resp.output)          # reasoning + assistant message
history.append({"role": "user", "content": "You stopped early. Continue with write_rows until the painting is complete."})
```

This is the documented stateless multi-turn pattern (`history += response.output`; `history.append({"role": "user", ...})`).
`[DOC: https://developers.openai.com/api/docs/guides/conversation-state.md]`

### 4.5 Rules that silently degrade quality if broken (GPT-6.2 especially)

- "Starting with GPT-6.2, the Responses API drops reasoning items after mid-conversation edits to the conversation history (prefix
  edits), certain model changes (including downgrades), or changes to the end user or organization."
  `[DOC: https://developers.openai.com/api/docs/guides/latest-model/gpt-6.2-astra.md]`
- A prefix change includes "updating Responses API parameters such as `tools` or `text.format`, changing `instructions` on
  pre-GPT-6.2 models, or editing earlier `input` items", and changing request-level `reasoning.effort`. So: append-only history,
  constant `tools` / `reasoning` / `model` for the whole conversation.
  `[DOC: https://developers.openai.com/api/docs/guides/keep-prefix-unchanged.md]`
- Diagnose with the response field `input_transformations` (list of `{id, action: "removed", reason, type: "reasoning"}`, reasons
  `prefix_mismatch`, `end_user_mismatch`, `model_mismatch`, `organization_mismatch`, `turn_boundary`). Log it when non-empty. In
  SDK 2.53.0 read it with `getattr(resp, "input_transformations", None)`.
  `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]` `[OFFLINE]`
- `turn_boundary` is not an error: with reasoning context mode `current_turn`, reasoning items from earlier turns stay in the
  payload but are not rendered to the model ("Those reasoning items can remain in the API payload for continuity, but the service
  does not render them into the new sample"). The docs do not define where a "turn" ends (see UNVERIFIED). Keep replaying
  everything either way.
  `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]`

---

## 5. Reasoning settings

### 5.1 `reasoning.effort`

`reasoning` is an object: `{"effort": ..., "summary": ..., "context": ..., "mode": ...}` (all optional).
`[SDK: types/shared/reasoning.py :: Reasoning]`

- All values the API knows: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. `[SDK: types/shared/reasoning_effort.py :: ReasoningEffort]`
  "Not all reasoning models support every value." **`max` is the highest** ("Maximum reasoning for your most complex tasks. If you
  are currently using `xhigh`, evaluate if `max` results in stronger performance").
  `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]`
- Default: API reference says `effort` default `"medium"`; defaults are "model-dependent rather than universal".
  `[DOC: .../responses/methods/create.md]` `[DOC: .../guides/reasoning.md]`
- "The models also reason adaptively across reasoning efforts, using fewer tokens for simpler tasks".
  `[DOC: .../guides/reasoning.md]`

Per-model support, from each model page (`https://developers.openai.com/api/docs/models/<id>.md`):

| Model id | Documented `reasoning.effort` values | Default |
| --- | --- | --- |
| `gpt-6.2-astra` | Page states only: "The API rejects requests that set reasoning effort to `none`." The migration guide adds: "If you currently use `none` or `minimal`, use `low` instead." The full list is **not** enumerated. | not stated on the page |
| `gpt-6.2-sol` | same wording as `gpt-6.2-astra` | not stated on the page |
| `gpt-6.1-terra` | `none`, `low`, `medium`, `high`, `xhigh`, `max` | `medium` |
| `gpt-6-luna` | `none`, `low`, `medium`, `high`, `xhigh`, `max` | `medium` |
| `gpt-6.1-sol` | `low`, `medium`, `high`, `xhigh`, `max` | `medium` |
| `gpt-6-astra` | `low`, `medium`, `high`, `xhigh`, `max` (`none` -> HTTP 400) | not stated |
| `gpt-6-sol` | `none`, `low`, `medium`, `high`, `xhigh`, `max` | `medium` |
| `gpt-5.6-sol/terra/luna` | `none`, `low`, `medium`, `high`, `xhigh`, `max` `[DOC: .../guides/latest-model/gpt-5.6.md]` | `medium` |

**Use `"max"`.** Caveat for `gpt-6.2-astra` / `gpt-6.2-sol`: `max` is the generic top value and every other current model lists
it, but those two pages do not spell out their list (the docs' own GPT-6.2 examples go up to `xhigh`: the deployment checklist
sends `reasoning={"effort": "xhigh", "mode": "pro"}` to `gpt-6.2-astra`). Make the effort a config
value, and if the first live call returns HTTP 400 (`openai.BadRequestError`) naming `reasoning.effort`, fall back to `"xhigh"`.
The shape of that 400 is not documented (see UNVERIFIED), so detect it loosely (`e.param` or `e.message` mentioning
`effort`), and only ever fall back on the **first** request of a conversation: changing request-level `reasoning.effort`
later is a prefix change that drops reasoning (section 4.5).
`[DOC: https://developers.openai.com/api/docs/guides/deployment-checklist.md]`
(Since settled for `gpt-6.2-astra`: the live API accepted `"max"` on 20 November 2026. The adapter sends the configured
value as written and has no fallback.)

### 5.2 Other reasoning fields

- `reasoning.summary`: `"auto"` | `"concise"` | `"detailed"`; opt-in ("This output will not be included unless you explicitly opt
  in"); `auto` gives the most detailed summariser available. "Before using summarizers with our latest reasoning models, you may
  need to complete organization verification". Leave it **unset** by default so a missing verification cannot fail the run; the
  only effect is `summary: []` in reasoning items. `generate_summary` is the deprecated name.
  `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]` `[SDK: types/shared/reasoning.py :: Reasoning]`
- `reasoning.context`: `"auto"` (same as omitting) | `"current_turn"` | `"all_turns"`. Controls whether reasoning items from
  **earlier turns** are rendered to the model; `current_turn` "Makes reasoning from the active turn available" either way.
  GPT-5.6 defaults to `all_turns`, earlier models to `current_turn`; "`all_turns` ... GPT-5.6 models and GPT-6.1 Sol support this
  value". The default and support for GPT-6.2 are not stated. **Omit it** and log `response.reasoning.context`, which reports
  the effective mode. `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]`
- `reasoning.mode`: `"standard"` (default) | `"pro"`. A separate knob from effort ("Reasoning mode and reasoning effort are
  independent"); pro "applies more model work to a request before returning a single final answer" at higher latency and token
  use, billed at the model's normal rates. Documented for GPT-5.6 and GPT-6 models. Not part of "highest effort"; leave unset
  unless the benchmark decides to define the top setting as `max` + `pro`.
  `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]` `[DOC: .../guides/latest-model/gpt-5.6.md]`

### 5.3 Interaction with `max_output_tokens`

- `max_output_tokens` is "An upper bound for the number of tokens that can be generated for a response, including visible output
  tokens and reasoning tokens." Minimum 16. `[SDK: types/responses/response_create_params.py]` `[DOC: .../responses/methods/create.md]`
- So **reasoning counts against it**. A low limit at high effort yields `status: "incomplete"` with empty text. "OpenAI recommends
  reserving at least 25,000 tokens for reasoning and outputs when you start experimenting with these models."
  `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]`
- Reasoning tokens also "occupy space in the model's context window". `[DOC: .../guides/reasoning.md]`

### 5.4 Parameters to omit with reasoning on

"For Responses requests, omit `temperature`, `top_p`, and `top_logprobs` and remove `message.output_text.logprobs` from `include`."
(listed under "Unsupported parameters" for GPT-6.2; for GPT-6: "When reasoning effort is not `none`, remove ...").
`[DOC: https://developers.openai.com/api/docs/guides/latest-model/gpt-6.2-astra.md]`
`[DOC: https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md]`
The harness never sends them, which is correct.

---

## 6. Output-token limits and context windows

From the model pages and `https://developers.openai.com/api/docs/models/compare.md`:

| Model id | Context window | Max input tokens | Max output tokens |
| --- | ---: | ---: | ---: |
| `gpt-6.2-astra` | 1,050,000 | 922,000 | 128,000 |
| `gpt-6.2-sol` | 1,050,000 | 922,000 | 128,000 |
| `gpt-6.1-terra` | 1,050,000 | 922,000 | 128,000 |
| `gpt-6-luna` | 1,050,000 | 922,000 | 128,000 |
| `gpt-6.1-sol`, `gpt-6-astra`, `gpt-6-sol` | 1,050,000 | 922,000 | 128,000 |

- "Set the limit high": send `max_output_tokens=128000` for all GPT-6.x models (the documented per-model maximum). Omitting the
  parameter is also valid (it is optional/nullable); then no request-level cap is sent and the docs state no other default.
- Worst-case cost of one call at the cap on `gpt-6.2-astra`: 128,000 output tokens x $40 / 1M = $5.12, plus input.
- Long-context surcharge: "For requests with more than 272K input tokens, input and cache rates double and output costs 1.5
  times the short-context rate for the full request." A long tool loop replays its whole history every turn, so watch
  `usage.input_tokens`. `[DOC: https://developers.openai.com/api/docs/models/gpt-6.2-astra.md]`
- With the default `truncation: "disabled"`, exceeding the context window is an HTTP 400, not silent trimming.
  `[SDK: types/responses/response_create_params.py :: truncation]`
- **A high cap is not free for rate limits.** "Your rate limit is calculated as the maximum of `max_tokens` and the estimated
  number of tokens based on the character count of your request. Try to set the `max_tokens` value as close to your expected
  response size as possible." `[DOC: https://developers.openai.com/api/docs/guides/rate-limits.md]` The passage uses the Chat
  Completions parameter name; if it applies to `max_output_tokens` the same way (not stated, see UNVERIFIED), each request
  with a 128,000 cap counts at least 128,000 tokens against tokens-per-minute, i.e. about 7 requests per minute on the Build
  tier of `gpt-6.2-astra` (1,000,000 TPM, section 8). Make the cap a config value so it can be lowered if contestant runs
  hit 429s while actual usage is low.
- "For long context models like GPT-5.5, there is a separate rate limit for long context requests."
  `[DOC: https://developers.openai.com/api/docs/guides/rate-limits.md]`

---

## 7. Vision input for the judge

### 7.1 Request

One user message whose `content` is a list of parts, in display order. Text parts are `input_text`, images are `input_image` with a
data URL in `image_url`:

```json
{
  "model": "gpt-6.2-astra",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "You are judging renderings of a painting. Reference image:"},
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo...", "detail": "high"},
        {"type": "input_text", "text": "Candidate A:"},
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo...", "detail": "high"},
        {"type": "input_text", "text": "Candidate B:"},
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo...", "detail": "high"},
        {"type": "input_text", "text": "Answer with the letter of the better candidate and one sentence of justification."}
      ]
    }
  ],
  "reasoning": {"effort": "high"},
  "max_output_tokens": 32000,
  "store": true,
  "background": true
}
```

- Part shapes: `{"type": "input_text", "text": str}` and `{"type": "input_image", "image_url": str, "detail": ...}`.
  `image_url` is "A fully qualified URL or base64 encoded image in a data URL." (Alternative: `file_id`.)
  `[SDK: types/responses/response_input_text_param.py :: ResponseInputTextParam]` `[SDK: types/responses/response_input_image_param.py :: ResponseInputImageParam]`
- Data URL form: `f"data:image/png;base64,{b64}"` with standard base64 of the PNG bytes.
  `[SDK: openai-2.53.0.dist-info/METADATA :: "Vision"]` `[DOC: https://developers.openai.com/api/docs/guides/images-vision.md]`
- `image_url` here is a **string**, not the Chat Completions object `{"url": ...}`, and the part type is `input_image`, not
  `image_url`.
- Multiple images: "You can provide multiple images as input in a single request by including multiple images in the `content`
  array". Interleaving text and image parts in one list is how the docs' own examples are built.
  `[DOC: https://developers.openai.com/api/docs/guides/images-vision.md]`
- No `tools`. Read the answer from `response.output_text`; if it is empty, look for a `refusal` part and check `status`.
- A judge call is a normal Responses call; use the same `run_response` helper. The effort and token limit above are placeholders.

### 7.2 `detail`

- Values: `low`, `high`, `original`, `auto`; "If you omit the parameter, it defaults to `auto`". The SDK type marks `detail` as a
  required key, so always send it. `[DOC: .../guides/images-vision.md]` `[SDK: types/responses/response_input_image_param.py]`
- Behaviour listed for `gpt-6-astra` (the table has no separate `gpt-6.2-*`, `gpt-6.1-terra` or `gpt-6-luna` rows): "`low` fits
  within 512 x 512 pixels. `high` allows up to 2,500 patches and a 65,535-pixel maximum dimension. ... `original` preserves the
  image's dimensions ... `auto` uses the same sizing behavior as `original`." `[DOC: .../guides/images-vision.md]`
- For small grid renderings, `high` is a safe explicit choice: nothing is enlarged ("All resizing preserves aspect ratio without
  enlarging smaller images") and anything up to 2,500 patches (e.g. 1600 x 1600) is passed at full size.
- `high` is listed for every model family in the sizing table, but GPT-6.2 has no row; the docs' own `gpt-6.2-astra` vision
  examples send `"detail": "auto"` (or omit it). Keep `detail` a config value; if a live call returns a 400 naming `detail`,
  use `"auto"`.

### 7.3 Limits

`[DOC: https://developers.openai.com/api/docs/guides/images-vision.md]`

| Limit | Value |
| --- | --- |
| File types | PNG, JPEG, WEBP, non-animated GIF |
| Images per request | up to 1,500 |
| Request payload | up to 512 MB total |
| Per image | at most 30,000 patches (32 x 32 px each) after resizing, else the request is **rejected** (not auto-resized); sides above 65,535 px are scaled down to that limit (`gpt-6-astra` row) |

Nine PNGs are far inside every limit. Images also count toward tokens-per-minute rate limits.

### 7.4 Billing

Images are billed as input tokens at the model's input price. Patch-based formula:
`patches = ceil(width / 32) * ceil(height / 32)` (after any resizing for the chosen detail level), then
`tokens = ceil(patches * multiplier)`. Multiplier is 1.2 for `gpt-6-astra` and the GPT-5.6 / 5.5 / 5.4 families; the table does
not list `gpt-6.2-*`, `gpt-6.1-terra` or `gpt-6-luna`. Worked examples for `gpt-6-astra`, `detail: high`: 1024 x 1024 -> 1,024
patches -> 1,229 tokens; 2048 x 2048 -> resized to 1600 x 1600 -> 2,500 patches -> 3,000 tokens. "Billing can differ by one token
due to rounding." `[DOC: https://developers.openai.com/api/docs/guides/images-vision.md]`
`[DOC: https://developers.openai.com/api/docs/guides/image-cost-calculator.md]`

---

## 8. Current model ids and list prices

Model ids confirmed on the model pages, the pricing page, and the current SDK's model list
(`[DOC: https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/shared/chat_model.py]`).
Prices: Standard tier, USD per 1M tokens, short context (<= 272K input tokens), from
`https://developers.openai.com/api/docs/pricing.md` fetched 2026-11-19.

| Tier | API model id | Input | Cached input | Cache write (30 min) | Output | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Flagship | `gpt-6.2-astra` | $8.00 | $0.40 | $10.00 | $40.00 | released 2026-11-17; text + image in, text out |
| Mid | `gpt-6.2-sol` | $2.00 | $0.10 | $2.50 | $10.00 | released 2026-11-17 |
| Mid / vision-oriented | `gpt-6.1-terra` | $1.00 | $0.05 | $1.25 | $5.00 | "visual reasoning, precise computer use" |
| Small / cheap | `gpt-6-luna` | $0.10 | $0.01 | $0.125 | $0.50 | "focused, high-volume tasks" |
| Previous flagship | `gpt-6-astra` | $10.00 | $1.00 | $12.50 | $50.00 | |
| Previous mid | `gpt-6.1-sol` | $2.00 | $0.10 | $2.50 | $10.00 | |
| Previous mid | `gpt-6-sol` | $2.00 | $0.20 | $2.50 | $10.00 | |
| Older | `gpt-5.6-sol` / `-terra` / `-luna` | $4.00 / $2.00 / $0.20 | $0.40 / $0.20 / $0.02 | $5.00 / $2.50 / $0.25 | $20.00 / $12.00 / $1.20 | |

- The models overview names the tiers: "use GPT-6.2 Astra, our flagship model ... Choose GPT-6.2 Sol to balance intelligence and
  cost, or GPT-6 Luna for cost-sensitive, high-volume workloads." `[DOC: https://developers.openai.com/api/docs/models.md]`
- Long context (> 272K input tokens): input and cache rates x2, output x1.5, for the whole request (e.g. `gpt-6.2-astra`
  $16.00 / $60.00). `[DOC: .../pricing.md]`
- Batch is 50% of Standard; Fast mode is 2x; background `2h` / `8h` windows are 30% / 40% below Standard for
  `gpt-6.2-astra`, `gpt-6.2-sol`, `gpt-6.1-terra`, `gpt-6-luna`. `[DOC: .../pricing.md]`
- The model pages list only the bare ids (no dated snapshots) for the GPT-6.x models.
- The `gpt-6.2-astra` page lists endpoints Chat Completions, Responses, Batch, and features streaming, structured outputs,
  function calling, image input, prompt caching. `[DOC: https://developers.openai.com/api/docs/models/gpt-6.2-astra.md]`
- "GPT-5.6 Sol's promotional pricing is available until November 30, 2026." `[DOC: .../pricing.md]`
- Published rate limits for `gpt-6.2-astra` (requests / tokens per minute): Build 5,000 / 1,000,000; Launch 10,000 / 4,000,000;
  Grow 15,000 / 40,000,000. `[DOC: .../models/gpt-6.2-astra.md]`

---

## 9. Errors and retries

### 9.1 Exception classes (`import openai`)

`[SDK: _exceptions.py]` `[SDK: _client.py :: OpenAI._make_status_error]`

| HTTP | Exception | Retry? |
| --- | --- | --- |
| 400 | `BadRequestError` | No. Bad parameter, unsupported value, input too large. |
| 401 | `AuthenticationError` | No. |
| 403 | `PermissionDeniedError` | No (includes `misalignment_policy_violation`; unsupported region). |
| 404 | `NotFoundError` | No ("Requested resource does not exist", e.g. a response id). |
| 408 | `APIStatusError` (no dedicated subclass; check `e.status_code == 408`) | SDK retries. `[OFFLINE]` |
| 409 | `ConflictError` | SDK retries. |
| 422 | `UnprocessableEntityError` | No. |
| 429 | `RateLimitError` | Depends on `e.code` (below). SDK retries all 429s. |
| >= 500 | `InternalServerError` | Yes (503 `server_is_overloaded`, 500). |
| none | `APIConnectionError`; subclass `APITimeoutError` | Yes. |

All derive from `openai.APIError` -> `openai.OpenAIError`. Status errors derive from `APIStatusError` and expose `status_code`,
`request_id`, `response`, `body`, and (from the error JSON `{"error": {"message", "type", "param", "code"}}`) `e.message`,
`e.type`, `e.param`, `e.code`. `[SDK: _exceptions.py :: APIError, APIStatusError]` `[OFFLINE]`

### 9.2 SDK built-in retry behaviour

`[SDK: _base_client.py :: BaseClient._should_retry, _calculate_retry_timeout, _parse_retry_after_header]` `[SDK: _constants.py]`

- Retries up to `max_retries` (default 2) on: connection errors and timeouts, HTTP 408, 409, 429, and >= 500. The header
  `x-should-retry: true|false` overrides this.
- Delay: honours `retry-after-ms`, then `retry-after` (seconds or HTTP date), when the value is > 0 and <= 120 s; otherwise
  exponential backoff `0.5 s * 2^n` capped at 8 s, with jitter. If `Retry-After` exceeds 120 s the SDK does **not** retry and
  raises the status error.
- Configure with `OpenAI(max_retries=N)` or `client.with_options(max_retries=N)`; `0` disables.
- The SDK does not look at `error.code`, so it also retries 429s that can never succeed (observed: a 429 with a quota code was
  attempted three times). `[OFFLINE]`
- No idempotency key is sent for OpenAI requests (`_idempotency_header` is `None`), so a create call retried after a lost reply
  can start a second generation. `[SDK: _base_client.py :: BaseClient.__init__]`

### 9.3 Which 429s / 5xx are worth retrying at harness level

`[DOC: https://developers.openai.com/api/docs/guides/error-codes.md]` `[DOC: https://developers.openai.com/api/docs/guides/rate-limits.md]`

- Retry (after `Retry-After` if present, else exponential backoff with jitter): 429 "Rate limit reached"; 429 type
  `rate_limit_error` code `slow_down`; 503 type `service_unavailable_error` code `server_is_overloaded`; 500.
- Do **not** retry: 429 with code `credit_balance_exhausted`, `organization_spend_limit_exceeded`,
  `project_spend_limit_exceeded`, `organization_usage_limit_exceeded` (the type may read `insufficient_quota`: "The broader
  `error.type` can still be `insufficient_quota`"). "Retrying billing, spend, or quota errors won't restore API access."
  The rate-limit guide's own retry sample also stops on `e.code` `insufficient_quota` and `billing_hard_limit_reached`, so
  treat those two codes, and `e.type == "insufficient_quota"`, as non-retryable as well.
- "If you manage retries in your application, disable SDK retries or account for them in those limits so nested retry loops don't
  multiply requests."
- Response-level failures arrive with HTTP 200 (a polled background response with `status: "failed"`), so SDK retries never
  see them. The docs give no retry guidance for failed responses; going by the code names, re-sending the same request is
  reasonable for `error.code` `server_error` / `rate_limit_exceeded` and pointless for the policy and image codes.
  `[SDK: types/responses/response_error.py]`

### 9.4 Rate-limit headers

`x-ratelimit-limit-requests`, `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-requests`, `x-ratelimit-remaining-tokens`,
`x-ratelimit-reset-requests` (e.g. `1s`), `x-ratelimit-reset-tokens` (e.g. `6m0s`), optional
`x-ratelimit-limit-project-tokens` / `-remaining-project-tokens` / `-reset-project-tokens`, and `Retry-After` (seconds; "may be
present on `429` responses caused by a temporary rate limit and `503` responses caused by temporary model overload").
`[DOC: https://developers.openai.com/api/docs/guides/rate-limits.md]`
To read headers: `raw = client.responses.with_raw_response.create(...)`; `raw.headers[...]`; `raw.parse()` returns the `Response`.
`[SDK: resources/responses/responses.py :: ResponsesWithRawResponse]` `[OFFLINE]`

---

## 10. Worked example (fixtures)

Shapes follow the API reference examples and the installed SDK types; ids and the `encrypted_content` strings are invented and
opaque. Every response fixture below was parsed with the installed SDK (`Response` construction, `output_text`, item `to_dict()`).
`[OFFLINE]` The request fixtures 10.1 and 10.3 were compared with the JSON body the installed SDK actually sends for the same
keyword arguments and for the `to_dict()` replay of 10.2b: identical. `[OFFLINE]` The `queued` fixture (10.2a) is built from the
schema; the docs show no JSON for a queued response.

### 10.1 Request 1 (body of `POST /v1/responses`)

```json
{
  "model": "gpt-6.2-astra",
  "input": [
    {"role": "developer", "content": "You are a pixel painter. Use the write_rows tool to type the painting row by row. When the painting is finished, reply with a short confirmation and no tool call."},
    {"role": "user", "content": "Paint: The Starry Night (Vincent van Gogh). Canvas: 4 columns x 4 rows."}
  ],
  "tools": [
    {
      "type": "function",
      "name": "write_rows",
      "description": "Write consecutive rows of the canvas. Each string in rows is one row; each character is a palette index.",
      "parameters": {
        "type": "object",
        "properties": {
          "palette": {"type": "array", "items": {"type": "string"}, "description": "Colours, index = position."},
          "start_row": {"type": "integer", "description": "Index of the first row being written."},
          "rows": {"type": "array", "items": {"type": "string"}, "description": "One string per row."}
        },
        "required": ["start_row", "rows"]
      },
      "strict": false
    }
  ],
  "reasoning": {"effort": "max"},
  "max_output_tokens": 128000,
  "include": ["reasoning.encrypted_content"],
  "store": true,
  "background": true
}
```

### 10.2a Response 1, as returned by the create call (background job accepted)

```json
{
  "id": "resp_0f3a9c1e5b7d42a68c0e1f2a3b4c5d6e",
  "object": "response",
  "created_at": 1795078800,
  "status": "queued",
  "background": true,
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": 128000,
  "model": "gpt-6.2-astra",
  "output": [],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {"effort": "max", "summary": null},
  "store": true,
  "temperature": 1.0,
  "text": {"format": {"type": "text"}},
  "tool_choice": "auto",
  "tools": [
    {
      "type": "function",
      "name": "write_rows",
      "description": "Write consecutive rows of the canvas. Each string in rows is one row; each character is a palette index.",
      "parameters": {
        "type": "object",
        "properties": {
          "palette": {"type": "array", "items": {"type": "string"}, "description": "Colours, index = position."},
          "start_row": {"type": "integer", "description": "Index of the first row being written."},
          "rows": {"type": "array", "items": {"type": "string"}, "description": "One string per row."}
        },
        "required": ["start_row", "rows"]
      },
      "strict": false
    }
  ],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": null,
  "user": null,
  "metadata": {}
}
```

Poll `GET /v1/responses/resp_0f3a9c1e5b7d42a68c0e1f2a3b4c5d6e?include[]=reasoning.encrypted_content`. For a mid-flight poll
fixture use the same object with `"status": "in_progress"` (also schema-shaped, not copied from the docs).

### 10.2b Response 1, final (reasoning + one tool call)

```json
{
  "id": "resp_0f3a9c1e5b7d42a68c0e1f2a3b4c5d6e",
  "object": "response",
  "created_at": 1795078800,
  "completed_at": 1795079712,
  "status": "completed",
  "background": true,
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": 128000,
  "model": "gpt-6.2-astra",
  "output": [
    {
      "id": "rs_0f3a9c1e5b7d42a68c0e1f2a3b4c5d6e01",
      "type": "reasoning",
      "summary": [],
      "encrypted_content": "gAAAAABpHk3ZfakeEncryptedReasoningBlobOne=="
    },
    {
      "id": "fc_0f3a9c1e5b7d42a68c0e1f2a3b4c5d6e02",
      "type": "function_call",
      "status": "completed",
      "call_id": "call_Qm8xT2vLr4Jd0aWk",
      "name": "write_rows",
      "arguments": "{\"palette\":[\"#0b1f4b\",\"#f4d35e\"],\"start_row\":0,\"rows\":[\"0010\",\"0111\"]}"
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {"effort": "max", "summary": null},
  "store": true,
  "temperature": 1.0,
  "text": {"format": {"type": "text"}},
  "tool_choice": "auto",
  "tools": [
    {
      "type": "function",
      "name": "write_rows",
      "description": "Write consecutive rows of the canvas. Each string in rows is one row; each character is a palette index.",
      "parameters": {
        "type": "object",
        "properties": {
          "palette": {"type": "array", "items": {"type": "string"}, "description": "Colours, index = position."},
          "start_row": {"type": "integer", "description": "Index of the first row being written."},
          "rows": {"type": "array", "items": {"type": "string"}, "description": "One string per row."}
        },
        "required": ["start_row", "rows"]
      },
      "strict": false
    }
  ],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 412,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 9345,
    "output_tokens_details": {"reasoning_tokens": 9280},
    "total_tokens": 9757
  },
  "user": null,
  "metadata": {}
}
```

Expected adapter reading: `status == "completed"`, `output_text == ""`, one tool call with
`call_id == "call_Qm8xT2vLr4Jd0aWk"`, `name == "write_rows"`,
`json.loads(arguments) == {"palette": ["#0b1f4b", "#f4d35e"], "start_row": 0, "rows": ["0010", "0111"]}`;
usage: input 412, output 9345 of which reasoning 9280.

### 10.3 Request 2 (replayed assistant turn + tool result)

Only `input` grows; every other key is identical to request 1.

```json
{
  "model": "gpt-6.2-astra",
  "input": [
    {"role": "developer", "content": "You are a pixel painter. Use the write_rows tool to type the painting row by row. When the painting is finished, reply with a short confirmation and no tool call."},
    {"role": "user", "content": "Paint: The Starry Night (Vincent van Gogh). Canvas: 4 columns x 4 rows."},
    {
      "id": "rs_0f3a9c1e5b7d42a68c0e1f2a3b4c5d6e01",
      "type": "reasoning",
      "summary": [],
      "encrypted_content": "gAAAAABpHk3ZfakeEncryptedReasoningBlobOne=="
    },
    {
      "id": "fc_0f3a9c1e5b7d42a68c0e1f2a3b4c5d6e02",
      "type": "function_call",
      "status": "completed",
      "call_id": "call_Qm8xT2vLr4Jd0aWk",
      "name": "write_rows",
      "arguments": "{\"palette\":[\"#0b1f4b\",\"#f4d35e\"],\"start_row\":0,\"rows\":[\"0010\",\"0111\"]}"
    },
    {
      "type": "function_call_output",
      "call_id": "call_Qm8xT2vLr4Jd0aWk",
      "output": "OK: wrote rows 0-1. Rows remaining: 2."
    }
  ],
  "tools": [
    {
      "type": "function",
      "name": "write_rows",
      "description": "Write consecutive rows of the canvas. Each string in rows is one row; each character is a palette index.",
      "parameters": {
        "type": "object",
        "properties": {
          "palette": {"type": "array", "items": {"type": "string"}, "description": "Colours, index = position."},
          "start_row": {"type": "integer", "description": "Index of the first row being written."},
          "rows": {"type": "array", "items": {"type": "string"}, "description": "One string per row."}
        },
        "required": ["start_row", "rows"]
      },
      "strict": false
    }
  ],
  "reasoning": {"effort": "max"},
  "max_output_tokens": 128000,
  "include": ["reasoning.encrypted_content"],
  "store": true,
  "background": true
}
```

(The key order inside replayed items does not matter; the SDK's `to_dict()` emits `arguments, call_id, name, type, id, status`
for a function call. `[OFFLINE]`)

### 10.4 Response 2, final (reasoning + final text, no tool call)

```json
{
  "id": "resp_1a4b0d2f6c8e53b79d1f203142536475",
  "object": "response",
  "created_at": 1795079720,
  "completed_at": 1795079801,
  "status": "completed",
  "background": true,
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": 128000,
  "model": "gpt-6.2-astra",
  "output": [
    {
      "id": "rs_1a4b0d2f6c8e53b79d1f20314253647501",
      "type": "reasoning",
      "summary": [],
      "encrypted_content": "gAAAAABpHk4AfakeEncryptedReasoningBlobTwo=="
    },
    {
      "id": "msg_1a4b0d2f6c8e53b79d1f20314253647502",
      "type": "message",
      "status": "completed",
      "role": "assistant",
      "phase": "final_answer",
      "content": [
        {"type": "output_text", "text": "The painting is complete.", "annotations": [], "logprobs": []}
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {"effort": "max", "summary": null},
  "store": true,
  "temperature": 1.0,
  "text": {"format": {"type": "text"}},
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 9790,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 9700},
    "output_tokens": 1210,
    "output_tokens_details": {"reasoning_tokens": 1190},
    "total_tokens": 11000
  },
  "user": null,
  "metadata": {}
}
```

Expected: no `function_call` items -> the loop ends (or a nudge is sent, section 4.4); `output_text == "The painting is complete."`.
`phase` may be absent; do not require it. (`"tools": []` is abbreviated here; the real echo repeats the tool list.)

### 10.5 Other response fixtures (only the fields that differ; fill the rest from 10.4)

**(a) Refusal as a content part** (`status` is still `completed`; `output_text == ""`):

```json
{
  "id": "resp_2b5c1e3a7d9f64c8ae20314253647586",
  "object": "response",
  "created_at": 1795080000,
  "completed_at": 1795080004,
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "model": "gpt-6.2-astra",
  "output": [
    {
      "id": "msg_2b5c1e3a7d9f64c8ae2031425364758601",
      "type": "message",
      "status": "completed",
      "role": "assistant",
      "content": [
        {"type": "refusal", "refusal": "I'm sorry, I cannot assist with that request."}
      ]
    }
  ],
  "parallel_tool_calls": true,
  "tool_choice": "auto",
  "tools": [],
  "usage": {
    "input_tokens": 81,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 11,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 92
  }
}
```

**(b) Safety block via content filter** (`status: "incomplete"`):

```json
{
  "id": "resp_3c6d2f4b8e0a75d9bf31425364758697",
  "object": "response",
  "created_at": 1795080100,
  "status": "incomplete",
  "error": null,
  "incomplete_details": {"reason": "content_filter"},
  "model": "gpt-6.2-astra",
  "output": [],
  "parallel_tool_calls": true,
  "tool_choice": "auto",
  "tools": [],
  "usage": {
    "input_tokens": 412,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 37,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 449
  }
}
```

**(c) Max-token truncation during reasoning** (no visible output; `output_text == ""`):

```json
{
  "id": "resp_4d7e3a5c9f1b86eac0425364758697a8",
  "object": "response",
  "created_at": 1795080200,
  "status": "incomplete",
  "error": null,
  "incomplete_details": {"reason": "max_output_tokens"},
  "max_output_tokens": 128000,
  "model": "gpt-6.2-astra",
  "output": [
    {
      "id": "rs_4d7e3a5c9f1b86eac0425364758697a801",
      "type": "reasoning",
      "summary": [],
      "encrypted_content": "gAAAAABpHk5QfakeEncryptedReasoningBlobCut=="
    }
  ],
  "parallel_tool_calls": true,
  "tool_choice": "auto",
  "tools": [],
  "usage": {
    "input_tokens": 412,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 128000,
    "output_tokens_details": {"reasoning_tokens": 128000},
    "total_tokens": 128412
  }
}
```

**(d) Two tool calls in one reply** (with an interim `commentary` message; `output_text == "Writing the last two rows."`):

```json
{
  "id": "resp_5e8f4b6d0a2c97fbd1536475869708b9",
  "object": "response",
  "created_at": 1795080300,
  "completed_at": 1795080950,
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "model": "gpt-6.2-astra",
  "output": [
    {
      "id": "rs_5e8f4b6d0a2c97fbd1536475869708b901",
      "type": "reasoning",
      "summary": [],
      "encrypted_content": "gAAAAABpHk6AfakeEncryptedReasoningBlobTwoCalls=="
    },
    {
      "id": "msg_5e8f4b6d0a2c97fbd1536475869708b902",
      "type": "message",
      "status": "completed",
      "role": "assistant",
      "phase": "commentary",
      "content": [
        {"type": "output_text", "text": "Writing the last two rows.", "annotations": [], "logprobs": []}
      ]
    },
    {
      "id": "fc_5e8f4b6d0a2c97fbd1536475869708b903",
      "type": "function_call",
      "status": "completed",
      "call_id": "call_A1b2C3d4E5f6G7h8",
      "name": "write_rows",
      "arguments": "{\"start_row\":2,\"rows\":[\"1110\"]}"
    },
    {
      "id": "fc_5e8f4b6d0a2c97fbd1536475869708b904",
      "type": "function_call",
      "status": "completed",
      "call_id": "call_I9j0K1l2M3n4O5p6",
      "name": "write_rows",
      "arguments": "{\"start_row\":3,\"rows\":[\"0100\"]}"
    }
  ],
  "parallel_tool_calls": true,
  "tool_choice": "auto",
  "tools": [],
  "usage": {
    "input_tokens": 9790,
    "input_tokens_details": {"cached_tokens": 9700, "cache_write_tokens": 0},
    "output_tokens": 5120,
    "output_tokens_details": {"reasoning_tokens": 5050},
    "total_tokens": 14910
  }
}
```

The follow-up request appends, after the four replayed items, two results in call order:

```json
[
  {"type": "function_call_output", "call_id": "call_A1b2C3d4E5f6G7h8", "output": "OK: wrote row 2. Rows remaining: 1."},
  {"type": "function_call_output", "call_id": "call_I9j0K1l2M3n4O5p6", "output": "OK: wrote row 3. Rows remaining: 0."}
]
```

**(e) Failed generation** (HTTP 200; `status: "failed"`):

```json
{
  "id": "resp_6f905c7e1b3da80ce264758697081920",
  "object": "response",
  "created_at": 1795080400,
  "status": "failed",
  "error": {"code": "server_error", "message": "The server had an error while processing your request."},
  "incomplete_details": null,
  "model": "gpt-6.2-astra",
  "output": [],
  "parallel_tool_calls": true,
  "tool_choice": "auto",
  "tools": [],
  "usage": null
}
```

**(f) HTTP error body** (any 4xx / 5xx; the SDK raises the class from section 9.1 and exposes these four fields):

```json
{"error": {"message": "Rate limit reached for requests", "type": "rate_limit_error", "param": null, "code": "slow_down"}}
```

### 10.6 Streaming appendix (not the recommended path)

If a fake stream is ever needed: `client.responses.create(stream=True, ...)` returns an iterator of typed events; wire format is
server-sent events, `event: <type>\ndata: <json>\n\n`, each JSON carrying `type` and `sequence_number`.
`[DOC: https://developers.openai.com/api/reference/resources/responses/methods/create.md]` (Streaming example)
Event sequence for a response with one reasoning item and one function call (function-call events as in the function-calling
guide; `R0` = the response object with `status: "in_progress"`, `output: []`; `R1` = the final object from 10.2b):

```json
[
  {"type": "response.created", "sequence_number": 0, "response": "R0"},
  {"type": "response.in_progress", "sequence_number": 1, "response": "R0"},
  {"type": "response.output_item.added", "sequence_number": 2, "output_index": 0, "item": {"id": "rs_0f3a9c1e5b7d42a68c0e1f2a3b4c5d6e01", "type": "reasoning", "summary": []}},
  {"type": "response.output_item.done", "sequence_number": 3, "output_index": 0, "item": {"id": "rs_0f3a9c1e5b7d42a68c0e1f2a3b4c5d6e01", "type": "reasoning", "summary": [], "encrypted_content": "gAAAAABpHk3ZfakeEncryptedReasoningBlobOne=="}},
  {"type": "response.output_item.added", "sequence_number": 4, "output_index": 1, "item": {"id": "fc_0f3a9c1e5b7d42a68c0e1f2a3b4c5d6e02", "type": "function_call", "status": "in_progress", "call_id": "call_Qm8xT2vLr4Jd0aWk", "name": "write_rows", "arguments": ""}},
  {"type": "response.function_call_arguments.delta", "sequence_number": 5, "item_id": "fc_0f3a9c1e5b7d42a68c0e1f2a3b4c5d6e02", "output_index": 1, "delta": "{\"palette\":[\"#0b1f4b\",\"#f4d35e\"],"},
  {"type": "response.function_call_arguments.delta", "sequence_number": 6, "item_id": "fc_0f3a9c1e5b7d42a68c0e1f2a3b4c5d6e02", "output_index": 1, "delta": "\"start_row\":0,\"rows\":[\"0010\",\"0111\"]}"},
  {"type": "response.function_call_arguments.done", "sequence_number": 7, "item_id": "fc_0f3a9c1e5b7d42a68c0e1f2a3b4c5d6e02", "output_index": 1, "name": "write_rows", "arguments": "{\"palette\":[\"#0b1f4b\",\"#f4d35e\"],\"start_row\":0,\"rows\":[\"0010\",\"0111\"]}"},
  {"type": "response.output_item.done", "sequence_number": 8, "output_index": 1, "item": {"id": "fc_0f3a9c1e5b7d42a68c0e1f2a3b4c5d6e02", "type": "function_call", "status": "completed", "call_id": "call_Qm8xT2vLr4Jd0aWk", "name": "write_rows", "arguments": "{\"palette\":[\"#0b1f4b\",\"#f4d35e\"],\"start_row\":0,\"rows\":[\"0010\",\"0111\"]}"}},
  {"type": "response.completed", "sequence_number": 9, "response": "R1"}
]
```

(In a real fixture replace the strings `"R0"` / `"R1"` with the full response objects.) A text message streams as
`response.output_item.added` (message, `content: []`) -> `response.content_part.added` -> `response.output_text.delta` ... ->
`response.output_text.done` -> `response.content_part.done` -> `response.output_item.done`.

- Terminal events: `response.completed`, `response.incomplete`, `response.failed`; each has the full `response`. Take the final
  object from whichever arrives. `[SDK: types/responses/response_completed_event.py, response_incomplete_event.py, response_failed_event.py]`
- An `error` event (`{"type": "error", "code", "message", "param", "sequence_number"}`) is **yielded**, not raised; a stream that
  simply ends without a terminal event raises nothing either. The consumer must detect both. `[OFFLINE]`
  The SDK raises `openai.APIError` from the iterator only when a data frame carries a truthy top-level `"error"` key, which
  the flat `error` event above does not have. `[SDK: _streaming.py :: Stream.__stream__]`
  `[SDK: types/responses/response_error_event.py :: ResponseErrorEvent]`
- Helper: `with client.responses.stream(...) as s: final = s.get_final_response()` returns a `ParsedResponse`, but only when a
  `response.completed` event arrived; otherwise `RuntimeError`. `[SDK: lib/streaming/responses/_responses.py]` `[OFFLINE]`
- When streaming, take reasoning items from `response.output_item.done` (or the terminal response): "The `encrypted_content` in
  `response.output_item.added` may be incomplete." `[DOC: .../responses/methods/create.md]`
- The background guide's sample skips events with `event.type == "keepalive"`; that type is not in the installed SDK's event union
  nor in the streaming reference, and the SDK parses it into an unrelated class with `.type == "keepalive"`. `[DOC: .../guides/background.md]` `[OFFLINE]`

---

## 11. Gotchas and surprises

1. **Installed SDK is three months and one major version behind the API** (section 0). It still works; rely on `extra_body`,
   `model_extra`, and string comparisons on `.type` / `.status` / `.reason`.
2. **Tool calling on the current models needs Responses**; Chat Completions cannot do reasoning + tools on GPT-6.x.
3. **Omitting `strict` is not "non-strict"** in Responses: the server tries to normalise the schema into strict mode. Send
   `"strict": false`. Check the echoed `response.tools[0].strict` in a live run.
4. **`arguments` is a string**, may be invalid or truncated JSON (non-strict mode, or `status: "incomplete"`); `json.loads` in a
   `try`. `[OFFLINE]`
5. **`call_id` is not `id`.** Results go back under `call_id` (`call_...`), never the item `id` (`fc_...`).
6. **No finish reason field.** A `completed` response with zero `function_call` items is the "model stopped" signal. Do not use
   "has text" as the signal: a `commentary` message can accompany tool calls, and `output_text` then is non-empty.
7. **`output_text == ""` has four causes**: tool-call-only turn, refusal part, truncation during reasoning, content filter.
8. **Reasoning tokens are inside `output_tokens`** and count against `max_output_tokens`; cached and cache-write tokens are inside
   `input_tokens`. GPT-5.6+ bills cache writes at 1.25x, so `cache_write_tokens` matters for cost.
9. **`get_final_response()` raises on truncation**, and `ParsedResponse` objects (from `responses.stream()` / `responses.parse()`)
   serialise with extra keys: `"parsed": null` on every `output_text` part, and `"parsed_arguments": null` on function calls when
   `to_dict()` is used. `[OFFLINE]` Use plain `responses.create` / `responses.retrieve`, whose `Response` items round-trip
   exactly.
10. **SDK retries are blind**: timeouts and all 429s are retried (default twice). In foreground mode the 10-minute default
    timeout plus retries turns one long call into three aborted attempts; billing 429s are retried pointlessly.
11. **GPT-6.2 drops reasoning after any prefix edit** and after `tools` / `reasoning.effort` / `safety_identifier` changes. Keep the
    conversation append-only and the request parameters constant. Watch `input_transformations`.
12. **Top-level `instructions` is the wrong place for the system prompt on GPT-6.2** (appended after `input`, billed uncached every
    request). Use a first `developer` message.
13. **Background + `store=False`**: the finished response is only retrievable for "roughly 10 minutes". Hence `store=True` in the
    recommended setup.
14. **Long-context pricing** kicks in above 272K input tokens for the whole request (x2 input, x1.5 output). The replayed
    history grows every turn; reasoning "occup[ies] space in the model's context window", so expect it to grow fast at `max`
    effort. Track `usage.input_tokens` per call.
15. **Reasoning summaries are opt-in and may need organisation verification**; do not send `reasoning.summary` by default.
16. **`gpt-6.2-*` reject `effort: "none"`**; `gpt-6-astra` returns HTTP 400 for it. Irrelevant at `max`, relevant if the judge is
    configured with a low effort.
17. **`input_image.image_url` is a string** (data URL), and `detail` should always be sent.
18. **`response.usage` may be `None`**; `response.error` / `incomplete_details` are `None` on success.
19. A client object needs an API key at construction; build it lazily so mocked tests run without `OPENAI_API_KEY`.
20. **`max_output_tokens` may count against tokens-per-minute in full** (section 6): the rate-limit guide says the limit is
    computed from "the maximum of `max_tokens` and the estimated number of tokens" of the request. Keep the cap configurable.
21. **A `failed` or `incomplete` background response raises nothing.** `retrieve` returns normally; only `status`,
    `error` and `incomplete_details` tell the adapter what happened (section 3.5).
22. **Misalignment monitoring covers replayed-reasoning conversations** and can end one with HTTP 403
    `misalignment_policy_violation` (`openai.PermissionDeniedError`); treat it as terminal for that run, never retry
    (section 3.6).

---

## UNVERIFIED (from memory)

Nothing in this section was confirmed against a fetched page or the installed SDK. Treat each line as a hypothesis,
except the two that a live run has since settled, which say so.

- That `reasoning.effort: "max"` is accepted by `gpt-6.2-astra` and `gpt-6.2-sol`. The docs neither list it for these two models
  nor exclude it (they exclude only `none`, and tell `minimal` users to use `low`). Fall back to `xhigh` on HTTP 400.
  **Settled live for `gpt-6.2-astra` on 20 November 2026: it is accepted.** `gpt-6.2-sol` was not tried.
- The error `code` / `param` returned for an unsupported effort value, and for any other 400.
- That the API rejects unknown keys in replayed input items (e.g. `parsed`, `parsed_arguments`). I believe unknown parameters
  produce HTTP 400; avoid sending them regardless.
- That `retrieve` of an expired or unknown background response fails with HTTP 404 (`openai.NotFoundError`).
- That a request containing a `function_call` item without a matching `function_call_output` is rejected, and that a `reasoning`
  item replayed without the item that followed it is rejected. Replaying everything and answering every call avoids both.
- What "roughly 10 minutes" of temporary background storage is measured from (I read it as after completion).
- Whether `GET /v1/responses/{id}` returns `encrypted_content` without the `include` query parameter, and whether reasoning items
  carry `encrypted_content` by default when `store` is `true`. (Written when the adapter sent `include` on both calls. It no
  longer does for a stored response: see the next item.)
- ~~That `include=["reasoning.encrypted_content"]` is accepted together with `store=true`.~~ **Settled live on 20 November
  2026: it is not.** The first request this adapter ever sent (`gpt-6.2-astra`, `store: true`, `background: true`,
  `include: ["reasoning.encrypted_content"]`) was refused three times with HTTP 400, `invalid_request_error`, `param: include`:
  "Encrypted content cannot be requested for persisted responses." No tokens were used. The adapter now sends no `include`
  unless `options.include_encrypted_reasoning` is true, and refuses that option together with `options.store` true. With the
  response stored, the reasoning items that are replayed are found by their ids.
- What exactly delimits a "turn" for `reasoning.context` / `turn_boundary` (assumed: a new user message starts a new turn; tool
  outputs do not).
- That background mode works without `store=true` on every account (the current docs show `background=True, store=False`; I
  recall an older requirement that background responses be stored). `store=True` sidesteps the question.
- The exact JSON of a `queued` background response (10.2a is schema-shaped), and whether `usage` is `null` until completion.
- Whether polling `retrieve` counts against the requests-per-minute limit.
- The server-side maximum duration of a foreground (non-background) request, and whether idle long-lived connections are cut.
- Whether the rate-limit accounting sentence quoted in section 6 (written with the Chat Completions name `max_tokens`) applies
  unchanged to Responses `max_output_tokens`, and whether reasoning models are counted the same way.
- That `include=["reasoning.encrypted_content"]` is rejected for models without reasoning support (I recall a 400 for
  non-reasoning models). Every GPT-6.x model page read for section 8 lists "Reasoning token support", so this should not
  arise.
- Whether a truncated (`incomplete`, `max_output_tokens`) response can contain a `function_call` item with partial `arguments`
  and `status: "incomplete"`, and whether the output of an incomplete response may be replayed to continue. Treat an incomplete
  response as terminal for that step.
- Behaviour when `max_output_tokens` exceeds the model maximum (128,000); stay at or below it.
- The default `reasoning.context` for GPT-6.2 and whether sending `"all_turns"` to a model that lacks it is an error.
- Whether `phase` is populated on GPT-6.2 assistant messages.
- Image sizing rules and the token multiplier for `gpt-6.2-astra`, `gpt-6.2-sol`, `gpt-6.1-terra`, `gpt-6-luna` (the docs' tables
  list `gpt-6-astra`; I assume the family shares them).
- Whether a refusal in plain-text mode (no JSON schema) is delivered as a `refusal` part or as ordinary `output_text`.
- Id prefixes (`resp_`, `rs_`, `fc_`, `call_`, `msg_`) are taken from doc examples; treat all ids as opaque strings.
- How many input tokens a replayed reasoning item costs on the next request (the token numbers in the section 10 fixtures are
  illustrative, not measured).
- Whether `pro` reasoning mode can be combined with function tools.

---

## Sources

Fetched 2026-11-19 (some just after midnight UTC, 2026-11-20). `.md` variants of the official pages.

Official documentation (read in full or in the relevant sections):

- https://developers.openai.com/api/docs/llms.txt (index)
- https://developers.openai.com/api/docs/guides/latest-model/gpt-6.2-astra.md
- https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md
- https://developers.openai.com/api/docs/guides/latest-model/gpt-5.6.md
- https://developers.openai.com/api/docs/guides/reasoning.md
- https://developers.openai.com/api/docs/guides/function-calling.md
- https://developers.openai.com/api/docs/guides/background.md
- https://developers.openai.com/api/docs/guides/conversation-state.md
- https://developers.openai.com/api/docs/guides/keep-prefix-unchanged.md
- https://developers.openai.com/api/docs/guides/deployment-checklist.md
- https://developers.openai.com/api/docs/guides/migrate-to-responses.md
- https://developers.openai.com/api/docs/guides/text.md
- https://developers.openai.com/api/docs/guides/streaming-responses.md
- https://developers.openai.com/api/docs/guides/images-vision.md
- https://developers.openai.com/api/docs/guides/image-cost-calculator.md
- https://developers.openai.com/api/docs/guides/flex-processing.md
- https://developers.openai.com/api/docs/guides/model-selection.md
- https://developers.openai.com/api/docs/guides/safety-checks.md
- https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring.md
- https://developers.openai.com/api/docs/models.md
- https://developers.openai.com/api/docs/models/compare.md
- https://developers.openai.com/api/docs/models/gpt-6.2-astra.md
- https://developers.openai.com/api/docs/models/gpt-6.2-sol.md
- https://developers.openai.com/api/docs/models/gpt-6.1-terra.md
- https://developers.openai.com/api/docs/models/gpt-6-luna.md
- https://developers.openai.com/api/docs/models/gpt-6.1-sol.md
- https://developers.openai.com/api/docs/models/gpt-6-astra.md
- https://developers.openai.com/api/docs/models/gpt-6-sol.md
- https://developers.openai.com/api/docs/pricing.md
- https://developers.openai.com/api/docs/changelog.md (November 2026 entries)

Official documentation (searched for specific passages only):

- https://developers.openai.com/api/docs/guides/structured-outputs.md (strict-schema rules, refusal handling)
- https://developers.openai.com/api/docs/guides/prompt-caching.md (usage accounting, model differences table)
- https://developers.openai.com/api/docs/guides/token-counting.md (output token counts)
- https://developers.openai.com/api/docs/guides/your-data.md (`store`, background retention)
- https://developers.openai.com/api/docs/guides/error-codes.md
- https://developers.openai.com/api/docs/guides/rate-limits.md
- https://developers.openai.com/api/docs/guides/safety-checks/cybersecurity.md
- https://developers.openai.com/api/docs/models/gpt-5.5-pro.md and https://developers.openai.com/api/docs/models/gpt-5.4-pro.md (background-mode advice)
- https://developers.openai.com/api/reference/resources/responses/methods/create.md (body parameters, Response object, examples)
- https://developers.openai.com/api/reference/resources/responses/methods/retrieve.md (query parameters)
- https://developers.openai.com/api/reference/resources/responses/streaming-events.md (list of event type names only)
- https://learn.chatgpt.com/docs/models.md (reached by redirect from https://developers.openai.com/codex/models.md; product page, used only as a pointer about "Max" effort, not as API evidence)

SDK repository (current `main`, for comparison with the installed version):

- https://raw.githubusercontent.com/openai/openai-python/main/CHANGELOG.md
- https://raw.githubusercontent.com/openai/openai-python/main/src/openai/_version.py
- https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/shared/chat_model.py
- https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/shared/responses_model.py
- https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/shared/reasoning.py
- https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/responses/response_create_params.py
- https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/responses/response_usage.py
- https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/responses/response_reasoning_item.py
- https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/responses/function_tool_param.py
- https://raw.githubusercontent.com/openai/openai-python/main/src/openai/lib/streaming/responses/_responses.py
- https://raw.githubusercontent.com/openai/openai-python/main/src/openai/_constants.py

Installed SDK files read (under `site-packages/`, 2026-11-19):

- `openai/_version.py`, `openai/_constants.py`, `openai/_client.py`, `openai/_base_client.py`, `openai/_exceptions.py`,
  `openai/_streaming.py`, `openai/_models.py` (`to_dict`), `openai/_utils/_transform.py`
- `openai/resources/responses/responses.py` (`create`, `stream`, `retrieve`, `cancel`, `_make_tools`),
  `openai/resources/responses/input_tokens.py`
- `openai/lib/streaming/responses/_responses.py`, `openai/lib/_parsing/_responses.py`
- `openai/types/shared/reasoning.py`, `reasoning_effort.py`, `chat_model.py`, `responses_model.py`
- `openai/types/responses/`: `response.py`, `response_create_params.py`, `response_usage.py`, `response_status.py`,
  `response_error.py`, `response_includable.py`, `response_output_item.py`, `response_output_message.py` (+ `_param`),
  `response_output_text.py` (+ `_param`), `response_output_refusal.py` (+ `_param`), `response_reasoning_item.py` (+ `_param`),
  `response_function_tool_call.py` (+ `_param`), `response_input_item_param.py`, `easy_input_message_param.py`,
  `response_input_text_param.py`, `response_input_image_param.py`, `response_input_message_content_list_param.py`,
  `response_function_call_output_item_list_param.py`, `function_tool.py`, `function_tool_param.py`,
  `tool_choice_options.py`, `tool_choice_function_param.py`, `parsed_response.py`, `response_stream_event.py`,
  and the event files `response_created_event.py`, `response_in_progress_event.py`, `response_queued_event.py`,
  `response_completed_event.py`, `response_incomplete_event.py`, `response_failed_event.py`, `response_error_event.py`,
  `response_output_item_added_event.py`, `response_output_item_done_event.py`,
  `response_function_call_arguments_delta_event.py`, `response_function_call_arguments_done_event.py`,
  `response_content_part_added_event.py`, `response_text_delta_event.py`, `response_text_done_event.py`
- `openai-2.53.0.dist-info/METADATA` (the SDK's own README, as shipped with the installed version: usage, vision, errors, retries, timeouts)
- `httpx/_config.py` (`Timeout`), `httpx/_exceptions.py`
- `httpcore/_backends/sync.py` (`SyncStream.read`; added in the independent pass)

Offline experiments (installed SDK + `httpx.MockTransport`, no network): request-body serialisation for create / retrieve /
cancel; `extra_body` merging; item `to_dict()` round trip and replay; unknown fields, item types, enum values and stream events;
`responses.stream()` / `get_final_response()` on completed vs incomplete streams; `ParsedResponse` serialisation; error-class
mapping and retry counts for 400 / 403 / 408 / 409 / 429 / 500 / 503; timeout exception class; raw-response headers.

---

## Verification (independent pass)

Done 2026-11-20 by a second reviewer, who did not write the notes. No model/inference endpoint was called. Every `[DOC]` page
named below was downloaded again as markdown (generic User-Agent, HTTP 200) and searched for the exact passages; `[SDK]`
claims were checked by reading the installed `openai 2.53.0` source or, for a few behavioural ones (`with_options`,
`with_raw_response`, `to_dict`), by exercising it; the `[OFFLINE]` claims an adapter depends on were re-run against
`httpx.MockTransport`.

### What was checked, and how

Installed SDK (`site-packages/openai/`), read directly:

- `_version.py` (2.53.0), `_constants.py` (timeout 600 s / connect 5 s, `max_retries` 2, backoff 0.5 s to 8 s, 120 s
  `Retry-After` ceiling), `_client.py` (environment variables, credential check, `auth_headers`, `default_headers`,
  `_make_status_error`), `_base_client.py` (`_should_retry`, `_calculate_retry_timeout`, `_parse_retry_after_header`,
  `_idempotency_header = None`), `_exceptions.py`, `_streaming.py`, `resources/responses/responses.py` (`create` body keys,
  `retrieve`, `cancel`, `stream`; no polling helper and no long-request streaming rule exist).
- `types/shared/reasoning.py`, `reasoning_effort.py`, `chat_model.py`, `responses_model.py`; `types/responses/response.py`,
  `response_status.py`, `response_usage.py`, `response_error.py`, `response_includable.py`, `response_output_message.py`,
  `response_output_text.py`, `response_output_refusal.py`, `response_reasoning_item.py`, `response_function_tool_call.py`,
  `function_tool_param.py`, `easy_input_message_param.py`, `response_input_text_param.py`, `response_input_image_param.py`,
  `response_input_message_content_list_param.py`, `response_input_item_param.py` (`FunctionCallOutput`),
  `tool_choice_options.py`, `tool_choice_allowed_param.py`, `tool_choice_function_param.py`, `response_error_event.py`,
  `lib/streaming/responses/_responses.py`, `lib/_parsing/_responses.py`; `openai-2.53.0.dist-info/METADATA`.
- Confirmed absent from 2.53.0 by searching `types/` and `resources/responses/`: `background_options`, `access_programs`,
  `ultrafast`, `input_transformations`, `prompt_cache_diagnostics`, `tool_usage`, `cache_write_tokens_details`,
  `max_messages`, `steered`, `misalignment_policy_violation`, `configuration_update`, function-tool `async`.
- `../httpcore/_backends/sync.py :: SyncStream.read` (read timeout is set per socket read).

Offline runs (installed SDK, no network):

- Every JSON block in this file parses as JSON. All eight `Response` fixtures (10.2a, 10.2b, 10.4, 10.5 a-e) pass
  **strict** `Response.model_validate(...)`, not only the SDK's lenient construction; `status`, item types and `output_text`
  match the "expected" text under each fixture; a recursive scan for `model_extra` found no unknown field anywhere except the
  top-level `store` key (which the API reference examples also return and 2.53.0 does not type). Each item's `to_dict()`
  reproduces the fixture item exactly.
- The item snippets in 3.2 to 3.4 validate as `ResponseOutputItem`; the usage block in 3.7 as `ResponseUsage`; the ten
  stream events in 10.6 as `ResponseStreamEvent`, each landing in the expected event class.
- Request fixtures 2.1, 7.1, 10.1, 10.3: every key at every level exists in the SDK parameter types
  (`Responses.create` signature, `FunctionToolParam`, `Reasoning`, `EasyInputMessageParam`, `ResponseInputTextParam`,
  `ResponseInputImageParam`, `ResponseReasoningItemParam`, `ResponseFunctionToolCallParam`, `FunctionCallOutput`). Sending
  10.1 and 7.1 through `client.responses.create(**fixture)` produced a request body equal to the fixture. Running the 4.2
  procedure on 10.1 + 10.2b (create, `retrieve(..., include=[...])`, `to_dict()` replay, append `function_call_output`)
  produced a second request body equal to 10.3, both with dicts and with SDK objects in `input`.
- Paths and methods: `POST /v1/responses`, `GET /v1/responses/{id}?include[]=reasoning.encrypted_content`,
  `POST /v1/responses/{id}/cancel`; `Authorization: Bearer <key>`; `_request_id` from `x-request-id`; `extra_body` merge.
- Error mapping for 400 / 401 / 403 / 404 / 408 / 409 / 422 / 429 / 500 / 503 and the `code` / `type` / `param` /
  `request_id` attributes; attempts with default retries (429, 500, 408, 409: 3 attempts; 400: 1; 429 with
  `Retry-After: 300`: 1; read timeout: 3 attempts then `APITimeoutError`; connect error: 3 then `APIConnectionError`).
- Tolerance: unknown top-level fields, an `output_text` key in the JSON, unknown `incomplete_details.reason`, unknown output
  item type, usage without `cache_write_tokens`, extra fields on `response.error`, truncated `arguments` string.
- Streaming: full event sequence, flat `error` event (yielded), nested `error` key (raises `APIError`), stream cut short
  (no exception), `keepalive` (parsed into an unrelated class), `get_final_response()` on an `incomplete` stream
  (`RuntimeError`), `ParsedResponse` extra keys (`parsed`, `parsed_arguments`).

Documentation re-fetched and searched for the quoted passages (all under `https://developers.openai.com/api/docs/` unless
a full URL is given): `llms.txt`, `guides/latest-model/gpt-6.2-astra.md`, `guides/latest-model/gpt-6-astra.md`,
`guides/latest-model/gpt-5.6.md`, `guides/reasoning.md`, `guides/function-calling.md`, `guides/background.md`,
`guides/conversation-state.md`, `guides/keep-prefix-unchanged.md`, `guides/deployment-checklist.md`,
`guides/migrate-to-responses.md`, `guides/text.md`, `guides/images-vision.md`,
`guides/image-cost-calculator.md`, `guides/flex-processing.md`, `guides/model-selection.md`,
`guides/safety-checks/misalignment-monitoring.md`, `guides/safety-checks/cybersecurity.md`, `guides/structured-outputs.md`,
`guides/prompt-caching.md`, `guides/token-counting.md`, `guides/your-data.md`, `guides/error-codes.md`,
`guides/rate-limits.md`, `models.md`, `models/compare.md`, the model pages `models/gpt-6.2-astra.md`, `gpt-6.2-sol.md`,
`gpt-6.1-terra.md`, `gpt-6-luna.md`, `gpt-6.1-sol.md`, `gpt-6-astra.md`, `gpt-6-sol.md`, `gpt-5.5-pro.md`, `pricing.md`,
`changelog.md`; the API reference pages
`https://developers.openai.com/api/reference/resources/responses/methods/create.md`, `.../retrieve.md`, `.../cancel.md`
and `https://developers.openai.com/api/reference/resources/responses/streaming-events.md`; and from the SDK repository
`https://raw.githubusercontent.com/openai/openai-python/main/CHANGELOG.md`, `src/openai/_version.py` (3.56.0) and the
`main` versions of `types/shared/chat_model.py`, `types/shared/reasoning.py`, `types/shared/reasoning_effort.py`,
`types/responses/response.py`, `response_usage.py`, `response_error.py`, `function_tool_param.py`.

Confirmed against those pages: endpoint choice and the "tool calling requires Responses" statements; the flat function-tool
shape and the `strict` default rules; `tool_choice` / `parallel_tool_calls`; `function_call` / `function_call_output` shapes
and the replay loop; the replay rules and `input_transformations` reasons; `reasoning.effort` values, the per-model lists
and defaults in 5.1, `reasoning.summary` / `context` / `mode`; `max_output_tokens` semantics and the 25,000-token advice;
`status` values, `incomplete_details.reason` values, response error codes; usage accounting and the cache-write multipliers;
background mode, polling loop, cancel, `store` retention wording; context window / max input / max output for all seven
GPT-6.x models; every price in section 8 and the long-context, Batch, Fast and completion-window rules; rate limits for
`gpt-6.2-astra`; image part shapes, `detail` values, file types, 1,500 images, 512 MB, 30,000 patches, the 1.2 multiplier
and the worked token examples; 429 / 503 codes and rate-limit headers; SDK release dates (2.53.0 on 2026-08-03, 3.0.0 on
2026-08-12, 3.56.0 on 2026-11-19) and the GPT-6.2 release entry (2026-11-17).

No wrong field name, nesting, enum value, default, limit, model id or price was found in the original notes.

### Corrections made (old -> new)

1. Section 2.4, and the tool schema in 2.1, 10.1, 10.2a, 10.2b, 10.3. Old: "`write_rows` has only optional properties and
   no `required` list", with example schemas that had no `required`. New: only `palette` is optional (as `write_rows_tool()` in paintbench/prompts.py
   defines the tool); the example schemas carry `"required": ["start_row", "rows"]`; the reason strict mode does not fit is
   stated correctly (optional property, no `additionalProperties: false`). All fixtures were re-validated after the change.
2. Section 9.1. Old: no row for HTTP 408. New: 408 raises the base `openai.APIStatusError` (there is no dedicated
   subclass) and is retried by the SDK.
3. Section 3.5. Old: `response.error` described as `{"code", "message"}` only. New: the current API also returns optional
   `param`, `type` and `misalignment` on that object (untyped in 2.53.0, readable via `model_extra`).
4. Section 3.5 table. Old: "HTTP status was still 200" for `failed`. New: stated as what it means for the adapter (a
   successful `retrieve`, no exception; check `status`). The original wording was an inference with no source.
5. Section 3.4. Old: reasoning item described by the fixture keys only, and "a response typically starts with a reasoning
   item". New: lists the optional keys (`content`, `status`, `encrypted_content`), notes that the docs' examples show
   `"content": []`, and replaces "typically starts with" (unsourced generalisation) with what the doc example shows.
6. Section 3.6 and 4.5. Old: two speculative sentences inside verified sections (plain-text refusals "may also simply
   arrive as ordinary `output_text`"; "the natural reading" of what a turn is). New: both reduced to a pointer; the
   hypotheses stay under UNVERIFIED only.
7. Section 3.6 item 4. Added the documented fact that misalignment monitoring can block conversations that use persisted
   reasoning (which this adapter does), with the "match the error code" instruction.
8. Section 1.2. Old: the missing-credentials message was abbreviated and the check described as "no key". New: full message;
   notes that `OPENAI_ADMIN_KEY` alone satisfies the constructor; adds the `OpenAI-Organization` / `OpenAI-Project` headers.
9. Section 1.3 and UNVERIFIED. Old: "httpx applies the read timeout per socket read" listed as unverified. New: moved into
   1.3 with the source (`httpcore/_backends/sync.py :: SyncStream.read`) and the offline timeout result.
10. Section 1.4. Added: Zero Data Retention forces `store=false` behaviour regardless of the value sent.
11. Section 5.1. Added the concrete doc example behind "examples go up to `xhigh`", and the rule that the `xhigh` fallback may
    only happen on the first request of a conversation.
12. Section 6 and gotcha 20 (new). Added the rate-limit accounting statement: a 128,000 `max_output_tokens` cap may count
    in full against tokens-per-minute.
13. Section 7.2. Added: GPT-6.2 has no row in the image sizing table; the docs' `gpt-6.2-astra` examples use
    `"detail": "auto"`; keep `detail` configurable.
14. Section 9.3. Added the two extra non-retryable codes used in the rate-limit guide's sample
    (`insufficient_quota`, `billing_hard_limit_reached`).
15. Section 10.6. Added when the SDK does raise during streaming (data frame with a top-level `error` key).
16. Section 11. Added gotchas 20 to 22. UNVERIFIED: removed the httpx item (now verified), added two items (rate-limit
    accounting for `max_output_tokens`; `include` on non-reasoning models).

### Still unverified

(As this pass left it. The adapter has since been run against the live API, on 20 November 2026 with `gpt-6.2-astra`.
That settled the first two items below: `gpt-6.2-astra` accepts `"max"`, and `include` beside `store=true` is refused.
The adapter sends the effort as written, with no fallback, and sends no `include` for a stored response.)

Everything under "UNVERIFIED (from memory)" remains unverified. None of it can be settled without a live call. The items
that could make a first live run fail, most likely first:

1. `reasoning.effort: "max"` on `gpt-6.2-astra` / `gpt-6.2-sol`. No fetched page lists the accepted values for these two
   models; every other GPT-6.x page lists `max`. Mitigation in place: config value plus first-request fallback to `xhigh`
   (which the docs do show with `gpt-6.2-astra`).
2. `include=["reasoning.encrypted_content"]` together with `store=true`, and whether `GET /v1/responses/{id}` returns
   `encrypted_content`. The docs describe the include value only for stateless use. With `store=true` the replayed
   reasoning items also carry ids of stored items, which is the documented default replay (`input_list += response.output`
   in the function-calling guide runs with `store` at its default).
3. The exact JSON of `queued` / `in_progress` background responses (10.2a is built from the schema) and whether `usage` is
   null until completion.
4. How `max_output_tokens` is counted against tokens-per-minute.
5. `detail: "high"` and image token cost on the GPT-6.2 models.
6. The error `code` / `param` of any 400 (unsupported effort, unknown key, orphaned `function_call`), and the error raised
   when a background response has expired.
7. Behaviour of an `incomplete` response that contains a partial `function_call`.

Also not verifiable here: that the live API behaves as its documentation says. The fixtures prove that an adapter written
from these notes parses what the installed SDK types and the reference examples describe, not what the service returns.
