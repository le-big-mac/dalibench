# Anthropic Messages API: provider notes for the `anthropic` adapter

Researched 2026-11-19/20 for paintbench. Written from documentation and from the installed SDK source. While
they were written no request was sent to any model endpoint: the only executed code was offline, the installed
SDK driven through `httpx.MockTransport` (no network), to see what the SDK itself sends, raises and assembles.

The adapter has since been run against the live API. The statements tagged `[live]` were added on 20 November
2026, after the first live run. They are the only ones here that rest on requests sent to the API, and "Live
observations" under "Sources" says what was sent and what came back.

How to read the evidence tags:

- `[doc:NAME]` = an official page fetched for these notes (full URL in "Sources").
- `[sdk:PATH]` = a file of the installed `anthropic` package, version 0.121.0 (the path is relative to the package directory).
- `[probe]` = behaviour of the installed SDK observed offline with a mocked transport (what the SDK does, not what the API does).
- `[live]` = behaviour of the API itself, observed in requests that were sent to it on 20 November 2026 (what the API
  did for those requests on that day, not a documented guarantee). Each is listed under "Live observations" in
  "Sources". "Run 1" and "run 2" in these notes are the two full runs of that day, described there.
- Anything not backed this way is in "UNVERIFIED (from memory)" at the end and nowhere else.

Three facts to keep in mind while reading:

1. **The installed SDK (0.121.0) is older than the current docs.** The docs describe SDK 2.x (latest Python SDK is 2.11.0
   `[doc:sdk-versions]`), which sends `anthropic-version: 2026-10-21`, sends API keys as `Authorization: Bearer`, and names
   its HTTP library `httpx2` in code samples `[doc:migration]` `[doc:py-sdk]`. The installed 0.121.0 sends
   `anthropic-version: 2023-06-01`, sends API keys as `X-Api-Key`, and uses `httpx` `[sdk:_client.py]` `[probe]`. Both API
   versions are listed as valid `[doc:versioning]`. Everything in these notes is for 0.121.0 unless it says otherwise.
2. **The current models are Claude Fable 5.2, Opus 5.5, Sonnet 5.5 and Haiku 5.5** `[doc:models]`. Material from
   mid-2026 and before lists Fable 5 / Opus 5 / Sonnet 5 / Haiku 4.5 and old prices. Where the two disagree, these
   notes follow the documentation as it was fetched.
3. **Current models bind thinking blocks to the conversation prefix.** Changing `system`, `tools` or any earlier message
   after the model produced a thinking block makes that block invalid, and Claude Fable 5.2 answers with a 400
   `[doc:prefix-checks]`. The adapter and harness must be strictly append-only (section 4).

---

## 1. Endpoint, SDK method, client, timeouts, streaming

### Endpoint and method

- Endpoint: `POST https://api.anthropic.com/v1/messages` `[doc:api-ref]`. Used the way these notes describe, it is
  stateless: every request carries the whole conversation `[doc:api-ref]`. (A beta feature, "message threads", header
  `message-threads-2026-08-12`, can keep a conversation on the server; we do not use it `[doc:message-threads]`.)
- SDK method to use for every call (contestant and judge): **`client.messages.stream(**kwargs)`** as a context manager,
  then **`stream.get_final_message()`**. It returns one assembled `ParsedMessage` (a subclass of `anthropic.types.Message`)
  with the same fields a non-streaming call returns `[sdk:resources/messages/messages.py]` `[sdk:lib/streaming/_messages.py]`
  `[doc:streaming]` `[probe]`.
- Why not `client.messages.create()`: for a non-streaming call the SDK computes `expected = 3600 s * max_tokens / 128000`
  and raises before sending if that exceeds 600 s, i.e. for **`max_tokens >= 21334`**
  `[sdk:_base_client.py::_calculate_nonstreaming_timeout]` `[sdk:resources/messages/messages.py]` `[probe: 21333 passes, 21334 raises]`:

  ```text
  ValueError: Streaming is required for operations that may take longer than 10 minutes. See https://github.com/anthropics/anthropic-sdk-python#long-requests for more details
  ```

  The guard only runs when `stream` is falsy, no per-request `timeout=` was given, and the client's timeout is still the
  default; passing any `timeout` (client or request) silences it `[sdk:resources/messages/messages.py]` `[probe]`. The docs
  call it "a client-side validation, not an API restriction" and say to use `.stream()` + `.get_final_message()` when you
  do not need the events `[doc:thinking]` `[doc:errors]`. Do not silence the guard and go non-streaming: the docs warn that
  some networks drop idle connections and say to stream requests that may run longer than 10 minutes `[doc:errors]`.
- There is no background mode on the Messages API. The documented options for long requests are streaming and the Message
  Batches API (asynchronous, most batches finish within 1 hour, results within 24 hours, 50% price)
  `[doc:errors]` `[doc:batches]` `[doc:pricing]`. Batches do not fit an interactive tool loop; use streaming.
- Documented risk that streaming does not remove: "For workloads that push thinking above roughly 32k tokens per
  request, use batch processing to avoid networking issues: such requests can run long enough to hit system timeouts and
  open connection limits" `[doc:thinking]`. A contestant call at `max` effort can exceed that, so expect some long calls to
  die mid-stream and make sure they are retried (section 9). The API can also answer `timeout_error` (504, or a 500
  "when a request reaches the server's time limit before the response starts") `[doc:errors]`.

### Client construction: Workload Identity Federation (from a Google Cloud machine)

`anthropic.WorkloadIdentityCredentials` exists in 0.121.0 (re-exported from `anthropic.lib.credentials`) `[sdk:__init__.py]`
`[sdk:lib/credentials/_workload.py]`. Constructor (all keyword-only):

```python
WorkloadIdentityCredentials(
    *,
    identity_token_provider: Callable[[], str],   # returns the external OIDC JWT; called on every exchange
    federation_rule_id: str,                      # "fdrl_..."
    organization_id: str,                         # raw UUID
    service_account_id: Optional[str] = None,     # "svac_..."
    workspace_id: Optional[str] = None,           # "wrkspc_..."
    scope: Optional[str] = None,                  # informational only, not sent
    http_client: Optional[httpx.Client] = None,   # client used for the token exchange only
)
```

The pattern to use on a Google Cloud machine whose identity a federation rule trusts. The ids are kept in
`~/.anthropic/wif-<name>.env`, one `KEY=VALUE` per line, under a short name of the user's choosing (`research`
below). Pass them explicitly and do not put them in `os.environ`. `ANTHROPIC_API_KEY` must not be set in the
environment at all:

```python
import os
import httpx
import google.auth.transport.requests, google.oauth2.id_token
from anthropic import Anthropic, WorkloadIdentityCredentials

def wif_ids(name="research"):
    ids = {}
    with open(os.path.expanduser(f"~/.anthropic/wif-{name}.env")) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1); ids[k] = v
    return ids

def google_token():
    return google.oauth2.id_token.fetch_id_token(
        google.auth.transport.requests.Request(), "https://api.anthropic.com")

ids = wif_ids("research")
client = Anthropic(
    credentials=WorkloadIdentityCredentials(
        identity_token_provider=google_token,
        federation_rule_id=ids["ANTHROPIC_FEDERATION_RULE_ID_GCP"],
        organization_id=ids["ANTHROPIC_ORGANIZATION_ID"],
        service_account_id=ids["ANTHROPIC_SERVICE_ACCOUNT_ID"],
        workspace_id=ids["ANTHROPIC_WORKSPACE_ID"],
    ),
    base_url="https://api.anthropic.com",          # REQUIRED here, see below
    timeout=httpx.Timeout(1800.0, connect=10.0),   # see "Timeouts"
    max_retries=1,                                 # see section 9
)
```

What the SDK does with this `[sdk:_client.py]` `[sdk:lib/credentials/_auth.py]` `[sdk:lib/credentials/_cache.py]`
`[sdk:lib/credentials/_workload.py]` `[doc:wif]`:

- **`base_url` must be passed explicitly.** Precedence is `base_url=` kwarg > `ANTHROPIC_BASE_URL` > profile > default
  `https://api.anthropic.com`, and the client then calls `credentials.bind_base_url(<resolved base_url>)`, so the token
  exchange goes to the same host as the API calls. With `ANTHROPIC_BASE_URL` exported and no kwarg, both the API calls
  and the token exchange (`POST {base_url}/v1/oauth/token`) go to that address `[probe: construct-only, no request sent]`.
  `bind_base_url` rejects non-https hosts other than localhost.
- When any of `api_key=`, `auth_token=`, `credentials=`, `config=`, `profile=` is passed, the constructor does not read
  `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` from the environment `[sdk:_client.py]` `[probe]`. Without an explicit
  argument those variables win over federation, even when set to an empty string `[doc:wif]`.
- The provider is wrapped in a thread-safe `TokenCache`: the token is minted lazily on the first request, refreshed
  when under 120 s remain (best effort) and always when under 30 s remain. Requests get `Authorization: Bearer <token>`
  plus `anthropic-beta: oauth-2025-04-20` (merged with any beta header you set).
- On a 401 the client invalidates the cached token and retries once, but only if `max_retries >= 1` (the check sits
  behind `remaining_retries > 0`) `[sdk:_client.py::_should_retry]` `[sdk:_base_client.py::request]`.
- A failed token exchange raises `anthropic.WorkloadIdentityError` (subclass of `AnthropicError`, not of `APIError`;
  attributes `status_code`, `body`, `request_id`). The SDK does not retry it. Every assertion denial is an opaque 401
  with message `Authentication failed` `[doc:wif]`.
- The minted token is scoped to one workspace; the `anthropic-workspace-id` header is not supported with federation
  tokens `[sdk:lib/credentials/_workload.py]`.

### Client construction: API key

```python
from anthropic import Anthropic
client = Anthropic()                      # reads ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) and ANTHROPIC_BASE_URL
client = Anthropic(api_key="sk-ant-...", base_url="https://api.anthropic.com", timeout=..., max_retries=1)
```

- Environment variables read by the constructor: `ANTHROPIC_API_KEY` (sent as `X-Api-Key` in 0.121.0),
  `ANTHROPIC_AUTH_TOKEN` (sent as `Authorization: Bearer`), `ANTHROPIC_BASE_URL`, `ANTHROPIC_CUSTOM_HEADERS`
  `[sdk:_client.py]` `[probe]`. The API accepts an API key in either `Authorization: Bearer <key>` or the legacy
  `x-api-key` header `[doc:auth]`.
- Raw HTTP, if ever needed: headers `x-api-key` (or `Authorization: Bearer`), `anthropic-version`, `content-type:
  application/json` `[doc:api-ref]` `[doc:versioning]`.
- Select the auth path explicitly in config (the adapter's `options.auth`: `{type: api_key, env: NAME}` or
  `{type: wif, org: NAME}`); do not fall back from one to the other silently.

### Timeouts

- SDK default: `httpx.Timeout(timeout=600, connect=5.0)`, i.e. connect 5 s and read / write / pool 600 s each
  `[sdk:_constants.py]` `[probe]`. The read timeout is the longest allowed gap between received bytes, not a cap on the
  whole request. There is no total-duration timeout.
- A streamed response is kept alive by SSE traffic; streams "might also include any number of `ping` events"
  `[doc:streaming]`. With thinking display `omitted` (the default on current models) no thinking text is streamed at all
  during the thinking phase `[doc:thinking]`. The docs give no ping interval (see UNVERIFIED), so I recommend raising the
  read timeout, for example `timeout=httpx.Timeout(1800.0, connect=10.0)`, and asking for `display: "summarized"` so
  summary deltas flow while the model thinks (section 5). This is a judgement call, not a documented requirement.
- The SDK's default HTTP client enables TCP keep-alive (`SO_KEEPALIVE`, idle 60 s, interval 60 s, 5 probes), which
  surfaces a dead peer as a read error regardless of the read timeout. This is lost if you pass your own `http_client`
  (a plain `httpx.Client`, or one built with a custom `transport`) `[sdk:_base_client.py::_DefaultHttpxClient]` `[doc:py-sdk]`.
- A wall-clock cap per call cannot be enforced reliably from inside the event loop (below). Leaving the `with` block
  does close the connection `[sdk:lib/streaming/_messages.py]` `[probe]`, but the loop body only runs when the SDK yields
  an event, and the SDK swallows `ping` events without yielding them (`if sse.event == "ping": continue`
  `[sdk:_streaming.py]`). During a silent thinking phase no event is yielded, so a deadline check placed in the loop
  does not run until the next real event arrives. The read timeout is not a wall-clock cap either: any received bytes,
  pings included, restart it. If a hard cap is needed it has to come from outside the loop (for example a timer that
  closes the stream); that path was not tested here.
- WIF token lifetime (relevant to long calls): the minted token lives for the lesser of the federation rule's
  `token_lifetime_seconds` (default 3,600 s) and twice the remaining lifetime of the identity token presented, never
  less than 60 s `[doc:wif-guide]`. What happens to a stream that is still open when its token expires is not
  documented (see UNVERIFIED).

### The simplest reliable call

Verified offline against the installed SDK with a mocked SSE stream `[probe]`:

```python
raw_tool_json: dict[int, str] = {}     # content index -> concatenated partial_json of that tool_use block
thinking_tokens = None
with client.messages.stream(**kwargs) as stream:
    for event in stream:
        if event.type == "content_block_delta" and event.delta.type == "input_json_delta":
            raw_tool_json[event.index] = raw_tool_json.get(event.index, "") + event.delta.partial_json
        elif event.type == "message_delta" and event.usage.output_tokens_details is not None:
            thinking_tokens = event.usage.output_tokens_details.thinking_tokens
    message = stream.get_final_message()
    request_id = stream.request_id                 # "request-id" response header
    headers = stream.response.headers              # rate-limit headers live here
data = message.to_dict(mode="json")                # plain JSON, only the fields the API sent (plus SDK nulls, see 4)
```

Why the loop instead of calling `get_final_message()` straight away (all `[probe]` on 0.121.0):

1. `get_final_message().usage.output_tokens_details` stays `None` even when the final `message_delta` event carries it.
   The SDK's accumulator copies `output_tokens`, `input_tokens`, the cache counters and `server_tool_use` from
   `message_delta`, but not `output_tokens_details` `[sdk:lib/streaming/_messages.py::accumulate_event]`. The docs say the
   thinking-token breakdown "appears only on the final `message_delta` event" when streaming `[doc:thinking-steering]`.
   So read it from the event.
2. When a reply is cut off inside a tool call, the SDK parses the partial JSON leniently and hands back a shortened but
   well-formed `input` (in the test: two rows sent, the second cut, `input` held one row and no error). Keep the raw
   string so a truncated call can be recognised (section 3).
3. **Do not read `message._request_id` on a stream-assembled message: it raises `AttributeError`**
   (`'ParsedMessage' object has no attribute '_request_id'`). The attribute is only attached to objects returned by
   non-streaming calls `[sdk:_models.py::add_request_id]` `[probe]`. Use `stream.request_id`.
4. To hand `parse_response()` the same shape a non-streaming response has, write the value read from the event back
   into the dict: `if thinking_tokens is not None: data["usage"]["output_tokens_details"] = {"thinking_tokens":
   thinking_tokens}`. The result still validates as `anthropic.types.Message` `[probe]`.
5. `get_final_message()` raises a bare `AssertionError` when the stream delivered no events at all (HTTP 200 with an
   empty body), and the event loop raises `AttributeError: 'NoneType' object has no attribute 'output_tokens'` when a
   `message_start` or `message_delta` event has no `usage` `[probe]`. Neither is an `anthropic` exception; catch them
   around the whole block and treat them as a failed (retryable) call.

Async is the same shape: `AsyncAnthropic(...)`, `async with client.messages.stream(...) as stream`,
`async for event in stream`, `await stream.get_final_message()` `[sdk:lib/streaming/_messages.py]` `[doc:py-sdk]`.

---

## 2. Request shape for a contestant call

Keyword arguments for `client.messages.stream()`; the SDK adds `"stream": true` to the body itself. This exact body was
captured from the SDK offline `[probe]`; field names are confirmed in `[doc:api-ref]` and `[sdk:types/*_param.py]`.

```json
{
  "model": "claude-fable-5-2",
  "max_tokens": 128000,
  "system": "You are a contestant in a pixel-painting benchmark. Paint the named painting by calling write_rows.",
  "thinking": {"type": "adaptive", "display": "summarized"},
  "output_config": {"effort": "max"},
  "tools": [
    {
      "name": "write_rows",
      "description": "Write consecutive rows of the canvas, top to bottom, starting at start_row. Each string in rows is one row; each character is one cell and must be a key of the palette.",
      "input_schema": {
        "type": "object",
        "properties": {
          "palette": {"type": "array", "items": {"type": "string"}, "description": "Optional. Palette entries such as \"a=#1b2a6b\"."},
          "start_row": {"type": "integer", "description": "Zero-based index of the first row being written."},
          "rows": {"type": "array", "items": {"type": "string"}, "description": "Rows to write, one string per row."}
        }
      }
    }
  ],
  "messages": [
    {"role": "user", "content": "Painting: The Starry Night (Vincent van Gogh, 1889). Canvas: 8 columns x 4 rows."}
  ]
}
```

Field by field:

| Field | Type | Notes |
| --- | --- | --- |
| `model` | string | Required. Exact id, section 8. |
| `max_tokens` | integer | Required. Hard cap on thinking plus visible output of this one request `[doc:thinking-steering]`. Section 6. |
| `system` | string, or array of `{"type":"text","text":...}` | Top-level, optional. The initial system prompt goes here and nowhere else `[doc:api-ref]`. Omit the key when there is no system prompt. Must stay identical for the whole conversation (section 4). |
| `messages` | array of `{"role": "user" \| "assistant", "content": string \| array of blocks}` | A string is shorthand for one text block. Consecutive same-role messages are combined into one turn by the API. Limit 100,000 messages `[doc:api-ref]`. The conversation must end with a user message: current models reject a trailing assistant message (prefill) with a 400 `[doc:errors]`. A third role, `"system"`, exists for mid-conversation system messages (`{"role": "system", "content": string \| array of text blocks}`), but it has placement rules (it goes after the newest user message; "right after Claude's turn or right before a user message" it is a 400) and is not available on every model `[doc:api-ref]` `[doc:change-tools]` `[doc:mid-system]`. Do not use it: never map the harness's leading `role="system"` message to a `messages` entry. |
| `tools` | array | One custom tool: `name` (regex `^[a-zA-Z0-9_-]{1,128}$`), `description` (optional string), `input_schema` (JSON Schema object; `type: "object"` required, `properties` and `required` optional) `[doc:api-ref]` `[doc:define-tools]` `[sdk:types/tool_param.py]`. A schema with no `required` list is valid. Must stay identical for the whole conversation, down to the wording of descriptions (section 4). |
| `tool_choice` | object | Leave it out: the default is `{"type": "auto"}` `[doc:define-tools]`. Other values: `{"type":"none"}`, `{"type":"any"}`, `{"type":"tool","name":"..."}`. **`any` and `tool` return a 400 on Claude Fable 5.2, Opus 5.5 and Sonnet 5.5** (`tool_choice: type "tool" and "any" are not supported for this model.`) `[doc:errors]`. On models that accept forced tool use it skips thinking `[doc:define-tools]`. |
| parallel tool calls | flag inside `tool_choice` | On by default: one reply may hold several `tool_use` blocks. To limit to at most one call per reply send `"tool_choice": {"type": "auto", "disable_parallel_tool_use": true}`. It is not a top-level parameter `[doc:parallel]` `[sdk:types/tool_choice_auto_param.py]`. We leave the default. |
| `thinking` | object | Section 5. |
| `output_config` | object | `{"effort": "low" \| "medium" \| "high" \| "xhigh" \| "max"}`. Top-level; effort does not go inside `thinking` `[doc:effort]` `[sdk:types/output_config_param.py]`. |
| `temperature`, `top_p`, `top_k` | | Never send. Any non-default value is a 400 on all current models; `top_p: 1` is non-default (the default is 0.99); sending `temperature` and `top_p` together is a 400 even at defaults; under API version 2026-10-21 any of the three is a 400 on every model `[doc:thinking]` `[doc:migration]`. |
| `stop_sequences`, `metadata`, `service_tier`, `cache_control`, `container`, `inference_geo` | | Optional, not needed `[sdk:resources/messages/messages.py]`. |

Strict schema mode: a tool may carry `"strict": true`, which constrains sampling so `input` always matches the schema
`[doc:strict]`. **Do not use it here.** It requires `"additionalProperties": false` on objects, supports only a subset of
JSON Schema (no `minimum`/`maximum`, no `minLength`/`maxLength`, `minItems` only 0 or 1), counts every optional parameter
against a limit of 24, adds grammar-compilation latency on first use, and is unavailable for Claude Fable 5.2 in
organizations with customer-managed encryption keys `[doc:structured-outputs]` `[doc:fable-migration]`. It would also make
this provider's tool schema differ from the others. Without strict mode the model can send a wrong type or omit a field
`[doc:strict]`, so validate `input` in the harness and answer with an error result (section 4).

The API builds a tool-use system prompt from the tool definitions; that overhead is billed as input tokens (286 tokens
on Opus 5.5 / Sonnet 5.5 / Haiku 5.5 with `auto`; Fable 5.2 is not in that table) `[doc:pricing]`.

Do not word any prompt or tool field so that it asks the model to write out its reasoning ("think step by step in your
answer", a `reasoning` field in the tool input): current models can refuse that with category `reasoning_extraction`
`[doc:refusals]`.

---

## 3. Response shape

A `Message` `[doc:api-ref]` `[sdk:types/message.py]` (skeleton; complete examples are in section 10):

```text
{
  "id": "msg_01A1B2C3D4E5F6G7H8J9K0LM",
  "type": "message",
  "role": "assistant",
  "model": "claude-fable-5-2",
  "content": [ <blocks, see below> ],
  "stop_reason": "tool_use",
  "stop_sequence": null,
  "stop_details": null,
  "usage": { <see "Usage"> }
}
```

### Content blocks

`content` is an ordered array. Never read `content[0].text`; select by `type` `[doc:fable-migration]`. With one custom tool
and no beta features the types are:

| `type` | Fields | Meaning |
| --- | --- | --- |
| `thinking` | `thinking` (string, may be `""`), `signature` (string, opaque) | Reasoning or a progress note. Text is a summary or empty, never the raw chain of thought `[doc:thinking]`. |
| `redacted_thinking` | `data` (string, opaque) | Safety-redacted reasoning `[doc:thinking]`. |
| `text` | `text` (string), `citations` (null or array) | Visible answer text. A reply can have zero, one or several text blocks; concatenate them in order. |
| `tool_use` | `id` (string, pattern `^[a-zA-Z0-9_-]+$`, e.g. `toolu_01...`), `name` (string), `input` (**JSON object**, not a string), `caller` (object, `{"type": "direct"}` for a normal call) | One tool call `[doc:api-ref]` `[doc:handle-tool-calls]` `[sdk:types/tool_use_block.py]`. |

- Tool-call arguments are an object. When streaming, they arrive as `input_json_delta` string fragments and the SDK
  parses them; the final `tool_use.input` "is always an object" `[doc:streaming]`. With no arguments, `input` is `{}`.
- A reply can start with `thinking` blocks, mix `text` and `tool_use`, and hold several `tool_use` blocks. On Claude
  Fable 5.2, Opus 5.5 and Sonnet 5.5 the short notes the model writes between tool calls come back as extra `thinking`
  blocks ("progress updates"), not as `text` `[doc:thinking]` `[doc:fable-migration]`. So a `tool_use` reply often has no
  `text` block at all.
- With adaptive thinking the model may skip thinking; do not assume a reply starts with a thinking block
  `[doc:thinking-steering]`.
- `content` can be an empty array: on a refusal, and occasionally with `stop_reason: "end_turn"` after tool results
  `[doc:refusals]` `[doc:stop-reasons]`.

### `stop_reason` values and what to do

Complete list `[doc:stop-reasons]` `[doc:api-ref]` `[sdk:types/stop_reason.py]`:

| Value | Meaning | Action |
| --- | --- | --- |
| `end_turn` | The model finished its turn. | Final reply. If the painting is incomplete, send the nudge (section 4). Content may be empty. |
| `tool_use` | The model stopped to have client tools run. | Run every `tool_use` block, send all results back (section 4). |
| `max_tokens` | Output hit `max_tokens` (thinking counts). | Truncated. Text may be cut, there may be no text at all, and a trailing `tool_use` block may be incomplete `[doc:stop-reasons]` `[doc:thinking-troubleshooting]`. See below. |
| `model_context_window_exceeded` | Generation filled the context window (input + output). Returned instead of an error on Claude 4.5 and later `[doc:context-windows]`. | Treat as truncated, like `max_tokens`. |
| `refusal` | A safety classifier declined. HTTP 200, not an error. | Discard any partial output; read `stop_details`; do not retry on the same model. |
| `pause_turn` | A server-tool loop hit its iteration limit. "When Claude stops to call a tool that you run, the stop reason is `tool_use`, never `pause_turn`" `[doc:stop-reasons]`. | Should not occur with one client tool. If it does: append the content unchanged as an assistant message and re-send with no new user message. |
| `stop_sequence` | A custom stop sequence was produced. | Cannot occur; we send no `stop_sequences`. |
| `null` | Only inside `message_start` while streaming. | If the assembled message still has `stop_reason` `None`, the stream ended after `message_start` but without a `message_delta`: the SDK raises nothing in that case `[probe]`. Treat it as a failed call and retry. (A stream with no events at all raises `AssertionError` instead, section 1.) |

Truncation (`max_tokens`):

- Check `stop_reason` before acting on tool calls. The documented remedy for an incomplete `tool_use` block is to
  re-send the request with a higher `max_tokens` `[doc:stop-reasons]`; at 128,000 there is no higher value.
- The documented way to continue a truncated turn is: "pass the assistant turn back unchanged and append a new `user`
  message (with a `tool_result` for each `tool_use` block in that turn)" `[doc:thinking]`.
- **A reply cut off while the model was still thinking is continued the same way.** It is the commonest truncation
  at `max` effort: `stop_reason: "max_tokens"` and a `content` of one `thinking` block and nothing else. The block
  comes with its signature: all 291 such replies of run 1 had one, on six models `[live]`. Passed back unchanged
  (null-valued keys dropped) with a user message after it, the turn **was accepted** on `claude-haiku-5-5`,
  `claude-fable-5-2` and `claude-sonnet-5`, and the next call's input was about 2,535 tokens larger, for 2,500
  tokens of thinking and a 29-token reminder, so the earlier thinking was in the context `[live]`. An assistant turn
  whose only block, and so its last, is a signed thinking block is therefore not rejected when a user message
  follows it. Never leave such a turn out. The adapter of run 1 did, and that made every call of 55 attempts
  start its reasoning again from the task: four calls of 128,000 output tokens each, nothing typed, the input
  growing by the 29 tokens of the reminder and nothing else `[live]`.
- **At full size, through the adapter itself, the same held.** In the small check of 20 November 2026 (Live
  observations, item 3) four attempts at effort `max` each had a first reply of one signed thinking block cut off at
  128,000 output tokens. The adapter sent each back with the reminder, and each second request was accepted with
  about 128,035 more input tokens: on `claude-opus-5-5`, `claude-fable-5-2` and `claude-sonnet-5` the model then
  called the tool and the picture was completed; on `claude-sonnet-5-5` the request was accepted and the reply ended
  in `stop_reason: "refusal"` after 18,074 output tokens, which is the platform's block and not a rejected turn
  `[live]`.
- Not observed, and sent on the documented rule alone: a turn that ends in a signed thinking block after a `text` or
  `tool_use` block; a turn made only of `redacted_thinking`; `claude-fable-5` with a thinking-only turn (its one
  attempt in the check was blocked on its first call); `claude-haiku-5-5` with one larger than 2,500 tokens.
- For the last `tool_use` block of a `max_tokens` reply, compare with the raw string collected in section 1: if
  `json.loads(raw_tool_json[index])` fails, the call was cut. Pass the raw string on as the call's arguments (the
  harness's canvas then rejects it and returns an error result) rather than the SDK's shortened object `[probe]`.
  The replay payload keeps the SDK's shortened object, because a replayed `tool_use.input` must be a JSON object
  `[doc:api-ref]`. Whether a turn that holds such a cut call can be replayed is not documented (UNVERIFIED, item 6).
  Run 1 replayed nine of them, each with a `tool_result` for every call, and each next call was answered `[live]`.
- "By default, the API buffers each key and value in `input` until it is complete", then emits it as chunked
  `partial_json` deltas `[doc:streaming]`. What that means for the exact shape of a truncated argument string is not
  documented (see UNVERIFIED); apply the `json.loads` test above either way.
- A thinking block whose `signature` is `""` (its `signature_delta` never arrived) cannot be replayed: a signature
  that was "truncated, altered, or sent back empty" fails verification with a 400 `[doc:thinking-troubleshooting]`.
  The stream helper keeps such a block in the final message `[probe]`, so check for it on a truncated reply.

### Refusals and safety blocks

A refusal is HTTP 200 with `stop_reason: "refusal"` and a `stop_details` object `[doc:refusals]` `[sdk:types/refusal_stop_details.py]`:

```json
{
  "stop_reason": "refusal",
  "stop_details": {
    "type": "refusal",
    "category": "cyber",
    "explanation": "This request was declined because it could enable cyber harm."
  },
  "content": []
}
```

- `category`: `"cyber"`, `"bio"`, `"frontier_llm"`, `"reasoning_extraction"`, `"general_harms"` or `null`.
  `explanation`: string or `null`; its wording is not stable. `stop_details` is `null` for every other stop reason.
- Branch on `stop_reason == "refusal"`, not on `content` or on the inner fields `[doc:refusals]`.
- A refusal can arrive before any output (`content` empty) or mid-stream after partial output; "treat any partial
  output as incomplete and discard it" `[doc:refusals]`. When streaming, `stop_details` arrives in the `message_delta`
  event and the SDK copies it onto the final message `[sdk:lib/streaming/_messages.py]` `[probe]`.
- Re-sending to the same model "usually earns another refusal" `[doc:refusals]`. The API offers fallback to another
  model (beta `fallbacks` parameter, SDK middleware); a benchmark should not use it, since another model would answer.
- Billing: a pre-output refusal is billed for categories `bio`, `frontier_llm`, `reasoning_extraction`, not for the
  others or `null`; a mid-stream refusal bills input plus the output already streamed; all count against rate limits
  `[doc:refusals]`.
- A model can also decline in plain words with `stop_reason: "end_turn"`; that is ordinary text and only the harness's
  text heuristic can see it.

### Usage

`[doc:api-ref]` `[sdk:types/usage.py]` `[doc:thinking-steering]`

```json
{
  "usage": {
    "input_tokens": 1042,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
    "output_tokens": 2610,
    "output_tokens_details": {"thinking_tokens": 2480},
    "server_tool_use": null,
    "service_tier": "standard",
    "inference_geo": "global"
  }
}
```

| Field | Meaning |
| --- | --- |
| `input_tokens` | Input tokens that were neither written to nor read from the cache. |
| `cache_creation_input_tokens`, `cache_read_input_tokens` | Cache writes and cache reads (number or null). **Total input = `input_tokens` + `cache_creation_input_tokens` + `cache_read_input_tokens`** `[doc:api-ref]`. Both are 0 or null unless the request asks for caching (`cache_control`) `[doc:prompt-caching]`. |
| `output_tokens` | Everything billed as output: thinking plus text plus tool calls. "The inclusive, authoritative total used for billing." |
| `output_tokens_details.thinking_tokens` | The part of `output_tokens` spent on reasoning. **Included in `output_tokens`, never additional**; always <= `output_tokens`. Counts the raw reasoning, not the summary you see. The object may be null. When streaming it is only on the final `message_delta` (see section 1 for the SDK gap). |
| `cache_creation`, `server_tool_use`, `service_tier`, `inference_geo`, `speed` | Informational; may be null or absent. `speed` (`"standard"` / `"fast"` / null) is in the API reference but not in 0.121.0's `Usage` type; the SDK keeps it as an extra field, and it appears in `to_dict()` `[doc:api-ref]` `[probe]`. Read `usage` with `.get()`. |

Other points: the billed output count does not match the visible text, because thinking is billed in full whatever
`display` says `[doc:thinking-steering]`. Thinking blocks from earlier turns that are replayed are billed again as input
on every later request `[doc:thinking]`. `message_delta` usage counts are cumulative `[doc:streaming]`.

Suggested mapping onto paintbench's `Usage`: `input_tokens` = the three-way sum above; `output_tokens` =
`usage.output_tokens`; `reasoning_tokens` = `thinking_tokens` or `None`; `cached_input_tokens` = `cache_read_input_tokens`.

---

## 4. The tool-result turn, and the nudge

### The rules

All from `[doc:handle-tool-calls]`, `[doc:parallel]`, `[doc:thinking]`, `[doc:prefix-checks]`, `[doc:change-tools]`:

1. Keep every earlier message exactly as sent. Append the reply's **whole `content` array, unchanged and in order**, as
   one assistant message: `{"role": "assistant", "content": <content>}`.
2. Append **one** user message holding one `tool_result` block for **every** `tool_use` block of that reply, including
   calls that failed or that you did not run.
3. Send it with the same `system`, the same `tools`, the same `thinking` and `output_config`.
4. If the next reply's `stop_reason` is `tool_use` again, repeat.

What must be passed back verbatim for reasoning to stay valid:

- Every `thinking` block with its `signature` and its `thinking` text exactly as received, **including blocks whose
  `thinking` is `""`** (the default `display` is `omitted`, so that is the normal case). The signature carries the
  encrypted reasoning; an empty block is complete `[doc:thinking]` `[doc:change-tools]`.
- Every `redacted_thinking` block with its `data`. Filtering on `type == "thinking"` alone drops these and breaks the
  protocol `[doc:thinking]`.
- The order of blocks. Thinking blocks may not be rearranged, edited, or partly dropped: "the sequence of consecutive
  `thinking` blocks must match what the model generated" `[doc:thinking]`.
- The `tool_use` blocks (`id`, `name`, `input`, and `caller` when present) and the `text` blocks.
- A modified thinking block gets a 400: ``messages.1.content.0: `thinking` or `redacted_thinking` blocks in the latest
  assistant message cannot be modified. These blocks must remain as they were in the original response.`` `[doc:errors]`.
- Interleaved thinking (thinking between tool calls) is automatic with adaptive thinking; no beta header
  `[doc:thinking]`. A tool-use loop is one assistant turn from the model's point of view, and the thinking configuration
  must not change inside it `[doc:thinking]`.

### The prefix check: stay append-only

On Claude Fable 5.2, Opus 5.5, Sonnet 5.5 and Haiku 5.5 each thinking block's signature records a fingerprint of
everything before it: `system`, `tools`, and every earlier message. Each request that sends thinking blocks back is
checked `[doc:prefix-checks]` `[doc:preserved-thinking]`:

- A check fails if `system`, `tools` or an earlier message changed after the block was produced, or if a thinking block
  between two others was left out or moved. "A change can be as small as one word of a tool description." Inserting,
  moving or deleting an earlier message counts, whatever its role.
- Result: **400 on Claude Fable 5.2**; "possible 400" on Opus 5.5 / Sonnet 5.5 / Haiku 5.5 (otherwise the stale blocks
  silently reach the model). Error text: ``messages.{i}.content.{j}: Invalid `signature` in `thinking` block. The block
  is bound to a different conversation. Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to
  "drop_block".`` Retrying the same body fails the same way. Match only on `The block is bound to a different
  conversation` `[doc:fix-bound]` `[doc:errors]`.
- What does **not** count: `model`, `effort`, `max_tokens`, `tool_choice`, `metadata`, `thinking.display`, JSON
  formatting or key order ("the API compares parsed JSON"), `cache_control` markers `[doc:prefix-checks]`.
- The three habits that make failures impossible: send each assistant turn back exactly as returned; build `system`
  and `tools` once per conversation and send that copy every time; put every update in a new message at the end
  `[doc:prefix-checks]`.

Consequences for the adapter:

- `build_request()` must be a deterministic function of the message list: the same neutral messages must always
  produce the same `system`, `tools` and earlier `messages`. No timestamps, no re-wording, no trimming of old tool
  results, no shortening of old turns.
- Store the assistant content as plain JSON and replay that. A dict payload goes through the SDK untouched, unknown
  keys included `[probe]`.
- A recovery switch exists (`thinking.block_binding.prefix_mismatch_behavior: "drop_block"` with beta header
  `thinking-binding-controls-2026-08-01`) `[doc:prefix-checks]`. Do not build on it; an append-only adapter never needs it.
  For a benchmark a prefix failure is a bug to surface, and the harness's "discard the conversation and start again"
  path is the right response.
- Thinking blocks from Claude Fable 5.2, Sonnet 5.5 and Haiku 5.5 only work in the organization that produced them (or
  one linked to it); another organization's request has them dropped silently `[doc:preserved-thinking]`. Keep one conversation on one
  organization: do not carry it on with another organization's credentials.
- Do not route requests through a gateway or proxy that rewrites bodies `[doc:change-tools]`.

### What to store as the replay payload

```python
data = message.to_dict(mode="json")
payload = [{k: v for k, v in block.items() if v is not None} for block in data["content"]]
```

Reason for dropping null-valued keys (top level of each block only; never touch `input`): the SDK's stream accumulator
materialises optional fields the stream did not send as explicit nulls, so a stream-assembled text block carries
`"citations": null` and a `tool_use` block whose stream event had no `caller` carries `"caller": null` `[probe]`. The
reference allows `citations` to be null but lists `caller` on a `tool_use` input block as optional without "or null"
`[doc:api-ref]`. Omitting an optional key is valid in every case; sending a null there is not documented. Keys with real
values (`caller: {"type": "direct"}`, `thinking: ""`, `signature`) are kept. The official pattern is to pass
`message.content` straight back (`{"role": "assistant", "content": message.content}`), which the SDK's own tool runner
also does `[doc:thinking-tool-workflows]` `[sdk:lib/tools/_beta_runner.py]`; the filtered dicts are the same content minus
those nulls. For the record: with 0.121.0 that official pattern, applied to a stream-assembled message, puts
`"citations": null` and `"caller": null` on the wire `[probe]`, so both forms are in circulation; neither was sent to
the API here (see UNVERIFIED).

### `tool_result` format

```json
{
  "role": "user",
  "content": [
    {"type": "tool_result", "tool_use_id": "toolu_01AAAAAAAAAAAAAAAAAAAAAA", "content": "OK: rows 0-1 written (2 of 4)."},
    {"type": "tool_result", "tool_use_id": "toolu_01BBBBBBBBBBBBBBBBBBBBBB", "content": "Rejected: row 2 has 7 cells, expected 8.", "is_error": true}
  ]
}
```

- `tool_use_id` (required): the `id` of the `tool_use` block it answers. `content` (optional): a string, or an array of
  `text` / `image` / `document` / `search_result` blocks; an empty result is expressed by leaving `content` out.
  `is_error` (optional boolean): set `true` when the call failed or was invalid; the model may then call again
  `[doc:handle-tool-calls]` `[sdk:types/tool_result_block_param.py]`.
- There is no `tool` role. Results are blocks inside a **user** message `[doc:handle-tool-calls]`.
- `is_error` is optional. paintbench's neutral tool message (`paintbench/types.py::tool_result`) carries only the
  result text and no error flag, so the adapter has nothing to set it from; leave it out rather than guess from the
  text. The model reads the rejection in `content` either way.
- `tool_use_id` must match `^[a-zA-Z0-9_-]+$` `[doc:api-ref]`.

Ordering rules `[doc:handle-tool-calls]` `[doc:parallel]`:

- The user message with the results must come immediately after the assistant message with the calls; no message in
  between. Otherwise: `tool_use ids were found without tool_result blocks immediately after`.
- Inside that user message the `tool_result` blocks come first. Text before a `tool_result` is a 400. Text after all
  results is allowed, but do not add any: "Never add text blocks immediately after tool results", because the model
  then learns to end its turn after each tool call `[doc:stop-reasons]`.
- Several tool calls in one reply: return **all results together in one user message**, one `tool_result` per
  `tool_use`, matched by `tool_use_id`. A separate user message per result is listed as wrong (it also suppresses
  parallel calls). Execution order is up to you; if a call is not run, still return a result for it with
  `is_error: true` and a short reason `[doc:parallel]`.

paintbench keeps one neutral `role="tool"` message per call, so the adapter must **merge each run of consecutive tool
messages into a single user message**, in the order of the `tool_use` blocks.

### The follow-up user message (the nudge)

After an assistant reply with no tool call (`end_turn`), append the assistant content (same payload rule) and then a
plain user message:

```json
[
  {"role": "assistant", "content": [
    {"type": "thinking", "thinking": "", "signature": "EqQBCgIYAhIM1gbcDa9G...opaque-signature-2...=="},
    {"type": "text", "text": "The painting is complete: all 4 rows are written."}
  ]},
  {"role": "user", "content": "Rows 2-3 are still missing. Write them with write_rows."}
]
```

- Passing thinking blocks back across turns is "recommended"; within a tool-use turn it is "required" `[doc:thinking]`.
  Always replay the full payload; the API decides what to keep.
- If the reply's content is an empty array, do not add an assistant message for it; append only the new user message.
  This is the documented remedy for empty `end_turn` replies ("add a continuation prompt in a new user message"), and
  the API combines consecutive user messages into one turn `[doc:stop-reasons]` `[doc:api-ref]`. Two user messages
  in a row were accepted in 236 requests of run 1 `[live]`. Whether an empty assistant message would be rejected was
  never tried (UNVERIFIED, item 4).
- If a reply holds only `thinking` / `redacted_thinking` blocks (no `text`, no `tool_use`), it is **not** empty:
  append it like any other turn, thinking block last, and put the user message after it. A reply cut off by
  `max_tokens` while thinking is the usual case, and the API accepts the turn (section 3, `[live]`). An earlier
  version of these notes recommended skipping such a turn, as a defence against a rejection remembered and never
  checked. That advice was wrong: with the turn skipped, the model never sees its own reasoning again. The 55
  attempts of run 1 that ended incomplete were four such replies each, and no call of theirs was shown the
  thinking of the call before `[live]`.
- The only thinking to leave out is the reasoning at the end of a turn (the `thinking` and `redacted_thinking`
  blocks after its last `text` or `tool_use` block) when one of its `thinking` blocks has a `signature` that is
  missing or `""` (section 3). It is then left out whole, the signed blocks in it included: that reasoning goes
  back whole (documented, and seen live for one signed block) or not at all (what the adapter did before, accepted
  in 236 requests of run 1), and never in part, which nothing documents and nothing has tried. No reply of run 1
  or run 2 held a thinking block without a signature, so this has not arisen. Leaving out the newest thinking is
  explicitly allowed; leaving out a block between two kept ones is not `[doc:change-tools]`. If that leaves the
  turn with nothing in it, the empty-reply rule above applies. Whichever way a turn goes, it goes the same way in every later request (the prefix check).
- A turn passed back this way must have a user message after it. A request whose last message is an assistant turn
  is a prefill, a 400 on current models `[doc:errors]`; the adapter refuses to build one. The harness appends the
  tool results or the nudge after every reply it goes on from, and the judge its one follow-up question. (The
  `pause_turn` re-send of section 3 is the one documented case of a last assistant message. It cannot arise with one
  client tool, and the adapter would refuse it as well.)
- After `stop_reason: "refusal"` nothing is passed back at all: the harness and the judge stop there, so a refused
  reply's thinking is stored in the transcript and never sent.
- What it costs: the thinking passed back is input on every later call (section 6), about 128,000 tokens for each
  reply cut off at `max_tokens: 128000`. After three such replies the fourth call carries about 385,000 input
  tokens, inside the 1M window; on Claude Haiku 5.5 any request over 100,000 input tokens is billed at the higher
  rate (section 8). Run 1 showed turns of that size going back: nine replies of 128,000 output tokens that ended in
  a cut tool call were replayed, and each next call reported about 128,000 input tokens `[live]`.
- Text blocks must not be empty: `text` has `minLength: 1` `[doc:api-ref]`. Never send `{"type": "text", "text": ""}`.
- New instructions always go in a new message at the end; never edit `system` or the first user message
  `[doc:change-tools]`.

### Mapping from paintbench's neutral messages (suggestion)

| Neutral message | Anthropic request |
| --- | --- |
| `role="system"` | top-level `system` string (same text on every call) |
| `role="user"` | `{"role": "user", "content": [...]}`: `Text` -> `{"type": "text", "text": ...}`, `Image` -> image block (section 7) |
| `role="assistant"`, `provider == "anthropic"` | `provider_payload` itself, verbatim (the adapter stores the whole message, `{"role": "assistant", "content": [...]}`), a turn of signed thinking alone included; skipped only when the payload holds nothing to send back (no `text`, no `tool_use`, no signed `thinking`, no `redacted_thinking`) |
| `role="assistant"` from elsewhere | rebuilt: one `text` block if the text is non-empty, then one `{"type": "tool_use", "id", "name", "input"}` per call (`input` must be an object; ids must match `^[a-zA-Z0-9_-]+$`) |
| run of `role="tool"` | one `{"role": "user", "content": [tool_result, ...]}` |

---

## 5. Reasoning settings

Parameters `[doc:thinking]` `[doc:effort]` `[doc:api-ref]`:

| Parameter | Values | Notes |
| --- | --- | --- |
| `thinking.type` | `"adaptive"`, `"enabled"` (needs `budget_tokens`), `"disabled"`, `"between_tools"` | Per-model table below. |
| `thinking.display` | `"summarized"`, `"omitted"`; `"updates"` is beta (header `thinking-display-updates-2026-08-18`) | Only changes what text you see. Billing and signatures are identical; may be switched between turns. Default on all four current models: `"omitted"` (thinking blocks arrive with `thinking: ""`). |
| `thinking.budget_tokens` | integer >= 1024 | Only with `type: "enabled"`. Legacy. |
| `output_config.effort` | `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"` | The control for how much the model thinks and writes. |

What each current model does with each `thinking` value `[doc:thinking]`:

| Model | no `thinking` field | `adaptive` | `enabled` + `budget_tokens` | `between_tools` | `disabled` |
| --- | --- | --- | --- | --- | --- |
| Claude Fable 5.2 | adaptive | adaptive | 400 | 400 | 400 |
| Claude Opus 5.5 | adaptive | adaptive | 400 | 400 | 400 |
| Claude Sonnet 5.5 | adaptive | adaptive | 400 | up-front thinking off, effort `high` or below only | 400 |
| Claude Haiku 5.5 | adaptive | adaptive | 400 | 400 | thinking off, effort `high` or below only |

So:

- **`budget_tokens` still exists in the API but is rejected (400) by every current model.** It works only on older
  models: Opus 4.5, Sonnet 4.5, Haiku 4.5; deprecated on Opus 4.6 and Sonnet 4.6. Error text:
  `"thinking.type.enabled" is not supported for this model. Use "thinking.type.adaptive" and "output_config.effort" to
  control thinking behavior.` `[doc:thinking]` `[doc:errors]`.
- Thinking cannot be turned off on Fable 5.2 and Opus 5.5 at all.
- Send `thinking: {"type": "adaptive"}` explicitly (same as omitting it on these four models, and it also turns thinking
  on for Opus 4.8 / 4.7 / 4.6 and Sonnet 4.6, where the default is off). Add `"display": "summarized"` to get readable
  summaries in transcripts and steadier stream traffic.

Effort `[doc:effort]` `[doc:models]`:

| Model | Allowed effort values | Default | Highest |
| --- | --- | --- | --- |
| `claude-fable-5-2` | low, medium, high, xhigh, max | `high` | `max` |
| `claude-opus-5-5` | low, medium, high, xhigh, max | `medium` | `max` |
| `claude-sonnet-5-5` | low, medium, high, xhigh, max | `high` | `max` |
| `claude-haiku-5-5` | low, medium, high, xhigh, max | `medium` | `max` |

- **Highest setting for every current model: `thinking: {"type": "adaptive"}` + `output_config: {"effort": "max"}`.**
  `max` is "absolute maximum capability with no constraints on token spending"; on Fable 5.2 the docs note it "runs
  slower and costs more than at `xhigh`, often for only a small gain" `[doc:effort]`. That is still the highest setting.
- Older models: `max` exists on Opus 4.6 and Sonnet 4.6 but `xhigh` does not; Opus 4.5 supports effort alongside
  `budget_tokens`; Haiku 4.5 and Sonnet 4.5 do not support effort `[doc:effort]`.
- Setting effort to the model's default is identical to omitting it `[doc:effort]`.
- Effort is guidance, not a budget. No level guarantees a thinking block on every request; follow-up requests that only
  process tool results can skip thinking even at `max` `[doc:thinking-steering]`.
- Keep `thinking` and `effort` constant for a whole conversation; you cannot toggle thinking inside a tool loop, and
  changes break prompt caching `[doc:thinking]`.

Interaction with `max_tokens`: **thinking counts against `max_tokens`**, which is a hard cap on thinking plus response
text of one request; each request in a tool loop has its own cap. At `high` and above "set a large `max_tokens`". A
too-small value ends in `stop_reason: "max_tokens"`, possibly after a thinking block and before any text
`[doc:thinking-steering]` `[doc:effort]` `[doc:fable-migration]`.

Rejected alongside thinking / on current models `[doc:thinking]` `[doc:errors]`:

- `temperature`, `top_p`, `top_k` at non-default values: 400 (see section 2). Never send them.
- Forced `tool_choice` (`any`, `tool`): 400 on Fable 5.2, Opus 5.5, Sonnet 5.5.
- Prefilled last assistant message: 400 on all current models.
- `thinking.display` together with `type: "disabled"` or `"between_tools"`: invalid.
- `xhigh` / `max` effort together with `disabled` (Haiku 5.5) or `between_tools` (Sonnet 5.5): 400.

SDK note: 0.121.0's typed dicts know only `adaptive` / `enabled` / `disabled` and `summarized` / `omitted`, but request
params are not validated at runtime, so other documented values pass through unchanged `[sdk:types/thinking_config_param.py]`
`[probe]`.

---

## 6. Output limits and context windows

`[doc:models]` `[doc:thinking]` `[doc:context-windows]`

| Model | Context window | Max output (`max_tokens` ceiling, synchronous API) |
| --- | --- | --- |
| `claude-fable-5-2` | 1,000,000 tokens | 128,000 |
| `claude-opus-5-5` | 1,000,000 | 128,000 |
| `claude-sonnet-5-5` | 1,000,000 | 128,000 |
| `claude-haiku-5-5` | 1,000,000 | 128,000 |
| older: Opus 5 / 4.8 / 4.7 / 4.6, Sonnet 5 / 4.6 | 1,000,000 | 128,000 |
| older: Opus 4.5, Sonnet 4.5, Haiku 4.5 | 200,000 | 64,000 |

- The 1M window is the default; no beta header; standard pricing across the window except Claude Haiku 5.5, whose
  prompts over 100,000 tokens cost more (section 8).
- "Set the limit high" = **`max_tokens: 128000`** for the four current models. The docs recommend exactly that for
  Sonnet 5.5 agentic work ("set `max_tokens` to 128,000, the model's maximum, and stream the response") and "a large
  `max_tokens`" at high effort for the others `[doc:effort]`. Any value above 21,333 needs the streaming path (section 1).
- `max_tokens` is per request, not per conversation.
- A high `max_tokens` costs nothing in rate-limit terms: output-token limits count only tokens actually generated;
  "the `max_tokens` parameter does not factor into OTPM rate limit calculations" `[doc:rate-limits]`.
- If input plus `max_tokens` exceeds the window, current models accept the request and stop with
  `model_context_window_exceeded` if generation actually reaches the limit `[doc:context-windows]`.
- Replayed thinking blocks stay in context and are billed as input on these models ("keep all prior turns"), so a long
  tool loop at `max` effort grows quickly `[doc:thinking]`.
- Limits can be read at runtime from the Models API (`client.models.retrieve(id)` -> `max_input_tokens`, `max_tokens`,
  `capabilities`) `[doc:models]`.
- The Message Batches API can go to 300k output with a beta header on some models; not relevant here `[doc:models]`.

---

## 7. Vision input for the judge

One user message whose `content` is an array mixing text and image blocks in any order. This is the documented
multi-image pattern, labels included ("introduce each one with a short text label") `[doc:vision]`; the block shapes were
confirmed by capturing what the SDK serialises `[probe]`:

```json
{
  "model": "claude-opus-5-5",
  "max_tokens": 32000,
  "thinking": {"type": "adaptive"},
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Reference painting:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgoAAAANSUhEUg..."}},
        {"type": "text", "text": "Entry A:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgoAAAANSUhEUg..."}},
        {"type": "text", "text": "Which entry is closer to the reference? Answer in plain text."}
      ]
    }
  ]
}
```

- Image block: `{"type": "image", "source": {"type": "base64", "media_type": <mime>, "data": <base64 string>}}`.
  `media_type` is one of `image/jpeg`, `image/png`, `image/gif`, `image/webp`. `data` is standard base64 of the file
  bytes with no `data:` prefix and no line breaks (`base64.standard_b64encode(bytes).decode("ascii")`)
  `[doc:vision]` `[sdk:types/base64_image_source_param.py]`. Other source types: `{"type": "url", "url": ...}` and Files
  API `file_id`; not needed.
- No tools: leave `tools` out. The answer is the concatenation of the `text` blocks; on Fable 5.2 and Opus 5.5 thinking
  is always on, so expect `thinking` blocks first and size `max_tokens` for thinking plus the answer.
- Text labels must be non-empty (`minLength: 1`) `[doc:api-ref]`.
- "Claude works best when images come before text. Images placed after text or interpolated with text still perform
  well" `[doc:vision]`.
- The judge prompt is exposed to the same refusal as contestant prompts: on Claude Fable 5.2, Opus 5.5 and Sonnet 5.5
  a request that asks for the model's reasoning in the response text (a `<reasoning>` section, a `reasoning` field,
  "reasoning verbatim or in full") can come back as `stop_reason: "refusal"` with category `reasoning_extraction`.
  Asking for "a short explanation" or "the evidence behind a result" is the documented safe wording `[doc:refusals]`
  `[doc:thinking]`. The judge code must check `stop_reason` like the contestant loop does.

Limits `[doc:vision]` `[doc:errors]`:

| Limit | Value |
| --- | --- |
| Images per request | 600 on models with a 1M window; 100 on models with a 200k window |
| Dimensions per image | 8000 x 8000 px maximum. If a request holds **more than 20 images**, a stricter per-image limit applies to all of them (keep each side at or under 3,000 px, or stay at 20 images or fewer). 9 images is under that threshold. |
| Size per image | 10 MB, measured base64-encoded, on the Claude API |
| Request size | 128 MB for the Messages API; over that is a 413 `request_too_large` |
| Formats | JPEG, PNG, GIF, WebP; only the first frame of an animation |

Token cost `[doc:vision]` `[doc:vision-coordinates]`:

- An image costs `ceil(width / 28) * ceil(height / 28)` visual tokens, billed as input tokens at the model's input
  price.
- Before counting, an image larger than the model's limits is downscaled, aspect ratio preserved, to the largest size
  that satisfies both: long edge <= 2,576 px and <= 4,784 visual tokens ("high-resolution tier: Claude 4.7 and later
  models"); 1,568 px and 1,568 tokens on the standard tier (all other models).
- Examples from the docs: 200x200 -> 64 tokens; 1000x1000 -> 1,296; 1920x1080 -> 2,691 (high-res tier);
  3840x2160 -> downscaled to 2576x1449, 4,784 tokens.
- Worst case for 9 images on the high-res tier: 9 x 4,784 = 43,056 input tokens. To control cost, resize before
  sending. A reference implementation of the resize rule is in `[doc:vision-coordinates]`.
- Limitations listed by the docs: no identification of real people in images; possible errors on low-quality, rotated
  or very small (under 200 px) images; counting and spatial reasoning are approximate `[doc:vision]`.

---

## 8. Current models and list prices

Model ids from `[doc:models]` (https://platform.claude.com/docs/en/about-claude/models/overview) and prices from
`[doc:pricing]` (https://platform.claude.com/docs/en/about-claude/pricing), both fetched 2026-11-19. USD per million
tokens (MTok). On the Claude API the id and the alias are the same string; each is a pinned snapshot.

| Tier | Model | API id | Input | Output | Cache read | 5m cache write | Released |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Flagship | Claude Fable 5.2 | `claude-fable-5-2` | $8 | $40 | $0.20 | $10 | 2026-11-09 |
| High (the docs' default pick: "start with Claude Opus 5.5 for most workloads") | Claude Opus 5.5 | `claude-opus-5-5` | $4 | $20 | $0.20 | $5 | 2026-09-22 |
| Mid | Claude Sonnet 5.5 | `claude-sonnet-5-5` | $2 | $10 | $0.10 | $2.50 | 2026-09-28 |
| Small / cheap | Claude Haiku 5.5 | `claude-haiku-5-5` | $0.10 (prompt <= 100k tokens); $0.50 (prompt > 100k) | $0.50; $2.50 | $0.01; $0.05 | $0.125; $0.625 | 2026-10-07 |

- Claude Haiku 5.5 is priced per request by prompt length: when a request's total input (cache reads and writes
  included) is over 100,000 tokens, that request pays the higher prices for input and output `[doc:pricing]`.
- Thinking tokens are billed as output tokens `[doc:thinking]`. Message Batches are 50% off `[doc:pricing]`.
- Claude Mythos 5.2 (`claude-mythos-5-2`, same price as Fable 5.2) is limited to verified organizations `[doc:models]`.
- Previous generation still listed: Claude Fable 5.1 and Fable 5 ($10 / $50), Claude Opus 5, 4.8, 4.7, 4.6 and 4.5
  ($5 / $25), Claude Sonnet 5 ($2 / $10), Sonnet 4.6 and 4.5 ($3 / $15), Claude Haiku 4.5 ($1 / $5) `[doc:models]`.
- Claude 4.7 and later use a tokenizer that produces about 30% more tokens for the same text than earlier models
  `[doc:pricing]`.
- Claude Fable 5.2 has its own rate-limit bucket `[doc:fable-migration]`, requires 30-day data retention (a zero-data-
  retention organization gets a 400) `[doc:fable]` `[doc:fable-migration]`, and does not support Priority Tier.
- The SDK's `Model` literal in 0.121.0 does not list the 5.2 / 5.5 ids, but the type is `Union[Literal[...], str]`, so
  any id string is accepted `[sdk:types/model_param.py]`.

---

## 9. Errors and retries

### HTTP statuses and SDK exception classes

`[doc:errors]` `[sdk:_exceptions.py]` `[sdk:_client.py::_make_status_error]` `[probe]`

| Status | API `error.type` | SDK class (`anthropic.`) | Retry? |
| --- | --- | --- | --- |
| 400 | `invalid_request_error` | `BadRequestError` | No. Includes prefix-check failures, rejected parameters, and a spend limit you set yourself. |
| 401 | `authentication_error` | `AuthenticationError` | No (the SDK already retried once with a fresh token when a credentials provider is used and `max_retries >= 1`). |
| 402 | `billing_error` | `APIStatusError` | No. |
| 403 | `permission_error` | `PermissionDeniedError` | No. |
| 404 | `not_found_error` | `NotFoundError` | No (unknown model id, wrong path). |
| 408 | | `APIStatusError` | SDK retries. |
| 409 | `conflict_error` | `ConflictError` | SDK retries. |
| 413 | `request_too_large` | `RequestTooLargeError` | No. |
| 422 | | `UnprocessableEntityError` | No. |
| 429 | `rate_limit_error` | `RateLimitError` | Yes, after `retry-after`. A spend-cap 429 has no `retry-after` and keeps failing. |
| 500 | `api_error` | `InternalServerError` | Yes. |
| 503, 504 | `timeout_error` (504) | `InternalServerError` (any other status >= 500) | Yes. |
| 529 | `overloaded_error` | `OverloadedError` | Yes. |
| no response | | `APIConnectionError`; `APITimeoutError` is a subclass | Yes. |

Things that differ from what one would guess:

- `OverloadedError` (529) is **not** a subclass of `InternalServerError`. Catch `anthropic.APIStatusError` and branch on
  `e.status_code`, or catch both `[probe]`.
- Every `APIStatusError` has `status_code`, `response` (httpx), `request_id`, `body` (parsed JSON) and `type` (the
  `error.type` string) `[sdk:_exceptions.py]`.
- Error body: `{"type": "error", "error": {"type": "...", "message": "..."}, "request_id": "req_..."}` `[doc:errors]`.
- Some Messages API errors add a finer code at `error.details.error_code` (in the SDK: `e.body["error"]["details"]
  ["error_code"]`, absent on most errors). Two matter here, both HTTP 429 with `error.type == "rate_limit_error"`:
  `enforced_spend_limit_reached` (monthly spend cap: no `retry-after`, "retrying, including the SDK's automatic
  retries, fails until access resumes") and `ramp_limit_exceeded` (retry after `retry-after`)
  `[doc:api-ref]` `[doc:rate-limits]`. A spend-cap 429 is better raised as `FatalError`.

### Errors after the stream has started

The SDK's retries only cover the time before the HTTP response starts. Once events are flowing `[probe]`:

- An SSE `error` event (`event: error` / `data: {"type": "error", "error": {"type": "overloaded_error", "message":
  "Overloaded"}}` `[doc:streaming]`) is raised as a plain **`anthropic.APIStatusError` with `status_code == 200`**; the
  real kind is in `e.type` (`"overloaded_error"`, `"api_error"`, `"rate_limit_error"`, `"timeout_error"`, ...) and
  `e.body` `[sdk:_streaming.py]`. Classify mid-stream errors by `e.type`, not by status code. The reference says the
  same: once the stream has started "a later error comes through the stream, with no status of its own. To identify
  it, use `error.details.error_code` if the error has one, and `error.type` if not" `[doc:api-ref]`.
- A transport failure while reading the body is raised as a raw **`httpx` exception** (`httpx.ReadTimeout`,
  `httpx.ReadError`, `httpx.RemoteProtocolError`; all subclasses of `httpx.HTTPError`), not wrapped in
  `anthropic.APIConnectionError` `[sdk:_base_client.py::_attempt_request]`.
- A stream that closes cleanly after `message_start` but without a `message_delta` raises nothing; the final message
  has `stop_reason` `None`.
- A 200 response with no events at all makes `get_final_message()` raise a bare `AssertionError`; an event without
  `usage` raises `AttributeError` (section 1). Neither is an `anthropic` or `httpx` exception.
- In all of these cases nothing was retried, and partial output cannot be resumed for thinking or tool blocks
  `[doc:streaming]`. Discard it and re-send the same request.

Suggested mapping to paintbench's errors:

| Raise `TransientError` | Raise `FatalError` |
| --- | --- |
| `APIConnectionError` (incl. `APITimeoutError`); `httpx.HTTPError` during the stream; `APIStatusError` with status 408, 409, 429, >= 500, or with `type` in `overloaded_error`, `api_error`, `timeout_error`, `rate_limit_error`; final message with `stop_reason is None`; `AssertionError` / `AttributeError` raised while consuming the stream (empty or malformed stream); `WorkloadIdentityError` with `status_code` `None`, 429 or >= 500 | `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, `RequestTooLargeError`, `UnprocessableEntityError`, status 402; a 429 whose `error.details.error_code` is `enforced_spend_limit_reached`; `WorkloadIdentityError` with status 400 / 401 / 403; the SDK's `ValueError` / `TypeError` raised before any request is sent |

`retry_after`: `float(e.response.headers["retry-after"])` when present.

### The SDK's built-in retries

`[sdk:_base_client.py]` `[sdk:_constants.py]` `[doc:py-sdk]` `[probe]`

- Default `max_retries = 2` (three attempts). Retried: connection errors and timeouts, 408, 409, 429, every status
  >= 500 (so 529 too). A response header `x-should-retry: true|false` overrides.
- Delay: `retry-after-ms` or `retry-after` if it is between 0 and 60 s; otherwise `min(0.5 * 2**n, 8)` seconds times a
  jitter factor of 0.75 to 1.0.
- Configure per client (`Anthropic(max_retries=N)`) or per call (`client.with_options(max_retries=N, timeout=...)`);
  `0` disables.
- paintbench owns backoff, so keep it low, but use **`max_retries=1`, not 0**, with WIF: the 401 token-refresh retry
  only runs when a retry is left. Offline check: with `max_retries=0` a 401 surfaces after one attempt and the token
  provider is not called again; with 1 or 2 the token is re-minted and the request is sent exactly once more `[probe]`.
- A retried request is a new generation from scratch.

### Rate limits

`[doc:rate-limits]` `[doc:context-windows]`

- Limits are per model class: requests per minute, input tokens per minute, output tokens per minute. Thinking tokens
  count as output. Exceeding one gives a 429 naming the limit; wait `retry-after` seconds when the header is present.
- Response headers (on `stream.response.headers`): `retry-after`; `anthropic-ratelimit-requests-limit|remaining|reset`;
  `anthropic-ratelimit-tokens-limit|remaining|reset`; `anthropic-ratelimit-input-tokens-limit|remaining|reset`;
  `anthropic-ratelimit-output-tokens-limit|remaining|reset` (reset values are RFC 3339 times; remaining token counts are
  rounded to the nearest thousand). Also `request-id`, `anthropic-organization-id`, `anthropic-workspace-id`
  `[doc:api-overview]`.
- A sharp rise in traffic can cause 429s below the limits (ramp limits) `[doc:errors]`.

---

## 10. Worked example (mock fixtures)

Every JSON object below was validated offline against the installed SDK: the response objects with
`anthropic.types.Message.model_validate(...)`, the event lists by serving them as an SSE body through
`httpx.MockTransport` and calling `client.messages.stream(...).get_final_message()`, and the two request bodies are what
the SDK actually serialised `[probe]`. Signatures and ids are made-up placeholders; real signatures are much longer.

### Request 1 (system + user + tool)

The body shown at the top of section 2, plus `"stream": true`.

### Response 1 (reasoning + one tool call)

```json
{
  "id": "msg_01A1B2C3D4E5F6G7H8J9K0LM",
  "type": "message",
  "role": "assistant",
  "model": "claude-fable-5-2",
  "content": [
    {
      "type": "thinking",
      "thinking": "Plan: dark blue sky, yellow stars, a cypress on the left. Start with the palette and the top two rows.",
      "signature": "EuYBCkYICxgCKkDx1nQ0...opaque-signature-1...=="
    },
    {
      "type": "tool_use",
      "id": "toolu_01T1x1fJ34qAmk2tNTrN7Up6",
      "name": "write_rows",
      "input": {"palette": ["a=#1b2a6b", "b=#f4d03f", "c=#0b3d2e"], "start_row": 0, "rows": ["aabaaaba", "caaabaaa"]},
      "caller": {"type": "direct"}
    }
  ],
  "stop_reason": "tool_use",
  "stop_sequence": null,
  "stop_details": null,
  "usage": {
    "input_tokens": 1042,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "output_tokens": 2610,
    "output_tokens_details": {"thinking_tokens": 2480},
    "service_tier": "standard"
  }
}
```

With the default `display` (`omitted`) the thinking block would be `{"type": "thinking", "thinking": "", "signature":
"..."}`. Also test a variant without the `caller` key and one with a `redacted_thinking` block
(`{"type": "redacted_thinking", "data": "..."}`).

A real non-streaming response body carries more top-level and `usage` keys than this fixture. The reference's own
example has `"container"`, `"diagnostics": null`, `"citations": null` on text blocks, and in `usage`: `cache_creation`,
`inference_geo`, `server_tool_use`, `service_tier`, `speed` `[doc:api-ref]`. That example validates against 0.121.0's
`Message` (`diagnostics` and `usage.speed` land in `model_extra`) `[probe]`. `parse_response()` must ignore keys it does
not know, at every level.

### Response 1 as a stream (the events the SDK assembles)

Wire format: each event is `event: <name>\ndata: <one-line JSON>\n\n`, response header `content-type: text/event-stream`.
Order: `message_start`, then per content block `content_block_start` / zero or more `content_block_delta` /
`content_block_stop`, then `message_delta`, then `message_stop`; `ping` events may appear anywhere `[doc:streaming]`.

```json
[
  {"event": "message_start", "data": {"type": "message_start", "message": {"id": "msg_01A1B2C3D4E5F6G7H8J9K0LM", "type": "message", "role": "assistant", "model": "claude-fable-5-2", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 1042, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 1}}}},
  {"event": "content_block_start", "data": {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}},
  {"event": "ping", "data": {"type": "ping"}},
  {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Plan: dark blue sky, yellow stars, a cypress on the left."}}},
  {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": " Start with the palette and the top two rows."}}},
  {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "EuYBCkYICxgCKkDx1nQ0...opaque-signature-1...=="}}},
  {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 0}},
  {"event": "content_block_start", "data": {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_01T1x1fJ34qAmk2tNTrN7Up6", "name": "write_rows", "input": {}, "caller": {"type": "direct"}}}},
  {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ""}}},
  {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"palette\": [\"a=#1b2a6b\", \"b=#f4d03f\", \"c=#0b3d2e\"]"}}},
  {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ", \"start_row\": 0, \"rows\": [\"aabaaaba\", \"caaabaaa\"]}"}}},
  {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 1}},
  {"event": "message_delta", "data": {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null}, "usage": {"output_tokens": 2610, "output_tokens_details": {"thinking_tokens": 2480}}}},
  {"event": "message_stop", "data": {"type": "message_stop"}}
]
```

Details that matter when building a fake stream `[doc:streaming]` `[doc:thinking]` `[probe]`:

- **`usage` is mandatory** in `message_start.message` and in `message_delta`. Several doc traces abbreviate it away;
  feeding such a trace to SDK 0.121.0 crashes with `AttributeError: 'NoneType' object has no attribute 'output_tokens'`.
- A thinking block starts as `{"type": "thinking", "thinking": "", "signature": ""}`, gets `thinking_delta` events, and
  exactly one `signature_delta` just before its `content_block_stop`. With `display: "omitted"` there is a single
  `thinking_delta` whose `thinking` is `""`, then the `signature_delta`.
- A `tool_use` block starts with `"input": {}`; the arguments arrive as `input_json_delta.partial_json` string
  fragments that concatenate to one JSON document. An empty first fragment is normal. A call with no arguments has no
  non-empty fragment and keeps `input: {}`.
- `redacted_thinking` blocks arrive whole in `content_block_start` (`{"type": "redacted_thinking", "data": "..."}`) with
  no deltas.
- `stop_reason` is `null` in `message_start` and is set by `message_delta`; `message_delta.usage` is cumulative.
- Iterating the SDK stream yields the raw events plus SDK-made ones (`text`, `thinking`, `signature`, `input_json`);
  ignore types you do not handle. Unknown SSE event names are skipped by the SDK.
- Helper that returns the assembled message: `stream.get_final_message()` (`MessageStream`, sync) /
  `await stream.get_final_message()` (`AsyncMessageStream`).

A test client that exercises the real SDK with no network (works on 0.121.0 `[probe]`):

```python
import json, httpx
from anthropic import Anthropic

def sse(events):   # events: list of (name, data)
    return "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in events).encode()

def fake_client(events, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))                     # the exact body the SDK sent
        return httpx.Response(200, headers={"content-type": "text/event-stream", "request-id": "req_test"},
                              content=sse(events))
    return Anthropic(api_key="test", base_url="https://api.anthropic.com", max_retries=0,
                     http_client=httpx.Client(transport=httpx.MockTransport(handler)))
```

### Request 2 (replayed assistant turn + tool result)

Top-level fields are identical to request 1. `messages` as serialised by the SDK:

```json
[
  {"role": "user", "content": "Painting: The Starry Night (Vincent van Gogh, 1889). Canvas: 8 columns x 4 rows."},
  {
    "role": "assistant",
    "content": [
      {
        "type": "thinking",
        "thinking": "Plan: dark blue sky, yellow stars, a cypress on the left. Start with the palette and the top two rows.",
        "signature": "EuYBCkYICxgCKkDx1nQ0...opaque-signature-1...=="
      },
      {
        "type": "tool_use",
        "id": "toolu_01T1x1fJ34qAmk2tNTrN7Up6",
        "name": "write_rows",
        "input": {"palette": ["a=#1b2a6b", "b=#f4d03f", "c=#0b3d2e"], "start_row": 0, "rows": ["aabaaaba", "caaabaaa"]},
        "caller": {"type": "direct"}
      }
    ]
  },
  {
    "role": "user",
    "content": [
      {"type": "tool_result", "tool_use_id": "toolu_01T1x1fJ34qAmk2tNTrN7Up6", "content": "OK: rows 0-1 written (2 of 4)."}
    ]
  }
]
```

### Response 2 (final text)

```json
{
  "id": "msg_01N2P3Q4R5S6T7U8V9W0XYZA",
  "type": "message",
  "role": "assistant",
  "model": "claude-fable-5-2",
  "content": [
    {"type": "thinking", "thinking": "", "signature": "EqQBCgIYAhIM1gbcDa9G...opaque-signature-2...=="},
    {"type": "text", "text": "The painting is complete: all 4 rows are written."}
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "stop_details": null,
  "usage": {
    "input_tokens": 3720,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "output_tokens": 215,
    "output_tokens_details": {"thinking_tokens": 180},
    "service_tier": "standard"
  }
}
```

As a stream:

```json
[
  {"event": "message_start", "data": {"type": "message_start", "message": {"id": "msg_01N2P3Q4R5S6T7U8V9W0XYZA", "type": "message", "role": "assistant", "model": "claude-fable-5-2", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 3720, "output_tokens": 1}}}},
  {"event": "content_block_start", "data": {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}},
  {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": ""}}},
  {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "EqQBCgIYAhIM1gbcDa9G...opaque-signature-2...=="}}},
  {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 0}},
  {"event": "content_block_start", "data": {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}},
  {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "The painting is complete: "}}},
  {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "all 4 rows are written."}}},
  {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 1}},
  {"event": "message_delta", "data": {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 215, "output_tokens_details": {"thinking_tokens": 180}}}},
  {"event": "message_stop", "data": {"type": "message_stop"}}
]
```

### Refusal

The response body is the documented example `[doc:refusals]`:

```json
{
  "id": "msg_01XFUDYJgAACzvnptvVoYEL",
  "type": "message",
  "role": "assistant",
  "model": "claude-fable-5-2",
  "content": [],
  "stop_reason": "refusal",
  "stop_sequence": null,
  "stop_details": {
    "type": "refusal",
    "category": "cyber",
    "explanation": "This request was declined because it could enable cyber harm."
  },
  "usage": {"input_tokens": 412, "output_tokens": 0}
}
```

As a stream: `message_start` (empty content, `usage`), then
`{"type": "message_delta", "delta": {"stop_reason": "refusal", "stop_sequence": null, "stop_details": {"type":
"refusal", "category": "cyber", "explanation": "..."}}, "usage": {"output_tokens": 0}}`, then `message_stop`. Also test
`"category": null, "explanation": null`, and a mid-stream refusal that follows a partial text block (the partial text
must be discarded).

### `max_tokens` truncation

```json
{
  "id": "msg_01MAXTOKENS0000000000000",
  "type": "message",
  "role": "assistant",
  "model": "claude-fable-5-2",
  "content": [
    {"type": "thinking", "thinking": "", "signature": "EuYBCkYICxgCKkDx1nQ0...opaque-signature-1...=="},
    {"type": "text", "text": "Row 0 uses the dark blue of the sky and the first sta"}
  ],
  "stop_reason": "max_tokens",
  "stop_sequence": null,
  "stop_details": null,
  "usage": {"input_tokens": 1042, "output_tokens": 128000, "output_tokens_details": {"thinking_tokens": 127950}}
}
```

Cut off while the model was still thinking: one `thinking` block and nothing else. Every incomplete attempt of run 1
was four replies of this shape `[live]`. The signature is shortened here; in the 291 live replies it was 250,000 to
397,000 characters long, beside 34,000 to 109,000 characters of summary text:

```json
{
  "id": "msg_01CUTWHILETHINKING000000",
  "type": "message",
  "role": "assistant",
  "model": "claude-fable-5-2",
  "content": [
    {
      "type": "thinking",
      "thinking": "Sixteen columns. Row 1 is sky. The crest starts at column 3 of row 2, so row 3 must",
      "signature": "EvkJCkYICxgCKkBq7Zc2...opaque-signature-of-a-block-cut-off-by-the-output-limit...=="
    }
  ],
  "stop_reason": "max_tokens",
  "stop_sequence": null,
  "stop_details": null,
  "usage": {"input_tokens": 1079, "output_tokens": 128000, "output_tokens_details": {"thinking_tokens": 128000}}
}
```

The next request appends that content as one assistant message, unchanged, and then the nudge (section 4):

```json
[
  {"role": "assistant", "content": [
    {
      "type": "thinking",
      "thinking": "Sixteen columns. Row 1 is sky. The crest starts at column 3 of row 2, so row 3 must",
      "signature": "EvkJCkYICxgCKkBq7Zc2...opaque-signature-of-a-block-cut-off-by-the-output-limit...=="
    }
  ]},
  {"role": "user", "content": "The picture is not finished. Missing rows: 1-11. Send them with write_rows."}
]
```

Stream variant for a tool call cut mid-argument: the response-1 stream up to the empty `input_json_delta`, then

```json
[
  {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"start_row\": 0, \"rows\": [\"aabaaaba\", \"caaab"}}},
  {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 1}},
  {"event": "message_delta", "data": {"type": "message_delta", "delta": {"stop_reason": "max_tokens", "stop_sequence": null}, "usage": {"output_tokens": 128000}}},
  {"event": "message_stop", "data": {"type": "message_stop"}}
]
```

With that input the SDK's final message holds `"input": {"start_row": 0, "rows": ["aabaaaba"]}` and raises nothing, while
the raw string collected from the deltas does not parse `[probe]`.

### Two tool calls in one reply

```json
{
  "id": "msg_01TWOTOOLS00000000000000",
  "type": "message",
  "role": "assistant",
  "model": "claude-fable-5-2",
  "content": [
    {"type": "thinking", "thinking": "", "signature": "EuYBCkYICxgCKkDx1nQ0...opaque-signature-1...=="},
    {
      "type": "tool_use",
      "id": "toolu_01AAAAAAAAAAAAAAAAAAAAAA",
      "name": "write_rows",
      "input": {"palette": ["a=#1b2a6b", "b=#f4d03f"], "start_row": 0, "rows": ["aabaaaba", "aaaabaaa"]},
      "caller": {"type": "direct"}
    },
    {
      "type": "tool_use",
      "id": "toolu_01BBBBBBBBBBBBBBBBBBBBBB",
      "name": "write_rows",
      "input": {"start_row": 2, "rows": ["abaaaaab", "aaaaaaaa"]},
      "caller": {"type": "direct"}
    }
  ],
  "stop_reason": "tool_use",
  "stop_sequence": null,
  "stop_details": null,
  "usage": {"input_tokens": 1042, "output_tokens": 3100, "output_tokens_details": {"thinking_tokens": 2900}}
}
```

The next request appends that content as one assistant message and then exactly one user message:

```json
{
  "role": "user",
  "content": [
    {"type": "tool_result", "tool_use_id": "toolu_01AAAAAAAAAAAAAAAAAAAAAA", "content": "OK: rows 0-1 written (2 of 4)."},
    {"type": "tool_result", "tool_use_id": "toolu_01BBBBBBBBBBBBBBBBBBBBBB", "content": "OK: rows 2-3 written (4 of 4)."}
  ]
}
```

On Fable 5.2 / Opus 5.5 / Sonnet 5.5 a second `thinking` block (a progress update) can sit directly before each
`tool_use` block; a fixture with `thinking, thinking, tool_use, thinking, tool_use` is realistic `[doc:thinking]`.

---

## 11. Gotchas and surprises

Ordered by how likely each is to break an untested adapter.

1. **Tool results are blocks in one user message, not messages of their own.** One `tool_result` per `tool_use`, all in
   a single user message placed directly after the assistant message, results before any text. The neutral harness has
   one tool message per call; the adapter must merge them.
2. **Replay the assistant content exactly, empty thinking blocks included.** With the default `display` every thinking
   block has `thinking: ""` and looks droppable; dropping or editing one is a 400 inside a tool loop. `redacted_thinking`
   is a separate type that must survive too.
3. **Append-only or 400.** `system`, `tools` and all earlier messages must be identical on every request of a
   conversation (prefix check, hard 400 on Fable 5.2). No dates in the system prompt, no rewording of tool descriptions,
   no history trimming, no moving a conversation to another organization.
4. **`messages.create()` with a large `max_tokens` raises `ValueError` before any request is sent.** Use
   `messages.stream()` everywhere.
5. **The stream helper loses `usage.output_tokens_details`** on 0.121.0; read thinking tokens from the `message_delta`
   event.
6. **A reply cut by `max_tokens` can contain a shortened but valid-looking tool call.** Check `stop_reason` and the raw
   argument string before applying it. **More often, at `max` effort, it contains one signed thinking block and
   nothing else. Pass that turn back, then send the nudge.** Skipping it looks safe, because the request is valid
   either way, and it throws away up to 128,000 tokens of reasoning on every call. Of the 300 replies of run 1 that
   were cut by `max_tokens`, 291 had this shape and 9 held a cut tool call; 55 attempts ended with nothing typed
   after four such replies each, none of which the model saw again `[live]` (section 3).
7. **Mid-stream failures do not look like HTTP failures**: `APIStatusError` with `status_code == 200` (kind in `.type`),
   raw `httpx` exceptions, a bare `AssertionError` (stream with no events), or no exception at all with `stop_reason`
   `None`. None is retried by the SDK. And `message._request_id` raises `AttributeError` on a stream-assembled message.
8. **529 is `OverloadedError`, which is not an `InternalServerError`.**
9. **Thinking is on by default and cannot be switched off** on Fable 5.2 and Opus 5.5; `budget_tokens`, `disabled` and
   forced `tool_choice` are 400s there. The first block of a reply is usually `thinking`, so `content[0].text` fails.
10. **A tool-call reply often has no text.** Narration between tool calls arrives as `thinking` blocks on Fable 5.2,
    Opus 5.5 and Sonnet 5.5.
11. **Empty replies happen**: `end_turn` with `content: []`. Do not replay an empty assistant message; send the nudge.
    Empty means that no block can go back. A reply that holds a signed `thinking` block and nothing else is not
    empty and is replayed (gotcha 6); only a `thinking` block that never got its signature counts for nothing.
    Claude Fable 5.2 also "sometimes ends a turn with a list of what remains, an offer to continue" on long tasks
    `[doc:fable-whats-new]`, which is what the nudge is for.
12. **No text after tool results.** Extra text in the tool-result message teaches the model to stop after each call.
13. **Text blocks need at least one character**; an empty label or empty nudge is rejected.
14. **Never send `temperature` / `top_p` / `top_k`.**
15. **`ANTHROPIC_BASE_URL` silently redirects both API calls and the WIF token exchange**; always pass `base_url`.
    `ANTHROPIC_API_KEY`, even empty, beats federation when no explicit credential argument is passed.
16. **`max_retries=0` disables the WIF 401 refresh-and-retry.**
17. **`thinking_tokens` is part of `output_tokens`**, and `input_tokens` excludes cached tokens; add the three input
    counters for a total.
18. **Docs vs installed SDK**: docs show `anthropic-version: 2026-10-21`, `httpx2`, `Authorization: Bearer` for API keys
    and the `client.beta.messages` namespace for newer features. 0.121.0 has no types for `between_tools`,
    `block_binding`, `input_transformations` or `display: "updates"`; unknown request keys still pass through, and
    `extra_body` replaces a whole top-level key (it is a shallow merge) `[probe]`.
    **These notes and the test helper in section 10 hold for 0.121.0 only.** A requirement of `anthropic>=0.121`
    with no upper bound resolves to 2.x on a fresh install (2.11.0 on 2026-11-20 `[doc:sdk-versions]`). On 2.x the
    `http_client` "must be an `httpx2` client. Passing a client from the separate `httpx` package raises a `TypeError`"
    `[doc:py-sdk]` (so an `httpx.Client(transport=httpx.MockTransport(...))` fixture breaks), requests go out under API
    version 2026-10-21, and `MessageParam` became a union with one type per role `[doc:migration]`. The same page
    points 0.x users to a separate "v1 migration guide", so 1.x has breaking changes of its own. `pyproject.toml`
    therefore pins `anthropic>=0.121,<1`, until the adapter is re-checked against a newer major version.
19. **Block types unknown to 0.121.0** (for example the `fallback` block of the server-side fallback beta) are assembled
    by the stream helper as a mis-typed text block with extra null fields `[probe]`. We use no beta, so this should not
    arise; if it ever does, that payload is not a faithful copy.
20. **Refusals are HTTP 200** and invisible to error-rate monitoring; a `reasoning_extraction` refusal means a prompt or
    tool field asks the model to write out its reasoning.
21. **Signatures are long opaque strings**; store them in full and never truncate transcripts that will be replayed.
22. **Prompt caching is opt-in** (top-level `"cache_control": {"type": "ephemeral"}` caches the growing prefix
    automatically; 5-minute TTL by default, `"ttl": "1h"` available) `[doc:prompt-caching]`. It changes no content and
    does not affect the prefix check, but it changes the cost arithmetic (cache writes 1.25x, reads 0.025x to 0.1x of
    the input price `[doc:pricing]`). Off by default keeps accounting simple.
23. **Claude Haiku 5.5 doubles as a pricing trap**: a prompt over 100k tokens (easy to reach with 9 large images plus
    history) moves the whole request to the 5x price tier.

---

## UNVERIFIED (from memory)

The open point in each item is not backed by a page fetched for these notes or by the SDK source (where an item quotes
a documented fact for context, it says so). Treat each item as an open question, except where it says that a live
run has since settled it.

1. **Ping cadence.** I believe the API sends `ping` events every few seconds to tens of seconds during long silent
   phases, which would keep a stream alive through a 30-minute thinking phase. The docs I read only say pings "might"
   appear. Hence the recommendation to raise the read timeout and request `display: "summarized"`.
2. **WIF token lifetime versus stream length.** The token lifetime itself is documented (section 1: default 3,600 s,
   capped at twice the remaining life of the identity token). I found nothing on whether a stream that is already open
   is cut when its bearer token expires. If it is, calls longer than the token lifetime would fail mid-stream every
   time and be retried in a loop. Watch for this on a long call.
3. **`"caller": null` and `"citations": null` in a replayed block.** Whether the API accepts a null `caller` is
   undocumented (the reference lists `caller` on an input `tool_use` block as optional, without "or null"); the notes
   avoid the question by dropping null-valued keys. The opposite risk is equally unverified: that dropping a null key
   could count as a change for the prefix check. The docs say the check "compares parsed JSON, not its text" and that
   thinking blocks "aren't compared as content", but say nothing about null versus absent keys. Whether the live
   stream's `content_block_start` for `tool_use` includes `caller` at all is also unconfirmed (the reference lists it
   with default `{"type": "direct"}`; the docs' SSE traces omit it).
4. **Assistant messages with empty content.** From memory the API rejects an assistant turn whose content is an
   empty array. The notes recommend not sending such a message (section 4: the documented remedy for an empty reply
   is a new user message, and that path is the one in use); the claim that it would be rejected is not verified,
   and nothing sent so far has tested it.
   This item used to cover a second claim, that a turn "whose final block is a thinking block" is rejected too.
   That half is no longer here: for a turn of one signed thinking block followed by a user message it was tested on
   20 November 2026 and is false (section 3, `[live]`). The shapes that the test did not cover are listed there.
5. **Whitespace-only text blocks.** From memory these are rejected like empty ones ("text content blocks must contain
   non-whitespace text"). Only the `minLength: 1` rule is documented.
6. **Truncated tool calls.** What the API returns as `input` for a `tool_use` block cut by `max_tokens` (the exact
   shape of the argument string on the stream: cut inside a value, cut between values, or closed by the server), and
   whether a turn containing such a block can be replayed, is not documented. A thinking block cut before its
   signature arrived cannot be replayed (documented, section 3). If the replay is rejected (400), the harness's
   discard-and-restart path applies. Since then run 1 replayed nine such turns and none was rejected (section 3,
   `[live]`); the documentation is still silent, and the shape of the cut argument string is still open.
7. **`max_tokens` above the model ceiling** returns a 400 (from memory); the docs only state the ceilings.
8. **API version 2023-06-01 with the newest models.** The versioning page lists 2023-06-01 in its version history, the
   SDK page says an SDK keeps "the API version that was current when your SDK's major version was released", and the
   thinking page gives Claude Fable 5.2 a sampling-parameter rule (non-default values are a 400) that differs from the
   2026-10-21 rule (any value is a 400) only under an older API version, which implies the model is served there. No
   page states it outright and nothing was sent, so `claude-fable-5-2` through SDK 0.121.0 is untested here.
9. **First message role.** Older material says the first message must be `user`; the reference text I read only
   describes alternating turns. The adapter always starts with a user message anyway.
10. **Usage fields on live stream events.** Whether `message_start.usage` and `message_delta.usage` carry the cache
    counters and `input_tokens` on every stream is unconfirmed; the schema marks them optional and the doc traces show
    only `input_tokens` / `output_tokens`. Read them defensively.
11. **Empty-string `tool_result.content`.** Undocumented; the docs express an empty result by omitting `content`. Send
    non-empty result text.
12. **Claude Haiku 5.5 image tier.** The docs put "Claude 4.7 and later models" on the high-resolution image tier
    without a per-model list; I assume Haiku 5.5 is included.
13. **Order of `tool_result` blocks.** The docs say results are matched by `tool_use_id` and do not require the order of
    the calls; I recommend the same order anyway.
14. **A response text block with empty text.** The stream helper can hand back `{"type": "text", "text": ""}` (a block
    that opened and received no delta) `[probe]`, and the request schema forbids sending one (`minLength: 1`,
    documented). Whether the live API ever returns such a block in a completed reply is unknown. If it does, replaying
    it verbatim is a documented 400, and dropping it is an edit whose effect on the prefix check is unknown. Suggested
    handling, untested: drop empty text blocks from the replay payload and let a resulting 400 take the
    discard-and-restart path.
15. **A hard wall-clock cap.** Closing the stream from another thread (a timer calling `stream.close()`) while the
    event loop is blocked in a read was not tested; I do not know which exception the blocked read raises.

---

## Sources

### Official documentation (fetched 2026-11-19 and 2026-11-20 with a generic User-Agent, as markdown via the `.md` suffix)

| Tag | URL |
| --- | --- |
| `[doc:models]` | https://platform.claude.com/docs/en/models/overview (the older `/about-claude/models/overview` URL redirects here) |
| `[doc:pricing]` | https://platform.claude.com/docs/en/about-claude/pricing |
| `[doc:fable]` | https://platform.claude.com/docs/en/models/fable-5-2/overview |
| `[doc:fable-migration]` | https://platform.claude.com/docs/en/models/fable-5-2/migration-guide |
| `[doc:fable-whats-new]` | https://platform.claude.com/docs/en/models/fable-5-2/whats-new-fable-5-2 |
| (model pages) | https://platform.claude.com/docs/en/models/opus-5-5/overview , https://platform.claude.com/docs/en/models/sonnet-5-5/overview , https://platform.claude.com/docs/en/models/haiku-5-5/overview |
| `[doc:thinking]` | https://platform.claude.com/docs/en/build-with-claude/thinking |
| `[doc:thinking-steering]` | https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost (the old `/adaptive-thinking` URL serves this page) |
| `[doc:thinking-tool-workflows]` | https://platform.claude.com/docs/en/build-with-claude/thinking-tool-workflows |
| `[doc:thinking-troubleshooting]` | https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting |
| `[doc:effort]` | https://platform.claude.com/docs/en/build-with-claude/effort |
| `[doc:preserved-thinking]` | https://platform.claude.com/docs/en/build-with-claude/preserved-thinking |
| `[doc:prefix-checks]` | https://platform.claude.com/docs/en/build-with-claude/preserved-thinking/prefix-checks |
| `[doc:change-tools]` | https://platform.claude.com/docs/en/build-with-claude/preserved-thinking/change-tools-or-instructions |
| `[doc:fix-bound]` | https://platform.claude.com/docs/en/build-with-claude/preserved-thinking/fix-bound-to-a-different-conversation |
| `[doc:stop-reasons]` | https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons |
| `[doc:refusals]` | https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback |
| `[doc:streaming]` | https://platform.claude.com/docs/en/build-with-claude/streaming |
| `[doc:vision]` | https://platform.claude.com/docs/en/build-with-claude/vision |
| `[doc:vision-coordinates]` | https://platform.claude.com/docs/en/build-with-claude/vision-coordinates |
| `[doc:context-windows]` | https://platform.claude.com/docs/en/build-with-claude/context-windows |
| `[doc:structured-outputs]` | https://platform.claude.com/docs/en/build-with-claude/structured-outputs (limitations and compatibility sections) |
| `[doc:prompt-caching]` | https://platform.claude.com/docs/en/build-with-claude/prompt-caching (overview and automatic-caching sections only) |
| `[doc:batches]` | https://platform.claude.com/docs/en/build-with-claude/batch-processing (timing statements only) |
| `[doc:define-tools]` | https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools |
| `[doc:handle-tool-calls]` | https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls |
| `[doc:parallel]` | https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use |
| `[doc:strict]` | https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use |
| `[doc:api-ref]` | https://platform.claude.com/docs/en/api/messages/create |
| `[doc:errors]` | https://platform.claude.com/docs/en/api/errors |
| `[doc:rate-limits]` | https://platform.claude.com/docs/en/api/rate-limits |
| `[doc:versioning]` | https://platform.claude.com/docs/en/api/versioning |
| `[doc:migration]` | https://platform.claude.com/docs/en/api/migration-2026-10-21 |
| `[doc:api-overview]` | https://platform.claude.com/docs/en/api/overview (request size and response header tables) |
| `[doc:py-sdk]` | https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python (the older `/api/sdks/python` URL redirects here; the page describes SDK 2.x) |
| `[doc:auth]` | https://platform.claude.com/docs/en/manage-claude/authentication |
| `[doc:wif]` | https://platform.claude.com/docs/en/manage-claude/wif-reference |
| `[doc:wif-guide]` | https://platform.claude.com/docs/en/manage-claude/workload-identity-federation ("Token lifetime and refresh" section; added by the independent pass) |
| `[doc:mid-system]` | https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages (intro and availability only; added by the independent pass) |
| `[doc:message-threads]` | https://platform.claude.com/docs/en/build-with-claude/message-threads (intro only; added by the independent pass) |
| `[doc:sdk-versions]` | https://platform.claude.com/docs/sdk-versions.json |
| (index) | https://platform.claude.com/llms.txt ; https://platform.claude.com/docs/en/api/beta-headers |

Fetched but only searched for single statements: https://platform.claude.com/docs/en/build-with-claude/extended-thinking .

### Installed SDK source (anthropic 0.121.0, read 2026-11-19/20)

Root: the installed package directory, `site-packages/anthropic/`.

- `_version.py`, `_constants.py` (default timeout, retries, backoff constants)
- `_client.py` (constructor, credential precedence, headers, `_should_retry` for 401, `_make_status_error`)
- `_base_client.py` (`_calculate_nonstreaming_timeout`, `request`, `_attempt_request`, `_should_retry`,
  `_calculate_retry_timeout`, `_DefaultHttpxClient` keep-alive)
- `_exceptions.py`, `_streaming.py` (SSE decoding, `error` event handling), `_models.py` (`to_dict`, `to_json`),
  `_utils/_transform.py` and `_compat.py` (how response models are serialised when passed back)
- `resources/messages/messages.py` (`create`, `stream` signatures and bodies)
- `lib/streaming/_messages.py` (`MessageStream`, `MessageStreamManager`, `accumulate_event`, `build_events`)
- `lib/credentials/__init__.py`, `_types.py`, `_workload.py`, `_auth.py`, `_cache.py`, `_constants.py`
- `lib/tools/_beta_runner.py` (`append_messages`: how the SDK's own runner replays assistant content)
- `types/`: `message.py`, `usage.py`, `message_delta_usage.py`, `output_tokens_details.py`, `stop_reason.py`,
  `refusal_stop_details.py`, `content_block.py`, `text_block.py`, `thinking_block.py`, `redacted_thinking_block.py`,
  `tool_use_block.py`, `raw_message_delta_event.py`, `message_param.py`, `content_block_param.py`,
  `text_block_param.py`, `image_block_param.py`, `base64_image_source_param.py`, `tool_param.py`,
  `tool_choice_auto_param.py` / `_any_` / `_tool_` / `_none_`, `tool_result_block_param.py`,
  `thinking_block_param.py`, `redacted_thinking_block_param.py`, `tool_use_block_param.py`,
  `thinking_config_param.py` and its three members, `output_config_param.py`, `model.py`, `model_param.py`,
  `anthropic_beta_param.py`, `shared/error_type.py`

### Offline experiments (`[probe]`)

Run against the installed SDK, HTTP served by `httpx.MockTransport` inside the process; no socket was opened to
any API. The scratch scripts were not kept. They established: the non-streaming guard threshold and message;
the request bodies and headers the SDK sends; assembly of thinking / redacted_thinking / text / tool_use blocks from SSE
events; the missing `output_tokens_details` on the final message; lenient parsing of truncated tool JSON; the exception
types for pre-stream 400 / 429 / 529, mid-stream `error` events, mid-stream transport errors and incomplete streams;
retry counts; that dict payloads pass through unchanged; that the stream helper materialises null `caller` /
`citations`; that fake streams need `usage`; and (construct-only, with dummy ids and a provider that raises if called)
how `base_url` and `ANTHROPIC_BASE_URL` bind the WIF token exchange.

### Live observations (`[live]`)

Added 20 November 2026. These are requests that were sent to the API that day with this adapter's client on SDK
0.121.0. Run 1's requests were built by the adapter. In the two-call check, call 1 was built by the adapter and
call 2 by the probe script, which appended the turn (the block from the SDK's `model_dump()`) and the reminder as a
plain string; the adapter sends the reminder as one text block, which the API treats as the same thing (section 2),
and the adapter's own request for this shape was first seen live in item 3. They say what the API
did for those requests; none of it is a documented guarantee.

1. **Replay of a turn cut off while thinking: a two-call check per model.** The script is not in the repository.
   Models `claude-haiku-5-5`, `claude-fable-5-2`, `claude-sonnet-5`; the contestant's system prompt, task (the Great
   Wave, 64 wide) and tool; `thinking: {"type": "adaptive", "display": "summarized"}`, effort `high`,
   `max_tokens: 2500`.
   - Call 1 on each model: `stop_reason: "max_tokens"`, `content` of one `thinking` block with a signature. Usage
     in / out: 1,079 / 2,500 (Haiku 5.5, Fable 5.2), 1,145 / 2,500 (Sonnet 5).
   - Call 2: the same request with two messages appended, that assistant turn unchanged (null-valued keys dropped)
     and the user message "The picture is not finished. Missing rows: 1-43. Send them with write_rows.". **Accepted
     on all three models.** Input 3,614 / 3,614 / 3,677, which is 2,535 / 2,535 / 2,532 more than call 1.
   - What it shows: an assistant turn whose only block is a signed thinking block is accepted when a user message
     follows it, and the thinking is counted in the input. What it does not show: whether a model that gets its
     thinking back then finishes sooner (call 2 was itself cut off at 2,500 tokens, still thinking); effort `max`;
     a 128,000-token block in a thinking-only turn; `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-fable-5`; a turn
     that ends in thinking after a `text` or `tool_use` block; `redacted_thinking`; `display: "omitted"`.
2. **Run 1**, the first full run (six models at effort `max` with `display: "summarized"`, and Claude Haiku 5.5
   once more at its default effort; `max_tokens: 128000`), read from the transcripts it saved (405 attempts, 1,364
   replies). Run 2 was the same models, paintings and widths at effort `high`.
   - 300 replies ended with `stop_reason: "max_tokens"`, all at effort `max`. 291 of them held one `thinking` block
     and nothing else, and every one of the 291 had a non-empty signature (`claude-sonnet-5-5` 105,
     `claude-fable-5-2` 62, `claude-sonnet-5` 58, `claude-haiku-5-5` 46, `claude-opus-5-5` 19, `claude-fable-5` 1).
     No thinking block without a signature occurred in any reply of run 1 or run 2 (2,996 replies).
   - The adapter of that day did not send those turns back. 236 requests therefore held two or more user messages in
     a row (the nudge straight after the task, the tool results or an earlier nudge, with no assistant turn
     between). All were answered, and each one's input was 29 tokens larger than that of the call before. 55
     attempts ended incomplete after four such replies each.
   - The other 9 replies cut by `max_tokens` ended inside a tool call (`claude-sonnet-5` 5, `claude-fable-5-2` 2,
     `claude-sonnet-5-5` 2). Each turn was replayed with the SDK's shortened `input` object and a `tool_result` for
     every call; each next call was answered (`stop_reason: "tool_use"`) and reported 127,644 to 129,249 input
     tokens. No conversation of run 1 or run 2 was discarded.
   - 34 replies ended with `stop_reason: "refusal"`, each with one signed `thinking` block as its whole content. The
     harness stops at a refusal, so none was sent back.
3. **The small check at effort max** (six attempts at the Great Wave with prompt version A2, made with the
   adapter once it passed a cut-off thinking turn back; `max_tokens: 128000`, `display: "summarized"`). Every
   request was built by the adapter.
   - Four attempts began with a reply cut off while thinking (`stop_reason: "max_tokens"`, one signed `thinking`
     block): usage in / out 1,192 / 128,000 on `claude-opus-5-5` (64 wide), `claude-fable-5-2` and
     `claude-sonnet-5-5` (128 wide), 1,258 / 128,000 on `claude-sonnet-5` (128 wide).
   - Each second request held that turn and then the reminder as one text block. All four were accepted. Input:
     129,227 (`claude-opus-5-5`), 129,228 (`claude-fable-5-2`), 129,282 (`claude-sonnet-5`), 129,227
     (`claude-sonnet-5-5`), which is 128,024 to 128,036 more than the first call.
   - What followed: `claude-opus-5-5` called the tool after 8,852 more thinking tokens and finished in five calls;
     `claude-fable-5-2` after 1,350 more, one strip per reply, eight calls; `claude-sonnet-5` after 22,057 more, six
     tool calls in one reply, four calls in all. `claude-sonnet-5-5`'s second reply ended in `stop_reason:
     "refusal"` (category `reasoning_extraction`) after 18,074 output tokens. No conversation was discarded.
   - The other two attempts never reached a second request: `claude-fable-5` (128 wide) was blocked on its first
     call after 46,377 output tokens, `claude-opus-5` (16 wide) within three seconds with none.
   - What it shows: a thinking-only turn of 128,000 tokens goes back through this adapter at effort `max` on those
     four models, and three of the four models went on from it to finish. What it does not show: how often, since
     each is one attempt on one painting.

---

## Verification (independent pass)

Done 2026-11-20 by a second reviewer, who did not write the notes above. No request was sent to any model endpoint. The
only network traffic was documentation fetches from `platform.claude.com` with a generic User-Agent; the only code run
was the installed SDK against `httpx.MockTransport`.

### What was checked, and how

1. **Documentation, re-fetched rather than taken from the Sources table.** 46 pages plus `sdk-versions.json` and
   `llms.txt` were downloaded as markdown (`curl`, `.md` suffix).
   - Prose read from start to end (code samples in other languages skipped): thinking, thinking-steering-and-cost,
     thinking-troubleshooting, handling-stop-reasons, streaming, handle-tool-calls, errors, prefix-checks,
     versioning, models overview, Fable 5.2 overview.
   - Read in the parts these notes rely on: effort (through the per-model recommendations), refusals-and-fallback
     (response shape, billing, pitfalls), change-tools-or-instructions (through the tool-loop details), vision
     (sending images, limits, costs, limitations), context-windows (through thinking and tool use), define-tools
     (definition table, forcing tool use), parallel-tool-use (execution, disabling), the Messages create reference
     (headers, every top-level body parameter, the message / tool / tool_use / tool_result / image block schemas,
     the response schema, the error-code list, the example), migration-2026-10-21 (upgrade steps, Messages API
     changes, section list), the Python SDK page (all prose), pricing (model table, long-context, tool-overhead and
     batch sections), rate-limits (spend cap, Messages limits, header table).
   - Searched for the specific statements cited: Fable 5.2 migration guide and what's-new page, the Opus 5.5 /
     Sonnet 5.5 / Haiku 5.5 / Mythos 5.2 model pages, preserved-thinking, thinking-tool-workflows,
     vision-coordinates, API overview, authentication, wif-reference, the WIF guide, structured-outputs,
     strict-tool-use, prompt-caching, batch-processing, fix-bound-to-a-different-conversation,
     mid-conversation-system-messages, message-threads, fallback-credit, llms.txt.
   - Downloaded but not opened: beta-headers, extended-thinking.
2. **Quoted error messages, enums, limits, prices, model ids, release dates and defaults** in sections 1 to 9 and 11
   were compared one by one with the page or SDK file they are attributed to. The per-model thinking table, the
   effort table, the stop-reason list, the refusal categories and billing column, the usage field list, the image
   limits and token table, the price table (including cache-read multipliers and the Haiku 5.5 long-prompt tier) and
   the tool-overhead token counts all match the live pages. No mismatch was found beyond the corrections below.
3. **SDK source read directly**: `types/` response and request models (`message.py`, `usage.py`,
   `output_tokens_details.py`, `stop_reason.py`, `refusal_stop_details.py`, `content_block.py`, the four block types,
   `message_delta_usage.py`, `raw_message_delta_event.py`, `tool_param.py`, the four `tool_choice_*_param.py`,
   `tool_result_block_param.py`, `tool_use_block_param.py`, the thinking config params, `output_config_param.py`,
   `message_param.py`, `model_param.py`, image params), `_client.py` (constructor, auth headers, 401 retry,
   `_make_status_error`), `_base_client.py` (non-streaming guard, retry policy, backoff, keep-alive client),
   `_exceptions.py`, `_streaming.py`, `_models.py`, `_constants.py`, `resources/messages/messages.py` (`stream`
   signature, guard condition), `lib/streaming/_messages.py` (`accumulate_event`, `build_events`),
   `lib/credentials/_workload.py`, `_constants.py`, `_cache.py`, `_auth.py`, `lib/tools/_beta_runner.py`.
4. **Section 10 fixtures, field by field.** All ten JSON blocks were extracted from this file by script. The five
   response objects were validated with `anthropic.types.Message.model_validate` (no extra or unknown fields
   reported). The stream event lists were served as SSE through `httpx.MockTransport` and assembled with
   `client.messages.stream(...).get_final_message()`; the request-1 body and the request-2 `messages` array were
   compared for equality with what the SDK put on the wire; the replay payload rule applied to the assembled
   response 1 reproduces the request-2 assistant content exactly. The reference page's own example response and the
   refusal page's example were validated the same way.
5. **The call recipe in section 1** was run as written against the fixtures (event loop, `get_final_message()`,
   `stream.request_id`, `stream.response.headers`, `to_dict(mode="json")`).
6. **Behaviour claims tagged `[probe]` were re-run independently**: guard threshold (21,333 passes, 21,334 raises;
   silenced by a client timeout), missing `output_tokens_details` after assembly, lenient parsing of a truncated
   tool call, null `caller` / `citations` materialised by the stream helper, pass-through of dict payloads and of
   request keys unknown to the SDK types, shallow `extra_body` merge, exception classes for 400 / 402 / 408 / 413 /
   429 / 503 / 504 / 529, mid-stream `error` event (plain `APIStatusError`, status 200), mid-body transport failures
   (raw `httpx` exceptions, not retried), connect failure and pre-response timeout (wrapped and retried), retry
   counts, default timeout, WIF header set, `base_url` binding with and without `ANTHROPIC_BASE_URL`, https
   requirement, 401 refresh-and-retry with `max_retries` 0 / 1 / 2, token-exchange failure class, credential
   precedence with an explicit `credentials=` and with an empty `ANTHROPIC_API_KEY`.

The scratch scripts and the fetched pages were not kept.

### Corrections made (old -> new)

1. Section 1, "`message._request_id` is `None` on a stream-assembled message" -> **it raises `AttributeError`**; the
   attribute exists only on objects returned by non-streaming calls. (An adapter that read it would crash on every
   call.)
2. Section 1, "for a wall-clock cap per call, check a deadline inside the event loop" -> the loop body does not run
   during silent phases, because the SDK swallows `ping` events without yielding them; a deadline in the loop cannot
   cut a silent call, and the read timeout is not a wall-clock cap. A hard cap needs something outside the loop
   (untested, now listed under UNVERIFIED).
3. Sections 3 and 9, "a stream that closes without a `message_delta` raises nothing" -> true only after
   `message_start`; a 200 response with no events makes `get_final_message()` raise a bare `AssertionError`. Added
   it, and the `AttributeError` for events without `usage`, to the error mapping as retryable.
4. Section 2, `messages` roles "`user` | `assistant`" and "there is no system role inside `messages`" -> a
   `"system"` role does exist (mid-conversation system messages, with placement rules); the instruction not to use
   it for the initial prompt stands and is now explicit.
5. Section 1, "it is stateless" -> qualified: a beta "message threads" feature exists and is not used.
6. Section 3, usage table: `speed` was listed without saying that 0.121.0's `Usage` type lacks it -> noted (kept
   as an extra field; read `usage` defensively).
7. Section 9: added the documented `error.details.error_code` field, with `enforced_spend_limit_reached` (a 429 that
   never recovers by retrying; map to `FatalError`) and `ramp_limit_exceeded`; added the reference's rule for
   identifying mid-stream errors.
8. Section 11 item 18: added that `anthropic>=0.121` with no upper bound resolves to SDK 2.x on a fresh install,
   where an `httpx` client passed as `http_client` raises `TypeError` and the API version changes; recommended
   pinning below 1.0, which `pyproject.toml` now does.
9. Sources: two URLs replaced by the addresses they now redirect to (`/docs/en/models/overview`,
   `/docs/en/cli-sdks-libraries/sdks/python`); three pages added.

### Additions (documented facts that were missing)

- Section 1: the docs' warning that requests with more than roughly 32k thinking tokens "can run long enough to hit
  system timeouts and open connection limits", and the server-side `timeout_error`.
- Section 1: documented WIF token lifetime (default 3,600 s; capped at twice the identity token's remaining life).
- Section 1: how to restore `output_tokens_details` in the dict handed to `parse_response()`.
- Section 3: a thinking block with an empty `signature` cannot be replayed; the replay payload keeps the SDK's
  shortened `input` object; default buffering of tool-input keys and values on the stream.
- Section 4: with 0.121.0 the documented `message.content` replay pattern sends `"citations": null` /
  `"caller": null` itself; `is_error` cannot be set from paintbench's neutral tool message; `tool_use_id` pattern.
- Section 6: `max_tokens` does not count toward output-token rate limits.
- Section 7: the judge prompt can trigger a `reasoning_extraction` refusal.
- Section 10: keys a real non-streaming response carries beyond the fixtures.
- UNVERIFIED: items 2, 3, 6 and 8 rewritten to separate what is documented from what is open; items 14 (empty
  response text block) and 15 (closing a stream from another thread) added.

One sentence written during this pass was removed again after an offline test contradicted it (a claim that
`exclude_none=True` strips nulls inside tool `input`; it does not).

### Checked and left unchanged

Endpoint and method; the tool definition format (`name`, `description`, `input_schema`; no wrapper object, no
`type` needed); `tool_choice` shapes and the forced-tool 400; where tool calls appear (`content[]` blocks of type
`tool_use` with `id`, `name`, `input` as an object, `caller`); the tool-result message format and ordering rules;
the replay rules for `thinking` / `redacted_thinking` blocks and the prefix check; `thinking` and
`output_config.effort` values and the per-model tables; the seven `stop_reason` values; the refusal shape; usage
field names and the inclusive meaning of `output_tokens`; output and context limits; image block format and
limits; model ids, release dates and prices; status-to-exception mapping; retry policy; every JSON fixture in
section 10.

### What remains unverified

(As this pass left it. Since then, on 20 November 2026, live calls settled one half of item 4: a turn of one signed
thinking block is accepted. They also showed a turn with a cut tool call replayed without a rejection, which bears on
item 6. See "Live observations" under "Sources". The rest of this list stands.)

- Everything under "UNVERIFIED (from memory)", items 1 to 15. The ones most likely to matter in a live run:
  ping cadence during long silent thinking (item 1), an open stream outliving its WIF token (item 2), replay of a
  `max_tokens`-truncated turn (item 6), and SDK 0.121.0's `anthropic-version: 2023-06-01` against
  `claude-fable-5-2` (item 8).
- All API behaviour is taken from documentation; nothing here was observed on the wire. In particular, whether a
  live `tool_use` `content_block_start` carries `caller`, and which optional `usage` keys appear on live stream
  events, are known only from the schema.
- Pages only searched, not read in full (listed in step 1 above). Statements taken from them are limited to the
  sentences quoted.
- The suggested mapping onto paintbench's types (section 4 table, section 9 table) is advice, not a documented
  contract; it was checked for consistency with `paintbench/types.py`, `paintbench/adapters/base.py` and
  `paintbench/harness.py`, not executed.
