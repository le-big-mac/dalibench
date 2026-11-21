# Google Gemini (Gemini Developer API, `generateContent`, `google-genai` SDK) - notes for the `google` adapter

Researched 2026-11-19 (UTC). No model/inference endpoint was called while these notes were written.
Independently re-verified 2026-11-20 (every doc page re-fetched, every [RUN] claim re-executed offline, every
JSON block in this file re-parsed through the installed SDK); what was checked and what was changed is listed
in the last section, "Verification (independent pass)".

The adapter written from these notes has since been run against the live API (20 November 2026, model
`gemini-4-carbon`). The notes were not revised after that run: nothing in them rests on a live call, and the
list under "UNVERIFIED (from memory)" stands as it was written.

Evidence tags used on every claim:

- **[RUN]** executed offline against the *installed* `google-genai 2.17.0` through `httpx.MockTransport`
  (no network). Wire JSON and parsed objects shown under this tag are real SDK output.
- **[SRC]** read in the installed SDK source, `site-packages/google/genai/<file>` + symbol.
- **[DOC:key]** official documentation page fetched for these notes; keys resolve in "Sources" at the end.
- **[SRC-NEW]** read (NOT installed, NOT executed) in the `google-genai` 2.34.0 and 3.6.1 wheels downloaded from PyPI.
  Use these only to know what changes if the SDK pin is raised.

Anything without one of these tags lives in "UNVERIFIED (from memory)" near the end.

---

## 0. Read this first (things that differ from what older knowledge would predict)

1. **The current models are Gemini 4**: `gemini-4-carbon` (strongest), `gemini-4-neon`, `gemini-4-hydrogen`,
   `gemini-4-argon`. Gemini 2.5 Pro and 2.5 Flash are deprecated (rerouted from 2026-12-01), Gemini 3 Pro Preview
   and Gemini 2.0 are shut down. [DOC:models]
2. **The installed SDK (2.17.0, released 2026-08-06) predates Gemini 4.** Google's docs for Gemini 4 require
   `google-genai>=2.30.1` for "long decode" (automatic continuation past the 64K per-request cap). [DOC:long-decode]
   On 2.17.0 the fields the current docs use do not exist: `types.GenerateContentConfig(automatic_continuation=...)`
   or `(continuation_token=...)` raises `pydantic.ValidationError: Extra inputs are not permitted`. [RUN]
   **The choice: raise the pin to `google-genai>=2.30.1,<3` (2.34.0 is the last 2.x) for continuation, or cap
   every call at one 64K hop. The adapter caps every call at one hop:** `pyproject.toml` pins
   `google-genai>=2.17,<3`, the adapter never sends more than 65,536 as `max_output_tokens`, and where the SDK has
   automatic continuation the adapter switches it off. Section 1.6 has the version matrix; the adapter recipe in
   these notes works on both.
3. **Hard server limits per HTTP request** [DOC:long-decode]: non-streaming calls are closed by the server after
   **10 minutes**, streaming calls after **20 minutes**; one request ("hop") returns at most **65,536 tokens of
   thinking + output combined**. A 10-30 minute call therefore *must* be streamed, and anything that needs more
   than 64K tokens or more than 20 minutes needs continuation hops.
4. **`max_output_tokens` counts thinking tokens** on Gemini 4 (it caps thinking + visible output together). A low
   value produces `MAX_TOKENS` with empty text. [DOC:thinking]
5. **Thinking cannot be turned off on Gemini 4**; control is `thinking_level`, not `thinking_budget`. Highest level
   is `high` on carbon/argon and `xhigh` on neon/hydrogen. [DOC:thinking]
6. **Thought signatures on function-call parts are mandatory**: drop one and the next request fails with HTTP 400.
   Replay the model's `Content` exactly as received. [DOC:thought-signatures]
7. The docs now call `generateContent` "legacy" (still "fully supported") and recommend the Interactions API for
   new projects. [DOC:llms][DOC:interactions] This adapter uses `generateContent`; see 1.7.

---

## 1. Endpoint, SDK method, client, timeouts, streaming

### 1.1 Endpoint and method

- REST: `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse`
  (streaming, server-sent events) and `POST .../v1beta/models/{model}:generateContent` (unary). The request body
  is identical for both. Auth header: `x-goog-api-key: <key>`. [RUN][DOC:ref-gc][DOC:text]
- SDK: `client.models.generate_content_stream(model=..., contents=..., config=...)` returns a **generator** of
  `types.GenerateContentResponse` chunks; `client.models.generate_content(...)` returns one
  `types.GenerateContentResponse`. Both are keyword-only. [SRC models.py `Models.generate_content`,
  `Models.generate_content_stream`]
- **Use `generate_content_stream` for both the contestant and the judge call, and assemble the chunks yourself.**
  There is no SDK helper that returns a final assembled message for a stream (none in 2.17.0, 2.34.0 or 3.6.1).
  [SRC][SRC-NEW] The assembler is ~20 lines (section 1.5) and was run offline. [RUN]

Why streaming:

| | non-streaming `generateContent` | streaming `streamGenerateContent` | source |
|---|---|---|---|
| server closes the connection after | 10 minutes | 20 minutes | [DOC:long-decode] "Handle long-running requests" |
| per-request cap, neon / hydrogen | 65,536 tokens | 65,536 tokens | [DOC:long-decode] |
| per-request cap, carbon / argon | 32,768 on a "direct REST" call; docs say 64K "through the Google GenAI SDK" | 65,536 tokens | [DOC:long-decode] |
| `max_output_tokens` > 32,768 on carbon/argon without a continuation token | HTTP 400 on a direct REST call | allowed | [DOC:long-decode] |

The SDK's non-streaming request is a plain POST to `:generateContent` (identical code in 2.17.0 and 2.34.0
[SRC][SRC-NEW]); how the server tells "SDK" from "direct REST" is not documented. Streaming avoids the ambiguity.

### 1.2 Client construction and authentication

```python
import httpx
from google import genai
from google.genai import types, errors

def make_client(api_key: str | None = None, *, base_url: str | None = None,
                httpx_client: httpx.Client | None = None) -> genai.Client:
    return genai.Client(
        api_key=api_key,                      # None -> SDK reads the environment (below)
        http_options=types.HttpOptions(
            api_version="v1beta",             # explicit: SDK 3.x defaults to "v1", 2.x to "v1beta"
            timeout=1_200_000,                # MILLISECONDS (20 min); see 1.3
            base_url=base_url,                # tests / proxy; None -> https://generativelanguage.googleapis.com/
            httpx_client=httpx_client,        # tests: httpx.Client(transport=httpx.MockTransport(handler))
            retry_options=types.HttpRetryOptions(attempts=1),   # explicit "no SDK retries"; see section 9
        ),
    )
```
This exact function was run offline. [RUN]

- `genai.Client.__init__` is keyword-only: `enterprise, vertexai, api_key, credentials, project, location,
  debug_config, http_options`. For the Developer API pass only `api_key` and `http_options`. [SRC client.py `Client.__init__`]
- **Environment variables** [SRC _api_client.py `get_env_api_key`][RUN][DOC:api-key]: `GOOGLE_API_KEY` first, then
  `GEMINI_API_KEY`. If both are set, `GOOGLE_API_KEY` wins and the SDK logs a warning. An explicit `api_key=`
  argument beats both. No key at all -> `ValueError("No API key was provided...")` at construction.
- Do not set `GOOGLE_GENAI_USE_VERTEXAI` / `GOOGLE_GENAI_USE_ENTERPRISE` (they switch the client to Vertex).
  [SRC _api_client.py `BaseApiClient.__init__`]
- **Base URL override** (priority order) [SRC _base_url.py `get_base_url`][RUN]: `HttpOptions(base_url=...)`, then
  `genai._base_url.set_default_base_urls(...)`, then env `GOOGLE_GEMINI_BASE_URL`. The SDK appends
  `/{api_version}/models/{model}:generateContent`, e.g. `http://127.0.0.1:9/fake/v1beta/models/gemini-4-carbon:generateContent`.
- **Injecting a transport for tests**: `HttpOptions(httpx_client=httpx.Client(transport=httpx.MockTransport(fn)))`
  (sync) and `HttpOptions(httpx_async_client=httpx.AsyncClient(...))` (async). [SRC types.py `HttpOptions`][RUN]
  Where `aiohttp` is not installed, the async client uses httpx. [RUN]
- Model name: pass the bare id (`"gemini-4-carbon"`); the SDK turns it into `models/gemini-4-carbon`. Names
  containing `..`, `?` or `&` raise `ValueError`. [SRC _transformers.py `t_model`]
- Async variant: `await client.aio.models.generate_content(...)`; streaming is
  `async for chunk in await client.aio.models.generate_content_stream(...)` (note the `await`). [RUN]
- `client.close()` closes the sync httpx client. [SRC client.py `Client.close`][RUN]

### 1.3 Timeouts

- `HttpOptions.timeout` is an **int in milliseconds**. [SRC types.py `HttpOptions.timeout`; _api_client.py `get_timeout_in_seconds`]
- **Default is no timeout at all**: with `timeout` unset the SDK sends every request with httpx
  `connect/read/write/pool = None`. A timeout configured on your own `httpx.Client(timeout=5.0)` is *ignored*,
  because the SDK passes its own per-request timeout. [RUN]
- With `timeout=1_200_000` the request gets `connect/read/write/pool = 1200.0` s and the SDK also sends the header
  `X-Server-Timeout: 1200`. [RUN][SRC _api_client.py `populate_server_timeout_header`]
- For a stream, httpx's read timeout is the maximum gap between received bytes, not a total deadline.
- Per-call override: `GenerateContentConfig(http_options=types.HttpOptions(timeout=600_000))`. [RUN]
- Google's guidance: "Set your HTTP read timeout to 20 minutes. A 3-minute or 5-minute timeout stops the request
  early", with the example `genai.Client(http_options=types.HttpOptions(timeout=1_200_000))`. [DOC:long-decode]
  The Gemini 4 migration checklists repeat "Set HTTP client timeouts to at least 20 minutes per call". [DOC:carbon]
- A timeout surfaces as a raw `httpx.ReadTimeout` (subclass of `httpx.TimeoutException`), not as an SDK error. [RUN]

### 1.4 What a long generation looks like (three nested limits) [DOC:long-decode][DOC:thinking][DOC:carbon]

1. **One HTTP request ("hop")**: at most 65,536 tokens of thinking + output, at most 20 minutes when streamed.
2. **One generation**: `max_output_tokens` is the total of thinking + output across all hops. If it is unset or
   <= the hop cap, the model stops at the cap with `finishReason: "MAX_TOKENS"`. If it is larger than the hop cap
   and the model has budget left when the hop fills up, the hop ends with `finishReason: "CONTINUATION"` and
   `candidates[0].continuationToken`; the client must send **the same request again** plus
   `"continuationToken": "<token>"` (top level of the request body) and keep going while the finish reason is
   `CONTINUATION`. "On hard problems, a hop can spend all of its tokens on internal thinking and return empty text
   with a `continuation_token`. This is normal."
3. **Context**: input + thinking + output share one 1,048,576-token window on Gemini 4.

Who runs the hops:

- `google-genai >= 2.30.1`: the SDK, by default (`automatic_continuation` defaults to on) for both
  `generate_content` and `generate_content_stream`; "Timeouts and retries apply to each hop, not to the full
  run". [DOC:long-decode][SRC-NEW models.py `_generate_content_with_continuation`,
  `_generate_content_stream_with_continuation`; _extra_utils.py `StreamContinuation`] When streaming, the SDK
  keeps yielding chunks across hops, clears the intermediate `CONTINUATION` finish reason on the chunks it
  yields, and makes each chunk's `usage_metadata` include all earlier hops - so the assembler in 1.5 sees one
  long stream ending in the final hop's finish reason. If a stream drops after a checkpoint the SDK resumes from
  it, giving up after 3 reconnects without progress. [SRC-NEW _extra_utils.py `StreamContinuation.observe`,
  `end_hop`, `MAX_STREAM_RECONNECTS_WITHOUT_PROGRESS`] None of this was executed here.
  **Caveat for the assembler on >= 2.30.1** [SRC-NEW models.py `_generate_content_stream_with_continuation`;
  _extra_utils.py `StreamContinuation.end_hop`, `_next_resume_token`]: the SDK yields every chunk to the caller
  immediately and, after a dropped stream, silently starts a new hop from the last checkpoint token. Chunks it
  had already yielded *after* that checkpoint are not retracted and the caller gets no reconnect signal, whereas
  the docs' rule for a hand-written loop is "If the stream drops, discard the pending chunks and resume from the
  last token" [DOC:long-decode]. So after a mid-hop reconnect the plain concatenation in 1.5 can contain a
  duplicated or divergent tail. Checkpoints start after the first 32,768 tokens of a hop on carbon/argon (after
  65,536 on neon/hydrogen), one every 5,000 tokens [DOC:long-decode], so this only concerns long hops. If the
  pin is raised and this matters, pass `automatic_continuation=False` (the SDK then makes exactly one request and
  re-raises stream errors, as 2.17.0 does) and keep `max_output_tokens <= 65_536`, or write the docs' own loop.
- `google-genai 2.17.0` (installed): nobody. The SDK has no continuation code; it parses `CONTINUATION` into an
  unknown-enum pseudo-member (with a `UserWarning`) and **drops** `continuationToken` from the candidate. [RUN]
  So on 2.17.0 keep `max_output_tokens <= 65536`, which makes the server end at the cap with `MAX_TOKENS`
  instead of `CONTINUATION`. A hand-written continuation loop through the 2.17.0 streaming method is not
  possible either: the token never reaches the parsed chunk, `chunk.sdk_http_response.body` is `None`, and
  `GenerateContentConfig(should_return_http_response=True)` raises `ValueError: Accessing the raw HTTP response
  is not supported in streaming methods`. (Only the request half works:
  `http_options=types.HttpOptions(extra_body={"continuationToken": "<token>"})` lands at the top level of the
  body.) [RUN]

### 1.5 The simplest reliable way to get one final assembled response [RUN]

```python
def call(client, model, contents, config):
    """Stream one request; return one assembled result. Works on 2.17.0 and on >=2.30.1."""
    parts, finish_reason, usage, block_reason = [], None, None, None
    model_version = response_id = None
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
        if chunk.usage_metadata is not None:
            usage = chunk.usage_metadata          # cumulative: keep the LAST one, do not sum chunks
        if chunk.prompt_feedback is not None and chunk.prompt_feedback.block_reason is not None:
            block_reason = chunk.prompt_feedback.block_reason
        model_version = chunk.model_version or model_version
        response_id = chunk.response_id or response_id
        if not chunk.candidates:                  # None when the prompt was blocked
            continue
        cand = chunk.candidates[0]
        if cand.content is not None and cand.content.parts:      # both can be None
            parts.extend(p for p in cand.content.parts if p != types.Part())   # NEVER merge parts
        if cand.finish_reason is not None:
            finish_reason = cand.finish_reason
    content = types.Content(role="model", parts=parts) if parts else None
    return dict(content=content, finish_reason=finish_reason, usage=usage, block_reason=block_reason,
                model_version=model_version, response_id=response_id)
```

Rules baked into it, each with its source:

- The generator is lazy: the HTTP request is sent on the first `next()`, so HTTP errors are raised inside the
  `for`, not at the call. Wrap the whole loop. [RUN]
- **Concatenate parts in arrival order into one model turn and never merge or rewrite them.** Docs: "Don't
  concatenate parts with signatures together. Don't merge one part with a signature with another part without a
  signature." [DOC:thinking] Google's own manual continuation loop does `parts.extend(candidate.content.parts)`
  and says "Combine the parts from all hops into one model turn for the conversation history" [DOC:long-decode];
  the newer SDK merges hops by plain list concatenation "keeping each hop's Part distinct and preserving empty
  placeholder Parts". [SRC-NEW _extra_utils.py `merge_pydantic_models`]
- Keep empty-text parts that carry a signature: "the model may return the thought signature in a part with an
  empty text content part. It is advisable to parse the entire request until the `finish_reason` is returned".
  [DOC:thought-signatures FAQ 3]
- `usage_metadata` is cumulative within a stream ("Keep only the last value of each stream"). [DOC:long-decode]
- **`finish_reason is None` after the loop means the stream ended early** (server cut, dropped connection, zero
  chunks): the SDK raises nothing when a stream simply ends. [RUN] Treat it as a failed call and retry; do not
  treat it as a completed turn.
- `finish_reason` is a `str`-based enum: `r["finish_reason"] == "MAX_TOKENS"` is `True`, `.value` is the string,
  and set/dict lookups with plain strings work (`fr in {"STOP"}`). **`str(fr)` and f-strings give
  `"FinishReason.MAX_TOKENS"`, not `"MAX_TOKENS"`** - store and log `fr.value`. `json.dumps` emits the bare
  value. [RUN]
- **A `functionCall` part always arrives whole, in one chunk, with complete `args`.** Docs: "With
  `generateContent`, streaming function calls arrived complete in a single chunk. You could not see the
  arguments being generated in real-time, so the handler simply checked for a complete `functionCall` object."
  [DOC:migrate "Streaming tools and function calls"] Incremental argument streaming
  (`FunctionCallingConfig.stream_function_call_arguments`, `FunctionCall.partial_args`,
  `FunctionCall.will_continue`) is "not supported in Gemini API" and the SDK raises `ValueError` if it is
  requested on the Developer API; the REST `FunctionCall` has only `name`, `id`, `args`. No partial-JSON
  accumulation is needed. [SRC types.py `FunctionCall`, `FunctionCallingConfig`; models.py
  `_FunctionCallingConfig_to_mldev`][DOC:ref-overview]
- The SDK's own chat helper applies the same completeness rule: a streamed turn is only kept in its history when
  a `finish_reason` was seen. [SRC chats.py `Chat.send_message_stream`]

### 1.6 SDK version matrix

| | 2.17.0 (installed) | 2.34.0 (last 2.x, 2026-10-29) | 3.6.1 (latest, 2026-11-18) |
|---|---|---|---|
| default API version for the Developer API | `v1beta` [RUN] | `v1beta` [SRC-NEW] | **`v1`** (`GOOGLE_AI_API_DEFAULT_VERSION`; env `GOOGLE_GENAI_API_VERSION` overrides) [SRC-NEW][DOC:api-versions] |
| HTTP library | `httpx` | `httpx` required, `httpx2` optional | **`httpx2` required**, `httpx` only as an extra; `httpx_client` accepts either [SRC-NEW METADATA, types.py] |
| `GenerateContentConfig.automatic_continuation`, `.continuation_token` | **absent -> ValidationError** [RUN] | present; continuation on by default | present |
| `Candidate.continuation_token` | absent; wire field silently dropped [RUN] | present (`bytes`) | present |
| `FinishReason.CONTINUATION` | unknown: pseudo-member + `UserWarning` [RUN] | member | member |
| `ThinkingLevel.XHIGH` | absent; `"xhigh"` is passed through with a `UserWarning` [RUN] | absent (same behaviour) | member |
| `MISSING_THOUGHT_SIGNATURE`, `MALFORMED_RESPONSE`, `ESCALATION`, `PUP_LIMITED_DISABLED` finish reasons | unknown (pseudo-member + warning) | unknown | members |
| `GenerateContentConfig.store` | absent | present | present |
| `usage_metadata.initial_prompt_token_count` | absent | present (property) | present |
| `Part.from_function_response(..., id=)` | **no `id` parameter -> TypeError** [RUN] | no `id` parameter | no `id` parameter |
| retries without `retry_options` | none [RUN] | none | none |

Release dates are from the PyPI JSON for `google-genai`. [PyPI]

Recommendation: pin `google-genai>=2.30.1,<3` (2.34.0). It keeps `httpx` and `v1beta`, so everything tagged [RUN]
in these notes is expected to carry over, and it adds continuation. If 3.x is chosen instead, keep
`api_version="v1beta"` explicit (as in `make_client`) and add the `httpx` extra or switch the test transport to
`httpx2`. (The adapter keeps the lower bound at 2.17 and makes one request per call on every version: section 0,
item 2.) Guard for both worlds in the adapter:

```python
HAS_CONTINUATION = "automatic_continuation" in types.GenerateContentConfig.model_fields   # False on 2.17.0 [RUN]
max_output_tokens = wanted if HAS_CONTINUATION else min(wanted, 65_536)
```

### 1.7 Why not the Interactions API (and what it would buy)

- Docs: "Use the Interactions API (`client.interactions`) as the default across all workflows. `generateContent`
  is still supported for existing integrations, but is considered a legacy API" [DOC:llms]; "As of June 2026, it
  is Generally Available and the recommended interface for all new projects. While it is now considered legacy,
  the original `generateContent` API remains fully supported." [DOC:interactions] No shutdown date for
  `generateContent` appears on the deprecations page. [DOC:deprecations]
- What it would buy: `background=True` runs the generation (including long decode) on the server; the client gets
  an id and polls `client.interactions.get(id=...)` through states `in_progress`, `requires_action`, `completed`,
  `failed`, `cancelled`. Background requires `store=True` (interaction stored server-side, 55 days by default on
  the paid tier). [DOC:background]
- It is a different request/response schema (`input`, `steps`, `previous_interaction_id`). **It was not
  researched to adapter depth here.** If the 20-minute-per-hop model proves painful, that is the documented
  escape hatch and needs its own research pass.
- Do not use the OpenAI-compatibility endpoint: "Long decode is not available. Output stops at the per-request
  cap." [DOC:long-decode]

---

## 2. Request shape for a contestant call

### 2.1 SDK call

```python
WRITE_ROWS = types.FunctionDeclaration(
    name="write_rows",
    description="Write consecutive rows of the painting grid, starting at start_row.",
    parameters_json_schema={                       # plain JSON Schema, passed through untouched
        "type": "object",
        "properties": {
            "palette":   {"type": "array", "items": {"type": "string"}, "description": "..."},
            "start_row": {"type": "integer", "description": "..."},
            "rows":      {"type": "array", "items": {"type": "string"}, "description": "..."},
        },
    },                                             # no "required" list
)

config = types.GenerateContentConfig(
    system_instruction="<system prompt>",
    tools=[types.Tool(function_declarations=[WRITE_ROWS])],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    thinking_config=types.ThinkingConfig(thinking_level="high", include_thoughts=True),
    max_output_tokens=65_536,
    # tool_config omitted -> AUTO.  temperature / top_p / top_k / candidate_count: never send (section 5).
)
contents = [types.Content(role="user", parts=[types.Part(text="<user message>")])]
result = call(client, "gemini-4-carbon", contents, config)
```

### 2.2 What the installed SDK actually puts on the wire for that call [RUN]

```json
{
  "contents": [
    {"parts": [{"text": "Paint: The Starry Night (16x16)."}], "role": "user"}
  ],
  "systemInstruction": {
    "parts": [{"text": "You are a painter. Type the painting cell by cell with write_rows."}],
    "role": "user"
  },
  "tools": [
    {"functionDeclarations": [
      {"description": "Write consecutive rows of the painting grid, starting at start_row.",
       "name": "write_rows",
       "parameters_json_schema": {
         "type": "object",
         "properties": {
           "palette":   {"type": "array", "items": {"type": "string"}, "description": "Optional list of colours."},
           "start_row": {"type": "integer", "description": "Index of the first row to write."},
           "rows":      {"type": "array", "items": {"type": "string"}, "description": "One string per row."}
         }
       }}
    ]}
  ],
  "generationConfig": {
    "maxOutputTokens": 65536,
    "thinkingConfig": {"include_thoughts": true, "thinking_level": "HIGH"}
  }
}
```

Notes on that body:

- **Mixed key casing is real.** The SDK converts the outer layers to camelCase but dumps nested pydantic objects
  with their Python names: `thinkingConfig.thinking_level`, `thinkingConfig.include_thoughts`,
  `functionDeclarations[].parameters_json_schema`. [RUN] The REST reference names are `thinkingLevel`,
  `includeThoughts`, `parametersJsonSchema` [DOC:ref-gc]; the API accepts the snake_case spellings too (Google's
  own REST samples send `inline_data`, `mime_type`, `generation_config`, `media_resolution`,
  `thought_signature`). [DOC:media-resolution][DOC:image][DOC:thought-signatures] Tests that assert on the raw
  request body must expect the casing above, or compare semantically.
- `systemInstruction` is a `Content`; the SDK wraps a string as `{"parts":[{"text":...}],"role":"user"}`. [RUN]
  "Currently, text only." [DOC:ref-gc]
- `tools`, `toolConfig`, `systemInstruction`, `safetySettings`, `cachedContent` are top-level request fields;
  `maxOutputTokens`, `thinkingConfig`, `mediaResolution`, `temperature` live in `generationConfig`.
  [SRC models.py `_GenerateContentConfig_to_mldev`][DOC:ref-gc]
- `automatic_function_calling` is SDK-only and never appears on the wire. [RUN]

### 2.3 Function declaration: `parameters` vs `parameters_json_schema`

- `FunctionDeclaration` fields: `name`, `description`, `parameters: Optional[Schema]`,
  `parameters_json_schema: Optional[Any]`, `response`, `response_json_schema`, `behavior`.
  [SRC types.py `FunctionDeclaration`] REST: `parameters` (OpenAPI-3.0-subset `Schema`) and
  `parametersJsonSchema` (JSON Schema object) are **mutually exclusive**; `name` and `description` are required.
  [DOC:ref-gc] The SDK does not check exclusivity locally. [RUN]
- `parameters_json_schema=<dict>` is sent unchanged (lower-case JSON Schema types). [RUN]
- `parameters=<dict>` is converted to `types.Schema` and sent with **upper-case** types:
  `{"properties": {"palette": {"items": {"type": "STRING"}, "type": "ARRAY"}, "start_row": {"type": "INTEGER"}, ...}, "type": "OBJECT"}`.
  [RUN] Most examples in the function-calling guide use this `parameters` form with a plain dict (sometimes
  passed as a whole dict declaration: `types.Tool(function_declarations=[{"name":..., "parameters": {...}}])`);
  one Python example (section "Slice and inspect files in function responses") builds the declaration exactly as
  in 2.1, `types.FunctionDeclaration(name=..., description=..., parameters_json_schema={"type": "object",
  "properties": {...}, "required": [...]})`. [DOC:function-calling] `types.Schema` is strict (`extra='forbid'`), so an unsupported JSON-Schema keyword in
  `parameters` raises a local `ValidationError`; `parameters_json_schema` has no local validation.
  [SRC _common.py `BaseModel.model_config`]
- Either works for `write_rows`. Function names: letters, digits, underscores, dots, colons, dashes; max 128;
  must start with a letter or underscore. [SRC types.py `FunctionDeclaration.name`][DOC:ref-gc]
- `Tool(function_declarations=[...])`: all declarations go in one `Tool`; the SDK merges several function tools
  into one anyway. [SRC _transformers.py `t_tools`] Max 512 declarations. [SRC types.py `Tool`]

### 2.4 Making sure automatic function calling (AFC) is off

- Pass `automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)`. With it the SDK takes
  the direct path and returns the raw function call. [SRC _extra_utils.py `should_disable_afc`; models.py
  `generate_content` / `generate_content_stream`][DOC:function-calling "You can disable automatic function calling with"]
- AFC only ever executes **Python callables** passed in `tools`. A `types.Tool` carrying `function_declarations`
  marks the request AFC-incompatible and the SDK also goes straight to the single raw request, even without
  `disable=True`. [SRC _extra_utils.py `find_afc_incompatible_tool_indexes`] Never put a Python function in
  `tools`.
- Without the flag and without function tools (e.g. the judge call) the SDK logs `AFC is enabled with max remote
  calls: 10.` at INFO and still makes exactly one request; with `disable=True` it logs nothing. [RUN] On that
  path the streaming method also re-normalises the whole `contents` list after every chunk. [SRC models.py
  `generate_content_stream`; _extra_utils.py `append_chunk_contents`] Set the flag on the judge call too.

### 2.5 Tool choice, parallel calls, strict modes

- Tool choice = `tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=..., allowed_function_names=[...]))`,
  wire `"toolConfig": {"functionCallingConfig": {"mode": "AUTO"}}`. [RUN] Modes [DOC:function-calling][DOC:ref-gc]:
  - `AUTO` - default when only function declarations are present; model chooses text or a call. **Leave it
    unset.**
  - `ANY` - must call a function (constrained decoding, schema adherence); optional `allowed_function_names`.
  - `NONE` - no calls.
  - `VALIDATED` - text or call, with constrained decoding for calls; default only when built-in tools or
    structured output are combined with functions.
- **Parallel calls: there is no request flag** to enable or disable them (no such field in the SDK types or the
  REST reference). [SRC types.py][DOC:ref-gc] The model may return several `functionCall` parts in one turn
  ("parallel function calling", supported on all Gemini 4 models). [DOC:function-calling] The adapter must
  always handle N calls.
- **Strict schema**: there is no per-function `strict` flag. [SRC types.py] The nearest things are modes `ANY` /
  `VALIDATED` (constrained decoding). Avoid them here: "For `ANY` mode, very large or deeply nested schemas may
  be rejected", and with constrained decoding "Optional properties, `enum` values, and `anyOf` branches multiply
  the branching paths ... the API rejects the request with a `400 INVALID_ARGUMENT`". [DOC:function-calling]
  `AUTO` plus tolerant argument validation in the harness is the documented default.

---

## 3. Response shape

### 3.1 Object layout (SDK attribute -> wire name) [SRC types.py `GenerateContentResponse`, `Candidate`, `Content`, `Part`, `FunctionCall`][DOC:ref-overview][RUN]

```
GenerateContentResponse
  .candidates: Optional[list[Candidate]]            "candidates"   (None when the prompt was blocked)
     [0].content: Optional[Content]                 "content"      (None when blocked / some MAX_TOKENS chunks)
           .role: Optional[str]                     "role"         "model"
           .parts: Optional[list[Part]]             "parts"
               .text: Optional[str]                 "text"
               .thought: Optional[bool]             "thought"            True = thought SUMMARY text
               .thought_signature: Optional[bytes]  "thoughtSignature"   base64 string on the wire
               .function_call: Optional[FunctionCall] "functionCall"
                     .id: Optional[str]             "id"
                     .name: Optional[str]           "name"
                     .args: Optional[dict[str, Any]] "args"         ALREADY A JSON OBJECT, not a string
               .function_response, .inline_data, ...
     [0].finish_reason: Optional[FinishReason]      "finishReason"
     [0].finish_message: Optional[str]              "finishMessage"  (see gotcha 11.9: always None on 2.17.0)
     [0].safety_ratings: Optional[list[SafetyRating]] "safetyRatings"
     [0].index, .token_count, .citation_metadata, .grounding_metadata, ...
  .prompt_feedback: Optional[...PromptFeedback]     "promptFeedback"
        .block_reason: Optional[BlockedReason]      "blockReason"
        .safety_ratings
  .usage_metadata: Optional[...UsageMetadata]       "usageMetadata"
  .model_version: Optional[str]                     "modelVersion"
  .response_id: Optional[str]                       "responseId"
  .sdk_http_response.headers: dict                  (response headers, also on stream chunks)
```

- **Text**: concatenate `part.text` over parts where `part.text` is set and `part.thought` is not true. The
  convenience property `response.text` does exactly that for candidate 0, returns `None` when there is no
  non-thought text, and logs a warning when other part kinds are present. [SRC types.py
  `GenerateContentResponse._get_text`][RUN] Prefer iterating parts.
- **Thought summaries**: parts with `thought == True` (only when `include_thoughts=True`). [DOC:thinking]
- **Tool calls**: every part with `part.function_call is not None`; `response.function_calls` returns that list
  (or `None`). Do not assume position: "don't assume the `functionCall` will always be the last item in the
  parts array ... always iterate through the parts array". [DOC:function-calling]
  - `id`: "Gemini 3 now always returns a unique `id` with every `functionCall`. Include this exact `id` in your
    `functionResponse`". [DOC:function-calling] Doc sample value: `'8f2b1a3c'`. The schema still marks it
    optional, so tolerate `None`.
  - `args`: a `dict` (JSON object), or `None`/absent for a call without arguments - use `fc.args or {}`.
    [SRC][RUN]
- Detect "the model called a tool" from the parts, never from `finish_reason`. There is no tool-call finish
  reason: the one documented function-call response that shows its finish reason (non-streaming,
  `gemini-4-hydrogen`) ends with `"finishReason": "STOP"`, the same value as a plain text reply.
  [DOC:migrate, section "Function calling" > "Before (generateContent)", REST tab] The function-calling guide
  also says "Always check the
  `finishReason` ... to handle cases where the model failed to generate a valid function call"
  (`MALFORMED_FUNCTION_CALL`). [DOC:function-calling best practices]
- Because calls arrive whole (1.5), a turn that ends `MAX_TOKENS` can still contain complete, executable
  `functionCall` parts; a turn that ends `MAX_TOKENS` without any is truncated, not finished.

### 3.2 Finish reasons - full list for the Developer API [DOC:ref-overview "Candidate.finishReason"]

| value | meaning (doc wording, shortened) | adapter action |
|---|---|---|
| `STOP` | natural stop point or stop sequence | normal completion |
| `MAX_TOKENS` | `max_output_tokens` reached (thinking counts) | truncation; text may be empty |
| `CONTINUATION` | hit the per-request cap but generation is not complete; continue with `continuation_token`; `max_output_tokens` "is cumulative across continuation hops and must be monotonically increased or omitted" | only seen if `max_output_tokens` > hop cap and nothing continued; on 2.17.0 treat as truncation |
| `SAFETY` | candidate flagged for safety; blocked content is not returned | refusal/safety block |
| `RECITATION` | flagged for recitation of training data | block |
| `LANGUAGE` | unsupported language | block |
| `BLOCKLIST` | contains forbidden terms | block |
| `PROHIBITED_CONTENT` | potentially prohibited content | block |
| `SPII` | sensitive personally identifiable information | block |
| `OTHER` | unknown reason | block/unknown |
| `MALFORMED_FUNCTION_CALL` | the function call generated by the model is invalid | no usable call; retry the request |
| `UNEXPECTED_TOOL_CALL` | tool call generated but no tools enabled | error |
| `TOO_MANY_TOOL_CALLS` | too many consecutive tool calls, system exited | error |
| `MISSING_THOUGHT_SIGNATURE` | request has at least one thought signature missing | adapter bug: history was altered |
| `MALFORMED_RESPONSE` | finished due to malformed response | retry |
| `ESCALATION` | filtered by an escalation rule | block |
| `PUP_LIMITED_DISABLED` | account limited/disabled for Prohibited Use Policy violations | fatal |
| `IMAGE_SAFETY`, `IMAGE_PROHIBITED_CONTENT`, `IMAGE_OTHER`, `NO_IMAGE`, `IMAGE_RECITATION` | image generation only | n/a |
| `FINISH_REASON_UNSPECIFIED` | unused | - |
| absent / `None` | "the model has not stopped generating tokens" (mid-stream chunk) | at end of stream = incomplete, retry |

SDK 2.17.0's `types.FinishReason` has only: `FINISH_REASON_UNSPECIFIED, STOP, MAX_TOKENS, SAFETY, RECITATION,
LANGUAGE, OTHER, BLOCKLIST, PROHIBITED_CONTENT, SPII, MALFORMED_FUNCTION_CALL, IMAGE_SAFETY,
UNEXPECTED_TOOL_CALL, TOO_MANY_TOOL_CALLS, IMAGE_PROHIBITED_CONTENT, NO_IMAGE, IMAGE_RECITATION, IMAGE_OTHER`.
[RUN] Any other string still parses: `types.FinishReason("CONTINUATION")` yields a pseudo-member whose `.value`
and `.name` are the string, emits `UserWarning: CONTINUATION is not a valid FinishReason`, and compares equal to
the plain string. [RUN][SRC _common.py `CaseInSensitiveEnum._missing_`] **Compare with strings
(`fr == "CONTINUATION"` or `fr.value`), not with enum attributes that may not exist**, and do not run with
warnings-as-errors: with `warnings.simplefilter("error")` (or pytest `filterwarnings = error`) the same unknown
value is no longer a warning but a `pydantic.ValidationError` ("Input should be 'FINISH_REASON_UNSPECIFIED',
'STOP', ...") raised while the response chunk is parsed, and `ThinkingConfig(thinking_level="xhigh")` fails the
same way at construction. [RUN] The repo's pytest config has no such filter today.

### 3.3 Refusals and safety blocks

There is no "refusal" field or stop reason in the schema (`Candidate` and `Part` fields are listed in 3.1). [SRC]
Three shapes:

1. **Prompt blocked**: `response.candidates` is `None`, `response.prompt_feedback.block_reason` is set. Values:
   `BLOCK_REASON_UNSPECIFIED`, `SAFETY`, `OTHER`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `IMAGE_SAFETY`. "Returns no
   candidates at all only if there was something wrong with the prompt (check `prompt_feedback`)."
   [DOC:ref-overview][RUN] The SDK enum is `types.BlockedReason` and does not match the wire list exactly: its
   members are `BLOCKED_REASON_UNSPECIFIED` (note the extra "ED"), `SAFETY`, `OTHER`, `BLOCKLIST`,
   `PROHIBITED_CONTENT`, `IMAGE_SAFETY`, `MODEL_ARMOR`, `JAILBREAK`, so the wire value
   `BLOCK_REASON_UNSPECIFIED` parses as an unknown pseudo-member with a `UserWarning`. Compare block reasons as
   strings too, and treat any non-`None` value as "prompt blocked". [RUN][SRC types.py `BlockedReason`]
   When streaming, prompt feedback is "sent only in the first stream chunk". [SRC types.py
   `GenerateContentResponsePromptFeedback`] `response.text`, `.parts`, `.function_calls` are all `None`. [RUN]
   `BlockedReason.OTHER` = "the query or response may violate the terms of service or be otherwise unsupported".
   [DOC:troubleshooting]
2. **Candidate blocked**: a candidate with `finish_reason` in the "block" rows of 3.2, usually **without**
   `content` ("The content that was blocked is not returned"), and `safety_ratings` entries
   `{category, probability, blocked}`. [DOC:safety][RUN]
3. **The model declines in words**: ordinary text with `STOP`; indistinguishable from any other text reply.

Adjustable safety filters are **off by default** on Gemini 2.5 and later ("If the threshold is not set, the
default block threshold is Off"); built-in protections for core harms cannot be adjusted. [DOC:safety] The
adapter does not need `safety_settings`.

### 3.4 Usage [SRC types.py `GenerateContentResponseUsageMetadata`][DOC:ref-overview][DOC:tokens][DOC:thinking]

| SDK field | wire | meaning |
|---|---|---|
| `prompt_token_count` | `promptTokenCount` | input tokens (text, images, tools, history). Includes cached tokens. |
| `cached_content_token_count` | `cachedContentTokenCount` | "Number of tokens in the cached part of the prompt"; a subset of `prompt_token_count`. It is the only cache field, and the caching guide says implicit-cache hits are visible in `usage_metadata`. [DOC:caching] |
| `candidates_token_count` | `candidatesTokenCount` | "Total number of tokens across all the generated response candidates" - the non-thinking output (see below) |
| `thoughts_token_count` | `thoughtsTokenCount` | thinking tokens (the full thoughts, not the summaries) |
| `tool_use_prompt_token_count` | `toolUsePromptTokenCount` | built-in-tool results fed back as input; not used here |
| `total_token_count` | `totalTokenCount` | prompt + tool-use prompt + thoughts + candidates |
| `prompt_tokens_details`, `candidates_tokens_details`, `cache_tokens_details`, `tool_use_prompt_tokens_details` | `...TokensDetails` | lists of `{modality: TEXT|IMAGE|VIDEO|AUDIO|DOCUMENT, token_count}` |
| `traffic_type` | - | SDK field with no counterpart in the Developer API reference; expect `None` |
| (none on 2.17.0) | `serviceTier` | "Service tier of the request" (`standard`, `flex`, `priority`); listed in the REST reference, absent from the 2.17.0 type, so the SDK drops it silently [DOC:ref-overview][RUN] |

- **Thinking tokens are additional to `candidates_token_count`, not included in it.** Evidence: the doc sample
  `{"promptTokenCount": 142, "candidatesTokenCount": 0, "thoughtsTokenCount": 32768, "totalTokenCount": 32910}`
  (142 + 0 + 32768) [DOC:long-decode]; reference: total = "prompt + tool-use prompt + thoughts + response
  candidates" [DOC:ref-overview]; "response pricing is the sum of output tokens and thinking tokens".
  [DOC:thinking] So billable output = `candidates_token_count + thoughts_token_count`; billable input =
  `prompt_token_count`, of which `cached_content_token_count` is charged at the cached rate.
- Any field can be `None` (absent on the wire) - coalesce to 0. Mid-stream chunks often carry partial usage.
  [RUN]
- With SDK >= 2.30.1 and automatic continuation, usage is **summed over hops**, and every later hop re-sends
  the prompt plus everything generated so far as input, so `prompt_token_count`/`total_token_count` overstate
  context size (doc example: a 1,000-token prompt with 200,000 output tokens reports ~397,000 prompt tokens).
  Use it as-is for cost; use `usage_metadata.initial_prompt_token_count` (SDK-side property, not in
  `model_dump()`) for prompt size. [DOC:long-decode][SRC-NEW types.py]

---

## 4. The tool-result turn

### 4.1 What to send [DOC:function-calling step 4][DOC:thought-signatures][RUN]

After a reply that contains function calls, the next request's `contents` is the previous `contents` plus two
entries:

```python
calls = [p.function_call for p in result["content"].parts if p.function_call is not None]

contents.append(result["content"])                       # 1. the model turn, EXACTLY as received
contents.append(types.Content(role="user", parts=[       # 2. ONE user turn holding every function response
    types.Part(function_response=types.FunctionResponse(
        id=fc.id,                                        #    echo the call id
        name=fc.name,                                    #    required
        response={"output": "<tool result text>"},       #    required; must be a JSON object (dict)
    ))
    for fc in calls                                      #    same order as the calls
]))
```

Wire form of those two entries (the tail of `contents`) as produced by the installed SDK [RUN]:

```json
[
 {"parts": [
    {"text": "**Planning the canvas**\n\nI'll lay out a 16x16 grid: sky first, then the village.", "thought": true},
    {"functionCall": {"id": "fc_8f2b1a3c", "args": {"palette": ["#0b1d51", "#f5d547"], "start_row": 0,
                      "rows": ["0000111100001111", "0001111110011111"]}, "name": "write_rows"},
     "thoughtSignature": "b3BhcXVlLXNpZ25hdHVyZS1B"}
  ], "role": "model"},
 {"parts": [
    {"functionResponse": {"id": "fc_8f2b1a3c", "name": "write_rows",
                          "response": {"output": "ok: rows 0-1 written (2 of 16)"}}}
  ], "role": "user"}
]
```

- **`Part.from_function_response` has no `id` parameter** in 2.17.0, 2.34.0 or 3.6.1 (signature
  `(*, name, response, parts=None)`), although the function-calling guide calls it with `id=tool_call.id`; that
  sample raises `TypeError: Part.from_function_response() got an unexpected keyword argument 'id'`. Build the
  part as shown above. [RUN][SRC-NEW][DOC:function-calling]
- `FunctionResponse` fields: `name` (required), `response: dict` (required), `id`, `parts` (multimodal, unused),
  `will_continue`/`scheduling` (Live API only). [SRC types.py `FunctionResponse`]
- `response` payload: "Use "output" key to specify function output and "error" key to specify error details (if
  any). If "output" and "error" keys are not specified, then whole "response" is treated as function output."
  [SRC types.py `FunctionResponse.response`] The REST reference says any keys are fine ("output", "result", ...)
  and that a failed call "can have an "error" key". [DOC:ref-gc] Use `{"output": "<text>"}` for success and
  `{"error": "<text>"}` for a rejected/invalid call.
- Role of the function-response turn is `"user"` (`Content.role` "Must be either 'user' or 'model'").
  [SRC types.py `Content.role`][DOC:function-calling]
- The full `tools`, `system_instruction` and config must be re-sent on every request (the API is stateless).
  [DOC:thinking "The Gemini API is stateless"]

### 4.2 What must be replayed verbatim for reasoning to stay valid [DOC:thought-signatures][DOC:thinking][DOC:function-calling]

- Thought signatures are "encrypted representations of the model's internal thought process". They arrive as
  `thoughtSignature` on ordinary parts (function-call parts, text parts), never as separate items.
- **Mandatory**: "When using Gemini 4 and Gemini 3 models, you must pass back thought signatures during function
  calling, otherwise you will get a validation error (4xx status code)." Validation covers the **current turn**:
  the API walks back to the most recent user message with standard content (text, not a `functionResponse`);
  every model function-call step after it is in the turn; "The first `functionCall` part in each step of the
  current turn must include its `thought_signature`"; omit it and "the request will fail with a 400 error", with
  a message of the form `Function call <name> in the <index>. content block is missing a thought_signature`.
  Earlier turns are not validated. The enum value `MISSING_THOUGHT_SIGNATURE` also exists as a finish reason.
  [DOC:ref-overview]
- **Recommended, not validated**: signatures on non-function-call parts (e.g. the last text part of a text-only
  reply). "You won't receive a blocking error if you omit them, though performance may degrade."
- Where signatures sit: single call -> on that `functionCall` part. Parallel calls -> **only on the first**
  `functionCall` part; later ones have none ("Do not copy a thought signature across parallel `functionCall`
  parts" [DOC:neon]). Sequential steps -> one per step, and all of them must be present in later requests of the
  same turn. Text-only reply -> on the last part, possibly a part whose text is `""`.
- Part hygiene: return the signature "in the exact part where it was received"; "Don't merge a `Part`
  containing a signature with one that does not"; "Don't combine two `Parts` that both contain signatures".
- "Return the entire response with all parts back to the model in subsequent turns" - that includes
  `thought: true` summary parts when `include_thoughts` is on. [DOC:thinking] The SDK serialises them back as
  `{"text": ..., "thought": true}`. [RUN]
- The SDK does nothing clever here; "handled automatically" in the docs means "append
  `response.candidates[0].content` unchanged". There is no signature logic in the SDK outside the type
  definitions. [SRC][SRC-NEW]
- Signature bytes: the SDK base64-decodes `thoughtSignature` into `bytes` and re-encodes with **URL-safe**
  base64 when sending, so `"...//g=="` comes back as `"...__g=="`. Same bytes; this is the official SDK's
  behaviour. [RUN][SRC _common.py `encode_unserializable_types`]
- Signatures survive model switches within Gemini 4 ("return prior model turns and thought signatures unchanged
  when switching models across turns"). [DOC:choosing]
- They "increase the input tokens billed when sent back". [DOC:thinking]

### 4.3 Ordering rules and several calls in one reply

- Roles alternate `user` / `model`. [DOC:text]
- For a reply with calls FC1 (+signature), FC2: the model turn holds both `functionCall` parts; the following
  user turn holds FR1, FR2. "If you have them interleaved as FC1 + signature, FR1, FC2, FR2 the API will return a
  400 error." [DOC:thought-signatures FAQ 2]
- The docs contradict themselves on response order: "you don't need to return the `function_result` objects in
  the same order ... The Gemini API maps each result back to its corresponding call using the `id`" and, further
  down the same page, "To send the results back, include the responses in the same order as they were
  requested." [DOC:function-calling] Do both: same order **and** matching `id`.
- One `functionResponse` per `functionCall` ("User should provide a FunctionResponse for each function call in
  the next turn" [SRC types.py `Tool.function_declarations`]), including calls the harness rejected (answer
  those with `{"error": ...}`).
- Keep the function-response user turn free of text parts: the signature validator finds the start of the
  current turn by looking for the latest user message with "standard content (e.g., `text`)".
  [DOC:thought-signatures]
- Verified wire shape for two calls (the tail of `contents`) [RUN]:

```json
[
 {"parts": [
    {"functionCall": {"id": "fc_1", "args": {"start_row": 0, "rows": ["0000"]}, "name": "write_rows"},
     "thoughtSignature": "b3BhcXVlLXNpZ25hdHVyZS1D"},
    {"functionCall": {"id": "fc_2", "args": {"start_row": 1, "rows": ["1111"]}, "name": "write_rows"}}
  ], "role": "model"},
 {"parts": [
    {"functionResponse": {"id": "fc_1", "name": "write_rows", "response": {"output": "ok: row 0 written"}}},
    {"functionResponse": {"id": "fc_2", "name": "write_rows", "response": {"output": "ok: row 1 written"}}}
  ], "role": "user"}
]
```

### 4.4 Plain follow-up ("nudge") after an assistant text reply

```python
contents.append(result["content"])     # keeps the text part(s) and their (optional) thought signature
contents.append(types.Content(role="user", parts=[types.Part(text="<nudge text>")]))
```

Wire (the tail of `contents`) [RUN]:

```json
[
 {"parts": [{"text": "The painting is complete: 2 rows written."},
            {"text": "", "thoughtSignature": "b3BhcXVlLXNpZ25hdHVyZS1C"}], "role": "model"},
 {"parts": [{"text": "The canvas still has 14 empty rows. Please continue with write_rows."}], "role": "user"}
]
```

If `result["content"]` is `None` (no parts at all: blocked, `MALFORMED_FUNCTION_CALL`, some `MAX_TOKENS`), do
not append an empty model turn and then a nudge. The SDK's own chat helper treats a model turn with no parts as
invalid and removes it *and the user input before it* from the history it sends next. [SRC chats.py
`_validate_content`, `_extract_curated_history`] Re-issue the same request instead (see UNVERIFIED 7-8).

### 4.5 Persisting a transcript

`content.model_dump(mode="json", exclude_none=True)` gives a JSON-safe dict (snake_case keys, signature and
image bytes as URL-safe base64); `types.Content.model_validate(d)` restores it with identical signature bytes.
Plain `model_dump()` leaves raw `bytes` in the dict. [RUN]

---

## 5. Reasoning ("thinking") settings

`config.thinking_config = types.ThinkingConfig(thinking_level=..., include_thoughts=..., thinking_budget=...)`
[SRC types.py `ThinkingConfig`]; wire `generationConfig.thinkingConfig.{thinkingLevel, includeThoughts, thinkingBudget}`. [DOC:ref-gc]

### 5.1 `thinking_level` per model [DOC:thinking table][DOC:long-decode "Set max_output_tokens"]

(The model-card pages do not list thinking levels; the table below is the one on the thinking page, and the
long-decode page repeats it: "Carbon and Argon default to `medium` and support up to `high`. Neon and Hydrogen
default to `high` and support up to `xhigh`.")

| model | default | allowed | **highest** |
|---|---|---|---|
| `gemini-4-carbon` | `medium` | `low`, `medium`, `high` | **`high`** |
| `gemini-4-argon` | `medium` | `low`, `medium`, `high` | **`high`** |
| `gemini-4-neon` | `high` | `low`, `medium`, `high`, `xhigh` | **`xhigh`** |
| `gemini-4-hydrogen` | `high` | `low`, `medium`, `high`, `xhigh` | **`xhigh`** |
| `gemini-3.8-flash` | `medium` | `low`, `medium`, `high` | `high` |
| `gemini-3.6-flash` | `medium` | `minimal`, `low`, `medium`, `high` | `high` |
| `gemini-3.5-flash-lite` | `minimal` | `minimal`, `low`, `medium`, `high` | `high` |
| `gemini-3.1-pro-preview` | `high` | `low`, `medium`, `high` | `high` |

- REST enum: `THINKING_LEVEL_UNSPECIFIED, MINIMAL, LOW, MEDIUM, HIGH, XHIGH`. [DOC:ref-gc] REST samples in the
  guides send lower case (`"thinkingLevel": "medium"`, `"xhigh"`). [DOC:carbon][DOC:thinking]
- SDK 2.17.0 enum `types.ThinkingLevel`: `THINKING_LEVEL_UNSPECIFIED, MINIMAL, LOW, MEDIUM, HIGH` - **no
  `XHIGH`**. `thinking_level="high"`, `"HIGH"` and `types.ThinkingLevel.HIGH` all serialise as `"HIGH"`.
  `thinking_level="xhigh"` is accepted, emits `UserWarning: xhigh is not a valid ThinkingLevel`, and is sent as
  the literal `"xhigh"` (the spelling the REST samples use). `types.ThinkingLevel.XHIGH` is an `AttributeError`
  on 2.17.0 and 2.34.0 - always pass the **string**. [RUN][SRC-NEW] An unknown string is passed through in the
  case you give it (`"XHIGH"` goes out as `"XHIGH"`, the REST enum spelling; `"xhigh"` as `"xhigh"`, the REST
  sample spelling) - either is documented, pick one and keep it. Under warnings-as-errors the same call raises
  `pydantic.ValidationError` instead of warning (3.2). [RUN]
- "You cannot disable thinking for Gemini 4 Carbon, Gemini 4 Argon, Gemini 4 Neon, or Gemini 4 Hydrogen."
  `minimal` is "Not supported on Gemini 4 models or `gemini-3.8-flash`"; on Gemini 4 it is rejected with an
  error. [DOC:thinking][DOC:long-decode][DOC:neon]
- `thinkingLevel` is "required for Gemini 4 and recommended for Gemini 3"; if omitted the model default applies.
  [DOC:thinking] Send it explicitly, mapped per model from the table (do not send `xhigh` to carbon/argon; the
  result of an unsupported level is not documented).
- `xhigh` "increases latency and token use and can cause overthinking or exhaust `max_output_tokens`".
  [DOC:thinking]

### 5.2 `thinking_budget`

Gemini 2.5-era integer budget (`0` = off, `-1` = dynamic). On Gemini 4 it is deprecated: "If passed alone, it
maps to the nearest `thinkingLevel` and returns a warning; sending `thinkingBudget` and `thinkingLevel` together
returns a `400` error." Gemini 2.5 models do not accept `thinkingLevel`. [DOC:thinking] Never send it.

### 5.3 `include_thoughts`

`True` -> the response contains thought **summary** parts (`thought: true`): one final summary when not
streaming, "rolling, incremental summaries" when streaming. Levels and billing apply to the full raw thoughts,
not the summaries. Summaries are opt-in ("You can enable thought summaries by setting `includeThoughts` to
`true`"). [DOC:thinking] The thought-signature rules in 4.2 are stated without any dependence on this flag. If
summaries are requested they become part of the model turn and are replayed with it (4.2).

### 5.4 Interaction with `max_output_tokens`

- "`max_output_tokens` ... sets the maximum number of tokens a response can generate, including thought tokens
  ... acts as a hard cutoff ... without changing how the model allocates its thinking budget. If the model hits
  this limit while reasoning, it stops generating with `finish_reason: MAX_TOKENS` and returns truncated or
  empty output (while still billing for any thinking tokens generated)." [DOC:thinking]
- "To use fewer tokens, lower `thinking_level`"; lowering `max_output_tokens` "doesn't make the model think more
  concisely, and instead cuts the response off sooner". [DOC:long-decode]
- "Empty text with `MAX_TOKENS`: the model used all tokens on thinking. Raise `max_output_tokens` or lower
  `thinking_level`." [DOC:long-decode troubleshooting]

### 5.5 Parameters that are rejected or ignored with current models

| parameter | behaviour on Gemini 4 / 3.6+ | source |
|---|---|---|
| `temperature`, `top_p`, `top_k` | "no longer supported and is silently ignored on Gemini 3.6 Flash and newer"; in-range values ignored with "deprecation warning headers (and will return `400` errors in a future release)"; "Out-of-range values return a `400` validation error" | [DOC:ref-gc][DOC:text][DOC:neon] |
| `candidate_count` | "unsupported on Gemini 3 and 4"; remove | [DOC:troubleshooting][DOC:carbon] |
| `thinking_budget` + `thinking_level` together | 400 | [DOC:thinking] |
| `thinking_level: "minimal"` | error on Gemini 4 | [DOC:long-decode] |
| `cached_content` (explicit caching) | "not supported on Gemini 4 ... returns an error" | [DOC:carbon] |

The harness never sends temperature - correct. Implicit caching needs nothing (on by default for prompts of at
least 4,096 tokens on Gemini 4; cached input is discounted 95%). [DOC:caching][DOC:carbon]

---

## 6. Output-token limits, context windows, and "set the limit high"

| model | context | max output | source |
|---|---|---|---|
| `gemini-4-carbon`, `-argon`, `-neon`, `-hydrogen` | 1,048,576 tokens, **unified** (input + thinking + output) | 65,536 per request; up to 1,048,576 minus input with long decode | [DOC:model-cards][DOC:tokens] |
| `gemini-3.8-flash`, `gemini-3.5-flash-lite` | 1,048,576 input | 65,536 output (separate budget) | [DOC:model-cards][DOC:choosing] |

- If input + `max_output_tokens` exceeds 1,048,576 the API "automatically caps the output limit to the remaining
  space". [DOC:thinking]
- Default when `max_output_tokens` is unset on Gemini 4: 64K for streaming and SDK calls (32K for direct
  non-streaming REST on carbon/argon). [DOC:long-decode][DOC:carbon]
- Google's guidance: "Run your evaluations ... with `max_output_tokens` set to at least 65,536" [DOC:choosing];
  "Long coding and agentic tasks: Start at 65,536"; "Hard math, science, research, and proofs: test values from
  262,144 up to 1,048,576 minus your input tokens"; "Always set `max_output_tokens` greater than 65,536 for long
  decode"; long decode is listed as *not* useful for "Agents that call tools in many short turns".
  [DOC:long-decode]

What to set:

- **Staying on SDK 2.17.0: `max_output_tokens=65_536`**, the largest value one hop can serve. Larger values ask
  for continuation hops that 2.17.0 cannot perform (the turn would come back `CONTINUATION` with possibly empty
  content). Accept that a turn whose thinking alone exceeds 64K ends as `MAX_TOKENS`; the harness must treat
  `MAX_TOKENS` with no function call as a truncated turn, not as "the model is done".
- **With SDK >= 2.30.1: `max_output_tokens=262_144`** (the value used in Google's long-decode examples) is a
  reasonable "high"; the SDK then continues hops automatically and the assembler in 1.5 needs no change for
  clean hops (but see the reconnect caveat in 1.4: after a dropped stream the newer SDK can hand the assembler a
  duplicated tail).
  Worst-case output spend per call at carbon's price from 2026-12-01 ($20/1M): about $5.24, plus re-sent input
  on each hop (mostly at the cached rate).
- Judge call: no tools, plain text, but thinking still counts. Use at least 32,768 ("Existing apps with a limit
  lower than 32K: Raise `max_output_tokens` to 32,768" [DOC:carbon]); 65,536 is fine.

---

## 7. Vision input for the judge

### 7.1 Request

```python
parts = []
for label, png_bytes in items:                                   # up to 9
    parts.append(types.Part(text=label))
    parts.append(types.Part.from_bytes(data=png_bytes, mime_type="image/png"))   # RAW bytes, not base64
parts.append(types.Part(text="<question>"))

judge_config = types.GenerateContentConfig(
    system_instruction="<judge system prompt>",                  # optional
    max_output_tokens=65_536,
    thinking_config=types.ThinkingConfig(thinking_level="high"),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)
result = call(client, "gemini-4-carbon", [types.Content(role="user", parts=parts)], judge_config)
content = result["content"]                      # None when blocked / no parts: check before reading text
text = "".join(p.text for p in (content.parts if content else []) if p.text and not p.thought)
# Accept the answer only when result["finish_reason"] == "STOP"; "MAX_TOKENS" with text == "" means the
# judge spent its whole budget thinking (section 5.4).
```

Wire for that call with two images (image data shortened here; the SDK sends the full base64) [RUN]:

```json
{"contents": [{"parts": [
    {"text": "Image 1: candidate 1"},
    {"inlineData": {"data": "iVBORw0KGgoAAAANSUhEUgAA...", "mimeType": "image/png"}},
    {"text": "Image 2: candidate 2"},
    {"inlineData": {"data": "iVBORw0KGgoAAAANSUhEUgAA...", "mimeType": "image/png"}},
    {"text": "Which is closer to the reference? Answer in plain text."}
  ], "role": "user"}],
 "systemInstruction": {"parts": [{"text": "You are a strict judge."}], "role": "user"},
 "generationConfig": {"maxOutputTokens": 65536, "thinkingConfig": {"thinking_level": "HIGH"}}}
```

- `types.Part.from_bytes(*, data: bytes, mime_type: str, media_resolution=None)` builds
  `Part(inline_data=Blob(data=..., mime_type=...))`. [SRC types.py `Part.from_bytes`, `Blob`] **Pass the raw PNG
  bytes**; the SDK base64-encodes them into `inlineData.data` using the URL-safe alphabet (`-`/`_`, never
  `+`/`/`). [RUN] Encoding pitfalls, all run offline [RUN]: `data=base64.b64encode(png)` (base64 *bytes*) is
  encoded a second time and the model receives garbage; `data="<base64 str>"` (a `str`) is base64-decoded by
  pydantic and happens to work. If the harness holds images as base64 strings, decode them first
  (`base64.b64decode(s)`) and pass bytes.
- Parts are sent in list order inside one `Content`; text and image parts interleave freely. [RUN]
  "You can provide multiple images in a single prompt by including multiple image `Part` objects". [DOC:image]
- A bare list `contents=["text a", part, "text b"]` is also accepted and becomes one user `Content` [RUN]; the
  explicit `Content` is clearer.
- Dict form also works: `{"role": "user", "parts": [{"text": ...}, {"inline_data": {"mime_type": "image/png",
  "data": <raw bytes>}}]}`. [RUN][SRC _transformers.py `t_contents`]
- Doc tip (single image): "place the text prompt *after* the image part". [DOC:image]

### 7.2 Limits

- Formats: `image/png`, `image/jpeg`, `image/webp`, `image/heic`, `image/heif`. [DOC:image]
- "Gemini models support a maximum of 3,600 image files per request." [DOC:image] Nine is nowhere near it.
- Inline size: the image guide says inline data "limits your total request size (text prompts, system
  instructions, and inline bytes) to 20MB" [DOC:image]; the file-input guide says inline data is "100 MB per
  request/payload (50 MB for PDFs)". [DOC:file-input] The two pages disagree - **stay under 20 MB total** (base64
  inflates bytes by 4/3).
- No per-image pixel maximum is stated on the pages fetched; large images are scaled/tiled to the token budget
  below.

### 7.3 Token cost per image and `media_resolution`

- Gemini 4 / Gemini 3: `media_resolution` sets "the **maximum number of tokens** allocated per input image".
  Image tokens by level: `MEDIA_RESOLUTION_UNSPECIFIED` (default) **1120**, `MEDIA_RESOLUTION_LOW` 280,
  `MEDIA_RESOLUTION_MEDIUM` 560, `MEDIA_RESOLUTION_HIGH` 1120, `MEDIA_RESOLUTION_ULTRA_HIGH` 2240 (per-part
  only). "`MEDIA_RESOLUTION_HIGH` provides the optimal performance for most use cases"; recommended for images.
  [DOC:media-resolution] Nine images at the default are at most about 10,080 input tokens.
- The image and token guides still carry the older formula: "258 tokens if both dimensions <= 384 pixels.
  Larger images are tiled into 768x768 pixel tiles, each costing 258 tokens." [DOC:image][DOC:tokens] For
  Gemini 4 take the media-resolution table as the cap and read the real number from
  `usage_metadata.prompt_tokens_details` (modality `IMAGE`).
- Images are billed as input tokens at the model's single input price (Gemini 4 prices are not split by
  modality). [DOC:pricing]
- Setting it: globally `GenerateContentConfig(media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH)` ->
  `generationConfig.mediaResolution` [RUN]; per part
  `types.Part.from_bytes(..., media_resolution="MEDIA_RESOLUTION_HIGH")` -> `{"mediaResolution": {"level":
  "MEDIA_RESOLUTION_HIGH"}, "inlineData": {...}}` [RUN] ("Per-part media resolution is an experimental
  feature"; per-part overrides global). [DOC:media-resolution] The default already equals HIGH for images on
  Gemini 4, so the judge can leave it unset.
- Global enum `types.MediaResolution`: `MEDIA_RESOLUTION_UNSPECIFIED, _LOW, _MEDIUM, _HIGH`; per-part enum
  `types.PartMediaResolutionLevel` adds `MEDIA_RESOLUTION_ULTRA_HIGH`. [SRC types.py]

---

## 8. Current model ids and list prices

Model ids from [DOC:models] and [DOC:model-cards]; prices (USD per 1M tokens, paid tier, Standard) from
[DOC:pricing] (`https://aistudio.google.com/docs/pricing.md`, fetched 2026-11-19), cross-checked with
[DOC:choosing]. Output prices **include thinking tokens**.

| tier | API id | input | output | cached input | released | notes |
|---|---|---|---|---|---|---|
| flagship ("our strongest reasoning model") | `gemini-4-carbon` | $2.00 through 2026-11-30, **$4.00 from 2026-12-01** | $10.00 through 2026-11-30, **$20.00 from 2026-12-01** | $0.10 -> $0.20 | 2026-11-10 | paid tier only |
| mid ("sits between Hydrogen and Carbon") | `gemini-4-neon` | $3.00 | $15.00 | $0.15 | 2026-11-18 | paid tier only |
| small / default ("start here") | `gemini-4-hydrogen` | $1.50 | $7.50 | $0.075 | 2026-11-12 | free tier exists |
| cheapest general model | `gemini-3.5-flash-lite` | $0.30 | $2.50 | $0.03 | - | 1M input / 64K output |
| efficient Gemini 4 (earlier release) | `gemini-4-argon` | same as carbon ($2.00 -> $4.00) | same as carbon ($10.00 -> $20.00) | $0.10 -> $0.20 | 2026-10-13 | paid tier only |
| previous-generation Flash | `gemini-3.8-flash` | $0.75 through 2026-12-31, $1.50 from 2027-01-01 | $3.75 -> $7.50 | $0.075 -> $0.15 | - | |

- Batch and Flex tiers are half price; Priority for carbon is $3.60 / $18.00 (doubling on 2026-12-01).
  [DOC:pricing]
- Gemini 4 has no >200k-token price step on the pricing page (the older `gemini-3.1-pro-preview` did:
  $2/$12 up to 200k, $4/$18 above). [DOC:pricing]
- All ids above are stable ids with "No shutdown date announced" for the Gemini 4 family. [DOC:deprecations]
  `gemini-2.5-pro` / `gemini-2.5-flash` are deprecated and will be rerouted on 2026-12-01;
  `gemini-3-pro-preview` and `gemini-2.0-flash` are shut down. [DOC:models]
- Judge candidates ("strong vision model"): all four Gemini 4 models accept "Text, Image, Video, Audio, and
  PDF" and output text. [DOC:model-cards]
- Spend guard that can bite a benchmark: paid Tier 1 projects have a spend-based rate limit of **$10 per rolling
  10 minutes** (Tier 2/3: $200), returned as `429 RESOURCE_EXHAUSTED`. [DOC:rate-limits] RPM/TPM numbers for
  Gemini 4 are not published on the page ("can be viewed in Google AI Studio").

---

## 9. Errors and retries

### 9.1 Exception classes [SRC errors.py][RUN]

- `errors.APIError(Exception)` - base. Attributes: `.code: int` (HTTP status), `.status: str` (e.g.
  `"RESOURCE_EXHAUSTED"`), `.message: str`, `.details` (the whole JSON error body as a dict), `.response` (the
  `httpx.Response`; `e.response.headers` works). `str(e)` = `"429 RESOURCE_EXHAUSTED. {...}"`.
- `errors.ClientError(APIError)` - any 4xx. `errors.ServerError(APIError)` - any 5xx.
- Not wrapped: transport errors propagate as raw httpx exceptions (`httpx.ReadTimeout`, `httpx.ConnectError`,
  `httpx.RemoteProtocolError`, ...), which are **not** `APIError`. [RUN]
- `errors.UnknownApiResponseError(ValueError)` - a **stream** event that is not JSON; it is raised from inside
  the iteration. This includes anything in the SSE body that is not a `data: <json>` line: a `[DONE]` sentinel,
  an `event:`/`id:` line, an SSE comment line such as `: keepalive`, or `data:` without the following space.
  [RUN][SRC _api_client.py `HttpResponse._iter_response_stream`] On the **non-streaming** path a 200 response
  whose body is not JSON raises a raw `json.JSONDecodeError` instead (also a `ValueError`). [RUN]
- An HTTP error whose body is not JSON (e.g. an HTML 502 from a proxy) still becomes `ServerError`/`ClientError`:
  `.code` is the HTTP status, `.status` is the HTTP reason phrase (`"Bad Gateway"`), `.details` is
  `{"message": <body text>, "status": <reason phrase>}`. [RUN]
- `pydantic.ValidationError` - a response that parses as JSON but violates a type (e.g. a `thoughtSignature`
  that is not base64). [RUN]
- Mid-stream failures: an SSE chunk shaped `{"error": {...}}` raises `ClientError`/`ServerError` from inside the
  iteration, after earlier chunks were already yielded. [RUN][SRC _api_client.py `request_streamed`]

Error body: `{"error": {"code": <int HTTP status>, "message": <str>, "status": <str gRPC status>, "details":
[<objects such as ErrorInfo / LocalizedMessage>]}}` [DOC:errors]. Minimal fixture (message text illustrative):

```json
{"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota).",
           "status": "RESOURCE_EXHAUSTED", "details": []}}
```

### 9.2 Status codes [DOC:errors][DOC:troubleshooting]

| HTTP | status | meaning | retry? |
|---|---|---|---|
| 400 | `INVALID_ARGUMENT` | malformed request. Other pages name these 400 causes: missing thought signature [DOC:thought-signatures], `thinking_budget` sent together with `thinking_level` [DOC:thinking], tool schema too complex for constrained decoding [DOC:function-calling], out-of-range sampling values [DOC:neon], continuation token without a raised `max_output_tokens` [DOC:long-decode] | no (fix the request) |
| 400 | `INVALID_ARGUMENT` | **invalid API key** - the error-format sample on the errors page is `{"code": 400, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT", "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID", ...}]}`. So a key the API does not recognise can come back as 400, while the table's own 403 row covers "a wrong, blocked, or leaked API key"; treat both as fatal configuration errors | no |
| 400 | `FAILED_PRECONDITION` | free tier not available in region | no |
| 402 | `RESOURCE_EXHAUSTED` | prepay credits depleted ("Don't retry this request") | **no** |
| 403 | `PERMISSION_DENIED` | bad/blocked key, project restricted | no |
| 404 | `NOT_FOUND` | "The requested resource wasn't found" (doc example: a referenced media file) | no |
| 429 | `RESOURCE_EXHAUSTED` | RPM / TPM / RPD / spend limit | yes, backoff |
| 499 | `CANCELLED` | client closed the connection | caller's choice |
| 500 | `INTERNAL` | server error ("Your input context is too long" is given as an example) | yes |
| 503 | `UNAVAILABLE` | overloaded / out of capacity | yes |
| 504 | `DEADLINE_EXCEEDED` | not finished within the deadline ("Set a larger timeout") | yes |

`RESOURCE_EXHAUSTED` is both 402 (fatal) and 429 (retryable): **branch on `e.code`, not `e.status`.**
Doc guidance: "Only retry on transient errors (like `429`, `408`, or `5xx`). Do not retry on client errors (like
`400`, `402`, or `403`)"; exponential backoff with jitter. [DOC:troubleshooting]

### 9.3 SDK built-in retry [SRC _api_client.py `retry_args`, `_RETRY_*`; types.py `HttpRetryOptions`][RUN]

- **Off unless configured.** `HttpOptions.retry_options=None` -> one attempt. Verified: a 503 is raised after
  exactly one HTTP call. The troubleshooting page claims the Python SDK "automatically retries transient errors
  up to four times" by default; the source of 2.17.0, 2.34.0 and 3.6.1 says otherwise (`if options is None:
  stop_after_attempt(1)`). [RUN][SRC-NEW][DOC:troubleshooting]
- Enable: `HttpOptions(retry_options=types.HttpRetryOptions(attempts=5, initial_delay=1.0, max_delay=60.0,
  exp_base=2, jitter=1, http_status_codes=[408, 429, 500, 502, 503, 504]))`. Those are also the values used for
  any field left `None`. `attempts` includes the first call; 0 or 1 = no retry. Per-call override via
  `GenerateContentConfig(http_options=types.HttpOptions(retry_options=...))`.
- Retried: `APIError` with a listed code, `httpx.TimeoutException`, `httpx.ConnectError`. Not retried: other
  4xx, `httpx.RemoteProtocolError`, anything raised **after a stream has started** (mid-stream error chunk,
  dropped stream). [RUN] With a 20-minute timeout an SDK-level timeout retry repeats the whole 20 minutes.
- Recommendation: leave SDK retries off (`attempts=1`) and retry in the harness around `call()`, where partial
  streams and `finish_reason is None` can be handled too. Retry on: `APIError` with `code in {408, 429, 500,
  502, 503, 504}`, `httpx.TransportError`, and "stream ended without a finish reason". A retry re-sends the
  identical request; nothing is resumable on 2.17.0.

### 9.4 Rate-limit headers

None are documented on the rate-limit, API-error or troubleshooting pages (no `Retry-After`, no
`x-ratelimit-*`). [DOC:rate-limits][DOC:errors][DOC:troubleshooting] Limits are per project across RPM, input
TPM, RPD and spend; RPD resets at midnight Pacific. Response headers are reachable as
`response.sdk_http_response.headers` (also on every stream chunk) and `e.response.headers`. [RUN]

---

## 10. Worked example (usable as mocked-response fixtures)

All requests below are the bodies the installed SDK produced, and all responses below were fed through the
installed SDK and parsed, via `httpx.MockTransport`. [RUN] Signatures are valid base64 of readable placeholders
(`b3BhcXVlLXNpZ25hdHVyZS1B` = `opaque-signature-A`); real ones are opaque values issued by the API. **A fixture
signature must be valid base64** or the SDK raises `ValidationError` (the docs' `"<Signature A>"` placeholder
does exactly that). All token counts, response ids and message texts in the fixtures are invented.

### 10.1 How to fake the stream

- URL the SDK calls: `POST {base}/v1beta/models/gemini-4-carbon:streamGenerateContent?alt=sse`.
- Body to return: `Content-Type: text/event-stream`, one event per chunk, each exactly
  `data: <GenerateContentResponse JSON>` followed by a blank line (`\n\n` or `\r\n\r\n`). No `event:` names, no
  ids, and **no `[DONE]` sentinel** (the SDK would raise `UnknownApiResponseError`). An empty body yields zero
  chunks without error. [RUN][SRC _api_client.py `HttpResponse._iter_response_stream`]

```python
def handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)                       # assert on it here
    sse = "".join("data: " + json.dumps(c) + "\n\n" for c in NEXT_CHUNKS)
    return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})
client = make_client("test-key", httpx_client=httpx.Client(transport=httpx.MockTransport(handler)))
```

- Non-streaming fake: `httpx.Response(200, json=<one GenerateContentResponse>)`; errors:
  `httpx.Response(429, json={"error": {...}})`. [RUN]
- Building chunk objects without HTTP: `types.GenerateContentResponse.model_validate(<wire dict>)` accepts the
  camelCase wire names and snake_case names, but **rejects unknown fields** (e.g. `continuationToken` on
  2.17.0), whereas the real HTTP path strips unknown fields silently. [RUN] Prefer the transport fake.
- Each chunk is a complete `GenerateContentResponse` holding only the **new** parts (deltas), and one logical
  text can be split over several chunks (doc streaming sample: `"The image displays"` then `" the following
  materials:..."`, same `responseId`). [DOC:text] `finishReason` is absent while "the model has not stopped
  generating tokens" [DOC:ref-overview]; `usageMetadata` is cumulative within the stream. [DOC:long-decode]
- "The final assembled message" helper: none in the SDK - use `call()` from 1.5.

### 10.2 Request 1 (system + user + tool)

Exactly the JSON in section 2.2.

### 10.3 Response 1 (reasoning summary + one tool call), as stream chunks

```json
[
 {"candidates": [{"content": {"role": "model", "parts": [
      {"text": "**Planning the canvas**\n\nI'll lay out a 16x16 grid: sky first, then the village.", "thought": true}
    ]}, "index": 0}],
  "usageMetadata": {"promptTokenCount": 412, "thoughtsTokenCount": 1310, "totalTokenCount": 1722},
  "modelVersion": "gemini-4-carbon", "responseId": "r1AbCd"},

 {"candidates": [{"content": {"role": "model", "parts": [
      {"functionCall": {"name": "write_rows",
                        "args": {"palette": ["#0b1d51", "#f5d547"], "start_row": 0,
                                 "rows": ["0000111100001111", "0001111110011111"]},
                        "id": "fc_8f2b1a3c"},
       "thoughtSignature": "b3BhcXVlLXNpZ25hdHVyZS1B"}
    ]}, "finishReason": "STOP", "index": 0}],
  "usageMetadata": {"promptTokenCount": 412, "candidatesTokenCount": 96, "thoughtsTokenCount": 2140,
                    "totalTokenCount": 2648, "cachedContentTokenCount": 0},
  "modelVersion": "gemini-4-carbon", "responseId": "r1AbCd"}
]
```

Assembled by `call()` [RUN]: `finish_reason == "STOP"`; one call
`("fc_8f2b1a3c", "write_rows", {"palette": [...], "start_row": 0, "rows": [...]})` with `args` a `dict`;
visible text `""`; thought summary present; usage prompt 412 / candidates 96 / thoughts 2140 / cached 0 / total
2648; `parts[1].thought_signature == b"opaque-signature-A"`. (`finishReason: "STOP"` on a function-call
response matches the one documented sample, a non-streaming `gemini-4-hydrogen` response [DOC:migrate]; that it
is also `STOP` on the last chunk of a *streamed* function-call response is not shown anywhere, so the adapter
must not depend on it - see 3.1.)

The same response as a single non-streaming body is the object with both parts in one `parts` array and the
second chunk's `finishReason` / `usageMetadata`.

### 10.4 Request 2 (replayed model turn + tool result)

`systemInstruction`, `tools`, `generationConfig` identical to request 1; `contents` is [RUN]:

```json
[
 {"parts": [{"text": "Paint: The Starry Night (16x16)."}], "role": "user"},
 {"parts": [
    {"text": "**Planning the canvas**\n\nI'll lay out a 16x16 grid: sky first, then the village.", "thought": true},
    {"functionCall": {"id": "fc_8f2b1a3c",
                      "args": {"palette": ["#0b1d51", "#f5d547"], "start_row": 0,
                               "rows": ["0000111100001111", "0001111110011111"]},
                      "name": "write_rows"},
     "thoughtSignature": "b3BhcXVlLXNpZ25hdHVyZS1B"}
  ], "role": "model"},
 {"parts": [
    {"functionResponse": {"id": "fc_8f2b1a3c", "name": "write_rows",
                          "response": {"output": "ok: rows 0-1 written (2 of 16)"}}}
  ], "role": "user"}
]
```

### 10.5 Response 2 (final text; signature on a trailing empty-text part)

```json
[
 {"candidates": [{"content": {"role": "model", "parts": [{"text": "The painting is complete: 2 rows written."}]}, "index": 0}],
  "usageMetadata": {"promptTokenCount": 6710, "thoughtsTokenCount": 220, "totalTokenCount": 6930},
  "modelVersion": "gemini-4-carbon", "responseId": "r2EfGh"},
 {"candidates": [{"content": {"role": "model", "parts": [{"text": "", "thoughtSignature": "b3BhcXVlLXNpZ25hdHVyZS1C"}]},
                  "finishReason": "STOP", "index": 0}],
  "usageMetadata": {"promptTokenCount": 6710, "candidatesTokenCount": 11, "thoughtsTokenCount": 220,
                    "totalTokenCount": 6941, "cachedContentTokenCount": 4096},
  "modelVersion": "gemini-4-carbon", "responseId": "r2EfGh"}
]
```

(Token counts are invented; they were raised in the verification pass so that the cache hit is at least
plausible - implicit caching on Gemini 4 only applies to prompts of 4,096 tokens or more. [DOC:caching])

Assembled [RUN]: no function calls -> the loop ends; text `"The painting is complete: 2 rows written."`; the
model turn has two parts (the second is `text == ""` with a signature) and both are replayed if a nudge follows
(wire form in 4.4).

### 10.6 Two tool calls in one reply

```json
[
 {"candidates": [{"content": {"role": "model", "parts": [
      {"functionCall": {"name": "write_rows", "args": {"start_row": 0, "rows": ["0000"]}, "id": "fc_1"},
       "thoughtSignature": "b3BhcXVlLXNpZ25hdHVyZS1D"},
      {"functionCall": {"name": "write_rows", "args": {"start_row": 1, "rows": ["1111"]}, "id": "fc_2"}}
    ]}, "finishReason": "STOP", "index": 0}],
  "usageMetadata": {"promptTokenCount": 412, "candidatesTokenCount": 60, "thoughtsTokenCount": 900, "totalTokenCount": 1372}}
]
```

Signature on the first call only [DOC:thought-signatures]. The follow-up request's last two `contents` entries
are the JSON in 4.3. [RUN]

### 10.7 Safety block / refusal

(a) Prompt blocked - no candidates [RUN]:

```json
[{"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
  "usageMetadata": {"promptTokenCount": 412, "totalTokenCount": 412},
  "modelVersion": "gemini-4-carbon", "responseId": "r5"}]
```
`call()` -> `content None`, `finish_reason None`, `block_reason == "PROHIBITED_CONTENT"`. Check `block_reason`
**before** applying the "no finish reason = retry" rule.

(b) Candidate blocked - candidate without content [RUN]:

```json
[{"candidates": [{"finishReason": "SAFETY", "index": 0,
                  "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH", "blocked": true}]}],
  "usageMetadata": {"promptTokenCount": 412, "totalTokenCount": 412}}]
```
`call()` -> `content None`, `finish_reason == "SAFETY"`.

### 10.8 Max-tokens truncation

Thinking consumed the whole budget [RUN] (shape taken from the doc's all-thinking hop sample [DOC:long-decode]):

```json
[{"candidates": [{"content": {"role": "model", "parts": [{"text": ""}]}, "finishReason": "MAX_TOKENS", "index": 0}],
  "usageMetadata": {"promptTokenCount": 412, "candidatesTokenCount": 0, "thoughtsTokenCount": 65536, "totalTokenCount": 65948}}]
```

Partial text, with the finish reason on a content-less last chunk [RUN]:

```json
[{"candidates": [{"content": {"role": "model", "parts": [{"text": "The painting is"}]}, "index": 0}]},
 {"candidates": [{"finishReason": "MAX_TOKENS", "index": 0}],
  "usageMetadata": {"promptTokenCount": 412, "candidatesTokenCount": 4, "thoughtsTokenCount": 65532, "totalTokenCount": 65948}}]
```

Assembled by `call()` [RUN]: first fixture -> `finish_reason == "MAX_TOKENS"`, `content` is **not** `None` (one
part, `text == ""`), no function calls, visible text `""`; second fixture -> `finish_reason == "MAX_TOKENS"`,
one part `"The painting is"`. Neither is a finished turn: "no function call" here means "truncated", so branch
on `finish_reason` before deciding that the model is done.

On Gemini 4 a `MAX_TOKENS` response "still includes a `continuation_token`" [DOC:long-decode "Resume when the
model uses all output tokens"], so a realistic fixture also has `"continuationToken": "CiQ4ZmQ3"` next to
`"finishReason"`. Through the transport fake 2.17.0 drops it silently; `model_validate` on the same dict rejects
it (10.1).

### 10.9 Other shapes worth a test

Malformed function call (content without parts) [RUN]:

```json
[{"candidates": [{"content": {"role": "model"}, "finishReason": "MALFORMED_FUNCTION_CALL",
                  "finishMessage": "Malformed function call: write_rows(rows=[", "index": 0}],
  "usageMetadata": {"promptTokenCount": 412, "thoughtsTokenCount": 800, "totalTokenCount": 1212}}]
```
-> `content None`, `finish_reason == "MALFORMED_FUNCTION_CALL"`.

Continuation hop that nobody continued (doc sample [DOC:long-decode]; only reachable when `max_output_tokens`
> 65,536):

```json
[{"candidates": [{"content": {"role": "model", "parts": [{"text": ""}]},
                  "finishReason": "CONTINUATION", "continuationToken": "CiQ4ZmQ3", "index": 0}],
  "usageMetadata": {"promptTokenCount": 142, "candidatesTokenCount": 0, "thoughtsTokenCount": 65536, "totalTokenCount": 65678}}]
```
On 2.17.0 -> `finish_reason == "CONTINUATION"` (plus a `UserWarning`), token dropped. [RUN]

HTTP errors: `429` + `{"error": {"code": 429, "message": "...", "status": "RESOURCE_EXHAUSTED"}}` ->
`errors.ClientError`, `.code == 429`; `503` + `{"error": {"code": 503, "message": "The model is overloaded.
Please try again later.", "status": "UNAVAILABLE"}}` -> `errors.ServerError`; `400` + `{"error": {"code": 400,
"message": "Function call `write_rows` in the 1. content block is missing a `thought_signature`.", "status":
"INVALID_ARGUMENT"}}` -> `errors.ClientError`. [RUN] (The 400 message text is modelled on the pattern quoted in
[DOC:thought-signatures]; the other messages are illustrative.)

---

## 11. Gotchas and surprises

1. **SDK too old for the current models.** See section 0.2 / 1.6. Code copied from today's docs
   (`automatic_continuation=`, `continuation_token=`, `types.FinishReason.CONTINUATION`,
   `candidate.continuation_token`, `types.ThinkingLevel.XHIGH`, `usage_metadata.initial_prompt_token_count`,
   `Part.from_function_response(id=...)`) fails on 2.17.0 with `ValidationError` / `AttributeError` /
   `TypeError`. [RUN]
2. **`max_output_tokens` includes thinking**, and a hop is capped at 64K thinking + output. At the highest
   thinking level an entire hop can be thinking with no text or tool call. Distinguish `MAX_TOKENS` /
   `CONTINUATION` / "no finish reason" from a genuine "model stopped without a tool call" before nudging.
3. **10-minute server limit on non-streaming calls**, 20 minutes on streams. Stream everything.
4. **No default client timeout, and a timeout on your own httpx client is ignored.** Set `HttpOptions.timeout`
   in **milliseconds**. [RUN]
5. **No default retries**, despite the docs; mid-stream errors are never retried by the SDK. [RUN]
6. **A stream that just ends raises nothing.** `finish_reason is None` after iteration = incomplete. [RUN]
7. **Unknown enum values do not raise**; they warn (`UserWarning`) and become pseudo-members. Compare finish and
   block reasons as strings. Keep `-W error` off for this adapter, or filter `UserWarning` from
   `google.genai._common`: with warnings turned into errors an unknown finish reason, block reason or
   `thinking_level="xhigh"` surfaces as `pydantic.ValidationError`, not as a warning. [RUN]
8. **Wire casing is mixed** (`thinkingConfig.thinking_level`, `parameters_json_schema`). Do not "fix" it and do
   not assert camelCase in request tests. [RUN]
9. **`candidate.finish_message` is always `None` on the Developer API with 2.17.0**: the converter does not copy
   `finishMessage`. [SRC models.py `_Candidate_from_mldev`][RUN]
10. **`response.text` is not a safe accessor**: `None` when there is no visible text, silently excludes thought
    parts, and logs a warning when function calls are present. Iterate parts. [RUN]
11. **`candidates`, `candidates[0].content`, `content.parts`, `function_call.args`, `function_call.id` and every
    usage field can each be `None`.** [RUN][SRC]
12. **Never merge, reorder, drop or re-create parts of a model turn.** The trailing `{"text": "",
    "thoughtSignature": ...}` part is real content. [DOC:thought-signatures]
13. **Signature base64 alphabet changes on the way back** (standard -> URL-safe). Tests comparing the outgoing
    request with the incoming fixture should compare decoded bytes. [RUN]
14. **Explicit caching, temperature, top_p, top_k, candidate_count, thinking_budget, `minimal`** are all
    unsupported / ignored / errors on Gemini 4 (5.5).
15. **Carbon's price doubles on 2026-12-01** ($4 / $20). Cost tables must be date-aware. [DOC:pricing]
16. **Request storage is on by default for new paid-tier projects** since 2026-10-26 (`store=true` for
    `generateContent` too); opt out per request with `store=false`. [DOC:changelog] 2.17.0 has no `store` config
    field; `GenerateContentConfig(http_options=types.HttpOptions(extra_body={"store": False}))` puts
    `"store": false` at the top level of the body, where the REST reference places it. [RUN][DOC:ref-gc]
17. **Tier-1 spend limit of $10 per 10 minutes** can throttle parallel contestants with 429s. [DOC:rate-limits]
18. **`system_instruction` is sent with `role: "user"`** by the SDK - expected, not a bug. [RUN]
19. **Python callables in `tools` trigger automatic function calling** (up to 10 hidden round trips). Only pass
    `types.Tool` objects, and set `disable=True` anyway. [SRC]
20. **Dummy signatures exist for injected history**: `"context_engineering_is_the_way_to_go"` or
    `"skip_thought_signature_validator"` "in the thought signature field to skip validation" - "strongly
    discouraged". The harness never fabricates function calls, so it should not need them.
    [DOC:thought-signatures FAQ 1]
21. The docs' own function-calling page still says "Use a low temperature (e.g., 0)" under best practices while
    other pages say temperature is ignored on Gemini 4. Ignore the former. [DOC:function-calling][DOC:text]
22. Prompting pitfall from the docs: requiring structured text (XML/JSON) immediately before a tool call "may
    occasionally fail with `Malformed_Function_Call`"; and asking for raw-text JSON tool calls with zero declared
    tools can return `MALFORMED_FUNCTION_CALL`. [DOC:function-calling][DOC:neon]
23. **`str(finish_reason)` is `"FinishReason.STOP"`**, not `"STOP"` (same for every SDK enum, including block
    reasons). Persist and log `.value`. [RUN]
24. **`types.BlockedReason.BLOCKED_REASON_UNSPECIFIED` is not the wire spelling** (`BLOCK_REASON_UNSPECIFIED`);
    the SDK enum also has `MODEL_ARMOR` and `JAILBREAK`, which the Developer API reference does not list. [RUN]
25. **An invalid API key can be HTTP 400 `INVALID_ARGUMENT`** (ErrorInfo reason `API_KEY_INVALID`, the errors
    page's own sample body) as well as 403 `PERMISSION_DENIED` (its table), so "400 = my request body is wrong"
    is not always true; read `e.message`. [DOC:errors]
26. **Any non-`data: ` line in a stream body is fatal to the SDK's parser** (`UnknownApiResponseError`), which
    matters for hand-written SSE fixtures: no comments, no `event:` lines, no `[DONE]`. [RUN]
27. **The docs disagree with themselves on the non-streaming time limit**: "Non-streaming calls: The server
    closes the connection after 10 minutes" versus a comparison table on the same page that lists "Time limit
    per request: 20 minutes" for the non-streaming loop too. [DOC:long-decode] Streaming (20 minutes in both
    places) sidesteps it.
28. **The SDK identifies itself on every request** with `user-agent` and `x-goog-api-client:
    google-genai-sdk/2.17.0 gl-python/<version>`, plus `x-server-timeout: <seconds>` when a timeout is set. [RUN]
    Tests that assert on headers should expect them.

---

## UNVERIFIED (from memory)

The open question in each item is not backed by a fetched page or by SDK source. Treat as hypotheses. Where an
item carries a [RUN] or [DOC] tag, the tag marks the neighbouring fact that *is* backed, not the question.

1. `finishReason` on the last chunk of a **streamed** response whose parts are function calls. (The
   non-streaming case is no longer unverified: the migration guide shows `"finishReason": "STOP"` on a
   function-call response, see 3.1. No fetched page shows the streamed case; the fixtures in section 10 assume
   `STOP` there as well. The adapter must not depend on it.)
2. Integer arguments might arrive as JSON floats (`3.0`) because function args travel as a protobuf `Struct`.
   Cheap defence: accept integral floats for `start_row`. (The only evidence either way is the printed doc
   sample `args={'color_temp': 'warm', 'brightness': 25}`, an `int`. [DOC:function-calling])
3. How the 20-minute (stream) / 10-minute (unary) server cut shows up client-side: a clean end of stream
   without `finishReason`, an `httpx.RemoteProtocolError`, or a 504. Handle all three as retryable.
4. Whether the server sends any bytes during a long thinking phase when `include_thoughts` is off. If it does
   not, only the 20-minute read timeout protects against a dead connection. `include_thoughts=True` makes the
   stream carry rolling summaries.
5. 429 bodies carrying a `google.rpc.RetryInfo` detail with `retryDelay`. Parse opportunistically from
   `e.details["error"]["details"]`; never rely on it.
6. Encoding of the dummy signatures through the SDK (`thought_signature=b"skip_thought_signature_validator"`,
   which goes out base64-encoded) - unverified that the API accepts that form.
7. Whether two consecutive `user` turns (e.g. a function-response turn followed directly by a text nudge) are
   accepted. Avoid the situation.
8. Whether a model turn with `parts: []` is accepted on input.
9. What the API returns for a thinking level the model does not list (e.g. `xhigh` on carbon); presumably 400.
10. The mechanism by which the server grants "SDK" non-streaming calls a 64K hop on carbon/argon while "direct
    REST" gets 32K (the docs state the outcome only).
11. Actual per-image token counts for small PNGs on Gemini 4 (whether a tiny image costs fewer than 1120
    tokens).
12. Whether replayed `thought: true` summary parts are billed as input.
13. Whether the server ever puts anything other than `data: <json>` events on the stream (SSE comments or
    keep-alive lines during long thinking). If it did, SDK 2.17.0 would raise `UnknownApiResponseError`
    mid-stream [RUN]; treat that exception as retryable.
14. Whether mid-stream checkpoint `continuationToken`s appear on carbon/argon streams when `max_output_tokens`
    is 65,536 or lower (the docs describe checkpoints "after the first 32,768 tokens" without saying whether
    they depend on `max_output_tokens`). Harmless on 2.17.0, which drops the field. [RUN]
15. That the API accepts the exact mixed-case body the SDK emits (`thinkingConfig.thinking_level`,
    `thinkingConfig.include_thoughts`, `functionDeclarations[].parameters_json_schema`). This is what the
    official SDK sends [RUN] and the docs' REST samples use snake_case for other fields, but no fetched page
    states it for these three fields. If a live call ever returns `400 INVALID_ARGUMENT` naming an unknown
    field, switch the declaration to `parameters=<dict>` (2.3), whose wire name `parameters` has no case
    variants.
16. Everything about SDK >= 2.30.1 behaviour at run time. The newer wheels were read, not installed or run.

Not obtainable from the pages fetched: RPM/TPM limits for Gemini 4 (AI Studio only), any rate-limit response
headers, a `generateContent` retirement date.

---

## Sources

All fetched 2026-11-19 (UTC) with a generic User-Agent, and all fetched again on 2026-11-20 for the independent
pass (every URL below answered 200). Pages marked (gc) are the `generateContent` variants
(`?generate-content=true`); unmarked guide pages are the default (Interactions) variants or pages with a single
version. `https://ai.google.dev/gemini-api/docs/<page>.md.txt` now answers `301` to
`https://aistudio.google.com/docs/<page>.md.txt` and returns the default (Interactions) variant, so the
`generateContent` variants must be requested from aistudio.google.com with the query parameter.

Documentation:

- [DOC:llms] https://aistudio.google.com/llms.txt (reached via the redirect from
  https://ai.google.dev/gemini-api/docs/llms.txt; index of all pages; states `generateContent` is legacy)
- [DOC:long-decode] https://aistudio.google.com/docs/long-decode.md?generate-content=true (gc)
- [DOC:thought-signatures] https://aistudio.google.com/docs/thought-signatures.md?generate-content=true (gc)
- [DOC:thinking] https://aistudio.google.com/docs/thinking.md?generate-content=true (gc)
- [DOC:function-calling] https://aistudio.google.com/docs/function-calling.md?generate-content=true (gc)
- [DOC:text] https://aistudio.google.com/docs/text-generation.md?generate-content=true (gc)
- [DOC:image] https://aistudio.google.com/docs/image-understanding.md?generate-content=true (gc)
- [DOC:media-resolution] https://aistudio.google.com/docs/media-resolution.md?generate-content=true (gc)
- [DOC:tokens] https://aistudio.google.com/docs/tokens.md?generate-content=true (gc)
- [DOC:file-input] https://aistudio.google.com/docs/file-input-methods.md?generate-content=true (gc)
- [DOC:safety] https://aistudio.google.com/docs/safety-settings.md?generate-content=true (gc)
- [DOC:errors] https://aistudio.google.com/docs/api-errors.md?generate-content=true (gc)
- [DOC:caching] https://aistudio.google.com/docs/caching.md?generate-content=true (gc)
- [DOC:api-key] https://aistudio.google.com/docs/api-key.md?generate-content=true (gc)
- [DOC:carbon] https://aistudio.google.com/docs/whats-new-gemini-4-carbon.md?generate-content=true (gc)
- [DOC:neon] https://aistudio.google.com/docs/whats-new-gemini-4-neon.md?generate-content=true (gc)
- https://aistudio.google.com/docs/whats-new-gemini-4-hydrogen.md?generate-content=true (gc)
- https://aistudio.google.com/docs/whats-new-gemini-4-argon.md?generate-content=true (gc)
- https://aistudio.google.com/docs/latest-model.md?generate-content=true (gc)
- [DOC:models] https://aistudio.google.com/docs/models.md
- [DOC:model-cards] https://aistudio.google.com/docs/models/gemini-4-carbon.md, .../gemini-4-argon.md,
  .../gemini-4-neon.md, .../gemini-4-hydrogen.md, .../gemini-3.8-flash.md, .../gemini-3.5-flash-lite.md
- [DOC:choosing] https://aistudio.google.com/docs/choosing-a-model.md
- [DOC:pricing] https://aistudio.google.com/docs/pricing.md
- [DOC:rate-limits] https://aistudio.google.com/docs/rate-limits.md
- [DOC:troubleshooting] https://aistudio.google.com/docs/troubleshooting.md
- [DOC:deprecations] https://aistudio.google.com/docs/deprecations.md
- [DOC:changelog] https://aistudio.google.com/docs/changelog.md
- [DOC:interactions] https://aistudio.google.com/docs/interactions-overview.md
- [DOC:background] https://aistudio.google.com/docs/background-execution.md
- [DOC:migrate] https://aistudio.google.com/docs/migrate-to-interactions.md (its "Before (generateContent)" tabs
  hold sample `generateContent` request/response bodies, including the only function-call response that shows
  `"finishReason": "STOP"`)
- https://aistudio.google.com/docs/libraries.md
- [DOC:ref-overview] https://aistudio.google.com/api-reference/generate-content.md (v1beta data models)
- [DOC:ref-gc] https://aistudio.google.com/api-reference/generate-content/generate-content.md (v1beta request/response)
- https://aistudio.google.com/api-reference/generate-content/generate-content.md?api-version=v1 (checked that
  `continuationToken`, `thoughtSignature`, `thinkingLevel`/`XHIGH`, `parametersJsonSchema`, `store` also exist in v1)
- https://aistudio.google.com/api-reference/generate-content/stream-generate-content.md
- [DOC:api-versions] https://aistudio.google.com/api-reference/api-versions.md
- [PyPI] https://pypi.org/pypi/google-genai/json (release list and dates)

Installed SDK source read (google-genai 2.17.0, under `site-packages/google/genai/`), 2026-11-19:

- `client.py` (`Client`, `AsyncClient`, `DebugConfig`)
- `_api_client.py` (`get_env_api_key`, `BaseApiClient.__init__`, `_build_request`, `_request_once`, `_request`,
  `request_streamed`, `HttpResponse._iter_response_stream`, `retry_args`, `_RETRY_*`, `get_timeout_in_seconds`,
  `populate_server_timeout_header`, `_ensure_httpx_ssl_ctx`)
- `_base_url.py` (`get_base_url`)
- `errors.py` (`APIError`, `ClientError`, `ServerError`, `UnknownApiResponseError`)
- `models.py` (`Models.generate_content`, `Models.generate_content_stream`, `_generate_content`,
  `_generate_content_stream`, `_GenerateContentParameters_to_mldev`, `_GenerateContentConfig_to_mldev`,
  `_Content_to_mldev`, `_Part_to_mldev`, `_Blob_to_mldev`, `_FunctionCall_to_mldev`, `_Tool_to_mldev`,
  `_ToolConfig_to_mldev`, `_FunctionCallingConfig_to_mldev`, `_GenerateContentResponse_from_mldev`,
  `_Candidate_from_mldev`)
- `types.py` (`ThinkingLevel`, `FinishReason`, `BlockedReason`, `FunctionCallingConfigMode`, `MediaResolution`,
  `PartMediaResolutionLevel`, `HarmCategory`, `HarmBlockThreshold`, `Part`, `Content`, `Blob`, `FunctionCall`,
  `FunctionResponse`, `FunctionDeclaration`, `Schema`, `Tool`, `ToolConfig`, `FunctionCallingConfig`,
  `AutomaticFunctionCallingConfig`, `ThinkingConfig`, `GenerateContentConfig`, `HttpOptions`,
  `HttpRetryOptions`, `Candidate`, `SafetyRating`, `GenerateContentResponsePromptFeedback`,
  `GenerateContentResponseUsageMetadata`, `ModalityTokenCount`, `GenerateContentResponse`)
- `_extra_utils.py` (`find_afc_incompatible_tool_indexes`, `get_function_map`, `should_disable_afc`,
  `get_function_response_parts`, `append_chunk_contents`)
- `_transformers.py` (`t_model`, `t_part`, `t_content`, `t_contents`, `t_schema`, `t_tool`, `t_tools`)
- `_common.py` (`BaseModel`, `_remove_extra_fields`, `CaseInSensitiveEnum`, `convert_to_dict`,
  `encode_unserializable_types`)
- `chats.py` (`_validate_content`, `_extract_curated_history`, `Chat.send_message_stream`)

Offline executions against the installed SDK (httpx.MockTransport; the scratch scripts were not kept): request
serialisation, response parsing, SSE parsing, error mapping, retry behaviour, timeouts, env-var precedence, enum
handling, transcript round-trip.

Newer SDK source read only (wheels from files.pythonhosted.org, unpacked, not installed, not run):
`google-genai 2.34.0` and `3.6.1` - `types.py` (`ThinkingLevel`, `FinishReason`,
`GenerateContentConfig.continuation_token/automatic_continuation/store`, `Candidate.continuation_token`,
`GenerateContentResponseUsageMetadata.initial_prompt_token_count`, `Part.from_function_response`), `models.py`
(`_generate_content_with_continuation`, `_generate_content_stream_with_continuation`, `_Candidate_from_mldev`),
`_extra_utils.py` (`should_enable_automatic_continuation`, `StreamContinuation`, `should_continue_generation`,
`prepare_continuation_config`, `merge_pydantic_models`, `merge_continuation_responses`), `_api_client.py`
(`GOOGLE_AI_API_DEFAULT_VERSION`, `retry_args`), and `METADATA` (dependencies).

---

## Verification (independent pass)

Done 2026-11-20 by a second reviewer, who did not write these notes. No model/inference endpoint was called; every
SDK execution went through `httpx.MockTransport` with `socket.connect` patched to raise. The scratch material
(fetched pages, scripts, wheels) was not kept.

### What was checked, and how

1. **Documentation, re-fetched.** Every documentation URL in "Sources" (41 pages) and the PyPI JSON were
   downloaded again (all HTTP 200), along with five further pages used for cross-checks (the `get-started` and
   `structured-output` generateContent guides, the `gemini-3.6-flash` and `gemini-3.1-pro-preview` model cards,
   the v1 reference overview). They were read directly, not through the first pass's copies. Checked against
   them: every quoted sentence;
   the long-decode limits (10/20 minutes, 64K/32K hop caps, checkpoints, `CONTINUATION`/`MAX_TOKENS` rules,
   minimum SDK versions); the thinking-level table and defaults; the thought-signature rules, 400 message
   pattern and both dummy signatures; function-calling `id` rule, modes, ordering statements and best
   practices; the complete `finishReason`, `blockReason`, `thinkingLevel`, function-calling `mode` and
   `mediaResolution` enumerations in the REST reference (v1beta and v1); request-body field placement
   (`tools`, `toolConfig`, `systemInstruction`, `generationConfig.*`, top-level `store` and
   `continuationToken`); usage field names; image formats, limits and token table; all model ids, release
   dates, deprecations and every price in section 8; the error table; rate-limit and retry guidance.
2. **Installed SDK 2.17.0, re-executed offline.** Re-ran the whole contestant loop from these notes
   (`make_client`, `call`, request 1, tool-result turn, nudge) and the judge call, and compared the captured
   wire bodies with the JSON printed in 2.2, 4.1, 4.3, 4.4, 7.1 and 10.4 (equal as parsed JSON; 7.1 apart from
   the image data it abbreviates). Re-ran: env-var
   precedence, base-URL override, model-name validation, timeout propagation and `X-Server-Timeout`, lazy
   generator, SSE parsing variants, mid-stream errors, error-class mapping, retry behaviour (default, enabled,
   per-call, which exceptions retry), unknown-enum handling, `parameters` vs `parameters_json_schema`
   serialisation, tool merging, AFC logging and paths, `response.text`, image encoding and its pitfalls,
   media-resolution serialisation, transcript round trip, signature re-encoding, `extra_body`, async variants,
   strict `model_validate` vs the transport path.
3. **Section 10 fixtures.** Every fenced `json` block in this file was extracted programmatically and
   parsed. All 16 are valid JSON. Every response fixture was (a) validated with
   `types.GenerateContentResponse.model_validate` (all pass except the `CONTINUATION` one, which fails on
   `continuationToken` exactly as 10.1 says) and (b) fed through the SDK via the transport fake and `call()`,
   giving the assembled results stated next to each fixture. Every `contents` fixture was validated as
   `types.Content` and re-serialised by the SDK byte-for-byte equal to the printed JSON. Token arithmetic in
   every `usageMetadata` adds up.
4. **Newer SDKs, source only.** Downloaded the 2.30.1, 2.34.0 and 3.6.1 wheels from PyPI (sha256 equal to the
   PyPI JSON), unzipped, read; not installed, not imported. Confirmed every cell of the 1.6 matrix and the
   continuation description in 1.4.

### Corrections made (old -> new)

1. UNVERIFIED 1 / 3.1 / 10.3: "No fetched page shows a function-call response with its finish reason" ->
   wrong; the migration guide shows one, with `"finishReason": "STOP"` (non-streaming). Moved into 3.1 with its
   source; only the streamed case remains unverified.
2. 2.3: "Every example in the function-calling guide uses this `parameters` form" -> one Python example uses
   `parameters_json_schema=` exactly as section 2.1 does; text corrected (this strengthens the 2.1 recipe).
3. 9.1: "`UnknownApiResponseError` - a response/stream chunk that is not JSON" -> stream events only; a
   non-JSON 200 body on the non-streaming path raises `json.JSONDecodeError`; a non-JSON error body still maps
   to `ClientError`/`ServerError` with the HTTP reason phrase as `.status`.
4. 3.4: "`traffic_type` ... Vertex only" (unbacked) -> "no counterpart in the Developer API reference"; added
   the wire field `usageMetadata.serviceTier`, which 2.17.0 drops.
5. 3.3: block-reason list was the wire list only -> added that the SDK enum spells the first member
   `BLOCKED_REASON_UNSPECIFIED` and has two extra members, so the wire value warns as unknown.
6. 3.2 / 5.1 / gotcha 7: "do not run with warnings-as-errors" (reason unstated) -> with warnings as errors an
   unknown enum value is a `pydantic.ValidationError`, both when parsing a response and when building
   `ThinkingConfig(thinking_level="xhigh")`.
7. 5.1: source tag `[DOC:model-cards]` for the thinking-level table -> the model cards do not carry it;
   re-sourced to the thinking and long-decode pages (the values themselves were right).
8. 1.4 / 6: "the assembler in 1.5 needs no change" on SDK >= 2.30.1 -> true for clean hops only; added the
   reconnect caveat (already-yielded chunks after the last checkpoint are not retracted) and the
   `automatic_continuation=False` way out.
9. 7.1: judge snippet read `result["content"].parts` unguarded (`AttributeError` when the prompt is blocked or
   no parts came back) -> guarded, and tied to `finish_reason`.
10. 10.5: fixture reported a 2,048-token cache hit on a 2,710-token prompt, below the documented 4,096-token
    implicit-cache minimum -> 4,096 of 6,710 (totals recomputed).
11. 10.8: assembled results were not stated -> added (`content` is not `None` for the all-thinking fixture), plus
    the documented fact that `MAX_TOKENS` responses also carry `continuationToken`.

Added, not corrections: function-call parts are never streamed partially (1.5); `str(enum)` pitfall (1.5,
gotcha 23); invalid API key can be a 400 (9.2, gotcha 25); a manual continuation loop is impossible through the
2.17.0 streaming method (1.4); SSE strictness (9.1, gotcha 26); the docs' 10-vs-20-minute inconsistency (gotcha
27); SDK identification headers (gotcha 28); UNVERIFIED items 13-16.

### Checked and left unchanged

Everything else. In particular, re-confirmed without change: the tool definition and where it sits on the wire;
`functionCall.{id,name,args}` and `args` being a dict; the function-response part (`id`, `name`, `response`
dict, role `user`, one user turn for all responses); thought-signature replay rules and the URL-safe base64
re-encoding; `thinking_level` values per model and the absence of `XHIGH` in the 2.17.0 enum; `max_output_tokens`
counting thinking; the full finish-reason list; usage field names and the "thinking is additional to
candidates" reading; the timeout unit (milliseconds) and the no-default-timeout / no-default-retry behaviour;
`inlineData` image parts; model ids and prices.

### Still unverified

The numbered list under "UNVERIFIED (from memory)" (items 1-16), which nobody can settle without a live call.
(As this pass left it. The adapter has since been run live, and these notes were not revised after that run.)
The ones most likely to matter on first contact with the real API:

- how the 20-minute server cut on a stream surfaces (3), and whether anything is sent during long silent
  thinking (4, 13);
- the finish reason on a *streamed* function-call turn (1) - harmless if the adapter keys on parts as told;
- acceptance of the SDK's mixed-case request body for Gemini 4 (15) - this is the official SDK's own output, so
  the risk is low, but it has not been observed against `gemini-4-*`;
- whether one 65,536-token, 20-minute hop is enough for a contestant at the highest thinking level. On the
  installed 2.17.0 there is no way to continue a turn that hits either limit (1.4); that is a property of the
  SDK version, and section 0 says which way the adapter went.
