# OpenAI-compatible Chat Completions notes ("openai_compat" adapter)

Researched 2026-11-19 and 2026-11-20 (UTC). No model/inference endpoint was called.
The only non-documentation HTTP request was an unauthenticated `GET https://openrouter.ai/api/v1/models`
(OpenRouter's public model catalog: ids, prices, limits, supported parameters; it runs no model).

The adapter written from these notes has not been run against a live API. It is tested against mocked responses
only, so everything here is what the documentation and the SDK source say, not what a server was seen to do.

Target: any server that speaks `POST {base_url}/chat/completions` with tools. OpenRouter first; also vLLM, Ollama,
llama.cpp server, LM Studio, Together, Fireworks, Groq, DeepSeek, Mistral, xAI.

Evidence tags:

- `[SDK: <path> :: <symbol>]` = read in the installed SDK `openai 2.53.0`, path relative to
  `site-packages/openai/`.
- `[OFFLINE]` = observed by running the installed SDK against `httpx.MockTransport` (no network): the exact request
  body/headers the SDK builds, and how it parses canned replies and canned SSE streams.
- `[OR: <page>]` = `https://openrouter.ai/docs/<page>.md`, fetched as markdown.
- `[OR-API]` = the OpenAPI excerpt for the endpoint at
  `https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion.md` (schema names in backticks).
- `[OR-CAT]` = `https://openrouter.ai/api/v1/models`, fetched 2026-11-19.
- `[DOC: <url>]` = another official documentation page fetched for these notes.
- `[SRC: <url>]` = a source file in the vendor's own public repository, fetched for these notes.
- Anything not backed this way is in "UNVERIFIED (from memory)" at the end and nowhere else.

"Canonical" below means the shape in the OpenAI SDK types, which are generated from OpenAI's API spec.

---

## 0. Read this first: decisions

| Question | Answer |
| --- | --- |
| Endpoint / method | `POST {base_url}/chat/completions`; SDK `client.chat.completions.create(...)`. |
| Response handling | Treat every response as **plain JSON (dict)**, never as SDK model objects. One internal shape for both transport modes: a dict shaped like a non-streaming `chat.completion`. |
| Long calls | Default `stream=True` and assemble the chunks with **our own** accumulator (section 1.6, code tested offline). Config switch `stream: false` uses `with_raw_response.create(...)` + `json.loads`. Never use the SDK's `.stream()` helper / `ChatCompletionStreamState` (it corrupts compat-server streams, section 1.7). |
| Reasoning across tool turns | Replay the **whole assistant `message` object exactly as received** (all keys, including `reasoning_details`, `reasoning`, `reasoning_content`, list-valued `content`), then one `role:"tool"` message per call. History is append-only: never edit the system prompt, the tool list or any earlier message. |
| Highest reasoning setting | Not standardised. OpenRouter: body key `reasoning: {"effort": <highest listed for the model>}` (`max` for current Claude / GPT-6 / DeepSeek / GLM / Kimi / Muse, `xhigh` for Gemini 4 Neon/Hydrogen, Grok 4.7, Qwen4 Max, `high` for Gemini 4 Argon/Carbon and Mistral Large 4). Discover with `GET /api/v1/models` -> `reasoning.supported_efforts[0]`. Other servers: section 5. |
| Output-limit parameter | Config switch `max_tokens_param`: `"max_tokens"` (default; right for OpenRouter, Ollama, DeepSeek, Together, LM Studio, llama.cpp), `"max_completion_tokens"` (OpenAI direct, Groq, xAI), or `null` to omit. |
| Tool | Canonical `{"type":"function","function":{name,description,parameters}}`; no `strict`, no `required`, no `additionalProperties`. Omit `tool_choice` and `parallel_tool_calls`. |
| Is "extra_body + verbatim replay" enough? | Yes for the request knobs and for replay, **plus** four things it does not cover: the `max_tokens_param` switch, the stream-reassembly rule for `reasoning_details`, error objects inside HTTP 200, and (Gemini 4 on OpenRouter only) the long-decode continuation. Details in section 5.6. |
| Not for | First-party OpenAI GPT-6 Astra / GPT-6.1 Sol / GPT-6.2 with tools: OpenAI says tool calling for these requires the Responses API (section 11, item 1). Use the `openai` adapter for `api.openai.com`. Through OpenRouter the same models work on Chat Completions. |

---

## 1. Endpoint, client, auth, timeouts, long requests

### 1.1 Endpoint and method

- `POST {base_url}/chat/completions`, JSON body, `Authorization: Bearer <key>`.
  `[OFFLINE]` (URL the SDK built for `base_url="https://example.invalid/api/v1"` was `.../api/v1/chat/completions`).
- OpenRouter: "OpenRouter supports two request shapes: the Responses API (`POST /api/v1/responses`) and the Chat
  Completions API (`POST /api/v1/chat/completions`)." `[OR: api_reference/overview]`
  The Chat Completions row of the compatibility matrix: "Full support for streaming, tools, structured outputs, and
  vision across 400+ models." `[OR: guides/features/compatibility]`
- Why Chat Completions and not OpenRouter's Responses or Messages skins: it is the one protocol all the listed
  servers share (vLLM, Ollama, llama.cpp, LM Studio, DeepSeek, Groq, Together, Fireworks, Mistral's
  `/openai/v1`, xAI's legacy endpoint all document `/v1/chat/completions`; sources in section "Sources").
- SDK method: `client.chat.completions.create(*, messages, model, ..., extra_headers=None, extra_query=None,
  extra_body=None, timeout=not_given)`. `[SDK: resources/chat/completions/completions.py :: Completions.create]`

### 1.2 Client construction

```python
import os, httpx
from openai import OpenAI, Omit

client = OpenAI(
    base_url=base_url,                    # e.g. "https://openrouter.ai/api/v1"; always pass it explicitly
    api_key=api_key,                      # value of os.environ[api_key_env]; MUST be a non-empty string
    timeout=httpx.Timeout(connect=15.0, read=900.0, write=60.0, pool=60.0),
    max_retries=0,                        # the harness owns retries (section 9)
    default_headers={                     # optional: drop the SDK's platform headers; keep its generic User-Agent
        "X-Stainless-Lang": Omit(), "X-Stainless-Package-Version": Omit(), "X-Stainless-OS": Omit(),
        "X-Stainless-Arch": Omit(), "X-Stainless-Runtime": Omit(), "X-Stainless-Runtime-Version": Omit(),
        "X-Stainless-Async": Omit(), "OpenAI-Organization": Omit(), "OpenAI-Project": Omit(),
    },
)
```

Facts behind each line:

- If `api_key` is `None` the constructor reads `OPENAI_API_KEY`; if `base_url` is `None` it reads `OPENAI_BASE_URL`,
  else `https://api.openai.com/v1`. It also reads `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID` (sent as headers
  `OpenAI-Organization` / `OpenAI-Project` to whatever `base_url` is), `OPENAI_CUSTOM_HEADERS` (extra headers,
  one `Name: value` per line, sent on every request), `OPENAI_ADMIN_KEY` and `OPENAI_WEBHOOK_SECRET` (neither is
  sent on a chat-completions request).
  `[SDK: _client.py :: OpenAI.__init__, OpenAI.default_headers]` `[OFFLINE]` (with `OPENAI_ORG_ID` set, the header
  went to a non-OpenAI base URL; with the `Omit()` map below it did not.)
  So: always pass `base_url` and `api_key` explicitly, validate the key is non-empty first, and omit the two
  `OpenAI-*` headers. OpenRouter documents this exact trap: with `api_key=None` "the SDK will send that OpenAI key
  (`sk-proj-...`) to `openrouter.ai`" and gets `401` "Invalid API key: this is not an OpenRouter API key".
  `[OR: guides/features/compatibility]`
- An empty key raises `OpenAIError("Missing credentials ...")` at construction, **unless `OPENAI_ADMIN_KEY` is set
  in the environment** (the constructor also reads that variable): then construction succeeds and the first
  `chat.completions.create` raises `TypeError("Could not resolve authentication method ...")` before any request is
  sent. With a non-empty `api_key` the admin key is never sent on a chat-completions request (`Authorization` is
  `Bearer <api_key>`). `[SDK: _client.py :: OpenAI.__init__, auth_headers, _validate_headers]` `[OFFLINE]`
  So validate the key yourself; do not rely on the constructor. Keyless local servers need a dummy non-empty string
  (vLLM docs use `"EMPTY"`/`"dummy"`, Ollama `'ollama'` "required but ignored", LM Studio `"lm-studio"`).
  `[DOC: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/reasoning_outputs.md]`
  `[DOC: https://raw.githubusercontent.com/ollama/ollama/main/docs/api/openai-compatibility.mdx]`
  `[DOC: https://lmstudio.ai/llms.txt]`
- Headers the SDK sends by default (observed): `authorization`, `accept: application/json`,
  `content-type: application/json`, `user-agent: OpenAI/Python 2.53.0`, `x-stainless-lang`,
  `x-stainless-package-version`, `x-stainless-os` (`Linux`), `x-stainless-arch` (`x64`), `x-stainless-runtime`
  (`CPython`), `x-stainless-runtime-version` (`3.12.13`), `x-stainless-async`, `x-stainless-retry-count`,
  `x-stainless-read-timeout`. With the `Omit()` map above plus per-request
  `extra_headers={"X-Stainless-Retry-Count": Omit(), "X-Stainless-Read-Timeout": Omit()}` only `authorization`,
  `accept`, `content-type`, `user-agent` (and httpx's `host`, `accept-encoding`, `connection`, `content-length`)
  remain. `[OFFLINE]` None of these identify the user or the project; omitting them is optional hygiene. The
  SDK's default `User-Agent` (`OpenAI/Python 2.53.0`) is generic too, and nothing is added to it.
- OpenRouter attribution headers are optional and we send none: `HTTP-Referer`, `X-OpenRouter-Title` (`X-Title`
  also accepted), `X-OpenRouter-Categories`, `X-OpenRouter-App-Visibility`. Without `HTTP-Referer` "no app page
  will be created and your usage will not appear in rankings". `[OR: app-attribution]`
  `[OR: api_reference/overview]`

### 1.3 Base URLs and key variables

| Server | `base_url` | Key | Source |
| --- | --- | --- | --- |
| OpenRouter | `https://openrouter.ai/api/v1` (regional hosts `https://eu.openrouter.ai` and `https://us.openrouter.ai`, Business/Enterprise only) | `OPENROUTER_API_KEY`; format `sk-or-v1-` + 64 lowercase hex (73 chars) | `[OR: guides/features/compatibility]` `[OR: api_reference/authentication]` `[OR: guides/routing/provider-selection]` |
| vLLM | `http://localhost:8000/v1` | none unless server started with `--api-key` / `VLLM_API_KEY` | `[DOC: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/serving/online_serving/openai_compatible_server.md]` |
| Ollama (local) | `http://localhost:11434/v1/` | ignored | `[DOC: https://raw.githubusercontent.com/ollama/ollama/main/docs/api/openai-compatibility.mdx]` |
| Ollama (cloud) | `https://ollama.com/v1` | `OLLAMA_API_KEY` | same |
| llama.cpp server | `http://127.0.0.1:9931/v1` (README defaults: `--host 127.0.0.1`, `--port 9931`) | none unless `--api-key` | `[DOC: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md]` |
| LM Studio | `http://localhost:1234/v1` | ignored | `[DOC: https://lmstudio.ai/llms.txt]` |
| DeepSeek | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | `[DOC: https://api-docs.deepseek.com/]` |
| Groq | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` | `[DOC: https://console.groq.com/docs/openai.md]` |
| Together | `https://api.together.ai/v1` | `TOGETHER_API_KEY` | `[DOC: https://docs.together.ai/docs/inference/openai-compatibility.md]` |
| Fireworks | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` | `[DOC: https://docs.fireworks.ai/guides/function-calling.md]` |
| Mistral | `https://api.mistral.ai/openai/v1` (OpenAI-compatible, "public preview"); `https://api.mistral.ai/v1` is the native Mistral format | `MISTRAL_API_KEY` | `[DOC: https://docs.mistral.ai/llms-full.txt]` (section "chat-completion") |
| xAI | `https://api.x.ai/v1` (Chat Completions is "a legacy endpoint") | `XAI_API_KEY` | `[DOC: https://docs.x.ai/developers/model-capabilities/legacy/chat-completions.md]` |

OpenRouter model ids are `<author>/<slug>` (e.g. `anthropic/claude-fable-5.2`); "Do not hardcode raw model names
like `gpt-4o` without the `openai/` namespace". `[OR: guides/features/compatibility]` Each catalog record has `id`
(routable) and `canonical_slug` (dated, permanent, e.g. `anthropic/claude-fable-5.2-20261109`). Aliases with a `~`
prefix (`~anthropic/claude-sonnet-latest`) roll forward automatically: do not use them for a benchmark. Suffixes:
`:free`, `:batch` are separate catalog entries; `:nitro`, `:floor`, `:exacto` are request-time routing modifiers;
`:thinking`, `:extended`, `:online` are deprecated. `[OR: api_reference/models]` `[OR-CAT]`

### 1.4 Timeouts and SDK retries (defaults)

- `DEFAULT_TIMEOUT = httpx.Timeout(timeout=600, connect=5.0)`, `DEFAULT_MAX_RETRIES = 2`,
  `INITIAL_RETRY_DELAY = 0.5`, `MAX_RETRY_DELAY = 8.0`, `MAX_RETRY_AFTER_DELAY = 120`. `[SDK: _constants.py]`
- "By default requests time out after 10 minutes"; "requests that time out are retried twice by default"; a timeout
  raises `APITimeoutError`. `[SDK: ../openai-2.53.0.dist-info/METADATA (README, sections Retries, Timeouts)]`
- `httpx.Timeout(x)` applies `x` to connect, read, write and pool. "The read timeout specifies the maximum duration
  to wait for a chunk of data to be received (for example, a chunk of the response body)." So on a stream it is an
  inactivity limit; on a non-streamed call with no keep-alive bytes it has to cover the whole generation.
  `[SDK: ../httpx/_config.py :: Timeout]` (httpx 0.28.1) `[DOC: https://www.python-httpx.org/advanced/timeouts/]`
- The OpenAI SDK has no "must stream" guard for long Chat Completions calls (nothing in `_base_client.py` or
  `resources/chat/completions/completions.py` rejects a long non-streaming request).
  `[SDK: _base_client.py]` There is no background mode in Chat Completions.

### 1.5 What servers do during a long request

| Server | Streaming | Non-streaming |
| --- | --- | --- |
| OpenRouter | "OpenRouter sends periodic SSE comment lines to prevent intermediate proxies and cloud load balancers from timing out": `: OPENROUTER PROCESSING`. "The official OpenAI SDKs parse and ignore SSE comment lines automatically." `[OR: guides/features/compatibility]` `[OR: api_reference/streaming]` | "OpenRouter sends whitespace while a non-streaming model call is in progress, so a response that started before the check finished has status `200` and the error in the body." `[OR: guides/features/plugins/alignment]` (only place this is stated) |
| DeepSeek | "Streaming requests: Continuously return SSE keep-alive comments (`: keep-alive`)" | "Non-streaming requests: Continuously return empty lines". "If the request has not started inference after 10 minutes, the server will close the connection." `[DOC: https://api-docs.deepseek.com/quick_start/rate_limit]` |
| xAI | (not stated) | Its own OpenAI-SDK examples for reasoning models set `timeout=httpx.Timeout(3600.0)`. `[DOC: https://docs.x.ai/developers/model-capabilities/text/reasoning.md]` |
| Together | "Reasoning models produce longer outputs, so you should stream the response". `[DOC: https://docs.together.ai/docs/inference/chat/reasoning.md]` | - |
| vLLM, Ollama, llama.cpp, LM Studio | content, reasoning and tool-call deltas are streamed as generated (chunk examples in the vLLM and LM Studio docs) | no keep-alive mechanism is documented |

Other OpenRouter facts that bear on the choice:

- Cancellation: "Streaming requests can be cancelled by aborting the connection. For supported providers, this
  immediately stops model processing and billing." Supported list includes OpenAI, Azure, Anthropic, Fireworks,
  Together, DeepInfra, xAI, DeepSeek; not supported: AWS Bedrock, Groq, Google, Google AI Studio, Mistral and others.
  "For non-streaming requests or unsupported providers, the model will continue processing and you will be billed
  for the complete response." `[OR: api_reference/streaming]`
- No maximum request duration is documented for OpenRouter Chat Completions. Documented timeout statuses: `408`
  ("Your request timed out"), `error_type: "timeout"` -> `504` ("The provider did not respond within the allowed
  time"), and `524` ("Infrastructure Timeout - Provider request timed out at edge network").
  `[OR: api_reference/errors-and-debugging]` `[OR-API]`
- The SDK's SSE decoder drops lines starting with `:` and leading whitespace before a JSON body is ignored by the
  JSON parser, so both keep-alive forms are harmless. `[SDK: _streaming.py :: SSEDecoder.decode]` `[OFFLINE]`

### 1.6 Recommendation: one function, two transports, one dict out

Default `stream: true`. Reasons: in this mode servers send output as it is generated and OpenRouter and DeepSeek
document keep-alive comments for the silent stretches (so the read timeout works as an inactivity watchdog rather
than a cap on total duration); OpenRouter can stop billing when we drop the connection; and mid-stream provider
errors surface as exceptions. Keep `stream: false` as a config switch for servers whose streaming tool calls are
unreliable, and as the fallback when a streamed replay is rejected (section 5.6, item 3); with it set
`timeout.read` to at least the longest call you accept (3600 s, as xAI's examples do), because only OpenRouter and
DeepSeek document keep-alive bytes for non-streaming requests.

```python
import json, httpx, openai

def call(client, body: dict, *, stream: bool, stream_usage: bool = True, extra_headers: dict | None = None) -> dict:
    """body = the JSON body without transport keys. Returns a dict shaped like a non-streaming chat.completion."""
    std = {k: body[k] for k in ("model", "messages", "tools") if k in body}
    rest = {k: v for k, v in body.items() if k not in std}       # everything else goes through extra_body
    rest.pop("stream", None); rest.pop("stream_options", None)   # the transport decides these, not the config
    if stream:
        if stream_usage:
            rest["stream_options"] = {"include_usage": True}
        it = client.chat.completions.create(**std, stream=True, extra_body=rest, extra_headers=extra_headers)
        return accumulate(chunk.to_dict() for chunk in it)
    raw = client.chat.completions.with_raw_response.create(**std, extra_body=rest, extra_headers=extra_headers)
    return json.loads(raw.text)                                   # leading keep-alive whitespace is fine
```

- `extra_body` keys are merged into the top level of the JSON body and override same-named explicit arguments
  (observed: `max_tokens=100` plus `extra_body={"max_tokens": 7}` sent `"max_tokens": 7`). `[OFFLINE]`
  `[SDK: ../openai-2.53.0.dist-info/METADATA ("Undocumented request params")]` Passing the whole non-standard part of
  the body this way sidesteps the SDK's parameter typing (`max_tokens=None` would be sent as `"max_tokens": null`,
  so never pass `None`; leave keys out). `[OFFLINE]`
- Message dicts are sent exactly as given, including unknown keys (`reasoning_details`, `reasoning_content`,
  `reasoning`) and `"content": null`. `[OFFLINE]`
- `with_raw_response.create(...)` returns a `LegacyAPIResponse` with `.text`, `.headers`, `.status_code`,
  `.retries_taken`, `.parse()`. `[OFFLINE]` `[SDK: ../openai-2.53.0.dist-info/METADATA ("Accessing raw response data (e.g. headers)")]`
  Use it rather than the parsed object because, for a `200` whose body is not a normal completion, the SDK does not
  raise: an error-only body becomes a `ChatCompletion` with `choices=None` and the error in `model_extra`; a
  non-JSON body with a non-JSON content type comes back as a plain `str`; a whitespace-only body raises a bare
  `json.JSONDecodeError`. `[OFFLINE]`
- `stream_options: {"include_usage": true}` is the canonical way to get usage in a stream ("an additional chunk
  will be streamed before the `data: [DONE]` message ... the `choices` field will always be an empty array").
  `[SDK: types/chat/chat_completion_stream_options_param.py]` Accepted by vLLM, Ollama, DeepSeek, Fireworks, xAI;
  on OpenRouter it "has no effect. Full usage details are always included." `[OR-API: ChatStreamOptions]`
  vLLM and DeepSeek reject `stream_options` when `stream` is not true, so never send it on the non-streaming path.
  `[SRC: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/entrypoints/openai/chat_completion/protocol.py]`
  `[DOC: https://api-docs.deepseek.com/api/create-chat-completion]` For OpenRouter set `stream_usage: false`: the
  option does nothing there, and whether an inert parameter interacts with `provider.require_parameters` is not
  documented. Not documented for llama.cpp or LM Studio:
  give the config a way to omit it.

The accumulator (this exact code was re-run offline by the independent verification pass: through the real SDK
stream parser on the 10.6 fixture, and directly on the chunk variants listed at the end of 10.6; the index-less
tool-call rule and the id-collision guard were added in that pass):

```python
import json

TEXT_KEYS = ("content", "refusal", "reasoning", "reasoning_content")

def _merge_reasoning_details(acc: list, deltas: list) -> None:
    # Merge by *type transition*, never by `index` (rule taken from OpenRouter's own AI SDK provider, see below).
    for d in deltas:
        d = dict(d)
        last = acc[-1] if acc else None
        t = d.get("type")
        if t == "reasoning.text" and last and last.get("type") == "reasoning.text" and not last.get("signature"):
            last["text"] = (last.get("text") or "") + (d.get("text") or "")
            if d.get("signature"):
                last["signature"] = d["signature"]
            if not last.get("format") and d.get("format"):
                last["format"] = d["format"]
        elif t == "reasoning.summary" and last and last.get("type") == "reasoning.summary":
            last["summary"] = (last.get("summary") or "") + (d.get("summary") or "")
            if not last.get("format") and d.get("format"):
                last["format"] = d["format"]
        else:
            acc.append(d)          # new block; encrypted blocks are opaque and never merged

def _complete_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except ValueError:
        return False

def accumulate(chunks) -> dict:
    """chunks: iterable of dicts (chunk.to_dict()). Returns a dict shaped like a non-streaming chat.completion."""
    msg: dict = {"role": "assistant", "content": None}
    tool_calls: dict = {}; order: list = []; last_key = None
    out: dict = {"object": "chat.completion", "choices": [], "usage": None}
    finish = None; native = None; n = 0
    for ch in chunks:
        n += 1
        for k in ("id", "model", "created", "provider", "system_fingerprint"):
            if ch.get(k) is not None and k not in out:
                out[k] = ch[k]
        if ch.get("usage"):
            out["usage"] = ch["usage"]                      # last non-null wins
        for choice in ch.get("choices") or []:              # may be [] on a usage-only chunk
            if choice.get("index", 0) != 0:
                continue
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            if choice.get("native_finish_reason"):
                native = choice["native_finish_reason"]
            delta = choice.get("delta") or {}
            for k in TEXT_KEYS:
                v = delta.get(k)
                if isinstance(v, str) and v != "":
                    msg[k] = (msg.get(k) or "") + v
                elif isinstance(v, list):                   # Mistral-native thinking chunks: not supported here
                    raise NotImplementedError("list-valued delta." + k + "; use stream=false for this server")
            if delta.get("reasoning_details"):
                _merge_reasoning_details(msg.setdefault("reasoning_details", []), delta["reasoning_details"])
            for tc in delta.get("tool_calls") or []:
                fn = tc.get("function") or {}
                idx = tc.get("index")
                if idx is not None:
                    key = idx
                else:                                       # no index (out of spec, but seen): does it continue the latest call?
                    last = tool_calls.get(last_key) if last_key is not None else None
                    if last is None:
                        key = None
                    elif tc.get("id"):
                        key = last_key if tc["id"] == last["id"] else None
                    elif fn.get("name") and (fn["name"] != last["function"]["name"] or _complete_json(last["function"]["arguments"])):
                        key = None                          # a new name, or the same name after a finished call => new call
                    else:
                        key = last_key
                slot = tool_calls.get(key) if key is not None else None
                if slot is not None and tc.get("id") and slot["id"] and tc["id"] != slot["id"]:
                    slot = None                             # same index, different id => a new call reusing the index
                if slot is None:
                    key = ("slot", len(order))              # unique internal key, preserves arrival order
                    slot = {"id": None, "type": "function", "function": {"name": None, "arguments": ""}}
                    tool_calls[key] = slot; order.append(slot)
                    if idx is not None:
                        tool_calls[idx] = slot              # later deltas at this index continue this call
                last_key = key
                if tc.get("id") and not slot["id"]:
                    slot["id"] = tc["id"]                   # set once; some servers repeat it on every chunk
                if fn.get("name") and not slot["function"]["name"]:
                    slot["function"]["name"] = fn["name"]   # set once
                a = fn.get("arguments")
                if isinstance(a, str):
                    slot["function"]["arguments"] += a
                elif a is not None:                         # object instead of string fragment
                    slot["function"]["arguments"] = json.dumps(a)
    if n == 0:
        raise RuntimeError("stream produced no chunks (non-SSE body or empty stream)")
    if finish is None:
        raise RuntimeError("stream ended without a finish_reason (truncated)")
    if order:
        taken = {tc["id"] for tc in order if tc["id"]}
        seen = set()
        for i, tc in enumerate(order):                      # ids must be non-empty and unique
            if not tc["id"] or tc["id"] in seen:
                new_id = f"call_{i}"
                while new_id in taken:                      # never collide with an id the server did send
                    new_id += "_"
                tc["id"] = new_id
                taken.add(new_id)
            seen.add(tc["id"])
        msg["tool_calls"] = order
    out["choices"] = [{"index": 0, "message": msg, "finish_reason": finish,
                       **({"native_finish_reason": native} if native else {})}]
    return out
```

Why each rule exists:

- `reasoning_details` merge. OpenRouter: in streaming, `reasoning_details` "appears in
  `choices[].delta.reasoning_details` for each chunk"; "The `reasoning_details` array in each chunk may contain one
  or more reasoning objects"; "The complete reasoning sequence is built by concatenating all chunks in order."
  `[OR: guides/best-practices/reasoning-tokens]` The docs stop there; the concrete algorithm above is the one in
  OpenRouter's own AI SDK provider, whose comments say: "Providers (e.g. OpenAI `openai-responses-v1`) stream a
  logical reasoning block as many small deltas that all carry the same `index` (frequently `index: 0`), so `index`
  cannot be used to reassemble the array - reassembly is driven by the `type` transition instead"; "A signature
  closes the text detail it signs: once the last text detail is signed, the next text delta starts a new block";
  "Encrypted details are opaque blobs (each has its own id); they are discrete and must be pushed as-is, never
  merged"; and a signature can arrive as "a signature-only delta" after the text.
  `[SRC: https://raw.githubusercontent.com/OpenRouterTeam/ai-sdk-provider/main/src/chat/index.ts]`
- Tool-call slots. Canonical deltas carry `index` (required), and `id`, `type`, `function.name`,
  `function.arguments` as optional fragments. `[SDK: types/chat/chat_completion_chunk.py :: ChoiceDeltaToolCall]`
  OpenRouter "returns all information except the arguments in the first chunk"; "Some providers return empty, null,
  or duplicate IDs for parallel tool calls"; a delta at a known index with a different non-empty id "is a new tool
  call reusing the index"; a delta may have no index at all ("No index (allowed by the schema): a delta with a new
  id or, without an id, a new function name starts a new tool call; anything else continues the most recent one";
  a delta that repeats the latest call's id continues it; a repeated name alone only starts a separate call when the
  previous call's arguments are already a complete JSON value). `accumulate()` follows these rules.
  `[SRC: .../ai-sdk-provider/main/src/chat/index.ts]` `[OFFLINE]` (OpenRouter's own schema marks `index` as required
  on `ChatStreamToolCall`, so the index-less branch is for other servers. `[OR-API: ChatStreamToolCall]`)
  xAI: "With streaming, the function call is returned in whole in a single chunk, not streamed across chunks."
  `[DOC: https://docs.x.ai/developers/tools/function-calling.md]` LM Studio's doc shows `id` and `name` only on the
  first chunk and argument fragments after. `[DOC: https://lmstudio.ai/llms.txt]`
- Usage. OpenAI style: a final chunk with `choices: []` and `usage`. OpenRouter: "the usage chunk contains one
  choice with a content-free `delta` that repeats the `finish_reason` (and `native_finish_reason`) of the stream"
  so "the terminal `finish_reason` appears twice". `[OR: api_reference/streaming]` DeepSeek: "no separate
  usage-only chunk is emitted: the statistics ride on the last content chunk".
  `[DOC: https://api-docs.deepseek.com/api/create-chat-completion]` "Last non-null wins" covers all three.
- The two `RuntimeError`s. A stream that ends cleanly without `[DONE]` or a `finish_reason` raises nothing in the
  SDK, and a `200` JSON (non-SSE) reply to a `stream=True` request yields zero chunks, silently. `[OFFLINE]`
  Map both to a retryable error.

### 1.7 Do NOT use the SDK stream helper

`client.chat.completions.stream(...)` / `ChatCompletionStream.get_final_completion()` /
`ChatCompletionStreamState` accumulate with `accumulate_delta`, which string-concatenates every repeated string
key except `index` and `type`, and requires an integer `index` on every dict inside a list delta.
`[SDK: lib/streaming/_deltas.py :: accumulate_delta]` Fed an OpenRouter-style stream (every delta carries
`"role": "assistant"`, reasoning details repeat `id` and `format`) the helper returned
`role == "assistantassistantassistant..."`, `reasoning_details[0].id == "r1r1r1"`,
`format == "anthropic-claude-v1anthropic-claude-v1anthropic-claude-v1"`; with `index` missing from a reasoning detail
it raised `RuntimeError: Expected list delta entry to have an `index` key`. `[OFFLINE]` OpenRouter documents `index`
on reasoning details as optional. `[OR: guides/best-practices/reasoning-tokens]`

---

## 2. Request shape for a contestant call

### 2.1 The body

```json
{
  "model": "anthropic/claude-fable-5.2",
  "messages": [
    {"role": "system", "content": "<system prompt text>"},
    {"role": "user", "content": "<task message text>"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "write_rows",
        "description": "Store palette colours and picture rows. It checks format only. A stored row can be replaced by sending it again.",
        "parameters": {
          "type": "object",
          "properties": {
            "palette": {
              "type": "array",
              "items": {"type": "string"},
              "description": "Optional. Colours to define, each written as a one-character code then red, green and blue from 0 to 255, for example 'k 20 40 90'."
            },
            "start_row": {
              "type": "integer",
              "description": "The row number of the first row in 'rows'. Row 1 is the top."
            },
            "rows": {
              "type": "array",
              "items": {"type": "string"},
              "description": "Consecutive rows, starting at start_row. Spaces are ignored."
            }
          }
        }
      }
    }
  ],
  "max_tokens": 128000,
  "stream": true,
  "reasoning": {"effort": "max"},
  "provider": {"require_parameters": true}
}
```

`reasoning` and `provider` are OpenRouter extensions (sections 5 and 2.8); they are examples of what the config's
`extra_body` carries. Everything above them is canonical. `stream` (and, for servers other than OpenRouter,
`stream_options: {"include_usage": true}`) are transport keys: the `call()` function in 1.6 adds them, the request
builder should not.

- `messages`: required, at least one. Roles: `system`, `developer`, `user`, `assistant`, `tool` (plus deprecated
  `function`). `[SDK: types/chat/chat_completion_message_param.py]` `[OR-API: ChatMessages]`
- `tools`: list of `{"type": "function", "function": {"name": str (required; a-z, A-Z, 0-9, underscores, dashes,
  max 64), "description": str, "parameters": <JSON Schema object>, "strict": bool|null}}`.
  `[SDK: types/chat/chat_completion_function_tool_param.py]` `[SDK: types/shared_params/function_definition.py]`
  `[OR-API: ChatFunctionTool]`
- OpenRouter: "the `tools` parameter must be included in every request (Steps 1 and 3) so the router can validate
  the tool schema on each call." `[OR: guides/features/tool-calling]` Send the identical `tools` array on every call
  of a conversation (also required by prefix-bound reasoning, section 4.6).

### 2.2 System prompt

Send it as the first message with `role: "system"` and a string `content`. `[SDK:
types/chat/chat_completion_system_message_param.py]` The canonical API also has `role: "developer"` ("With o1 models
and newer, `developer` messages replace the previous `system` messages"), and OpenRouter accepts both; the vLLM, Ollama
and llama.cpp pages read here never mention a `developer` role. `[SDK: types/chat/chat_completion_developer_message_param.py]`
`[OR-API: ChatSystemMessage, ChatDeveloperMessage]` OpenRouter: "system messages should contain text only".
`[OR: guides/overview/multimodal/image-understanding]`

### 2.3 Tool choice

Omit `tool_choice`. Canonical default: "`none` is the default when no tools are present. `auto` is the default if
tools are present." `[SDK: types/chat/completion_create_params.py :: CompletionCreateParamsBase.tool_choice]`
Reasons not to send it at all, even as `"auto"`:

- Ollama lists `tool_choice` as an unsupported request field. `[DOC: .../ollama/main/docs/api/openai-compatibility.mdx]`
- Forcing is rejected by current Claude models through OpenRouter: Chat Completions `tool_choice: "required"` or a
  named function "return the provider's 400 (`tool_choice: type "tool" and "any" are not supported for this
  model.`)" on Opus 5.5 / Sonnet 5.5, and on Fable 5.1 and later. `[DOC:
  https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/claude-5-5]` `[DOC:
  https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/fable-5-1]`
- vLLM needs server flags `--enable-auto-tool-choice` and `--tool-call-parser <name>` for any tool calling.
  `[DOC: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/tool_calling.md]`

Values, for reference: `"none"`, `"auto"`, `"required"`, `{"type":"function","function":{"name":"write_rows"}}`.
`[SDK: types/chat/chat_completion_tool_choice_option_param.py]` `[OR-API: ChatToolChoice]`

### 2.4 Parallel tool calls

`parallel_tool_calls: bool`. Leave it out.

- Canonical default `true`. `[DOC: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create.md]`
- OpenRouter: default "true for most models"; `false` means "the model will only request one tool call at a time".
  `[OR: guides/features/tool-calling]` `[OR: api_reference/parameters]`
- vLLM: `false` "ensures vLLM only returns zero or one tool call per request"; default `true`.
  `[DOC: .../vllm/main/docs/serving/online_serving/openai_compatible_server.md]`
- llama.cpp: "Multiple/parallel tool calling is supported on some models but disabled by default, enable it by
  passing `"parallel_tool_calls": true`". `[DOC: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/function-calling.md]`
- Only 18 of the 387 chat-capable OpenRouter catalog models list `parallel_tool_calls` in `supported_parameters`
  (none of the current Claude, GPT-6, Gemini 4 or Grok entries do), so sending it together with
  `provider.require_parameters: true` can leave no eligible provider. `[OR-CAT]`

The adapter must handle several calls in one reply regardless (section 4.4).

### 2.5 Strict schemas: avoid

- Canonical strict mode (`function.strict: true`) requires "`additionalProperties` must be set to `false` for each
  object in the `parameters`" and "All fields in `properties` must be marked as `required`"; a non-conforming schema "will be rejected".
  "Chat Completions requests remain non-strict by default."
  `[DOC: https://developers.openai.com/api/docs/guides/function-calling.md]` Our tool has an optional `palette` and
  no `required` list, so it is not strict-compatible as given, and the harness must send the schema unchanged.
- OpenRouter with Anthropic: "For strict tool use (`strict: true` on tools), you must explicitly pass the
  `structured-outputs-2025-11-13` header [`x-anthropic-beta`]. Without this header, OpenRouter will strip the
  `strict` field and route normally." `[OR: guides/routing/provider-selection]`
- vLLM: with `tool_choice="auto"` and no `strict`, "the model generates tool calls without any grammar and malformed
  markup can leak into the response" unless the operator starts the server with `--tool-strict-level function`.
  `[DOC: .../vllm/main/docs/features/tool_calling.md]` That is a server-launch choice, not something to send.
- So: do not add the `strict` key at all. An absent key is the canonical non-strict default on Chat Completions
  (`"strict": false` is also valid canonically, but it is one more key for a compat server to trip over).

### 2.6 Output-limit parameter (`max_tokens_param` switch)

| Server | `max_tokens` | `max_completion_tokens` | Source |
| --- | --- | --- | --- |
| OpenAI (canonical) | "deprecated in favor of `max_completion_tokens`, and is not compatible with o-series models" | "upper bound ... including visible output tokens and reasoning tokens" | `[SDK: types/chat/completion_create_params.py]` |
| OpenRouter | accepted (OpenAPI text: "Maximum tokens (deprecated, use max_completion_tokens). Note: some providers enforce a minimum of 16."); listed in `supported_parameters` of 370 of the 387 chat-capable catalog models | accepted, "shares `max_tokens` semantics"; listed for only 94 of 387 (mostly OpenAI and Anthropic entries; not Gemini, Grok, Mistral, Meta) | `[OR-API: ChatRequest]` `[OR: api_reference/parameters]` `[OR-CAT]` |
| vLLM | accepted, marked deprecated | accepted, takes precedence | `[SRC: .../vllm/entrypoints/openai/chat_completion/protocol.py :: ChatCompletionRequest]` |
| Ollama | accepted | not in the supported list and not a field of the request struct, so it has no effect | `[DOC: .../ollama/main/docs/api/openai-compatibility.mdx]` `[SRC: https://raw.githubusercontent.com/ollama/ollama/main/openai/openai.go :: ChatCompletionRequest]` |
| llama.cpp | used in README examples | not mentioned in README | `[DOC: .../llama.cpp/master/tools/server/README.md]` |
| LM Studio | listed under "Supported payload parameters" | not listed | `[DOC: https://lmstudio.ai/llms.txt]` |
| DeepSeek | the documented parameter (1 to 393216; default 8K non-thinking, 64K thinking, 128K at `reasoning_effort: max`) | not documented | `[DOC: https://api-docs.deepseek.com/api/create-chat-completion]` |
| Groq | "Deprecated in favor of `max_completion_tokens`" | preferred | `[DOC: https://console.groq.com/docs/api-reference.md]` |
| Fireworks | accepted | "Alias for max_tokens. Cannot be specified together with max_tokens." | `[DOC: https://docs.fireworks.ai/api-reference/post-chatcompletions.md]` |
| Together | documented | not documented | `[DOC: https://docs.together.ai/reference/chat-completions-1.md]` |
| xAI | "[DEPRECATED]" | preferred; "only applies to visible output tokens (i.e. does not apply to tokens used for reasoning or function calls). Defaults to 128,000 when unset" | `[DOC: https://docs.x.ai/developers/rest-api-reference/inference/chat-completions.md]` |

Recommendation: config key `max_tokens_param` with values `"max_tokens"` (default), `"max_completion_tokens"`,
`null` (send neither). Never send both (Fireworks rejects it). Default `"max_tokens"` because it is the only name
Ollama, LM Studio, DeepSeek and Together document, and on OpenRouter it is the one almost every model lists.

### 2.7 Never send

`temperature`, `top_p`, `top_k`, penalties, `seed`, `stop`, `n`, `logprobs`, `response_format`, `user`,
`name` on any message. Reasons beyond harness policy: Claude 4.7 and later ignore or reject sampling parameters
through OpenRouter; xAI reasoning models error on `presencePenalty`, `frequencyPenalty`, `stop`; Groq returns 400
for `logprobs`, `logit_bias`, `top_logprobs`, `messages[].name`, and `n != 1`. `[DOC:
https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/claude-4-7]` `[DOC:
https://docs.x.ai/developers/model-capabilities/text/reasoning.md]` `[DOC: https://console.groq.com/docs/openai.md]`

### 2.8 OpenRouter-only body fields worth knowing (all go through `extra_body`)

- `provider` (object, `additionalProperties: false`): `order` (list of provider slugs), `only`, `ignore`,
  `allow_fallbacks` (default `true`), `require_parameters` (default `false`), `data_collection` (`"allow"`|`"deny"`),
  `zdr`, `quantizations`, `sort` (`"price"`|`"throughput"`|`"latency"`|`"exacto"` or `{by, partition}`), `max_price`,
  `preferred_min_throughput`, `preferred_max_latency`, `enforce_distillable_text`. `[OR-API: ProviderPreferences]`
  `[OR: guides/routing/provider-selection]`
  - `require_parameters`: by default "providers that don't support all the LLM parameters specified in your request
    can still receive the request, but will ignore unknown parameters. When you set `require_parameters` to `true`,
    the request won't even be routed to that provider." Even when it is false there is a soft preference for `tools`, the
    response format and `verbosity` ("the request is only routed to the supporting providers"), and setting
    `max_tokens` restricts routing to providers that allow that length; nothing comparable is documented for
    `reasoning`. Recommended for contestants: `{"require_parameters": true}`, and send
    only parameters the model lists in `supported_parameters` (`tools`, `max_tokens`, `reasoning`).
  - Any tool-calling request is also re-ordered by "Auto Exacto" (tool-call success rate, throughput, benchmarks);
    setting `provider.order` disables sticky routing. `[OR: guides/routing/auto-exacto]`
    `[OR: guides/best-practices/prompt-caching]`
- `models` (list of fallback model ids), `route` (deprecated). Do not use: a benchmark must not silently switch
  models. `[OR-API: ChatRequest]` The response's top-level `model` says which model answered. `[OR: api_reference/overview]`
- `session_id` (string, max 256): sticky routing key and log grouping; "routing all requests in the session to the
  same provider to maximize prompt cache hits". Useful: one id per attempt. `[OR-API: ChatRequest]`
- `cache_control: {"type": "ephemeral"}` or `{"type": "ephemeral", "ttl": "1h"}` at the top level: Anthropic
  "automatic caching", breakpoint "advances ... as conversations grow". Default TTL 5 minutes; cache writes cost
  1.25x input (2x for 1h), reads 0.1x. (By arithmetic: with 10-30 minutes between calls a 5-minute cache has
  expired by the next turn; only the 1h TTL can hit.)
  `[OR: guides/best-practices/prompt-caching]`
- `plugins`, `debug`, `service_tier`, `trace`, `metadata`, `modalities`, `prediction`: not needed. `[OR-API: ChatRequest]`
- "If the chosen model doesn't support a request parameter ..., then the parameter is ignored. The rest are
  forwarded to the underlying model API." `[OR: api_reference/overview]`

---

## 3. Response shape

### 3.1 Non-streaming body

```json
{
  "id": "gen-1795000000-AbCdEfGhIjKlMnOpQrSt",
  "object": "chat.completion",
  "created": 1795000000,
  "model": "anthropic/claude-fable-5.2",
  "system_fingerprint": null,
  "choices": [
    {
      "index": 0,
      "finish_reason": "tool_calls",
      "native_finish_reason": "tool_use",
      "message": {
        "role": "assistant",
        "content": null,
        "refusal": null,
        "reasoning": "<plain-text reasoning or summary>",
        "reasoning_details": [ {"type": "reasoning.text", "text": "...", "signature": "...", "id": "...", "format": "anthropic-claude-v1", "index": 0} ],
        "tool_calls": [
          {"id": "call_abc123", "type": "function", "function": {"name": "write_rows", "arguments": "{\"start_row\": 1, \"rows\": [\"kkkk\"]}"}}
        ]
      }
    }
  ],
  "usage": {"prompt_tokens": 812, "completion_tokens": 2304, "total_tokens": 3116}
}
```

- Canonical fields: `id`, `object: "chat.completion"`, `created`, `model`, `choices[]` (`index`, `finish_reason`,
  `logprobs`, `message`), `usage`, `system_fingerprint`, `service_tier`. `[SDK: types/chat/chat_completion.py]`
- Canonical `message`: `role: "assistant"`, `content: str|null`, `refusal: str|null`, `tool_calls: list|null`,
  `annotations`, `audio`, `function_call` (deprecated). `[SDK: types/chat/chat_completion_message.py]`
- OpenRouter additions:
  - on the choice: `native_finish_reason: string | null` ("The raw finish_reason string returned by the model") and
    optionally `error: {code, message, metadata?}`. Both are documented **only** in the TypeScript types and prose
    of `[OR: api_reference/overview]` (`NonStreamingChoice`, `StreamingChoice`) and in the errors page; the OpenAPI
    schemas `ChatChoice` / `ChatStreamChoice` list just `finish_reason`, `index`, `logprobs`, `message`/`delta`.
    Treat both keys as optional (`choice.get(...)`), never as required.
  - on the message: `reasoning: str|null`, `reasoning_details: list`, `images`, `model`, `audio`.
    `[OR-API: ChatAssistantMessage]`
  - on `usage`: `cost`, `is_byok`, `cost_details`, `server_tool_use_details`, `deferred_tools_details`.
    `[OR-API: ChatUsage]`
  - top level, optional: `openrouter_metadata`, `service_tier`, `alignment`, `safeguard_results`. `[OR-API: ChatResult]`
  - `ChatResult` marks `id`, `choices`, `created`, `model`, `object`, `system_fingerprint` as required, yet the
    documented error-only `200` body has none of them but `id` (3.7): do not validate a 200 body against the schema.
- Other servers put reasoning in `message.reasoning_content` (DeepSeek, llama.cpp, Fireworks, xAI, LM Studio when
  enabled, some Together models) or `message.reasoning` (vLLM, Ollama, Groq with `reasoning_format: parsed`, some
  Together models); table in section 5.
- "`choices` is always an array, even if the model only returns one completion." `[OR: api_reference/overview]`
  Read `choices[0]` only; never send `n`.

### 3.2 Text

`choices[0].message.content`:

- `str`, or `null` (normal when the reply is only tool calls), or `""`. `[SDK: types/chat/chat_completion_message.py]`
- OpenRouter's schema also allows an array of content parts. `[OR-API: ChatAssistantMessage]`
- Mistral's native endpoint returns a **list of chunks** when reasoning is on:
  `[{"type": "thinking", "thinking": [{"type": "text", "text": "..."}]}, {"type": "text", "text": "..."}]`; for
  `zai-glm-5-3` there "is no plain-string mode". `[DOC: https://docs.mistral.ai/llms-full.txt]` (section "reasoning")
- Rule: `text = content if str; "".join(p["text"] for p in content if p.get("type") == "text") if list; "" if None`.
  The SDK passes a list through untouched: parsing raises nothing and warns nothing; a pydantic serializer
  `UserWarning` appears only when the parsed object is serialised again (`to_dict()` / `model_dump()`), which the
  streaming path of `call()` does per chunk. `json.loads(raw.text)` on the non-streaming path never warns. `[OFFLINE]`
- Some servers leave reasoning inside `content` as `<think>...</think>` (Groq `reasoning_format: raw`, llama.cpp
  `--reasoning-format none`, DeepSeek-R1 on Together). `[DOC: https://console.groq.com/docs/reasoning.md]`
  `[DOC: .../llama.cpp/master/tools/server/README.md]` `[DOC: https://docs.together.ai/docs/inference/chat/reasoning.md]`
  Do not strip anything; record what arrived.

### 3.3 Tool calls

`choices[0].message.tool_calls[i]` = `{"id": str, "type": "function", "function": {"name": str, "arguments": str}}`.
`arguments` is a **JSON-encoded string**, not an object. `[SDK:
types/chat/chat_completion_message_function_tool_call.py]` `[OR-API: ChatToolCall]` ("Function arguments as JSON
string") Extra keys can appear on a call (DeepSeek's sample output shows `index=0`). `[DOC:
https://api-docs.deepseek.com/guides/thinking_mode]`

Do not trust it:

- Canonical docstring: "the model does not always generate valid JSON, and may hallucinate parameters not defined by
  your function schema. Validate the arguments in your code before calling your function." `[SDK: same file]`
- OpenRouter measures exactly these failures per provider: `InvalidJson` ("`JSON.parse(arguments)` throws"),
  `UnknownName`, `MissingName`, `SchemaMismatch`, and `NameLeakedInContent`. `[OR: guides/routing/auto-exacto]`
- vLLM (see 2.5): malformed markup can leak into `content`; with its `hf` parser, in streaming "a call that does
  not parse is dropped". `[DOC: .../vllm/main/docs/features/tool_calling.md]`
- LM Studio: models not trained for tools "may output improperly formatted tool calls, resulting in LM Studio being
  unable to parse them into the `tool_calls` field" (the text stays in `content`). `[DOC: https://lmstudio.ai/llms.txt]`
- The SDK does not validate: an `arguments` object arrives as a `dict`, a missing `id` as `None`. `[OFFLINE]`

Parsing rule for the adapter (the harness wants the parsed value, or the raw string when it is not valid JSON):

```python
def parse_args(raw):
    if isinstance(raw, (dict, list)):      # non-conforming server sent an object
        return raw
    if raw is None or raw == "":           # no-argument call
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw                         # harness rejects it with a format message
```

Ids: opaque strings; if empty, null or duplicated within one reply, substitute a unique id and use the same id in
the replayed assistant message and in the tool message (see 1.6).

### 3.4 Finish reasons

`choices[0].finish_reason`. Decide "did the model call tools" from `message.tool_calls` being non-empty, **not**
from this field.

| Value | Meaning | Where | Map to |
| --- | --- | --- | --- |
| `stop` | natural end or a stop sequence | canonical | `end` (or `tool_calls` if calls are present) |
| `tool_calls` | the model called tools | canonical | `tool_calls` |
| `length` | output limit reached ("the maximum number of tokens specified in the request was reached"); may come with empty `content` when reasoning used the whole budget; may come with a half-written tool call | canonical | `max_tokens` |
| `content_filter` | "content was omitted due to a flag from our content filters"; OpenRouter also uses it for a model refusal, together with `message.refusal` | canonical, OpenRouter | `refusal` |
| `function_call` | deprecated form of `tool_calls` | canonical | `tool_calls` |
| `error` | provider failed during generation; `choices[0].error` or top-level `error` carries details | OpenRouter | raise (section 9) |
| `insufficient_system_resource` | "the request is interrupted due to insufficient resource of the inference system" | DeepSeek | transient error |
| `aborted` | "the generation was interrupted" | DeepSeek | transient error |
| `end_turn` | xAI's reference: "`"end_turn"` or `null` in streaming mode when the chunk is not the last" (so it may show up on non-final stream chunks; the final value still decides) | xAI | `end` if it is the final value |
| `null` | not finished (every non-final streaming chunk). OpenRouter's types also allow `null` on a non-streaming choice, and its zero-completion insurance speaks of "a blank/null finish reason" | all | `other` |
| anything else | vLLM types the field as an arbitrary string | - | `other`, keep raw |

Sources: `[SDK: types/chat/chat_completion.py :: Choice.finish_reason]`; OpenRouter "normalizes each model's
`finish_reason` to one of the following values: `tool_calls`, `stop`, `length`, `content_filter`, `error`"
`[OR: api_reference/overview]` `[OR-API: ChatFinishReasonEnum]`; DeepSeek `[DOC:
https://api-docs.deepseek.com/api/create-chat-completion]`; xAI `[DOC:
https://docs.x.ai/developers/rest-api-reference/inference/chat-completions.md]`; vLLM `[SRC:
.../vllm/entrypoints/openai/chat_completion/protocol.py :: ChatCompletionResponseChoice]`. The SDK does not validate
the value; `"error"` came through as-is. `[OFFLINE]`

Known mismatches between `finish_reason` and content:

- `stop` with tool calls present. OpenRouter's own SDK overrides it: "when there are tool calls with encrypted
  reasoning (thoughtSignature), the model returns 'stop' but expects continuation", and it infers tool calls when the
  finish reason is unknown or missing. `[SRC: .../ai-sdk-provider/main/src/chat/index.ts]` Ollama rewrites `stop`
  to `tool_calls` itself when calls are present. `[SRC: .../ollama/main/openai/openai.go]`
- `length` with a tool call whose arguments are cut off. OpenRouter's SDK forwards, on `length` / `error` /
  `content_filter`, "only ... tool calls whose arguments are complete, parsable JSON". `[SRC: same]`
- Gemini 4 on OpenRouter: `finish_reason: "length"` with `native_finish_reason: "CONTINUATION"` or `"MAX_TOKENS"`
  is a resumable stop (section 6.3).

### 3.5 Refusals and safety blocks

Four different signals:

1. Refusal written by the model as output (HTTP 200): `message.refusal` is a non-empty string. Canonical: "The
   refusal message generated by the model." `[SDK: types/chat/chat_completion_message.py]` OpenRouter: "Chat
   Completions returns `message.refusal` with `finish_reason: "content_filter"`".
   `[OR: api_reference/errors-and-debugging]`
2. Filter stop: `finish_reason == "content_filter"` with empty or partial `content`.
3. Provider-reported refusal or policy block delivered as an **error**: `403` with
   `error.metadata.error_type` = `"refusal"` ("The provider reported a model refusal as an error rather than as
   output") or `"content_policy_violation"` (a filter around the model, "such as Gemini's `SAFETY` block");
   `metadata.provider_code` e.g. `bio_policy`, `cyber_policy` for GPT-6. It can arrive as the HTTP status, or after a
   `200` was committed "in which case it is the `code` inside the error body". "Retrying the same input returns the
   same block." `[OR: api_reference/errors-and-debugging]` `[DOC:
   https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/gpt-6]`
4. OpenRouter-side blocks before any provider: moderation flag (`403`, `error.metadata` =
   `{reasons: string[], flagged_input: string, provider_name, model_slug}`) and guardrails (`403`,
   "Request blocked: ..."). `[OR: api_reference/errors-and-debugging]` The catalog marks first-party Anthropic and
   OpenAI endpoints `top_provider.is_moderated: true`. `[OR-CAT]`

Adapter mapping: 1 and 2 -> `Reply(stop_reason="refusal", refusal_text=message.refusal or content)`. 3 and 4 ->
also a refusal `Reply` (`refusal_text = error.message`), not a retryable error.

### 3.6 Usage

`usage` (may be `null`/absent, e.g. on an error body or a stream without the usage chunk):

| Field | Meaning | Source |
| --- | --- | --- |
| `prompt_tokens` | input tokens; OpenRouter: "Including images, input audio, and tools if any" | `[SDK: types/completion_usage.py]` `[OR: api_reference/overview]` |
| `completion_tokens` | output tokens | same |
| `total_tokens` | canonical: prompt + completion | same |
| `prompt_tokens_details.cached_tokens` | tokens read from cache; a **subset** of `prompt_tokens` (OpenRouter example: `prompt_tokens: 10339`, `cached_tokens: 10318`) | `[SDK]` `[OR: guides/best-practices/prompt-caching]` |
| `prompt_tokens_details.cache_write_tokens` | tokens written to cache | `[SDK]` `[OR: api_reference/overview]` |
| `prompt_tokens_details.audio_tokens`, `.video_tokens` | not relevant | `[OR-API: ChatUsage]` |
| `completion_tokens_details.reasoning_tokens` | reasoning/thinking tokens | `[SDK]` `[OR-API: ChatUsage]` |
| `completion_tokens_details.accepted_prediction_tokens`, `.rejected_prediction_tokens`, `.audio_tokens` | not relevant | `[SDK]` `[OR-API: ChatUsage]` |
| `completion_tokens_details.image_tokens` | not relevant (only in OpenRouter's TypeScript type; not in the SDK type or the OpenAPI schema) | `[OR: api_reference/overview]` |
| `cost` | OpenRouter: "The total amount charged to your account"; "OpenRouter uses a credit system where the base currency is US dollars" | `[DOC: https://openrouter.ai/docs/cookbook/administration/usage-accounting]` `[OR: faq]` |
| `is_byok`, `cost_details.{upstream_inference_cost, upstream_inference_prompt_cost, upstream_inference_completions_cost}` | OpenRouter | `[OR: api_reference/overview]` |
| `prompt_cache_hit_tokens`, `prompt_cache_miss_tokens` (top level of `usage`) | DeepSeek; `prompt_tokens` = hit + miss | `[DOC: https://api-docs.deepseek.com/api/create-chat-completion]` |
| top-level `timings` (`prompt_n`, `predicted_n`, ...) | llama.cpp, Ollama extra | `[DOC: .../llama.cpp/master/tools/server/README.md]` `[SRC: .../ollama/main/openai/openai.go]` |

Are reasoning tokens inside `completion_tokens`?

- Canonical and OpenRouter: **included**. OpenAI: rejected prediction tokens, "like reasoning tokens, ... are still
  counted in the total completion tokens for purposes of billing, output, and context window limits". OpenRouter:
  "subtract `usage.completion_tokens_details.reasoning_tokens` from `usage.completion_tokens`" to get visible
  tokens; example `max_tokens: 300` -> `completion_tokens: 302`, `reasoning_tokens: 301`. "Reasoning tokens are
  considered output tokens and charged accordingly." `[SDK: types/completion_usage.py]`
  `[OR: guides/best-practices/reasoning-tokens]`
- xAI direct: **additional**. Its reference example has `prompt_tokens: 32`, `completion_tokens: 9`,
  `reasoning_tokens: 94`, `total_tokens: 135` (= 32 + 9 + 94). `[DOC:
  https://docs.x.ai/developers/rest-api-reference/inference/chat-completions.md]`
- Generic rule for `Usage.output_tokens`: if `prompt + completion == total` use `completion_tokens`; else if
  `prompt + completion + reasoning == total` use `completion + reasoning`; if `total_tokens` is missing use
  `completion_tokens`. Keep the raw `usage` object in `Reply.raw`.
- Summarised reasoning: on Claude through OpenRouter the visible reasoning text is a summary and "may contain fewer
  tokens than the reasoning token count reported in `usage`". `[OR: guides/best-practices/reasoning-tokens]`
- Ollama's `usage` has no reasoning breakdown; vLLM reports `completion_tokens_details.reasoning_tokens` when a
  reasoning parser is configured. `[SRC: .../ollama/main/openai/openai.go :: Usage]` `[DOC:
  .../vllm/main/docs/serving/online_serving/openai_compatible_server.md]`

OpenRouter: "Token counts are calculated using the model's native tokenizer"; usage is "always returned for
non-streaming" and once in the last chunk when streaming; the response header `X-Generation-Id` and the body `id`
(`gen-...`) can be used with `GET /api/v1/generation?id=...` later. `[OR: api_reference/overview]`
`[OR: api_reference/streaming]`

### 3.7 Errors inside HTTP 200 (OpenRouter)

"Otherwise, the returned HTTP response status will be `200` and any error occurred while the LLM is producing the
output will be emitted in the response body or as an SSE data event." `[OR: api_reference/errors-and-debugging]`
Check every 200 body, in this order:

1. top-level `error` object (non-streaming: "a `200 OK` whose JSON body holds only an `error` object and no
   `choices`; its `id` identifies the generation");
2. `choices` missing or empty;
3. `choices[0].error` object, or `choices[0].finish_reason == "error"` ("the error is embedded in the final response
   alongside any partial content").

Shapes in section 10.5. Also "treat a `200` carrying an `error` chunk with no content as a failure, not a success".
`[OR: api_reference/streaming]`

---

## 4. The tool-result turn

### 4.1 What to send

After a reply with N tool calls, the next request's `messages` are: everything sent before, then the assistant
message, then N tool messages, in the order of `tool_calls`:

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  { "...": "the assistant message object exactly as received (see 4.2)" },
  {"role": "tool", "tool_call_id": "call_abc123", "content": "rows 1-16 stored"},
  {"role": "tool", "tool_call_id": "call_def456", "content": "row 17 rejected: 31 cells, expected 32"}
]
```

- Assistant message, canonical minimum: `role: "assistant"`, `content` ("Required unless `tool_calls` or
  `function_call` is specified", may be `null`), `tool_calls: [{id, type: "function", function: {name,
  arguments}}]` with `arguments` the same string that was received. `[SDK:
  types/chat/chat_completion_assistant_message_param.py]` `[SDK:
  types/chat/chat_completion_message_function_tool_call_param.py]`
- Tool message: `{"role": "tool", "tool_call_id": <id of the call>, "content": <string>}`; all three required.
  `[SDK: types/chat/chat_completion_tool_message_param.py]` `[OR-API: ChatToolMessage]`
  - `content`: always send a **string**. Canonical also allows an array of text parts and OpenRouter an array of
    content items, but OpenRouter's TypeScript request type and the vendor examples read here (OpenRouter, DeepSeek, vLLM, Groq,
    Mistral, Fireworks) use a plain string. For
    structured results, `json.dumps` them. `[OR: api_reference/overview]`
  - Do not send `name` on tool messages (vLLM and Mistral examples do; Groq returns 400 for `messages[].name`).
    `[DOC: https://console.groq.com/docs/openai.md]` Ollama looks the tool name up from the matching call id when
    `name` is absent. `[SRC: .../ollama/main/openai/openai.go :: FromChatRequest]`
- One tool message per call, each with its own `tool_call_id`; every documented loop does exactly this and nothing
  else between the assistant message and the tool messages. `[OR: guides/best-practices/reasoning-tokens]`
  `[DOC: https://api-docs.deepseek.com/guides/thinking_mode]`
  `[DOC: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/interleaved_thinking.md]`
- Resend the same `tools` array and the same extra-body settings on every call.

### 4.2 What must be replayed for reasoning to survive

Replay the assistant message **as the server sent it**: keep every key. Which key carries the state depends on the
server, which is why whitelisting fields is the wrong design:

| Server | State lives in | Rule (quoted) | Source |
| --- | --- | --- | --- |
| OpenRouter | `message.reasoning_details` (array); `message.reasoning` (string) is the plain-text alternative, `reasoning_content` an accepted alias | see quotes below | `[OR: guides/best-practices/reasoning-tokens]` |
| DeepSeek | `message.reasoning_content` | "for requests carrying the `tools` parameter, the `reasoning_content` must be fully passed back to the API in all subsequent requests - even for turns where the model did not perform a tool call. If your code does not correctly pass back `reasoning_content`, the API will return a 400 error." | `[DOC: https://api-docs.deepseek.com/guides/thinking_mode]` |
| Fireworks | `message.reasoning_content` | for interleaved thinking "you **MUST** include the `reasoning_content` from previous" assistant messages; recommended: "Pass the `Message` object directly" | `[DOC: https://docs.fireworks.ai/guides/reasoning.md]` |
| Together | `reasoning` or `reasoning_content` ("depending on the model") | preserved thinking: "include the unmodified `reasoning_content` from previous turns back in the conversation" (and `chat_template_kwargs: {"clear_thinking": false}`) | `[DOC: https://docs.together.ai/docs/inference/chat/reasoning.md]` |
| vLLM | `message.reasoning` (old name `reasoning_content` is renamed on input) | interleaved-thinking example appends `{"role": "assistant", "tool_calls": ..., "reasoning": ...}` | `[DOC: .../vllm/main/docs/features/interleaved_thinking.md]` `[SRC: .../vllm/entrypoints/openai/chat_completion/protocol.py]` |
| Ollama | `message.reasoning` | request `Message` struct reads `reasoning` and maps it to the model's thinking | `[SRC: .../ollama/main/openai/openai.go]` |
| llama.cpp | `message.reasoning_content` | "Reasoning input (preserve reasoning in history) is also supported by some specific templates." | `[DOC: .../llama.cpp/master/tools/server/README.md]` |
| Mistral native | `message.content` as a list including the `thinking` chunk | "always replay the full assistant message (including `ThinkChunk`) back into the message history" | `[DOC: https://docs.mistral.ai/llms-full.txt]` |
| xAI direct | nothing: encrypted reasoning exists only on the Responses API; "Chat Completions is unaffected; it has no field for the ciphertext" | reasoning state cannot be carried on this endpoint | `[DOC: https://docs.x.ai/developers/model-capabilities/text/reasoning.md]` |

OpenRouter's rule, verbatim `[OR: guides/best-practices/reasoning-tokens]`:

> To preserve reasoning context across multiple turns, you can pass it back to the API in one of two ways:
> 1. **`message.reasoning`** (string): Pass the plaintext reasoning as a string field on the assistant message
> 2. **`message.reasoning_details`** (array): Pass the full reasoning_details block
>
> Use `reasoning_details` when working with models that return special reasoning types (such as encrypted or
> summarized) - this preserves the full structure needed for those models.

> Preserving reasoning blocks is useful specifically for tool calling. When models like Claude invoke tools, it is
> pausing its construction of a response to await external information. When tool results are returned, the model
> will continue building that existing response. This necessitates preserving reasoning blocks during tool use

> When providing reasoning_details blocks, the entire sequence of consecutive reasoning blocks must match the
> outputs generated by the model during the original request; you cannot rearrange or modify the sequence of these
> blocks.

The documented Chat Completions loop appends the received message object itself: `messages = append(messages, msg)
// includes reasoning_details as-is`, and in the cURL tab "Send the assistant turn back unmodified (including
reasoning_details) plus the tool results". "The `reasoning_details` functionality works identically across all
supported reasoning models." `[OR: guides/best-practices/reasoning-tokens]`

`reasoning_details` entries (`ReasoningDetailUnion`), discriminated by `type`:

| `type` | Required | Other fields |
| --- | --- | --- |
| `reasoning.text` | `type` | `text: str|null`, `signature: str|null`, `id: str|null`, `format`, `index: int` |
| `reasoning.summary` | `type`, `summary: str` | `id`, `format`, `index` |
| `reasoning.encrypted` | `type`, `data: str` | `id`, `format`, `index` |
| `reasoning.server_tool_call` | `type`, `tool_name`, `arguments`, `result` | `tool_call_id`, `id`, `format`, `index` (only with OpenRouter server tools; we use none) |

`format` values: `unknown`, `openai-responses-v1`, `azure-openai-responses-v1`, `bedrock-openai-responses-v1`,
`bedrock-xai-responses-v1`, `xai-responses-v1`, `meta-responses-v1`, `google-vertex-meta-responses-v1`,
`google-vertex-xai-responses-v1`, `openrouter-stealth-v1`, `anthropic-claude-v1` (default), `google-gemini-v1`.
`[OR-API: ReasoningDetailText, ReasoningDetailSummary, ReasoningDetailEncrypted, ReasoningDetailServerToolCall,
ReasoningFormat]` `[OR: guides/best-practices/reasoning-tokens]` Treat every entry as opaque: store and resend.

### 4.3 Minimal normalisation before replay

Start from the received `message` dict and apply only these:

1. Drop keys whose value is `null`, except `content` (`refusal: null`, `reasoning: null`, `annotations`, `audio`,
   `function_call`). An absent optional key and a `null` one mean the same; strict servers are less likely to object
   to an absent one. Build the replay from the raw dict, not from `message.model_dump()`: the SDK's `model_dump()`
   adds `annotations`, `audio`, `function_call` as `null`; `to_dict()` returns only what the server sent. `[OFFLINE]`
2. If there are no tool calls and `content` is `null`, set `content` to `""`. Ollama rejects it otherwise ("content
   is only optional if tool calls are present" -> `invalid message content type`). `[SRC:
   .../ollama/main/openai/openai.go :: FromChatRequest]`
3. If a tool-call id had to be synthesised (empty/duplicate), write the synthesised id into the replayed
   `tool_calls` entry too.
4. Leave `tool_calls[i].function.arguments` byte-identical, with one optional escape hatch: Ollama rejects a
   replayed call whose arguments are not valid JSON (`invalid tool call arguments`). `[SRC: same ::
   FromCompletionToolCall]` If a server answers 400 to such a replay, retry once with that call's `arguments`
   replaced by `"{}"`. Make this a config option, off by default.

Nothing else: do not reorder keys in `reasoning_details`, do not merge `reasoning` into `content`, do not strip
`<think>` text.

The harness's `Reply.provider_payload` should be exactly this normalised message dict (JSON-serialisable), and
`build_request` should emit it unchanged for assistant messages whose `provider` is `openai_compat`.

### 4.4 Several tool calls in one reply

- All calls are in one `tool_calls` array of one assistant message; reasoning (`reasoning_details` etc.) is on the
  message, once, not per call. `[OR-API: ChatAssistantMessage]`
- Execute in array order; send one tool message per call, same order, each with its `tool_call_id`.
- A call with a bad name or bad arguments still gets a tool message (carrying the error text), so that every call
  id in the assistant message is answered.

### 4.5 Follow-up user message after a text reply (the "nudge")

Append the assistant message (same verbatim rule, it may carry reasoning) and then a plain user message:

```json
[
  "... earlier messages ...",
  {"role": "assistant", "content": "I have stored rows 1 to 16.", "reasoning_details": ["...as received..."]},
  {"role": "user", "content": "The picture is not finished: rows 17 to 22 are missing. Keep calling write_rows."}
]
```

- The conversation must end with a user or tool message. Current Claude models reject "a final assistant turn"
  (prefill) through OpenRouter. `[DOC: https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/claude-5-5]`
- DeepSeek: with `tools` in the request, the `reasoning_content` of this non-tool turn must be replayed too (quote in
  4.2). Verbatim replay already does that.
- On OpenRouter, `reasoning.context` (`auto` | `all_turns` | `current_turn`, "only supported by OpenAI GPT-5.6 and
  newer") controls whether the model sees reasoning from earlier turns; "The default value of `context` may vary by
  model." Leave it unset. `[OR: guides/best-practices/reasoning-tokens]` `[OR-API: ReasoningContext]`

### 4.6 History is append-only (prefix-bound reasoning)

For Claude Fable 5.1+, Opus 5.5, Sonnet 5.5, Haiku 5.5: "Preserved thinking ties each thinking block to the
conversation that produced it, meaning the system prompt, the tool list, and every message before it." Replaying a
thinking block after "injected or removed messages, in-place summarization, a changed system prompt, a changed tool
list" "returns a 400 `invalid_request_error`" on enforcing providers; "OpenRouter passes the upstream error through
and does not drop thinking blocks for you"; "If your harness replays history exactly as received and only appends,
nothing changes for you." `[DOC: https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/claude-5-5]`
`[DOC: https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/fable-5-1]`

Consequences: a retry of a failed call must resend the identical prefix; a restart is a fresh conversation, not an
edited one; never switch model or provider family inside one conversation (Sonnet 5.5 / Haiku 5.5 thinking blocks
are also account-bound; for Sonnet 5.5 OpenRouter only routes replays to Anthropic, Claude Platform on AWS, or
Azure and "returns a 400" if `provider.only` / `provider.ignore` excludes all of them). `[DOC: .../model-migrations/claude-5-5]`

---

## 5. Reasoning settings

Nothing here is standardised. The canonical API has one knob; every vendor adds its own.

### 5.1 Request controls by server

| Server | Body key(s) | Values | Default | Source |
| --- | --- | --- | --- | --- |
| OpenAI (canonical) | `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; "Not all reasoning models support every value" | reference says `medium`; guide says "model-dependent" | `[SDK: types/shared/reasoning_effort.py]` `[DOC: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create.md]` `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]` |
| OpenRouter | `reasoning: {effort, max_tokens, exclude, enabled, summary, context, mode}`; shorthand `reasoning_effort`; legacy `include_reasoning` | 5.2 | per model (`reasoning.default_effort`, `default_enabled` in the catalog) | `[OR-API: ChatRequest]` `[OR: guides/best-practices/reasoning-tokens]` |
| DeepSeek | `reasoning_effort`; `thinking: {"type": "enabled"|"disabled"}` | `none`, `low`, `high`, `max` (`minimal`->`low`, `medium`/`xhigh`->`high`, `ultra`->`max`) | thinking on, `high` | `[DOC: https://api-docs.deepseek.com/guides/thinking_mode]` `[DOC: https://api-docs.deepseek.com/api/create-chat-completion]` |
| vLLM | `reasoning_effort`; `chat_template_kwargs: {"enable_thinking": bool}` or `{"thinking": bool}`; `thinking_token_budget: int`; `include_reasoning: bool` (default true) | `none`..`max`; any non-`none` effort sets `enable_thinking=true`; the chat template decides what the levels mean ("some silently fall back to another effort") | model/template dependent; needs server flag `--reasoning-parser <name>` to get a separate `reasoning` field | `[DOC: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/reasoning_outputs.md]` |
| Ollama | `reasoning_effort` or `reasoning: {"effort": ...}` | model-defined names; aliases `minimal`->`low`, `xhigh`/`ultra`->`max`; `none` turns thinking off where possible; "Use `/api/show` to discover each model's supported values and default" | per model | `[DOC: .../ollama/main/docs/api/openai-compatibility.mdx]` |
| llama.cpp | `reasoning_effort` ("If `none`, reasoning/thinking is disabled. Otherwise, the value is made available to the jinja template"); `chat_template_kwargs`; `reasoning_format`; server flags `--reasoning`, `--reasoning-effort`, `--reasoning-budget`, `--reasoning-format` | template dependent; `GET /props` -> `chat_template_caps.reasoning_effort_levels` lists them lowest to highest | `auto` | `[DOC: .../llama.cpp/master/tools/server/README.md]` |
| LM Studio | app setting to return `reasoning_content` separately; no request parameter documented | - | - | `[DOC: https://lmstudio.ai/llms.txt]` |
| Groq | `reasoning_effort`; `reasoning_format: "parsed"|"raw"|"hidden"`; `include_reasoning: bool` (not together with `reasoning_format`) | per model: GPT-OSS `low`/`medium`/`high`; Qwen 3.8 27B `none`/`default`/`low`/`medium`/`high`; Muse Spark Open `low`/`medium`/`high`/`xhigh` (default `xhigh`) | per model | `[DOC: https://console.groq.com/docs/reasoning.md]` |
| Together | `reasoning: {"enabled": bool}`; `reasoning_effort` (`low`/`medium`/`high`; DeepSeek V4 Pro `high`/`max`); `chat_template_kwargs` | per model | per model | `[DOC: https://docs.together.ai/docs/inference/chat/reasoning.md]` |
| Fireworks | `reasoning_effort` (`low`/`medium`/`high` ...) **or** `thinking: {"type": "enabled", "budget_tokens": >=1024}`, not both | - | - | `[DOC: https://docs.fireworks.ai/guides/reasoning.md]` |
| Mistral | `reasoning_effort` | Large 4: `low`/`medium`/`high` (default `high`; rejects `none`, `minimal`, `xhigh`, `max`); Small 4 / Medium 3.5: `high` or `none`; `zai-glm-5-3`: `low`/`high`/`max` | per model | `[DOC: https://docs.mistral.ai/llms-full.txt]` |
| xAI | `reasoning_effort` | `low`, `medium`, `high`, `xhigh` (`xhigh` on grok-4.6+; treated as `high` on grok-4.5) | `high`; "Reasoning cannot be disabled" | `[DOC: https://docs.x.ai/developers/model-capabilities/text/reasoning.md]` |

Where reasoning text comes back: `reasoning` (OpenRouter, vLLM, Ollama, Groq parsed, some Together models),
`reasoning_content` (DeepSeek, llama.cpp, Fireworks, xAI, LM Studio, some Together models), list-valued `content`
(Mistral native), `reasoning_details` (OpenRouter only). vLLM: "`reasoning` used to be called `reasoning_content`
... your client code could silently read an empty `reasoning_content`, even when `reasoning` is populated." Read
both names. Sources as in the table.

### 5.2 OpenRouter's unified `reasoning` object

```json
{"reasoning": {"effort": "max"}}
```

| Key | Type / values | Notes |
| --- | --- | --- |
| `effort` | `"max"`, `"xhigh"`, `"high"`, `"medium"`, `"low"`, `"minimal"`, `"none"`, `null` | "OpenAI-style". `none` "Disables reasoning entirely". |
| `max_tokens` | integer | "Anthropic-style" budget. "Cannot be combined with `reasoning.effort` or the top-level `reasoning_effort`: a request that sets both is rejected with "Only one of "reasoning.effort" and "reasoning.max_tokens" can be specified"". Ignored/not forwarded by current Claude (adaptive thinking only) and unsupported by Gemini 4. |
| `enabled` | boolean | "Enables reasoning at the "medium" effort level"; default "inferred from `effort` or `max_tokens`". |
| `exclude` | boolean, default false | "The model will still use reasoning, but it won't be returned in the response. The tokens are still billed". Chat Completions only. **Leave false**: excluded reasoning text cannot be logged, though for Gemini 4 "the continuation item is still returned". |
| `summary` | `"auto"`, `"concise"`, `"detailed"`, `null` | reasoning-summary verbosity (schema `ChatReasoningSummaryVerbosityEnum`; no prose documentation found). |
| `context` | `"auto"`, `"all_turns"`, `"current_turn"`, `null` | "Only supported by OpenAI GPT-5.6 and newer." |
| `mode` | `"standard"`, `"pro"`, `null` | `pro` reroutes to the `*-pro` model; GPT-5.6+ only. Not a reasoning-depth setting for our purposes: choose the `-pro` slug explicitly if wanted. |

Also: top-level `reasoning_effort` is "Shorthand for setting reasoning effort. Equivalent to setting
reasoning.effort. Cannot be used simultaneously with reasoning.effort if they differ."; `include_reasoning: true`
== `reasoning: {}`, `include_reasoning: false` == `reasoning: {exclude: true}` (legacy); top-level `verbosity`
(`low`..`max`) also maps to Anthropic's `output_config.effort` and "wins if both are passed", so never send it.
`[OR-API: ChatRequest, ReasoningContext, ReasoningMode, ChatReasoningSummaryVerbosityEnum]`
`[OR: guides/best-practices/reasoning-tokens]` `[OR: api_reference/parameters]`
`[DOC: https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/claude-4-6]`

How effort is applied per vendor (OpenRouter):

- Claude 4.6 and newer: `reasoning.effort` -> Anthropic `output_config.effort`; thinking is adaptive; accepted
  `low`, `medium`, `high`, `xhigh`, `max` ("`minimal` is sent as `low`, `none` is rejected"). "Requests that set no
  effort are sent without one and run at Anthropic's default" (`medium` for Opus 5.5, `high` for Sonnet 5.5).
  Reasoning text is a summary by default (OpenRouter sends `display: "summarized"`). `[DOC:
  .../model-migrations/claude-5-5]` `[OR: guides/best-practices/reasoning-tokens]`
- Mandatory-reasoning models (`reasoning.mandatory: true` in the catalog): `reasoning: {"enabled": false}` or
  `{"effort": "none"}` returns `400` "Reasoning is mandatory for this endpoint and cannot be disabled." `[DOC:
  .../model-migrations/claude-5-5]`
- GPT-6: "Sol and Luna accept `none` as a reasoning effort, Astra does not." `[DOC: .../model-migrations/gpt-6]`
- Gemini 3 / 4: effort -> `thinkingLevel`; "If a model doesn't support a specific effort level ..., OpenRouter will
  map your requested effort to the nearest supported level" (table maps `xhigh` -> `high`). Gemini 4 Argon accepts
  `low`, `medium`, `high`; "reasoning cannot be turned off". `[OR: guides/best-practices/reasoning-tokens]`
  `[DOC: https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/gemini-4]`
- Older budget-based models: effort becomes a share of the output limit: `max`/`xhigh` ~95%, `high` ~80%, `medium`
  ~50%, `low` ~20%, `minimal` ~10%; Anthropic formula `budget_tokens = max(min(max_tokens * ratio, 128000), 1024)`.
  `[OR: guides/best-practices/reasoning-tokens]`
- "Some reasoning models do not return their reasoning tokens ... some (like the OpenAI o-series) do not."
  `[OR: guides/best-practices/reasoning-tokens]` Encrypted state still comes back as `reasoning_details` entries
  of type `reasoning.encrypted` ("such as OpenAI's encrypted reasoning content or Anthropic's redacted thinking");
  then "the trace carries no reasoning text even though the reasoning token count ... is non-zero".
  `[DOC: https://openrouter.ai/docs/guides/features/broadcast]`

### 5.3 Highest setting per current model (OpenRouter catalog, 2026-11-19)

`GET /api/v1/models` gives each model a `reasoning` object: `supported_efforts` ("returned in descending effort
order (highest first)"; `null` means "all gateway effort values are accepted"; omitted means no effort selection),
`default_effort`, `default_enabled`, `mandatory`, `supports_max_tokens`. `[OR: guides/best-practices/reasoning-tokens]`

| Model id | `supported_efforts` | Default | Mandatory | Highest |
| --- | --- | --- | --- | --- |
| `anthropic/claude-fable-5.2`, `anthropic/claude-opus-5.5`, `anthropic/claude-sonnet-5.5` | max, xhigh, high, medium, low | high | yes | `max` |
| `anthropic/claude-haiku-5.5` | max, xhigh, high, medium, low, none | medium | no | `max` |
| `openai/gpt-6.2-astra`, `openai/gpt-6.2-sol` | max, xhigh, high, medium, low | medium | yes | `max` |
| `openai/gpt-6.1-terra`, `openai/gpt-6-luna` | max, xhigh, high, medium, low, none | medium | no | `max` |
| `google/gemini-4-neon`, `google/gemini-4-hydrogen` | xhigh, high, medium, low | medium | yes | `xhigh` |
| `google/gemini-4-argon`, `google/gemini-4-carbon`, `google/gemini-3.8-flash` | high, medium, low | medium | yes | `high` |
| `x-ai/grok-4.7` | xhigh, high, medium, low | high | yes | `xhigh` |
| `deepseek/deepseek-v4-pro-0813`, `deepseek/deepseek-v4.1-flash` | max, high, low, none | high | no | `max` |
| `mistralai/mistral-large-4-0` | high, none | high | no | `high` |
| `meta/muse-wonder`, `meta/muse-spark-1.3` | max, xhigh, high, medium, low, minimal | medium | yes | `max` |
| `qwen/qwen4-max` | xhigh, high, low, none | high | no | `xhigh` |
| `z-ai/glm-5.3` | max, high, low | max | yes | `max` |
| `moonshotai/kimi-k3.1-pro` | max, high, low, none | max | no | `max` |

`[OR-CAT]` Put the highest value in the model's config explicitly (do not rely on OpenRouter mapping `max` down, and
do not rely on defaults: the catalog lists `default_effort: high` for `anthropic/claude-opus-5.5` while the
migration guide says an unset effort runs at Anthropic's default of `medium`). A small helper that reads the
catalog and prints `supported_efforts[0]` for a model id is a cheap pre-flight check.

### 5.4 Reasoning and the output limit

- OpenRouter: "On most providers, the request's `max_output_tokens` limit applies to reasoning and visible output
  combined." ... "On Chat Completions (`/api/v1/chat/completions`), the same budget is `max_tokens` (or
  `max_completion_tokens`, which shares the same budget). If the model spends all of it reasoning, the response
  returns `finish_reason: "length"` with an empty `content`, and the reasoning tokens are still billed." "Retrying
  does not help; raise `max_tokens` or cap reasoning instead." `[OR: guides/best-practices/reasoning-tokens]`
  `[OR: api_reference/errors-and-debugging]`
- Claude 5.5 via OpenRouter: "Thinking and the reply share `max_tokens`. Anthropic reports that 64k worked well for
  long agentic coding turns". `[DOC: .../model-migrations/claude-5-5]`
- OpenAI canonical: `max_completion_tokens` includes reasoning tokens (2.6); OpenAI recommends "reserving at least
  25,000 tokens for reasoning and outputs". `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]`
- xAI direct is the exception: `max_completion_tokens` "does not apply to tokens used for reasoning" (2.6).
- vLLM: without `thinking_token_budget` "no explicit reasoning limit is applied beyond normal generation constraints
  such as `max_tokens`". `[DOC: .../vllm/main/docs/features/reasoning_outputs.md]`
- DeepSeek: default `max_tokens` is 64K in thinking mode, 128K at `reasoning_effort: max`. (2.6)

### 5.5 Parameters that are rejected or ignored with reasoning

| What | Behaviour | Source |
| --- | --- | --- |
| `temperature`, `top_p`, `top_k` on Claude 4.7 Opus and later | "silently ignored" (4.7); "Non-default `temperature`, `top_p`, and `top_k` are not supported" (5.5) | `[DOC: .../model-migrations/claude-4-7]` `[DOC: .../model-migrations/claude-5-5]` |
| forced `tool_choice` on Claude Fable 5.1+, Opus 5.5, Sonnet 5.5 | 400 | section 2.3 |
| assistant prefill on Claude Opus 5 and later | not supported | `[DOC: .../model-migrations/claude-5-5]` |
| `reasoning.effort` together with `reasoning.max_tokens` | 400 on OpenRouter | `[OR-API: ChatRequest]` |
| disabling reasoning on a mandatory model | 400 on OpenRouter | 5.2 |
| `effort: "none"` on GPT-6 Astra, GPT-6.2 Astra/Sol (OpenAI direct) | HTTP 400 | `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]` |
| `max_tokens` on OpenAI o-series (direct) | "not compatible" | `[SDK: types/chat/completion_create_params.py]` |
| `stop` on OpenAI o3 / o4-mini (direct) | "Not supported" | same |
| `presencePenalty`, `frequencyPenalty`, `stop` on xAI reasoning models | "Requests that include them return an error." | `[DOC: https://docs.x.ai/developers/model-capabilities/text/reasoning.md]` |
| `temperature`, `presence_penalty`, `frequency_penalty` on DeepSeek thinking mode | accepted, no effect | `[DOC: https://api-docs.deepseek.com/guides/thinking_mode]` |
| `thinking` together with `reasoning_effort` on Fireworks | validation error | `[DOC: https://docs.fireworks.ai/guides/reasoning.md]` |
| `reasoning_format: "raw"` with tools on Groq | 400 | `[DOC: https://console.groq.com/docs/reasoning.md]` |
| a `reasoning_effort` value the chat template raises on, llama.cpp | "answered with a 400" (`GET /props` -> `chat_template_caps.reasoning_effort_rejected`) | `[DOC: .../llama.cpp/master/tools/server/README.md]` |
| function calling with reasoning on Together DeepSeek V3.1 | "only works in non-reasoning mode" | `[DOC: https://docs.together.ai/docs/inference/chat/reasoning.md]` |

### 5.6 Is "extra_body in config + verbatim replay" sufficient?

Yes as the core design. Every reasoning control found is a top-level body key (`reasoning`, `reasoning_effort`,
`thinking`, `chat_template_kwargs`, `thinking_token_budget`, `include_reasoning`, `reasoning_format`), as are the
OpenRouter routing keys (`provider`, `session_id`, `cache_control`). A free-form dict merged into the body reaches
all of them, and the SDK supports it directly (`extra_body`, section 1.6). Verbatim replay of the assistant message
covers every server in 4.2 because each server reads back the field it emitted.

What the design needs in addition:

1. `max_tokens_param` (a rename, not an extra key), section 2.6.
2. `extra_headers` (dict, default empty) for the rare header-only switches (`x-anthropic-beta`, `X-OpenRouter-Metadata`).
3. `stream` (bool) and, if streaming, the `reasoning_details` merge rule in 1.6. Verbatim replay is only as good as
   the reassembly. If a Claude/GPT conversation gets a 400 on the first tool-result turn with `stream: true`, set
   `stream: false` for that model: the non-streaming message is byte-exact.
4. Replay must be of the whole message object, not "known keys + a list of extras": the state key differs per
   server and Mistral puts it inside `content`.
5. Response-side handling that no request setting provides: errors inside 200, `content` as list, the two usage
   conventions, and the Gemini 4 continuation (6.3).
6. A per-model `reasoning` value: there is no universal "highest". Suggested mapping for the harness config:
   `ModelSettings.reasoning` (dict), if non-empty, is sent verbatim as the top-level `reasoning` key (OpenRouter's
   object; Ollama and Together also read a `reasoning` object); servers that want `reasoning_effort` or `thinking`
   get them through `options.extra_body`. `extra_body` is applied last and wins.

Suggested `options` keys: `base_url` (required), `api_key_env` (required; `null` for keyless local servers ->
dummy key), `extra_body` (dict), `extra_headers` (dict), `max_tokens_param`, `stream` (default true),
`stream_usage` (default true; false omits `stream_options`), `timeout` (dict of connect/read/write/pool seconds),
`sanitize_bad_tool_arguments` (default false, see 4.3).

---

## 6. Output-token limits and context windows

### 6.1 Current models (OpenRouter catalog, 2026-11-19)

`context_length` = context window; `top_provider.max_completion_tokens` = "Maximum tokens in response". "Input and
output tokens share the model's context window, so `max_completion_tokens` is a ceiling for `max_tokens`, not a
guaranteed output size." `[OR: guides/overview/models]`

| Model id | Context | Max output |
| --- | --- | --- |
| `anthropic/claude-fable-5.2`, `claude-opus-5.5`, `claude-sonnet-5.5`, `claude-haiku-5.5` | 1,000,000 | 128,000 |
| `openai/gpt-6.2-astra`, `gpt-6.2-sol`, `gpt-6.1-terra`, `gpt-6-luna` | 1,050,000 | 128,000 |
| `google/gemini-4-neon`, `gemini-4-hydrogen`, `gemini-4-carbon`, `gemini-4-argon` | 1,048,576 | 943,718 total (65,536 per request for Argon, see 6.3) |
| `google/gemini-3.8-flash` | 1,048,576 | 65,536 |
| `x-ai/grok-4.7` | 500,000 | 450,000 |
| `deepseek/deepseek-v4-pro-0813`, `deepseek/deepseek-v4.1-flash` | 1,048,576 | 943,718 |
| `mistralai/mistral-large-4-0` | 1,048,576 | 262,144 |
| `meta/muse-wonder`, `meta/muse-spark-1.3` | 1,048,576 | 943,718 |
| `qwen/qwen4-max` | 1,000,000 | 131,072 |
| `z-ai/glm-5.3`, `moonshotai/kimi-k3.1-pro` | 1,048,576 | 943,718 |

`[OR-CAT]` These are the top provider's numbers; other providers of the same model can be lower (per-endpoint
values at `GET /api/v1/models/{author}/{slug}/endpoints`). `[OR: api_reference/models]`

### 6.2 What "set the limit high" should mean

- OpenRouter range: `max_tokens` "Range: [1, context_length)". `[OR: api_reference/overview]`
- Setting it has routing and billing side effects: "if you set `max_output_tokens` (or `max_tokens` on Chat
  Completions), then OpenRouter will only route to providers that support a response of that length"
  `[OR: guides/routing/provider-selection]`; and the in-flight spending hold is estimated from "the input tokens,
  plus the completion tokens allowed by `max_tokens` up to a fixed per-request cap", so on a low-balance prepaid
  account a huge value can be refused with `402` `"reason": "weight_exceeds_budget"`. `[OR: api_reference/limits]`
- Recommendation: per-model `max_output_tokens` in the config, set to the catalog's `max_completion_tokens` for
  128K-class models (Claude, GPT-6: `128000`), and to a deliberate value for the models whose catalog figure is a
  near-context number (`64000` to `128000`; for Gemini 4 see 6.3). Never derive it from `context_length`.
- If the adapter omits the parameter, the documented defaults are: DeepSeek 64K/128K in thinking mode; xAI
  128,000; OpenRouter sends Gemini 4 "its default output budget of 65,536 tokens"; llama.cpp `n_predict` default
  is `-1` (infinity). Other servers' defaults were not found in the pages read. Sources in 2.6 and `[DOC: .../model-migrations/gemini-4]`,
  `[DOC: .../llama.cpp/master/tools/server/README.md]`.
- Local servers: the context window is a server-launch setting. Ollama: "The OpenAI API does not have a way of
  setting the context size for a model"; change it with a `Modelfile` `PARAMETER num_ctx`. `[DOC:
  .../ollama/main/docs/api/openai-compatibility.mdx]`

### 6.3 Gemini 4 "long decode" on OpenRouter (continuation)

`[DOC: https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/gemini-4]`

- "Gemini 4 Argon ... returns at most 65,536 output tokens per request (its per-request output limit), but it can
  keep working on one answer for up to 1M output tokens in total."
- `max_tokens` at or below the per-request limit: a normal single request; hitting the limit ends with
  `finish_reason: "length"`, `native_finish_reason: "MAX_TOKENS"`.
- `max_tokens` above the per-request limit: on reaching the per-request limit the response ends with
  `finish_reason: "length"`, `native_finish_reason: "CONTINUATION"` and a continuation token: a
  `reasoning_details` entry `{"type": "reasoning.encrypted", "id": "gemini-continuation", "format":
  "google-gemini-v1", "data": "<opaque>", "index": 0}`.
- "At `high` effort the model can spend a hop's full per-request limit on reasoning and return little or no visible
  text in that hop. A hop with empty content and `native_finish_reason: "CONTINUATION"` is normal: resume it."
- To resume: resend "the same conversation and parameters with the stopped turn appended as the last assistant
  turn", `max_tokens` "above the total output of every hop so far"; "Do not add a new user message in the middle of
  a continuation"; concatenate the visible text of all hops; sum usage ("Every hop is its own generation with its
  own usage and cost"). A continuation "only works on the provider that issued it". Tokens "can reach several MB".
- "A hop that finishes normally can still carry checkpoint items [when streaming], so decide whether to continue
  from the finish reason, not from whether an item is present." After a `MAX_TOKENS` hop the budget is spent:
  "Resending it with the same `max_tokens` returns a 400 error."
- The guide documents the 65,536 per-request figure for Argon and points to the model page for limits; it was not
  checked for Neon, Hydrogen or Carbon.
- The stopped turn **replaces** the previous stopped turn; it is not appended after it. "Only a continuation item
  in the last assistant turn resumes the decode. Each hop replaces the previous continuation turn, and OpenRouter
  removes continuation items from earlier turns"; "OpenRouter drops the trailing continuation turn before
  forwarding the request, because the token already carries its state"; "OpenRouter resumes from the last
  continuation item in the turn". The guide's own loop keeps the original conversation fixed and sends
  `[...conversation, continuation]`, where `continuation` is rebuilt on every hop as
  `{"role": "assistant", "content": "", "reasoning_details": [<latest item with id "gemini-continuation">]}`.
  A request that stacked two stopped assistant turns would no longer have "the conversation before it ... resent
  unchanged".
- A `MAX_TOKENS` hop also returns a continuation token; the guide's loop continues after it by raising
  `max_tokens` ("set `max_tokens` above the total output of every hop so far"), and breaks only when
  `finish_reason != "length"` or no continuation item is present.
- Adapter guidance. Simplest, and what the fixtures assume: do **not** loop. Send Gemini 4 with
  `max_tokens <= 65536` (or omit it), report any `length` stop as `max_tokens`, and let the harness decide. If the
  loop is wanted, implement it inside `complete()` exactly as the guide does: keep `base = messages` (the request's
  messages, unchanged); while `finish_reason == "length"` and `native_finish_reason == "CONTINUATION"` and
  `reasoning_details` contains an entry with `id == "gemini-continuation"`: set
  `messages = base + [{"role": "assistant", "content": "", "reasoning_details": [last such entry]}]` and call again
  with the same body; concatenate `content` across hops and sum `usage` across hops; stop on any other finish, and
  on `MAX_TOKENS` report `max_tokens` (the budget the config allowed is spent). This is the one case where a
  request legitimately ends with an assistant message.
- Not documented, so UNVERIFIED: how a continuation interacts with tool calls in the stopped turn, and what the
  replay payload of a multi-hop turn should be on the *next* tool-result turn (the guide only says that a finished
  long answer in history "does not affect later requests"). This is a second reason to prefer the no-loop option
  for contestants.

---

## 7. Vision input for the judge

### 7.1 Message shape

One user message whose `content` is an array of parts, text and images interleaved in order:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "REFERENCE"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}},
    {"type": "text", "text": "Picture K"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}},
    {"type": "text", "text": "<the question>"}
  ]
}
```

- Canonical part types for user content: `{"type": "text", "text": str}` and `{"type": "image_url", "image_url":
  {"url": str, "detail": "auto"|"low"|"high"}}` where `url` is "Either a URL of the image or the base64 encoded image
  data". `[SDK: types/chat/chat_completion_content_part_text_param.py]`
  `[SDK: types/chat/chat_completion_content_part_image_param.py]` The SDK sends the parts unchanged. `[OFFLINE]`
- OpenRouter: same shape (`ChatContentImage`: "URL of the image (data: URLs supported)"; `detail` enum `auto`,
  `low`, `high`, `original`, where `original` is an OpenRouter extension "downgraded to `high` for providers that
  lack an original-resolution tier"). `[OR-API: ChatContentItems, ChatContentImage]` Omit `detail` (default `auto`).
- Data URL form: `data:image/png;base64,<standard base64, no line breaks>`. Media types OpenRouter lists:
  `image/png`, `image/jpeg`, `image/webp`, `image/gif`. `[OR: guides/overview/multimodal/image-understanding]`
- OpenRouter's ordering advice: "Due to how the content is parsed, we recommend sending the text prompt first, then
  the images. If the images must come first, put them before the text within the same user message; system messages
  should contain text only." "ContentParts are only for the "user" role". `[OR: same]` `[OR: api_reference/overview]`
  Interleaved label/image/label/image inside one user message is within the schema; keep the judge's system prompt
  text-only.
- The judge call sends no `tools`, no `reasoning` unless configured, and reads `choices[0].message.content` (3.2).

### 7.2 Which servers accept it

| Server | Images | Notes | Source |
| --- | --- | --- | --- |
| OpenRouter | model-dependent: 251 of the 387 chat-capable catalog models list `image` in `architecture.input_modalities`; filter `GET /api/v1/models?input_modalities=image` | "The number of images you can send in a single request varies per provider and per model." Image errors are typed (`error.metadata.error_type`): `invalid_image`, `image_too_large`, `image_too_small`, `unsupported_image_format`, `image_download_failed` (all 400) and `image_not_found` (404); the last two concern image URLs, not data URLs | `[OR-CAT]` `[OR: guides/overview/multimodal/image-understanding]` `[OR: api_reference/errors-and-debugging]` |
| OpenAI (direct) | yes | PNG, JPEG, WEBP, non-animated GIF; "Up to 512 MB total payload per request"; "Up to 1,500 images per request"; at most 30,000 patches per image | `[DOC: https://developers.openai.com/api/docs/guides/images-vision.md]` |
| vLLM | with a multimodal model | "`image_url.detail` parameter is not supported" | `[DOC: .../vllm/main/docs/serving/online_serving/openai_compatible_server.md]` |
| Ollama | yes | base64 data URLs only ("[ ] Image URL"); accepts `image_url` as an object `{"url": ...}` or as a bare string; data URL must start with `data:image/<jpeg|jpg|png|webp>;base64,` (or `data:;base64,`) | `[DOC: .../ollama/main/docs/api/openai-compatibility.mdx]` `[SRC: .../ollama/main/openai/openai.go]` |
| llama.cpp | when the model is loaded with a multimodal projector (`--mmproj`; "experimental") | `image_url.url` "can be a remote URL, base64 (raw or URI-encoded via `data:image/...;base64`) or path to local file"; formats of `stb_image` (jpeg, png, tga, bmp, gif) | `[DOC: .../llama.cpp/master/tools/server/README.md]` |
| DeepSeek | on vision-capable models | http(s) URL "(max 8192 characters)" or base64 data URL; JPEG, PNG, GIF, WebP; `detail: low` "downsamples the image to 512x512" | `[DOC: https://api-docs.deepseek.com/api/create-chat-completion]` |
| Groq | on vision models | `qwen/qwen3.8-27b`: "maximum of 3 images", 2048 tokens each; `meta-models/muse-spark-open`: "maximum of 5 images", up to 1,026 tokens each; 20MB request limit for image URLs | `[DOC: https://console.groq.com/docs/vision.md]` |
| Fireworks | on vision models | "120 images maximum"; "Total base64-encoded images must be less than 10MB"; URL images under 5MB | `[DOC: https://docs.fireworks.ai/guides/querying-vision-language-models.md]` |
| xAI | yes | "Maximum image size: `20MiB`"; "Maximum number of images: No limit" | `[DOC: https://docs.x.ai/developers/model-capabilities/images/understanding.md]` |
| Together | on vision models | base64 data URLs and multiple images supported | `[DOC: https://docs.together.ai/docs/inference/vision/inputs.md]` |
| Mistral | on vision models | (limits not extracted) | `[DOC: https://docs.mistral.ai/llms-full.txt]` |

A 9-image judge call therefore cannot go to Groq's current vision models. For Claude and Gemini through OpenRouter
the per-request image limits are the upstream provider's and are not stated in OpenRouter's docs: take them from
the `anthropic` and `google` provider notes.

### 7.3 Billing

- OpenRouter bills images as input: `prompt_tokens` is "Including images, input audio, and tools if any", counted
  with "the model's native tokenizer". `[OR: api_reference/overview]` The catalog `pricing.image` field ("Cost per
  image input") is set for Gemini models (equal to the per-token prompt price, e.g. `"0.000002"` for
  `google/gemini-4-argon`) and absent for Claude and GPT. `[OR: guides/overview/models]` `[OR-CAT]`
  The in-flight spending estimate ignores image pricing. `[OR: api_reference/limits]`
- OpenAI models tokenise images in 32x32 px patches: `patch_count = ceil(width/32) * ceil(height/32)` after
  resizing to the detail level's limits, times a per-model multiplier (worked example: 1024x1024 ->
  1024 patches -> `ceil(1024 * 1.2) = 1229` tokens). `[DOC: https://developers.openai.com/api/docs/guides/images-vision.md]`
- The only reliable number for a given judge model is `usage.prompt_tokens` (and `usage.cost` on OpenRouter) from
  the response. Record both.

---

## 8. Model ids and prices

All from the OpenRouter catalog, `GET https://openrouter.ai/api/v1/models`, fetched 2026-11-19 `[OR-CAT]`. Prices
are USD per million tokens ("pricing from the top provider for this model"; the JSON holds USD per token as
strings). `T`/`I` = `tools` in `supported_parameters` / `image` in `input_modalities`.

| Tier | Model id (exact) | Input | Output | Cache read | T | I |
| --- | --- | --- | --- | --- | --- | --- |
| Anthropic flagship | `anthropic/claude-fable-5.2` | 8 | 40 | 0.20 | yes | yes |
| Anthropic high | `anthropic/claude-opus-5.5` | 4 | 20 | 0.20 | yes | yes |
| Anthropic mid | `anthropic/claude-sonnet-5.5` | 2 | 10 | 0.10 | yes | yes |
| Anthropic small | `anthropic/claude-haiku-5.5` | 0.10 | 0.50 | 0.01 | yes | yes |
| OpenAI flagship | `openai/gpt-6.2-astra` | 8 (16 above 272,000 prompt tokens) | 40 (60) | 0.40 | yes | yes |
| OpenAI mid | `openai/gpt-6.2-sol` | 2 | 10 | 0.10 | yes | yes |
| OpenAI mid-low | `openai/gpt-6.1-terra` | 1 | 5 | 0.05 | yes | yes |
| OpenAI small | `openai/gpt-6-luna` | 0.10 | 0.50 | 0.01 | yes | yes |
| Google flagship reasoning | `google/gemini-4-argon` | 2 | 10 | 0.10 | yes | yes |
| Google frontier | `google/gemini-4-carbon` | 2 | 10 | 0.10 | yes | yes |
| Google mid | `google/gemini-4-neon` | 3 | 15 | 0.15 | yes | yes |
| Google workhorse | `google/gemini-4-hydrogen` | 1.5 | 7.5 | 0.075 | yes | yes |
| Google flash | `google/gemini-3.8-flash` | 0.825 | 4.125 | 0.0825 | yes | yes |
| xAI flagship | `x-ai/grok-4.7` | 2 | 6 | 0.50 | yes | yes |
| DeepSeek pro | `deepseek/deepseek-v4-pro-0813` | 0.22 | 9.5 | 0.17 | yes | no |
| DeepSeek flash | `deepseek/deepseek-v4.1-flash` | 0.30 | 1.20 | 0.006 | yes | yes |
| Mistral flagship | `mistralai/mistral-large-4-0` | 0.7 | 2.1 | 0.07 | yes | yes |
| Meta flagship | `meta/muse-wonder` | 4 | 20 | 0.20 | yes | yes |
| Meta mid | `meta/muse-spark-1.3` | 1.25 | 4.25 | 0.15 | yes | yes |
| Qwen flagship | `qwen/qwen4-max` | 1.425 | 4.275 | 0.07125 | yes | yes |
| Z.ai | `z-ai/glm-5.3` | 0.0699 | 1.4 | 0.0689 | yes | no |
| Moonshot | `moonshotai/kimi-k3.1-pro` | 0.253 | 8.8 | 0.18 | yes | yes |

Notes:

- Re-fetched 2026-11-20 by the verification pass: every id, price, context, max-output, effort list, default and
  `mandatory` flag in this section and in 5.3 / 6.1 matched the catalog, **except** `deepseek/deepseek-v4.1-flash`,
  which the first pass listed at 0.051 / 0.567 / 0.018 and the catalog now lists at 0.30 / 1.20 / 0.006 (corrected
  above; no catalog entry carries the old figures). "The catalog updates continuously as providers deploy new
  models, adjust pricing, and add hardware endpoints" `[OR: api_reference/models]`: read prices at run time, or
  better use `usage.cost`. The catalog also has a newer dated sibling `deepseek/deepseek-v4.1-flash-1017`
  (0.036 / 0.50 / 0.01).
- The tier labels are my reading of the catalog descriptions and prices, not an OpenRouter field. Google's Gemini 4
  names do not encode a tier; the catalog calls Argon "Google's flagship reasoning model", Carbon "a frontier Gemini
  4 model", Neon "a fast, mid-priced model", Hydrogen "the high-efficiency workhorse".
- `pricing.overrides` holds conditional prices keyed on `min_prompt_tokens`: all four OpenAI entries above at
  272,000 prompt tokens (input x2, output x1.5; e.g. Astra 16 / 60), `x-ai/grok-4.7` at 200,000 (4 / 12),
  `anthropic/claude-haiku-5.5` at 100,000 (0.50 / 2.50). The other rows have no override. A paintbench
  conversation is far below these thresholds. Anthropic entries also carry `input_cache_write` (10 for Fable 5.2) and
  `input_cache_write_1h` (16). Reasoning tokens bill at the output price.
- `:batch` variants only serve `/api/v1/batches` (their catalog `endpoints` field) and are cheaper: exactly half
  price for 69 of the 89 batch entries that have a base entry, including the Claude, GPT-6 and Gemini 4 ones; the
  rest vary (`x-ai/grok-4.7:batch` is 0.8x) `[OR-CAT]`; `:free` variants are rate-limited (20
  requests/minute; 50 or 1000 per day). Neither is suitable. `[OR: api_reference/limits]`
- Dated ids for reproducibility are in `canonical_slug`, e.g. `anthropic/claude-fable-5.2-20261109`,
  `openai/gpt-6.2-astra-20261117`, `google/gemini-4-argon-20261013`.
- Capability filters: `GET /api/v1/models?supported_parameters=tools&input_modalities=image`. At fetch time the
  catalog had 479 entries, 387 of them served by `/api/v1/chat/completions` (the rest are `:batch` variants); of
  those 387, 327 list `tools`, 251 take image input and 223 do both.
- The response's `usage.cost` is the authoritative per-call charge; prefer it to price x tokens for OpenRouter runs.

First-party prices fetched for cross-checking: OpenAI's own page lists the same short-context prices
(`gpt-6.2-astra` $8 / $40, `gpt-6.2-sol` $2 / $10, `gpt-6.1-terra` $1 / $5, `gpt-6-luna` $0.10 / $0.50) and a
long-context tier at double input. `[DOC: https://developers.openai.com/api/docs/pricing.md]` I did not fetch
first-party pricing pages for DeepSeek, Groq, Together, Fireworks, Mistral or xAI; do not assume the OpenRouter
figures apply to their direct APIs. Direct model ids seen in vendor docs (not priced here): DeepSeek
`deepseek-flash`, `deepseek-v4-pro`; xAI `grok-4.7`; Mistral `mistral-large-4-0`, `mistral-medium-3-5`,
`mistral-small-2603`; Groq `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, `meta-models/muse-spark-open`.

---

## 9. Errors and retries

### 9.1 SDK exception classes

`[SDK: _exceptions.py]` `[SDK: _client.py :: OpenAI._make_status_error]` `[OFFLINE]`

| Situation | Exception | Notes |
| --- | --- | --- |
| HTTP 400 | `BadRequestError` | |
| 401 | `AuthenticationError` | |
| 403 | `PermissionDeniedError` | on OpenRouter also moderation, guardrail, budget, refusal blocks |
| 404 | `NotFoundError` | |
| 409 | `ConflictError` | |
| 422 | `UnprocessableEntityError` | |
| 429 | `RateLimitError` | |
| >= 500 (500, 502, 503, 504, 524, 529) | `InternalServerError` | |
| other 4xx (402, 408, 413) | `APIStatusError` (base class) | |
| connect failure, protocol error before a response | `APIConnectionError` | |
| timeout before a response | `APITimeoutError` (subclass of `APIConnectionError`) | |
| SSE chunk containing a top-level `error` key | `APIError` (not an `APIStatusError`; `e.body` is the error object, `e.message` its `message`) | this is how OpenRouter mid-stream errors arrive |
| transport failure **while iterating** a stream | raw `httpx.ReadTimeout`, `httpx.RemoteProtocolError`, `httpx.ReadError` (not `OpenAIError`) | catch `httpx.HTTPError` too |

All status errors carry `.status_code`, `.response` (so `.response.headers`), `.body` (the value under `"error"`
when the body has that key, else the whole body, else the raw text). `[OFFLINE]`

### 9.2 What the SDK retries by itself

- Retries when: connection errors and timeouts; status `408`, `409`, `429`, `>= 500`; header `x-should-retry: true`.
  Does not retry `402` (even with `Retry-After`), `400`, `401`, `403`, `404`, `413`, `422`.
  `[SDK: _base_client.py :: BaseClient._should_retry]` `[OFFLINE]`
- Backoff `min(0.5 * 2**n, 8.0)` seconds with jitter; honours `retry-after-ms` / `retry-after` when
  `0 < value <= 120` s; a `Retry-After` above 120 s means **no retry**. `[SDK: _base_client.py ::
  _calculate_retry_timeout, _parse_retry_after_header]` `[OFFLINE]` (429 with `retry-after: 7` slept 7.0 twice;
  with `retry-after: 500` made one attempt.)
- Default 2 retries (3 attempts). Configure with `OpenAI(max_retries=N)` or
  `client.with_options(max_retries=N)`. `[SDK: ../openai-2.53.0.dist-info/METADATA]`
- A read timeout on a non-streaming call is retried like any other timeout, i.e. the whole (possibly 30-minute,
  already billed) request is sent again. `[OFFLINE]` Mid-stream failures are never retried by the SDK.
- Recommendation (matches the harness rule "keep the SDK's own retries low"): `max_retries=0`, classify in the
  adapter, let the harness back off.

### 9.3 Classification for `TransientError` / `FatalError`

| Signal | Class | `retry_after` |
| --- | --- | --- |
| `APITimeoutError`, `APIConnectionError`, `httpx.HTTPError` during a stream | transient | - |
| 408, 409, 429, 500, 502, 503, 504, 524, 529 | transient | `Retry-After` header when present |
| OpenRouter 402 with `error.metadata.limit_source == "openrouter_in_flight_budget"` | transient | `Retry-After` header ("wait for it and retry") |
| other 402 ("insufficient credits", `weight_exceeds_budget`, `openrouter_key_limit`) | fatal | - |
| 400, 401, 404, 413, 422 | fatal | - |
| 403 with `error.metadata.error_type` in `refusal`, `content_policy_violation`, or moderation metadata (`reasons`) | not an error: refusal `Reply` (3.5) | - |
| 403 budget (`limit_source` = `openrouter_guardrail_budget` / `openrouter_workspace_budget` / `openrouter_key_limit`), guardrail, permissions | fatal | - |
| error inside 200 / stream (`error.code` + `error.metadata.error_type`) | by `error.code` using the rows above; `error_type` `provider_overloaded`, `provider_unavailable`, `rate_limit_exceeded`, `timeout`, `server`, `unmapped` -> transient; `context_length_exceeded`, `max_tokens_exceeded`, `invalid_request`, `invalid_prompt`, `authentication`, `payment_required`, `permission_denied` -> fatal | - |
| stream ended without `finish_reason`, zero chunks, body not JSON, `choices` empty with no `error` | transient | - |
| DeepSeek `finish_reason` `insufficient_system_resource` / `aborted` | transient | - |

OpenRouter status meanings: `400` bad request, `401` invalid credentials, `402` insufficient credits, `403`
forbidden (permissions, guardrail, moderation), `408` request timed out, `429` rate limited, `502` "Your chosen
model is down or we received an invalid response from it", `503` "no available model provider that meets your
routing requirements"; the OpenAPI adds `404`, `413`, `422`, `500`, `524`, `529` ("Provider Overloaded").
`[OR: api_reference/errors-and-debugging]` `[OR-API]` Error body: `{"error": {"code": <int>, "message": <str>,
"metadata": {...}}}`; `error.metadata.error_type` is "the field to rely on"; `metadata.provider_code` is the
upstream code "when available". `[OR: api_reference/errors-and-debugging]` Full `error_type` list: token limits
`context_length_exceeded`, `max_tokens_exceeded`, `token_limit_exceeded`, `string_too_long` (400); auth
`authentication` (401), `permission_denied` (403), `payment_required` (402); availability `rate_limit_exceeded`
(429), `provider_overloaded` (503), `provider_unavailable` (502); validation `invalid_request`, `invalid_prompt`
(400), `not_found` (404), `precondition_failed` (412), `payload_too_large` (413), `unprocessable` (422); policy
`content_policy_violation`, `refusal` (403); image errors `invalid_image`, `image_too_large`, `image_too_small`,
`unsupported_image_format`, `image_download_failed` (400), `image_not_found` (404); generic `server` (500),
`timeout` (504), `unmapped` (500). That is the complete `ApiErrorType` enum (27 values).
`[OR: api_reference/errors-and-debugging]` `[OR-API: ApiErrorType]` Image errors are fatal for the judge call
(the same input fails again).

OpenRouter retries inside one request: before any token reaches the client it "automatically tries a backup
provider"; "Failover also stops once part of the answer has reached you". `[OR: api_reference/errors-and-debugging]`
Zero-completion insurance: no charge when a response has zero completion tokens and a blank finish reason, or an
error finish reason. `[OR: guides/features/zero-completion-insurance]`

### 9.4 Rate limits and headers

- OpenRouter paid models have "no platform-level request cap"; `429` then comes from the upstream provider (with
  `metadata.provider_code`). "Successful inference responses do not include `X-RateLimit-*` headers." On a
  platform-limit `429` the error response carries `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`;
  `Retry-After` may be present on `429` and `503` ("When every attempted provider returned a retry hint") and on the
  in-flight-budget `402`. `GET /api/v1/key` reports credit and budget headroom. "Making additional accounts or API
  keys will not affect your rate limits". `[OR: api_reference/limits]` `[OR: api_reference/errors-and-debugging]`
- OpenAI direct sends `x-ratelimit-limit-requests`, `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-requests`,
  `x-ratelimit-remaining-tokens`, `x-ratelimit-reset-requests`, `x-ratelimit-reset-tokens` on responses, and
  `Retry-After` on some `429`/`503`. `[DOC: https://developers.openai.com/api/docs/guides/rate-limits.md]`
- DeepSeek limits concurrency (2500 for `deepseek-flash`, 500 for `deepseek-v4-pro`) and answers `429` beyond it.
  `[DOC: https://api-docs.deepseek.com/quick_start/rate_limit]`
- Groq adds non-standard statuses `498` (flex capacity) and `499`; "You will not be charged for requests that
  return server error codes." `[DOC: https://console.groq.com/docs/errors.md]`

---

## 10. Worked example (fixtures)

Field names and nesting are as documented (OpenRouter flavour unless stated). Ids, text, token counts, cost and
signature strings are invented placeholders. `native_finish_reason` values are illustrative: the docs only show
`"stop"`, `"max_tokens"` / `"MAX_TOKENS"` ("any native finish reason that normalizes to `length`, such as
`max_tokens` or `MAX_TOKENS`" `[OR: guides/routing/auto-exacto]`) and, for Gemini 4, `"CONTINUATION"`; never
branch on it except for the two Gemini 4 values.
Checked by the verification pass: every response fixture below parses under strict pydantic validation of the
SDK's `ChatCompletion` type (`ChatCompletion.model_validate`), with the OpenRouter/DeepSeek additions landing in
`model_extra`, **except** the two error fixtures in 10.5, which fail it on purpose (`finish_reason: "error"` is
not in the SDK literal; the error-only body lacks `choices`/`created`/`model`/`object`) and which the SDK's
default lenient parser nevertheless returns as a `ChatCompletion` without raising. Every 10.6 chunk validates as
`ChatCompletionChunk`; the request fixtures validate against `CompletionCreateParams` once the two OpenRouter
keys (`reasoning`, `provider`) are set aside.
Unknown keys may appear anywhere: fixtures should also be run with extra keys added, to prove the parser ignores them.

### 10.1 Request 1 (system + user + tool)

```json
{
  "model": "anthropic/claude-fable-5.2",
  "messages": [
    {"role": "system", "content": "You are taking part in a memory test. ..."},
    {"role": "user", "content": "Painting: The Great Wave off Kanagawa, by Katsushika Hokusai, c. 1831.\n\nType it as a grid 32 cells wide and 22 cells tall. ..."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "write_rows",
        "description": "Store palette colours and picture rows. It checks format only. A stored row can be replaced by sending it again.",
        "parameters": {
          "type": "object",
          "properties": {
            "palette": {"type": "array", "items": {"type": "string"}, "description": "Optional. Colours to define, each written as a one-character code then red, green and blue from 0 to 255, for example 'k 20 40 90'."},
            "start_row": {"type": "integer", "description": "The row number of the first row in 'rows'. Row 1 is the top."},
            "rows": {"type": "array", "items": {"type": "string"}, "description": "Consecutive rows, starting at start_row. Spaces are ignored."}
          }
        }
      }
    }
  ],
  "max_tokens": 128000,
  "reasoning": {"effort": "max"},
  "provider": {"require_parameters": true}
}
```

(The streaming transport adds `"stream": true`, plus `"stream_options": {"include_usage": true}` when
`stream_usage` is on; recommended off for OpenRouter, section 1.6.)

### 10.2 Response 1 (reasoning + one tool call), non-streaming

```json
{
  "id": "gen-1795000000-AbCdEfGhIjKlMnOpQrSt",
  "object": "chat.completion",
  "created": 1795000000,
  "model": "anthropic/claude-fable-5.2",
  "system_fingerprint": null,
  "choices": [
    {
      "index": 0,
      "finish_reason": "tool_calls",
      "native_finish_reason": "tool_use",
      "message": {
        "role": "assistant",
        "content": null,
        "refusal": null,
        "reasoning": "The wave curls from the left; Fuji is small, low and right of centre.",
        "reasoning_details": [
          {
            "type": "reasoning.text",
            "text": "The wave curls from the left; Fuji is small, low and right of centre.",
            "signature": "EqQBCkYIBxgCKkDPLACEHOLDERSIGNATURE==",
            "id": "reasoning-text-1",
            "format": "anthropic-claude-v1",
            "index": 0
          }
        ],
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "write_rows",
              "arguments": "{\"palette\": [\"k 20 40 90\", \"w 240 240 230\"], \"start_row\": 1, \"rows\": [\"kkkkkkkk kkkkkkkk kkkkwwww wwwwkkkk\"]}"
            }
          }
        ]
      }
    }
  ],
  "usage": {
    "prompt_tokens": 812,
    "completion_tokens": 2304,
    "total_tokens": 3116,
    "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 2101},
    "cost": 0.098656,
    "is_byok": false,
    "cost_details": {"upstream_inference_cost": null, "upstream_inference_prompt_cost": 0.006496, "upstream_inference_completions_cost": 0.09216}
  }
}
```

Expected parse: text `""`; one tool call `id="call_abc123"`, `name="write_rows"`, arguments parsed to a dict;
stop `tool_calls`; usage input 812, output 2304, reasoning 2101, cached 0; `provider_payload` = the `message`
object with the `null` `refusal` dropped and `content: null` kept.

OpenAI-family models through OpenRouter return summary and encrypted entries instead (same parse, same replay):

```json
{
  "role": "assistant",
  "content": null,
  "reasoning": "Planning the composition before typing rows.",
  "reasoning_details": [
    {"type": "reasoning.summary", "summary": "Planning the composition before typing rows.", "id": "rs_0a1b2c", "format": "openai-responses-v1", "index": 0},
    {"type": "reasoning.encrypted", "data": "gAAAAABPLACEHOLDERCIPHERTEXT", "id": "rs_0a1b2c", "format": "openai-responses-v1", "index": 0}
  ],
  "tool_calls": [
    {"id": "call_5T0GBcLA27", "type": "function", "function": {"name": "write_rows", "arguments": "{\"start_row\": 1, \"rows\": [\"kkkk\"]}"}}
  ]
}
```

A `reasoning_content` server (DeepSeek, llama.cpp, Fireworks) returns, for the same turn:

```json
{
  "id": "071f229d-daa0-4557-a580-106a22e8df73",
  "object": "chat.completion",
  "created": 1795000000,
  "model": "deepseek-flash",
  "choices": [
    {
      "index": 0,
      "finish_reason": "tool_calls",
      "message": {
        "role": "assistant",
        "content": "",
        "reasoning_content": "The wave curls from the left; Fuji is small.",
        "tool_calls": [
          {"index": 0, "id": "call_00_kw66qNnNto11bSfJVIdlV5Oo", "type": "function", "function": {"name": "write_rows", "arguments": "{\"start_row\": 1, \"rows\": [\"kkkk\"]}"}}
        ]
      }
    }
  ],
  "usage": {"prompt_tokens": 700, "completion_tokens": 300, "total_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 250}, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 700}
}
```

### 10.3 Request 2 (replayed assistant turn + tool result)

```json
{
  "model": "anthropic/claude-fable-5.2",
  "messages": [
    {"role": "system", "content": "You are taking part in a memory test. ..."},
    {"role": "user", "content": "Painting: The Great Wave off Kanagawa, by Katsushika Hokusai, c. 1831.\n\nType it as a grid 32 cells wide and 22 cells tall. ..."},
    {
      "role": "assistant",
      "content": null,
      "reasoning": "The wave curls from the left; Fuji is small, low and right of centre.",
      "reasoning_details": [
        {
          "type": "reasoning.text",
          "text": "The wave curls from the left; Fuji is small, low and right of centre.",
          "signature": "EqQBCkYIBxgCKkDPLACEHOLDERSIGNATURE==",
          "id": "reasoning-text-1",
          "format": "anthropic-claude-v1",
          "index": 0
        }
      ],
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "write_rows",
            "arguments": "{\"palette\": [\"k 20 40 90\", \"w 240 240 230\"], \"start_row\": 1, \"rows\": [\"kkkkkkkk kkkkkkkk kkkkwwww wwwwkkkk\"]}"
          }
        }
      ]
    },
    {"role": "tool", "tool_call_id": "call_abc123", "content": "Stored row 1. 21 rows to go."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "write_rows",
        "description": "Store palette colours and picture rows. It checks format only. A stored row can be replaced by sending it again.",
        "parameters": {
          "type": "object",
          "properties": {
            "palette": {"type": "array", "items": {"type": "string"}, "description": "Optional. Colours to define, each written as a one-character code then red, green and blue from 0 to 255, for example 'k 20 40 90'."},
            "start_row": {"type": "integer", "description": "The row number of the first row in 'rows'. Row 1 is the top."},
            "rows": {"type": "array", "items": {"type": "string"}, "description": "Consecutive rows, starting at start_row. Spaces are ignored."}
          }
        }
      }
    }
  ],
  "max_tokens": 128000,
  "reasoning": {"effort": "max"},
  "provider": {"require_parameters": true}
}
```

A test should assert that `messages[2]` equals the received message minus `null`-valued keys other than `content`,
and that `messages[0:2]`, `tools` and the settings are byte-identical to request 1.

### 10.4 Response 2 (final text)

```json
{
  "id": "gen-1795000900-ZyXwVuTsRqPoNmLkJiHg",
  "object": "chat.completion",
  "created": 1795000900,
  "model": "anthropic/claude-fable-5.2",
  "system_fingerprint": null,
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "native_finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "All 22 rows are stored. The wave fills the left two thirds; Fuji sits low and right. I was unsure about the boats.",
        "refusal": null,
        "reasoning": "All rows accepted; summarise.",
        "reasoning_details": [
          {"type": "reasoning.text", "text": "All rows accepted; summarise.", "signature": "EqQBPLACEHOLDER2==", "id": "reasoning-text-2", "format": "anthropic-claude-v1", "index": 0}
        ]
      }
    }
  ],
  "usage": {"prompt_tokens": 3150, "completion_tokens": 190, "total_tokens": 3340, "prompt_tokens_details": {"cached_tokens": 3000}, "completion_tokens_details": {"reasoning_tokens": 60}, "cost": 0.0087}
}
```

Expected parse: text as given, no tool calls, stop `end`, cached input 3000.

### 10.5 Other responses

Refusal as model output (HTTP 200):

```json
{
  "id": "gen-1795001000-refusal0000000000001",
  "object": "chat.completion",
  "created": 1795001000,
  "model": "anthropic/claude-fable-5.2",
  "system_fingerprint": null,
  "choices": [
    {
      "index": 0,
      "finish_reason": "content_filter",
      "native_finish_reason": "refusal",
      "message": {"role": "assistant", "content": null, "refusal": "I can't help with that request."}
    }
  ],
  "usage": {"prompt_tokens": 812, "completion_tokens": 12, "total_tokens": 824}
}
```

Refusal / policy block as an error (HTTP 403 body; the same object can be the whole body of a 200, or the `error`
of a stream chunk):

```json
{
  "error": {
    "code": 403,
    "message": "This content was flagged for possible biological risk. If this seems wrong, try rephrasing your request.",
    "metadata": {"error_type": "refusal", "provider_code": "bio_policy"}
  }
}
```

OpenRouter moderation block (HTTP 403):

```json
{
  "error": {
    "code": 403,
    "message": "Input flagged by moderation.",
    "metadata": {"reasons": ["violence"], "flagged_input": "...", "provider_name": "OpenAI", "model_slug": "openai/gpt-6.2-astra"}
  }
}
```

Max-tokens truncation where reasoning consumed the budget (HTTP 200):

```json
{
  "id": "gen-1795001100-length00000000000001",
  "object": "chat.completion",
  "created": 1795001100,
  "model": "openai/gpt-6.2-sol",
  "system_fingerprint": null,
  "choices": [
    {
      "index": 0,
      "finish_reason": "length",
      "native_finish_reason": "length",
      "message": {
        "role": "assistant",
        "content": "",
        "refusal": null,
        "reasoning": null,
        "reasoning_details": [
          {"type": "reasoning.encrypted", "data": "gAAAAABPLACEHOLDER", "id": "rs_0d0d0d", "format": "openai-responses-v1", "index": 0}
        ]
      }
    }
  ],
  "usage": {"prompt_tokens": 812, "completion_tokens": 302, "total_tokens": 1114, "completion_tokens_details": {"reasoning_tokens": 301}}
}
```

Max-tokens truncation in the middle of a tool call (arguments are not valid JSON):

```json
{
  "id": "gen-1795001200-length00000000000002",
  "object": "chat.completion",
  "created": 1795001200,
  "model": "anthropic/claude-fable-5.2",
  "system_fingerprint": null,
  "choices": [
    {
      "index": 0,
      "finish_reason": "length",
      "native_finish_reason": "max_tokens",
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {"id": "call_trunc1", "type": "function", "function": {"name": "write_rows", "arguments": "{\"start_row\": 9, \"rows\": [\"kkkkkkkk kkkkww"}}
        ]
      }
    }
  ],
  "usage": {"prompt_tokens": 2000, "completion_tokens": 128000, "total_tokens": 130000}
}
```

Expected parse: stop `max_tokens`; the tool call's `arguments` is the raw string (not valid JSON).

Two tool calls in one reply:

```json
{
  "id": "gen-1795001300-twocalls000000000001",
  "object": "chat.completion",
  "created": 1795001300,
  "model": "openai/gpt-6.2-astra",
  "system_fingerprint": null,
  "choices": [
    {
      "index": 0,
      "finish_reason": "tool_calls",
      "native_finish_reason": "completed",
      "message": {
        "role": "assistant",
        "content": "Sending rows 1-8 and 9-16.",
        "refusal": null,
        "tool_calls": [
          {"id": "call_aaa111", "type": "function", "function": {"name": "write_rows", "arguments": "{\"palette\": [\"k 20 40 90\"], \"start_row\": 1, \"rows\": [\"kkkk\", \"kkkk\"]}"}},
          {"id": "call_bbb222", "type": "function", "function": {"name": "write_rows", "arguments": "{\"start_row\": 9, \"rows\": [\"kkkk\"]}"}}
        ]
      }
    }
  ],
  "usage": {"prompt_tokens": 900, "completion_tokens": 400, "total_tokens": 1300}
}
```

Follow-up messages for it (after the replayed assistant message):

```json
[
  {"role": "tool", "tool_call_id": "call_aaa111", "content": "Stored rows 1-2."},
  {"role": "tool", "tool_call_id": "call_bbb222", "content": "Stored row 9."}
]
```

HTTP 200 whose body is only an error (non-streaming; may be preceded by whitespace):

```json
{
  "id": "gen-1795001400-err00000000000000001",
  "error": {"code": 502, "message": "Provider returned error", "metadata": {"error_type": "provider_unavailable"}}
}
```

HTTP 200 with partial content and a choice-level error:

```json
{
  "id": "gen-1795001500-err00000000000000002",
  "object": "chat.completion",
  "created": 1795001500,
  "model": "anthropic/claude-fable-5.2",
  "choices": [
    {
      "index": 0,
      "finish_reason": "error",
      "message": {"role": "assistant", "content": "partial output..."},
      "error": {"code": 502, "message": "Provider disconnected mid-stream", "metadata": {"error_type": "provider_unavailable"}}
    }
  ]
}
```

Pre-response error with a status code (HTTP 429, header `Retry-After: 60` possible):

```json
{"error": {"code": 429, "message": "Rate limit exceeded", "metadata": {"error_type": "rate_limit_exceeded", "provider_code": "rate_limited"}}}
```

### 10.6 Stream for response 1 (SSE bytes the server sends)

Each `data:` line is one JSON object on a single line followed by a blank line; comment lines start with `:`.
The reasoning deltas follow the one streaming example in OpenRouter's docs (`type`, `text`, `signature: null`, `id`,
`format`, `index` on each delta) and the "signature-only delta" described in OpenRouter's SDK source; treat the
exact split as representative, not guaranteed.

```text
: OPENROUTER PROCESSING

data: {"id":"gen-1795000000-AbCdEfGhIjKlMnOpQrSt","object":"chat.completion.chunk","created":1795000000,"model":"anthropic/claude-fable-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning":"The wave curls from the left; ","reasoning_details":[{"type":"reasoning.text","text":"The wave curls from the left; ","signature":null,"id":"reasoning-text-1","format":"anthropic-claude-v1","index":0}]},"finish_reason":null,"native_finish_reason":null}]}

: OPENROUTER PROCESSING

data: {"id":"gen-1795000000-AbCdEfGhIjKlMnOpQrSt","object":"chat.completion.chunk","created":1795000000,"model":"anthropic/claude-fable-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning":"Fuji is small, low and right of centre.","reasoning_details":[{"type":"reasoning.text","text":"Fuji is small, low and right of centre.","signature":null,"id":"reasoning-text-1","format":"anthropic-claude-v1","index":0}]},"finish_reason":null,"native_finish_reason":null}]}

data: {"id":"gen-1795000000-AbCdEfGhIjKlMnOpQrSt","object":"chat.completion.chunk","created":1795000000,"model":"anthropic/claude-fable-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning_details":[{"type":"reasoning.text","signature":"EqQBCkYIBxgCKkDPLACEHOLDERSIGNATURE==","id":"reasoning-text-1","format":"anthropic-claude-v1","index":0}]},"finish_reason":null,"native_finish_reason":null}]}

data: {"id":"gen-1795000000-AbCdEfGhIjKlMnOpQrSt","object":"chat.completion.chunk","created":1795000000,"model":"anthropic/claude-fable-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"write_rows","arguments":""}}]},"finish_reason":null,"native_finish_reason":null}]}

data: {"id":"gen-1795000000-AbCdEfGhIjKlMnOpQrSt","object":"chat.completion.chunk","created":1795000000,"model":"anthropic/claude-fable-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"function":{"arguments":"{\"palette\": [\"k 20 40 90\", \"w 240 240 230\"], \"start_row\": 1, "}}]},"finish_reason":null,"native_finish_reason":null}]}

data: {"id":"gen-1795000000-AbCdEfGhIjKlMnOpQrSt","object":"chat.completion.chunk","created":1795000000,"model":"anthropic/claude-fable-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"function":{"arguments":"\"rows\": [\"kkkkkkkk kkkkkkkk kkkkwwww wwwwkkkk\"]}"}}]},"finish_reason":null,"native_finish_reason":null}]}

data: {"id":"gen-1795000000-AbCdEfGhIjKlMnOpQrSt","object":"chat.completion.chunk","created":1795000000,"model":"anthropic/claude-fable-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":"tool_calls","native_finish_reason":"tool_use"}]}

data: {"id":"gen-1795000000-AbCdEfGhIjKlMnOpQrSt","object":"chat.completion.chunk","created":1795000000,"model":"anthropic/claude-fable-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":"tool_calls","native_finish_reason":"tool_use"}],"usage":{"prompt_tokens":812,"completion_tokens":2304,"total_tokens":3116,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens_details":{"reasoning_tokens":2101},"cost":0.098656}}

data: [DONE]

```

`accumulate()` on this stream must produce the `choices[0].message` of 10.2 (without the `refusal: null` key; with
`content: null`), `finish_reason: "tool_calls"` and the `usage` of the last chunk.

Variants worth a test each (all handled by `accumulate()` in 1.6, all exercised offline):

- OpenAI-standard ending: last content chunk `{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
  then a usage chunk `{"choices":[],"usage":{...}}`, then `data: [DONE]`.
- `reasoning_content` deltas (`"delta":{"reasoning_content":"..."}`) and vLLM/Ollama `reasoning` deltas.
- A whole tool call in one chunk (`id`, `name` and complete `arguments` together).
- `id` and `name` repeated on every argument chunk.
- Two parallel calls with `index` 0 and 1 interleaved; two calls that both use `index` 0 with different ids; a call
  without `index` and without `id`.
- No `index` at all: the same `id` repeated on every fragment (one call); two whole calls with different ids (two
  calls); no id and the same name on two whole calls (two calls); no id and the name repeated on argument fragments
  (one call).
- Empty-string ids on two parallel calls (-> `call_0`, `call_1`); duplicate ids (second is replaced); a server id
  `call_1` next to an empty id (-> `call_1`, `call_1_`).
- `arguments` sent as an object instead of a string (-> `json.dumps`).
- Stream that ends with no `finish_reason` and no `[DONE]` -> error; stream with zero chunks -> error.

Mid-stream error (last line of a stream; the SDK raises `openai.APIError` when it reads it, with `e.body` equal to
the `error` object):

```text
data: {"id":"gen-1795000000-AbCdEfGhIjKlMnOpQrSt","object":"chat.completion.chunk","created":1795000000,"model":"anthropic/claude-fable-5.2","provider":"Anthropic","error":{"code":502,"message":"Provider disconnected unexpectedly","metadata":{"error_type":"provider_unavailable"}},"choices":[{"index":0,"delta":{"content":""},"finish_reason":"error"}]}

```

### 10.7 How to build a fake stream in a test

The most faithful mock is at the transport level, so the SDK's real SSE parser, comment handling and error-chunk
handling are exercised: `[OFFLINE]`

```python
import httpx
from openai import OpenAI

def fake_client(sse_bytes: bytes = b"", *, json_body=None, status=200, headers=None) -> OpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        handler.requests.append(request)                       # inspect json.loads(request.content) in tests
        if json_body is not None:
            return httpx.Response(status, json=json_body, headers=headers)
        return httpx.Response(status, content=sse_bytes, headers={"content-type": "text/event-stream"})
    handler.requests = []
    client = OpenAI(base_url="https://example.invalid/api/v1", api_key="test", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    client._handler = handler
    return client
```

There is no SDK helper to recommend for the final message: the adapter's `accumulate()` is the helper. (The SDK's
own `.stream(...).get_final_completion()` is the thing not to use, section 1.7.)

---

## 11. Gotchas and surprises

1. **First-party OpenAI and tools.** OpenAI's guides: "Use the Responses API for function calling. Chat Completions
   does not support function calling with GPT-6 Astra or GPT-6.1 Sol."; "GPT-6 Astra and GPT-6.1 Sol require the
   Responses API for tool calling."; "GPT-6.2 Astra and GPT-6.2 Sol support Chat Completions, but tool calling
   requires Responses." `[DOC: https://developers.openai.com/api/docs/guides/reasoning.md]`
   `[DOC: https://developers.openai.com/api/docs/guides/function-calling.md]`
   `[DOC: https://developers.openai.com/api/docs/guides/latest-model.md]` Pointing `openai_compat` at
   `api.openai.com` for contestants is therefore wrong for the flagship models; via OpenRouter the catalog lists
   `tools` for `openai/gpt-6.2-astra` on `/api/v1/chat/completions`. `[OR-CAT]`
2. **The SDK stream helper corrupts compat streams** (role `"assistantassistant..."`, concatenated `id`/`format`,
   `RuntimeError` without `index`). Section 1.7. `[OFFLINE]`
3. **HTTP 200 is not success.** Error-only bodies, choice-level errors, `finish_reason: "error"`, and (streaming)
   error chunks, truncated streams and zero-chunk streams. The SDK turns an error-only 200 into a `ChatCompletion`
   with `choices=None` without raising. Sections 3.7, 1.6. `[OFFLINE]`
4. **Mid-stream transport failures are raw `httpx` exceptions**, not `openai.OpenAIError`. Section 9.1. `[OFFLINE]`
5. **`finish_reason` lies about tool calls** (`stop` with calls present; unknown values). Use the presence of
   `tool_calls`. Section 3.4.
6. **`content` is not always a string**: `null`, `""`, or a list (Mistral native thinking chunks; OpenRouter schema
   allows arrays). Section 3.2.
7. **Three names for reasoning text** (`reasoning`, `reasoning_content`, inside `content`) plus OpenRouter's
   `reasoning_details`; vLLM renamed its field from `reasoning_content` to `reasoning`. Section 5.1.
8. **DeepSeek returns 400 if `reasoning_content` is not replayed** on any assistant turn once `tools` is in the
   request, including turns without a tool call. Section 4.2.
9. **OpenRouter drops unsupported parameters silently** unless `provider.require_parameters: true`; with it, an
   unlisted parameter (`parallel_tool_calls`, `max_completion_tokens` for most models) can leave no provider.
   Sections 2.4, 2.6, 2.8.
10. **Disabling or forcing things fails on current Claude**: `effort: "none"` / `enabled: false` (400), forced
    `tool_choice` (400), assistant prefill (unsupported). Sections 2.3, 5.2, 5.5.
11. **History edits break reasoning replay** (prefix-bound thinking): identical system prompt, tool list and
    earlier messages on every call. Section 4.6.
12. **Gemini 4 (Argon) through OpenRouter stops with `length` by design** when `max_tokens > 65,536`, and a hop at
    `high` effort can be all reasoning with empty content; such a stop is resumed by resending with the stopped
    turn as the last assistant message (each hop's stopped turn *replaces* the previous one, never stacks), not by
    a user nudge. With `max_tokens <= 65,536` there is no continuation hop at all. Section 6.3.
13. **`max_tokens` naming and meaning differ**: Ollama only reads `max_tokens`; OpenAI o-series only
    `max_completion_tokens`; Fireworks rejects both together; on xAI the limit excludes reasoning. Section 2.6.
14. **Usage conventions differ**: reasoning tokens are inside `completion_tokens` on OpenAI/OpenRouter, outside on
    xAI direct. Section 3.6.
15. **`max_tokens=None` is sent as `null`** by the SDK; omit keys instead. `extra_body` overrides explicit
    arguments. Section 1.6. `[OFFLINE]`
16. **Environment leakage**: with `api_key=None` / `base_url=None` the SDK falls back to `OPENAI_API_KEY` /
    `OPENAI_BASE_URL`, and it forwards `OPENAI_ORG_ID` / `OPENAI_PROJECT_ID` as headers to any base URL. Section 1.2.
17. **SDK retries re-send long calls**: a read timeout on a non-streaming request is retried twice by default.
    Section 9.2. `[OFFLINE]`
18. **Tool-call ids can be empty, null or duplicated**, two parallel calls can share `index` 0 in a stream, and a
    delta can come with no `index` at all. Section 1.6.
19. **Ollama specifics**: no `tool_choice`; `max_completion_tokens` ignored; rejects a replayed assistant message
    with `content: null` and no tool calls, and a replayed call with invalid-JSON arguments; context size is fixed by
    the model's `num_ctx`. Sections 2.3, 2.6, 4.3, 6.2.
20. **llama.cpp specifics**: default port is 9931; parallel tool calls off by default; reasoning in
    `reasoning_content`; images need `--mmproj`. Sections 1.3, 2.4, 7.2.
21. **vLLM specifics**: tool calling needs `--enable-auto-tool-choice --tool-call-parser ...`; separate reasoning
    needs `--reasoning-parser ...`; "tool calling only parses functions from the `content` field, not from the
    `reasoning`"; `user` is ignored; `image_url.detail` unsupported. `[DOC: .../vllm/main/docs/features/tool_calling.md]`
    `[DOC: .../vllm/main/docs/features/reasoning_outputs.md]`
22. **Groq specifics**: 400 for `messages[].name`, `logprobs`, `logit_bias`, `n != 1`; `temperature: 0` is
    converted to `1e-8`; vision models take 3 or 5 images. Sections 2.7, 7.2.
23. **Mistral has two endpoints**: `/openai/v1` (OpenAI-compatible, preview) and `/v1` (native: list-valued
    content, `tool_choice: "any"`). Use the first. Section 1.3.
24. **xAI Chat Completions is "legacy"** and cannot carry encrypted reasoning; for Grok use OpenRouter (its
    `reasoning_details` formats include `xai-responses-v1`). Section 4.2.
25. **OpenRouter reasoning text is often a summary or absent** (Claude "summarized", OpenAI summaries, o-series
    none); the token count in `usage` is the real measure. Section 3.6.
26. **`stream_options` on OpenRouter is a no-op and usage always arrives**, in a chunk that repeats the finish
    reason rather than one with empty `choices`. Section 1.6.
27. **Streaming + Anthropic tools on OpenRouter** turns on fine-grained tool streaming ("OpenRouter sets
    `eager_input_streaming: true` on each tool so Anthropic emits tool arguments as a sequence of incremental
    chunks rather than buffering them"). `[OR: guides/routing/provider-selection]` Expect many small argument
    fragments; validate the assembled JSON.
28. **Installed SDK is older than the API** (`openai 2.53.0`); irrelevant here because every non-standard field
    goes through `extra_body` and every response is read as a dict, but its typed `ChatModel` list and literals must
    not be used for validation.

---

## UNVERIFIED (from memory)

Nothing below was confirmed from a fetched page or the SDK. Treat as hints only.

- OpenAI-style servers reject a request in which an assistant message with `tool_calls` is not followed by a tool
  message for every `tool_call_id` (error text along the lines of "An assistant message with 'tool_calls' must be
  followed by tool messages responding to each 'tool_call_id'"). The documented loops all satisfy this, but I did
  not find the rule stated.
- Some servers or chat templates reject `content: null` on an assistant message with tool calls and want `""`.
  (Verified only for the opposite Ollama case in 4.3.)
- Some providers reject an empty-string tool result; a non-empty string such as `"ok"` is safer.
- Mistral's API validates tool-call ids as 9 alphanumeric characters and uses `finish_reason` values
  `model_length` and `error`. (vLLM's doc mentions truncating ids "to the last 9 digits" for Mistral models, which is
  consistent, but I did not fetch Mistral's API reference.)
- Which servers, if any, currently return `function.arguments` as a JSON object rather than a string. The parser in
  3.3 handles it either way.
- Ollama's default context window is small (a few thousand tokens) and prompts beyond it are truncated silently.
- Whether llama.cpp's server accepts `max_completion_tokens` and `stream_options.include_usage`.
- On GCP, idle TCP flows through Cloud NAT are dropped after about 20 minutes; relevant only to non-streaming calls
  to servers that send no keep-alive bytes.
- How often OpenRouter emits `: OPENROUTER PROCESSING` or non-streaming whitespace (the docs say "occasionally" /
  "periodic"); the `read=900` timeout in 1.2 is a judgement, not a documented figure.
- How OpenRouter answers when `provider.require_parameters: true` leaves no endpoint (I recall a 404 saying no
  endpoints support the requested parameters), and whether `stream_options` counts as a parameter for that check.
- Whether OpenRouter includes an `index` key on non-streaming `tool_calls` entries and a top-level `provider` key on
  non-streaming responses (the schema shows neither; the stream-error example shows `provider`).
- Google's own OpenAI-compatible endpoint (not covered here) is said to put thought signatures in
  `tool_calls[].extra_content`; verbatim replay of the whole message would cover it.
- (Added by the verification pass.) Every concrete value of `native_finish_reason` used in the section 10 fixtures
  other than `"stop"`, `"max_tokens"`, `"MAX_TOKENS"` and `"CONTINUATION"` (that is: `"tool_use"`, `"refusal"`,
  `"length"`, `"completed"`) is a guess at the upstream vocabulary; no fetched page shows them. Tests may carry
  them; adapter logic must not depend on them.
- (Added by the verification pass.) The fixture tool-call ids (`call_abc123`, ...) and reasoning ids
  (`reasoning-text-1`, `rs_...`) are shaped like the documentation examples, not like any real provider's ids
  (ids are opaque; nothing should parse them).
- (Added by the verification pass.) Whether OpenRouter always sends `native_finish_reason` on a normal choice. It
  is in the TypeScript types and examples of the overview page but not in the OpenAPI `ChatChoice` schema.
- (Added by the verification pass.) Gemini 4 long decode together with tool calls, and what to replay on the turn
  after a multi-hop answer (6.3).
- (Added by the verification pass.) That a streamed OpenRouter reply for a current Claude / GPT-6 / Gemini model
  really splits `reasoning_details` the way fixture 10.6 does (one text block, signature in a trailing
  signature-only delta). The merge rule is OpenRouter's own; the fixture's split is an illustration. If a replay
  after streaming is rejected, the documented fallback is `stream: false` (5.6 item 3).

---

## Sources

Read 2026-11-19 / 2026-11-20.

Installed SDK (paths under `site-packages/`):

- `openai/_version.py` (2.53.0), `openai/_constants.py`, `openai/_client.py`, `openai/_base_client.py`,
  `openai/_exceptions.py`, `openai/_streaming.py`
- `openai/resources/chat/completions/completions.py`
- `openai/lib/streaming/chat/_completions.py`, `openai/lib/streaming/_deltas.py`
- `openai/types/chat/completion_create_params.py`, `chat_completion.py`, `chat_completion_message.py`,
  `chat_completion_chunk.py`, `chat_completion_message_function_tool_call.py`, `chat_completion_message_tool_call.py`,
  `chat_completion_message_param.py`, `chat_completion_system_message_param.py`,
  `chat_completion_developer_message_param.py`, `chat_completion_user_message_param.py`,
  `chat_completion_assistant_message_param.py`, `chat_completion_tool_message_param.py`,
  `chat_completion_content_part_param.py`, `chat_completion_content_part_text_param.py`,
  `chat_completion_content_part_image_param.py`, `chat_completion_function_tool_param.py`,
  `chat_completion_message_function_tool_call_param.py`, `chat_completion_tool_choice_option_param.py`,
  `chat_completion_stream_options_param.py`
- `openai/types/completion_usage.py`, `openai/types/shared/reasoning_effort.py`,
  `openai/types/shared_params/function_definition.py`, `function_parameters.py`
- `openai-2.53.0.dist-info/METADATA` (the SDK README)
- `httpx/_config.py`, `httpx/_transports/default.py`, `httpx/_exceptions.py` (httpx 0.28.1)

Offline probes (`[OFFLINE]`): throwaway scripts run with the installed SDK against
`httpx.MockTransport`; no network. They covered: headers and URL built by the SDK; `extra_body` merge; pass-through
of unknown message keys, image parts and `content: null`; parsing of unknown fields, `finish_reason: "error"`,
object-valued `arguments`, list-valued `content`; error-only and whitespace-prefixed 200 bodies; non-JSON 200
bodies; retry counts and sleeps for statuses 400-529 with and without `Retry-After`; timeout and connection
exceptions; SSE comments, error chunks, truncated and empty streams; mid-stream transport exceptions; the SDK
stream helper's output on OpenRouter-style chunks; and the `accumulate()` function in section 1.6.

OpenRouter:

- https://openrouter.ai/docs/llms.txt, https://openrouter.ai/docs/_llms/docs.txt,
  https://openrouter.ai/docs/_llms/api-reference.txt, https://openrouter.ai/docs/llms-full.txt
- https://openrouter.ai/docs/api_reference/overview.md
- https://openrouter.ai/docs/api_reference/streaming.md
- https://openrouter.ai/docs/api_reference/errors-and-debugging.md
- https://openrouter.ai/docs/api_reference/limits.md
- https://openrouter.ai/docs/api_reference/authentication.md
- https://openrouter.ai/docs/api_reference/parameters.md
- https://openrouter.ai/docs/api_reference/models.md
- https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion.md (OpenAPI for `POST /chat/completions`)
- https://openrouter.ai/docs/guides/features/tool-calling.md
- https://openrouter.ai/docs/guides/best-practices/reasoning-tokens.md
- https://openrouter.ai/docs/guides/features/compatibility.md
- https://openrouter.ai/docs/guides/community/openai-sdk.md
- https://openrouter.ai/docs/guides/overview/multimodal/image-understanding.md
- https://openrouter.ai/docs/guides/overview/multimodal/overview.md
- https://openrouter.ai/docs/guides/overview/models.md
- https://openrouter.ai/docs/guides/routing/provider-selection.md
- https://openrouter.ai/docs/guides/routing/auto-exacto.md
- https://openrouter.ai/docs/guides/best-practices/prompt-caching.md
- https://openrouter.ai/docs/guides/best-practices/latency-and-performance.md
- https://openrouter.ai/docs/guides/features/zero-completion-insurance.md
- https://openrouter.ai/docs/guides/features/message-transforms.md
- https://openrouter.ai/docs/guides/features/plugins/response-healing.md
- https://openrouter.ai/docs/app-attribution.md
- https://openrouter.ai/docs/faq.md
- From llms-full.txt: https://openrouter.ai/docs/guides/features/plugins/alignment,
  https://openrouter.ai/docs/guides/features/broadcast,
  https://openrouter.ai/docs/cookbook/administration/usage-accounting,
  https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/claude-4-6, `.../claude-4-7`,
  `.../claude-5-5`, `.../fable-5-1`, `.../gemini-4`, `.../gpt-6`
- https://openrouter.ai/api/v1/models (catalog; also with `?supported_parameters=tools&input_modalities=image`)
- https://raw.githubusercontent.com/OpenRouterTeam/ai-sdk-provider/main/src/chat/index.ts (OpenRouter's AI SDK
  provider: streaming merge of `reasoning_details`, tool-call delta handling, finish-reason fixes)

OpenAI:

- https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create.md
- https://developers.openai.com/api/docs/guides/function-calling.md
- https://developers.openai.com/api/docs/guides/reasoning.md
- https://developers.openai.com/api/docs/guides/latest-model.md
- https://developers.openai.com/api/docs/guides/images-vision.md
- https://developers.openai.com/api/docs/guides/rate-limits.md
- https://developers.openai.com/api/docs/guides/error-codes.md
- https://developers.openai.com/api/docs/pricing.md, https://developers.openai.com/api/docs/models.md
- https://platform.openai.com/docs/llms.txt, https://developers.openai.com/llms.txt

vLLM:

- https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/tool_calling.md
- https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/reasoning_outputs.md
- https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/interleaved_thinking.md
- https://raw.githubusercontent.com/vllm-project/vllm/main/docs/serving/online_serving/openai_compatible_server.md
- https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/entrypoints/openai/chat_completion/protocol.py

Ollama:

- https://raw.githubusercontent.com/ollama/ollama/main/docs/api/openai-compatibility.mdx
- https://raw.githubusercontent.com/ollama/ollama/main/openai/openai.go

llama.cpp:

- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/function-calling.md
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/multimodal.md

LM Studio: https://lmstudio.ai/llms.txt (full documentation export)

httpx: https://www.python-httpx.org/advanced/timeouts/

DeepSeek:

- https://api-docs.deepseek.com/ , https://api-docs.deepseek.com/guides/thinking_mode ,
  https://api-docs.deepseek.com/api/create-chat-completion , https://api-docs.deepseek.com/quick_start/rate_limit ,
  https://api-docs.deepseek.com/quick_start/error_codes

Groq: https://console.groq.com/docs/openai.md , https://console.groq.com/docs/reasoning.md ,
https://console.groq.com/docs/tool-use/local-tool-calling.md , https://console.groq.com/docs/vision.md ,
https://console.groq.com/docs/errors.md , https://console.groq.com/docs/api-reference.md

Together: https://docs.together.ai/docs/inference/openai-compatibility.md ,
https://docs.together.ai/docs/inference/chat/reasoning.md ,
https://docs.together.ai/docs/inference/function-calling/overview.md ,
https://docs.together.ai/docs/inference/vision/inputs.md , https://docs.together.ai/reference/chat-completions-1.md

Fireworks: https://docs.fireworks.ai/guides/reasoning.md , https://docs.fireworks.ai/guides/function-calling.md ,
https://docs.fireworks.ai/guides/querying-vision-language-models.md ,
https://docs.fireworks.ai/api-reference/post-chatcompletions.md

Mistral: https://docs.mistral.ai/llms.txt , https://docs.mistral.ai/llms-full.txt (sections chat-completion,
function-calling, reasoning)

xAI: https://docs.x.ai/llms.txt , https://docs.x.ai/developers/model-capabilities/text/reasoning.md ,
https://docs.x.ai/developers/tools/function-calling.md ,
https://docs.x.ai/developers/model-capabilities/images/understanding.md ,
https://docs.x.ai/developers/rest-api-reference/inference/chat-completions.md ,
https://docs.x.ai/developers/model-capabilities/legacy/chat-completions.md

---

## Verification (independent pass)

Done 2026-11-20 by a second reviewer, who did not write the notes. No model/inference endpoint was called. Network
use: `GET`, with a generic User-Agent, of 74 URLs (all HTTP 200): every page and source file that a claim in
sections 0-11 is cited to (all `[OR: ...]`, `[OR-API]`, `[DOC: ...]` and `[SRC: ...]` targets, checked by script),
plus `llms.txt` / `llms-full.txt` and the public catalog `https://openrouter.ai/api/v1/models` (unauthenticated;
it runs no model). A few index pages that are listed under "Sources" but that no claim cites were not re-fetched.
Everything else was the installed SDK, read and executed offline against `httpx.MockTransport`.

### What was checked, and how

1. **Verbatim quotes.** A script extracted double-quoted prose phrases from the notes and searched for each
   (whitespace- and markup-normalised) in the re-fetched pages and in the SDK source. About 170 phrases were
   isolated and found in a fetched source; the only genuine misses were two quotes that elided words (corrected
   below). The script cannot isolate quotes that sit next to code spans containing quote characters, so those
   were not all machine-checked; the load-bearing ones among them (replay rules, reasoning parameters, finish
   reasons, error shapes, limits, vendor-specific rejections) were located by hand with `grep` in the cited page.
2. **OpenRouter request/response schema.** Read the OpenAPI excerpt for `POST /chat/completions` schema by schema:
   `ChatRequest` (including `reasoning.{effort,max_tokens,enabled,exclude,summary,context,mode}` and the effort
   enum `max, xhigh, high, medium, low, minimal, none`), `ProviderPreferences`, `ChatFunctionTool`,
   `ChatToolChoice`, `ChatStreamOptions`, `ChatResult`, `ChatChoice`, `ChatAssistantMessage`, `ChatToolMessage`,
   `ChatUserMessage`, `ChatToolCall`, `ChatUsage`, `CostDetails`, `ChatStreamChunk`, `ChatStreamChoice`,
   `ChatStreamDelta`, `ChatStreamToolCall`, `ChatFinishReasonEnum`, `ApiErrorType`, `ChatContentImage`,
   `ReasoningDetailUnion` and its four members, `ReasoningFormat`, and the declared status codes.
3. **OpenRouter guides.** Read in full or in the relevant sections: reasoning tokens (replay rule, the Go and
   cURL replay loops, `reasoning_details` shapes, legacy parameters, per-model discovery), API overview (TypeScript
   request/response types), streaming (keep-alive comments, final usage chunk, cancellation, mid-stream errors),
   errors and debugging (200-with-error bodies, typed error codes, refusal vs policy block), limits, provider
   selection (`require_parameters`, soft preferences, Anthropic beta features), tool calling, prompt caching,
   image understanding, models reference/guide, and the Claude 4.6 / 4.7 / 5.5, Fable 5.1, Gemini 4 and GPT-6
   migration guides.
4. **OpenRouter's AI SDK provider source** (`src/chat/index.ts`): the `reasoning_details` merge, the tool-call
   delta slotting rules, and the finish-reason overrides were read and compared line by line with `accumulate()`.
5. **Catalog.** Parsed the re-fetched catalog and compared every model row in 5.3, 6.1 and 8 (id,
   `canonical_slug`, context, max output, input/output/cache-read price, `pricing.overrides`, `supported_efforts`,
   `default_effort`, `mandatory`, `tools`, image input) and every count quoted in 2.4, 2.6, 7.2 and 8
   (479 entries, 387 chat-capable, 327 / 251 / 223, 18 with `parallel_tool_calls`, 370 / 94 for the two
   output-limit names). All matched except one price row.
6. **Other vendors.** For vLLM, Ollama, llama.cpp, LM Studio, DeepSeek, Groq, Together, Fireworks, Mistral, xAI and
   first-party OpenAI, the claims that an adapter acts on were located by `grep` in the cited page or source file:
   base URL and key variable, the output-limit parameter name, reasoning controls and their values, the field that
   carries reasoning text and the replay rule, extra finish reasons, extra usage fields, keep-alive behaviour,
   `stream_options` handling, unsupported request fields, and image limits. No discrepancy was found. Purely
   informational cells were spot-checked rather than exhaustively checked.
7. **SDK types** (`openai 2.53.0`): `ChatCompletion`, `Choice.finish_reason` literal, `ChatCompletionMessage`,
   `ChatCompletionMessageFunctionToolCall`, `ChatCompletionChunk` / `ChoiceDelta` / `ChoiceDeltaToolCall`,
   `CompletionUsage` and its two detail types, `FunctionDefinition`, the tool / assistant / developer / image
   param types, `ReasoningEffort`, `ChatCompletionStreamOptionsParam`, `ChatCompletionToolChoiceOptionParam`,
   the `max_tokens` / `max_completion_tokens` / `stop` / `tool_choice` docstrings, `_constants.py`,
   `OpenAI.__init__`, `auth_headers`, `default_headers`, `_make_status_error`, `_should_retry`,
   `_calculate_retry_timeout`, `_parse_retry_after_header`, `Stream.__stream__`, `accumulate_delta`.
8. **Fixtures (section 10), field by field.** Every JSON block in the notes parses as JSON. Each response fixture
   was passed to `ChatCompletion.model_validate` (strict pydantic validation of the SDK's own type): all validate
   except the two error fixtures, which are not meant to (see the note at the top of section 10). Each 10.6 chunk
   validates as `ChatCompletionChunk`. Request fixtures validate against `CompletionCreateParams`, the tool
   definition against `ChatCompletionFunctionToolParam`, tool messages against `ChatCompletionToolMessageParam`,
   the judge message against `ChatCompletionUserMessageParam`.
9. **Code in the notes, executed as written** (extracted from this file, not retyped): `call()` on the 10.6 SSE
   bytes through the real SDK stream parser returns exactly the 10.2 message (minus `refusal: null`), with
   `finish_reason: "tool_calls"` and the last chunk's usage; `call(stream=False)` returns the 10.2 body unchanged;
   the mid-stream error line raises `openai.APIError` (not `APIStatusError`) with `e.body` equal to the `error`
   object; a JSON 200 reply to a streamed request and a truncated stream raise the two `RuntimeError`s; `parse_args`
   and `fake_client` behave as described.
10. **Every `[OFFLINE]` claim re-run**: request URL and default headers; the `Omit()` header map; `OPENAI_ORG_ID`
    / `OPENAI_PROJECT_ID` forwarding; `extra_body` overriding an explicit argument; `max_tokens=None` sent as
    `null`; unknown message keys and `content: null` passed through; `with_raw_response` attributes; `model_dump()`
    vs `to_dict()`; error-only 200 -> `ChatCompletion` with `choices=None`; non-JSON 200 -> `str`; whitespace-only
    body -> `JSONDecodeError`; list-valued `content`, object-valued `arguments` and a missing id accepted without
    error; status -> exception class for 400-529; retry counts and sleeps (429 x3, `retry-after: 7` -> 7.0 s twice,
    `retry-after: 500` -> one attempt, 402 never retried, 408/409/5xx retried, `x-should-retry: true` honoured);
    timeouts and connect errors -> `APITimeoutError` / `APIConnectionError` after 3 attempts; mid-stream transport
    failures -> raw `httpx` exceptions, one attempt; the SDK stream helper's corruption of an OpenRouter-style
    stream (`role` = `"assistant"` x8, `id` and `format` tripled, `RuntimeError` when `index` is missing).

### Corrections made (old -> new)

1. **8, price row** `deepseek/deepseek-v4.1-flash`: `0.051 / 0.567 / 0.018` -> `0.30 / 1.20 / 0.006` (catalog on
   2026-11-20; no entry carries the old figures). Added a note that catalog prices move and that `usage.cost` is
   the figure to use.
2. **6.3, Gemini 4 continuation loop**: "append the assistant message (verbatim) and call again" -> each hop's
   stopped turn **replaces** the previous one (`base + [{"role": "assistant", "content": "", "reasoning_details":
   [latest gemini-continuation item]}]`), as the guide's own loop does; stacking stopped turns breaks the
   "resent unchanged" requirement. Also stated that a `MAX_TOKENS` hop still returns a token, and recommended the
   no-loop option (`max_tokens <= 65536`) for contestants. Gotcha 12 reworded to match.
3. **1.6, `accumulate()`**: a tool-call delta **without `index`** that repeated the call's `id` (or its name) was
   treated as a new call, splitting one call into two (second one with `name: None`). Replaced the one-line rule
   by the rule in OpenRouter's provider source (repeat of the latest id continues; no id and no name continues; a
   repeated name starts a new call only after the previous arguments are complete JSON). Added `_complete_json`.
4. **1.6, `accumulate()`**: a synthesised id `call_<i>` could equal an id the server sent for another call (for
   example a real `call_1` followed by an empty id) -> the synthesised id is now made unique against all ids.
   Added `import json` to the block so it is self-contained.
5. **3.1, citation**: `native_finish_reason` and choice-level `error` were attributed to the OpenAPI `ChatChoice`
   schema -> they are only in the overview page's TypeScript types and the errors page; the OpenAPI `ChatChoice` /
   `ChatStreamChoice` do not list them. Both must be read as optional. Listed the other OpenAPI-only extras
   (`audio`, `deferred_tools_details`, `service_tier`, `alignment`, `safeguard_results`).
6. **3.6, usage table**: `completion_tokens_details.image_tokens` was cited to the SDK -> it is not in the SDK type
   nor in the OpenAPI schema, only in OpenRouter's TypeScript type; split into its own row.
7. **1.2**: "An empty key raises `OpenAIError` at construction" -> true only when `OPENAI_ADMIN_KEY` is unset; with
   it set the constructor succeeds and the first call raises `TypeError` before sending. Added `OPENAI_ADMIN_KEY`
   and `OPENAI_WEBHOOK_SECRET` to the variables the constructor reads (neither is sent on this endpoint) and what
   `OPENAI_CUSTOM_HEADERS` does.
8. **3.2**: "pydantic only warns" -> parsing does not warn; the warning appears on re-serialisation
   (`to_dict()` / `model_dump()`).
9. **7.2 and 9.3, image error types**: four listed -> six (`image_download_failed` 400, `image_not_found` 404
   added); 9.3 now gives the complete 27-value `ApiErrorType` enum.
10. **2.6 and 5.4, two quotes** restored to the exact source wording ("Maximum tokens (deprecated, use
    max_completion_tokens). ..."; "On Chat Completions (`/api/v1/chat/completions`), the same budget is ...").
11. **8**: "`:batch` variants are half price" -> exactly half for 69 of 89 batch entries (all Claude, GPT-6 and
    Gemini 4 ones); others vary.
12. **10 intro**: said the docs show only `"stop"`, `"MAX_TOKENS"`, `"CONTINUATION"` as native finish reasons ->
    `"max_tokens"` is also shown (auto-exacto page). Added the fixture-validation statement.
13. **10.6 variant list** extended with the index-less, id-collision and object-arguments cases now covered.
14. **UNVERIFIED**: added five items (invented `native_finish_reason` values and ids in fixtures; whether
    `native_finish_reason` is always present; Gemini 4 long decode with tool calls; the real chunking of streamed
    `reasoning_details`).

Nothing was found wrong in: the tool definition shape, where tool calls / ids / arguments appear, the tool
message shape, the replay rule (whole assistant message including `reasoning_details`), the `reasoning` object and
its allowed values, the per-model highest efforts, the finish-reason enumeration, the refusal signals, the usage
field names, the keep-alive and timeout facts, the SDK exception and retry tables, or the model ids.

### What remains unverified

- Everything under "UNVERIFIED (from memory)".
- Live behaviour of any server: nothing here was observed on the wire. In particular the exact shape and split of
  streamed `reasoning_details` for current Claude / GPT-6 / Gemini models, which the streamed replay depends on
  (non-streaming replay does not), and how often OpenRouter sends keep-alive bytes.
- Server-side limits that the docs do not state: OpenRouter's maximum request duration, the per-request image
  count for Claude and Gemini through OpenRouter, and the default output limit when `max_tokens` is omitted for
  models other than Gemini 4.
- Vendor rows other than OpenRouter were checked claim by claim against the cited page, but those pages were not
  read end to end; a contradicting statement elsewhere on a vendor's site would not have been seen.
- First-party prices other than OpenAI's were not fetched (as the notes already say).
- The catalog is a snapshot; ids, prices and effort lists can change after 2026-11-20.
