"""The adapter interface: one class per API family, one method that matters.

An adapter turns the neutral messages in `paintbench.types` into one
provider's request, makes the call, and turns the response back into a
`Reply`. The harness and the judge know nothing else about any provider.

Rules every adapter follows:

- `complete()` is the only method that touches the network.
- Split the work into `build_request()` (pure: messages, tools and settings in,
  the exact keyword arguments or JSON body out) and `parse_response()` (pure:
  the provider's response as plain JSON data in, a `Reply` out). Tests inspect
  the first and feed documented response shapes to the second.
- Send the tool exactly as given: same name, same descriptions, same schema.
  Never add a tool, a system prompt or any other content of your own.
- Never send temperature or any other sampling parameter; they stay at the
  provider default.
- An assistant `Message` whose `provider` is this adapter's must be replayed
  from its `provider_payload` verbatim, so reasoning state survives tool-result
  turns. An assistant message from elsewhere (a test, another adapter) is
  rebuilt from its text and tool calls.
- `Reply.provider_payload` and `Reply.raw` must be JSON-serialisable.
- Raise `TransientError` for failures worth retrying (rate limits, overload,
  timeouts, connection errors, 5xx) and `FatalError` for the rest (bad
  request, authentication, unknown model). The harness owns backoff and
  restarts; keep the SDK's own retries low.
- Build the client lazily on first use, so constructing an adapter needs no
  credentials and no SDK import. A client passed to the constructor is used
  as is (tests pass fakes).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..types import Message, ModelSettings, Reply, ToolSpec


class AdapterError(Exception):
    """A model call failed."""

    retryable = False

    def __init__(self, message: str, *, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after  # seconds, when the provider said how long to wait


class TransientError(AdapterError):
    """Worth retrying: rate limit, overload, timeout, connection error, 5xx."""

    retryable = True


class FatalError(AdapterError):
    """Not worth retrying: bad request, authentication, unknown model, missing SDK."""


class Adapter(ABC):
    provider: ClassVar[str] = ""

    def __init__(self, client: Any = None):
        self._client = client

    @abstractmethod
    def build_request(self, messages: list[Message], tools: list[ToolSpec], settings: ModelSettings) -> dict[str, Any]:
        """The exact request this adapter would send. Pure; no network, no client."""

    @abstractmethod
    def parse_response(self, response: Any) -> Reply:
        """Turn the provider's response (as plain JSON data) into a Reply. Pure."""

    @abstractmethod
    def complete(self, messages: list[Message], tools: list[ToolSpec], settings: ModelSettings) -> Reply:
        """Make one model call."""
