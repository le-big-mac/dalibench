"""Adapters, one per API family. Each is imported only when asked for, so the
core installs and runs without any provider SDK."""

from __future__ import annotations

import importlib
from typing import Any

from .base import Adapter, AdapterError, FatalError, TransientError

# provider name -> (module, class)
REGISTRY: dict[str, tuple[str, str]] = {
    "anthropic": ("paintbench.adapters.anthropic_api", "AnthropicAdapter"),
    "openai": ("paintbench.adapters.openai_responses", "OpenAIAdapter"),
    "google": ("paintbench.adapters.google_genai", "GoogleAdapter"),
    "openai_compat": ("paintbench.adapters.openai_compat", "OpenAICompatAdapter"),
    "fake": ("paintbench.adapters.fake", "FakeAdapter"),
}


def get_adapter(provider: str, client: Any = None) -> Adapter:
    """A new adapter for a provider name. `client` replaces the real SDK client (tests)."""
    try:
        module_name, class_name = REGISTRY[provider]
    except KeyError:
        raise ValueError(f"unknown provider {provider!r}; known: {sorted(REGISTRY)}") from None
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(client=client)


__all__ = ["Adapter", "AdapterError", "FatalError", "TransientError", "REGISTRY", "get_adapter"]
