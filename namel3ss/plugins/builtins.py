"""Registration of built-in tool plugins."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .base import (
    PLUGIN_CATEGORY_LLM_PROVIDER,
    ToolPlugin,
)
from .registry import register_plugin


class _LLMProviderPluginBase:
    """Shared helper for synchronous configuration bookkeeping."""

    name = "llm"
    input_schema: Optional[Mapping[str, Any]] = None
    output_schema: Optional[Mapping[str, Any]] = None

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}

    def configure(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)

    async def call(self, context: Dict[str, Any], payload: Dict[str, Any]) -> Any:
        raise RuntimeError(
            "LLM provider plugins are not yet invoked through the runtime plugin registry. "
            "Continue using the existing connector APIs."
        )


class OpenAILLMProviderPlugin(_LLMProviderPluginBase):
    """Adapter entry for OpenAI-compatible chat completion providers."""

    name = "openai"


class AnthropicLLMProviderPlugin(_LLMProviderPluginBase):
    """Adapter entry for Anthropic Claude-compatible providers."""

    name = "anthropic"


register_plugin(PLUGIN_CATEGORY_LLM_PROVIDER, OpenAILLMProviderPlugin.name, OpenAILLMProviderPlugin)
register_plugin(PLUGIN_CATEGORY_LLM_PROVIDER, AnthropicLLMProviderPlugin.name, AnthropicLLMProviderPlugin)

__all__ = [
    "OpenAILLMProviderPlugin",
    "AnthropicLLMProviderPlugin",
]
