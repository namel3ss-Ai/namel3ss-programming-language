"""Core plugin abstractions for Namel3ss tool integrations."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

PLUGIN_CATEGORY_LLM_PROVIDER = "llm_provider"
PLUGIN_CATEGORY_EMBEDDING_PROVIDER = "embedding_provider"
PLUGIN_CATEGORY_VECTOR_STORE = "vector_store"
PLUGIN_CATEGORY_GRAPH_DB = "graph_db"
PLUGIN_CATEGORY_CUSTOM_TOOL = "custom_tool"
PLUGIN_CATEGORY_EVALUATOR = "evaluator"


@runtime_checkable
class ToolPlugin(Protocol):
    """Base interface for Namel3ss tools/connectors."""

    name: str
    input_schema: Optional[Mapping[str, Any]]
    output_schema: Optional[Mapping[str, Any]]

    def configure(self, config: Mapping[str, Any]) -> None:
        """Configure the plugin with validated settings."""

    async def call(self, context: Dict[str, Any], payload: Dict[str, Any]) -> Any:
        """Execute the tool using the supplied context and payload."""


__all__ = [
    "ToolPlugin",
    "PLUGIN_CATEGORY_LLM_PROVIDER",
    "PLUGIN_CATEGORY_EMBEDDING_PROVIDER",
    "PLUGIN_CATEGORY_VECTOR_STORE",
    "PLUGIN_CATEGORY_GRAPH_DB",
    "PLUGIN_CATEGORY_CUSTOM_TOOL",
    "PLUGIN_CATEGORY_EVALUATOR",
]
