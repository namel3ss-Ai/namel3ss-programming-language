"""Utilities for working with plugin metadata."""

from __future__ import annotations

from typing import Dict

from .base import (
    PLUGIN_CATEGORY_CUSTOM_TOOL,
    PLUGIN_CATEGORY_EMBEDDING_PROVIDER,
    PLUGIN_CATEGORY_GRAPH_DB,
    PLUGIN_CATEGORY_LLM_PROVIDER,
    PLUGIN_CATEGORY_VECTOR_STORE,
)

_CATEGORY_ALIASES: Dict[str, str] = {
    "llm": PLUGIN_CATEGORY_LLM_PROVIDER,
    "llm_provider": PLUGIN_CATEGORY_LLM_PROVIDER,
    "vector": PLUGIN_CATEGORY_VECTOR_STORE,
    "vector_store": PLUGIN_CATEGORY_VECTOR_STORE,
    "embedding": PLUGIN_CATEGORY_EMBEDDING_PROVIDER,
    "embedding_provider": PLUGIN_CATEGORY_EMBEDDING_PROVIDER,
    "graph": PLUGIN_CATEGORY_GRAPH_DB,
    "graph_db": PLUGIN_CATEGORY_GRAPH_DB,
    "custom": PLUGIN_CATEGORY_CUSTOM_TOOL,
    "custom_tool": PLUGIN_CATEGORY_CUSTOM_TOOL,
    "tool": PLUGIN_CATEGORY_CUSTOM_TOOL,
}


def normalize_plugin_category(value: str) -> str:
    """Return a canonical plugin category for ``value``."""

    key = (value or "").strip().lower()
    if not key:
        return ""
    return _CATEGORY_ALIASES.get(key, key)


__all__ = ["normalize_plugin_category"]
