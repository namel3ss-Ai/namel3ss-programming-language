"""Central registry for Namel3ss tool plugins."""

from __future__ import annotations

from typing import Dict, Type

from .base import ToolPlugin


class PluginRegistryError(RuntimeError):
    """Raised when plugin registration or lookup fails."""


_PLUGINS: Dict[str, Dict[str, Type[ToolPlugin]]] = {}


def register_plugin(category: str, name: str, plugin_cls: Type[ToolPlugin]) -> None:
    """Register ``plugin_cls`` for the (category, name) tuple."""

    category_key = (category or "").strip()
    plugin_name = (name or "").strip()
    if not category_key or not plugin_name:
        raise PluginRegistryError("Plugin category and name must be provided")
    bucket = _PLUGINS.setdefault(category_key, {})
    if plugin_name in bucket:
        raise PluginRegistryError(f"Plugin '{plugin_name}' already registered for category '{category_key}'")
    bucket[plugin_name] = plugin_cls


def get_plugin(category: str, name: str) -> Type[ToolPlugin]:
    """Return the registered plugin class for ``(category, name)``."""

    category_key = (category or "").strip()
    plugin_name = (name or "").strip()
    bucket = _PLUGINS.get(category_key)
    if not bucket:
        raise PluginRegistryError(f"No plugins registered for category '{category_key}'")
    plugin_cls = bucket.get(plugin_name)
    if plugin_cls is None:
        raise PluginRegistryError(f"Plugin '{plugin_name}' is not registered for category '{category_key}'")
    return plugin_cls


def clear_registry() -> None:
    """Internal helper for tests to reset registry state."""

    _PLUGINS.clear()


__all__ = ["PluginRegistryError", "register_plugin", "get_plugin", "clear_registry"]
