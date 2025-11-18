from __future__ import annotations

from typing import Any, Dict, Mapping

import pytest

from namel3ss.plugins.base import ToolPlugin
from namel3ss.plugins.registry import PluginRegistryError, get_plugin, register_plugin
from namel3ss.plugins import registry as registry_module


@pytest.fixture(autouse=True)
def _restore_registry():
    original = {category: dict(entries) for category, entries in registry_module._PLUGINS.items()}
    try:
        yield
    finally:
        registry_module._PLUGINS.clear()
        registry_module._PLUGINS.update({category: dict(entries) for category, entries in original.items()})


class _FakeToolPlugin:
    name = "fake"

    def __init__(self) -> None:
        self.config: Dict[str, Any] = {}

    def configure(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    async def call(self, context: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"context": context, "payload": payload}


def test_register_and_lookup_plugin():
    register_plugin("custom_category", _FakeToolPlugin.name, _FakeToolPlugin)
    plugin_cls = get_plugin("custom_category", "fake")
    assert plugin_cls is _FakeToolPlugin


def test_register_duplicate_raises():
    register_plugin("dup_category", _FakeToolPlugin.name, _FakeToolPlugin)
    with pytest.raises(PluginRegistryError):
        register_plugin("dup_category", _FakeToolPlugin.name, _FakeToolPlugin)


def test_lookup_missing_plugin_raises():
    with pytest.raises(PluginRegistryError):
        get_plugin("missing_category", "unknown")
