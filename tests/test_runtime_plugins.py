from __future__ import annotations

from pathlib import Path
import uuid

import pytest

from namel3ss.codegen.backend import generate_backend
from namel3ss.parser import Parser
from namel3ss.plugins.registry import register_plugin
from namel3ss.plugins import registry as registry_module
from tests.backend_test_utils import load_backend_module


@pytest.fixture(autouse=True)
def restore_plugin_registry():
    snapshot = {category: dict(entries) for category, entries in registry_module._PLUGINS.items()}
    try:
        yield
    finally:
        registry_module._PLUGINS.clear()
        for category, entries in snapshot.items():
            registry_module._PLUGINS[category] = dict(entries)


class SampleToolPlugin:
    name = "test_tool_plugin"
    input_schema = None
    output_schema = None

    def __init__(self) -> None:
        self.config = {}
        self.calls = []

    def configure(self, config):
        self.config = dict(config)

    async def call(self, context, payload):
        self.calls.append(payload)
        return {"echo": payload}


class FailingToolPlugin:
    name = "failing_plugin"
    input_schema = None
    output_schema = None

    def configure(self, config):
        if "required" not in config:
            raise ValueError("missing required config")

    async def call(self, context, payload):
        return {}


class RecordingToolPlugin:
    name = "recording_plugin"
    input_schema = {"query": {"type": "string", "required": True}}
    output_schema = None

    def __init__(self) -> None:
        self.config = {}
        self.calls = []

    def configure(self, config):
        self.config = dict(config)

    async def call(self, context, payload):
        self.calls.append(payload)
        return {"matches": [payload.get("query"), payload.get("note")]}


class SchemaEnforcingPlugin:
    name = "schema_plugin"
    input_schema = {"query": {"type": "string", "required": True}}
    output_schema = None

    def __init__(self) -> None:
        self.called = False

    def configure(self, config):
        self.config = dict(config)

    async def call(self, context, payload):
        self.called = True
        return payload


def _build_runtime(source: str, tmp_path: Path, monkeypatch):
    app = Parser(source).parse_app()
    backend_dir = tmp_path / f"backend_{uuid.uuid4().hex}"
    generate_backend(app, backend_dir)
    return load_backend_module(tmp_path, backend_dir, monkeypatch)


def test_build_context_instantiates_registered_tool(tmp_path, monkeypatch):
    register_plugin("custom_tool", SampleToolPlugin.name, SampleToolPlugin)
    source = (
        'app "PluginApp".\n'
        '\n'
        'connector "test_tool" type custom_tool:\n'
        '  provider = "test_tool_plugin"\n'
        '  foo = "bar"\n'
        '  secret = env:TEST_SECRET\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "hello"\n'
    )
    monkeypatch.setenv("TEST_SECRET", "resolved-secret")
    with _build_runtime(source, tmp_path, monkeypatch) as module:
        runtime = module.runtime
        context = runtime.build_context(None)
        tools = context.get("tools")
        assert tools is not None
        plugin_instance = tools["test_tool"]
        assert isinstance(plugin_instance, SampleToolPlugin)
        assert plugin_instance.config["foo"] == "bar"
        assert plugin_instance.config["secret"] == "resolved-secret"


def test_unknown_plugin_raises_clear_error(tmp_path, monkeypatch):
    source = (
        'app "PluginApp".\n'
        '\n'
        'connector "missing_plugin" type custom_tool:\n'
        '  provider = "unregistered_plugin"\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "hi"\n'
    )
    with _build_runtime(source, tmp_path, monkeypatch) as module:
        runtime = module.runtime
        with pytest.raises(RuntimeError) as excinfo:
            runtime.build_context(None)
    assert "unknown plugin" in str(excinfo.value).lower()


def test_plugin_configuration_failure_surfaces(tmp_path, monkeypatch):
    register_plugin("custom_tool", FailingToolPlugin.name, FailingToolPlugin)
    source = (
        'app "PluginApp".\n'
        '\n'
        'connector "failing_tool" type custom_tool:\n'
        '  provider = "failing_plugin"\n'
        '  optional = "value"\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "hi"\n'
    )
    with _build_runtime(source, tmp_path, monkeypatch) as module:
        runtime = module.runtime
        with pytest.raises(RuntimeError) as excinfo:
            runtime.build_context(None)
    assert "failed to configure connector 'failing_tool'" in str(excinfo.value).lower()


def test_tool_chain_step_invocation(tmp_path, monkeypatch):
    register_plugin("custom_tool", RecordingToolPlugin.name, RecordingToolPlugin)
    source = (
        'app "ToolApp".\n'
        '\n'
        'connector "search_tool" type custom_tool:\n'
        '  provider = "recording_plugin"\n'
        '  note = "defaults"\n'
        '\n'
        'define chain "tool_chain":\n'
        '  steps:\n'
        '    - step "search":\n'
        '        kind: tool\n'
        '        target: search_tool\n'
        '        options:\n'
        '          query: ctx:payload.query\n'
        '          note: "beta"\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "hi"\n'
    )
    with _build_runtime(source, tmp_path, monkeypatch) as module:
        context = module.runtime.build_context("home")
        payload = {"query": "alpha"}
        result = module.run_chain("tool_chain", payload, context=context)
        assert result["status"] == "ok"
        plugin_instance = context["tools"]["search_tool"]
        assert isinstance(plugin_instance, RecordingToolPlugin)
        assert plugin_instance.calls[-1]["query"] == "alpha"
        assert result["result"]["matches"][0] == "alpha"


def test_tool_chain_unknown_target(tmp_path, monkeypatch):
    register_plugin("custom_tool", RecordingToolPlugin.name, RecordingToolPlugin)
    source = (
        'app "ToolApp".\n'
        '\n'
        'connector "search_tool" type custom_tool:\n'
        '  provider = "recording_plugin"\n'
        '\n'
        'define chain "tool_chain":\n'
        '  steps:\n'
        '    - step "missing":\n'
        '        kind: tool\n'
        '        target: unknown_tool\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "hi"\n'
    )
    with _build_runtime(source, tmp_path, monkeypatch) as module:
        context = module.runtime.build_context("home")
        result = module.run_chain("tool_chain", {"query": "alpha"}, context=context)
        assert result["status"] == "error"
        error_step = result["steps"][0]
        assert "unknown_tool" in error_step["output"]["error"]


def test_tool_schema_validation_failure(tmp_path, monkeypatch):
    register_plugin("custom_tool", SchemaEnforcingPlugin.name, SchemaEnforcingPlugin)
    source = (
        'app "ToolApp".\n'
        '\n'
        'connector "schema_tool" type custom_tool:\n'
        '  provider = "schema_plugin"\n'
        '\n'
        'define chain "tool_chain":\n'
        '  steps:\n'
        '    - step "invalid":\n'
        '        kind: tool\n'
        '        target: schema_tool\n'
        '        options:\n'
        '          top_k: 5\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "hi"\n'
    )
    with _build_runtime(source, tmp_path, monkeypatch) as module:
        context = module.runtime.build_context("home")
        plugin_instance = context["tools"]["schema_tool"]
        result = module.run_chain("tool_chain", {"query": "alpha"}, context=context)
        assert result["status"] == "error"
        assert plugin_instance.called is False
        error_step = result["steps"][0]
        assert "missing required field" in error_step["output"]["error"]
