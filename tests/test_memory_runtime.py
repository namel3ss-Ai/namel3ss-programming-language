"""Tests covering memory declarations and runtime behaviour."""

from __future__ import annotations

import pytest

from namel3ss.codegen.backend import generate_backend
from namel3ss.codegen.backend.state import build_backend_state
from namel3ss.parser import Parser

from tests.backend_test_utils import load_backend_module


@pytest.fixture
def runtime_module(tmp_path, monkeypatch):
    source = (
        'app "MemoryApp".\n'
        '\n'
        'memory "chat_history":\n'
        '  scope: session\n'
        '  kind: list\n'
        '  max_items: 2\n'
        '\n'
        'define template "echo":\n'
        '  prompt = "{input}"\n'
        '\n'
        'define chain "echo_chain":\n'
        '  input -> template echo read_memory chat_history write_memory chat_history\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "ok"\n'
    )
    app = Parser(source).parse_app()
    backend_dir = tmp_path / "backend_memory"
    generate_backend(app, backend_dir)
    with load_backend_module(tmp_path, backend_dir, monkeypatch) as module:
        yield module


def test_chain_updates_memory_entries(runtime_module):
    context = runtime_module.runtime.build_context("home")
    runtime_module.run_chain("echo_chain", {"input": "hello"}, context=context)
    runtime_module.run_chain("echo_chain", {"input": "second"}, context=context)
    result = runtime_module.run_chain("echo_chain", {"input": "third"}, context=context)

    memory_state = context["memory_state"]["chat_history"]["entries"]
    assert [entry["value"] for entry in memory_state] == ["second", "third"]
    inputs = result["steps"][0]["inputs"]
    assert "chat_history" in inputs["memory"]


def test_chain_invalid_memory_reference_fails(tmp_path):
    source = (
        'app "Invalid".\n'
        '\n'
        'define template "echo":\n'
        '  prompt = "{input}"\n'
        '\n'
        'define chain "broken":\n'
        '  input -> template echo write_memory missing_store\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "ok"\n'
    )
    app = Parser(source).parse_app()
    with pytest.raises(ValueError):
        build_backend_state(app)
