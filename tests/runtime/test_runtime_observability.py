"""Runtime observability helpers tests."""

from __future__ import annotations

from pathlib import Path

from namel3ss.codegen.backend import generate_backend
from namel3ss.parser import Parser

from tests.backend_test_utils import load_backend_module


def _mini_app():
    source = (
        'app "Telemetry".\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "hello"\n'
    )
    return Parser(source).parse_app()


def test_request_id_helpers(tmp_path: Path, monkeypatch) -> None:
    backend_dir = tmp_path / "backend_obs"
    generate_backend(_mini_app(), backend_dir)

    with load_backend_module(tmp_path, backend_dir, monkeypatch) as module:
        runtime = module.runtime
        assert runtime.current_request_id() is None
        runtime.bind_request_id("req-123")
        assert runtime.current_request_id() == "req-123"
        runtime.clear_request_id()
        assert runtime.current_request_id() is None


def test_tracing_span_is_noop_without_sdk(tmp_path: Path, monkeypatch) -> None:
    backend_dir = tmp_path / "backend_obs_tracing"
    generate_backend(_mini_app(), backend_dir)

    with load_backend_module(tmp_path, backend_dir, monkeypatch) as module:
        runtime = module.runtime
        with runtime.tracing_span("namel3ss.test", {"component": "unit"}) as span:
            if span is not None:
                assert hasattr(span, "get_span_context")
