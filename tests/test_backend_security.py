from __future__ import annotations

from pathlib import Path

import pytest

from namel3ss.parser import Parser
from namel3ss.codegen.backend.core.generator import generate_backend

from tests.backend_test_utils import load_backend_module


def _build_secure_backend(tmp_path: Path, source: str | None = None) -> Path:
    sample_source = source or (
        'app "Secure".\n'
        '\n'
        'page "Home" at "/":\n'
        '  show text "Hello"\n'
    )
    app = Parser(sample_source).parse()
    backend_dir = tmp_path / "secure_backend"
    generate_backend(app, backend_dir)
    return backend_dir


def test_models_router_does_not_expose_python_endpoint(tmp_path: Path) -> None:
    backend_dir = _build_secure_backend(tmp_path)
    models_router = (backend_dir / "generated" / "routers" / "models.py").read_text(encoding="utf-8")
    assert "/api/python/" not in models_router


def test_rate_limits_trigger(monkeypatch, tmp_path: Path) -> None:
    backend_dir = _build_secure_backend(tmp_path)
    monkeypatch.setenv("NAMEL3SS_RATE_LIMIT_AI", "2/second")
    with load_backend_module(tmp_path, backend_dir, monkeypatch) as module:
        limiter = getattr(module.runtime, "enforce_rate_limit")
        limiter("ai", "tester")
        limiter("ai", "tester")
        with pytest.raises(Exception) as excinfo:
            limiter("ai", "tester")
        assert "Rate limit" in str(excinfo.value)


def test_csrf_tokens_are_validated(monkeypatch, tmp_path: Path) -> None:
    backend_dir = _build_secure_backend(tmp_path)
    with load_backend_module(tmp_path, backend_dir, monkeypatch) as module:
        ensure_cookie = getattr(module.runtime, "ensure_csrf_cookie")
        cookie_value, should_set = ensure_cookie({})
        assert should_set is True
        cookie_name = module.runtime.csrf_cookie_name()
        header_name = module.runtime.csrf_header_name()
        headers = {header_name: cookie_value.split(":", 1)[0]}
        cookies = {cookie_name: cookie_value}
        assert module.runtime.validate_csrf_request("POST", headers, cookies) is True
        headers[header_name] = "tampered"
        assert module.runtime.validate_csrf_request("POST", headers, cookies) is False
        assert module.runtime.validate_csrf_request("GET", {}, {}) is True
