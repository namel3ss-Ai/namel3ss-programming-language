from __future__ import annotations

import sys
import types
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from namel3ss.codegen.backend.core import BackendState, _render_runtime_module

STUB_SCHEMA_TYPES = [
    "ActionResponse",
    "ChartResponse",
    "FormResponse",
    "InsightResponse",
    "PredictionResponse",
    "ExperimentResult",
    "TableResponse",
    "RuntimeErrorPayload",
]


def _install_runtime_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    httpx_module = types.ModuleType("httpx")

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self) -> None:  # pragma: no cover - stub
            return None

        def json(self):  # pragma: no cover - stub
            return self._payload

    class AsyncClient:  # pragma: no cover - stub
        def __init__(self, *args, **kwargs):
            self._init_kwargs = dict(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return _Response([])

        async def post(self, *args, **kwargs):
            return _Response({})

        async def request(self, method: str, *args, **kwargs):  # pragma: no cover - stub
            method_lower = (method or "").lower()
            handler = getattr(self, method_lower, self.get)
            return await handler(*args, **kwargs)

    httpx_module.AsyncClient = AsyncClient
    monkeypatch.setitem(sys.modules, "httpx", httpx_module)

    fastapi_module = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code=500, detail=None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    fastapi_module.HTTPException = HTTPException
    fastapi_module.FastAPI = type("FastAPI", (), {})
    fastapi_module.APIRouter = type("APIRouter", (), {})
    fastapi_module.Depends = lambda dependency=None: dependency
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_module)

    fastapi_responses = types.ModuleType("fastapi.responses")

    class StreamingResponse:  # pragma: no cover - stub
        def __init__(self, content=None, media_type: str | None = None):
            self.content = content
            self.media_type = media_type

    fastapi_responses.StreamingResponse = StreamingResponse
    monkeypatch.setitem(sys.modules, "fastapi.responses", fastapi_responses)
    fastapi_module.responses = fastapi_responses

    sqlalchemy_module = types.ModuleType("sqlalchemy")

    class MetaData:  # pragma: no cover - stub
        pass

    def text(value: str) -> SimpleNamespace:  # pragma: no cover - stub
        return SimpleNamespace(text=value)

    sqlalchemy_module.MetaData = MetaData
    sqlalchemy_module.text = text

    sqlalchemy_ext = types.ModuleType("sqlalchemy.ext")
    sqlalchemy_ext_asyncio = types.ModuleType("sqlalchemy.ext.asyncio")

    class AsyncSession:  # pragma: no cover - stub
        pass

    sqlalchemy_ext_asyncio.AsyncSession = AsyncSession
    sqlalchemy_ext.asyncio = sqlalchemy_ext_asyncio
    monkeypatch.setitem(sys.modules, "sqlalchemy.ext", sqlalchemy_ext)
    monkeypatch.setitem(sys.modules, "sqlalchemy.ext.asyncio", sqlalchemy_ext_asyncio)

    sqlalchemy_sql = types.ModuleType("sqlalchemy.sql")
    sqlalchemy_sql.Select = type("Select", (), {})
    sqlalchemy_sql.table = lambda name: name
    sqlalchemy_sql.column = lambda name: name
    monkeypatch.setitem(sys.modules, "sqlalchemy.sql", sqlalchemy_sql)

    sqlalchemy_module.ext = sqlalchemy_ext
    sqlalchemy_module.sql = sqlalchemy_sql
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy_module)


def _build_runtime_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    _install_runtime_stubs(monkeypatch)

    generated_pkg = types.ModuleType("generated")
    generated_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "generated", generated_pkg)

    schemas_module = types.ModuleType("generated.schemas")
    for name in STUB_SCHEMA_TYPES:
        setattr(schemas_module, name, type(name, (), {}))
    generated_pkg.schemas = schemas_module
    monkeypatch.setitem(sys.modules, "generated.schemas", schemas_module)

    state = BackendState(
        app={"name": "Test", "database": None, "theme": {}, "variables": []},
        datasets={},
        connectors={},
        ai_connectors={},
        insights={},
        models={},
        templates={},
        chains={},
        experiments={},
        crud_resources={},
        pages=[],
        env_keys=[],
    )

    source = _render_runtime_module(state, embed_insights=False, enable_realtime=False)
    module_name = "generated.runtime_test_connectors"
    runtime_module = types.ModuleType(module_name)
    runtime_module.__package__ = "generated"
    monkeypatch.setitem(sys.modules, module_name, runtime_module)
    exec(source, runtime_module.__dict__)
    return runtime_module


@pytest.fixture()
def runtime_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _build_runtime_module(monkeypatch)


@pytest.mark.asyncio
async def test_default_grpc_driver_not_configured(
    runtime_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NAMEL3SS_ALLOW_STUBS", raising=False)
    connector = {
        "name": "grpc_misconfigured",
        "options": {"service": "ExampleService", "method": "DoWork"},
    }

    result = await runtime_module._default_grpc_driver(connector, {})

    assert result["status"] == "not_configured"
    assert "Missing gRPC configuration" in result["error"]
    assert result["config"] == {"service": "ExampleService", "method": "DoWork"}
    assert result["inputs"] == {"connector": "grpc_misconfigured"}
    assert result["metadata"]["elapsed_ms"] >= 0.0


@pytest.mark.asyncio
async def test_default_grpc_driver_with_pluggable_callable(
    runtime_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NAMEL3SS_ALLOW_STUBS", raising=False)
    captured_kwargs: dict = {}

    async def fake_driver(**kwargs):
        captured_kwargs.update(kwargs)
        return {"rows": [{"value": 1}]}

    monkeypatch.setattr(runtime_module, "_load_python_callable", lambda path: fake_driver)

    connector = {
        "name": "grpc_ok",
        "options": {
            "host": "api.example.com",
            "service": "ExampleService",
            "method": "DoWork",
            "metadata": {"authorization": "Bearer secret"},
            "payload": {"request": "data"},
            "driver": "package:driver",
        },
    }

    result = await runtime_module._default_grpc_driver(connector, {})

    assert result["status"] == "ok"
    assert result["result"] == {"rows": [{"value": 1}]}
    assert result["config"] is None
    assert result["metadata"]["endpoint"] == "api.example.com:443"
    assert result["metadata"]["service"] == "ExampleService"
    assert result["metadata"]["method"] == "DoWork"
    assert captured_kwargs["host"] == "api.example.com"
    assert captured_kwargs["service"] == "ExampleService"
    assert captured_kwargs["method"] == "DoWork"
    assert captured_kwargs["metadata"] == {"authorization": "Bearer secret"}
    assert captured_kwargs["payload"] == {"request": "data"}


@pytest.mark.asyncio
async def test_default_grpc_driver_reports_driver_errors(
    runtime_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NAMEL3SS_ALLOW_STUBS", raising=False)

    async def failing_driver(**kwargs):  # pragma: no cover - helper
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime_module, "_load_python_callable", lambda path: failing_driver)

    connector = {
        "name": "grpc_error",
        "options": {
            "host": "localhost",
            "service": "ExampleService",
            "method": "DoWork",
            "driver": "package:driver",
            "api_key": "super-secret",
        },
    }

    result = await runtime_module._default_grpc_driver(connector, {})

    assert result["status"] == "error"
    assert "RuntimeError" in result["error"]
    assert result["result"] is None
    assert result["config"]["api_key"] == "***"
    assert "elapsed_ms" in result["metadata"]


@pytest.mark.asyncio
async def test_default_grpc_driver_demo_mode_respects_env(
    runtime_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NAMEL3SS_ALLOW_STUBS", "true")
    connector = {
        "name": "grpc_demo",
        "options": {
            "host": "demo.example.com",
            "service": "ExampleService",
            "method": "DoWork",
            "api_key": "hidden",
        },
    }

    result = await runtime_module._default_grpc_driver(connector, {})

    assert result["status"] == "demo"
    assert result["result"][0]["service"] == "ExampleService"
    assert result["config"]["api_key"] == "***"
    assert result["metadata"]["endpoint"] == "demo.example.com:443"


@pytest.mark.asyncio
async def test_default_rest_driver_reports_missing_endpoint(
    runtime_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NAMEL3SS_ALLOW_STUBS", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    connector = {
        "name": "rest_missing",
        "options": {
            "endpoint": "env:API_BASE_URL",
            "params": {"api_key": "secret"},
            "headers": {"Authorization": "Bearer top-secret"},
        },
    }

    result = await runtime_module._default_rest_driver(connector, {})

    assert result["status"] == "not_configured"
    assert "missing env" in (result["error"] or "")
    assert result["config"]["params"]["api_key"] == "***"
    assert "missing_env" in result["metadata"]
    assert "API_BASE_URL" in result["metadata"]["missing_env"]
    assert result["inputs"]["headers"] == ["Authorization"]


@pytest.mark.asyncio
async def test_default_rest_driver_demo_mode(
    runtime_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NAMEL3SS_ALLOW_STUBS", "1")
    connector = {
        "name": "rest_demo",
        "options": {"demo": True, "headers": {"Authorization": "Bearer hidden"}},
    }

    result = await runtime_module._default_rest_driver(connector, {})

    assert result["status"] == "demo"
    assert result["config"]["headers"]["Authorization"] == "***"
    assert result["metadata"]["demo"] is True


@pytest.mark.asyncio
async def test_default_rest_driver_empty_response(
    runtime_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NAMEL3SS_ALLOW_STUBS", raising=False)
    connector = {
        "name": "rest_empty",
        "options": {
            "endpoint": "https://api.example.com/orders",
            "method": "GET",
            "params": {"tenant": "ctx:user.tenant"},
        },
    }

    context = {"user": {"tenant": "acme"}}
    result = await runtime_module._default_rest_driver(connector, context)

    assert result["status"] == "empty"
    assert result["rows"] is None
    assert result["inputs"]["params"] == ["tenant"]
    assert "status_code" not in result["metadata"]


def test_extract_rows_from_connector_response(runtime_module: ModuleType) -> None:
    rows = runtime_module._extract_rows_from_connector_response(
        {"status": "ok", "result": [1, {"value": 2}]}
    )
    assert rows == [{"value": 1}, {"value": 2}]

    batch_rows = runtime_module._extract_rows_from_connector_response(
        {"status": "ok", "batch": {"item": 3}}
    )
    assert batch_rows == [{"item": 3}]

    fallback_rows = runtime_module._extract_rows_from_connector_response([{"foo": "bar"}])
    assert fallback_rows == [{"foo": "bar"}]


def test_redact_secrets_masks_sensitive_fields(runtime_module: ModuleType) -> None:
    payload = {
        "api_key": "secret",
        "token": "hidden",
        "nested": {
            "password": "p@ss",
            "value": 42,
            "headers": [{"x-api-key": "also-secret"}],
        },
    }

    redacted = runtime_module._redact_secrets(payload)

    assert redacted["api_key"] == "***"
    assert redacted["token"] == "***"
    assert redacted["nested"]["password"] == "***"
    assert redacted["nested"]["value"] == 42
    assert redacted["nested"]["headers"][0]["x-api-key"] == "***"


@pytest.mark.asyncio
async def test_default_streaming_driver_seed_rows(
    runtime_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NAMEL3SS_ALLOW_STUBS", raising=False)
    connector = {
        "name": "seed_stream",
        "options": {"seed_rows": [{"foo": 1}, 2], "batch_size": 1},
    }
    context: dict = {}

    first = await runtime_module._default_streaming_driver(connector, context)
    second = await runtime_module._default_streaming_driver(connector, context)

    assert first["status"] == "ok"
    assert first["batch"] == [{"foo": 1}]
    assert second["batch"] == [{"value": 2}]
    cursor_state = context["stream_cursors"]["seed_stream"]
    assert cursor_state["index"] == 2
    assert second["metadata"]["exhausted"] is True


@pytest.mark.asyncio
async def test_default_streaming_driver_python_source(
    runtime_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NAMEL3SS_ALLOW_STUBS", raising=False)

    def driver():
        yield {"foo": "bar"}
        yield 5

    monkeypatch.setattr(runtime_module, "_load_python_callable", lambda path: driver)

    connector = {
        "name": "python_stream",
        "options": {
            "batch_size": 2,
            "source": {"type": "python", "driver": "module:callable"},
        },
    }
    context: dict = {}

    first = await runtime_module._default_streaming_driver(connector, context)
    second = await runtime_module._default_streaming_driver(connector, context)

    assert first["status"] == "ok"
    assert first["batch"] == [{"foo": "bar"}, {"value": 5}]
    assert second["batch"] == []
    assert second["metadata"]["exhausted"] is True


@pytest.mark.asyncio
async def test_default_streaming_driver_demo_mode(
    runtime_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NAMEL3SS_ALLOW_STUBS", "1")
    connector = {
        "name": "demo_stream",
        "options": {"batch_size": 3, "api_key": "secret"},
    }

    result = await runtime_module._default_streaming_driver(connector, {})

    assert result["status"] == "demo"
    assert result["batch"]
    assert result["config"]["api_key"] == "***"


@pytest.mark.asyncio
async def test_default_streaming_driver_not_configured(
    runtime_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NAMEL3SS_ALLOW_STUBS", raising=False)
    connector = {
        "name": "no_source_stream",
        "options": {"api_key": "secret"},
    }

    result = await runtime_module._default_streaming_driver(connector, {})

    assert result["status"] == "not_configured"
    assert result["batch"] is None
    assert result["config"]["api_key"] == "***"
    assert not result["metadata"]["exhausted"]
