"""Shared helpers for dynamic backend module tests."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Dict, Iterator


def install_backend_stubs(monkeypatch) -> None:
    """Install lightweight stubs for FastAPI, SQLAlchemy, and Pydantic used in backend tests."""

    fake_fastapi = ModuleType("fastapi")

    class FakeFastAPI:  # pragma: no cover - minimal FastAPI stand-in
        def __init__(self, *args, **kwargs) -> None:
            _ = (args, kwargs)
            self._middleware = []

        def get(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def post(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def websocket(self, *args, **kwargs):  # pragma: no cover - stub
            def decorator(func):
                return func

            return decorator

        def include_router(self, *args, **kwargs):  # pragma: no cover - hook
            return None

        def middleware(self, _event: str):  # pragma: no cover - decorator shim
            def decorator(func):
                return func

            return decorator

        def add_middleware(self, *args, **kwargs):  # pragma: no cover - storage shim
            self._middleware.append((args, kwargs))
            return None

        def exception_handler(self, _exc_type):  # pragma: no cover - decorator shim
            def decorator(func):
                return func

            return decorator

    class FakeAPIRouter:  # pragma: no cover - minimal APIRouter stand-in
        def __init__(self, *args, **kwargs) -> None:
            _ = (args, kwargs)

        def get(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def post(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def websocket(self, *args, **kwargs):  # pragma: no cover - stub
            def decorator(func):
                return func

            return decorator

    class FakeRequest:  # pragma: no cover - minimal Request stand-in
        def __init__(self, headers: dict | None = None, cookies: dict | None = None, method: str = "GET") -> None:
            self.headers = headers or {}
            self.cookies = cookies or {}
            self.state = SimpleNamespace()
            self.method = method
            self.client = SimpleNamespace(host="testclient")

    class FakeResponse:  # pragma: no cover - minimal Response stand-in
        def __init__(self, content=None, status_code: int = 200) -> None:
            self.content = content
            self.status_code = status_code
            self.headers: Dict[str, str] = {}

    class FakeWebSocket:  # pragma: no cover - WebSocket stand-in
        async def accept(self) -> None:
            return None

        async def close(self, code: int = 1000) -> None:
            _ = code
            return None

        async def send_json(self, message):
            _ = message
            return None

        async def receive_json(self):
            raise NotImplementedError

    fake_fastapi.Request = FakeRequest
    fake_fastapi.Response = FakeResponse
    fake_fastapi.WebSocket = FakeWebSocket
    fake_fastapi.WebSocketDisconnect = type("WebSocketDisconnect", (Exception,), {})
    fake_fastapi.FastAPI = FakeFastAPI
    fake_fastapi.APIRouter = FakeAPIRouter
    fake_fastapi.HTTPException = type("HTTPException", (Exception,), {})
    fake_fastapi.Depends = lambda dependency=None: dependency
    fake_fastapi.Query = lambda default=None, *args, **kwargs: default

    fake_fastapi_responses = ModuleType("fastapi.responses")

    class _StreamingResponse:  # pragma: no cover - minimal response stub
        def __init__(self, content=None, media_type: str | None = None) -> None:
            self.content = content
            self.media_type = media_type

    class _JSONResponse:  # pragma: no cover - minimal JSON response stand-in
        def __init__(self, content=None, status_code: int = 200) -> None:
            self.content = content or {}
            self.status_code = status_code
            self.headers = {}

        def set_cookie(self, *args, **kwargs) -> None:
            _ = (args, kwargs)
            return None

    fake_fastapi_responses.StreamingResponse = _StreamingResponse
    fake_fastapi_responses.JSONResponse = _JSONResponse
    monkeypatch.setitem(sys.modules, "fastapi.responses", fake_fastapi_responses)
    fake_fastapi.responses = fake_fastapi_responses

    fake_fastapi_websockets = ModuleType("fastapi.websockets")
    fake_fastapi_websockets.WebSocket = FakeWebSocket
    fake_fastapi_websockets.WebSocketDisconnect = fake_fastapi.WebSocketDisconnect
    monkeypatch.setitem(sys.modules, "fastapi.websockets", fake_fastapi_websockets)
    fake_fastapi.websockets = fake_fastapi_websockets

    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)

    fake_sqlalchemy = ModuleType("sqlalchemy")
    fake_sqlalchemy.__path__ = []

    class MetaData:  # pragma: no cover - stub metadata
        def __init__(self, *args, **kwargs) -> None:
            self.tables = {}

    def text(value):  # pragma: no cover - stubbed text helper
        return SimpleNamespace(text=value)

    def bindparam(name, *args, **kwargs):  # pragma: no cover - stubbed bindparam
        return SimpleNamespace(key=name)

    def update(table_name, *args, **kwargs):  # pragma: no cover - stubbed update expression
        return SimpleNamespace(table=table_name)

    fake_sqlalchemy.MetaData = MetaData
    fake_sqlalchemy.text = text
    fake_sqlalchemy.bindparam = bindparam
    fake_sqlalchemy.update = update
    monkeypatch.setitem(sys.modules, "sqlalchemy", fake_sqlalchemy)

    fake_sqlalchemy_sql = ModuleType("sqlalchemy.sql")

    class Select:  # pragma: no cover - stub select
        pass

    def table(name: str, *args, **kwargs):  # pragma: no cover - stub table factory
        return SimpleNamespace(name=name, columns=args)

    def column(name: str, *args, **kwargs):  # pragma: no cover - stub column factory
        return SimpleNamespace(name=name)

    fake_sqlalchemy_sql.Select = Select
    fake_sqlalchemy_sql.table = table
    fake_sqlalchemy_sql.column = column
    monkeypatch.setitem(sys.modules, "sqlalchemy.sql", fake_sqlalchemy_sql)
    fake_sqlalchemy.sql = fake_sqlalchemy_sql

    fake_sqlalchemy_ext = ModuleType("sqlalchemy.ext")
    fake_sqlalchemy_ext.__path__ = []
    monkeypatch.setitem(sys.modules, "sqlalchemy.ext", fake_sqlalchemy_ext)
    fake_sqlalchemy.ext = fake_sqlalchemy_ext

    fake_sqlalchemy_ext_asyncio = ModuleType("sqlalchemy.ext.asyncio")

    class AsyncSession:  # pragma: no cover - async session stub
        async def execute(self, *args, **kwargs):
            raise NotImplementedError

    class AsyncEngine:  # pragma: no cover - placeholder
        pass

    def async_sessionmaker(*args, **kwargs):  # pragma: no cover - stub factory
        def _factory(*_a, **_k):
            raise NotImplementedError

        return _factory

    def create_async_engine(*args, **kwargs):  # pragma: no cover - stub engine
        return SimpleNamespace()

    fake_sqlalchemy_ext_asyncio.AsyncSession = AsyncSession
    fake_sqlalchemy_ext_asyncio.AsyncEngine = AsyncEngine
    fake_sqlalchemy_ext_asyncio.async_sessionmaker = async_sessionmaker
    fake_sqlalchemy_ext_asyncio.create_async_engine = create_async_engine
    monkeypatch.setitem(sys.modules, "sqlalchemy.ext.asyncio", fake_sqlalchemy_ext_asyncio)
    fake_sqlalchemy_ext.asyncio = fake_sqlalchemy_ext_asyncio
    fake_sqlalchemy.ext.asyncio = fake_sqlalchemy_ext_asyncio

    fake_sqlalchemy_sql = ModuleType("sqlalchemy.sql")

    class Select:  # pragma: no cover - placeholder
        pass

    fake_sqlalchemy_sql.Select = Select
    monkeypatch.setitem(sys.modules, "sqlalchemy.sql", fake_sqlalchemy_sql)
    fake_sqlalchemy.sql = fake_sqlalchemy_sql

    fake_pydantic = ModuleType("pydantic")

    class BaseModel:  # pragma: no cover - simple stand-in
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

        def dict(self, *args, **kwargs):  # type: ignore[override]
            return dict(self.__dict__)

        def model_dump(self, *args, **kwargs):  # pragma: no cover - pydantic v2 shim
            return dict(self.__dict__)

    def Field(*args, **kwargs):  # pragma: no cover - placeholder
        return kwargs.get("default", None)

    fake_pydantic.BaseModel = BaseModel
    fake_pydantic.Field = Field
    monkeypatch.setitem(sys.modules, "pydantic", fake_pydantic)


def _clear_backend_modules(module_name: str, package_name: str) -> None:
    sys.modules.pop(module_name, None)
    sys.modules.pop(package_name, None)


@contextmanager
def load_backend_module(tmp_path, backend_dir, monkeypatch) -> Iterator[ModuleType]:
    """Load the generated backend module under test with required stubs."""

    install_backend_stubs(monkeypatch)
    sys.path.insert(0, str(tmp_path))
    module_name = f"{backend_dir.name}.main"
    try:
        module = importlib.import_module(module_name)
        yield module
    finally:
        sys.path.pop(0)
        _clear_backend_modules(module_name, backend_dir.name)


__all__ = ["install_backend_stubs", "load_backend_module"]
