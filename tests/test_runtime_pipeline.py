import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

from namel3ss.codegen.backend.core import BackendState, _render_runtime_module


STUB_SCHEMA_TYPES = [
    "ActionResponse",
    "ChartResponse",
    "FormResponse",
    "InsightResponse",
    "PredictionResponse",
    "ExperimentResult",
    "TableResponse",
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
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return _Response([])

        async def post(self, *args, **kwargs):
            return _Response({})

    httpx_module.AsyncClient = AsyncClient
    monkeypatch.setitem(sys.modules, "httpx", httpx_module)

    fastapi_module = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code=500, detail=None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    fastapi_module.HTTPException = HTTPException
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_module)

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

    class Select:  # pragma: no cover - stub
        pass

    sqlalchemy_sql.Select = Select
    monkeypatch.setitem(sys.modules, "sqlalchemy.sql", sqlalchemy_sql)

    sqlalchemy_module.ext = sqlalchemy_ext
    sqlalchemy_module.sql = sqlalchemy_sql
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy_module)


def _build_runtime_module(monkeypatch: pytest.MonkeyPatch):
    _install_runtime_stubs(monkeypatch)

    generated_pkg = types.ModuleType("generated")
    generated_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "generated", generated_pkg)

    schemas_module = types.ModuleType("generated.schemas")
    for name in STUB_SCHEMA_TYPES:
        setattr(schemas_module, name, type(name, (), {}))
    monkeypatch.setitem(sys.modules, "generated.schemas", schemas_module)
    generated_pkg.schemas = schemas_module

    dataset_definition = {
        "name": "sales",
        "source_type": "inline",
        "source": "inline",
        "connector": None,
        "operations": [
            {
                "type": "computed_column",
                "name": "gross",
                "expression": "row.get('revenue', 0) + row.get('tax', 0)",
            },
            {"type": "filter", "condition": "row.get('region') == 'EU'"},
            {"type": "group_by", "columns": ["month"]},
            {"type": "aggregate", "function": "sum", "expression": "gross as total_gross"},
        ],
        "transforms": [
            {"type": "select", "options": {"columns": ["month", "total_gross"]}},
        ],
        "quality_checks": [
            {"name": "positive", "condition": "row.get('total_gross', 0) > 0", "severity": "warning"},
            {"name": "minimum_rows", "metric": "row_count", "threshold": 1},
        ],
        "features": ["gross"],
        "targets": ["revenue"],
        "sample_rows": [],
    }

    state = BackendState(
        app={"name": "Test", "database": None, "theme": {}, "variables": []},
        datasets={"sales": dataset_definition},
        connectors={},
        ai_connectors={},
        insights={},
        models={},
        templates={},
        chains={},
        experiments={},
        pages=[],
        env_keys=[],
    )

    source = _render_runtime_module(state, embed_insights=False, enable_realtime=False)
    module_name = "generated.runtime_test"
    runtime_module = types.ModuleType(module_name)
    runtime_module.__package__ = "generated"
    monkeypatch.setitem(sys.modules, module_name, runtime_module)
    exec(source, runtime_module.__dict__)
    return runtime_module


@pytest.fixture()
def runtime_module(monkeypatch: pytest.MonkeyPatch):
    return _build_runtime_module(monkeypatch)


class StubSession:
    def __init__(self, rowcount: int = 1) -> None:
        self.rowcount = rowcount
        self.statements = []
        self.parameters = []
        self.committed = False
        self.rolled_back = False

    async def execute(self, statement, params=None):
        self.statements.append(statement)
        self.parameters.append(params or {})
        return SimpleNamespace(rowcount=self.rowcount)

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True


def test_execute_dataset_pipeline_with_quality_checks(runtime_module) -> None:
    dataset = runtime_module.DATASETS["sales"]
    rows = [
        {"month": "Jan", "region": "EU", "revenue": 120.0, "tax": 10.0},
        {"month": "Jan", "region": "NA", "revenue": 80.0, "tax": 8.0},
        {"month": "Feb", "region": "EU", "revenue": 150.0, "tax": 15.0},
    ]
    context: dict = {}

    processed = asyncio.run(runtime_module._execute_dataset_pipeline(dataset, rows, context))

    assert processed == [
        {"month": "Jan", "total_gross": 130.0},
        {"month": "Feb", "total_gross": 165.0},
    ]
    quality = context.get("quality", {}).get("sales")
    assert quality and all(item["passed"] for item in quality)
    assert context.get("features", {}).get("sales") == ["gross"]
    assert context.get("targets", {}).get("sales") == ["revenue"]


def test_execute_update_builds_parameterised_sql(runtime_module) -> None:
    session = StubSession(rowcount=3)
    context = {
        "vars": {"updated_status": "APPROVED"},
        "payload": {"target_id": 42},
    }

    rows_updated = asyncio.run(runtime_module._execute_update(
        "orders",
        'status = context.get("vars", {}).get("updated_status")',
        'id = context.get("payload", {}).get("target_id")',
        session,
        context,
    ))

    assert rows_updated == 3
    assert session.committed and not session.rolled_back
    assert session.statements, "No SQL statement executed"
    statement = session.statements[-1]
    params = session.parameters[-1]
    assert getattr(statement, "text", str(statement)) == "UPDATE orders SET status = :set_0 WHERE id = :where_0"
    assert params == {"set_0": "APPROVED", "where_0": 42}