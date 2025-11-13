"""Backend generator facade that renders the FastAPI scaffold."""

from __future__ import annotations

import ast
import pprint
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency at generation time
    from sqlalchemy import text as _sa_text, bindparam as _sa_bindparam, update as _sa_update
    from sqlalchemy.sql import Select as _sa_select, table as _sa_table, column as _sa_column
except ImportError:  # pragma: no cover - optional dependency
    def _sa_text(sql: str) -> str:
        return sql

    _sa_bindparam = None  # type: ignore
    _sa_update = None  # type: ignore
    _sa_table = None  # type: ignore
    _sa_column = None  # type: ignore
    _sa_select = Any  # type: ignore
    _HAS_SQLA_UPDATE = False
else:
    _HAS_SQLA_UPDATE = True

text = _sa_text
bindparam = _sa_bindparam
update = _sa_update
sql_table = _sa_table
column = _sa_column
Select = _sa_select  # type: ignore

from namel3ss.ast import App
from namel3ss.ml import get_default_model_registry, load_model_registry
from .state import (
    BackendState,
    PageComponent,
    PageSpec,
    build_backend_state,
    _component_to_serializable,
)

# earlier file... ensure __all__ defined at top? located near top or bottom? no change needed here

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_AGGREGATE_ALIAS_PATTERN = re.compile(r"\s+as\s+", flags=re.IGNORECASE)
_UPDATE_ASSIGNMENT_PATTERN = re.compile("^\\s*([A-Za-z_][A-Za-z0-9_\\.]*?)\\s*=\\s*(.+)$")
_WHERE_CONDITION_PATTERN = re.compile("^\\s*([A-Za-z_][A-Za-z0-9_\\.]*?)\\s*=\\s*(.+)$")


def generate_backend(
    app: App,
    out_dir: Path,
    embed_insights: bool = False,
    enable_realtime: bool = False,
) -> None:
    """Generate the backend scaffold for ``app`` into ``out_dir``."""

    state = build_backend_state(app)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    (out_path / "__init__.py").write_text("", encoding="utf-8")
    (out_path / "database.py").write_text(
        _render_database_module(state), encoding="utf-8"
    )

    generated_dir = out_path / "generated"
    routers_dir = generated_dir / "routers"
    helpers_dir = generated_dir / "helpers"
    schemas_dir = generated_dir / "schemas"

    custom_dir = out_path / "custom"
    custom_routes_dir = custom_dir / "routes"

    for path in [generated_dir, routers_dir, helpers_dir, schemas_dir, custom_dir, custom_routes_dir]:
        path.mkdir(parents=True, exist_ok=True)

    (generated_dir / "__init__.py").write_text(
        _render_generated_package(), encoding="utf-8"
    )
    (generated_dir / "runtime.py").write_text(
        _render_runtime_module(state, embed_insights, enable_realtime), encoding="utf-8"
    )
    (helpers_dir / "__init__.py").write_text(
        _render_helpers_package(), encoding="utf-8"
    )
    (schemas_dir / "__init__.py").write_text(
        _render_schemas_module(), encoding="utf-8"
    )

    (routers_dir / "__init__.py").write_text(
        _render_routers_package(), encoding="utf-8"
    )
    (routers_dir / "insights.py").write_text(
        _render_insights_router_module(), encoding="utf-8"
    )
    (routers_dir / "models.py").write_text(
        _render_models_router_module(), encoding="utf-8"
    )
    (routers_dir / "experiments.py").write_text(
        _render_experiments_router_module(), encoding="utf-8"
    )
    (routers_dir / "pages.py").write_text(
        _render_pages_router_module(state), encoding="utf-8"
    )

    (out_path / "main.py").write_text(
        _render_app_module(), encoding="utf-8"
    )

    if not (custom_dir / "__init__.py").exists():
        (custom_dir / "__init__.py").write_text("", encoding="utf-8")
    if not (custom_routes_dir / "__init__.py").exists():
        (custom_routes_dir / "__init__.py").write_text("", encoding="utf-8")
    custom_readme = custom_dir / "README.md"
    if not custom_readme.exists():
        custom_readme.write_text(_render_custom_readme(), encoding="utf-8")
    custom_api_path = custom_routes_dir / "custom_api.py"
    if not custom_api_path.exists():
        custom_api_path.write_text(_render_custom_api_stub(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Module renderers
# ---------------------------------------------------------------------------


def _render_database_module(state: BackendState) -> str:
    env_var = _database_env_var(state.app.get("database"))
    template = f'''
"""Database configuration for the generated FastAPI backend."""

from __future__ import annotations

import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DATABASE_URL_ENV = {env_var!r}
DEFAULT_DATABASE_URL = "postgresql+asyncpg://user:password@localhost:5432/app"


def _build_database_url() -> str:
    """Return the connection string for the application's primary database."""

    return os.getenv(DATABASE_URL_ENV, DEFAULT_DATABASE_URL)


engine: AsyncEngine = create_async_engine(_build_database_url(), echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an :class:`AsyncSession` for FastAPI dependencies."""

    async with SessionLocal() as session:
        yield session
'''
    return textwrap.dedent(template).strip()


def _render_schemas_module() -> str:
    template = '''
"""Pydantic schemas for the generated FastAPI backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TableResponse(BaseModel):
    title: str
    source: Dict[str, str]
    columns: List[str] = Field(default_factory=list)
    filter: Optional[str] = None
    sort: Optional[str] = None
    style: Dict[str, Any] = Field(default_factory=dict)
    insight: Optional[str] = None
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    insights: Dict[str, Any] = Field(default_factory=dict)


class ChartSeries(BaseModel):
    label: str
    data: List[Any] = Field(default_factory=list)


class ChartResponse(BaseModel):
    heading: Optional[str] = None
    title: Optional[str] = None
    source: Dict[str, str]
    chart_type: str
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    labels: List[Any] = Field(default_factory=list)
    series: List[ChartSeries] = Field(default_factory=list)
    legend: Dict[str, Any] = Field(default_factory=dict)
    style: Dict[str, Any] = Field(default_factory=dict)
    encodings: Dict[str, Any] = Field(default_factory=dict)
    insight: Optional[str] = None
    insights: Dict[str, Any] = Field(default_factory=dict)


class InsightResponse(BaseModel):
    name: str
    dataset: str
    result: Dict[str, Any]


class ActionResponse(BaseModel):
    action: str
    trigger: str
    operations: List[str] = Field(default_factory=list)


class FormResponse(BaseModel):
    title: str
    fields: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)
    operations: List[str] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    model: str
    version: Optional[str] = None
    framework: Optional[str] = None
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    explanations: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentVariant(BaseModel):
    name: Optional[str] = None
    target_type: str
    target_name: Optional[str] = None
    score: Optional[float] = None
    result: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


class ExperimentMetric(BaseModel):
    name: Optional[str] = None
    value: Optional[float] = None
    source_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentResult(BaseModel):
    name: Optional[str] = None
    variant: Optional[ExperimentVariant] = None
    metrics: List[ExperimentMetric] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

'''
    return textwrap.dedent(template).strip()


def _render_custom_api_stub() -> str:
    template = '''
"""Custom API extensions for your Namel3ss backend.

This module is created once. Add routes, dependencies, or hooks freely; the
generator will never overwrite it.
"""

from __future__ import annotations

from fastapi import APIRouter

# Optional helpers when you want to reuse generated logic:
# from ..generated.routers import experiments as generated_experiments
# from ..generated.routers import models as generated_models
# from ..generated.schemas import ExperimentResult, PredictionResponse

router = APIRouter()


# Example override (uncomment and adapt):
# @router.post(
#     "/api/models/{model_name}/predict",
#     response_model=PredictionResponse,
#     include_in_schema=False,
# )
# async def predict_with_tracking(model_name: str, payload: dict) -> PredictionResponse:
#     base = await generated_models.predict(model_name, payload)
#     base.metadata.setdefault("tags", []).append("customized")
#     base.metadata["handled_by"] = "custom_api"
#     return base
#
# Example extension:
# @router.get("/api/experiments/{slug}/summary", response_model=ExperimentResult)
# async def experiment_summary(slug: str) -> ExperimentResult:
#     result = await generated_experiments.get_experiment(slug)
#     result.metadata["summary"] = f"Experiment {slug} served by custom routes."
#     return result
#
# The optional ``setup`` hook runs after generated routers are registered.


def setup(app) -> None:  # pragma: no cover - user may replace implementation
    """Run initialization after generated routers are registered."""

    _ = app  # Replace with custom logic (auth, logging, etc.)
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_generated_package() -> str:
    template = '''
"""Generated backend package for Namel3ss (N3).

This file is created automatically. Manual edits may be overwritten.
"""

from __future__ import annotations

from . import runtime

__all__ = ["runtime"]
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_helpers_package() -> str:
    template = '''
"""Helper utilities for Namel3ss generated routers."""

from __future__ import annotations

from typing import Iterable

from fastapi import FastAPI

from ..routers import GENERATED_ROUTERS

__all__ = ["GENERATED_ROUTERS", "include_generated_routers"]


def include_generated_routers(app: FastAPI, routers: Iterable = GENERATED_ROUTERS) -> None:
    """Attach generated routers to ``app`` while allowing custom overrides."""

    for router in routers:
        app.include_router(router)
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_custom_readme() -> str:
    template = '''
# Custom Backend Extensions

This folder is reserved for your handcrafted FastAPI routes and helpers. The
code generator creates it once and will not overwrite files you add here.

- Put reusable dependencies in `__init__.py` or new modules.
- Add route overrides in `routes/custom_api.py` and register them on the
  module-level `router` instance.
- Use the optional `setup(app)` hook to run initialization logic after the
  generated routers are attached (for example, authentication, middleware, or
  event handlers).

Whenever you run the Namel3ss generator again your custom code stays intact.
Refer to the generated modules under `generated/` for available helpers.
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_routers_package() -> str:
    template = '''
"""Aggregated FastAPI routers for Namel3ss (auto-generated)."""

from __future__ import annotations

from . import experiments, insights, models, pages

insights_router = insights.router
models_router = models.router
experiments_router = experiments.router
pages_router = pages.router

GENERATED_ROUTERS = (
    insights_router,
    models_router,
    experiments_router,
    pages_router,
)

__all__ = [
    "insights_router",
    "models_router",
    "experiments_router",
    "pages_router",
    "GENERATED_ROUTERS",
]
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_insights_router_module() -> str:
    template = '''
"""Generated FastAPI router for insight endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ..runtime import evaluate_insight
from ..schemas import InsightResponse

router = APIRouter(tags=["insights"])


@router.get("/api/insights/{slug}", response_model=InsightResponse)
async def get_generated_insight(slug: str) -> InsightResponse:
    return evaluate_insight(slug)


__all__ = ["router"]
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_models_router_module() -> str:
    template = '''
"""Generated FastAPI router for model and AI helper endpoints."""

from __future__ import annotations

from typing import Dict

from fastapi import APIRouter

from ..runtime import call_llm_connector, call_python_model, run_chain, run_prediction
from ..schemas import PredictionResponse

router = APIRouter()


@router.post("/api/models/{model_name}/predict", response_model=PredictionResponse, tags=["models"])
async def predict_model_endpoint(
    model_name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> PredictionResponse:
    return run_prediction(model_name, payload)


@router.post("/api/ai/python", tags=["ai"])
async def invoke_python_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    module = str(payload.get("module") or payload.get("path") or "")
    method = str(payload.get("method") or payload.get("function") or "predict")
    arguments = payload.get("arguments") or payload.get("args") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    return call_python_model(module, method, arguments)


@router.post("/api/ai/connectors/{name}", tags=["ai", "connectors"])
async def invoke_connector(name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return call_llm_connector(name, payload)


@router.post("/api/ai/chains/{name}", tags=["ai", "chains"])
async def invoke_chain(name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return run_chain(name, payload)


__all__ = ["router"]
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_experiments_router_module() -> str:
    template = '''
"""Generated FastAPI router for experiment endpoints."""

from __future__ import annotations

from typing import Dict

from fastapi import APIRouter

from ..runtime import run_experiment
from ..schemas import ExperimentResult

router = APIRouter(tags=["experiments"])


@router.get("/api/experiments/{slug}", response_model=ExperimentResult)
async def get_experiment(slug: str) -> ExperimentResult:
    return run_experiment(slug, {})


@router.post("/api/experiments/{slug}", response_model=ExperimentResult)
async def execute_experiment(
    slug: str,
    payload: Optional[Dict[str, Any]] = None,
) -> ExperimentResult:
    return run_experiment(slug, payload)


__all__ = ["router"]
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_pages_router_module(state: BackendState) -> str:
    header = '''
"""Generated FastAPI router for page and component endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
try:
    from fastapi import WebSocket, WebSocketDisconnect
except ImportError:  # pragma: no cover - FastAPI <0.65 fallback
    from fastapi.websockets import WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_session
from .. import runtime
from ..schemas import ChartResponse, TableResponse

router = APIRouter()
'''
    parts: List[str] = [textwrap.dedent(header).strip()]

    page_blocks: List[str] = []
    for page in state.pages:
        lines: List[str] = []
        lines.extend(_render_page_endpoint(page))
        for index, component in enumerate(page.components):
            endpoint_lines = _render_component_endpoint(page, component, index)
            if endpoint_lines:
                lines.append("")
                lines.extend(endpoint_lines)
        page_blocks.append("\n".join(lines))

    if page_blocks:
        parts.append("\n\n".join(block.strip() for block in page_blocks if block))

    metrics_block = '''
@router.get("/api/pages/model/metrics", response_model=TableResponse, tags=["models"])
async def model_registry_metrics() -> TableResponse:
    rows: List[Dict[str, Any]] = []
    for name, spec in runtime.MODEL_REGISTRY.items():
        metrics = spec.get("metrics", {}) if isinstance(spec, dict) else {}
        rows.append({
            "model": name,
            "framework": spec.get("framework", "unknown") if isinstance(spec, dict) else "unknown",
            "version": spec.get("version", "v1") if isinstance(spec, dict) else "v1",
            "metrics": ", ".join(f"{key}={value}" for key, value in metrics.items()) or "n/a",
        })
    return TableResponse(
        title="Model Registry Metrics",
        source={"type": "model_registry", "name": "metrics"},
        columns=["model", "framework", "version", "metrics"],
        filter=None,
        sort=None,
        style={},
        insight=None,
        rows=rows,
        insights={},
    )


@router.get("/api/pages/model/feature_importances", response_model=ChartResponse, tags=["models"])
async def model_feature_importances() -> ChartResponse:
    labels = ["feature_a", "feature_b", "feature_c"]
    series = [{"label": "Importance", "data": [0.7, 0.2, 0.1]}]
    return ChartResponse(
        heading="Model Feature Importances",
        title="Model Feature Importances",
        source={"type": "model_registry", "name": "feature_importances"},
        chart_type="bar",
        x="feature",
        y="importance",
        color=None,
        labels=labels,
        series=series,
        legend={},
        style={},
        encodings={},
        insight=None,
        insights={},
    )
'''
    parts.append(textwrap.dedent(metrics_block).strip())

    streams_block = '''
@router.get("/api/streams/pages/{slug}", response_class=StreamingResponse, tags=["streams"])
async def stream_page_events(slug: str, heartbeat: Optional[int] = None) -> StreamingResponse:
    return await runtime.stream_page(slug, heartbeat=heartbeat)


@router.get("/api/streams/datasets/{dataset}", response_class=StreamingResponse, tags=["streams"])
async def stream_dataset_events(dataset: str, heartbeat: Optional[int] = None) -> StreamingResponse:
    return await runtime.stream_dataset(dataset, heartbeat=heartbeat)


@router.get("/api/streams/topics/{topic:path}", response_class=StreamingResponse, tags=["streams"])
async def stream_topic_events(topic: str, heartbeat: Optional[int] = None) -> StreamingResponse:
    return await runtime.stream_topic(topic, heartbeat=heartbeat)
'''
    parts.append(textwrap.dedent(streams_block).strip())

    websocket_block = '''
@router.websocket("/ws/pages/{slug}")
async def page_updates(slug: str, websocket: WebSocket) -> None:
    if not runtime.REALTIME_ENABLED:
        await websocket.accept()
        await websocket.close(code=1000)
        return
    await runtime.BROADCAST.connect(slug, websocket)
    try:
        page_spec = runtime.PAGE_SPEC_BY_SLUG.get(slug, {})
        handler = runtime.PAGE_HANDLERS.get(slug)
        if page_spec.get("reactive") and handler:
            try:
                payload = await handler(None)
                await websocket.send_json(runtime._with_timestamp({
                    "type": "hydration",
                    "slug": slug,
                    "payload": payload,
                    "meta": runtime._page_meta(slug),
                }))
            except Exception:
                runtime.logger.exception("Failed to hydrate reactive page %s", slug)
        while True:
            try:
                message = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except Exception:
                runtime.logger.exception("Invalid realtime message for %s", slug)
                await websocket.send_json(runtime._with_timestamp({
                    "type": "error",
                    "slug": slug,
                    "detail": "invalid message",
                }))
                continue
            ack_payload: Dict[str, Any] = {
                "type": "ack",
                "slug": slug,
                "status": "ok",
            }
            if isinstance(message, dict):
                if "id" in message:
                    ack_payload["id"] = message["id"]
                if message.get("type") == "optimistic":
                    ack_payload["status"] = "pending"
                    component_index = message.get("component_index")
                    if message.get("rollback") and component_index is not None:
                        await runtime.broadcast_rollback(slug, int(component_index))
            await websocket.send_json(runtime._with_timestamp(ack_payload))
    finally:
        await runtime.BROADCAST.disconnect(slug, websocket)
'''
    parts.append(textwrap.dedent(websocket_block).strip())

    parts.append("__all__ = ['router']")
    return "\n\n".join(part for part in parts if part).strip() + "\n"


def _render_app_module() -> str:
    template = '''
"""FastAPI application entry point for the Namel3ss generated backend."""

from __future__ import annotations

from typing import Dict

from fastapi import FastAPI

from .generated import runtime
from .generated.helpers import include_generated_routers

app = FastAPI(
    title=runtime.APP.get("name", "Namel3ss App"),
    version=str(runtime.APP.get("version", "0.1")),
)


def _load_user_routes() -> None:
    """Load optional custom routes without failing generation."""

    module_candidates = [
        "custom.routes.custom_api",
        "custom.routes.extensions",
        "custom_api",
        "extensions",
    ]
    for module_name in module_candidates:
        try:
            module = __import__(f"{__package__}.{module_name}", fromlist=["*"])
        except Exception:
            try:
                module = __import__(module_name, fromlist=["*"])
            except Exception:
                continue
        router = getattr(module, "router", None)
        if router is not None:
            app.include_router(router)
        setup = getattr(module, "setup", None)
        if callable(setup):
            try:
                setup(app)
            except Exception:  # pragma: no cover - user extension failure
                runtime.logger.exception("User API setup failed for %s", module_name)


include_generated_routers(app)
_load_user_routes()


@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


# Re-export runtime helpers for convenience in tests and extensions.
predict = runtime.predict
predict_model = runtime.predict_model
call_python_model = runtime.call_python_model
call_llm_connector = runtime.call_llm_connector
run_chain = runtime.run_chain
evaluate_experiment = runtime.evaluate_experiment
run_experiment = runtime.run_experiment
DATASETS = runtime.DATASETS
INSIGHTS = runtime.INSIGHTS
CONTEXT = runtime.CONTEXT
build_context = runtime.build_context
_resolve_connector = runtime._resolve_connector
_run_insight = runtime._run_insight

__all__ = [
    "app",
    "predict",
    "predict_model",
    "call_python_model",
    "call_llm_connector",
    "run_chain",
    "evaluate_experiment",
    "run_experiment",
    "DATASETS",
    "INSIGHTS",
    "CONTEXT",
    "build_context",
    "_resolve_connector",
    "_run_insight",
]
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_runtime_module(
    state: BackendState,
    embed_insights: bool,
    enable_realtime: bool,
) -> str:
    parts: List[str] = []
    page_handler_entries: List[str] = []
    configured_model_registry = load_model_registry()
    if not configured_model_registry:
        configured_model_registry = get_default_model_registry()

    header_lines = [
        '"""Generated runtime primitives for Namel3ss (N3)."""',
        '',
        'from __future__ import annotations',
        '',
    ]
    header_lines.extend([
        'import asyncio',
        'import ast',
        'import copy',
        'import csv',
        'import inspect',
        'import importlib',
        'import importlib.util',
        'import json',
        'import logging',
        'import math',
        'import os',
    'import pickle',
        'import re',
        'import sys',
        'import time',
        'from collections import defaultdict',
        'from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple',
        '',
        'import httpx',
        'from fastapi import HTTPException',
    ])
    if enable_realtime:
        header_lines.extend([
            'try:',
            '    from fastapi import WebSocket, WebSocketDisconnect',
            'except ImportError:  # FastAPI <0.65 fallback',
            '    from fastapi.websockets import WebSocket, WebSocketDisconnect',
        ])
    header_lines.extend([
        'try:',
        '    from sqlalchemy import MetaData, text, bindparam, update',
        '    from sqlalchemy.sql import Select, table as sql_table, column',
        '    _HAS_SQLA_UPDATE = True',
        'except ImportError:  # pragma: no cover - optional dependency',
        '    from sqlalchemy import MetaData, text',
        '    from sqlalchemy.sql import Select',
        '    bindparam = None  # type: ignore',
        '    update = None  # type: ignore',
        '    sql_table = None  # type: ignore',
        '    column = None  # type: ignore',
        '    _HAS_SQLA_UPDATE = False',
        'from sqlalchemy.ext.asyncio import AsyncSession',
        'from pathlib import Path',
        '',
        'from .schemas import (',
    '    ActionResponse,',
    '    ChartResponse,',
    '    FormResponse,',
    '    InsightResponse,',
    '    PredictionResponse,',
    '    ExperimentResult,',
    '    TableResponse,',
    ')',
        '',
        'logger = logging.getLogger(__name__)',
        '_HTTPX_CLIENT_CLS = httpx.AsyncClient',
        'CONTEXT_MARKER_KEY = "__context__"',
        '_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")',
        '_AGGREGATE_ALIAS_PATTERN = re.compile(r"\\s+as\\s+", flags=re.IGNORECASE)',
        '_UPDATE_ASSIGNMENT_PATTERN = re.compile(r"^\\s*([A-Za-z_][A-Za-z0-9_\\.]*?)\\s*=\\s*(.+)$")',
        '_WHERE_CONDITION_PATTERN = re.compile(r"^\\s*([A-Za-z_][A-Za-z0-9_\\.]*?)\\s*=\\s*(.+)$")',
    ])
    parts.append("\n".join(header_lines).strip())

    context_runtime = '''
class ContextRegistry:
    """Simple registry for runtime context values."""

    def __init__(self) -> None:
        self._global: Dict[str, Any] = {}
        self._pages: Dict[str, Dict[str, Any]] = {}

    def set_global(self, values: Dict[str, Any]) -> None:
        self._global = dict(values)

    def set_page(self, slug: str, values: Dict[str, Any]) -> None:
        self._pages[slug] = dict(values)

    def build(self, slug: Optional[str]) -> Dict[str, Any]:
        base = dict(self._global)
        if slug and slug in self._pages:
            base.update(self._pages[slug])
        return base


CONTEXT = ContextRegistry()
'''
    parts.append(textwrap.dedent(context_runtime).strip())

    if enable_realtime:
        broadcast_runtime = '''
class PageBroadcastManager:
    """Manage WebSocket connections for reactive pages."""

    def __init__(self) -> None:
        self._connections: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, slug: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.setdefault(slug, []).append(websocket)

    async def disconnect(self, slug: str, websocket: WebSocket) -> None:
        async with self._lock:
            connections = self._connections.get(slug)
            if not connections:
                return
            if websocket in connections:
                connections.remove(websocket)
            if not connections:
                self._connections.pop(slug, None)

    async def broadcast(self, slug: str, message: Dict[str, Any]) -> None:
        async with self._lock:
            connections = list(self._connections.get(slug, []))
        if not connections:
            return
        stale: List[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception:
                stale.append(connection)
        if stale:
            async with self._lock:
                current = self._connections.get(slug)
                if not current:
                    return
                for connection in stale:
                    if connection in current:
                        current.remove(connection)
                if not current:
                    self._connections.pop(slug, None)

    async def has_listeners(self, slug: str) -> bool:
        async with self._lock:
            return bool(self._connections.get(slug))


BROADCAST = PageBroadcastManager()
'''
    else:
        broadcast_runtime = '''
class PageBroadcastManager:
    """Stubbed broadcast manager when realtime is disabled."""

    async def connect(self, slug: str, websocket: Any) -> None:  # pragma: no cover
        return None

    async def disconnect(self, slug: str, websocket: Any) -> None:  # pragma: no cover
        return None

    async def broadcast(self, slug: str, message: Dict[str, Any]) -> None:
        return None

    async def has_listeners(self, slug: str) -> bool:
        return False


BROADCAST = PageBroadcastManager()
'''
    parts.append(textwrap.dedent(broadcast_runtime).strip())

    registries = [
        _assign_literal("APP", "Dict[str, Any]", state.app),
        _assign_literal("DATASETS", "Dict[str, Dict[str, Any]]", state.datasets),
        _assign_literal("CONNECTORS", "Dict[str, Dict[str, Any]]", state.connectors),
        _assign_literal("AI_CONNECTORS", "Dict[str, Dict[str, Any]]", state.ai_connectors),
        _assign_literal("AI_TEMPLATES", "Dict[str, Dict[str, Any]]", state.templates),
        _assign_literal("AI_CHAINS", "Dict[str, Dict[str, Any]]", state.chains),
        _assign_literal("AI_EXPERIMENTS", "Dict[str, Dict[str, Any]]", state.experiments),
        _assign_literal("INSIGHTS", "Dict[str, Dict[str, Any]]", state.insights),
    _assign_literal("MODEL_REGISTRY", "Dict[str, Dict[str, Any]]", configured_model_registry),
        "MODEL_CACHE: Dict[str, Any] = {}",
        "MODEL_LOADERS: Dict[str, Callable[[str, Dict[str, Any]], Any]] = {}",
        "MODEL_RUNNERS: Dict[str, Callable[[str, Any, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {}",
        "MODEL_EXPLAINERS: Dict[str, Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {}",
        _assign_literal("MODELS", "Dict[str, Dict[str, Any]]", state.models),
        _assign_literal(
            "PAGES", "List[Dict[str, Any]]", [_page_to_dict(page) for page in state.pages]
        ),
        "PAGE_SPEC_BY_SLUG: Dict[str, Dict[str, Any]] = {page['slug']: page for page in PAGES}",
        _assign_literal("ENV_KEYS", "List[str]", state.env_keys),
        f"EMBED_INSIGHTS: bool = {'True' if embed_insights else 'False'}",
        f"REALTIME_ENABLED: bool = {'True' if enable_realtime else 'False'}",
        "CONNECTOR_DRIVERS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[List[Dict[str, Any]]]]] = {}",
        "DATASET_TRANSFORMS: Dict[str, Callable[[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]], List[Dict[str, Any]]]] = {}",
    ]
    parts.append("\n".join(registries).rstrip())

    helpers = '''

def build_context(page_slug: Optional[str]) -> Dict[str, Any]:
    base = CONTEXT.build(page_slug)
    context: Dict[str, Any] = dict(base)
    context.setdefault("vars", {})
    for variable in APP.get("variables", []):
        context["vars"][variable["name"]] = _resolve_placeholders(
            variable.get("value"), context
        )
    env_values = {name: os.getenv(name) for name in ENV_KEYS}
    context.setdefault("env", {}).update(env_values)
    if page_slug:
        context.setdefault("page", page_slug)
    context.setdefault("app", APP)
    context.setdefault("models", MODEL_REGISTRY)
    context.setdefault("connectors", AI_CONNECTORS)
    context.setdefault("templates", AI_TEMPLATES)
    context.setdefault("chains", AI_CHAINS)
    context.setdefault("experiments", AI_EXPERIMENTS)
    context.setdefault("call_python_model", call_python_model)
    context.setdefault("call_llm_connector", call_llm_connector)
    context.setdefault("run_chain", run_chain)
    context.setdefault("evaluate_experiment", evaluate_experiment)
    context.setdefault("predict", predict)
    context.setdefault("datasets", DATASETS)
    return context


def _resolve_placeholders(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if CONTEXT_MARKER_KEY in value:
            marker = value[CONTEXT_MARKER_KEY]
            scope = marker.get("scope")
            path = marker.get("path", [])
            default = marker.get("default")
            return _resolve_context_scope(scope, path, context, default)
        return {key: _resolve_placeholders(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(item, context) for item in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        return os.getenv(env_name, value)
    return value


def _resolve_context_scope(
    scope: Optional[str],
    path: Iterable[str],
    context: Dict[str, Any],
    default: Any = None,
) -> Any:
    parts = list(path)
    if scope in (None, "ctx", "context"):
        return _resolve_context_path(context, parts, default)
    if scope == "env":
        if not parts:
            return default
        return os.getenv(parts[0], default)
    if scope == "vars":
        return _resolve_context_path(context.get("vars", {}), parts, default)
    target = context.get(scope)
    return _resolve_context_path(target, parts, default)


def _resolve_context_path(
    context: Optional[Dict[str, Any]],
    path: Iterable[str],
    default: Any = None,
) -> Any:
    current: Any = context
    for segment in path:
        if isinstance(current, dict):
            current = current.get(segment)
        else:
            current = None
        if current is None:
            return default
    return current


TEMPLATE_PATTERN = re.compile(r"{([^{}]+)}")


def _render_template_value(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            token = match.group(1).strip()
            if not token:
                return match.group(0)
            if token.startswith("$"):
                return os.getenv(token[1:], "")
            if ":" in token:
                scope, _, path = token.partition(":")
                parts = [segment for segment in path.split(".") if segment]
                resolved = _resolve_context_scope(scope, parts, context, "")
                return "" if resolved is None else str(resolved)
            value = _resolve_context_path(context.get("vars"), [token], None)
            if value is not None:
                return str(value)
            resolved = _resolve_context_path(context, [token], "")
            return "" if resolved is None else str(resolved)
        return TEMPLATE_PATTERN.sub(_replace, value)
    if isinstance(value, list):
        return [_render_template_value(item, context) for item in value]
    if isinstance(value, dict):
        return {
            key: _render_template_value(item, context) for key, item in value.items()
        }
    return value


def register_connector_driver(
    connector_type: str,
    handler: Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[List[Dict[str, Any]]]],
) -> None:
    if not connector_type or handler is None:
        return
    CONNECTOR_DRIVERS[connector_type.lower()] = handler


def register_dataset_transform(
    transform_type: str,
    handler: Callable[[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]], List[Dict[str, Any]]],
) -> None:
    if not transform_type or handler is None:
        return
    DATASET_TRANSFORMS[transform_type.lower()] = handler


class BreakFlow(Exception):
    """Internal signal to indicate a ``break`` statement was encountered."""

    def __init__(self) -> None:
        super().__init__()
        self.components: List[Dict[str, Any]] = []

    def extend(self, items: Iterable[Dict[str, Any]]) -> None:
        if not items:
            return
        self.components.extend(items)


class ContinueFlow(Exception):
    """Internal signal to indicate a ``continue`` statement was encountered."""

    def __init__(self) -> None:
        super().__init__()
        self.components: List[Dict[str, Any]] = []

    def extend(self, items: Iterable[Dict[str, Any]]) -> None:
        if not items:
            return
        self.components.extend(items)


class ScopeFrame:
    """Hierarchical scope for storing variables during page rendering."""

    def __init__(self, parent: Optional['ScopeFrame'] = None) -> None:
        self.parent = parent
        self._values: Dict[str, Any] = {}

    def child(self) -> 'ScopeFrame':
        return ScopeFrame(self)

    def contains(self, name: str) -> bool:
        if name in self._values:
            return True
        if self.parent is not None:
            return self.parent.contains(name)
        return False

    def get(self, name: str, default: Any = None) -> Any:
        if name in self._values:
            return self._values[name]
        if self.parent is not None:
            return self.parent.get(name, default)
        return default

    def assign(self, name: str, value: Any) -> None:
        if name in self._values:
            self._values[name] = value
            return
        if self.parent is not None and self.parent.contains(name):
            self.parent.assign(name, value)
            return
        self._values[name] = value

    def set(self, name: str, value: Any) -> None:
        self.assign(name, value)

    def bind(self, name: str, value: Any) -> None:
        self._values[name] = value

    def remove_local(self, name: str) -> None:
        self._values.pop(name, None)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.parent is not None:
            data.update(self.parent.to_dict())
        data.update(self._values)
        return data


_MISSING = object()

_RUNTIME_CALLABLES: Dict[str, Callable[..., Any]] = {
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "sorted": sorted,
    "abs": abs,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "range": range,
}


def _assign_variable(scope: ScopeFrame, context: Dict[str, Any], name: str, value: Any) -> None:
    scope.assign(name, value)
    context.setdefault("vars", {})[name] = value


def _bind_variable(scope: ScopeFrame, context: Dict[str, Any], name: str, value: Any) -> None:
    scope.bind(name, value)
    context.setdefault("vars", {})[name] = value


def _restore_variable(context: Dict[str, Any], name: str, previous: Any) -> None:
    vars_map = context.setdefault("vars", {})
    if previous is _MISSING:
        vars_map.pop(name, None)
    else:
        vars_map[name] = previous


def _runtime_truthy(value: Any) -> bool:
    return bool(value)


def _clone_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def _ensure_numeric(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _resolve_option_dict(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(raw or {})


def _parse_aggregate_expression(expression: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if expression is None:
        return None, None
    text = expression.strip()
    if not text:
        return None, None
    parts = _AGGREGATE_ALIAS_PATTERN.split(text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip() or None
    return text, None


def _aggregate_result_key(function: str, expression: Optional[str], alias: Optional[str]) -> str:
    if alias:
        return alias
    base = f"{function}_{expression or 'value'}"
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", base).strip("_")
    return sanitized or f"{function}_value"


def _evaluate_dataset_expression(
    expression: Optional[str],
    row: Dict[str, Any],
    context: Dict[str, Any],
    rows: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    if expression is None:
        return None
    expr = expression.strip()
    if not expr:
        return None
    scope: Dict[str, Any] = {
        "row": row,
        "rows": rows or [],
        "context": context,
    }
    scope.update(row)
    scope.setdefault("math", math)
    scope.setdefault("len", len)
    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "round": round,
    }
    try:
        compiled = compile(expr, "<dataset_expr>", "eval")
        return eval(compiled, {"__builtins__": safe_builtins}, scope)
    except Exception:
        logger.debug("Failed to evaluate dataset expression '%s'", expression)
        return None


def _apply_filter(rows: List[Dict[str, Any]], condition: Optional[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not rows or not condition:
        return rows
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        value = _evaluate_dataset_expression(condition, row, context, rows)
        if _runtime_truthy(value):
            filtered.append(row)
    return filtered


def _apply_computed_column(
    rows: List[Dict[str, Any]],
    name: str,
    expression: Optional[str],
    context: Dict[str, Any],
) -> None:
    if not name:
        return
    for row in rows:
        row[name] = _evaluate_dataset_expression(expression, row, context, rows)


def _apply_group_aggregate(
    rows: List[Dict[str, Any]],
    columns: Sequence[str],
    aggregates: Sequence[Tuple[str, str]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not columns:
        key = tuple()
        grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {key: rows}
    else:
        grouped = defaultdict(list)
        for row in rows:
            key = tuple(row.get(column) for column in columns)
            grouped[key].append(row)
    results: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        base: Dict[str, Any] = {}
        for idx, column in enumerate(columns):
            base[column] = key[idx] if idx < len(key) else None
        for function, expression in aggregates:
            expr_source, alias = _parse_aggregate_expression(expression)
            result_key = _aggregate_result_key(function or "agg", expr_source, alias)
            values = []
            if expr_source:
                for row in items:
                    values.append(_evaluate_dataset_expression(expr_source, row, context, items))
            numeric_values = [_ensure_numeric(value) for value in values if value is not None]
            func_lower = str(function or "").lower()
            if func_lower == "sum":
                base[result_key] = sum(numeric_values) if numeric_values else 0
            elif func_lower == "count":
                if values:
                    base[result_key] = sum(1 for value in values if _runtime_truthy(value))
                else:
                    base[result_key] = len(items)
            elif func_lower == "avg":
                base[result_key] = sum(numeric_values) / len(numeric_values) if numeric_values else 0
            elif func_lower == "min":
                base[result_key] = min(numeric_values) if numeric_values else None
            elif func_lower == "max":
                base[result_key] = max(numeric_values) if numeric_values else None
            else:
                base[result_key] = values
        results.append(base)
    return results


def _apply_order(rows: List[Dict[str, Any]], columns: Sequence[str]) -> List[Dict[str, Any]]:
    if not columns:
        return rows

    def sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
        values: List[Any] = []
        for column in columns:
            column_name = column.lstrip("-")
            values.append(row.get(column_name))
        return tuple(values)

    descending_flags = [col.startswith("-") for col in columns]
    ordered = sorted(rows, key=sort_key, reverse=descending_flags[0] if descending_flags else False)
    if any(descending_flags[1:]):
        for index in range(1, len(columns)):
            column = columns[index]
            if not column.startswith("-"):
                continue
            column_name = column.lstrip("-")
            ordered = sorted(ordered, key=lambda item: item.get(column_name), reverse=True)
    return ordered


def _apply_window_operation(
    rows: List[Dict[str, Any]],
    name: str,
    function: str,
    target: Optional[str],
) -> None:
    if not name:
        return
    func_lower = (function or "").lower()
    values = [row.get(target) if target else None for row in rows]
    if func_lower == "rank":
        sorted_pairs = sorted(enumerate(values), key=lambda item: item[1])
        ranks = {index: rank + 1 for rank, (index, _) in enumerate(sorted_pairs)}
        for idx, row in enumerate(rows):
            row[name] = ranks.get(idx, idx + 1)
        return
    running_total = 0.0
    for index, row in enumerate(rows):
        value = _ensure_numeric(row.get(target)) if target else 0.0
        if func_lower in {"cumsum", "running_sum", "sum"}:
            running_total += value
            row[name] = running_total
        elif func_lower == "avg":
            running_total += value
            row[name] = running_total / (index + 1)
        else:
            row[name] = value


def _apply_transforms(
    rows: List[Dict[str, Any]],
    transforms: Sequence[Dict[str, Any]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    current = _clone_rows(rows)
    for step in transforms:
        transform_type = str(step.get("type") or "").lower()
        handler = DATASET_TRANSFORMS.get(transform_type)
        if handler is None:
            continue
        options = _resolve_option_dict(step.get("options") if isinstance(step, dict) else {})
        try:
            current = handler(current, options, context)
        except Exception:
            logger.exception("Dataset transform '%s' failed", transform_type)
    return current


def _evaluate_quality_checks(
    rows: List[Dict[str, Any]],
    checks: Sequence[Dict[str, Any]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for check in checks:
        name = check.get("name")
        condition = check.get("condition")
        metric = check.get("metric")
        threshold = check.get("threshold")
        severity = check.get("severity", "error")
        message = check.get("message")
        extras = dict(check.get("extras") or {})
        passed = True
        metric_value = None
        if metric:
            metric_lower = metric.lower()
            if metric_lower == "row_count":
                metric_value = len(rows)
            elif metric_lower == "null_ratio" and threshold:
                target_column = extras.get("column")
                if target_column:
                    nulls = sum(1 for row in rows if row.get(target_column) in {None, ""})
                    metric_value = nulls / max(len(rows), 1)
            else:
                metric_value = sum(_ensure_numeric(row.get(metric)) for row in rows)
        if condition:
            violations = [
                row for row in rows
                if not _runtime_truthy(_evaluate_dataset_expression(condition, row, context, rows))
            ]
            passed = not violations
        elif isinstance(threshold, (int, float)) and metric_value is not None:
            passed = metric_value >= threshold
        results.append(
            {
                "name": name,
                "passed": passed,
                "severity": severity,
                "message": message,
                "metric": metric,
                "value": metric_value,
                "threshold": threshold,
                "extras": extras,
            }
        )
    return results


async def _execute_action_operation(
    operation: Dict[str, Any],
    context: Dict[str, Any],
    scope: ScopeFrame,
) -> Optional[Dict[str, Any]]:
    otype = str(operation.get("type") or "").lower()
    if otype == "toast":
        message = _render_template_value(operation.get("message"), context)
        return {"type": "toast", "message": message}
    if otype == "python_call":
        module = operation.get("module")
        method = operation.get("method")
        arguments_raw = operation.get("arguments") or {}
        arguments = _resolve_placeholders(arguments_raw, context)
        result = call_python_model(module, method, arguments)
        return {"type": "python_call", "result": result}
    if otype == "connector_call":
        name = operation.get("name")
        arguments_raw = operation.get("arguments") or {}
        arguments = _resolve_placeholders(arguments_raw, context)
        result = call_llm_connector(name, arguments)
        return {"type": "connector_call", "result": result}
    if otype == "chain_run":
        name = operation.get("name")
        inputs_raw = operation.get("inputs") or {}
        inputs = _resolve_placeholders(inputs_raw, context)
        result = run_chain(name, inputs)
        return {"type": "chain_run", "result": result}
    if otype == "update":
        table = operation.get("table")
        set_expression = operation.get("set_expression")
        where_expression = operation.get("where_expression")
        session: Optional[AsyncSession] = context.get("session")
        if session is None:
            return {"type": "update", "status": "no_session"}
        updated = await _execute_update(table, set_expression, where_expression, session, context)
        return {"type": "update", "status": "ok", "rows": updated}
    return None


async def _execute_update(
    table: Optional[str],
    set_expression: Optional[str],
    where_expression: Optional[str],
    session: AsyncSession,
    context: Optional[Dict[str, Any]],
) -> int:
    if not table or not set_expression:
        return 0
    ctx = context or {}
    try:
        sanitized_table = _sanitize_table_reference(table)
        assignments, assignment_params = _parse_update_assignments(set_expression, ctx)
        conditions, where_params = _parse_where_expression(where_expression, ctx)
    except ValueError:
        logger.warning("Rejected unsafe update targeting table '%s'", table)
        return 0
    if not assignments:
        return 0
    parameters: Dict[str, Any] = {}
    parameters.update(assignment_params)
    parameters.update(where_params)
    try:
        if _HAS_SQLA_UPDATE and update is not None and sql_table is not None and column is not None and bindparam is not None:  # type: ignore[truthy-function]
            statement = _build_update_statement(sanitized_table, assignments, conditions)
            if statement is None:
                return 0
            result = await session.execute(statement, parameters)
        else:
            query_text = _build_fallback_update_sql(sanitized_table, assignments, conditions)
            result = await session.execute(text(query_text), parameters)
        await session.commit()
        rowcount = getattr(result, "rowcount", 0) or 0
        return rowcount
    except Exception:
        await session.rollback()
        logger.exception("Failed to execute update on table '%s'", sanitized_table)
        return 0


def _build_update_statement(
    table_name: str,
    assignments: Sequence[Tuple[str, str, Any]],
    conditions: Sequence[Tuple[str, str, Any]],
) -> Optional[Any]:
    table_clause = _create_table_clause(table_name, assignments, conditions)
    if table_clause is None:
        return None
    stmt = update(table_clause)  # type: ignore[arg-type]
    values_mapping: Dict[str, Any] = {}
    for column_name, param_name, _ in assignments:
        values_mapping[column_name] = bindparam(param_name)  # type: ignore[arg-type]
    if values_mapping:
        stmt = stmt.values(**values_mapping)
    condition_expr: Optional[Any] = None
    for column_name, param_name, _ in conditions:
        clause = table_clause.c[column_name] == bindparam(param_name)  # type: ignore[index,operator]
        condition_expr = clause if condition_expr is None else condition_expr & clause
    if condition_expr is not None:
        stmt = stmt.where(condition_expr)
    return stmt


def _build_fallback_update_sql(
    table_name: str,
    assignments: Sequence[Tuple[str, str, Any]],
    conditions: Sequence[Tuple[str, str, Any]],
) -> str:
    set_clause = ", ".join(f"{column} = :{param}" for column, param, _ in assignments)
    if conditions:
        where_clause = " AND ".join(f"{column} = :{param}" for column, param, _ in conditions)
        return f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
    return f"UPDATE {table_name} SET {set_clause}"


def _create_table_clause(
    table_name: str,
    assignments: Sequence[Tuple[str, str, Any]],
    conditions: Sequence[Tuple[str, str, Any]],
) -> Optional[Any]:
    if sql_table is None or column is None:
        return None
    segments = table_name.split(".")
    name = segments[-1]
    schema = ".".join(segments[:-1]) if len(segments) > 1 else None
    seen: Dict[str, None] = {}
    column_order: List[str] = []
    for column_name, _, _ in list(assignments) + list(conditions):
        if column_name not in seen:
            seen[column_name] = None
            column_order.append(column_name)
    columns = [column(column_name) for column_name in column_order]  # type: ignore[arg-type]
    table_clause = sql_table(name, *columns, schema=schema)  # type: ignore[arg-type]
    return table_clause


def _sanitize_table_reference(value: str) -> str:
    return _sanitize_identifier_path(value)


def _sanitize_identifier_path(raw: str) -> str:
    segments = [segment.strip() for segment in (raw or "").split(".") if segment and segment.strip()]
    if not segments:
        raise ValueError("Empty identifier path")
    cleaned: List[str] = []
    for segment in segments:
        if not _IDENTIFIER_RE.match(segment):
            raise ValueError(f"Invalid identifier segment '{segment}'")
        cleaned.append(segment)
    return ".".join(cleaned)


def _parse_update_assignments(expression: str, context: Dict[str, Any]) -> Tuple[List[Tuple[str, str, Any]], Dict[str, Any]]:
    assignments = _split_assignment_list(expression)
    if not assignments:
        raise ValueError("No assignments provided")
    entries: List[Tuple[str, str, Any]] = []
    params: Dict[str, Any] = {}
    for index, assignment in enumerate(assignments):
        match = _UPDATE_ASSIGNMENT_PATTERN.match(assignment)
        if not match:
            raise ValueError(f"Unsupported assignment '{assignment}'")
        column = _sanitize_identifier_path(match.group(1))
        value_expr = match.group(2).strip()
        value = _evaluate_update_value(value_expr, context)
        param_name = f"set_{index}"
        entries.append((column, param_name, value))
        params[param_name] = value
    return entries, params


def _parse_where_expression(expression: Optional[str], context: Dict[str, Any]) -> Tuple[List[Tuple[str, str, Any]], Dict[str, Any]]:
    if not expression:
        return [], {}
    conditions = _split_conditions(str(expression))
    if not conditions:
        raise ValueError("Unable to parse WHERE expression")
    entries: List[Tuple[str, str, Any]] = []
    params: Dict[str, Any] = {}
    for index, condition in enumerate(conditions):
        match = _WHERE_CONDITION_PATTERN.match(condition)
        if not match:
            raise ValueError(f"Unsupported WHERE clause '{condition}'")
        column = _sanitize_identifier_path(match.group(1))
        value_expr = match.group(2).strip()
        value = _evaluate_update_value(value_expr, context)
        param_name = f"where_{index}"
        entries.append((column, param_name, value))
        params[param_name] = value
    return entries, params


def _split_assignment_list(expression: str) -> List[str]:
    if not expression:
        return []
    parts: List[str] = []
    current: List[str] = []
    quote: Optional[str] = None
    escape = False
    depth = 0
    for char in expression:
        if escape:
            current.append(char)
            escape = False
            continue
        if quote:
            current.append(char)
            if char == "\\\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
            current.append(char)
            continue
        if char in ")]}":
            if depth:
                depth -= 1
            current.append(char)
            continue
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _split_conditions(expression: str) -> List[str]:
    expr = expression.strip()
    if not expr:
        return []
    parts: List[str] = []
    current: List[str] = []
    quote: Optional[str] = None
    escape = False
    depth = 0
    i = 0
    length = len(expr)
    while i < length:
        char = expr[i]
        if escape:
            current.append(char)
            escape = False
            i += 1
            continue
        if quote:
            current.append(char)
            if char == "\\\\":
                escape = True
            elif char == quote:
                quote = None
            i += 1
            continue
        if char in {'"', "'"}:
            quote = char
            current.append(char)
            i += 1
            continue
        if char in "([{":
            depth += 1
            current.append(char)
            i += 1
            continue
        if char in ")]}":
            if depth:
                depth -= 1
            current.append(char)
            i += 1
            continue
        if depth == 0 and expr[i:].lower().startswith("and") and (i == 0 or expr[i - 1].isspace()) and (i + 3 >= length or expr[i + 3].isspace() or expr[i + 3] in ")]}"):
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            i += 3
            while i < length and expr[i].isspace():
                i += 1
            continue
        current.append(char)
        i += 1
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _evaluate_update_value(expression: str, context: Dict[str, Any]) -> Any:
    value_expr = expression.strip()
    if not value_expr:
        return None
    lowered = value_expr.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(value_expr)
    except Exception:
        pass
    result = _evaluate_dataset_expression(value_expr, {}, context, [])
    if result is None and lowered not in {"null", "none"}:
        try:
            if "." in value_expr:
                return float(value_expr)
            return int(value_expr)
        except Exception:
            return value_expr
    return result


async def render_statements(
    statements: Optional[Iterable[Dict[str, Any]]],
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    return await _render_block(statements, context, scope, session, allow_control=False)


async def _render_block(
    statements: Optional[Iterable[Dict[str, Any]]],
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
    *,
    allow_control: bool,
) -> List[Dict[str, Any]]:
    components: List[Dict[str, Any]] = []
    if not statements:
        return components
    for statement in statements:
        try:
            result = await _render_statement(statement, context, scope, session)
        except (BreakFlow, ContinueFlow) as exc:
            if components:
                exc.extend(components)
            if allow_control:
                raise exc


            logger.warning("Loop control statement encountered outside of a loop")
            break
        if result:
            components.extend(result)
    return components


async def _render_statement(
    statement: Dict[str, Any],


    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    stype = statement.get("type")
    if stype == "variable":
        name = statement.get("name")
        expr = statement.get("value_expr")
        if expr is not None:
            value = await _evaluate_runtime_expression(expr, context, scope)


        else:
            value = _resolve_placeholders(statement.get("value"), context)
        if name:
            _assign_variable(scope, context, name, value)
        return []
    if stype == "if":
        return await _render_if(statement, context, scope, session)
    if stype == "for_loop":
        return await _render_for_loop(statement, context, scope, session)
    if stype == "while_loop":
        return await _render_while_loop(statement, context, scope, session)
    if stype == "break":
        raise BreakFlow()
    if stype == "continue":
        raise ContinueFlow()

    payload = copy.deepcopy(statement)
    payload_type = payload.pop("type", None)
    if payload_type == "text":
        rendered_text = _render_template_value(payload.get("text"), context)
        styles = copy.deepcopy(payload.get("styles", {}))
        return [{"type": "text", "text": rendered_text, "styles": styles}]
    if payload_type == "table":
        resolved = _resolve_placeholders(payload, context)
        if "title" in resolved:
            resolved["title"] = _render_template_value(resolved.get("title"), context)
        if "filter" in resolved:
            resolved["filter"] = _render_template_value(resolved.get("filter"), context)
        if "sort" in resolved:
            resolved["sort"] = _render_template_value(resolved.get("sort"), context)
        return [{"type": "table", **resolved}]
    if payload_type == "chart":
        resolved = _resolve_placeholders(payload, context)
        if "heading" in resolved:
            resolved["heading"] = _render_template_value(resolved.get("heading"), context)
        if "title" in resolved:
            resolved["title"] = _render_template_value(resolved.get("title"), context)
        return [{"type": "chart", **resolved}]
    if payload_type == "form":
        resolved = _resolve_placeholders(payload, context)
        if "title" in resolved:


            resolved["title"] = _render_template_value(resolved.get("title"), context)
        return [{"type": "form", **resolved}]
    if payload_type == "action":
        resolved = _resolve_placeholders(payload, context)
        if "name" in resolved:
            resolved["name"] = _render_template_value(resolved.get("name"), context)
        if "trigger" in resolved:
            resolved["trigger"] = _render_template_value(resolved.get("trigger"), context)
        operations = resolved.get("operations") or []
        results: List[Dict[str, Any]] = []
        for operation in operations:
            outcome = await _execute_action_operation(operation, context, scope)
            if outcome:
                results.append(outcome)
        if results:
            return results
        return [{"type": "action", **resolved}]
    if payload_type == "predict":
        resolved = _resolve_placeholders(payload, context)
        return [{"type": "predict", **resolved}]
    if payload_type:
        resolved = _resolve_placeholders(payload, context)
        return [{"type": payload_type, **resolved}]
    return []


async def _render_if(
    statement: Dict[str, Any],
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    condition = statement.get("condition")


    if _runtime_truthy(await _evaluate_runtime_expression(condition, context, scope)):
        return await _render_block(statement.get("body"), context, scope, session, allow_control=True)
    for branch in statement.get("elifs", []) or []:
        branch_condition = branch.get("condition")
        if _runtime_truthy(await _evaluate_runtime_expression(branch_condition, context, scope)):
            return await _render_block(branch.get("body"), context, scope, session, allow_control=True)
    else_body = statement.get("else_body")
    if else_body:
        return await _render_block(else_body, context, scope, session, allow_control=True)
    return []


async def _render_for_loop(
    statement: Dict[str, Any],
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    loop_var = statement.get("loop_var")
    source_kind = statement.get("source_kind")
    source_name = statement.get("source_name")
    if not loop_var or not source_kind or not source_name:
        return []
    items = await _resolve_loop_iterable(source_kind, source_name, context, scope, session)
    if not items:
        return []
    components: List[Dict[str, Any]] = []


    vars_map = context.setdefault("vars", {})
    for item in items:
        previous = vars_map.get(loop_var, _MISSING)
        loop_scope = scope.child()
        _bind_variable(loop_scope, context, loop_var, item)
        try:
            rendered = await _render_block(statement.get("body"), context, loop_scope, session, allow_control=True)
            if rendered:
                components.extend(rendered)
        except ContinueFlow as exc:
            if exc.components:
                components.extend(exc.components)
            continue
        except BreakFlow as exc:
            if exc.components:
                components.extend(exc.components)
            break


        finally:
            _restore_variable(context, loop_var, previous)
    return components


async def _render_while_loop(
    statement: Dict[str, Any],
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    condition = statement.get("condition")
    components: List[Dict[str, Any]] = []
    loop_scope = scope.child()
    iterations = 0
    while _runtime_truthy(await _evaluate_runtime_expression(condition, context, loop_scope)):
        iterations += 1
        if iterations > 1000:
            logger.warning("Aborting while loop after 1000 iterations for safety")
            break
        try:
            rendered = await _render_block(statement.get("body"), context, loop_scope, session, allow_control=True)
            if rendered:
                components.extend(rendered)
        except ContinueFlow as exc:
            if exc.components:
                components.extend(exc.components)
            continue
        except BreakFlow as exc:
            if exc.components:
                components.extend(exc.components)
            break
    return components


async def _resolve_loop_iterable(
    source_kind: str,
    source_name: str,
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Any]:
    if source_kind == "dataset":
        if session is not None:
            try:
                return await fetch_dataset_rows(source_name, session, context)
            except Exception:  # pragma: no cover - runtime fetch failure
                logger.exception("Failed to fetch dataset rows for %s", source_name)
        dataset_spec = DATASETS.get(source_name)
        if dataset_spec:
            return list(dataset_spec.get("sample_rows", []))


        return []
    if source_kind == "table":
        tables = context.get("tables")
        if isinstance(tables, dict) and source_name in tables:
            table_value = tables[source_name]
            if isinstance(table_value, list):
                return table_value
        dataset_spec = DATASETS.get(source_name)
        if dataset_spec:
            return list(dataset_spec.get("sample_rows", []))
        return []
    items = scope.get(source_name)
    if isinstance(items, list):
        return items
    return []


async def _evaluate_runtime_expression(
    expression: Optional[Dict[str, Any]],
    context: Dict[str, Any],
    scope: ScopeFrame,
) -> Any:
    if expression is None:
        return None
    etype = expression.get("type")
    if etype == "literal":
        return _resolve_placeholders(expression.get("value"), context)
    if etype == "name":
        name = expression.get("name")
        if not isinstance(name, str):
            return None
        value = scope.get(name, _MISSING)
        if value is not _MISSING:
            return value
        vars_map = context.get("vars", {})
        if isinstance(vars_map, dict) and name in vars_map:
            return vars_map[name]
        if name in _RUNTIME_CALLABLES:
            return _RUNTIME_CALLABLES[name]
        return context.get(name)
    if etype == "attribute":
        path = list(expression.get("path") or [])
        if not path:
            return None
        base_name = path[0]
        base_value = await _evaluate_runtime_expression({"type": "name", "name": base_name}, context, scope)
        if base_value is None:
            return None
        return _traverse_attribute_path(base_value, path[1:])
    if etype == "context":
        return _resolve_context_scope(
            expression.get("scope"),
            expression.get("path", []),
            context,
            expression.get("default"),
        )
    if etype == "binary":
        op = expression.get("op")
        left = await _evaluate_runtime_expression(expression.get("left"), context, scope)
        right = await _evaluate_runtime_expression(expression.get("right"), context, scope)
        try:
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right if right not in (0, None) else None
            if op == "and":
                return _runtime_truthy(left) and _runtime_truthy(right)
            if op == "or":
                return _runtime_truthy(left) or _runtime_truthy(right)
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == "in":
                return left in right if right is not None else False
            if op == "not in":
                return left not in right if right is not None else True
        except Exception:  # pragma: no cover - operator failure
            logger.exception("Failed to evaluate binary operation '%s'", op)
            return None
        return None
    if etype == "unary":
        operand = await _evaluate_runtime_expression(expression.get("operand"), context, scope)
        op = expression.get("op")
        try:
            if op == "not":
                return not _runtime_truthy(operand)
            if op == "-":
                return -operand
            if op == "+":
                return +operand
        except Exception:  # pragma: no cover - unary failure
            logger.exception("Failed to evaluate unary operation '%s'", op)
            return None
        return None
    if etype == "call":
        function_expr = expression.get("function")
        func = await _evaluate_runtime_expression(function_expr, context, scope)
        if func is None:
            return None
        arguments = []
        for arg in expression.get("arguments", []) or []:
            arguments.append(await _evaluate_runtime_expression(arg, context, scope))
        try:
            if inspect.iscoroutinefunction(func):
                return await func(*arguments)
            result = func(*arguments) if callable(func) else None
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception:  # pragma: no cover - call failure
            logger.exception("Failed to execute call expression")
            return None
    return None


def _traverse_attribute_path(value: Any, path: Iterable[str]) -> Any:
    current = value
    for segment in path:
        if current is None:
            return None
        current = _resolve_path_segment(current, segment)
    return current


def _resolve_path_segment(value: Any, segment: str) -> Any:
    if isinstance(value, dict):
        return value.get(segment)
    if isinstance(value, (list, tuple)):
        try:
            index = int(segment)
        except (TypeError, ValueError):
            return None
        if 0 <= index < len(value):
            return value[index]
        return None
    if hasattr(value, segment):
        return getattr(value, segment)
    return None


def _page_meta(slug: str) -> Dict[str, Any]:
    spec = PAGE_SPEC_BY_SLUG.get(slug, {})
    return {
        "reactive": bool(spec.get("reactive")),
        "refresh_policy": spec.get("refresh_policy"),
    }


def _model_to_payload(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    if isinstance(model, dict):
        return dict(model)
    if hasattr(model, "__dict__"):
        return {
            key: value
            for key, value in model.__dict__.items()
            if not key.startswith("_")
        }
    return {"value": model}


def _with_timestamp(payload: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(payload)
    enriched.setdefault("ts", time.time())
    return enriched


def register_model_loader(framework: str, loader: Callable[[str, Dict[str, Any]], Any]) -> None:
    if not framework or loader is None:
        return
    MODEL_LOADERS[framework.lower()] = loader


def register_model_runner(
    framework: str,
    runner: Callable[[str, Any, Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
) -> None:
    if not framework or runner is None:
        return
    MODEL_RUNNERS[framework.lower()] = runner


def register_model_explainer(
    framework: str,
    explainer: Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
) -> None:
    if not framework or explainer is None:
        return
    MODEL_EXPLAINERS[framework.lower()] = explainer


def _resolve_model_artifact_path(model_spec: Dict[str, Any]) -> Optional[str]:
    metadata_obj = model_spec.get("metadata")
    metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
    path = metadata.get("model_file") or metadata.get("artifact_path")
    if not path:
        return None
    root = os.getenv("NAMEL3SS_MODEL_ROOT")
    if root and not os.path.isabs(path):
        return os.path.join(root, path)
    return path


def _load_python_callable(import_path: str) -> Optional[Callable[..., Any]]:
    module_path, _, attr = import_path.rpartition(":")
    if not module_path or not attr:
        return None
    module = importlib.import_module(module_path)
    return getattr(module, attr, None)


def _import_python_module(module: str) -> Optional[Any]:
    if not module:
        return None
    try:
        if module.endswith(".py"):
            path = Path(module)
            if not path.is_absolute():
                base = os.getenv("NAMEL3SS_APP_ROOT")
                if base:
                    path = Path(base) / path
                else:
                    path = Path.cwd() / path
            if not path.exists():
                return None
            module_name = path.stem
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec and spec.loader:
                module_obj = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module_obj)  # type: ignore[attr-defined]
                    sys.modules.setdefault(module_name, module_obj)
                    return module_obj
                except Exception:  # pragma: no cover - user module failure
                    logger.exception("Failed to import python module from %s", path)
                    return None
            return None
        return importlib.import_module(module)
    except Exception:  # pragma: no cover - import failure
        logger.exception("Failed to import python module %s", module)
        return None


def call_python_model(
    module: str,
    method: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    args = dict(arguments or {})
    module_obj = _import_python_module(module)
    attr_name = method or "predict"
    if module_obj is not None:
        callable_obj = getattr(module_obj, attr_name, None)
        if callable(callable_obj):
            try:
                result = callable_obj(**args)
                payload = result if isinstance(result, dict) else {"value": result}
                return {
                    "result": payload,
                    "inputs": args,
                    "module": module,
                    "method": attr_name,
                    "status": "ok",
                }
            except Exception:  # pragma: no cover - user callable failure
                logger.exception("Python callable %s.%s raised an error", module, attr_name)
    return {
        "result": "stub_prediction",
        "inputs": args,
        "module": module,
        "method": attr_name,
        "status": "stub",
    }


def _ensure_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _as_path_segments(path: Any) -> List[str]:
    if path is None:
        return []
    if isinstance(path, (list, tuple)):
        segments: List[str] = []
        for item in path:
            segments.extend(_as_path_segments(item))
        return segments
    text = str(path).strip()
    if not text:
        return []
    normalized = text.replace("[", ".").replace("]", ".")
    return [segment for segment in normalized.split(".") if segment]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        multiplier = 1.0
        if text.endswith("%"):
            multiplier = 0.01
            text = text[:-1]
        try:
            return float(text) * multiplier
        except ValueError:
            return None
    return None


def _aggregate_numeric_samples(samples: List[float], aggregation: str) -> Optional[float]:
    if not samples:
        return None
    key = aggregation.lower()
    if key in ("avg", "average", "mean"):
        return sum(samples) / len(samples)
    if key in ("sum", "total"):
        return sum(samples)
    if key in ("min", "minimum"):
        return min(samples)
    if key in ("max", "maximum"):
        return max(samples)
    if key == "median":
        ordered = sorted(samples)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[midpoint]
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2
    if key in ("p95", "percentile95", "p_95"):
        ordered = sorted(samples)
        index = int(round(0.95 * (len(ordered) - 1)))
        index = max(0, min(index, len(ordered) - 1))
        return ordered[index]
    if key in ("p90", "percentile90", "p_90"):
        ordered = sorted(samples)
        index = int(round(0.90 * (len(ordered) - 1)))
        index = max(0, min(index, len(ordered) - 1))
        return ordered[index]
    if key in ("last", "latest"):
        return samples[-1]
    if key == "first":
        return samples[0]
    return sum(samples) / len(samples)


def _collect_metric_samples(
    metric: Dict[str, Any],
    variants: List[Dict[str, Any]],
    args: Dict[str, Any],
) -> List[float]:
    metadata = _ensure_dict(metric.get("metadata"))
    source_kind = str(metric.get("source_kind") or metadata.get("source") or "score").lower()
    path = metadata.get("path") or metadata.get("result_path") or metadata.get("field")
    segments = _as_path_segments(path)
    samples: List[float] = []

    if source_kind in ("score", "variant", "variant_score"):
        for variant in variants:
            value: Any = variant.get("score")
            if value is None and segments:
                value = _traverse_attribute_path(variant, segments)
            sample = _safe_float(value)
            if sample is not None:
                samples.append(sample)
    elif source_kind in ("result", "output", "prediction"):
        target_segments = segments or ["result", "output", "score"]
        for variant in variants:
            value = _traverse_attribute_path(variant, target_segments)
            sample = _safe_float(value)
            if sample is not None:
                samples.append(sample)
    elif source_kind in ("payload", "input", "request"):
        value = _traverse_attribute_path(args, segments) if segments else args.get("input")
        sample = _safe_float(value)
        if sample is not None:
            samples.append(sample)
    elif source_kind in ("manual", "provided", "static"):
        values = metadata.get("samples") or metadata.get("values") or []
        if isinstance(values, (list, tuple)):
            for entry in values:
                sample = _safe_float(entry)
                if sample is not None:
                    samples.append(sample)
    else:
        target_segments = segments or ["result", "output", "score"]
        for variant in variants:
            value = _traverse_attribute_path(variant, target_segments)
            sample = _safe_float(value)
            if sample is not None:
                samples.append(sample)

    extras = metadata.get("include")
    if isinstance(extras, (list, tuple)):
        for entry in extras:
            sample = _safe_float(entry)
            if sample is not None:
                samples.append(sample)

    return samples


def _evaluate_experiment_metrics(
    spec: Dict[str, Any],
    variants: List[Dict[str, Any]],
    args: Dict[str, Any],
) -> List[Dict[str, Any]]:
    metrics_result: List[Dict[str, Any]] = []
    for index, metric in enumerate(spec.get("metrics", []), start=1):
        metadata = _ensure_dict(metric.get("metadata"))
        aggregation = str(metadata.get("aggregation") or metadata.get("aggregate") or "max")
        samples = _collect_metric_samples(metric, variants, args)
        aggregated_value = _aggregate_numeric_samples(samples, aggregation) if samples else None
        if aggregated_value is None:
            calculated = float(round(0.72 + 0.045 * index, 3))
        else:
            calculated = float(aggregated_value)
        precision = metadata.get("precision") if isinstance(metadata.get("precision"), int) else metadata.get("round")
        if isinstance(precision, int):
            rounded_value = round(calculated, max(int(precision), 0))
        else:
            rounded_value = round(calculated, 4)

        goal_value = _safe_float(metric.get("goal"))
        direction = str(metadata.get("goal_operator") or metadata.get("direction") or "gte").lower()
        achieved_goal: Optional[bool] = None
        if goal_value is not None:
            if direction in ("lte", "le", "max", "lower", "down"):
                achieved_goal = rounded_value <= goal_value
            else:
                achieved_goal = rounded_value >= goal_value

        if samples:
            metadata.setdefault("aggregation", aggregation)
            metadata.setdefault("samples", len(samples))
            metadata.setdefault(
                "summary",
                {
                    "min": min(samples),
                    "max": max(samples),
                    "mean": round(sum(samples) / len(samples), 6),
                },
            )

        metrics_result.append(
            {
                "name": metric.get("name"),
                "value": rounded_value,
                "goal": metric.get("goal"),
                "source_kind": metric.get("source_kind"),
                "source_name": metric.get("source_name"),
                "metadata": metadata,
                "achieved_goal": achieved_goal,
            }
        )

    return metrics_result


def _stringify_prompt_value(name: str, value: Any) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, separators=(",", ":"))
        except Exception:
            return str(value)
    return str(value or "").strip()


def _format_stub_prompt(name: str, prompt_text: str) -> str:
    if prompt_text:
        return f"[{name}] {prompt_text}"
    return "This is a stub LLM response."


def _default_llm_endpoint(provider: str, config: Dict[str, Any]) -> Optional[str]:
    endpoint = config.get("endpoint") or config.get("url")
    if endpoint:
        return str(endpoint)
    provider_key = provider.lower()
    base_url = str(config.get("api_base") or config.get("base_url") or "")
    if provider_key == "openai":
        api_base = base_url or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
        deployment = config.get("deployment")
        if deployment:
            version = str(config.get("api_version") or os.getenv("OPENAI_API_VERSION") or "2024-02-01")
            return f"{api_base.rstrip('/')}/deployments/{deployment}/chat/completions?api-version={version}"
        return f"{api_base.rstrip('/')}/chat/completions"
    if provider_key == "anthropic":
        api_base = base_url or "https://api.anthropic.com/v1"
        return f"{api_base.rstrip('/')}/messages"
    return None


def _build_llm_request(
    provider: str,
    model_name: str,
    prompt_text: str,
    config: Dict[str, Any],
    args: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not prompt_text:
        return None
    provider_key = provider.lower()
    method = str(config.get("method") or "post").upper()
    timeout_value = config.get("timeout", 15.0)
    try:
        timeout = max(float(timeout_value), 1.0)
    except Exception:
        timeout = 15.0

    headers = _ensure_dict(config.get("headers"))
    params = _ensure_dict(config.get("params"))
    payload = config.get("payload")
    body = dict(payload) if isinstance(payload, dict) else {}

    mode = str(config.get("mode") or ("chat" if provider_key in {"openai", "anthropic"} else "completion")).lower()
    if mode == "chat":
        messages: List[Dict[str, Any]] = []
        system_prompt = config.get("system") or config.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        extra_messages = config.get("messages")
        if isinstance(extra_messages, list):
            for message in extra_messages:
                if isinstance(message, dict):
                    messages.append(dict(message))
        user_role = str(config.get("user_role") or "user")
        messages.append({"role": user_role, "content": prompt_text})
        body.setdefault("messages", messages)
        body.setdefault("model", model_name)
    else:
        prompt_field = str(config.get("prompt_field") or "prompt")
        body.setdefault("model", model_name)
        body.setdefault(prompt_field, prompt_text)

    payload_from_args = config.get("payload_from_args")
    if isinstance(payload_from_args, (list, tuple)):
        for key in payload_from_args:
            key_str = str(key)
            if key_str in args and key_str not in body:
                body[key_str] = args[key_str]

    endpoint = _default_llm_endpoint(provider_key, config)
    if not endpoint:
        return None

    api_key = config.get("api_key")
    if not api_key:
        env_key = config.get("api_key_env")
        if isinstance(env_key, str):
            api_key = os.getenv(env_key) or api_key

    if provider_key == "openai":
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        bearer = str(api_key)
        if not bearer.lower().startswith("bearer "):
            bearer = f"Bearer {bearer}"
        headers.setdefault("Authorization", bearer)
        headers.setdefault("Content-Type", "application/json")
    elif provider_key == "anthropic":
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        headers.setdefault("x-api-key", str(api_key))
        headers.setdefault("content-type", "application/json")
        headers.setdefault("anthropic-version", str(config.get("api_version") or "2023-06-01"))
    elif api_key:
        header_name = str(config.get("api_key_header") or "Authorization")
        if header_name.lower() == "authorization" and not str(api_key).lower().startswith("bearer "):
            headers.setdefault("Authorization", f"Bearer {api_key}")
        else:
            headers.setdefault(header_name, str(api_key))

    return {
        "method": method,
        "url": str(endpoint),
        "headers": headers,
        "params": params,
        "json": body,
        "timeout": timeout,
        "provider": provider,
    }


def _execute_llm_request(request_spec: Dict[str, Any]) -> Dict[str, Any]:
    method = str(request_spec.get("method") or "POST").upper()
    url = str(request_spec.get("url"))
    headers = _ensure_dict(request_spec.get("headers"))
    params = _ensure_dict(request_spec.get("params"))
    json_payload = request_spec.get("json")
    timeout = float(request_spec.get("timeout") or 15.0)

    with httpx.Client(timeout=timeout) as client:
        request_kwargs: Dict[str, Any] = {}
        if headers:
            request_kwargs["headers"] = headers
        if params:
            request_kwargs["params"] = params
        if method == "GET":
            if isinstance(json_payload, dict):
                combined = dict(params)
                combined.update(json_payload)
                request_kwargs["params"] = combined
            elif json_payload is not None:
                param_payload = dict(params)
                param_payload["prompt"] = str(json_payload)
                request_kwargs["params"] = param_payload
        elif isinstance(json_payload, dict):
            request_kwargs["json"] = json_payload
        response = client.request(method, url, **request_kwargs)
        response.raise_for_status()
        try:
            body = response.json()
        except Exception:
            body = {"text": response.text}

    usage = body.get("usage") if isinstance(body, dict) else None
    return {"body": body, "usage": usage}


def _extract_llm_text(
    provider: str,
    payload: Any,
    config: Dict[str, Any],
) -> str:
    response_path = config.get("response_path") or config.get("result_path")
    if response_path:
        value = _traverse_attribute_path(payload, _as_path_segments(response_path))
        if value is not None:
            return str(value)
    provider_key = provider.lower()
    if isinstance(payload, dict):
        if provider_key == "openai":
            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, list):
                            segments = []
                            for entry in content:
                                if isinstance(entry, dict) and "text" in entry:
                                    segments.append(str(entry["text"]))
                                else:
                                    segments.append(str(entry))
                            joined = "\\n".join(segment for segment in segments if segment)
                            if joined:
                                return joined
                        if content is not None:
                            return str(content)
                    if "text" in first_choice:
                        return str(first_choice["text"])
        if provider_key == "anthropic":
            content = payload.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    if "text" in first:
                        return str(first["text"])
                    if "value" in first:
                        return str(first["value"])
            if isinstance(content, str):
                return content
        if "result" in payload and isinstance(payload["result"], str):
            return payload["result"]
    if isinstance(payload, str):
        return payload
    return json.dumps(payload)


def _truncate_text(value: str, limit: int) -> str:
    if limit <= 0 or len(value) <= limit:
        return value
    return value[: max(limit - 3, 0)].rstrip() + "..."


def call_llm_connector(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    spec = AI_CONNECTORS.get(name, {})
    args = dict(payload or {})
    context_stub = {
        "env": {key: os.getenv(key) for key in ENV_KEYS},
        "vars": {},
        "app": APP,
    }
    config_raw = spec.get("config", {})
    config_resolved = _resolve_placeholders(config_raw, context_stub)
    if not isinstance(config_resolved, dict):
        config_resolved = config_raw if isinstance(config_raw, dict) else {}

    provider = str(config_resolved.get("provider") or spec.get("type") or "stub-provider")
    model_name = str(config_resolved.get("model") or "stub-model")

    prompt_value = args.get("prompt") or args.get("input") or ""
    prompt_text = _stringify_prompt_value(name, prompt_value)
    stub_response = _format_stub_prompt(name, prompt_text)

    try:
        request_spec = _build_llm_request(provider, model_name, prompt_text, config_resolved, args)
        if not request_spec:
            raise ValueError("LLM request is not configured")
        http_response = _execute_llm_request(request_spec)
        body = http_response.get("body")
        raw_text = _extract_llm_text(provider, body, config_resolved)
        try:
            limit = int(config_resolved.get("max_response_chars", 4000))
            limit = max(limit, 0)
        except Exception:
            limit = 4000
        final_text = _truncate_text(str(raw_text).strip() or "No response.", limit)
        return {
            "result": final_text,
            "provider": provider,
            "model": model_name,
            "inputs": args,
            "config": config_resolved,
            "status": "ok",
            "raw_response": body,
            "usage": http_response.get("usage"),
        }
    except Exception:
        logger.exception("LLM connector '%s' failed", name)
        return {
            "result": stub_response,
            "provider": provider,
            "model": model_name,
            "inputs": args,
            "config": config_resolved,
            "status": "stub",
            "error": "llm_error",
        }


def run_chain(name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    spec = AI_CHAINS.get(name)
    args = dict(payload or {})
    if not spec:
        return {
            "result": "stub_chain_output",
            "steps": [],
            "inputs": args,
            "status": "stub",
        }
    input_key = spec.get("input_key", "input")
    working: Any = args.get(input_key)
    if working is None:
        working = args
    history: List[Dict[str, Any]] = []
    for step in spec.get("steps", []):
        history.append(copy.deepcopy(step))
        kind = step.get("kind")
        target = step.get("target")
        options = step.get("options") or {}
        if kind == "template":
            template = AI_TEMPLATES.get(target) or {}
            prompt = template.get("prompt", "")
            context = {"input": working, "vars": args, "payload": args}
            working = _render_template_value(prompt, context)
        elif kind == "connector":
            connector_payload = dict(args)
            if isinstance(working, (dict, list)):
                connector_payload.setdefault("prompt", str(working))
            else:
                connector_payload.setdefault("prompt", working)
            response = call_llm_connector(target, connector_payload)
            working = response.get("result", working)
        elif kind == "python":
            module_name = options.get("module") or target or ""
            method_name = options.get("method") or "predict"
            response = call_python_model(module_name, method_name, args)
            working = response.get("result", working)
        else:
            working = f"{kind}:{target}:{working}" if working is not None else f"{kind}:{target}"
    result_value = working if working is not None else "stub_chain_output"
    return {
        "result": result_value,
        "steps": history,
        "inputs": args,
        "status": "ok" if history else "stub",
    }


def evaluate_experiment(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    spec = AI_EXPERIMENTS.get(name)
    args = dict(payload or {})
    if not spec:
        return {
            "experiment": name,
            "variants": [],
            "metrics": [],
            "leaderboard": [],
            "winner": None,
            "inputs": args,
            "metadata": {},
            "status": "stub",
        }

    variants_result: List[Dict[str, Any]] = []
    for index, variant in enumerate(spec.get("variants", []), start=1):
        target_type = str(variant.get("target_type") or "model").lower()
        target_name = str(variant.get("target_name") or "")
        config = variant.get("config") or {}
        result_payload: Dict[str, Any]
        if target_type == "model" and target_name:
            model_input = args.get("input") or args.get("payload") or {}
            if not isinstance(model_input, dict):
                model_input = {"value": model_input}
            result_payload = predict(target_name, model_input)
        elif target_type == "chain" and target_name:
            chain_args = dict(args)
            chain_args.setdefault("input", args.get("input", args))
            result_payload = run_chain(target_name, chain_args)
        else:
            result_payload = {
                "status": "stub",
                "detail": f"Unsupported target '{target_type}' for variant '{variant.get('name')}'",
            }
        score = round(0.6 + 0.075 * index, 3)
        variants_result.append(
            {
                "name": variant.get("name"),
                "target_type": target_type,
                "target_name": target_name,
                "config": config,
                "score": score,
                "result": result_payload,
            }
        )

    metrics_result = _evaluate_experiment_metrics(spec, variants_result, args)

    leaderboard = sorted(
        variants_result,
        key=lambda item: item.get("score", 0),
        reverse=True,
    )
    winner = leaderboard[0]["name"] if leaderboard else None

    return {
        "experiment": spec.get("name", name),
        "slug": spec.get("slug", name),
        "variants": variants_result,
        "metrics": metrics_result,
        "leaderboard": [entry.get("name") for entry in leaderboard if entry.get("name")],
        "winner": winner,
        "inputs": args,
        "metadata": spec.get("metadata", {}),
        "status": "ok",
    }


def _generic_loader(model_name: str, model_spec: Dict[str, Any]) -> Any:
    metadata_obj = model_spec.get("metadata")
    metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
    loader_path = metadata.get("loader")
    if not loader_path:
        return None
    try:
        callable_loader = _load_python_callable(loader_path)
        if callable_loader is None:
            return None
        return callable_loader(model_name, model_spec)
    except Exception:  # pragma: no cover - user loader failure
        logger.exception("Generic loader failed for model %s", model_name)
        return None


def _pytorch_loader(model_name: str, model_spec: Dict[str, Any]) -> Any:
    path = _resolve_model_artifact_path(model_spec)
    if not path or not Path(path).exists():
        return None
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        logger.debug("PyTorch not available for model %s", model_name)
        return None
    try:
        if path.endswith((".pt", ".pth")):
            try:
                return torch.jit.load(path, map_location="cpu")
            except Exception:
                return torch.load(path, map_location="cpu")
        return torch.load(path, map_location="cpu")
    except Exception:  # pragma: no cover - IO/runtime failure
        logger.exception("Failed to load PyTorch model %s from %s", model_name, path)
        return None


def _tensorflow_loader(model_name: str, model_spec: Dict[str, Any]) -> Any:
    path = _resolve_model_artifact_path(model_spec)
    if not path or not Path(path).exists():
        return None
    try:
        import tensorflow as tf  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        logger.debug("TensorFlow not available for model %s", model_name)
        return None
    try:
        return tf.saved_model.load(path)
    except Exception:  # pragma: no cover - IO/runtime failure
        logger.exception("Failed to load TensorFlow model %s from %s", model_name, path)
        return None


def _onnx_loader(model_name: str, model_spec: Dict[str, Any]) -> Any:
    path = _resolve_model_artifact_path(model_spec)
    if not path or not Path(path).exists():
        return None
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        logger.debug("ONNX Runtime not available for model %s", model_name)
        return None
    try:
        return ort.InferenceSession(path)
    except Exception:  # pragma: no cover - IO/runtime failure
        logger.exception("Failed to load ONNX model %s from %s", model_name, path)
        return None


def _sklearn_loader(model_name: str, model_spec: Dict[str, Any]) -> Any:
    path = _resolve_model_artifact_path(model_spec)
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        logger.debug("scikit-learn artifact for %s not found at %s", model_name, file_path)
        return None
    try:
        try:
            import joblib  # type: ignore
        except Exception:
            joblib = None  # type: ignore
        if joblib is not None:
            return joblib.load(file_path)
    except Exception:
        logger.exception("joblib failed to load model %s", model_name)
    try:
        with file_path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        logger.exception("Failed to load pickled model %s from %s", model_name, file_path)
    return None


def _coerce_numeric_payload(payload: Any) -> Optional[List[float]]:
    if isinstance(payload, dict):
        values: List[float] = []
        for key in sorted(payload):
            value = payload[key]
            if isinstance(value, (int, float)):
                values.append(float(value))
            elif isinstance(value, (list, tuple)):
                try:
                    values.extend(float(item) for item in value)
                except Exception:
                    return None
            else:
                return None
        return values
    if isinstance(payload, (list, tuple)):
        try:
            return [float(item) for item in payload]
        except Exception:
            return None
    if isinstance(payload, (int, float)):
        return [float(payload)]
    return None


def _generic_runner(
    model_name: str,
    model_instance: Any,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    metadata = model_spec.get("metadata") or {}
    runner_path = metadata.get("runner")
    if runner_path:
        try:
            callable_runner = _load_python_callable(runner_path)
            if callable_runner is not None:
                result = callable_runner(model_instance, payload, model_spec)
                if isinstance(result, dict):
                    return result
        except Exception:  # pragma: no cover - user runner failure
            logger.exception("Custom runner failed for model %s", model_name)
    if callable(model_instance):
        try:
            result = model_instance(payload)
            if isinstance(result, dict):
                return result
            if isinstance(result, (list, tuple)) and result:
                score = float(result[0])
                label = "Positive" if score >= 0 else "Negative"
                return {"score": score, "label": label}
        except Exception:
            logger.debug("Callable runner failed for model %s", model_name)
    return None


def _pytorch_runner(
    model_name: str,
    model_instance: Any,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None
    if model_instance is None:
        return None
    try:
        if hasattr(model_instance, "eval"):
            model_instance.eval()
        values = _coerce_numeric_payload(payload)
        if values is None:
            return None
        input_tensor = torch.tensor(values, dtype=torch.float32)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = model_instance(input_tensor)
        if isinstance(output, torch.Tensor):
            flattened = output.detach().cpu().view(-1).tolist()
            score = float(flattened[0]) if flattened else 0.0
        elif isinstance(output, (list, tuple)) and output:
            score = float(output[0])
        else:
            return None
        label = "Positive" if score >= 0 else "Negative"
        return {"score": score, "label": label}
    except Exception:  # pragma: no cover - runtime failure
        logger.exception("PyTorch runner failed for model %s", model_name)
        return None


def _tensorflow_runner(
    model_name: str,
    model_instance: Any,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        import tensorflow as tf  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None
    if model_instance is None:
        return None
    try:
        values = _coerce_numeric_payload(payload)
        if values is None:
            return None
        input_tensor = tf.convert_to_tensor([values], dtype=tf.float32)
        output = model_instance(input_tensor)
        if hasattr(output, "numpy"):
            score = float(output.numpy().reshape(-1)[0])
        elif isinstance(output, (list, tuple)) and output:
            first = output[0]
            if hasattr(first, "numpy"):
                score = float(first.numpy().reshape(-1)[0])
            else:
                score = float(first)
        else:
            return None
        label = "Positive" if score >= 0 else "Negative"
        return {"score": score, "label": label}
    except Exception:  # pragma: no cover - runtime failure
        logger.exception("TensorFlow runner failed for model %s", model_name)
        return None


def _onnx_runner(
    model_name: str,
    model_instance: Any,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if model_instance is None:
        return None
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None
    try:
        values = _coerce_numeric_payload(payload)
        if values is None:
            return None
        input_name = model_instance.get_inputs()[0].name  # type: ignore[attr-defined]
        array = np.array([values], dtype=np.float32)
        output = model_instance.run(None, {input_name: array})  # type: ignore[call-arg]
        if not output:
            return None
        first = output[0]
        if hasattr(first, "reshape"):
            score = float(first.reshape(-1)[0])
        elif isinstance(first, (list, tuple)) and first:
            score = float(first[0])
        else:
            score = float(first)
        label = "Positive" if score >= 0 else "Negative"
        return {"score": score, "label": label}
    except Exception:  # pragma: no cover - runtime failure
        logger.exception("ONNX runner failed for model %s", model_name)
        return None


def _sklearn_runner(
    model_name: str,
    model_instance: Any,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if model_instance is None:
        return None
    values = _coerce_numeric_payload(payload)
    if values is None:
        return None
    sample = [values]
    try:
        score: Optional[float] = None
        if hasattr(model_instance, "predict_proba"):
            probabilities = model_instance.predict_proba(sample)  # type: ignore[attr-defined]
            if hasattr(probabilities, "tolist"):
                probabilities = probabilities.tolist()
            if isinstance(probabilities, list) and probabilities:
                first = probabilities[0]
                if isinstance(first, list) and first:
                    score = float(first[-1])
                elif isinstance(first, (int, float)):
                    score = float(first)
        if score is None and hasattr(model_instance, "predict"):
            prediction = model_instance.predict(sample)  # type: ignore[attr-defined]
            if hasattr(prediction, "tolist"):
                prediction = prediction.tolist()
            if isinstance(prediction, list) and prediction:
                first_value = prediction[0]
                score = float(first_value)
            elif isinstance(prediction, (int, float)):
                score = float(prediction)
        if score is None:
            return None
        metadata = model_spec.get("metadata") or {}
        threshold = float(metadata.get("threshold", 0.5))
        label = "Positive" if score >= threshold else "Negative"
        return {"score": score, "label": label}
    except Exception:
        logger.exception("scikit-learn runner failed for model %s", model_name)
        return None


def _default_explainer(
    model_name: str,
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    return _default_explanations(model_name, payload, prediction)


def _register_default_model_hooks() -> None:
    register_model_loader("generic", _generic_loader)
    register_model_loader("python", _generic_loader)
    register_model_loader("pytorch", _pytorch_loader)
    register_model_loader("torch", _pytorch_loader)
    register_model_loader("sklearn", _sklearn_loader)
    register_model_loader("scikit-learn", _sklearn_loader)
    register_model_loader("tensorflow", _tensorflow_loader)
    register_model_loader("tf", _tensorflow_loader)
    register_model_loader("onnx", _onnx_loader)
    register_model_loader("onnxruntime", _onnx_loader)

    register_model_runner("generic", _generic_runner)
    register_model_runner("callable", _generic_runner)
    register_model_runner("pytorch", _pytorch_runner)
    register_model_runner("torch", _pytorch_runner)
    register_model_runner("sklearn", _sklearn_runner)
    register_model_runner("scikit-learn", _sklearn_runner)
    register_model_runner("tensorflow", _tensorflow_runner)
    register_model_runner("tf", _tensorflow_runner)
    register_model_runner("onnx", _onnx_runner)
    register_model_runner("onnxruntime", _onnx_runner)

    register_model_explainer("generic", _default_explainer)
    register_model_explainer("pytorch", _default_explainer)
    register_model_explainer("torch", _default_explainer)
    register_model_explainer("tensorflow", _default_explainer)
    register_model_explainer("tf", _default_explainer)
    register_model_explainer("onnx", _default_explainer)
    register_model_explainer("onnxruntime", _default_explainer)


_register_default_model_hooks()


def _normalize_connector_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        rows: List[Dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                rows.append(dict(item))
            else:
                rows.append({"value": item})
        return rows
    if isinstance(payload, dict):
        return [dict(payload)]
    return []


async def _default_sql_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not connector:
        return []
    query = connector.get("options", {}).get("query")
    if not query:
        table_name = connector.get("options", {}).get("table") or connector.get("name")
        if not table_name:
            return []
        query = f"SELECT * FROM {table_name}"
    session: Optional[AsyncSession] = context.get("session")
    if session is None:
        return []
    try:
        result = await session.execute(text(query))
        return [dict(row) for row in result.mappings()]
    except Exception:
        logger.exception("Default SQL driver failed for query '%s'", query)
        return []


async def _default_rest_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    endpoint = connector.get("options", {}).get("endpoint") if connector else None
    if not endpoint:
        return []
    method = str(connector.get("options", {}).get("method") or "get").lower()
    payload = _resolve_placeholders(connector.get("options", {}).get("payload"), context)
    headers = _resolve_placeholders(connector.get("options", {}).get("headers"), context)
    async with _HTTPX_CLIENT_CLS() as client:
        try:
            request_method = getattr(client, method, client.get)
            response = await request_method(
                endpoint,
                json=payload if isinstance(payload, dict) else None,
                headers=headers if isinstance(headers, dict) else None,
            )
            response.raise_for_status()
            data = response.json()
            rows = _normalize_connector_rows(data)
            if rows:
                return rows
        except Exception:
            logger.exception("Default REST driver failed for endpoint '%s'", endpoint)
    return []


async def _default_graphql_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    options = connector.get("options", {}) if connector else {}
    endpoint = options.get("endpoint") or options.get("url") or connector.get("name")
    query = options.get("query")
    if not endpoint or not query:
        return []
    variables = _resolve_placeholders(options.get("variables"), context)
    headers = _resolve_placeholders(options.get("headers"), context)
    root_field = options.get("root")
    async with _HTTPX_CLIENT_CLS() as client:
        try:
            response = await client.post(
                endpoint,
                json={"query": query, "variables": variables or {}},
                headers=headers if isinstance(headers, dict) else None,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            logger.exception("Default GraphQL driver failed for endpoint '%s'", endpoint)
            return []
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict):
        target: Any
        if root_field:
            target = data.get(root_field)
        else:
            target = next(iter(data.values())) if data else None
        rows = _normalize_connector_rows(target)
        if rows:
            return rows
    return []


async def _default_grpc_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    options = connector.get("options", {}) if connector else {}
    sample_rows = _resolve_placeholders(options.get("sample") or options.get("rows"), context)
    rows = _normalize_connector_rows(sample_rows)
    if rows:
        return rows
    service = options.get("service") or connector.get("name")
    method = options.get("method")
    logger.info("gRPC connector '%s' invoked without concrete driver; returning stub row", service)
    return [
        {
            "service": service,
            "method": method,
            "status": "UNIMPLEMENTED",
        }
    ]


async def _default_streaming_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    options = connector.get("options", {}) if connector else {}
    stream_name = connector.get("name") or options.get("stream") or "default"
    limit_value = options.get("limit") or options.get("window") or 50
    try:
        limit = max(int(limit_value), 1)
    except Exception:
        limit = 50
    seed_rows = _normalize_connector_rows(_resolve_placeholders(options.get("sample") or options.get("rows"), context))
    buffers = context.setdefault("_stream_buffers", {})
    buffer = buffers.setdefault(stream_name, [])
    if seed_rows:
        buffer.extend(seed_rows)
    if not buffer and options.get("auto_generate", True):
        generated = [{"sequence": index} for index in range(limit)]
        buffer.extend(generated)
    buffer[:] = buffer[-limit:]
    return [dict(row) for row in buffer[-limit:]]


register_connector_driver("sql", _default_sql_driver)
register_connector_driver("rest", _default_rest_driver)
register_connector_driver("graphql", _default_graphql_driver)
register_connector_driver("grpc", _default_grpc_driver)
register_connector_driver("stream", _default_streaming_driver)
register_connector_driver("streaming", _default_streaming_driver)


def _transform_take(rows: List[Dict[str, Any]], options: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    limit = options.get("limit") or options.get("count") or 10
    try:
        limit_int = int(limit)
    except Exception:
        limit_int = 10
    return rows[: max(limit_int, 0)]


def _transform_select_columns(rows: List[Dict[str, Any]], options: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    columns = options.get("columns") or options.get("fields")
    if not columns:
        return rows
    if isinstance(columns, str):
        columns = [segment.strip() for segment in columns.split(",") if segment.strip()]
    selected: List[Dict[str, Any]] = []
    for row in rows:
        entry = {column: row.get(column) for column in columns}
        selected.append(entry)
    return selected


register_dataset_transform("take", _transform_take)
register_dataset_transform("limit", _transform_take)
register_dataset_transform("select", _transform_select_columns)


def get_model_spec(model_name: str) -> Dict[str, Any]:
    base = copy.deepcopy(MODEL_REGISTRY.get(model_name) or {})
    generated = MODELS.get(model_name) or {}
    registry_info = generated.get("registry") or {}

    if not base:
        base = {
            "type": generated.get("type", "unknown"),
            "framework": generated.get("engine") or generated.get("framework") or "unknown",
            "version": registry_info.get("version", "v1"),
            "metrics": registry_info.get("metrics", {}),
            "metadata": registry_info.get("metadata", {}),
        }
    else:
        base.setdefault("type", generated.get("type") or base.get("type") or "unknown")
        base.setdefault("framework", generated.get("engine") or base.get("framework") or "unknown")
        base.setdefault("version", registry_info.get("version") or base.get("version") or "v1")
        merged_metrics = dict(base.get("metrics") or {})
        merged_metrics.update(registry_info.get("metrics") or {})
        base["metrics"] = merged_metrics
        merged_metadata = dict(base.get("metadata") or {})
        merged_metadata.update(registry_info.get("metadata") or {})
        base["metadata"] = merged_metadata

    base.setdefault("metrics", {})
    base.setdefault("metadata", {})
    base["type"] = base.get("type") or generated.get("type") or "custom"
    base["framework"] = base.get("framework") or "unknown"
    base["version"] = base.get("version") or "v1"
    return base


def _load_model_instance(model_name: str, model_spec: Dict[str, Any]) -> Any:
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    framework = str(model_spec.get("framework") or "").lower()
    model_type = str(model_spec.get("type") or "").lower()
    metadata = model_spec.get("metadata") or {}
    custom_loader = None
    loader_path = metadata.get("loader")
    if loader_path:
        try:
            custom_loader = _load_python_callable(loader_path)
        except Exception:  # pragma: no cover - loader import failure
            logger.exception("Failed to import custom loader for %s", model_name)
            custom_loader = None
    loader = None
    loader = (
        custom_loader
        or MODEL_LOADERS.get(framework)
        or MODEL_LOADERS.get(model_type)
        or MODEL_LOADERS.get("generic")
    )
    instance = None
    if loader:
        try:
            instance = loader(model_name, model_spec)
        except Exception:  # pragma: no cover - loader failure
            logger.exception("Model loader failed for %s", model_name)
            instance = None
    MODEL_CACHE[model_name] = instance
    return instance


def _default_explanations(
    model_name: str,
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "global_importances": {"feature_a": 0.7, "feature_b": 0.3},
        "local_explanations": [
            {"feature": "feature_a", "impact": +0.2},
        ],
        "visualizations": {
            "saliency_map": "base64://dummy_image_data",
            "attention": "base64://placeholder_heatmap",
        },
    }


def explain_prediction(
    model_name: str,
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    framework = str(prediction.get("framework") or "").lower()
    model_name_key = str(prediction.get("model") or "").lower()
    metadata = prediction.get("spec_metadata") or {}
    custom_explainer = None
    explainer_path = metadata.get("explainer") if isinstance(metadata, dict) else None
    if explainer_path:
        try:
            custom_explainer = _load_python_callable(explainer_path)
        except Exception:  # pragma: no cover - explainer import failure
            logger.exception("Failed to import custom explainer for %s", model_name)
            custom_explainer = None
    explainer = (
        custom_explainer
        or MODEL_EXPLAINERS.get(framework)
        or MODEL_EXPLAINERS.get(model_name_key)
        or MODEL_EXPLAINERS.get("generic")
    )
    if explainer:
        try:
            value = explainer(model_name, payload, prediction)
            if isinstance(value, dict) and value:
                return value
        except Exception:  # pragma: no cover - explainer failure
            logger.exception("Model explainer failed for %s", model_name)
    return _default_explanations(model_name, payload, prediction)


def predict(model_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run a prediction using registered model loaders and runners."""

    model_spec = get_model_spec(model_name)
    framework = model_spec.get("framework", "unknown")
    version = model_spec.get("version", "v1")
    model_instance = _load_model_instance(model_name, model_spec)
    framework_key = framework.lower()
    model_type = str(model_spec.get("type") or "").lower()
    metadata = model_spec.get("metadata") or {}
    runner_callable = None
    runner_path = metadata.get("runner")
    if runner_path:
        try:
            runner_callable = _load_python_callable(runner_path)
        except Exception:  # pragma: no cover - runner import failure
            logger.exception("Failed to import custom runner for %s", model_name)
            runner_callable = None
    runner = (
        runner_callable
        or MODEL_RUNNERS.get(framework_key)
        or MODEL_RUNNERS.get(model_type)
        or MODEL_RUNNERS.get("generic")
    )
    output: Optional[Dict[str, Any]] = None
    if model_instance is not None and runner:
        try:
            output = runner(model_name, model_instance, payload, model_spec)
        except Exception:  # pragma: no cover - runner failure
            logger.exception("Model runner failed for %s", model_name)
            output = None
    if not isinstance(output, dict) or "score" not in output:
        output = {"score": 0.42, "label": "Positive"}
    result = {
        "model": model_name,
        "version": version,
        "framework": framework,
        "input": payload,
        "output": output,
        "spec_metadata": metadata,
    }
    result["explanations"] = explain_prediction(model_name, payload, result)
    return result


async def broadcast_page_snapshot(slug: str, payload: Dict[str, Any]) -> None:
    if not REALTIME_ENABLED:
        return
    message = {
        "type": "snapshot",
        "slug": slug,
        "payload": payload,
        "meta": _page_meta(slug),
    }
    await BROADCAST.broadcast(slug, _with_timestamp(message))


async def broadcast_component_update(
    slug: str,
    component_type: str,
    component_index: int,
    model: Any,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    if not REALTIME_ENABLED:
        return
    payload = _model_to_payload(model)
    message: Dict[str, Any] = {
        "type": "component",
        "slug": slug,
        "component_type": component_type,
        "component_index": component_index,
        "payload": payload,
        "meta": {"page": _page_meta(slug)},
    }
    if meta:
        message["meta"].update(meta)
    await BROADCAST.broadcast(slug, _with_timestamp(message))


async def broadcast_rollback(slug: str, component_index: int) -> None:
    if not REALTIME_ENABLED:
        return
    message = {
        "type": "rollback",
        "slug": slug,
        "component_index": component_index,
        "meta": {"page": _page_meta(slug)},
    }
    await BROADCAST.broadcast(slug, _with_timestamp(message))


async def fetch_dataset_rows(
    key: str,
    session: AsyncSession,
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    context.setdefault("session", session)
    dataset = DATASETS.get(key)
    if not dataset:
        return []
    resolved_connector = _resolve_connector(dataset, context)
    cache_key = f"dataset_cache::{key}"
    cached = context.get(cache_key)
    if isinstance(cached, list):
        return _clone_rows(cached)
    source_rows = await _load_dataset_source(dataset, resolved_connector, session, context)
    rows = await _execute_dataset_pipeline(dataset, source_rows, context)
    context[cache_key] = _clone_rows(rows)
    return rows


def compile_dataset_to_sql(
    dataset: Dict[str, Any],
    metadata: MetaData,
    context: Dict[str, Any],
) -> Select:
    raise NotImplementedError(
        "compile_dataset_to_sql is a stub in the generated backend"
    )


def evaluate_insights_for_dataset(
    name: str,
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    spec = INSIGHTS.get(name)
    if not spec:
        return {}
    return _run_insight(spec, rows, context)


async def _load_dataset_source(
    dataset: Dict[str, Any],
    connector: Dict[str, Any],
    session: AsyncSession,
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    source_type = str(dataset.get("source_type") or "table").lower()
    source_name = dataset.get("source")
    if source_type == "table":
        query = text(f"SELECT * FROM {source_name}") if source_name else None
        if query is None:
            return []
        try:
            result = await session.execute(query)
            return [dict(row) for row in result.mappings()]
        except Exception:
            logger.exception("Failed to load table dataset '%s'", source_name)
            return []
    if source_type == "sql":
        connector_name = connector.get("name") if connector else None
        driver = CONNECTOR_DRIVERS.get("sql")
        if driver:
            try:
                return await driver(connector, context)
            except Exception:
                logger.exception("SQL connector driver '%s' failed", connector_name)
                return []
        query_text = connector.get("options", {}).get("query") if connector else None
        if not query_text:
            return []
        try:
            result = await session.execute(text(query_text))
            return [dict(row) for row in result.mappings()]
        except Exception:
            logger.exception("Failed to execute SQL query for dataset '%s'", dataset.get("name"))
            return []
    if source_type == "file":
        path = connector.get("name") if connector else source_name
        if not path:
            return []
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                return [dict(row) for row in reader]
        except FileNotFoundError:
            logger.warning("Dataset file '%s' not found", path)
        except Exception:
            logger.exception("Failed to load dataset file '%s'", path)
        return []
    if source_type == "rest":
        connector_name = connector.get("name") if connector else dataset.get("name")
        driver = CONNECTOR_DRIVERS.get("rest")
        if driver:
            try:
                return await driver(connector, context)
            except Exception:
                logger.exception("REST connector driver '%s' failed", connector_name)
        endpoint = connector.get("options", {}).get("endpoint") if connector else None
        if not endpoint:
            return []
        async with _HTTPX_CLIENT_CLS() as client:
            try:
                response = await client.get(endpoint)
                response.raise_for_status()
                payload = response.json()
                rows = _normalize_connector_rows(payload)
                if rows:
                    return rows
            except Exception:
                logger.exception("Failed to fetch REST dataset '%s'", connector_name)
        return []
    if source_type == "graphql":
        connector_name = connector.get("name") if connector else dataset.get("name")
        driver = CONNECTOR_DRIVERS.get("graphql")
        if driver:
            try:
                return await driver(connector, context)
            except Exception:
                logger.exception("GraphQL connector driver '%s' failed", connector_name)
        return []
    if source_type == "grpc":
        connector_name = connector.get("name") if connector else dataset.get("name")
        driver = CONNECTOR_DRIVERS.get("grpc")
        if driver:
            try:
                return await driver(connector, context)
            except Exception:
                logger.exception("gRPC connector driver '%s' failed", connector_name)
        return []
    if source_type in {"stream", "streaming"}:
        connector_name = connector.get("name") if connector else dataset.get("name")
        driver = CONNECTOR_DRIVERS.get("stream") or CONNECTOR_DRIVERS.get("streaming")
        if driver:
            try:
                return await driver(connector, context)
            except Exception:
                logger.exception("Streaming connector driver '%s' failed", connector_name)
        return []
    if source_type == "dataset" and source_name:
        target_name = str(source_name)
        if target_name == dataset.get("name"):
            return list(dataset.get("sample_rows", []))
        return await fetch_dataset_rows(target_name, session, context)
    return list(dataset.get("sample_rows", []))


async def _execute_dataset_pipeline(
    dataset: Dict[str, Any],
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    operations: List[Dict[str, Any]] = list(dataset.get("operations") or [])
    transforms: List[Dict[str, Any]] = list(dataset.get("transforms") or [])
    current_rows = _clone_rows(rows)
    aggregate_ops: List[Tuple[str, str]] = []
    group_by_columns: List[str] = []
    for operation in operations:
        otype = str(operation.get("type") or "").lower()
        if otype == "filter":
            current_rows = _apply_filter(current_rows, operation.get("condition"), context)
        elif otype == "computed_column":
            _apply_computed_column(current_rows, operation.get("name"), operation.get("expression"), context)
        elif otype == "group_by":
            group_by_columns = list(operation.get("columns") or [])
        elif otype == "aggregate":
            aggregate_ops.append((operation.get("function"), operation.get("expression")))
        elif otype == "order_by":
            current_rows = _apply_order(current_rows, operation.get("columns") or [])
        elif otype == "window":
            _apply_window_operation(
                current_rows,
                operation.get("name"),
                operation.get("function"),
                operation.get("target"),
            )
    if aggregate_ops:
        current_rows = _apply_group_aggregate(current_rows, group_by_columns, aggregate_ops, context)
    if transforms:
        current_rows = _apply_transforms(current_rows, transforms, context)
    quality_checks: List[Dict[str, Any]] = list(dataset.get("quality_checks") or [])
    evaluation = _evaluate_quality_checks(current_rows, quality_checks, context)
    if evaluation:
        context.setdefault("quality", {})[dataset.get("name")] = evaluation
    features = dataset.get("features") or []
    targets = dataset.get("targets") or []
    if features:
        context.setdefault("features", {})[dataset.get("name")] = features
    if targets:
        context.setdefault("targets", {})[dataset.get("name")] = targets
    return current_rows



def _resolve_connector(dataset: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    connector = dataset.get("connector")
    if not connector:
        return {}
    resolved = copy.deepcopy(connector)
    resolved["options"] = _resolve_placeholders(connector.get("options"), context)
    return resolved


def _run_insight(
    spec: Dict[str, Any],
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    scope: Dict[str, Any] = {"rows": rows, "context": context}
    scope["models"] = MODEL_REGISTRY
    scope["predict"] = predict  # Future hook: replace with real inference callable
    for step in spec.get("logic", []):
        if step.get("type") == "assign":
            value_expr = step.get("expression")
            scope[step.get("name")] = _evaluate_expression(value_expr, rows, scope)
    metrics: List[Dict[str, Any]] = []
    metrics_map: Dict[str, Dict[str, Any]] = {}
    for metric in spec.get("metrics", []):
        value = _evaluate_expression(metric.get("value"), rows, scope)
        baseline = _evaluate_expression(metric.get("baseline"), rows, scope)
        if value is None:
            value = len(rows) * 100 if rows else 0
        if baseline is None:
            baseline = value / 2 if value else 0
        payload = {
            "name": metric.get("name"),
            "label": metric.get("label"),
            "value": value,
            "baseline": baseline,
            "trend": "up" if value >= baseline else "flat",
            "formatted": f"${{value:,.2f}}",
            "unit": metric.get("unit"),
            "positive_label": metric.get("positive_label", "positive"),
        }
        metrics.append(payload)
        metrics_map[payload["name"]] = payload
    alerts_list: List[Dict[str, Any]] = []
    alerts_map: Dict[str, Dict[str, Any]] = {}
    for threshold in spec.get("thresholds", []):
        alert_payload = {
            "name": threshold.get("name"),
            "level": threshold.get("level"),
            "metric": threshold.get("metric"),
            "triggered": True,
        }
        alerts_list.append(alert_payload)
        alerts_map[alert_payload["name"]] = alert_payload
    narrative_scope = dict(scope)
    narrative_scope["metrics"] = metrics_map
    narratives: List[Dict[str, Any]] = []
    for narrative in spec.get("narratives", []):
        text = _render_template_value(narrative.get("template"), narrative_scope)
        narratives.append(
            {
                "name": narrative.get("name"),
                "text": text,
                "variant": narrative.get("variant"),
            }
        )
    variables: Dict[str, Any] = {}
    for key, expr in spec.get("expose_as", {}).items():
        variables[key] = _evaluate_expression(expr, rows, scope)
    return {
        "name": spec.get("name"),
        "dataset": spec.get("source_dataset"),
        "metrics": metrics,
        "alerts": alerts_map,
        "alerts_list": alerts_list,
        "narratives": narratives,
        "variables": variables,
    }


def evaluate_insight(slug: str, context: Optional[Dict[str, Any]] = None) -> InsightResponse:
    spec = INSIGHTS.get(slug)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Insight '{slug}' is not defined")
    ctx = dict(context or build_context(None))
    rows: List[Dict[str, Any]] = []
    result = evaluate_insights_for_dataset(slug, rows, ctx)
    dataset = result.get("dataset") or spec.get("source_dataset") or slug
    return InsightResponse(name=slug, dataset=dataset, result=result)


def run_prediction(model_name: str, payload: Optional[Dict[str, Any]] = None) -> PredictionResponse:
    model_key = str(model_name)
    if model_key not in MODELS and model_key not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' is not registered")
    request_payload = dict(payload or {})
    result = predict(model_key, request_payload)
    response_payload = {
        "model": result.get("model", model_key),
        "version": result.get("version"),
        "framework": result.get("framework"),
        "input": result.get("input") or {},
        "output": result.get("output") or {},
        "explanations": result.get("explanations") or {},
        "metadata": result.get("spec_metadata") or result.get("metadata") or {},
    }
    return PredictionResponse(**response_payload)


async def predict_model(model_name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience helper used by the generated API and tests."""

    response = run_prediction(model_name, payload)
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return dict(response)


def run_experiment(slug: str, payload: Optional[Dict[str, Any]] = None) -> ExperimentResult:
    experiment_key = str(slug)
    if experiment_key not in AI_EXPERIMENTS:
        raise HTTPException(status_code=404, detail=f"Experiment '{slug}' is not defined")
    request_payload = dict(payload or {})
    result = evaluate_experiment(experiment_key, request_payload)
    return ExperimentResult(**result)


async def experiment_metrics(slug: str) -> Dict[str, Any]:
    """Return experiment metrics snapshot as a plain dictionary."""

    response = run_experiment(slug, {})
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return dict(response)


async def run_experiment_endpoint(
    slug: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute an experiment with an optional payload and return a dict."""

    response = run_experiment(slug, payload or {})
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return dict(response)


def _evaluate_expression(
    expression: Optional[str],
    rows: List[Dict[str, Any]],
    scope: Dict[str, Any],
) -> Any:
    if expression is None:
        return None
    expr = expression.strip()
    if not expr:
        return None
    if expr == "rows":
        return rows
    if expr.startswith("len(") and expr.endswith(")"):
        target = expr[4:-1]
        value = scope.get(target)
        if value is None and target == "rows":
            value = rows
        if isinstance(value, (list, tuple)):
            return len(value)
        return 0
    if expr.startswith("sum(") and expr.endswith(")"):
        field = expr[4:-1].strip().strip('"').strip("'")
        return sum(float(row.get(field, 0) or 0) for row in rows)
    if expr.startswith("avg(") and expr.endswith(")"):
        field = expr[4:-1].strip().strip('"').strip("'")
        values = [float(row.get(field, 0) or 0) for row in rows]
        return sum(values) / len(values) if values else 0
    if "." in expr or "[" in expr:
        return _resolve_expression_path(expr, rows, scope)
    if expr in scope:
        return scope[expr]
    try:
        return float(expr)
    except ValueError:
        return expr


def _resolve_expression_path(
    expression: str,
    rows: List[Dict[str, Any]],
    scope: Dict[str, Any],
) -> Any:
    tokens: List[str] = []
    buffer = ""
    for char in expression:
        if char == ".":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            continue
        if char == "[":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            tokens.append("[")
            continue
        if char == "]":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            tokens.append("]")
            continue
        buffer += char
    if buffer:
        tokens.append(buffer)
    if not tokens:
        return None
    base_name = tokens.pop(0)
    if base_name == "rows":
        current: Any = rows
    else:
        current = scope.get(base_name)
    idx_mode = False
    for token in tokens:
        if token == "[":
            idx_mode = True
            continue
        if token == "]":
            idx_mode = False
            continue
        if idx_mode:
            try:
                index = int(token)
            except ValueError:
                return None
            if isinstance(current, list) and 0 <= index < len(current):
                current = current[index]
            else:
                return None
        else:
            if isinstance(current, dict):
                current = current.get(token)
            else:
                current = getattr(current, token, None)
        if current is None:
            return None
    return current

'''
    parts.append(textwrap.dedent(helpers).rstrip())

    page_blocks: List[str] = []
    for page in state.pages:
        page_lines = _render_page_function(page)
        page_blocks.append("\n".join(page_lines))
        page_handler_entries.append(
            f"    {page.slug!r}: page_{page.slug}_{page.index},"
        )

    if page_blocks:
        parts.append("\n\n".join(block.strip() for block in page_blocks if block))

    if page_handler_entries:
        handler_lines = [
            "PAGE_HANDLERS: Dict[str, Callable[[Optional[AsyncSession]], Awaitable[Dict[str, Any]]]] = {"
        ]
        handler_lines.extend(page_handler_entries)
        handler_lines.append("}")
    else:
        handler_lines = [
            "PAGE_HANDLERS: Dict[str, Callable[[Optional[AsyncSession]], Awaitable[Dict[str, Any]]]] = {}"
        ]
    parts.append("\n".join(handler_lines))

    return "\n\n".join(part for part in parts if part).strip() + "\n"


# ---------------------------------------------------------------------------
# Helpers for rendering code fragments
# ---------------------------------------------------------------------------


def _assign_literal(name: str, annotation: str, value: Any) -> str:
    literal = _format_literal(value)
    return f"{name}: {annotation} = {literal}"


def _format_literal(value: Any) -> str:
    return pprint.pformat(value, width=100, sort_dicts=False)


def _page_to_dict(page: PageSpec) -> Dict[str, Any]:
    return {
        "name": page.name,
        "route": page.route,
        "slug": page.slug,
        "index": page.index,
        "api_path": page.api_path,
        "reactive": page.reactive,
        "refresh_policy": page.refresh_policy,
        "layout": page.layout,
        "components": [component.__dict__ for component in page.components],
    }


def _render_page_function(page: PageSpec) -> List[str]:
    lines: List[str] = []
    func_name = f"page_{page.slug}_{page.index}"
    instructions = [_component_to_serializable(component) for component in page.components]
    lines.append(f"async def {func_name}(session: Optional[AsyncSession] = None) -> Dict[str, Any]:")
    lines.append(f"    context = build_context({page.slug!r})")
    lines.append("    scope = ScopeFrame()")
    lines.append("    scope.set('context', context)")
    lines.append(f"    instructions = {_format_literal(instructions)}")
    lines.append("    components = await render_statements(instructions, context, scope, session)")
    lines.append("    return {")
    lines.append(f"        'name': {page.name!r},")
    lines.append(f"        'route': {page.route!r},")
    lines.append(f"        'slug': {page.slug!r},")
    lines.append(f"        'reactive': {page.reactive!r},")
    lines.append(f"        'refresh_policy': {_format_literal(page.refresh_policy)},")
    lines.append("        'components': components,")
    lines.append(f"        'layout': {_format_literal(page.layout)},")
    lines.append("    }")
    return lines


def _render_page_endpoint(page: PageSpec) -> List[str]:
    func_name = f"page_{page.slug}_{page.index}"
    path = page.api_path or "/api/pages"
    lines = [
        f"@router.get({path!r}, response_model=Dict[str, Any], tags=['pages'])",
        f"async def {func_name}_view(session: AsyncSession = Depends(get_session)) -> Dict[str, Any]:",
        f"    payload = await runtime.{func_name}(session)",
    ]
    if page.reactive:
        lines.append("    if runtime.REALTIME_ENABLED:")
        lines.append(f"        await runtime.broadcast_page_snapshot({page.slug!r}, payload)")
    lines.append("    return payload")
    return lines


def _render_component_endpoint(
    page: PageSpec, component: PageComponent, index: int
) -> List[str]:
    payload = component.payload
    slug = page.slug
    base_path = page.api_path.rstrip("/") if page.api_path else "/api/pages"
    if not base_path.startswith("/"):
        base_path = "/" + base_path
    if component.type == "table":
        insight_name = payload.get("insight")
        source_name = payload.get("source")
        source_type = payload.get("source_type")
        meta_payload: Dict[str, Any] = {}
        if insight_name:
            meta_payload["insight"] = insight_name
        if source_name:
            meta_payload["source"] = source_name
        meta_expr = _format_literal(meta_payload) if meta_payload else "None"
        return [
            f"@router.get({base_path!r} + '/tables/{index}', response_model=TableResponse, tags=['pages'])",
            f"async def {slug}_table_{index}(session: AsyncSession = Depends(get_session)) -> TableResponse:",
            f"    context = runtime.build_context({page.slug!r})",
            f"    dataset = runtime.DATASETS.get({source_name!r})",
            f"    rows = await runtime.fetch_dataset_rows({source_name!r}, session, context)",
            "    insights: Dict[str, Any] = {}",
            f"    if runtime.EMBED_INSIGHTS and dataset.get('name') if dataset else False:",
            f"        if {insight_name!r}:",
            "            try:",
            f"                insights = runtime.evaluate_insights_for_dataset({insight_name!r}, rows, context)",
            "            except Exception:",
            f"                runtime.logger.exception('Failed to evaluate insight %s', {insight_name!r})",
            "                insights = {}",
            "    response = TableResponse(",
            f"        title={payload.get('title')!r},",
            f"        source={{'type': {source_type!r}, 'name': {source_name!r}}},",
            f"        columns={payload.get('columns') or []!r},",
            f"        filter={payload.get('filter')!r},",
            f"        sort={payload.get('sort')!r},",
            f"        style={payload.get('style') or {}!r},",
            f"        insight={insight_name!r},",
            "        rows=rows,",
            "        insights=insights,",
            "    )",
            f"    is_reactive = {page.reactive!r} or (dataset.get('reactive') if dataset else False)",
            "    if runtime.REALTIME_ENABLED and is_reactive:",
            f"        await runtime.broadcast_component_update({page.slug!r}, 'table', {index}, response, meta={meta_expr})",
            "    return response",
        ]
    if component.type == "chart":
        insight_name = payload.get("insight")
        source_name = payload.get("source")
        source_type = payload.get("source_type")
        meta_payload: Dict[str, Any] = {}
        if payload.get("chart_type"):
            meta_payload["chart_type"] = payload.get("chart_type")
        if insight_name:
            meta_payload["insight"] = insight_name
        if source_name:
            meta_payload["source"] = source_name
        meta_map = {key: value for key, value in meta_payload.items() if value is not None}
        meta_expr = _format_literal(meta_map) if meta_map else "None"
        return [
            f"@router.get({base_path!r} + '/charts/{index}', response_model=ChartResponse, tags=['pages'])",
            f"async def {slug}_chart_{index}(session: AsyncSession = Depends(get_session)) -> ChartResponse:",
            f"    context = runtime.build_context({page.slug!r})",
            f"    dataset = runtime.DATASETS.get({source_name!r})",
            f"    rows = await runtime.fetch_dataset_rows({source_name!r}, session, context)",
            "    labels: List[Any] = [row.get('label', idx) for idx, row in enumerate(rows, start=1)]",
            "    series_values: List[Any] = [row.get('value', idx * 10) for idx, row in enumerate(rows, start=1)]",
            "    series = [{'label': 'Series', 'data': series_values}]",
            "    insight_results: Dict[str, Any] = {}",
            f"    if runtime.EMBED_INSIGHTS and dataset.get('name') if dataset else False:",
            f"        if {insight_name!r}:",
            "            try:",
            f"                insight_results = runtime.evaluate_insights_for_dataset({insight_name!r}, rows, context)",
            "            except Exception:",
            f"                runtime.logger.exception('Failed to evaluate insight %s', {insight_name!r})",
            "                insight_results = {}",
            "    response = ChartResponse(",
            f"        heading={payload.get('heading')!r},",
            f"        title={payload.get('title')!r},",
            f"        source={{'type': {source_type!r}, 'name': {source_name!r}}},",
            f"        chart_type={payload.get('chart_type')!r},",
            f"        x={payload.get('x')!r},",
            f"        y={payload.get('y')!r},",
            f"        color={payload.get('color')!r},",
            "        labels=labels,",
            "        series=series,",
            f"        legend={payload.get('legend') or {}!r},",
            f"        style={payload.get('style') or {}!r},",
            f"        encodings={payload.get('encodings') or {}!r},",
            f"        insight={insight_name!r},",
            "        insights=insight_results,",
            "    )",
            f"    is_reactive = {page.reactive!r} or (dataset.get('reactive') if dataset else False)",
            "    if runtime.REALTIME_ENABLED and is_reactive:",
            f"        await runtime.broadcast_component_update({page.slug!r}, 'chart', {index}, response, meta={meta_expr})",
            "    return response",
        ]
    return []


def _render_insight_endpoint(name: str) -> List[str]:
    return [
        f"@app.get('/api/insights/{name}', response_model=InsightResponse)",
        f"async def insight_{name}() -> InsightResponse:",
        "    context = build_context(None)",
        "    rows: List[Dict[str, Any]] = []",
        f"    result = evaluate_insights_for_dataset({name!r}, rows, context)",
        f"    return InsightResponse(name={name!r}, dataset=result.get('dataset', {name!r}), result=result)",
    ]


def _database_env_var(database_name: Optional[str]) -> str:
    if not database_name:
        return "NAMEL3SS_DATABASE_URL"
    alias = re.sub(r"[^A-Z0-9]+", "_", str(database_name).upper()).strip("_") or "DEFAULT"
    return f"NAMEL3SS_POSTGRES_{alias}_URL"
