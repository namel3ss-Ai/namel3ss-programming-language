"""Render the generated FastAPI routers."""

from __future__ import annotations

import textwrap
from typing import Any, Dict, Iterable, List, Optional

from ..state import BackendState, PageComponent, PageSpec, _component_to_serializable
from .utils import _format_literal

__all__ = [
    "_render_routers_package",
    "_render_insights_router_module",
    "_render_models_router_module",
    "_render_experiments_router_module",
    "_render_pages_router_module",
    "_render_page_endpoint",
    "_render_component_endpoint",
    "_render_insight_endpoint",
]


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
from ..helpers import router_dependencies
from ..schemas import InsightResponse

router = APIRouter(tags=["insights"], dependencies=router_dependencies())


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

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ..runtime import (
    PredictionResponse,
    call_llm_connector,
    explain_prediction,
    get_model_spec,
    predict,
    run_chain,
)
from ..helpers import router_dependencies

router = APIRouter(tags=["models"], dependencies=router_dependencies())


@router.post("/api/models/{model_name}/predict", response_model=PredictionResponse)
async def predict_model(model_name: str, payload: Dict[str, Any]) -> PredictionResponse:
    try:
        return predict(model_name, payload)
    except KeyError as exc:  # pragma: no cover - runtime failure
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/models/{model_name}/explain")
async def explain_model_prediction(model_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return explain_prediction(model_name, payload)


@router.get("/api/models/{model_name}/spec")
async def get_model_specification(model_name: str) -> Dict[str, Any]:
    return get_model_spec(model_name)


@router.post("/api/chains/{chain_name}")
async def run_registered_chain(chain_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return run_chain(chain_name, payload)


@router.post("/api/llm/{connector}")
async def run_llm_connector(connector: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return call_llm_connector(connector, payload)


__all__ = ["router"]
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_experiments_router_module() -> str:
    template = '''
"""Generated FastAPI router for experiment endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from ..runtime import ExperimentResult, evaluate_experiment, run_experiment
from ..helpers import router_dependencies

router = APIRouter(tags=["experiments"], dependencies=router_dependencies())


@router.get("/api/experiments/{slug}", response_model=ExperimentResult)
async def get_experiment(slug: str) -> ExperimentResult:
    return evaluate_experiment(slug, payload=None)


@router.post("/api/experiments/{slug}", response_model=ExperimentResult)
async def evaluate_experiment_endpoint(slug: str, payload: Dict[str, Any]) -> ExperimentResult:
    return evaluate_experiment(slug, payload)


@router.post("/api/experiments/{slug}/run", response_model=ExperimentResult)
async def run_experiment_endpoint(slug: str, payload: Dict[str, Any]) -> ExperimentResult:
    return run_experiment(slug, payload)


__all__ = ["router"]
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_pages_router_module(state: BackendState) -> str:
    header = '''
"""Generated FastAPI router for page and component endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
try:
    from fastapi import WebSocket, WebSocketDisconnect
except ImportError:  # pragma: no cover - FastAPI <0.65 fallback
    from fastapi.websockets import WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_session
from .. import runtime
from ..helpers import router_dependencies
from ..schemas import ChartResponse, TableResponse

router = APIRouter(dependencies=router_dependencies())
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
    if component.type == "form":
        return [
            f"@router.post({base_path!r} + '/forms/{index}', response_model=Dict[str, Any], tags=['pages'])",
            f"async def {slug}_form_{index}(payload: Dict[str, Any], session: AsyncSession = Depends(get_session)) -> Dict[str, Any]:",
            "    try:",
            f"        return await runtime.submit_form({page.slug!r}, {index}, payload, session=session)",
            "    except KeyError:",
            "        raise HTTPException(status_code=404, detail='Form not found')",
            "    except (IndexError, ValueError) as exc:",
            "        raise HTTPException(status_code=400, detail=str(exc)) from exc",
        ]
    if component.type == "action":
        return [
            f"@router.post({base_path!r} + '/actions/{index}', response_model=Dict[str, Any], tags=['pages'])",
            f"async def {slug}_action_{index}(payload: Optional[Dict[str, Any]] = None, session: AsyncSession = Depends(get_session)) -> Dict[str, Any]:",
            "    try:",
            f"        data = payload or {{}}",
            f"        return await runtime.trigger_action({page.slug!r}, {index}, data, session=session)",
            "    except KeyError:",
            "        raise HTTPException(status_code=404, detail='Action not found')",
            "    except (IndexError, ValueError) as exc:",
            "        raise HTTPException(status_code=400, detail=str(exc)) from exc",
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
