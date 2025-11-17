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

@router.get('/api/pages/root', response_model=Dict[str, Any], tags=['pages'])
async def page_home_0_view(session: AsyncSession = Depends(get_session)) -> Dict[str, Any]:
    payload = await runtime.page_home_0(session)
    return payload

@router.get('/api/pages/root' + '/tables/1', response_model=TableResponse, tags=['pages'])
async def home_table_1(session: AsyncSession = Depends(get_session)) -> TableResponse:
    context = runtime.build_context('home')
    dataset = runtime.DATASETS.get('active_users')
    rows = await runtime.fetch_dataset_rows('active_users', session, context)
    insights: Dict[str, Any] = {}
    if runtime.EMBED_INSIGHTS and dataset.get('name') if dataset else False:
        if None:
            try:
                insights = runtime.evaluate_insights_for_dataset(None, rows, context)
            except Exception as exc:
                runtime.logger.exception('Failed to evaluate insight %s', None)
                runtime._record_runtime_error(context, code='insight_evaluation_failed', message="Insight 'None' failed during evaluation.", scope=None, source='insight', detail=str(exc))
                insights = {}
    component_errors: List[Dict[str, Any]] = []
    if 'active_users':
        component_errors.extend(runtime._collect_runtime_errors(context, 'active_users'))
    if None:
        component_errors.extend(runtime._collect_runtime_errors(context, None))
    response = TableResponse(
        title='Active Users',
        source={'type': 'dataset', 'name': 'active_users'},
        columns=[],
        filter=None,
        sort=None,
        style={},
        insight=None,
        rows=rows,
        insights=insights,
        errors=component_errors,
    )
    is_reactive = False or (dataset.get('reactive') if dataset else False)
    if runtime.REALTIME_ENABLED and is_reactive:
        await runtime.broadcast_component_update('home', 'table', 1, response, meta={'source': 'active_users'})
    return response

@router.get('/api/pages/admin', response_model=Dict[str, Any], tags=['pages'])
async def page_admin_1_view(session: AsyncSession = Depends(get_session)) -> Dict[str, Any]:
    payload = await runtime.page_admin_1(session)
    return payload

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

@router.get("/api/streams/pages/{slug}", response_class=StreamingResponse, tags=["streams"])
async def stream_page_events(slug: str, heartbeat: Optional[int] = None) -> StreamingResponse:
    return await runtime.stream_page(slug, heartbeat=heartbeat)


@router.get("/api/streams/datasets/{dataset}", response_class=StreamingResponse, tags=["streams"])
async def stream_dataset_events(dataset: str, heartbeat: Optional[int] = None) -> StreamingResponse:
    return await runtime.stream_dataset(dataset, heartbeat=heartbeat)


@router.get("/api/streams/topics/{topic:path}", response_class=StreamingResponse, tags=["streams"])
async def stream_topic_events(topic: str, heartbeat: Optional[int] = None) -> StreamingResponse:
    return await runtime.stream_topic(topic, heartbeat=heartbeat)

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

__all__ = ['router']
