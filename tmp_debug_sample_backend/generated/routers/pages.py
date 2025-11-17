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
async def page_dashboard_0_view(session: AsyncSession = Depends(get_session)) -> Dict[str, Any]:
    payload = await runtime.page_dashboard_0(session)
    return payload

@router.get('/api/pages/root' + '/tables/1', response_model=TableResponse, tags=['pages'])
async def dashboard_table_1(session: AsyncSession = Depends(get_session)) -> TableResponse:
    context = runtime.build_context('dashboard')
    dataset = runtime.DATASETS.get('users')
    rows = await runtime.fetch_dataset_rows('users', session, context)
    insights: Dict[str, Any] = {}
    if runtime.EMBED_INSIGHTS and dataset.get('name') if dataset else False:
        if None:
            try:
                insights = runtime.evaluate_insights_for_dataset(None, rows, context)
            except Exception:
                runtime.logger.exception('Failed to evaluate insight %s', None)
                insights = {}
    response = TableResponse(
        title='Users',
        source={'type': 'table', 'name': 'users'},
        columns=[],
        filter=None,
        sort=None,
        style={},
        insight=None,
        rows=rows,
        insights=insights,
    )
    is_reactive = False or (dataset.get('reactive') if dataset else False)
    if runtime.REALTIME_ENABLED and is_reactive:
        await runtime.broadcast_component_update('dashboard', 'table', 1, response, meta={'source': 'users'})
    return response

@router.get('/api/pages/root' + '/charts/2', response_model=ChartResponse, tags=['pages'])
async def dashboard_chart_2(session: AsyncSession = Depends(get_session)) -> ChartResponse:
    context = runtime.build_context('dashboard')
    dataset = runtime.DATASETS.get('monthly_sales')
    rows = await runtime.fetch_dataset_rows('monthly_sales', session, context)
    labels: List[Any] = [row.get('label', idx) for idx, row in enumerate(rows, start=1)]
    series_values: List[Any] = [row.get('value', idx * 10) for idx, row in enumerate(rows, start=1)]
    series = [{'label': 'Series', 'data': series_values}]
    insight_results: Dict[str, Any] = {}
    if runtime.EMBED_INSIGHTS and dataset.get('name') if dataset else False:
        if None:
            try:
                insight_results = runtime.evaluate_insights_for_dataset(None, rows, context)
            except Exception:
                runtime.logger.exception('Failed to evaluate insight %s', None)
                insight_results = {}
    response = ChartResponse(
        heading='Sales',
        title=None,
        source={'type': 'dataset', 'name': 'monthly_sales'},
        chart_type='bar',
        x=None,
        y=None,
        color=None,
        labels=labels,
        series=series,
        legend={},
        style={},
        encodings={},
        insight=None,
        insights=insight_results,
    )
    is_reactive = False or (dataset.get('reactive') if dataset else False)
    if runtime.REALTIME_ENABLED and is_reactive:
        await runtime.broadcast_component_update('dashboard', 'chart', 2, response, meta={'chart_type': 'bar', 'source': 'monthly_sales'})
    return response

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
    try:
        context = await runtime.resolve_websocket_context(websocket)
    except WebSocketDisconnect:  # pragma: no cover - propagated disconnect
        raise
    except Exception:
        runtime.logger.exception("Realtime authentication failure for page %s", slug)
        await websocket.close(code=4403)
        return
    connection_id = await runtime.BROADCAST.connect(slug, websocket, context=context)
    try:
        page_spec = runtime.PAGE_SPEC_BY_SLUG.get(slug, {})
        handler = runtime.PAGE_HANDLERS.get(slug)
        if page_spec.get("reactive") and handler:
            try:
                payload = await handler(None)
                await websocket.send_json(runtime._with_timestamp({
                    "type": "snapshot",
                    "slug": slug,
                    "payload": payload,
                    "meta": {"page": runtime._page_meta(slug), "source": "hydration"},
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
                "connection_id": connection_id,
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
