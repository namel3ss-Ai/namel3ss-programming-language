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
from ..schemas import ChartResponse, TableResponse

router = APIRouter()

@router.get('/api/pages/root', response_model=Dict[str, Any], tags=['pages'])
async def page_home_0_view(session: AsyncSession = Depends(get_session)) -> Dict[str, Any]:
    payload = await runtime.page_home_0(session)
    return payload

@router.get('/api/pages/root' + '/charts/1', response_model=ChartResponse, tags=['pages'])
async def home_chart_1(session: AsyncSession = Depends(get_session)) -> ChartResponse:
    context = runtime.build_context('home')
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
        heading='Revenue Growth',
        title=None,
        source={'type': 'dataset', 'name': 'monthly_sales'},
        chart_type='line',
        x='month',
        y='total_revenue',
        color='var(--primary)',
        labels=labels,
        series=series,
        legend={},
        style={'title size': 'large'},
        encodings={},
        insight=None,
        insights=insight_results,
    )
    is_reactive = False or (dataset.get('reactive') if dataset else False)
    if runtime.REALTIME_ENABLED and is_reactive:
        await runtime.broadcast_component_update('home', 'chart', 1, response, meta={'chart_type': 'line', 'source': 'monthly_sales'})
    return response

@router.get('/api/pages/root' + '/tables/2', response_model=TableResponse, tags=['pages'])
async def home_table_2(session: AsyncSession = Depends(get_session)) -> TableResponse:
    context = runtime.build_context('home')
    dataset = runtime.DATASETS.get('orders')
    rows = await runtime.fetch_dataset_rows('orders', session, context)
    insights: Dict[str, Any] = {}
    if runtime.EMBED_INSIGHTS and dataset.get('name') if dataset else False:
        if None:
            try:
                insights = runtime.evaluate_insights_for_dataset(None, rows, context)
            except Exception:
                runtime.logger.exception('Failed to evaluate insight %s', None)
                insights = {}
    response = TableResponse(
        title='Recent Orders',
        source={'type': 'table', 'name': 'orders'},
        columns=['id', 'customer_name', 'total', 'status'],
        filter='status == "Pending"',
        sort=None,
        style={},
        insight=None,
        rows=rows,
        insights=insights,
    )
    is_reactive = False or (dataset.get('reactive') if dataset else False)
    if runtime.REALTIME_ENABLED and is_reactive:
        await runtime.broadcast_component_update('home', 'table', 2, response, meta={'source': 'orders'})
    return response

@router.get('/api/pages/feedback', response_model=Dict[str, Any], tags=['pages'])
async def page_feedback_1_view(session: AsyncSession = Depends(get_session)) -> Dict[str, Any]:
    payload = await runtime.page_feedback_1(session)
    return payload

@router.post('/api/pages/feedback' + '/forms/0', response_model=Dict[str, Any], tags=['pages'])
async def feedback_form_0(payload: Dict[str, Any], session: AsyncSession = Depends(get_session)) -> Dict[str, Any]:
    try:
        return await runtime.submit_form('feedback', 0, payload, session=session)
    except KeyError:
        raise HTTPException(status_code=404, detail='Form not found')
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get('/api/pages/admin', response_model=Dict[str, Any], tags=['pages'])
async def page_admin_2_view(session: AsyncSession = Depends(get_session)) -> Dict[str, Any]:
    payload = await runtime.page_admin_2(session)
    return payload

@router.get('/api/pages/admin' + '/tables/0', response_model=TableResponse, tags=['pages'])
async def admin_table_0(session: AsyncSession = Depends(get_session)) -> TableResponse:
    context = runtime.build_context('admin')
    dataset = runtime.DATASETS.get('orders')
    rows = await runtime.fetch_dataset_rows('orders', session, context)
    insights: Dict[str, Any] = {}
    if runtime.EMBED_INSIGHTS and dataset.get('name') if dataset else False:
        if None:
            try:
                insights = runtime.evaluate_insights_for_dataset(None, rows, context)
            except Exception:
                runtime.logger.exception('Failed to evaluate insight %s', None)
                insights = {}
    response = TableResponse(
        title='Orders',
        source={'type': 'table', 'name': 'orders'},
        columns=['id', 'customer_name', 'total', 'status'],
        filter=None,
        sort=None,
        style={},
        insight=None,
        rows=rows,
        insights=insights,
    )
    is_reactive = False or (dataset.get('reactive') if dataset else False)
    if runtime.REALTIME_ENABLED and is_reactive:
        await runtime.broadcast_component_update('admin', 'table', 0, response, meta={'source': 'orders'})
    return response

@router.post('/api/pages/admin' + '/actions/1', response_model=Dict[str, Any], tags=['pages'])
async def admin_action_1(payload: Optional[Dict[str, Any]] = None, session: AsyncSession = Depends(get_session)) -> Dict[str, Any]:
    try:
        data = payload or {}
        return await runtime.trigger_action('admin', 1, data, session=session)
    except KeyError:
        raise HTTPException(status_code=404, detail='Action not found')
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
