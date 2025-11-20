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
    "_render_frames_router_module",
    "_render_training_router_module",
    "_render_pages_router_module",
    "_render_crud_router_module",
    "_render_observability_router_module",
    "_render_page_endpoint",
    "_render_component_endpoint",
    "_render_insight_endpoint",
]


def _render_routers_package() -> str:
    template = '''
"""Aggregated FastAPI routers for Namel3ss (auto-generated)."""

from __future__ import annotations

from . import crud, experiments, frames, insights, models, observability, pages, training

insights_router = insights.router
models_router = models.router
experiments_router = experiments.router
frames_router = frames.router
training_router = training.router
pages_router = pages.router
crud_router = crud.router
observability_router = observability.router

GENERATED_ROUTERS = (
    insights_router,
    models_router,
    experiments_router,
    frames_router,
    training_router,
    pages_router,
    crud_router,
    observability_router,
)

__all__ = [
    "insights_router",
    "models_router",
    "experiments_router",
    "frames_router",
    "training_router",
    "pages_router",
    "crud_router",
    "observability_router",
    "GENERATED_ROUTERS",
]
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_insights_router_module() -> str:
    template = '''
"""Generated FastAPI router for insight endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ..helpers import rate_limit_dependency, router_dependencies
from ..runtime import evaluate_insight
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

from ..helpers import rate_limit_dependency, router_dependencies
from ..runtime import (
    PredictionResponse,
    call_llm_connector,
    call_python_model,
    explain_prediction,
    get_model_spec,
    predict,
    run_chain,
)

router = APIRouter(tags=["models"], dependencies=router_dependencies())


@router.post(
    "/api/models/{model_name}/predict",
    response_model=PredictionResponse,
    dependencies=[rate_limit_dependency("ai")],
)
async def predict_model(model_name: str, payload: Dict[str, Any]) -> PredictionResponse:
    try:
        return predict(model_name, payload)
    except KeyError as exc:  # pragma: no cover - runtime failure
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/models/{model_name}/explain",
    dependencies=[rate_limit_dependency("ai")],
)
async def explain_model_prediction(model_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return explain_prediction(model_name, payload)


@router.get("/api/models/{model_name}/spec")
async def get_model_specification(model_name: str) -> Dict[str, Any]:
    return get_model_spec(model_name)


@router.post(
    "/api/chains/{chain_name}",
    dependencies=[rate_limit_dependency("ai")],
)
async def run_registered_chain(chain_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return run_chain(chain_name, payload)


@router.post(
    "/api/llm/{connector}",
    dependencies=[rate_limit_dependency("ai")],
)
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

from ..helpers import rate_limit_dependency, router_dependencies
from ..runtime import ExperimentResult, evaluate_experiment, run_experiment

router = APIRouter(tags=["experiments"], dependencies=router_dependencies())


@router.get("/api/experiments/{slug}", response_model=ExperimentResult)
async def get_experiment(slug: str) -> ExperimentResult:
    return evaluate_experiment(slug, payload=None)


@router.post(
    "/api/experiments/{slug}",
    response_model=ExperimentResult,
    dependencies=[rate_limit_dependency("experiments")],
)
async def evaluate_experiment_endpoint(slug: str, payload: Dict[str, Any]) -> ExperimentResult:
    return evaluate_experiment(slug, payload)


@router.post(
    "/api/experiments/{slug}/run",
    response_model=ExperimentResult,
    dependencies=[rate_limit_dependency("experiments")],
)
async def run_experiment_endpoint(slug: str, payload: Dict[str, Any]) -> ExperimentResult:
    return run_experiment(slug, payload)


__all__ = ["router"]
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_frames_router_module() -> str:
    template = '''
"""Generated FastAPI router for frame endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_session
from .. import runtime
from ..helpers import router_dependencies
from ..schemas import FrameErrorResponse, FrameResponse, FrameSchemaPayload

router = APIRouter(prefix="/api/frames", tags=["frames"], dependencies=router_dependencies())


def _frame_not_found(name: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={
            "status_code": 404,
            "error": "FRAME_NOT_FOUND",
            "detail": f"Frame '{name}' is not registered.",
        },
    )


def _ensure_frame_access(name: str) -> Dict[str, Any]:
    frames = runtime.FRAMES if isinstance(runtime.FRAMES, dict) else None
    if not isinstance(frames, dict):
        raise _frame_not_found(name)
    spec = frames.get(name)
    if not isinstance(spec, dict):
        raise _frame_not_found(name)
    access = spec.get("access") or {}
    if isinstance(access, dict):
        if not access.get("public", True):
            # Placeholder for per-frame access control.
            pass
    return spec


@router.get("/", response_model=List[str])
async def list_frames() -> List[str]:
    if not isinstance(runtime.FRAMES, dict):
        return []
    return sorted(runtime.FRAMES.keys())


@router.get(
    "/{name}",
    response_model=FrameResponse,
    responses={404: {"model": FrameErrorResponse}, 400: {"model": FrameErrorResponse}},
)
async def get_frame(
    name: str,
    limit: Optional[int] = Query(None, ge=1),
    offset: Optional[int] = Query(None, ge=0),
    order_by: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_session),
) -> FrameResponse:
    _ensure_frame_access(name)
    context = runtime.build_context(None)
    payload = await runtime.fetch_frame_rows(
        name,
        session,
        context,
        limit=limit,
        offset=offset,
        order_by=order_by,
        as_response=True,
    )
    return FrameResponse(**payload)


@router.get(
    "/{name}/schema",
    response_model=FrameSchemaPayload,
    responses={404: {"model": FrameErrorResponse}},
)
async def get_frame_schema(
    name: str,
    session: AsyncSession = Depends(get_session),
) -> FrameSchemaPayload:
    _ensure_frame_access(name)
    context = runtime.build_context(None)
    payload = await runtime.fetch_frame_schema(name, session, context)
    return FrameSchemaPayload(**payload)


@router.get(
    "/{name}.csv",
    responses={404: {"model": FrameErrorResponse}, 400: {"model": FrameErrorResponse}},
)
async def download_frame_csv(
    name: str,
    limit: Optional[int] = Query(None, ge=1),
    offset: Optional[int] = Query(None, ge=0),
    order_by: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_session),
) -> Response:
    _ensure_frame_access(name)
    context = runtime.build_context(None)
    payload = await runtime.export_frame_csv(
        name,
        session,
        context,
        limit=limit,
        offset=offset,
        order_by=order_by,
    )
    headers = {"Content-Disposition": f"attachment; filename={name}.csv"}
    return Response(content=payload, media_type="text/csv", headers=headers)


@router.get(
    "/{name}.parquet",
    responses={404: {"model": FrameErrorResponse}, 400: {"model": FrameErrorResponse}},
)
async def download_frame_parquet(
    name: str,
    limit: Optional[int] = Query(None, ge=1),
    offset: Optional[int] = Query(None, ge=0),
    order_by: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_session),
) -> Response:
    _ensure_frame_access(name)
    context = runtime.build_context(None)
    payload = await runtime.export_frame_parquet(
        name,
        session,
        context,
        limit=limit,
        offset=offset,
        order_by=order_by,
    )
    headers = {"Content-Disposition": f"attachment; filename={name}.parquet"}
    return Response(content=payload, media_type="application/octet-stream", headers=headers)


__all__ = ["router"]
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_training_router_module() -> str:
    template = '''
"""Generated FastAPI router for training and tuning job endpoints."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_session
from ..helpers import rate_limit_dependency, router_dependencies
from ..runtime import (
    available_training_backends,
    get_training_job,
    get_tuning_job,
    list_training_jobs,
    list_tuning_jobs,
    resolve_training_job_plan,
    run_training_job,
    run_tuning_job,
    training_job_history,
    tuning_job_history,
)

router = APIRouter(prefix="/api/training", tags=["training"], dependencies=router_dependencies())


class TrainingRunRequest(BaseModel):
    payload: Dict[str, Any] = Field(default_factory=dict)
    overrides: Dict[str, Any] = Field(default_factory=dict)


# Training job endpoints

@router.get("/jobs", response_model=List[str])
async def list_training_jobs_endpoint() -> List[str]:
    """List all available training jobs."""
    return list_training_jobs()


@router.get("/jobs/{name}")
async def get_training_job_endpoint(name: str) -> Dict[str, Any]:
    """Get training job specification."""
    spec = get_training_job(name)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Training job '{name}' not found.")
    return spec


@router.get("/jobs/{name}/history")
async def get_training_job_history(name: str) -> List[Dict[str, Any]]:
    """Get training job execution history."""
    return training_job_history(name)


@router.post(
    "/jobs/{name}/plan",
    dependencies=[rate_limit_dependency("training")],
)
async def preview_training_plan(name: str, request: TrainingRunRequest) -> Dict[str, Any]:
    """Preview resolved training plan without execution."""
    try:
        return resolve_training_job_plan(name, request.payload, request.overrides)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime failure
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/jobs/{name}/run",
    dependencies=[rate_limit_dependency("training")],
)
async def execute_training_job(
    name: str,
    request: TrainingRunRequest,
    session: AsyncSession = Depends(get_session),
) -> Dict[str, Any]:
    """Execute a training job."""
    return await run_training_job(name, request.payload, request.overrides, session=session)


@router.get("/jobs/{name}/metrics")
async def get_training_job_metrics(name: str) -> Dict[str, Any]:
    """Get latest metrics for a training job."""
    history = training_job_history(name, limit=1)
    if not history:
        raise HTTPException(status_code=404, detail=f"No execution history for training job '{name}'")
    latest = history[-1]
    return {
        "job": name,
        "status": latest.get("status"),
        "metrics": latest.get("metrics", {}),
        "timestamp": latest.get("ts"),
    }


# Tuning job endpoints

@router.get("/tuning/jobs", response_model=List[str])
async def list_tuning_jobs_endpoint() -> List[str]:
    """List all available tuning jobs."""
    return list_tuning_jobs()


@router.get("/tuning/jobs/{name}")
async def get_tuning_job_endpoint(name: str) -> Dict[str, Any]:
    """Get tuning job specification."""
    spec = get_tuning_job(name)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Tuning job '{name}' not found.")
    return spec


@router.get("/tuning/jobs/{name}/history")
async def get_tuning_job_history(name: str) -> List[Dict[str, Any]]:
    """Get tuning job execution history."""
    return tuning_job_history(name)


@router.post(
    "/tuning/jobs/{name}/run",
    dependencies=[rate_limit_dependency("training")],
)
async def execute_tuning_job(
    name: str,
    request: TrainingRunRequest,
    session: AsyncSession = Depends(get_session),
) -> Dict[str, Any]:
    """Execute a hyperparameter tuning job."""
    return await run_tuning_job(name, request.payload, request.overrides, session=session)


@router.get("/tuning/jobs/{name}/trials")
async def get_tuning_trials(name: str) -> List[Dict[str, Any]]:
    """Get trial results from latest tuning job execution."""
    history = tuning_job_history(name, limit=1)
    if not history:
        raise HTTPException(status_code=404, detail=f"No execution history for tuning job '{name}'")
    latest = history[-1]
    return latest.get("trials", [])


@router.get("/tuning/jobs/{name}/best")
async def get_best_trial(name: str) -> Dict[str, Any]:
    """Get best trial from latest tuning job execution."""
    history = tuning_job_history(name, limit=1)
    if not history:
        raise HTTPException(status_code=404, detail=f"No execution history for tuning job '{name}'")
    latest = history[-1]
    best = latest.get("best_trial")
    if not best:
        raise HTTPException(status_code=404, detail=f"No successful trials for tuning job '{name}'")
    return best


# Backend management

@router.get("/backends", response_model=List[str])
async def list_training_backends() -> List[str]:
    """List available training backends."""
    return available_training_backends()


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
'''
    parts.append(textwrap.dedent(websocket_block).strip())

    parts.append("__all__ = ['router']")
    return "\n\n".join(part for part in parts if part).strip() + "\n"


def _render_crud_router_module(state: BackendState) -> str:
    template = '''
"""Generated FastAPI router for CRUD resources."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_session
from .. import runtime
from ..helpers import router_dependencies
from ..schemas import (
    CrudCatalogResponse,
    CrudDeleteResponse,
    CrudItemResponse,
    CrudListResponse,
)

router = APIRouter(prefix="/api/crud", tags=["crud"], dependencies=router_dependencies())


def _crud_not_found(slug: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"CRUD resource '{slug}' is not registered.")


def _crud_operation_forbidden(slug: str, operation: str) -> HTTPException:
    return HTTPException(status_code=403, detail=f"Operation '{operation}' is not allowed for resource '{slug}'.")


def _route(method: str, *args, **kwargs):
    """Return a router decorator, falling back when FastAPI stand-ins lack HTTP verbs."""
    decorator = getattr(router, method, None)
    if callable(decorator):
        return decorator(*args, **kwargs)
    fallback = router.post if method != "get" and hasattr(router, "post") else router.get
    return fallback(*args, **kwargs)


@router.get("/", response_model=CrudCatalogResponse)
async def list_crud_resources() -> CrudCatalogResponse:
    resources = runtime.describe_crud_resources()
    return CrudCatalogResponse(resources=resources)


@router.get("/{slug}", response_model=CrudListResponse)
async def list_crud_items(
    slug: str,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
) -> CrudListResponse:
    try:
        payload = await runtime.list_crud_resource(slug, session, limit=limit, offset=offset)
    except KeyError:
        raise _crud_not_found(slug)
    except PermissionError as exc:
        raise _crud_operation_forbidden(slug, str(exc) or "list")
    except RuntimeError as exc:
        raise HTTPException(500, detail=str(exc))
    return CrudListResponse(**payload)


@router.get("/{slug}/{identifier}", response_model=CrudItemResponse)
async def get_crud_item(
    slug: str,
    identifier: str,
    session: AsyncSession = Depends(get_session),
) -> CrudItemResponse:
    try:
        payload = await runtime.retrieve_crud_resource(slug, identifier, session)
    except KeyError:
        raise _crud_not_found(slug)
    except PermissionError as exc:
        raise _crud_operation_forbidden(slug, str(exc) or "retrieve")
    except RuntimeError as exc:
        raise HTTPException(500, detail=str(exc))
    result = CrudItemResponse(**payload)
    if result.status == "not_found":
        raise HTTPException(404, detail=f"Record '{identifier}' was not found for resource '{slug}'.")
    return result


@_route("post", "/{slug}", response_model=CrudItemResponse)
async def create_crud_item(
    slug: str,
    payload: Dict[str, Any],
    session: AsyncSession = Depends(get_session),
) -> CrudItemResponse:
    try:
        result = await runtime.create_crud_resource(slug, payload, session)
    except KeyError:
        raise _crud_not_found(slug)
    except PermissionError as exc:
        raise _crud_operation_forbidden(slug, str(exc) or "create")
    except RuntimeError as exc:
        raise HTTPException(500, detail=str(exc))
    return CrudItemResponse(**result)


@_route("put", "/{slug}/{identifier}", response_model=CrudItemResponse)
async def update_crud_item(
    slug: str,
    identifier: str,
    payload: Dict[str, Any],
    session: AsyncSession = Depends(get_session),
) -> CrudItemResponse:
    try:
        result = await runtime.update_crud_resource(slug, identifier, payload, session)
    except KeyError:
        raise _crud_not_found(slug)
    except PermissionError as exc:
        raise _crud_operation_forbidden(slug, str(exc) or "update")
    except RuntimeError as exc:
        raise HTTPException(500, detail=str(exc))
    return CrudItemResponse(**result)


@_route("delete", "/{slug}/{identifier}", response_model=CrudDeleteResponse)
async def delete_crud_item(
    slug: str,
    identifier: str,
    session: AsyncSession = Depends(get_session),
) -> CrudDeleteResponse:
    try:
        result = await runtime.delete_crud_resource(slug, identifier, session)
    except KeyError:
        raise _crud_not_found(slug)
    except PermissionError as exc:
        raise _crud_operation_forbidden(slug, str(exc) or "delete")
    except RuntimeError as exc:
        raise HTTPException(500, detail=str(exc))
    return CrudDeleteResponse(**result)


__all__ = ["router"]
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_observability_router_module() -> str:
    template = '''
"""Generated FastAPI router exposing health and metrics endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter
try:
    from fastapi.responses import PlainTextResponse
except ImportError:  # pragma: no cover - fallback for slim installs
    from starlette.responses import PlainTextResponse  # type: ignore

from ..runtime import health_summary, readiness_checks, render_prometheus_metrics

router = APIRouter(tags=["observability"])


@router.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return health_summary()


@router.get("/readyz")
async def readyz() -> Dict[str, Any]:
    return await readiness_checks()


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    payload = render_prometheus_metrics()
    return PlainTextResponse(payload, media_type="text/plain; version=0.0.4")


__all__ = ["router"]
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_page_endpoint(page: PageSpec) -> List[str]:
    func_name = f"page_{page.slug}_{page.index}"
    path = f"/api/pages/{page.slug}"
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
    base_path = f"/api/pages/{slug}"
    if component.type == "table":
        insight_name = payload.get("insight")
        source_name = payload.get("source") or ""
        source_type = (payload.get("source_type") or "dataset").lower()
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
            f"    source_type = {source_type!r}",
            f"    source_name = {source_name!r}",
            "    frame_spec = runtime.FRAMES.get(source_name) if source_type == 'frame' else None",
            "    if source_type == 'frame':",
            "        rows = await runtime.fetch_frame_rows(source_name, session, context)",
            "        dataset = runtime.DATASETS.get(frame_spec.get('source')) if frame_spec else None",
            "    else:",
            "        rows = await runtime.fetch_dataset_rows(source_name, session, context)",
            "        dataset = runtime.DATASETS.get(source_name)",
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
        source_name = payload.get("source") or ""
        source_type = (payload.get("source_type") or "dataset").lower()
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
            f"    source_type = {source_type!r}",
            f"    source_name = {source_name!r}",
            "    frame_spec = runtime.FRAMES.get(source_name) if source_type == 'frame' else None",
            "    if source_type == 'frame':",
            "        rows = await runtime.fetch_frame_rows(source_name, session, context)",
            "        dataset = runtime.DATASETS.get(frame_spec.get('source')) if frame_spec else None",
            "    else:",
            "        rows = await runtime.fetch_dataset_rows(source_name, session, context)",
            "        dataset = runtime.DATASETS.get(source_name)",
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
