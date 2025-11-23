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
