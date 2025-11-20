from __future__ import annotations

import textwrap
from typing import Any, Dict, Iterable, List, Optional

from ...state import BackendState, PageComponent, PageSpec, _component_to_serializable
from ..utils import _format_literal


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
