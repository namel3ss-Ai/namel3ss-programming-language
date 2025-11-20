from __future__ import annotations

import textwrap
from typing import Any, Dict, Iterable, List, Optional

from ...state import BackendState, PageComponent, PageSpec, _component_to_serializable
from ..utils import _format_literal


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
