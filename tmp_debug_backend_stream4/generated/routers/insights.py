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
