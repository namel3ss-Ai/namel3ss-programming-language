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
