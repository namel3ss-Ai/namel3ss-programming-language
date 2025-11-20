from __future__ import annotations

import textwrap
from typing import Any, Dict, Iterable, List, Optional

from ...state import BackendState, PageComponent, PageSpec, _component_to_serializable
from ..utils import _format_literal


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
