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
