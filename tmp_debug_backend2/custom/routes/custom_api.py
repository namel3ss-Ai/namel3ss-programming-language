"""Custom API extensions for your Namel3ss backend.

This module is created once. Add routes, dependencies, or hooks freely; the
generator will never overwrite it.
"""

from __future__ import annotations

from fastapi import APIRouter

# Optional helpers when you want to reuse generated logic:
# from ..generated.routers import experiments as generated_experiments
# from ..generated.routers import models as generated_models
# from ..generated.schemas import ExperimentResult, PredictionResponse

router = APIRouter()


# Example override (uncomment and adapt):
# @router.post(
#     "/api/models/{model_name}/predict",
#     response_model=PredictionResponse,
#     include_in_schema=False,
# )
# async def predict_with_tracking(model_name: str, payload: dict) -> PredictionResponse:
#     base = await generated_models.predict(model_name, payload)
#     base.metadata.setdefault("tags", []).append("customized")
#     base.metadata["handled_by"] = "custom_api"
#     return base
#
# Example extension:
# @router.get("/api/experiments/{slug}/summary", response_model=ExperimentResult)
# async def experiment_summary(slug: str) -> ExperimentResult:
#     result = await generated_experiments.get_experiment(slug)
#     result.metadata["summary"] = f"Experiment {slug} served by custom routes."
#     return result
#
# The optional ``setup`` hook runs after generated routers are registered.


def setup(app) -> None:  # pragma: no cover - user may replace implementation
    """Run initialization after generated routers are registered."""

    _ = app  # Replace with custom logic (auth, logging, etc.)
