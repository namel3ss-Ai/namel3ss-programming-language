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
