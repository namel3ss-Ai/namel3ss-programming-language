"""Aggregated FastAPI routers for Namel3ss (auto-generated)."""

from __future__ import annotations

from . import experiments, insights, models, pages

insights_router = insights.router
models_router = models.router
experiments_router = experiments.router
pages_router = pages.router

GENERATED_ROUTERS = (
    insights_router,
    models_router,
    experiments_router,
    pages_router,
)

__all__ = [
    "insights_router",
    "models_router",
    "experiments_router",
    "pages_router",
    "GENERATED_ROUTERS",
]
