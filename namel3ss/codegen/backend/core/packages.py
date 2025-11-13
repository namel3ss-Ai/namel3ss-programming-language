"""Render helper packages and custom extension stubs."""

from __future__ import annotations

import textwrap

__all__ = [
    "_render_generated_package",
    "_render_helpers_package",
    "_render_custom_readme",
    "_render_custom_api_stub",
]


def _render_generated_package() -> str:
    template = '''
"""Generated backend package for Namel3ss (N3).

This file is created automatically. Manual edits may be overwritten.
"""

from __future__ import annotations

from . import runtime

__all__ = ["runtime"]
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_helpers_package() -> str:
    template = '''
"""Helper utilities for Namel3ss generated routers."""

from __future__ import annotations

from typing import Iterable

from fastapi import FastAPI

from ..routers import GENERATED_ROUTERS

__all__ = ["GENERATED_ROUTERS", "include_generated_routers"]


def include_generated_routers(app: FastAPI, routers: Iterable = GENERATED_ROUTERS) -> None:
    """Attach generated routers to ``app`` while allowing custom overrides."""

    for router in routers:
        app.include_router(router)
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_custom_readme() -> str:
    template = '''
# Custom Backend Extensions

This folder is reserved for your handcrafted FastAPI routes and helpers. The
code generator creates it once and will not overwrite files you add here.

- Put reusable dependencies in `__init__.py` or new modules.
- Add route overrides in `routes/custom_api.py` and register them on the
  module-level `router` instance.
- Use the optional `setup(app)` hook to run initialization logic after the
  generated routers are attached (for example, authentication, middleware, or
  event handlers).

Whenever you run the Namel3ss generator again your custom code stays intact.
Refer to the generated modules under `generated/` for available helpers.
'''
    return textwrap.dedent(template).strip() + "\n"


def _render_custom_api_stub() -> str:
    template = '''
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
'''
    return textwrap.dedent(template).strip() + "\n"
