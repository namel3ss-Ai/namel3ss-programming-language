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

import importlib
import logging
import os
from typing import Any, Callable, Iterable, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request

__all__ = [
    "GENERATED_ROUTERS",
    "include_generated_routers",
    "router_dependencies",
]


_LOGGER = logging.getLogger("namel3ss.security")
_API_KEY = os.getenv("NAMEL3SS_API_KEY")
_CUSTOM_DEPENDENCY_PATH = os.getenv("NAMEL3SS_ROUTER_DEPENDENCY")
_CUSTOM_DEPENDENCY: Optional[Callable[..., Any]] = None


def _load_custom_dependency() -> Optional[Callable[..., Any]]:
    path = _CUSTOM_DEPENDENCY_PATH
    if not path:
        return None
    candidate_path = path.strip()
    if not candidate_path:
        return None
    module_name: Optional[str]
    attr_name: Optional[str]
    if ":" in candidate_path:
        module_name, attr_name = candidate_path.rsplit(":", 1)
    else:
        module_name, attr_name = candidate_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        candidate = getattr(module, attr_name)
    except Exception:
        _LOGGER.exception("Failed to import router dependency '%s'", candidate_path)
        return None
    if not callable(candidate):
        _LOGGER.warning("Router dependency '%s' is not callable", candidate_path)
        return None
    return candidate


def _clean_authorization(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    token = value.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token or None


async def _require_api_key(request: Request) -> None:
    if not _API_KEY:
        return
    headers = request.headers
    provided = headers.get("x-api-key") or _clean_authorization(headers.get("authorization"))
    if provided != _API_KEY:
        _LOGGER.warning("Rejected request with invalid API key header")
        raise HTTPException(status_code=401, detail="Invalid API key")


def router_dependencies() -> List[Any]:
    """Return dependencies applied to every generated router."""

    global _CUSTOM_DEPENDENCY
    dependencies: List[Any] = []
    if _CUSTOM_DEPENDENCY_PATH and _CUSTOM_DEPENDENCY is None:
        _CUSTOM_DEPENDENCY = _load_custom_dependency()
    if _CUSTOM_DEPENDENCY is not None:
        dependencies.append(Depends(_CUSTOM_DEPENDENCY))
    if _API_KEY:
        dependencies.append(Depends(_require_api_key))
    return dependencies


def _generated_routers() -> Iterable:
    from ..routers import GENERATED_ROUTERS  # local import to avoid circular dependency

    return GENERATED_ROUTERS


class _GeneratedRoutersProxy:
    def __iter__(self):
        return iter(_generated_routers())

    def __len__(self) -> int:
        return len(list(_generated_routers()))

    def __getitem__(self, index: int) -> Any:
        return list(_generated_routers())[index]


GENERATED_ROUTERS = _GeneratedRoutersProxy()


def include_generated_routers(app: FastAPI, routers: Optional[Iterable] = None) -> None:
    """Attach generated routers to ``app`` while allowing custom overrides."""

    target = routers or _generated_routers()
    for router in target:
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
