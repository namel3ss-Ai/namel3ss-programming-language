from __future__ import annotations

import textwrap
from typing import Any, Dict, Iterable, List, Optional

from ...state import BackendState, PageComponent, PageSpec, _component_to_serializable
from ..utils import _format_literal


def _render_routers_package(include_metadata: bool = False) -> str:
    """
    Render routers package __init__.py.
    
    Args:
        include_metadata: Whether to include metadata router for SDK generation
    
    Returns:
        Python code for routers/__init__.py
    """
    if include_metadata:
        template = '''
"""Aggregated FastAPI routers for Namel3ss (auto-generated)."""

from __future__ import annotations

from . import crud, datasets, experiments, frames, insights, metadata, models, observability, pages, training

insights_router = insights.router
models_router = models.router
experiments_router = experiments.router
frames_router = frames.router
training_router = training.router
pages_router = pages.router
crud_router = crud.router
datasets_router = datasets.router
observability_router = observability.router
metadata_router = metadata.router

GENERATED_ROUTERS = (
    insights_router,
    models_router,
    experiments_router,
    frames_router,
    training_router,
    pages_router,
    crud_router,
    datasets_router,
    observability_router,
    metadata_router,
)

__all__ = [
    "insights_router",
    "models_router",
    "experiments_router",
    "frames_router",
    "training_router",
    "pages_router",
    "crud_router",
    "datasets_router",
    "observability_router",
    "metadata_router",
    "GENERATED_ROUTERS",
]
'''
    else:
        template = '''
"""Aggregated FastAPI routers for Namel3ss (auto-generated)."""

from __future__ import annotations

from . import crud, datasets, experiments, frames, insights, models, observability, pages, training

insights_router = insights.router
models_router = models.router
experiments_router = experiments.router
frames_router = frames.router
training_router = training.router
pages_router = pages.router
crud_router = crud.router
datasets_router = datasets.router
observability_router = observability.router

GENERATED_ROUTERS = (
    insights_router,
    models_router,
    experiments_router,
    frames_router,
    training_router,
    pages_router,
    crud_router,
    datasets_router,
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
    "datasets_router",
    "observability_router",
    "GENERATED_ROUTERS",
]
'''
    return textwrap.dedent(template).strip() + "\n"
