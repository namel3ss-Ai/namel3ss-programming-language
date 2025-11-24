from __future__ import annotations

import textwrap
from typing import Any, Dict, Iterable, List, Optional

from ...state import BackendState, PageComponent, PageSpec, _component_to_serializable
from ..utils import _format_literal


def _render_routers_package(include_metadata: bool = False, include_websocket: bool = False) -> str:
    """
    Render routers package __init__.py.
    
    Args:
        include_metadata: Whether to include metadata router for SDK generation
        include_websocket: Whether to include WebSocket router for realtime
    
    Returns:
        Python code for routers/__init__.py
    """
    
    # Build import statement
    import_modules = ["crud", "datasets", "experiments", "frames", "insights", "models", "observability", "pages", "training"]
    if include_websocket:
        import_modules.append("websocket")
    if include_metadata:
        import_modules.append("metadata")
    
    import_line = f"from . import {', '.join(sorted(import_modules))}"
    
    # Build router assignments
    router_assignments = [
        "insights_router = insights.router",
        "models_router = models.router",
        "experiments_router = experiments.router",
        "frames_router = frames.router",
        "training_router = training.router",
        "pages_router = pages.router",
        "crud_router = crud.router",
        "datasets_router = datasets.router",
        "observability_router = observability.router",
    ]
    
    router_tuple = [
        "    insights_router,",
        "    models_router,",
        "    experiments_router,",
        "    frames_router,",
        "    training_router,",
        "    pages_router,",
        "    crud_router,",
        "    datasets_router,",
        "    observability_router,",
    ]
    
    all_exports = [
        "    \"insights_router\",",
        "    \"models_router\",",
        "    \"experiments_router\",",
        "    \"frames_router\",",
        "    \"training_router\",",
        "    \"pages_router\",",
        "    \"crud_router\",",
        "    \"datasets_router\",",
        "    \"observability_router\",",
    ]
    
    if include_websocket:
        router_assignments.append("websocket_router = websocket.router")
        router_tuple.append("    websocket_router,")
        all_exports.append("    \"websocket_router\",")
    
    if include_metadata:
        router_assignments.append("metadata_router = metadata.router")
        router_tuple.append("    metadata_router,")
        all_exports.append("    \"metadata_router\",")
    
    all_exports.append("    \"GENERATED_ROUTERS\",")
    
    template = f'''
"""Aggregated FastAPI routers for Namel3ss (auto-generated)."""

from __future__ import annotations

{import_line}

{chr(10).join(router_assignments)}

GENERATED_ROUTERS = (
{chr(10).join(router_tuple)}
)

__all__ = [
{chr(10).join(all_exports)}
]
'''
    return textwrap.dedent(template).strip() + "\n"
