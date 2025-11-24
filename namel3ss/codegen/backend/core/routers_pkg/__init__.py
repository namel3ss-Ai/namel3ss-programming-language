"""
Routers package - extracted modules for router code generation.

This package contains the modular components of the routers module.
Each module handles generation of a specific router type.
"""

from .package_init import _render_routers_package
from .insights_router import _render_insights_router_module
from .models_router import _render_models_router_module
from .experiments_router import _render_experiments_router_module
from .frames_router import _render_frames_router_module
from .training_router import _render_training_router_module
from .pages_router import _render_pages_router_module
from .crud_router import _render_crud_router_module
from .datasets_router import _render_datasets_router_module
from .websocket_router import _render_websocket_router_module
from .observability_router import _render_observability_router_module
from .planning_router import _render_planning_router_module
from .endpoint_generators import (
    _render_page_endpoint,
    _render_component_endpoint,
    _render_insight_endpoint,
)

__all__ = [
    "_render_routers_package",
    "_render_insights_router_module",
    "_render_models_router_module",
    "_render_experiments_router_module",
    "_render_frames_router_module",
    "_render_training_router_module",
    "_render_pages_router_module",
    "_render_crud_router_module",
    "_render_datasets_router_module",
    "_render_websocket_router_module",
    "_render_observability_router_module",
    "_render_planning_router_module",
    "_render_page_endpoint",
    "_render_component_endpoint",
    "_render_insight_endpoint",
]
