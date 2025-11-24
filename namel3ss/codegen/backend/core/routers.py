"""Render the generated FastAPI routers."""

from __future__ import annotations

from .routers_pkg import (
    _render_routers_package,
    _render_insights_router_module,
    _render_models_router_module,
    _render_experiments_router_module,
    _render_frames_router_module,
    _render_training_router_module,
    _render_pages_router_module,
    _render_crud_router_module,
    _render_datasets_router_module,
    _render_websocket_router_module,
    _render_observability_router_module,
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
    "_render_page_endpoint",
    "_render_component_endpoint",
    "_render_insight_endpoint",
]
