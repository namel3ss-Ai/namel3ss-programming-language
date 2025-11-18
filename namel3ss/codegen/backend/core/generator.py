"""Orchestrates the backend generation workflow."""

from __future__ import annotations

from pathlib import Path

from namel3ss.ast import App

from ..state import build_backend_state
from .app_module import _render_app_module
from .database import _render_database_module
from .deploy import emit_deployment_artifacts
from .packages import (
    _render_custom_api_stub,
    _render_custom_readme,
    _render_generated_package,
    _render_helpers_package,
)
from .routers import (
    _render_crud_router_module,
    _render_experiments_router_module,
    _render_frames_router_module,
    _render_insights_router_module,
    _render_models_router_module,
    _render_observability_router_module,
    _render_pages_router_module,
    _render_routers_package,
    _render_training_router_module,
)
from .runtime import _render_runtime_module
from .schemas import _render_schemas_module

__all__ = ["generate_backend"]


def generate_backend(
    app: App,
    out_dir: Path,
    embed_insights: bool = False,
    enable_realtime: bool = False,
    connector_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate the backend scaffold for ``app`` into ``out_dir``.
    
    Parameters
    ----------
    app : App
        The parsed application
    out_dir : Path
        Output directory for backend
    embed_insights : bool
        Whether to embed insight evaluations
    enable_realtime : bool
        Whether to enable websocket support
    connector_config : Optional[Dict[str, Any]]
        Runtime connector configuration (retry/concurrency settings)
    """

    state = build_backend_state(app)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    (out_path / "__init__.py").write_text("", encoding="utf-8")
    (out_path / "database.py").write_text(
        _render_database_module(state), encoding="utf-8"
    )

    generated_dir = out_path / "generated"
    routers_dir = generated_dir / "routers"
    helpers_dir = generated_dir / "helpers"
    schemas_dir = generated_dir / "schemas"

    custom_dir = out_path / "custom"
    custom_routes_dir = custom_dir / "routes"

    for path in [
        generated_dir,
        routers_dir,
        helpers_dir,
        schemas_dir,
        custom_dir,
        custom_routes_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    (generated_dir / "__init__.py").write_text(
        _render_generated_package(), encoding="utf-8"
    )
    (generated_dir / "runtime.py").write_text(
        _render_runtime_module(state, embed_insights, enable_realtime, connector_config), encoding="utf-8"
    )
    (helpers_dir / "__init__.py").write_text(
        _render_helpers_package(), encoding="utf-8"
    )
    (schemas_dir / "__init__.py").write_text(
        _render_schemas_module(), encoding="utf-8"
    )

    (routers_dir / "__init__.py").write_text(
        _render_routers_package(), encoding="utf-8"
    )
    (routers_dir / "insights.py").write_text(
        _render_insights_router_module(), encoding="utf-8"
    )
    (routers_dir / "models.py").write_text(
        _render_models_router_module(), encoding="utf-8"
    )
    (routers_dir / "experiments.py").write_text(
        _render_experiments_router_module(), encoding="utf-8"
    )
    (routers_dir / "frames.py").write_text(
        _render_frames_router_module(), encoding="utf-8"
    )
    (routers_dir / "training.py").write_text(
        _render_training_router_module(), encoding="utf-8"
    )
    (routers_dir / "pages.py").write_text(
        _render_pages_router_module(state), encoding="utf-8"
    )
    (routers_dir / "crud.py").write_text(
        _render_crud_router_module(state), encoding="utf-8"
    )
    (routers_dir / "observability.py").write_text(
        _render_observability_router_module(), encoding="utf-8"
    )

    (out_path / "main.py").write_text(
        _render_app_module(), encoding="utf-8"
    )

    if not (custom_dir / "__init__.py").exists():
        (custom_dir / "__init__.py").write_text("", encoding="utf-8")
    if not (custom_routes_dir / "__init__.py").exists():
        (custom_routes_dir / "__init__.py").write_text("", encoding="utf-8")
    custom_readme = custom_dir / "README.md"
    if not custom_readme.exists():
        custom_readme.write_text(_render_custom_readme(), encoding="utf-8")
    custom_api_path = custom_routes_dir / "custom_api.py"
    if not custom_api_path.exists():
        custom_api_path.write_text(_render_custom_api_stub(), encoding="utf-8")

    emit_deployment_artifacts(out_path, state)
