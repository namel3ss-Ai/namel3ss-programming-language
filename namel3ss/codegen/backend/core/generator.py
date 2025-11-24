"""Orchestrates the backend generation workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from namel3ss.ast import App

from ..inline_blocks import (
    collect_inline_blocks,
    generate_inline_python_module,
    generate_inline_react_components,
)
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
    _render_datasets_router_module,
    _render_experiments_router_module,
    _render_frames_router_module,
    _render_insights_router_module,
    _render_models_router_module,
    _render_observability_router_module,
    _render_pages_router_module,
    _render_planning_router_module,
    _render_routers_package,
    _render_training_router_module,
    _render_websocket_router_module,
)
from .runtime import _render_runtime_module
from .schemas import _render_schemas_module

__all__ = ["generate_backend"]


def generate_backend(
    app: Union[App, "BackendIR"],  # Accept both App AST and BackendIR
    out_dir: Path,
    embed_insights: bool = False,
    enable_realtime: bool = False,
    connector_config: Optional[Dict[str, Any]] = None,
    export_schemas: bool = True,
    schema_version: str = "1.0.0",
) -> None:
    """Generate the backend scaffold for ``app`` into ``out_dir``.
    
    PHASE 3 ARCHITECTURE:
    ---------------------
    Accepts either App AST or BackendIR for maximum flexibility:
    - Legacy path: App → build IR internally → generate
    - New path: BackendIR → generate directly
    
    This dual signature enables gradual migration while maintaining
    backward compatibility.
    
    Parameters
    ----------
    app : Union[App, BackendIR]
        Either the parsed application AST or pre-built BackendIR
    out_dir : Path
        Output directory for backend
    embed_insights : bool
        Whether to embed insight evaluations
    enable_realtime : bool
        Whether to enable websocket support
    connector_config : Optional[Dict[str, Any]]
        Runtime connector configuration (retry/concurrency settings)
    export_schemas : bool
        Whether to export schemas for SDK generation
    schema_version : str
        Version for exported schemas (default: "1.0.0")
    """

    # ========================================================================
    # PHASE 3: Handle both App and BackendIR inputs
    # ========================================================================
    from namel3ss.ir import BackendIR, build_backend_ir
    
    # Determine if input is IR or App
    if isinstance(app, BackendIR):
        backend_ir = app
        # Extract original app from IR metadata for backward compat
        app_ast = backend_ir.metadata.get("_original_app")
        if app_ast is None:
            raise ValueError(
                "BackendIR missing '_original_app' metadata. "
                "Please ensure IR was built with build_backend_ir()."
            )
    else:
        # Legacy path: App AST provided
        app_ast = app
        backend_ir = build_backend_ir(app_ast)
    
    # TODO Phase 4: Remove dependency on app_ast, use only backend_ir
    # For now, we continue with existing BackendState flow
    
    # ========================================================================
    # EXISTING LOGIC: Generate FastAPI backend from BackendState
    # ========================================================================
    
    # Work on a deep copy to avoid mutating the input AST across builds
    import copy
    app_copy = copy.deepcopy(app_ast)
    state = build_backend_state(app_copy)
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
    runtime_code = _render_runtime_module(state, embed_insights, enable_realtime, connector_config)
    (generated_dir / "runtime.py").write_text(runtime_code, encoding="utf-8")
    # Backward compatibility: also emit runtime.py at the root as tests expect
    (out_path / "runtime.py").write_text(runtime_code, encoding="utf-8")
    
    # Generate inline blocks module (use app_copy which is the App AST)
    inline_blocks = collect_inline_blocks(app_copy)
    if inline_blocks.get("python"):
        inline_python_module = generate_inline_python_module(inline_blocks["python"])
        (generated_dir / "inline_blocks.py").write_text(
            inline_python_module, encoding="utf-8"
        )
    
    # Generate React components
    if inline_blocks.get("react"):
        react_components = generate_inline_react_components(inline_blocks["react"])
        react_dir = generated_dir / "react_components"
        react_dir.mkdir(exist_ok=True)
        for filename, code in react_components.items():
            (react_dir / filename).write_text(code, encoding="utf-8")
    
    (helpers_dir / "__init__.py").write_text(
        _render_helpers_package(), encoding="utf-8"
    )
    (schemas_dir / "__init__.py").write_text(
        _render_schemas_module(), encoding="utf-8"
    )

    (routers_dir / "__init__.py").write_text(
        _render_routers_package(include_metadata=export_schemas, include_websocket=enable_realtime), encoding="utf-8"
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
    (routers_dir / "datasets.py").write_text(
        _render_datasets_router_module(backend_ir), encoding="utf-8"
    )
    (routers_dir / "websocket.py").write_text(
        _render_websocket_router_module(backend_ir, enable_realtime), encoding="utf-8"
    )
    (routers_dir / "observability.py").write_text(
        _render_observability_router_module(), encoding="utf-8"
    )
    (routers_dir / "planning.py").write_text(
        _render_planning_router_module(state), encoding="utf-8"
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

    # Generate metadata router for SDK generation
    if export_schemas:
        metadata_router_path = routers_dir / "metadata.py"
        try:
            from namel3ss.sdk_sync import (
                export_schemas_from_app,
                generate_metadata_router,
            )
            from namel3ss.sdk_sync.ir import SchemaVersion
            
            # Export schemas to registry
            version = SchemaVersion.parse(schema_version)
            spec = export_schemas_from_app(
                app=app,
                version=version,
                output_path=schemas_dir / "spec.json",
                namespace="app",
            )
            
            # Generate metadata router
            metadata_code = generate_metadata_router()
            metadata_router_path.write_text(metadata_code, encoding="utf-8")
            
        except ImportError:
            # SDK-Sync not available, skip schema export
            pass

    emit_deployment_artifacts(out_path, state)
