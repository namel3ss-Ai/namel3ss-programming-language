"""
HTTP Runtime Adapter - Converts Namel3ss IR to FastAPI backend.

This module provides the bridge between runtime-agnostic IR and concrete
FastAPI implementation. It's the main entry point for generating HTTP backends.

PHASE 2 IMPLEMENTATION:
-----------------------
In Phase 2, we create this adapter as a wrapper around existing codegen.
The existing namel3ss.codegen.backend logic is called directly for now.
Future phases will refactor to consume IR directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from namel3ss.ir import BackendIR


def generate_fastapi_backend(
    ir: BackendIR,
    output_dir: str | Path,
    embed_insights: bool = False,
    enable_realtime: bool = False,
    connector_config: Optional[Dict[str, Any]] = None,
    export_schemas: bool = True,
    schema_version: str = "1.0.0",
) -> None:
    """
    Generate FastAPI backend from Namel3ss IR.
    
    This is the main entry point for the HTTP runtime. It consumes
    runtime-agnostic IR and produces a complete FastAPI application.
    
    Args:
        ir: Backend intermediate representation
        output_dir: Directory for generated backend
        embed_insights: Whether to embed insight evaluations
        enable_realtime: Whether to enable WebSocket support
        connector_config: Runtime connector configuration
        export_schemas: Whether to export schemas for SDK generation
        schema_version: Version for exported schemas
        
    Example:
        >>> from namel3ss import build_backend_ir
        >>> from namel3ss_runtime_http import generate_fastapi_backend
        >>> 
        >>> ir = build_backend_ir(app)
        >>> generate_fastapi_backend(ir, "backend/")
    """
    # PHASE 3: Pass IR directly to codegen
    # generate_backend now accepts both App and BackendIR (Phase 3 update)
    
    from namel3ss.codegen.backend import generate_backend
    from pathlib import Path
    
    # Pass BackendIR directly - no need for _original_app anymore!
    generate_backend(
        app=ir,  # Pass IR directly (generate_backend handles both App and IR)
        out_dir=Path(output_dir),
        embed_insights=embed_insights,
        enable_realtime=enable_realtime,
        connector_config=connector_config,
        export_schemas=export_schemas,
        schema_version=schema_version,
    )


def adapt_ir_to_fastapi(ir: BackendIR) -> Any:
    """
    Convert BackendIR to FastAPI application object.
    
    This function creates an in-memory FastAPI app from IR,
    useful for programmatic usage without file generation.
    
    Args:
        ir: Backend intermediate representation
        
    Returns:
        FastAPI application instance
        
    Note:
        Phase 2: Not yet implemented. Will be added in future phases
        when codegen is refactored to work directly from IR.
    """
    raise NotImplementedError(
        "adapt_ir_to_fastapi not yet implemented. "
        "Use generate_fastapi_backend() to generate files instead."
    )


__all__ = [
    "generate_fastapi_backend",
    "adapt_ir_to_fastapi",
]
