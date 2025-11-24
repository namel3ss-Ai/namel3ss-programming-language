"""
Frontend Runtime Adapter - Converts Namel3ss IR to frontend applications.

This module provides the bridge between runtime-agnostic IR and concrete
frontend implementations (static sites, React, etc.).

PHASE 2 IMPLEMENTATION:
-----------------------
In Phase 2, we create this adapter as a wrapper around existing codegen.
Future phases will refactor to consume IR directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from namel3ss.ir import FrontendIR


def generate_static_site(
    ir: FrontendIR,
    output_dir: str | Path,
    enable_realtime: bool = False,
) -> None:
    """
    Generate static HTML/JS site from Namel3ss IR.
    
    Args:
        ir: Frontend intermediate representation
        output_dir: Directory for generated frontend
        enable_realtime: Whether to enable WebSocket support
        
    Example:
        >>> from namel3ss import build_frontend_ir
        >>> from namel3ss_runtime_frontend import generate_static_site
        >>> 
        >>> ir = build_frontend_ir(app)
        >>> generate_static_site(ir, "build/")
    """
    # PHASE 3: Pass IR directly to codegen
    from namel3ss.codegen.frontend import generate_site
    
    # Pass FrontendIR directly (generate_site now handles both App and IR)
    generate_site(
        app=ir,  # Pass IR directly!
        output_dir=str(output_dir),
        enable_realtime=enable_realtime,
        target="static",
    )


def generate_react_app(
    ir: FrontendIR,
    output_dir: str | Path,
    enable_realtime: bool = False,
) -> None:
    """
    Generate React application from Namel3ss IR.
    
    Args:
        ir: Frontend intermediate representation
        output_dir: Directory for generated React app
        enable_realtime: Whether to enable WebSocket support
        
    Example:
        >>> from namel3ss import build_frontend_ir
        >>> from namel3ss_runtime_frontend import generate_react_app
        >>> 
        >>> ir = build_frontend_ir(app)
        >>> generate_react_app(ir, "frontend/")
    """
    # PHASE 3: Pass IR directly to codegen
    from namel3ss.codegen.frontend import generate_site
    
    # Pass FrontendIR directly
    generate_site(
        app=ir,  # Pass IR directly!
        output_dir=str(output_dir),
        enable_realtime=enable_realtime,
        target="react-vite",
    )


__all__ = [
    "generate_static_site",
    "generate_react_app",
]
