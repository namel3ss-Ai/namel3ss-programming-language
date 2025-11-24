"""
Namel3ss HTTP Runtime

This package adapts Namel3ss IR (intermediate representation) to FastAPI HTTP servers.

Architecture:
-------------
The HTTP runtime consumes runtime-agnostic IR from the language core and generates
concrete FastAPI applications. This maintains separation between language semantics
and HTTP-specific implementation details.

Usage:
------
    from namel3ss import build_backend_ir, Parser
    from namel3ss_runtime_http import generate_fastapi_backend
    
    # Compile .ai to IR
    parser = Parser(source_code)
    module = parser.parse()
    app = module.body[0]
    ir = build_backend_ir(app)
    
    # Generate FastAPI backend from IR
    generate_fastapi_backend(ir, output_dir="backend/")

Components:
-----------
- adapter: IR → FastAPI conversion
- codegen: FastAPI code generation (moved from core)
- cli: Runtime-specific CLI commands (serve, dev)
- templates: FastAPI project templates
"""

__version__ = "0.5.0"

# Public API
from .adapter import generate_fastapi_backend, adapt_ir_to_fastapi

__all__ = [
    "__version__",
    "generate_fastapi_backend",
    "adapt_ir_to_fastapi",
]
