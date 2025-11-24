"""
Namel3ss Frontend Runtime

This package generates frontend applications from Namel3ss IR.

Architecture:
-------------
The frontend runtime consumes runtime-agnostic IR from the language core and generates
static sites, React apps, or other frontend implementations.

Usage:
------
    from namel3ss import build_frontend_ir, Parser
    from namel3ss_runtime_frontend import generate_static_site
    
    # Compile .ai to IR
    parser = Parser(source_code)
    module = parser.parse()
    app = module.body[0]
    ir = build_frontend_ir(app)
    
    # Generate static site from IR
    generate_static_site(ir, output_dir="build/")

Components:
-----------
- adapter: IR → Frontend conversion
- generators: Static, React, Vue generators (moved from core)
- templates: Frontend templates
"""

__version__ = "0.5.0"

# Public API
from .adapter import generate_static_site, generate_react_app

__all__ = [
    "__version__",
    "generate_static_site",
    "generate_react_app",
]
