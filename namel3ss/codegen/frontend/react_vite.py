"""
Generate a Vite + React + TypeScript frontend.

BACKWARD COMPATIBILITY WRAPPER
This module maintains backward compatibility by re-exporting from the new
modular react package. The original 1366-line monolithic implementation has
been refactored into focused modules:

- react/main.py - Main orchestration (generate_react_vite_site)
- react/config.py - Vite configuration generators
- react/scaffolding.py - HTML, main.tsx, CSS generation
- react/components.py - React component generators
- react/hooks.py - Custom React hooks
- react/client.py - API client library
- react/pages.py - Page component generation
- react/utils.py - Shared utility functions

For new code, prefer importing directly from namel3ss.codegen.frontend.react
"""

from __future__ import annotations

# Re-export for backward compatibility
from .react.main import generate_react_vite_site
from .react.pages import ReactPage

__all__ = ["generate_react_vite_site", "ReactPage"]
