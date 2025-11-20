"""
React + Vite frontend generation package.

This package contains modular generators for creating a complete
Vite + React + TypeScript frontend from a Namel3ss application.

Modules:
    main: Main orchestration function (generate_react_vite_site)
    config: Vite configuration files (package.json, tsconfig, vite.config)
    scaffolding: HTML entry point, main.tsx, CSS generation
    components: React component generators (Layout, Toast, widgets)  
    hooks: Custom React hooks (realtime updates)
    client: API client library and utilities
    pages: Page component generation and routing
    utils: Shared utility functions
"""

from .main import generate_react_vite_site

__all__ = ["generate_react_vite_site"]
