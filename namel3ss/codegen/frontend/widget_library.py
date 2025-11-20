"""
Widget library JavaScript template loader.

This module provides functions to load the widget runtime JavaScript
which is split into multiple parts for maintainability.
"""

from pathlib import Path


def _load_template(filename: str) -> str:
    """Load a JavaScript template file from the templates directory."""
    template_dir = Path(__file__).parent / "templates"
    template_path = template_dir / filename
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {filename}")
    return template_path.read_text(encoding="utf-8")


def generate_widget_library() -> str:
    """
    Generate the complete N3 widget runtime JavaScript library.
    
    The widget library provides:
    - Error handling and display
    - CRUD operations (fetch, submit)
    - Widget rendering (charts, tables, insights)
    - Form and action handling
    - Real-time WebSocket updates
    - State management
    
    Returns:
        Complete JavaScript code for the widget runtime
    
    The library is assembled from 3 modular template files:
    1. widget-core.js: Core utilities, CRUD operations, error handling (~500 lines)
    2. widget-rendering.js: Widget rendering, component registry, form/action handling (~700 lines)
    3. widget-realtime.js: WebSocket connections, fallback polling, state management (~300 lines)
    """
    # Load the 3 JavaScript template files
    core = _load_template("widget-core.js")
    rendering = _load_template("widget-rendering.js")
    realtime = _load_template("widget-realtime.js")
    
    # Concatenate in correct order with newline separators
    return "\n\n".join([core, rendering, realtime])

