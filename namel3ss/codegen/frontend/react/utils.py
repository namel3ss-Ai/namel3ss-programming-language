"""Utility functions for React-Vite generation."""

from __future__ import annotations

import re
from pathlib import Path


def normalize_route(route: str | None) -> str:
    """Normalize a route to ensure it starts with /."""
    value = (route or "").strip()
    if not value:
        return "/"
    if not value.startswith("/"):
        value = "/" + value
    return value


def make_component_name(base: str, index: int) -> str:
    """Generate a React component name from a base string."""
    cleaned = re.split(r"[^A-Za-z0-9]+", base)
    parts = [part for part in cleaned if part]
    if not parts:
        parts = ["Index"]
    name = "".join(part.capitalize() for part in parts)
    if not name or name.lower() == "index":
        name = "Index"
    suffix = "Page"
    if not name.endswith(suffix):
        name = f"{name}{suffix}"
    if index == 0:
        return "IndexPage"
    return name


def write_file(path: Path, content: str) -> None:
    """Write content to a file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
