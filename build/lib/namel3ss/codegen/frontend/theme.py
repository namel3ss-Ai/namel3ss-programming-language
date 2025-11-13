"""Theme and layout helpers for the frontend generator."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from namel3ss.ast import App, LayoutMeta, Page


def theme_palette(theme: Optional[str]) -> Dict[str, str]:
    """Return a basic color palette for the provided theme name."""
    mode = (theme or "").lower()
    if mode in {"dark", "night", "dim"}:
        return {
            "text": "#f5f5f5",
            "grid": "rgba(255, 255, 255, 0.15)",
            "background": "#111827",
        }
    return {
        "text": "#1f2937",
        "grid": "rgba(15, 23, 42, 0.08)",
        "background": "#ffffff",
    }


def layout_to_payload(layout: Optional[LayoutMeta]) -> Optional[Dict[str, Any]]:
    """Serialise ``LayoutMeta`` instances into JSON-friendly payloads."""
    if not layout:
        return None
    payload: Dict[str, Any] = {}
    if layout.width is not None:
        payload["width"] = layout.width
    if layout.height is not None:
        payload["height"] = layout.height
    if layout.variant:
        payload["variant"] = layout.variant
    if layout.align:
        payload["align"] = layout.align
    if layout.emphasis:
        payload["emphasis"] = layout.emphasis
    if layout.extras:
        payload["extras"] = copy.deepcopy(layout.extras)
    return payload or None


def infer_theme_mode(app: App, page: Page) -> Optional[str]:
    """Determine the effective theme mode for a page."""
    for key in ("theme", "mode", "appearance"):
        if isinstance(page.layout, dict):
            value = page.layout.get(key)
            if value:
                return str(value)
    for key in ("theme", "mode", "appearance"):
        value = app.theme.values.get(key)
        if value:
            return str(value)
    return None


def style_to_inline(styles: Dict[str, str]) -> str:
    """Convert a style dictionary into an inline CSS string."""
    if not styles:
        return ""
    css_map = {
        "color": "color",
        "background": "background-color",
        "background colour": "background-color",
        "size": "font-size",
        "align": "text-align",
        "weight": "font-weight",
    }
    parts = []
    for key, value in styles.items():
        if isinstance(value, (dict, list, tuple)):
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        value_str = str(value)
        prop = css_map.get(key.lower(), key.replace(" ", "-"))
        if prop == "font-size":
            lower_val = value_str.lower()
            if lower_val == "small":
                value_str = "0.875rem"
            elif lower_val == "medium":
                value_str = "1rem"
            elif lower_val == "large":
                value_str = "1.25rem"
            elif lower_val in {"xlarge", "xl"}:
                value_str = "1.5rem"
        parts.append(f"{prop}: {value_str};")
    return " ".join(parts)
