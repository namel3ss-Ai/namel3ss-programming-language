"""Utility functions for AI parser."""

from __future__ import annotations

from typing import Any, List, Optional

from namel3ss.ast import ContextValue


def to_float(value: Any) -> Optional[float]:
    """Convert value to float, returning None if invalid."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def split_memory_names(raw: str) -> List[str]:
    """Split comma-separated memory names, removing quotes."""
    if not raw:
        return []
    candidates = [part.strip() for part in raw.split(',') if part.strip()]
    normalized: List[str] = []
    for candidate in candidates:
        if (candidate.startswith('"') and candidate.endswith('"')) or \
           (candidate.startswith("'") and candidate.endswith("'")):
            normalized.append(candidate[1:-1])
        else:
            normalized.append(candidate)
    return normalized


def build_context_value(scope: str, path_text: str) -> ContextValue:
    """Build a ContextValue from scope and dotted path."""
    parts = [segment for segment in path_text.split('.') if segment]
    if not parts:
        return ContextValue(scope=scope, path=[])
    return ContextValue(scope=scope, path=parts)
