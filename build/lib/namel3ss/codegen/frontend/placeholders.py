"""Placeholder data used by the frontend renderer."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from namel3ss.ast import ShowChart

TEXT_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def placeholder_table_rows(columns: List[str]) -> List[Dict[str, Any]]:
    """Create deterministic table rows for preview rendering."""
    cols = columns or ["Column 1", "Column 2"]
    rows: List[Dict[str, Any]] = []
    for index in range(3):
        rows.append({col: f"{col} {index + 1}" for col in cols})
    return rows


def build_placeholder_chart_payload(chart_stmt: ShowChart) -> Dict[str, Any]:
    """Return a small chart dataset so charts render before live data exists."""
    labels = ["Jan", "Feb", "Mar", "Apr", "May"]
    values = [12, 19, 7, 15, 10]
    return {
        "labels": labels,
        "datasets": [
            {
                "label": chart_stmt.heading or "Series",
                "data": values,
            }
        ],
    }
