"""Frontend data helpers that are real-first and structured-error aware."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, TypedDict, Union, Literal

TEXT_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class TableRowsResult(TypedDict, total=False):
    """Result returned by :func:`placeholder_table_rows`."""

    status: Literal["ok", "error"]
    rows: List[Dict[str, Any]]
    error: str
    detail: str


class ChartPayloadResult(TypedDict, total=False):
    """Result returned by :func:`build_placeholder_chart_payload`."""

    status: Literal["ok", "error"]
    payload: Dict[str, Any]
    error: str
    detail: str


def placeholder_table_rows(
    columns: Sequence[Union[str, Dict[str, Any]]],
    *,
    real_rows: Optional[List[Dict[str, Any]]] = None,
) -> TableRowsResult:
    """Return table rows when provided, otherwise surface a structured error.

    Parameters
    ----------
    columns:
        Declared column metadata. Retained for parity with previous behaviour
        but not used to synthesise data.
    real_rows:
        Actual table rows gathered from a backend pipeline.

    Returns
    -------
    TableRowsResult
        ``{"status": "ok", "rows": [...]}`` if ``real_rows`` is supplied.
        Otherwise ``{"status": "error", "error": "no_data", "detail": ...}``.
    """

    _ = columns  # columns are kept for future validation hooks.

    if real_rows is not None:
        return {"status": "ok", "rows": list(real_rows)}

    return {
        "status": "error",
        "error": "no_data",
        "detail": "No table rows were provided by the runtime backend.",
    }


def build_placeholder_chart_payload(
    chart_kind: str,
    *,
    real_payload: Optional[Dict[str, Any]] = None,
) -> ChartPayloadResult:
    """Return chart payloads only when real data is present.

    Parameters
    ----------
    chart_kind:
        Chart type identifier (``bar``, ``line`` …). Preserved for caller
        awareness but not used to generate synthetic data.
    real_payload:
        Actual chart payload supplied by backend evaluation.

    Returns
    -------
    ChartPayloadResult
        ``{"status": "ok", "payload": ...}`` if ``real_payload`` exists.
        Otherwise ``{"status": "error", "error": "no_data", "detail": ...}``.
    """

    if real_payload is not None:
        payload = dict(real_payload)
        payload.setdefault("kind", chart_kind)
        return {"status": "ok", "payload": payload}

    return {
        "status": "error",
        "error": "no_data",
        "detail": "No chart data was supplied for the requested widget.",
    }
