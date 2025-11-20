"""Abstract base class for frame backend implementations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .constants import RuntimeExpressionEvaluator, RuntimeTruthiness


class _BackendFallback(Exception):
    """Raised internally when a backend requests a downgrade path."""


class _FrameBackendBase:
    """Base class for frame execution backends (Polars, Pandas, Python)."""

    def __init__(
        self,
        *,
        frame_name: str,
        context: Dict[str, Any],
        evaluate_expression: RuntimeExpressionEvaluator,
        runtime_truthy: RuntimeTruthiness,
    ) -> None:
        self._frame_name = frame_name
        self._context = context
        self._evaluate_expression = evaluate_expression
        self._runtime_truthy = runtime_truthy

    def columns(self) -> List[str]:
        """Get list of column names."""
        raise NotImplementedError()

    def filter(self, predicate: Any, predicate_source: Optional[str]) -> "_FrameBackendBase":
        """Filter rows by predicate."""
        raise NotImplementedError()

    def select(self, columns: Sequence[Dict[str, Any]]) -> "_FrameBackendBase":
        """Select and transform columns."""
        raise NotImplementedError()

    def order_by(self, columns: Sequence[Dict[str, Any]]) -> "_FrameBackendBase":
        """Order rows by columns."""
        raise NotImplementedError()

    def join(
        self,
        join_rows: Sequence[Dict[str, Any]],
        join_on: Sequence[str],
        join_how: str,
        join_schema: Optional[Sequence[Dict[str, Any]]],
        join_expressions: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> "_FrameBackendBase":
        """Join with another frame."""
        raise NotImplementedError()

    def summarise(
        self,
        aggregations: Sequence[Dict[str, Any]],
        group_by: Sequence[str],
    ) -> "_FrameBackendBase":
        """Aggregate rows with grouping."""
        raise NotImplementedError()

    def to_rows(self, column_order: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """Convert to list of row dictionaries."""
        raise NotImplementedError()

    def fallback(self) -> "_FrameBackendBase":
        """Fallback to a simpler backend."""
        raise _BackendFallback("No fallback available")


__all__ = ["_BackendFallback", "_FrameBackendBase"]
