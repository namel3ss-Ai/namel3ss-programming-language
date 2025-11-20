"""Core utility functions for frame execution."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .constants import (
    _TYPE_CASTERS,
    _POLARS_TYPE_MAP,
    pl,
    RuntimePlaceholderResolver,
)


def _coerce_value(dtype: Optional[str], value: Any) -> Any:
    """Coerce a value to the specified data type."""
    if value is None:
        return None
    caster = _TYPE_CASTERS.get(str(dtype or "").lower())
    if caster is None:
        return value
    try:
        return caster(value)
    except Exception:
        return value


def _resolve_default(value: Any, resolver: RuntimePlaceholderResolver, context: Dict[str, Any]) -> Any:
    """Resolve default values using the placeholder resolver."""
    if value is None:
        return None
    try:
        return resolver(value, context)
    except Exception:
        return value


def _evaluate_expression(
    expression: Any,
    row: Dict[str, Any],
    context: Dict[str, Any],
    rows: Optional[Sequence[Dict[str, Any]]],
    frame_name: str,
    evaluator: Callable[..., Any],
    expression_source: Optional[str] = None,
) -> Any:
    """Evaluate an expression in the context of a row."""
    if expression is None:
        return None
    return evaluator(
        expression,
        row,
        context,
        rows,
        frame_name,
        expression_source=expression_source,
    )


def _iter_columns(frame_spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Iterate over columns in a frame spec."""
    columns = frame_spec.get("columns")
    if not isinstance(columns, Iterable):
        return []
    return [column for column in columns if isinstance(column, dict)]


def _iter_constraints(frame_spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Iterate over constraints in a frame spec."""
    constraints = frame_spec.get("constraints")
    if not isinstance(constraints, Iterable):
        return []
    return [constraint for constraint in constraints if isinstance(constraint, dict)]


def _schema_columns(frame_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get schema columns from a frame spec."""
    columns = frame_spec.get("columns")
    if not isinstance(columns, list):
        return []
    return [column for column in columns if isinstance(column, dict)]


def _column_name_set(frame_spec: Dict[str, Any]) -> set[str]:
    """Get set of column names from a frame spec."""
    return {column.get("name") for column in _schema_columns(frame_spec) if column.get("name")}


def _clone_frame_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clone frame rows to avoid mutation."""
    safe_rows: List[Dict[str, Any]] = []
    for row in rows or []:
        if isinstance(row, dict):
            safe_rows.append(dict(row))
    return safe_rows


def _resolve_polars_dtype(dtype_name: Optional[str]) -> Optional["pl.datatypes.DataType"]:
    """Resolve a dtype name to a Polars data type."""
    if pl is None or not dtype_name:
        return None
    mapped = _POLARS_TYPE_MAP.get(str(dtype_name).lower())
    if mapped is None:
        return None
    return getattr(pl, mapped, None)


__all__ = [
    "_coerce_value",
    "_resolve_default",
    "_evaluate_expression",
    "_iter_columns",
    "_iter_constraints",
    "_schema_columns",
    "_column_name_set",
    "_clone_frame_rows",
    "_resolve_polars_dtype",
]
