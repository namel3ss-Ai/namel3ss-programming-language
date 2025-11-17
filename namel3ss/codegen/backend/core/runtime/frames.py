"""Reusable runtime helpers for N3Frame execution."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency for real parquet export
    import pyarrow as pa  # type: ignore[import]
    import pyarrow.parquet as pq  # type: ignore[import]
except Exception:  # pragma: no cover - fallback when pyarrow is unavailable
    pa = None  # type: ignore
    pq = None  # type: ignore

RuntimeExpressionEvaluator = Callable[
    [Any, Dict[str, Any], Dict[str, Any], Optional[Sequence[Dict[str, Any]]], Optional[str]],
    Any,
]
RuntimePlaceholderResolver = Callable[[Any, Dict[str, Any]], Any]
RuntimeTruthiness = Callable[[Any], bool]
RuntimeErrorRecorder = Callable[..., Dict[str, Any]]

DEFAULT_FRAME_LIMIT = 100
MAX_FRAME_LIMIT = 1000


_TYPE_CASTERS: Dict[str, Callable[[Any], Any]] = {
    "string": lambda value: str(value) if value is not None else None,
    "text": lambda value: str(value) if value is not None else None,
    "int": lambda value: int(value) if value is not None else None,
    "integer": lambda value: int(value) if value is not None else None,
    "number": lambda value: float(value) if value is not None else None,
    "float": lambda value: float(value) if value is not None else None,
    "decimal": lambda value: float(value) if value is not None else None,
    "bool": lambda value: bool(value) if value is not None else None,
    "boolean": lambda value: bool(value) if value is not None else None,
}


def _coerce_value(dtype: Optional[str], value: Any) -> Any:
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
    columns = frame_spec.get("columns")
    if not isinstance(columns, Iterable):
        return []
    return [column for column in columns if isinstance(column, dict)]


def _iter_constraints(frame_spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    constraints = frame_spec.get("constraints")
    if not isinstance(constraints, Iterable):
        return []
    return [constraint for constraint in constraints if isinstance(constraint, dict)]


def _schema_columns(frame_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    columns = frame_spec.get("columns")
    if not isinstance(columns, list):
        return []
    return [column for column in columns if isinstance(column, dict)]


def _column_name_set(frame_spec: Dict[str, Any]) -> set[str]:
    return {column.get("name") for column in _schema_columns(frame_spec) if column.get("name")}


@dataclass
class N3Frame:
    """Container for evaluated frame rows and schema helpers."""

    name: str
    spec: Dict[str, Any]
    rows: List[Dict[str, Any]]

    def schema_payload(self) -> Dict[str, Any]:
        return {
            "description": self.spec.get("description"),
            "tags": list(self.spec.get("tags") or []),
            "metadata": dict(self.spec.get("metadata") or {}),
            "columns": _schema_columns(self.spec),
            "indexes": list(self.spec.get("indexes") or []),
            "relationships": list(self.spec.get("relationships") or []),
            "constraints": list(self.spec.get("constraints") or []),
            "access": self.spec.get("access"),
            "options": dict(self.spec.get("options") or {}),
        }

    def _sorting_key(self, column: str, row: Dict[str, Any]) -> Tuple[int, Any]:
        value = row.get(column)
        if value is None:
            return (1, "")
        if isinstance(value, (int, float)):
            return (0, float(value))
        if isinstance(value, str):
            return (0, value.lower())
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (0, str(value))

    def ordered_rows(self, order_spec: Optional[Tuple[str, bool]]) -> List[Dict[str, Any]]:
        if not order_spec:
            return list(self.rows)
        column, descending = order_spec
        return sorted(
            self.rows,
            key=lambda row: self._sorting_key(column, row),
            reverse=descending,
        )

    def window_rows(
        self,
        order_spec: Optional[Tuple[str, bool]],
        limit: Optional[int],
        offset: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        ordered = self.ordered_rows(order_spec)
        total = len(ordered)
        start = max(offset, 0)
        if limit is None:
            window = ordered[start:]
        else:
            window = ordered[start : start + limit]
        return window, total

    def to_csv_bytes(self, *, columns: Optional[List[str]] = None, rows: Optional[Sequence[Dict[str, Any]]] = None) -> bytes:
        buffer = io.StringIO()
        active_rows = rows if rows is not None else self.rows
        active_columns = columns or [column.get("name") for column in _schema_columns(self.spec) if column.get("name")]
        writer = csv.DictWriter(buffer, fieldnames=active_columns)
        writer.writeheader()
        for row in active_rows:
            writer.writerow({key: row.get(key) for key in active_columns})
        return buffer.getvalue().encode("utf-8")

    def to_parquet_bytes(self, rows: Optional[Sequence[Dict[str, Any]]] = None) -> bytes:
        active_rows = rows if rows is not None else self.rows
        if pq is not None and pa is not None and active_rows:
            sink = io.BytesIO()
            try:  # pragma: no cover - depends on optional dependency
                table = pa.Table.from_pylist(list(active_rows))
                pq.write_table(table, sink)
                return sink.getvalue()
            except Exception:
                sink.seek(0)
        # Fallback to JSON bytes when pyarrow is unavailable or fails.
        return json.dumps(list(active_rows)).encode("utf-8")


def project_frame_rows(
    frame_name: str,
    frame_spec: Dict[str, Any],
    base_rows: Sequence[Any],
    context: Dict[str, Any],
    *,
    resolve_placeholders: RuntimePlaceholderResolver,
    evaluate_expression: Callable[..., Any],
    runtime_truthy: RuntimeTruthiness,
    record_error: RuntimeErrorRecorder,
) -> List[Dict[str, Any]]:
    """Apply column projections and validations for a frame against base rows."""

    if not isinstance(frame_spec, dict):
        return []
    shaped_rows: List[Dict[str, Any]] = []
    safe_rows: List[Dict[str, Any]] = []
    for entry in base_rows or []:
        if isinstance(entry, dict):
            safe_rows.append(dict(entry))
        else:
            safe_rows.append({"value": entry})
    for source_row in safe_rows:
        shaped_row: Dict[str, Any] = {}
        # Allow validations to access both original and shaped values.
        evaluation_row = dict(source_row)
        for column in _iter_columns(frame_spec):
            name = column.get("name")
            if not name:
                continue
            column_source = column.get("source") or name
            value = source_row.get(column_source)
            expression_expr = column.get("expression_expr")
            if expression_expr is not None:
                value = _evaluate_expression(
                    expression_expr,
                    evaluation_row,
                    context,
                    safe_rows,
                    frame_name,
                    evaluate_expression,
                    column.get("expression"),
                )
            if value is None and column.get("default") is not None:
                value = _resolve_default(column.get("default"), resolve_placeholders, context)
            value = _coerce_value(column.get("dtype"), value)
            if value is None and not column.get("nullable", True):
                record_error(
                    context,
                    code="frame_column_not_null",
                    message=f"Column '{name}' cannot be null in frame '{frame_name}'.",
                    scope=frame_name,
                    source="frame",
                    severity="error",
                )
            shaped_row[name] = value
            evaluation_row[name] = value
            for validation in column.get("validations", []) or []:
                if not isinstance(validation, dict):
                    continue
                validation_expr = validation.get("expression_expr")
                if validation_expr is None:
                    continue
                result = _evaluate_expression(
                    validation_expr,
                    evaluation_row,
                    context,
                    safe_rows,
                    frame_name,
                    evaluate_expression,
                    validation.get("expression"),
                )
                if runtime_truthy(result):
                    continue
                record_error(
                    context,
                    code="frame_column_validation_failed",
                    message=validation.get("message")
                    or f"Validation '{validation.get('name') or name}' failed for column '{name}'.",
                    scope=frame_name,
                    source="frame",
                    detail=validation.get("name"),
                    severity=str(validation.get("severity") or "error"),
                )
        shaped_rows.append(shaped_row)

    if not shaped_rows:
        return []

    for constraint in _iter_constraints(frame_spec):
        expr = constraint.get("expression_expr")
        if expr is None:
            continue
        message = constraint.get("message")
        for row in shaped_rows:
            result = _evaluate_expression(
                expr,
                row,
                context,
                shaped_rows,
                frame_name,
                evaluate_expression,
                constraint.get("expression"),
            )
            if runtime_truthy(result):
                continue
            record_error(
                context,
                code="frame_constraint_failed",
                message=message or f"Constraint '{constraint.get('name')}' failed for frame '{frame_name}'.",
                scope=frame_name,
                source="frame",
                detail=constraint.get("name"),
                severity=str(constraint.get("severity") or "error"),
            )
    return shaped_rows


__all__ = ["N3Frame", "project_frame_rows", "DEFAULT_FRAME_LIMIT", "MAX_FRAME_LIMIT"]
