"""Reusable runtime helpers for N3Frame execution."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency for dataframe execution
    import polars as pl  # type: ignore[import]
except Exception:  # pragma: no cover - fallback when polars is unavailable
    pl = None  # type: ignore

try:  # pragma: no cover - optional dependency for dataframe execution
    import pandas as pd  # type: ignore[import]
except Exception:  # pragma: no cover - fallback when pandas is unavailable
    pd = None  # type: ignore

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


class FramePipelineExecutionError(RuntimeError):
    """Raised when a frame pipeline fails validation or execution."""


def _clone_frame_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    safe_rows: List[Dict[str, Any]] = []
    for row in rows or []:
        if isinstance(row, dict):
            safe_rows.append(dict(row))
    return safe_rows


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


class _FramePipelineEngine:
    def __init__(
        self,
        frame_name: str,
        rows: Sequence[Dict[str, Any]],
        context: Dict[str, Any],
        *,
        evaluate_expression: RuntimeExpressionEvaluator,
        runtime_truthy: RuntimeTruthiness,
    ) -> None:
        self._frame_name = frame_name
        self._context = context
        self._evaluate_expression = evaluate_expression
        self._runtime_truthy = runtime_truthy
        self._group_by: List[str] = []
        self._engine: str = "python"
        self._pl_df: Optional[Any] = None
        self._pd_df: Optional[Any] = None
        self._rows: List[Dict[str, Any]] = _clone_frame_rows(rows)
        self._initialize_frame()

    def _initialize_frame(self) -> None:
        if pl is not None:
            try:
                self._pl_df = pl.DataFrame(self._rows)
                self._engine = "polars"
                self._rows = []
                return
            except Exception:  # pragma: no cover - fallback
                self._pl_df = None
        if pd is not None:
            try:
                self._pd_df = pd.DataFrame(self._rows)
                self._engine = "pandas"
                self._rows = []
                return
            except Exception:  # pragma: no cover - fallback
                self._pd_df = None
        self._engine = "python"

    def _current_columns(self) -> List[str]:
        if self._engine == "polars" and self._pl_df is not None:
            return list(self._pl_df.columns)
        if self._engine == "pandas" and self._pd_df is not None:
            return list(self._pd_df.columns)
        ordered: List[str] = []
        for row in self._rows:
            for key in row.keys():
                if key not in ordered:
                    ordered.append(key)
        return ordered

    def _ensure_columns_exist(self, columns: Sequence[str]) -> None:
        available = set(self._current_columns())
        missing = [column for column in columns if column and column not in available]
        if missing:
            joined = ", ".join(sorted(missing))
            raise FramePipelineExecutionError(
                f"Frame '{self._frame_name}' does not define columns: {joined}"
            )

    def _fallback_to_python(self) -> None:
        if self._engine == "polars" and self._pl_df is not None:
            self._rows = [dict(row) for row in self._pl_df.to_dicts()]
            self._pl_df = None
        elif self._engine == "pandas" and self._pd_df is not None:
            self._rows = self._pd_df.to_dict(orient="records")
            self._pd_df = None
        self._engine = "python"

    def _snapshot_rows(self) -> List[Dict[str, Any]]:
        if self._engine == "polars" and self._pl_df is not None:
            return [dict(row) for row in self._pl_df.to_dicts()]
        if self._engine == "pandas" and self._pd_df is not None:
            return self._pd_df.to_dict(orient="records")
        return [dict(row) for row in self._rows]

    def apply_filter(self, predicate: Any, predicate_source: Optional[str]) -> None:
        if predicate is None:
            return
        if self._engine == "polars" and self._pl_df is not None and pl is not None:
            try:
                expr = _build_polars_expression(predicate)
                self._pl_df = self._pl_df.filter(expr)
                return
            except FramePipelineExecutionError:
                self._fallback_to_python()
        if self._engine == "pandas" and self._pd_df is not None and pd is not None:
            rows_snapshot = self._pd_df.to_dict(orient="records")
            mask = self._pd_df.apply(
                lambda row: bool(
                    _evaluate_expression(
                        predicate,
                        row.to_dict(),
                        self._context,
                        rows_snapshot,
                        self._frame_name,
                        self._evaluate_expression,
                        predicate_source,
                    )
                ),
                axis=1,
            )
            self._pd_df = self._pd_df[mask]
            return
        self._filter_python(predicate, predicate_source)

    def _filter_python(self, predicate: Any, predicate_source: Optional[str]) -> None:
        rows_snapshot = self._snapshot_rows()
        filtered: List[Dict[str, Any]] = []
        for row in rows_snapshot:
            result = _evaluate_expression(
                predicate,
                row,
                self._context,
                rows_snapshot,
                self._frame_name,
                self._evaluate_expression,
                predicate_source,
            )
            if self._runtime_truthy(result):
                filtered.append(dict(row))
        self._rows = filtered
        self._engine = "python"

    def apply_select(self, columns: Sequence[str]) -> None:
        if not columns:
            return
        self._ensure_columns_exist(columns)
        if self._engine == "polars" and self._pl_df is not None:
            self._pl_df = self._pl_df.select(list(columns))
            return
        if self._engine == "pandas" and self._pd_df is not None:
            self._pd_df = self._pd_df.loc[:, list(columns)]
            return
        projected: List[Dict[str, Any]] = []
        for row in self._rows:
            projected.append({column: row.get(column) for column in columns})
        self._rows = projected
        self._engine = "python"

    def apply_order(self, columns: Sequence[str], descending: bool) -> None:
        if not columns:
            return
        self._ensure_columns_exist(columns)
        if self._engine == "polars" and self._pl_df is not None:
            self._pl_df = self._pl_df.sort(list(columns), descending=[descending] * len(columns))
            return
        if self._engine == "pandas" and self._pd_df is not None:
            self._pd_df = self._pd_df.sort_values(by=list(columns), ascending=not descending)
            return
        reverse = descending
        self._rows = sorted(self._rows, key=lambda item: tuple(item.get(column) for column in columns), reverse=reverse)

    def set_group_by(self, columns: Sequence[str]) -> None:
        self._ensure_columns_exist(columns)
        self._group_by = list(columns)

    def apply_join(
        self,
        join_rows: Sequence[Dict[str, Any]],
        join_on: Sequence[str],
        join_how: str,
        join_schema: Optional[Sequence[Dict[str, Any]]],
    ) -> None:
        join_keys = [column for column in join_on if column]
        if not join_keys:
            raise FramePipelineExecutionError("Join operation requires at least one key column")
        self._ensure_columns_exist(join_keys)
        right_schema_columns = [column.get("name") for column in join_schema or [] if column.get("name")]
        if not right_schema_columns and join_rows:
            sample = next((row for row in join_rows if isinstance(row, dict)), None)
            if isinstance(sample, dict):
                right_schema_columns = list(sample.keys())
        missing_right = [column for column in join_keys if column not in set(right_schema_columns)]
        if missing_right:
            raise FramePipelineExecutionError(
                f"Join target is missing columns: {', '.join(sorted(missing_right))}"
            )
        if self._engine == "polars" and self._pl_df is not None and pl is not None:
            right_df = pl.DataFrame(_clone_frame_rows(join_rows))
            how = (join_how or "inner").lower()
            if how == "right":
                result = right_df.join(self._pl_df, on=join_keys, how="left", suffix="_right")
            else:
                mapped = how if how in {"inner", "left", "outer"} else "inner"
                result = self._pl_df.join(right_df, on=join_keys, how=mapped, suffix="_right")
            for key in join_keys:
                dup = f"{key}_right"
                if dup in result.columns:
                    result = result.drop(dup)
            self._pl_df = result
            return
        if self._engine == "pandas" and self._pd_df is not None and pd is not None:
            right_df = pd.DataFrame(_clone_frame_rows(join_rows))
            how = (join_how or "inner").lower()
            if how not in {"inner", "left", "right", "outer"}:
                how = "inner"
            merged = self._pd_df.merge(right_df, how=how, on=join_keys, suffixes=("", "_right"))
            for key in join_keys:
                dup = f"{key}_right"
                if dup in merged.columns:
                    merged = merged.drop(columns=[dup])
            self._pd_df = merged
            return
        self._join_python(join_rows, join_keys, join_how, join_schema)

    def _join_python(
        self,
        join_rows: Sequence[Dict[str, Any]],
        join_on: Sequence[str],
        join_how: str,
        join_schema: Optional[Sequence[Dict[str, Any]]],
    ) -> None:
        left_rows = self._snapshot_rows()
        right_rows = _clone_frame_rows(join_rows)
        right_columns = [col.get("name") for col in join_schema or [] if col.get("name")]
        right_only = [column for column in right_columns if column not in join_on]
        index: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        for row in right_rows:
            key = tuple(row.get(column) for column in join_on)
            index.setdefault(key, []).append(row)
        results: List[Dict[str, Any]] = []
        matched_keys: set[Tuple[Any, ...]] = set()
        how = (join_how or "inner").lower()
        for left in left_rows:
            key = tuple(left.get(column) for column in join_on)
            matches = index.get(key)
            if matches:
                matched_keys.add(key)
                for match in matches:
                    combined = dict(left)
                    for column in right_only:
                        combined[column] = match.get(column)
                    results.append(combined)
            elif how in {"left", "outer"}:
                combined = dict(left)
                for column in right_only:
                    combined[column] = None
                results.append(combined)
        if how in {"right", "outer"}:
            left_columns = list(left_rows[0].keys()) if left_rows else list(join_on)
            for key, matches in index.items():
                if key in matched_keys:
                    continue
                for match in matches:
                    combined = {column: None for column in left_columns}
                    for idx, column in enumerate(join_on):
                        combined[column] = key[idx]
                    for column in right_only:
                        combined[column] = match.get(column)
                    results.append(combined)
        self._rows = results
        self._engine = "python"

    def apply_summarise(self, aggregations: Sequence[Dict[str, Any]]) -> None:
        if not aggregations:
            return
        if self._engine == "polars" and self._pl_df is not None and pl is not None:
            try:
                exprs = [_build_polars_aggregation(agg) for agg in aggregations]
                if self._group_by:
                    self._pl_df = self._pl_df.groupby(self._group_by).agg(exprs)
                else:
                    self._pl_df = self._pl_df.select(exprs)
                self._group_by = []
                return
            except FramePipelineExecutionError:
                self._fallback_to_python()
        if self._engine == "pandas" and self._pd_df is not None and pd is not None:
            self._rows = self._pd_df.to_dict(orient="records")
            self._pd_df = None
            self._engine = "python"
        self._aggregate_python(aggregations)

    def _aggregate_python(self, aggregations: Sequence[Dict[str, Any]]) -> None:
        rows = self._snapshot_rows()
        groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        if self._group_by:
            for row in rows:
                key = tuple(row.get(column) for column in self._group_by)
                groups.setdefault(key, []).append(row)
        else:
            groups[tuple()] = rows
        results: List[Dict[str, Any]] = []
        for key, items in groups.items():
            base: Dict[str, Any] = {}
            for idx, column in enumerate(self._group_by):
                base[column] = key[idx]
            for aggregation in aggregations:
                func = str(aggregation.get("function") or "").lower()
                expr = aggregation.get("expression")
                expr_source = aggregation.get("expression_source")
                values: List[Any] = []
                if expr is not None:
                    for row in items:
                        values.append(
                            _evaluate_expression(
                                expr,
                                row,
                                self._context,
                                items,
                                self._frame_name,
                                self._evaluate_expression,
                                expr_source,
                            )
                        )
                if func == "count":
                    if expr is None:
                        base[aggregation["name"]] = len(items)
                    else:
                        base[aggregation["name"]] = sum(1 for value in values if self._runtime_truthy(value))
                    continue
                numeric = []
                for value in values:
                    try:
                        numeric.append(float(value))
                    except (TypeError, ValueError):
                        continue
                if func == "sum":
                    base[aggregation["name"]] = sum(numeric)
                elif func in {"avg", "mean"}:
                    base[aggregation["name"]] = sum(numeric) / len(numeric) if numeric else 0.0
                elif func == "min":
                    base[aggregation["name"]] = min(numeric) if numeric else None
                elif func == "max":
                    base[aggregation["name"]] = max(numeric) if numeric else None
                else:
                    raise FramePipelineExecutionError(f"Unsupported aggregation '{func}'")
            results.append(base)
        self._rows = results
        self._engine = "python"
        self._group_by = []

    def finalize(self, column_order: Sequence[str]) -> List[Dict[str, Any]]:
        rows = self._snapshot_rows()
        if not column_order:
            return rows
        ordered: List[Dict[str, Any]] = []
        for row in rows:
            ordered_row: Dict[str, Any] = {}
            for column in column_order:
                ordered_row[column] = row.get(column)
            ordered.append(ordered_row)
        return ordered


def _build_polars_expression(spec: Any) -> "pl.Expr":
    if pl is None:
        raise FramePipelineExecutionError("Polars is not available for expression execution")
    if spec is None:
        raise FramePipelineExecutionError("Expression payload is missing")
    if isinstance(spec, dict):
        etype = spec.get("type")
        if etype == "literal":
            return pl.lit(spec.get("value"))
        if etype == "name":
            name = spec.get("name")
            if not name:
                raise FramePipelineExecutionError("Column reference is missing name")
            return pl.col(name)
        if etype == "binary":
            left = _build_polars_expression(spec.get("left"))
            right = _build_polars_expression(spec.get("right"))
            op = str(spec.get("op") or "").lower()
            if op in {"and", "&&"}:
                return left & right
            if op in {"or", "||"}:
                return left | right
            if op in {"==", "="}:
                return left == right
            if op in {"!=", "<>"}:
                return left != right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right
            if op == "%":
                return left % right
            raise FramePipelineExecutionError(f"Unsupported operator '{op}' in frame expression")
        if etype == "unary":
            operand = _build_polars_expression(spec.get("operand"))
            op = str(spec.get("op") or "").lower()
            if op in {"-", "neg"}:
                return -operand
            if op in {"+", "pos"}:
                return +operand
            if op in {"not", "!"}:
                return ~operand
            raise FramePipelineExecutionError(f"Unsupported unary operator '{op}'")
    raise FramePipelineExecutionError("Expression cannot be converted to Polars")


def _build_polars_aggregation(aggregation: Dict[str, Any]) -> "pl.Expr":
    if pl is None:
        raise FramePipelineExecutionError("Polars is not available for aggregation")
    func = str(aggregation.get("function") or "").lower()
    name = aggregation.get("name") or func
    expr_spec = aggregation.get("expression")
    if func == "count":
        if expr_spec is None:
            return pl.len().alias(name)
        raise FramePipelineExecutionError("count(expression) requires python fallback")
    if expr_spec is None:
        raise FramePipelineExecutionError(f"Aggregation '{func}' requires an expression")
    expr = _build_polars_expression(expr_spec)
    if func == "sum":
        return expr.sum().alias(name)
    if func in {"avg", "mean"}:
        return expr.mean().alias(name)
    if func == "min":
        return expr.min().alias(name)
    if func == "max":
        return expr.max().alias(name)
    raise FramePipelineExecutionError(f"Unsupported aggregation '{func}'")


def execute_frame_pipeline_plan(
    frame_name: str,
    plan_schema: Dict[str, Any],
    base_rows: Sequence[Dict[str, Any]],
    operations: Sequence[Dict[str, Any]],
    *,
    context: Dict[str, Any],
    evaluate_expression: RuntimeExpressionEvaluator,
    runtime_truthy: RuntimeTruthiness,
) -> List[Dict[str, Any]]:
    engine = _FramePipelineEngine(
        frame_name,
        base_rows,
        context,
        evaluate_expression=evaluate_expression,
        runtime_truthy=runtime_truthy,
    )
    for operation in operations:
        op = str(operation.get("op") or "").lower()
        if op == "filter":
            engine.apply_filter(operation.get("predicate"), operation.get("predicate_source"))
        elif op == "select":
            engine.apply_select(operation.get("columns") or [])
        elif op == "order_by":
            engine.apply_order(operation.get("columns") or [], bool(operation.get("descending")))
        elif op == "group_by":
            engine.set_group_by(operation.get("columns") or [])
        elif op == "summarise":
            engine.apply_summarise(operation.get("aggregations") or [])
        elif op == "join":
            engine.apply_join(
                operation.get("join_rows") or [],
                operation.get("join_on") or [],
                operation.get("join_how") or "inner",
                operation.get("join_schema"),
            )
    column_order = [column.get("name") for column in plan_schema.get("columns") or [] if column.get("name")]
    return engine.finalize(column_order)


def build_pipeline_frame_spec(
    plan_schema: Dict[str, Any],
    template_spec: Optional[Dict[str, Any]],
    frame_name: str,
) -> Dict[str, Any]:
    columns: List[Dict[str, Any]] = []
    for column in plan_schema.get("columns") or []:
        name = column.get("name")
        if not name:
            continue
        columns.append(
            {
                "name": name,
                "dtype": column.get("dtype"),
                "nullable": bool(column.get("nullable", True)),
                "role": column.get("role"),
            }
        )
    spec: Dict[str, Any] = {
        "name": frame_name,
        "source_type": (template_spec or {}).get("source_type") or "frame",
        "source": frame_name,
        "description": (template_spec or {}).get("description"),
        "columns": columns,
        "indexes": [],
        "relationships": [],
        "constraints": [],
        "access": (template_spec or {}).get("access"),
        "tags": list((template_spec or {}).get("tags") or []),
        "metadata": dict((template_spec or {}).get("metadata") or {}),
        "examples": [],
        "options": dict((template_spec or {}).get("options") or {}),
        "key": list(plan_schema.get("key") or (template_spec or {}).get("key") or []),
        "splits": dict(plan_schema.get("splits") or (template_spec or {}).get("splits") or {}),
    }
    return spec


__all__ = [
    "N3Frame",
    "project_frame_rows",
    "DEFAULT_FRAME_LIMIT",
    "MAX_FRAME_LIMIT",
    "FramePipelineExecutionError",
    "execute_frame_pipeline_plan",
    "build_pipeline_frame_spec",
]
