"""Reusable runtime helpers for N3Frame execution."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import inspect
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


class FrameSourceLoadError(RuntimeError):
    """Raised when external frame source loading fails."""


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


_POLARS_TYPE_MAP: Dict[str, str] = {
    "string": "Utf8",
    "text": "Utf8",
    "int": "Int64",
    "integer": "Int64",
    "number": "Float64",
    "float": "Float64",
    "decimal": "Float64",
    "bool": "Boolean",
    "boolean": "Boolean",
    "datetime": "Datetime",
    "timestamp": "Datetime",
    "date": "Date",
    "time": "Time",
}


def _resolve_polars_dtype(dtype_name: Optional[str]) -> Optional["pl.datatypes.DataType"]:
    if pl is None or not dtype_name:
        return None
    mapped = _POLARS_TYPE_MAP.get(str(dtype_name).lower())
    if mapped is None:
        return None
    return getattr(pl, mapped, None)


class _BackendFallback(Exception):
    """Raised internally when a backend requests a downgrade path."""


class _FrameBackendBase:
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
        raise NotImplementedError()

    def filter(self, predicate: Any, predicate_source: Optional[str]) -> "_FrameBackendBase":
        raise NotImplementedError()

    def select(self, columns: Sequence[Dict[str, Any]]) -> "_FrameBackendBase":
        raise NotImplementedError()

    def order_by(self, columns: Sequence[Dict[str, Any]]) -> "_FrameBackendBase":
        raise NotImplementedError()

    def join(
        self,
        join_rows: Sequence[Dict[str, Any]],
        join_on: Sequence[str],
        join_how: str,
        join_schema: Optional[Sequence[Dict[str, Any]]],
        join_expressions: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> "_FrameBackendBase":
        raise NotImplementedError()

    def summarise(
        self,
        aggregations: Sequence[Dict[str, Any]],
        group_by: Sequence[str],
    ) -> "_FrameBackendBase":
        raise NotImplementedError()

    def to_rows(self, column_order: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError()

    def fallback(self) -> "_FrameBackendBase":
        raise _BackendFallback("No fallback available")


class _PolarsFrameBackend(_FrameBackendBase):
    def __init__(
        self,
        df: "pl.DataFrame",
        *,
        frame_name: str,
        context: Dict[str, Any],
        evaluate_expression: RuntimeExpressionEvaluator,
        runtime_truthy: RuntimeTruthiness,
    ) -> None:
        super().__init__(
            frame_name=frame_name,
            context=context,
            evaluate_expression=evaluate_expression,
            runtime_truthy=runtime_truthy,
        )
        self._df = df

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Dict[str, Any]],
        *,
        frame_name: str,
        context: Dict[str, Any],
        evaluate_expression: RuntimeExpressionEvaluator,
        runtime_truthy: RuntimeTruthiness,
    ) -> "_PolarsFrameBackend":
        if pl is None:
            raise _BackendFallback("Polars is not available")
        try:
            df = pl.DataFrame(_clone_frame_rows(rows))
        except Exception as exc:
            raise _BackendFallback("Failed to initialise Polars backend") from exc
        return cls(
            df,
            frame_name=frame_name,
            context=context,
            evaluate_expression=evaluate_expression,
            runtime_truthy=runtime_truthy,
        )

    def columns(self) -> List[str]:
        return list(self._df.columns)

    def filter(self, predicate: Any, predicate_source: Optional[str]) -> "_PolarsFrameBackend":
        if predicate is None:
            return self
        if pl is None:
            raise _BackendFallback("Polars is not available")
        expr = _build_polars_expression(predicate)
        try:
            filtered = self._df.filter(expr)
        except Exception as exc:
            raise _BackendFallback("Polars filter failed") from exc
        return _PolarsFrameBackend(
            filtered,
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def select(self, columns: Sequence[Dict[str, Any]]) -> "_PolarsFrameBackend":
        if pl is None:
            raise _BackendFallback("Polars is not available")
        exprs: List["pl.Expr"] = []
        for column in columns:
            alias = column.get("alias")
            if not alias:
                raise FramePipelineExecutionError("Select column requires a name")
            expr_spec = column.get("expression")
            if expr_spec is None:
                source = column.get("source")
                if not source:
                    raise FramePipelineExecutionError("Select column requires a source when no expression is set")
                expr = pl.col(source)
            else:
                expr = _build_polars_expression(expr_spec)
            exprs.append(expr.alias(alias))
        try:
            selected = self._df.select(exprs)
        except Exception as exc:
            raise _BackendFallback("Polars select failed") from exc
        return _PolarsFrameBackend(
            selected,
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def order_by(self, columns: Sequence[Dict[str, Any]]) -> "_PolarsFrameBackend":
        if pl is None:
            raise _BackendFallback("Polars is not available")
        names = [spec["name"] for spec in columns]
        descending = [spec["descending"] for spec in columns]
        try:
            ordered = self._df.sort(names, descending=descending, maintain_order=True)
        except Exception as exc:
            raise _BackendFallback("Polars order_by failed") from exc
        return _PolarsFrameBackend(
            ordered,
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def join(
        self,
        join_rows: Sequence[Dict[str, Any]],
        join_on: Sequence[str],
        join_how: str,
        join_schema: Optional[Sequence[Dict[str, Any]]],
        join_expressions: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> "_PolarsFrameBackend":
        if join_expressions:
            raise _BackendFallback("Polars backend does not handle expression joins")
        if pl is None:
            raise _BackendFallback("Polars is not available")
        try:
            right_df = pl.DataFrame(_clone_frame_rows(join_rows))
        except Exception as exc:
            raise _BackendFallback("Failed to materialise join target for Polars") from exc
        how = (join_how or "inner").lower()
        try:
            if how == "right":
                result = right_df.join(self._df, on=list(join_on), how="left", suffix="_right")
            else:
                mapped = how if how in {"inner", "left", "outer"} else "inner"
                result = self._df.join(right_df, on=list(join_on), how=mapped, suffix="_right")
            for key in join_on:
                dup = f"{key}_right"
                if dup in result.columns:
                    result = result.drop(dup)
        except Exception as exc:
            raise _BackendFallback("Polars join failed") from exc
        return _PolarsFrameBackend(
            result,
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def summarise(
        self,
        aggregations: Sequence[Dict[str, Any]],
        group_by: Sequence[str],
    ) -> "_PolarsFrameBackend":
        if pl is None:
            raise _BackendFallback("Polars is not available for summarise")
        exprs = [_build_polars_aggregation(agg) for agg in aggregations]
        try:
            if group_by:
                summarised = self._df.groupby(list(group_by)).agg(exprs)
            else:
                summarised = self._df.select(exprs)
        except Exception as exc:
            raise _BackendFallback("Polars summarise failed") from exc
        return _PolarsFrameBackend(
            summarised,
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def to_rows(self, column_order: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        if column_order:
            return [dict(row) for row in self._df.select(list(column_order)).to_dicts()]
        return [dict(row) for row in self._df.to_dicts()]

    def fallback(self) -> "_FrameBackendBase":
        if pd is None:
            raise _BackendFallback("Pandas fallback is not available")
        return _PandasFrameBackend.from_rows(
            self.to_rows(),
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )


class _PandasFrameBackend(_FrameBackendBase):
    def __init__(
        self,
        df: "pd.DataFrame",
        *,
        frame_name: str,
        context: Dict[str, Any],
        evaluate_expression: RuntimeExpressionEvaluator,
        runtime_truthy: RuntimeTruthiness,
    ) -> None:
        super().__init__(
            frame_name=frame_name,
            context=context,
            evaluate_expression=evaluate_expression,
            runtime_truthy=runtime_truthy,
        )
        self._df = df

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Dict[str, Any]],
        *,
        frame_name: str,
        context: Dict[str, Any],
        evaluate_expression: RuntimeExpressionEvaluator,
        runtime_truthy: RuntimeTruthiness,
    ) -> "_PandasFrameBackend":
        if pd is None:
            raise _BackendFallback("Pandas is not available")
        try:
            df = pd.DataFrame(_clone_frame_rows(rows))
        except Exception as exc:
            raise _BackendFallback("Failed to initialise Pandas backend") from exc
        return cls(
            df,
            frame_name=frame_name,
            context=context,
            evaluate_expression=evaluate_expression,
            runtime_truthy=runtime_truthy,
        )

    def columns(self) -> List[str]:
        return list(self._df.columns)

    def filter(self, predicate: Any, predicate_source: Optional[str]) -> "_FrameBackendBase":
        if predicate is None:
            return self
        rows_snapshot = self._df.to_dict(orient="records")

        def _apply(row: "pd.Series") -> bool:
            result = _evaluate_expression(
                predicate,
                row.to_dict(),
                self._context,
                rows_snapshot,
                self._frame_name,
                self._evaluate_expression,
                predicate_source,
            )
            return bool(result)

        mask = self._df.apply(_apply, axis=1)
        filtered_df = self._df[mask]
        return _PandasFrameBackend(
            filtered_df,
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def select(self, columns: Sequence[Dict[str, Any]]) -> "_FrameBackendBase":
        requires_expression = any(column.get("expression") is not None for column in columns)
        if requires_expression:
            raise _BackendFallback("Computed selects require python backend")
        selection = [column.get("source") for column in columns]
        if any(source is None for source in selection):
            raise FramePipelineExecutionError("Select operation is missing column sources")
        projected = self._df.loc[:, selection]
        projected.columns = [column["alias"] for column in columns]
        return _PandasFrameBackend(
            projected,
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def order_by(self, columns: Sequence[Dict[str, Any]]) -> "_FrameBackendBase":
        ascending = [not spec["descending"] for spec in columns]
        ordered = self._df.sort_values(
            by=[spec["name"] for spec in columns],
            ascending=ascending,
            kind="mergesort",
        )
        return _PandasFrameBackend(
            ordered,
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def join(
        self,
        join_rows: Sequence[Dict[str, Any]],
        join_on: Sequence[str],
        join_how: str,
        join_schema: Optional[Sequence[Dict[str, Any]]],
        join_expressions: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> "_FrameBackendBase":
        if join_expressions:
            raise _BackendFallback("Expression joins require python backend")
        right_df = pd.DataFrame(_clone_frame_rows(join_rows))
        how = (join_how or "inner").lower()
        if how not in {"inner", "left", "right", "outer"}:
            how = "inner"
        merged = self._df.merge(right_df, how=how, on=list(join_on), suffixes=("", "_right"))
        for key in join_on:
            dup = f"{key}_right"
            if dup in merged.columns:
                merged = merged.drop(columns=[dup])
        return _PandasFrameBackend(
            merged,
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def summarise(
        self,
        aggregations: Sequence[Dict[str, Any]],
        group_by: Sequence[str],
    ) -> "_FrameBackendBase":
        raise _BackendFallback("Pandas summarise delegates to python backend")

    def to_rows(self, column_order: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        if column_order:
            df = self._df.loc[:, list(column_order)]
        else:
            df = self._df
        return df.to_dict(orient="records")

    def fallback(self) -> "_FrameBackendBase":
        return _PythonFrameBackend.from_rows(
            self.to_rows(),
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )


class _PythonFrameBackend(_FrameBackendBase):
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        *,
        frame_name: str,
        context: Dict[str, Any],
        evaluate_expression: RuntimeExpressionEvaluator,
        runtime_truthy: RuntimeTruthiness,
    ) -> None:
        super().__init__(
            frame_name=frame_name,
            context=context,
            evaluate_expression=evaluate_expression,
            runtime_truthy=runtime_truthy,
        )
        self._rows = [dict(row) for row in rows]

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Dict[str, Any]],
        *,
        frame_name: str,
        context: Dict[str, Any],
        evaluate_expression: RuntimeExpressionEvaluator,
        runtime_truthy: RuntimeTruthiness,
    ) -> "_PythonFrameBackend":
        return cls(
            rows,
            frame_name=frame_name,
            context=context,
            evaluate_expression=evaluate_expression,
            runtime_truthy=runtime_truthy,
        )

    def columns(self) -> List[str]:
        ordered: List[str] = []
        for row in self._rows:
            for key in row.keys():
                if key not in ordered:
                    ordered.append(key)
        return ordered

    def filter(self, predicate: Any, predicate_source: Optional[str]) -> "_FrameBackendBase":
        if predicate is None:
            return self
        rows_snapshot = [dict(row) for row in self._rows]
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
        return _PythonFrameBackend.from_rows(
            filtered,
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def select(self, columns: Sequence[Dict[str, Any]]) -> "_FrameBackendBase":
        rows = [dict(row) for row in self._rows]
        projected: List[Dict[str, Any]] = []
        for row in rows:
            evaluation_row = dict(row)
            shaped: Dict[str, Any] = {}
            for column in columns:
                alias = column.get("alias")
                if not alias:
                    raise FramePipelineExecutionError("Select column requires a name")
                expression = column.get("expression")
                if expression is not None:
                    value = _evaluate_expression(
                        expression,
                        evaluation_row,
                        self._context,
                        rows,
                        self._frame_name,
                        self._evaluate_expression,
                        column.get("expression_source"),
                    )
                else:
                    source = column.get("source")
                    if not source:
                        raise FramePipelineExecutionError("Select column requires a source when no expression is set")
                    value = row.get(source)
                shaped[alias] = value
                evaluation_row[alias] = value
            projected.append(shaped)
        return _PythonFrameBackend.from_rows(
            projected,
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def order_by(self, columns: Sequence[Dict[str, Any]]) -> "_FrameBackendBase":
        rows = [dict(row) for row in self._rows]
        for spec in reversed(columns):
            rows.sort(
                key=lambda item, column=spec["name"]: self._python_sort_key(item, column),
                reverse=spec["descending"],
            )
        return _PythonFrameBackend.from_rows(
            rows,
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def join(
        self,
        join_rows: Sequence[Dict[str, Any]],
        join_on: Sequence[str],
        join_how: str,
        join_schema: Optional[Sequence[Dict[str, Any]]],
        join_expressions: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> "_FrameBackendBase":
        if join_expressions:
            return self._join_with_expressions(join_rows, join_how, join_expressions, join_schema)
        right_columns = [col.get("name") for col in join_schema or [] if col.get("name")]
        right_only = [column for column in right_columns if column not in join_on]
        index: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        right_rows = _clone_frame_rows(join_rows)
        for row in right_rows:
            key = tuple(row.get(column) for column in join_on)
            index.setdefault(key, []).append(row)
        results: List[Dict[str, Any]] = []
        matched: set[Tuple[Any, ...]] = set()
        how = (join_how or "inner").lower()
        for left in self._rows:
            key = tuple(left.get(column) for column in join_on)
            matches = index.get(key)
            if matches:
                matched.add(key)
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
            left_columns = list(self._rows[0].keys()) if self._rows else list(join_on)
            for key, matches in index.items():
                if key in matched:
                    continue
                for match in matches:
                    combined = {column: None for column in left_columns}
                    for idx, column in enumerate(join_on):
                        combined[column] = key[idx]
                    for column in right_only:
                        combined[column] = match.get(column)
                    results.append(combined)
        return _PythonFrameBackend.from_rows(
            results,
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def summarise(
        self,
        aggregations: Sequence[Dict[str, Any]],
        group_by: Sequence[str],
    ) -> "_FrameBackendBase":
        rows = [dict(row) for row in self._rows]
        groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        if group_by:
            for row in rows:
                key = tuple(row.get(column) for column in group_by)
                groups.setdefault(key, []).append(row)
        else:
            groups[tuple()] = rows
        results: List[Dict[str, Any]] = []
        for key, items in groups.items():
            base: Dict[str, Any] = {}
            for idx, column in enumerate(group_by):
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
                elif func in {"nunique", "distinct"}:
                    base[aggregation["name"]] = self._count_unique(values)
                elif func in {"std", "stddev", "std_dev"}:
                    base[aggregation["name"]] = self._compute_stddev(numeric)
                else:
                    raise FramePipelineExecutionError(f"Unsupported aggregation '{func}'")
            results.append(base)
        return _PythonFrameBackend.from_rows(
            results,
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def to_rows(self, column_order: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        if not column_order:
            return [dict(row) for row in self._rows]
        ordered: List[Dict[str, Any]] = []
        for row in self._rows:
            ordered_row: Dict[str, Any] = {}
            for column in column_order:
                ordered_row[column] = row.get(column)
            ordered.append(ordered_row)
        return ordered

    def _python_sort_key(self, row: Dict[str, Any], column: str) -> Tuple[int, Any]:
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

    def _join_with_expressions(
        self,
        join_rows: Sequence[Dict[str, Any]],
        join_how: str,
        join_expressions: Sequence[Dict[str, Any]],
        join_schema: Optional[Sequence[Dict[str, Any]]],
    ) -> "_FrameBackendBase":
        right_rows = _clone_frame_rows(join_rows)
        right_columns = [col.get("name") for col in join_schema or [] if col.get("name")]
        rename_map = self._prepare_right_column_suffixes(right_columns, [])
        skip_columns: List[str] = []
        index: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        for row in right_rows:
            key = tuple(
                _evaluate_expression(
                    spec["right_expression"],
                    row,
                    self._context,
                    right_rows,
                    self._frame_name,
                    self._evaluate_expression,
                    spec.get("right_expression_source"),
                )
                for spec in join_expressions
            )
            index.setdefault(key, []).append(row)
        results: List[Dict[str, Any]] = []
        matched_keys: set[Tuple[Any, ...]] = set()
        how = (join_how or "inner").lower()
        for left in self._rows:
            left_key = tuple(
                _evaluate_expression(
                    spec["left_expression"],
                    left,
                    self._context,
                    self._rows,
                    self._frame_name,
                    self._evaluate_expression,
                    spec.get("left_expression_source"),
                )
                for spec in join_expressions
            )
            matches = index.get(left_key)
            if matches:
                matched_keys.add(left_key)
                for match in matches:
                    results.append(self._merge_join_rows(left, match, rename_map, skip_columns))
            elif how in {"left", "outer"}:
                results.append(self._merge_join_rows(left, None, rename_map, skip_columns))
        if how in {"right", "outer"}:
            left_columns = list(self._rows[0].keys()) if self._rows else []
            for key, matches in index.items():
                if key in matched_keys:
                    continue
                for match in matches:
                    placeholder = {column: None for column in left_columns}
                    for idx, spec in enumerate(join_expressions):
                        column_name = self._expression_column_name(spec["left_expression"])
                        if column_name and column_name in placeholder:
                            placeholder[column_name] = key[idx] if idx < len(key) else None
                    results.append(self._merge_join_rows(placeholder, match, rename_map, skip_columns))
        return _PythonFrameBackend.from_rows(
            results,
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def _merge_join_rows(
        self,
        left: Dict[str, Any],
        right: Optional[Dict[str, Any]],
        rename_map: Dict[str, str],
        skip_columns: Sequence[str],
    ) -> Dict[str, Any]:
        merged = dict(left)
        if right:
            skip_set = set(skip_columns)
            for column, value in right.items():
                if column in skip_set:
                    continue
                target = rename_map.get(column, column)
                merged[target] = value
        return merged

    def _prepare_right_column_suffixes(
        self,
        right_columns: Sequence[str],
        skip_columns: Sequence[str],
    ) -> Dict[str, str]:
        rename_map: Dict[str, str] = {}
        left_columns = set(self.columns())
        skip_set = set(skip_columns)
        for column in right_columns:
            if not column or column in skip_set:
                continue
            if column in left_columns:
                rename_map[column] = f"{column}_right"
        return rename_map

    def _expression_column_name(self, expression: Any) -> Optional[str]:
        if isinstance(expression, dict) and expression.get("type") == "name":
            name = expression.get("name")
            return str(name) if name else None
        return None

    def _count_unique(self, values: Sequence[Any]) -> int:
        seen: set[Any] = set()
        serialized: set[str] = set()
        for value in values:
            if value is None:
                continue
            try:
                seen.add(value)
            except TypeError:
                try:
                    serialized.add(json.dumps(value, sort_keys=True))
                except Exception:
                    serialized.add(str(value))
        return len(seen) + len(serialized)

    def _compute_stddev(self, values: Sequence[float]) -> Optional[float]:
        count = len(values)
        if count == 0:
            return None
        if count == 1:
            return 0.0
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        return math.sqrt(variance)


def _normalize_select_columns(columns: Sequence[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, column in enumerate(columns):
        if isinstance(column, str):
            normalized.append(
                {
                    "source": column,
                    "alias": column,
                    "expression": None,
                    "expression_source": None,
                }
            )
            continue
        if isinstance(column, dict):
            alias = column.get("alias") or column.get("name") or column.get("source")
            source = column.get("source") or column.get("name")
            expression = column.get("expression")
            normalized.append(
                {
                    "source": source,
                    "alias": alias or f"column_{idx}",
                    "expression": expression,
                    "expression_source": column.get("expression_source"),
                }
            )
    return normalized


def _normalize_order_columns(columns: Sequence[Any], default_descending: bool) -> List[Dict[str, Any]]:
    order_specs: List[Dict[str, Any]] = []
    for column in columns:
        name: Optional[str] = None
        descending_value: Optional[bool] = None
        if isinstance(column, dict):
            name = column.get("name") or column.get("column")
            if isinstance(column.get("descending"), bool):
                descending_value = bool(column.get("descending"))
            elif "descending" in column:
                descending_value = bool(column.get("descending"))
            elif "desc" in column:
                descending_value = bool(column.get("desc"))
        else:
            name = str(column)
        if not name:
            continue
        if descending_value is None:
            descending_value = bool(default_descending)
        order_specs.append({"name": name, "descending": descending_value})
    return order_specs


def _coerce_expression_spec(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return {"type": "name", "name": value}
    return {"type": "literal", "value": value}


def _normalize_join_expressions(join_expressions: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, expression in enumerate(join_expressions or []):
        entry: Optional[Dict[str, Any]] = None
        if isinstance(expression, dict):
            left_expr = _coerce_expression_spec(expression.get("left_expression") or expression.get("left"))
            right_expr = _coerce_expression_spec(expression.get("right_expression") or expression.get("right"))
            if left_expr is None or right_expr is None:
                continue
            entry = {
                "name": f"__n3_join_key_{idx}",
                "left_expression": left_expr,
                "right_expression": right_expr,
                "left_expression_source": expression.get("left_expression_source") or expression.get("left_source"),
                "right_expression_source": expression.get("right_expression_source") or expression.get("right_source"),
            }
        elif isinstance(expression, (list, tuple)) and len(expression) == 2:
            left_expr = _coerce_expression_spec(expression[0])
            right_expr = _coerce_expression_spec(expression[1])
            if left_expr is None or right_expr is None:
                continue
            entry = {
                "name": f"__n3_join_key_{idx}",
                "left_expression": left_expr,
                "right_expression": right_expr,
                "left_expression_source": None,
                "right_expression_source": None,
            }
        if entry:
            normalized.append(entry)
    return normalized


def _collect_right_column_names(
    join_rows: Sequence[Dict[str, Any]],
    join_schema: Optional[Sequence[Dict[str, Any]]],
) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for column in join_schema or []:
        name = column.get("name")
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    sample = next((row for row in join_rows if isinstance(row, dict)), None)
    if isinstance(sample, dict):
        for key in sample.keys():
            if key not in seen:
                names.append(key)
                seen.add(key)
    return names


def _resolve_frame_source_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    app_root = os.getenv("NAMEL3SS_APP_ROOT")
    if app_root:
        return Path(app_root) / path
    return Path.cwd() / path


def load_frame_file_source(
    source_config: Dict[str, Any],
    *,
    context: Dict[str, Any],
    resolve_placeholders: Callable[[Any, Dict[str, Any]], Any],
) -> List[Dict[str, Any]]:
    path_value = resolve_placeholders(source_config.get("path"), context) if source_config.get("path") is not None else source_config.get("path")
    if not path_value:
        raise FrameSourceLoadError("Frame file source requires a 'path' value.")
    if not isinstance(path_value, str):
        path_value = str(path_value)
    resolved_path = _resolve_frame_source_path(path_value)
    fmt = str(source_config.get("format") or "csv").lower()
    if fmt == "csv":
        return _read_csv_rows(resolved_path)
    if fmt == "parquet":
        return _read_parquet_rows(resolved_path)
    raise FrameSourceLoadError(f"Unsupported file format '{fmt}' for frame source.")


def _read_csv_rows(resolved_path: Path) -> List[Dict[str, Any]]:
    try:
        if pl is not None:
            frame = pl.read_csv(resolved_path)
            return [dict(row) for row in frame.to_dicts()]
        if pd is not None:
            df = pd.read_csv(resolved_path)
            return df.to_dict(orient="records")
        with resolved_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]
    except FileNotFoundError as exc:
        raise FrameSourceLoadError(f"CSV file '{resolved_path}' was not found.") from exc
    except Exception as exc:  # pragma: no cover - unexpected read failure
        raise FrameSourceLoadError(f"Failed to read CSV file '{resolved_path}': {exc}") from exc


def _read_parquet_rows(resolved_path: Path) -> List[Dict[str, Any]]:
    try:
        if pl is not None:
            frame = pl.read_parquet(resolved_path)
            return [dict(row) for row in frame.to_dicts()]
        if pd is not None:
            df = pd.read_parquet(resolved_path)
            return df.to_dict(orient="records")
        if pq is not None:
            table = pq.read_table(resolved_path)
            return table.to_pylist()
        raise FrameSourceLoadError("Parquet sources require polars, pandas, or pyarrow installed.")
    except FileNotFoundError as exc:
        raise FrameSourceLoadError(f"Parquet file '{resolved_path}' was not found.") from exc
    except Exception as exc:  # pragma: no cover - unexpected read failure
        raise FrameSourceLoadError(f"Failed to read Parquet file '{resolved_path}': {exc}") from exc


async def load_frame_sql_source(
    source_config: Dict[str, Any],
    session: Optional[Any],
    *,
    context: Dict[str, Any],
    resolve_placeholders: Callable[[Any, Dict[str, Any]], Any],
) -> List[Dict[str, Any]]:
    try:
        from sqlalchemy import text as _sql_text  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise FrameSourceLoadError("SQL frame sources require SQLAlchemy installed.") from exc

    connection_value = source_config.get("connection")
    query_value = source_config.get("query")
    table_value = source_config.get("table")
    resolved_query = resolve_placeholders(query_value, context) if query_value is not None else query_value
    if resolved_query is None:
        resolved_table = resolve_placeholders(table_value, context) if table_value is not None else table_value
        if not resolved_table:
            raise FrameSourceLoadError("SQL frame sources require a 'table' or explicit 'query'.")
        resolved_query = f"SELECT * FROM {resolved_table}"
    if not isinstance(resolved_query, str):
        resolved_query = str(resolved_query)
    resolved_connection = resolve_placeholders(connection_value, context) if connection_value is not None else None
    if resolved_connection:
        if not isinstance(resolved_connection, str):
            resolved_connection = str(resolved_connection)

        def _run_query() -> List[Dict[str, Any]]:
            try:
                from sqlalchemy import create_engine  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise FrameSourceLoadError("SQL frame sources require SQLAlchemy installed.") from exc
            engine = create_engine(resolved_connection)
            try:
                with engine.connect() as conn:
                    result = conn.execute(_sql_text(resolved_query))
                    return [dict(row) for row in result.mappings()]
            finally:
                engine.dispose()

        return await asyncio.to_thread(_run_query)
    if session is None:
        raise FrameSourceLoadError("SQL frame sources require a database session when no connection is provided.")
    statement = _sql_text(resolved_query)
    result = await _execute_sql_with_session(session, statement)
    return [dict(row) for row in result.mappings()]


async def _execute_sql_with_session(session: Any, statement: Any) -> Any:
    executor = getattr(session, "execute", None)
    if executor is None:
        raise FrameSourceLoadError("Database session does not expose an execute method.")
    maybe_result = executor(statement)
    if inspect.isawaitable(maybe_result):
        return await maybe_result
    return maybe_result


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
        self._backend = self._create_backend(rows)

    def _create_backend(self, rows: Sequence[Dict[str, Any]]) -> _FrameBackendBase:
        try:
            return _PolarsFrameBackend.from_rows(
                rows,
                frame_name=self._frame_name,
                context=self._context,
                evaluate_expression=self._evaluate_expression,
                runtime_truthy=self._runtime_truthy,
            )
        except _BackendFallback:
            pass
        try:
            return _PandasFrameBackend.from_rows(
                rows,
                frame_name=self._frame_name,
                context=self._context,
                evaluate_expression=self._evaluate_expression,
                runtime_truthy=self._runtime_truthy,
            )
        except _BackendFallback:
            pass
        return _PythonFrameBackend.from_rows(
            rows,
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def _ensure_columns_exist(self, columns: Sequence[str]) -> None:
        available = set(self._backend.columns())
        missing = [column for column in columns if column and column not in available]
        if missing:
            joined = ", ".join(sorted(missing))
            raise FramePipelineExecutionError(
                "Frame '%s' does not define columns: %s" % (self._frame_name, joined)
            )

    def _execute_with_backend(self, operator: Callable[[_FrameBackendBase], _FrameBackendBase]) -> None:
        backend = self._backend
        while True:
            try:
                self._backend = operator(backend)
                return
            except _BackendFallback:
                try:
                    backend = backend.fallback()
                except _BackendFallback as exc:
                    raise FramePipelineExecutionError(str(exc)) from exc

    def _force_python_backend(self) -> None:
        if isinstance(self._backend, _PythonFrameBackend):
            return
        self._backend = _PythonFrameBackend.from_rows(
            self._backend.to_rows(self._backend.columns()),
            frame_name=self._frame_name,
            context=self._context,
            evaluate_expression=self._evaluate_expression,
            runtime_truthy=self._runtime_truthy,
        )

    def apply_filter(self, predicate: Any, predicate_source: Optional[str]) -> None:
        if predicate is None:
            return
        self._execute_with_backend(lambda backend: backend.filter(predicate, predicate_source))

    def apply_select(self, columns: Sequence[Any]) -> None:
        normalized = _normalize_select_columns(columns)
        if not normalized:
            return
        base_columns = [column["source"] for column in normalized if column.get("source") and column.get("expression") is None]
        if base_columns:
            self._ensure_columns_exist(base_columns)
        self._execute_with_backend(lambda backend: backend.select(normalized))

    def apply_order(self, columns: Sequence[Any], descending: bool) -> None:
        order_specs = _normalize_order_columns(columns, descending)
        if not order_specs:
            return
        self._ensure_columns_exist([spec["name"] for spec in order_specs])
        self._execute_with_backend(lambda backend: backend.order_by(order_specs))

    def set_group_by(self, columns: Sequence[str]) -> None:
        self._ensure_columns_exist(columns)
        self._group_by = list(columns)

    def apply_join(
        self,
        join_rows: Sequence[Dict[str, Any]],
        join_on: Sequence[str],
        join_how: str,
        join_schema: Optional[Sequence[Dict[str, Any]]],
        join_expressions: Optional[Sequence[Any]] = None,
    ) -> None:
        expression_specs = _normalize_join_expressions(join_expressions)
        if expression_specs:
            self._force_python_backend()
            self._backend = self._backend.join(
                join_rows,
                join_on,
                join_how,
                join_schema,
                join_expressions=expression_specs,
            )
            return
        join_keys = [column for column in join_on if column]
        if not join_keys:
            raise FramePipelineExecutionError("Join operation requires at least one key column")
        self._ensure_columns_exist(join_keys)
        right_schema_columns = _collect_right_column_names(join_rows, join_schema)
        missing_right = [column for column in join_keys if column not in set(right_schema_columns)]
        if missing_right:
            raise FramePipelineExecutionError(
                "Join target is missing columns: %s" % ", ".join(sorted(missing_right))
            )
        self._execute_with_backend(
            lambda backend: backend.join(join_rows, join_keys, join_how, join_schema)
        )

    def apply_summarise(self, aggregations: Sequence[Dict[str, Any]]) -> None:
        if not aggregations:
            return
        self._execute_with_backend(
            lambda backend: backend.summarise(aggregations, self._group_by)
        )
        self._group_by = []

    def finalize(self, column_order: Sequence[str]) -> List[Dict[str, Any]]:
        rows = self._backend.to_rows(column_order)
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
            if op == "in":
                return left.is_in(_coerce_polars_in_rhs(spec.get("right"), right))
            if op in {"not in", "notin"}:
                return ~left.is_in(_coerce_polars_in_rhs(spec.get("right"), right))
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


def _coerce_polars_in_rhs(spec: Any, evaluated: Optional["pl.Expr"] = None) -> Any:
    if isinstance(spec, dict) and spec.get("type") == "literal":
        return spec.get("value")
    if evaluated is not None:
        return evaluated
    return _build_polars_expression(spec)


def _build_polars_aggregation(aggregation: Dict[str, Any]) -> "pl.Expr":
    if pl is None:
        raise FramePipelineExecutionError("Polars is not available for aggregation")
    func = str(aggregation.get("function") or "").lower()
    name = aggregation.get("name") or func
    expr_spec = aggregation.get("expression")
    if func == "count":
        if expr_spec is None:
            expr = pl.len().alias(name)
            dtype = _resolve_polars_dtype(aggregation.get("dtype"))
            return expr.cast(dtype) if dtype is not None else expr
        expr = _build_polars_expression(expr_spec)
        result = expr.count().alias(name)
        dtype = _resolve_polars_dtype(aggregation.get("dtype"))
        return result.cast(dtype) if dtype is not None else result
    if expr_spec is None:
        raise FramePipelineExecutionError(f"Aggregation '{func}' requires an expression")
    expr = _build_polars_expression(expr_spec)
    dtype = _resolve_polars_dtype(aggregation.get("dtype"))
    if func == "sum":
        result = expr.sum().alias(name)
    if func in {"avg", "mean"}:
        result = expr.mean().alias(name)
    if func == "min":
        result = expr.min().alias(name)
    if func == "max":
        result = expr.max().alias(name)
    if func in {"nunique", "distinct"}:
        result = expr.n_unique().alias(name)
    if func in {"std", "stddev", "std_dev"}:
        result = expr.std().alias(name)
    else:
        raise FramePipelineExecutionError(f"Unsupported aggregation '{func}'")
    return result.cast(dtype) if dtype is not None else result


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
            order_columns = operation.get("order") or operation.get("columns") or []
            engine.apply_order(order_columns, bool(operation.get("descending")))
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
                operation.get("join_expressions"),
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
    "FrameSourceLoadError",
    "load_frame_file_source",
    "load_frame_sql_source",
    "FramePipelineExecutionError",
    "execute_frame_pipeline_plan",
    "build_pipeline_frame_spec",
]
