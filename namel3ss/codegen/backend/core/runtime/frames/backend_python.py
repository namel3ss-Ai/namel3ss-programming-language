"""Pure Python frame backend implementation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from .constants import FramePipelineExecutionError, RuntimeExpressionEvaluator, RuntimeTruthiness, RuntimeErrorRecorder
from .backend_base import _BackendFallback, _FrameBackendBase
from .utilities import _clone_frame_rows, _evaluate_expression


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




__all__ = ["_PythonFrameBackend"]
