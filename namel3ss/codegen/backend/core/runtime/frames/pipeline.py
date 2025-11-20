"""Frame pipeline execution engine and plan builders."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .constants import (
    pl,
    RuntimeExpressionEvaluator,
    RuntimeTruthiness,
    FramePipelineExecutionError,
)
from .backend_base import _BackendFallback
from .backend_polars import _PolarsFrameBackend
from .backend_pandas import _PandasFrameBackend
from .backend_python import _PythonFrameBackend
from .normalizers import _normalize_select_columns, _normalize_order_columns, _normalize_join_expressions


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


__all__ = [
    "_FramePipelineEngine",
    "_build_polars_expression",
    "_coerce_polars_in_rhs",
    "_build_polars_aggregation",
    "execute_frame_pipeline_plan",
    "build_pipeline_frame_spec",
]
