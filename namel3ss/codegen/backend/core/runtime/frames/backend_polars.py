"""Polars-based frame backend implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .constants import pl, FramePipelineExecutionError, RuntimeExpressionEvaluator, RuntimeTruthiness
from .backend_base import _BackendFallback, _FrameBackendBase
from .utilities import _clone_frame_rows

# Forward declarations for functions defined later
def _build_polars_expression(spec: Any) -> "pl.Expr":
    """Build Polars expression - implementation in pipeline module."""
    from .pipeline import _build_polars_expression as impl
    return impl(spec)

def _build_polars_aggregation(aggregation: Dict[str, Any]) -> "pl.Expr":
    """Build Polars aggregation - implementation in pipeline module."""
    from .pipeline import _build_polars_aggregation as impl
    return impl(aggregation)


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




__all__ = ["_PolarsFrameBackend"]
