"""Pandas-based frame backend implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .constants import pd, FramePipelineExecutionError, RuntimeExpressionEvaluator, RuntimeTruthiness
from .backend_base import _BackendFallback, _FrameBackendBase
from .utilities import _clone_frame_rows


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




__all__ = ["_PandasFrameBackend"]
