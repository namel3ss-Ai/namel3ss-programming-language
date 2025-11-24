from __future__ import annotations

from typing import Any, Dict

import pytest

from namel3ss.ast.base import BinaryOp, ContextValue, Literal, NameRef
from namel3ss.ast.datasets import (
    ComputedColumnOp,
    Dataset,
    FilterOp,
    GroupByOp,
    JoinOp,
    OrderByOp,
    PaginationPolicy,
    WindowFrame,
    WindowOp,
)

from namel3ss.codegen.backend.core import compile_dataset_to_sql


class _ContextStub(dict):
    def __init__(self, env: Dict[str, Any] | None = None, ctx: Dict[str, Any] | None = None) -> None:
        super().__init__()
        self._env = env or {}
        self._ctx = ctx or {}
        self["pagination"] = {}

    def get_env(self, name: str | None) -> Any:
        if name is None:
            return None
        return self._env.get(name)

    def get_ctx(self, path: list[str]) -> Any:
        current: Any = self._ctx
        for part in path:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = None
            if current is None:
                break
        return current


def test_compile_dataset_with_computed_columns_and_filters() -> None:
    dataset = Dataset(
        name="orders",
        source_type="table",
        source="public.sales",
        operations=[
            ComputedColumnOp(
                name="total",
                expression=BinaryOp(NameRef("subtotal"), "+", NameRef("tax")),
            ),
            FilterOp(
                BinaryOp(
                    BinaryOp(NameRef("subtotal"), "+", NameRef("tax")),
                    ">",
                    Literal(100),
                )
            ),
            FilterOp(BinaryOp(NameRef("status"), "==", Literal("PAID"))),
            GroupByOp(columns=["region"]),
            OrderByOp(columns=["region ASC"]),
        ],
        pagination=PaginationPolicy(enabled=True, page_size=50, max_pages=None),
    )

    context = _ContextStub()
    result = compile_dataset_to_sql(dataset, metadata={}, context=context)

    assert result["status"] == "ok"
    assert result["tables"] == ['"public"."sales"']
    assert 'SELECT t0.*' in result["sql"]
    assert 'AS "total"' in result["sql"]
    assert 'GROUP BY "region"' in result["sql"]
    assert 'ORDER BY "region" ASC' in result["sql"]
    assert 'LIMIT :p3' in result["sql"]
    assert result["params"] == {"p1": 100, "p2": "PAID", "p3": 50, "p4": 0}
    assert "total" in result["columns"]


def test_compile_dataset_resolves_context_values_and_window_frame() -> None:
    dataset = Dataset(
        name="transactions",
        source_type="table",
        source="ledger",
        operations=[
            FilterOp(
                BinaryOp(
                    NameRef("account_id"),
                    "==",
                    ContextValue(scope="ctx", path=["user", "id"], default=None),
                )
            ),
            WindowOp(
                name="rolling_7",
                function="avg",
                target="amount",
                order_by=["timestamp"],
                frame=WindowFrame(interval_value=7, interval_unit="weeks"),
            ),
        ],
    )

    context = _ContextStub(ctx={"user": {"id": 42}})
    result = compile_dataset_to_sql(dataset, metadata={}, context=context)

    assert result["params"]["p1"] == 42
    assert any("warning: unsupported window frame unit 'weeks'" in note for note in result["notes"])
    assert result["status"] == "partial"


def test_compile_dataset_handles_non_sql_source() -> None:
    dataset = Dataset(name="events", source_type="rest", source="events")

    result = compile_dataset_to_sql(dataset, metadata={}, context=_ContextStub())

    assert result["sql"] is None
    assert result["status"] == "partial"
    assert result["notes"] == ["info: non-sql dataset source; skipping SQL compilation"]


def test_compile_dataset_with_join_and_metadata_mapping() -> None:
    join = JoinOp(
        target_type="dataset",
        target_name="regions",
        condition=BinaryOp(NameRef("region_id"), "==", NameRef("regions.id")),
        join_type="left",
    )
    dataset = Dataset(
        name="sales_by_region",
        source_type="table",
        source="sales",
        operations=[join],
    )
    metadata = {"dataset_tables": {"regions": "dim.regions"}}

    result = compile_dataset_to_sql(dataset, metadata=metadata, context=_ContextStub())

    assert any('LEFT JOIN "dim"."regions"' in line for line in result["sql"].splitlines())
    assert result["tables"] == ['"sales"', '"dim"."regions"']
    assert result["status"] == "ok"