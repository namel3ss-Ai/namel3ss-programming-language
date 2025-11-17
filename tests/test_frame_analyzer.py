from __future__ import annotations

import pytest

from namel3ss.ast import (
    BinaryOp,
    CallExpression,
    Frame,
    FrameColumn,
    FrameFilter,
    FrameGroupBy,
    FrameJoin,
    FrameRef,
    FrameSelect,
    FrameSummarise,
    Literal,
    NameRef,
)
from namel3ss.frames import FrameExpressionAnalyzer, FrameTypeError


def _build_analyzer() -> FrameExpressionAnalyzer:
    users = Frame(
        name="users",
        columns=[
            FrameColumn(name="id", dtype="int", nullable=False),
            FrameColumn(name="country", dtype="string"),
            FrameColumn(name="revenue", dtype="number"),
            FrameColumn(name="active", dtype="bool"),
        ],
    )
    regions = Frame(
        name="regions",
        columns=[
            FrameColumn(name="country", dtype="string"),
            FrameColumn(name="region_name", dtype="string"),
        ],
    )
    return FrameExpressionAnalyzer([users, regions])


def test_analyzer_selects_columns() -> None:
    analyzer = _build_analyzer()
    expr = FrameSelect(source=FrameRef("users"), columns=["id", "country"])
    plan = analyzer.analyze(expr)
    assert plan.schema.order == ["id", "country"]
    payload = plan.to_payload(lambda e: {"type": "expr"} if e else None, lambda e: "expr" if e else None)
    assert payload["root"] == "users"
    assert payload["operations"][0]["op"] == "select"


def test_analyzer_validates_missing_column() -> None:
    analyzer = _build_analyzer()
    expr = FrameSelect(source=FrameRef("users"), columns=["unknown"])
    with pytest.raises(FrameTypeError):
        analyzer.analyze(expr)


def test_group_by_and_summarise_updates_schema() -> None:
    analyzer = _build_analyzer()
    sum_call = CallExpression(function=NameRef(name="sum"), arguments=[NameRef(name="revenue")])
    expr = FrameSummarise(
        source=FrameGroupBy(source=FrameRef("users"), columns=["country"]),
        aggregations={"total": sum_call},
    )
    plan = analyzer.analyze(expr)
    assert "total" in plan.schema.columns
    assert plan.schema.columns["total"].dtype == "number"
    payload = plan.to_payload(lambda e: None, lambda e: None)
    assert payload["operations"][-1]["op"] == "summarise"


def test_join_detects_duplicate_columns() -> None:
    analyzer = _build_analyzer()
    expr = FrameJoin(
        left=FrameRef("users"),
        right="regions",
        on=["country"],
        how="inner",
    )
    plan = analyzer.analyze(expr)
    assert plan.schema.columns["region_name"].dtype == "string"

    conflicting = FrameJoin(
        left=FrameRef("users"),
        right="users",
        on=["country"],
        how="inner",
    )
    with pytest.raises(FrameTypeError):
        analyzer.analyze(conflicting)


def test_filter_expression_is_validated() -> None:
    analyzer = _build_analyzer()
    predicate = BinaryOp(left=NameRef(name="missing"), op="==", right=Literal(value=True))
    with pytest.raises(FrameTypeError):
        analyzer.analyze(FrameFilter(source=FrameRef("users"), predicate=predicate))
