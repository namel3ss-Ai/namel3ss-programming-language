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
from namel3ss.types import N3FrameType


def _build_analyzer_with_frames(
    extra_frames: list[Frame] | None = None,
) -> tuple[FrameExpressionAnalyzer, Frame, Frame]:
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
    frames = [users, regions]
    if extra_frames:
        frames.extend(extra_frames)
    analyzer = FrameExpressionAnalyzer(frames)
    return analyzer, users, regions


def _build_analyzer(extra_frames: list[Frame] | None = None) -> FrameExpressionAnalyzer:
    analyzer, _, _ = _build_analyzer_with_frames(extra_frames=extra_frames)
    return analyzer


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


def test_filter_type_mismatch_raises() -> None:
    analyzer = _build_analyzer()
    predicate = BinaryOp(left=NameRef(name="country"), op="==", right=Literal(value=True))
    with pytest.raises(FrameTypeError) as excinfo:
        analyzer.analyze(FrameFilter(source=FrameRef("users"), predicate=predicate))
    assert "filter" in str(excinfo.value).lower()


def test_aggregation_type_validation() -> None:
    analyzer = _build_analyzer()
    invalid_sum = CallExpression(function=NameRef(name="sum"), arguments=[NameRef(name="country")])
    expr = FrameSummarise(
        source=FrameGroupBy(source=FrameRef("users"), columns=["country"]),
        aggregations={"total": invalid_sum},
    )
    with pytest.raises(FrameTypeError) as excinfo:
        analyzer.analyze(expr)
    assert "aggregation 'total'" in str(excinfo.value).lower()


def test_join_type_mismatch_detected() -> None:
    numeric_regions = Frame(
        name="regions_numeric",
        columns=[
            FrameColumn(name="country", dtype="int"),
        ],
    )
    analyzer = _build_analyzer([numeric_regions])
    expr = FrameJoin(
        left=FrameRef("users"),
        right="regions_numeric",
        on=["country"],
        how="inner",
    )
    with pytest.raises(FrameTypeError) as excinfo:
        analyzer.analyze(expr)
    assert "join" in str(excinfo.value).lower()


def test_frame_definition_produces_n3_type() -> None:
    _, users, _ = _build_analyzer_with_frames()
    assert isinstance(users.type_info, N3FrameType)
    assert [name for name, _ in users.type_info.describe()] == ["id", "country", "revenue", "active"]
    assert users.type_info.columns["id"].dtype == "int"


def test_frame_type_propagates_through_select() -> None:
    analyzer = _build_analyzer()
    expr = FrameSelect(source=FrameRef("users"), columns=["country", "id"])
    plan = analyzer.analyze(expr)
    assert isinstance(plan.schema, N3FrameType)
    assert plan.schema.order == ["country", "id"]
    assert plan.schema.columns["country"].dtype == "string"
