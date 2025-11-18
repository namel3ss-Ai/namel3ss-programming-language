from __future__ import annotations

from typing import Any, Dict, List

import pytest

from namel3ss.codegen.backend.core.runtime.expression_sandbox import evaluate_expression_tree
from namel3ss.codegen.backend.core.runtime.frames import (
    FramePipelineExecutionError,
    execute_frame_pipeline_plan,
)


def _evaluate_expression(
    expression: Any,
    row: Dict[str, Any],
    context: Dict[str, Any],
    rows: List[Dict[str, Any]] | None,
    frame_name: str,
    expression_source: str | None = None,
) -> Any:
    scope = dict(row)
    scope.update({"row": row, "rows": rows or [], "context": context})
    return evaluate_expression_tree(expression, scope, context)


def _truthy(value: Any) -> bool:
    return bool(value)


def _binary_expression(left: Dict[str, Any], op: str, right: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "binary", "op": op, "left": left, "right": right}


def _name(name: str) -> Dict[str, Any]:
    return {"type": "name", "name": name}


def _literal(value: Any) -> Dict[str, Any]:
    return {"type": "literal", "value": value}


def test_execute_pipeline_filter_select_order() -> None:
    rows = [
        {"id": 1, "age": 34, "country": "BE", "active": True},
        {"id": 2, "age": 29, "country": "FR", "active": True},
        {"id": 3, "age": 45, "country": "BE", "active": False},
    ]
    predicate = _binary_expression(
        _binary_expression(_name("active"), "==", _literal(True)),
        "and",
        _binary_expression(_name("country"), "==", _literal("BE")),
    )
    operations = [
        {"op": "filter", "predicate": predicate},
        {"op": "select", "columns": ["id", "age", "country"]},
        {"op": "order_by", "columns": ["age"], "descending": True},
    ]
    schema = {
        "columns": [
            {"name": "id", "dtype": "int", "nullable": False},
            {"name": "age", "dtype": "int", "nullable": False},
            {"name": "country", "dtype": "string", "nullable": False},
        ]
    }
    result = execute_frame_pipeline_plan(
        "users",
        schema,
        rows,
        operations,
        context={},
        evaluate_expression=_evaluate_expression,
        runtime_truthy=_truthy,
    )
    assert result == [{"id": 1, "age": 34, "country": "BE"}]


def test_execute_pipeline_groupby_summarise() -> None:
    rows = [
        {"country": "BE", "amount": 120.0},
        {"country": "BE", "amount": 80.0},
        {"country": "FR", "amount": 200.0},
    ]
    operations = [
        {"op": "group_by", "columns": ["country"]},
        {
            "op": "summarise",
            "aggregations": [
                {
                    "name": "total_revenue",
                    "function": "sum",
                    "dtype": "number",
                    "expression": _name("amount"),
                    "expression_source": "amount",
                },
                {
                    "name": "avg_revenue",
                    "function": "mean",
                    "dtype": "number",
                    "expression": _name("amount"),
                    "expression_source": "amount",
                },
                {
                    "name": "order_count",
                    "function": "count",
                    "dtype": "int",
                    "expression": None,
                },
            ],
        },
    ]
    schema = {
        "columns": [
            {"name": "country", "dtype": "string", "nullable": False},
            {"name": "total_revenue", "dtype": "number", "nullable": True},
            {"name": "avg_revenue", "dtype": "number", "nullable": True},
            {"name": "order_count", "dtype": "int", "nullable": True},
        ]
    }
    result = execute_frame_pipeline_plan(
        "orders",
        schema,
        rows,
        operations,
        context={},
        evaluate_expression=_evaluate_expression,
        runtime_truthy=_truthy,
    )
    ordered = sorted(result, key=lambda row: row["country"])
    assert ordered == [
        {"country": "BE", "total_revenue": 200.0, "avg_revenue": 100.0, "order_count": 2},
        {"country": "FR", "total_revenue": 200.0, "avg_revenue": 200.0, "order_count": 1},
    ]


def test_execute_pipeline_join_merges_columns() -> None:
    rows = [
        {"id": 1, "country": "BE"},
        {"id": 2, "country": "FR"},
    ]
    join_rows = [
        {"country": "BE", "region": "EU"},
        {"country": "FR", "region": "EU"},
    ]
    operations = [
        {
            "op": "join",
            "join_target": "regions",
            "join_on": ["country"],
            "join_how": "left",
            "join_rows": join_rows,
            "join_schema": [
                {"name": "country", "dtype": "string"},
                {"name": "region", "dtype": "string"},
            ],
        },
        {"op": "select", "columns": ["id", "country", "region"]},
    ]
    schema = {
        "columns": [
            {"name": "id", "dtype": "int", "nullable": False},
            {"name": "country", "dtype": "string", "nullable": False},
            {"name": "region", "dtype": "string", "nullable": True},
        ]
    }
    result = execute_frame_pipeline_plan(
        "users",
        schema,
        rows,
        operations,
        context={},
        evaluate_expression=_evaluate_expression,
        runtime_truthy=_truthy,
    )
    assert result == [
        {"id": 1, "country": "BE", "region": "EU"},
        {"id": 2, "country": "FR", "region": "EU"},
    ]


def test_select_missing_column_raises() -> None:
    rows = [{"id": 1}]
    schema = {"columns": [{"name": "id", "dtype": "int"}]}
    operations = [{"op": "select", "columns": ["missing"]}]
    with pytest.raises(FramePipelineExecutionError):
        execute_frame_pipeline_plan(
            "users",
            schema,
            rows,
            operations,
            context={},
            evaluate_expression=_evaluate_expression,
            runtime_truthy=_truthy,
        )


def test_join_missing_key_raises() -> None:
    rows = [{"id": 1, "country": "BE"}]
    schema = {
        "columns": [
            {"name": "id", "dtype": "int"},
            {"name": "country", "dtype": "string"},
        ]
    }
    operations = [
        {
            "op": "join",
            "join_target": "regions",
            "join_on": ["country"],
            "join_how": "inner",
            "join_rows": [{"code": "BE"}],
            "join_schema": [{"name": "code"}],
        }
    ]
    with pytest.raises(FramePipelineExecutionError):
        execute_frame_pipeline_plan(
            "users",
            schema,
            rows,
            operations,
            context={},
            evaluate_expression=_evaluate_expression,
            runtime_truthy=_truthy,
        )
