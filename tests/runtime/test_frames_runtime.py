from __future__ import annotations

from os import PathLike
from typing import Any, Dict, List

import pytest

from namel3ss.codegen.backend.core.runtime.expression_sandbox import evaluate_expression_tree
from namel3ss.codegen.backend.core.runtime.frames import (
    FramePipelineExecutionError,
    FrameSourceLoadError,
    execute_frame_pipeline_plan,
    load_frame_file_source,
    load_frame_sql_source,
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


def test_filter_in_and_select_with_alias_and_expression() -> None:
    rows = [
        {"id": 1, "age": 20, "country": "BE"},
        {"id": 2, "age": 35, "country": "US"},
        {"id": 3, "age": 28, "country": "NL"},
    ]
    predicate = _binary_expression(_name("country"), "in", _literal(["BE", "NL"]))
    operations = [
        {"op": "filter", "predicate": predicate},
        {
            "op": "select",
            "columns": [
                {"name": "user_id", "source": "id"},
                {
                    "name": "age_plus_one",
                    "expression": _binary_expression(_name("age"), "+", _literal(1)),
                    "expression_source": "age + 1",
                },
            ],
        },
        {"op": "order_by", "columns": ["user_id"]},
    ]
    schema = {
        "columns": [
            {"name": "user_id", "dtype": "int"},
            {"name": "age_plus_one", "dtype": "int"},
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
        {"user_id": 1, "age_plus_one": 21},
        {"user_id": 3, "age_plus_one": 29},
    ]


def test_order_by_multiple_columns_with_descending() -> None:
    rows = [
        {"id": 1, "country": "BE", "age": 30},
        {"id": 2, "country": "BE", "age": 40},
        {"id": 3, "country": "NL", "age": 25},
        {"id": 4, "country": "NL", "age": 22},
    ]
    operations = [
        {
            "op": "order_by",
            "columns": [
                {"name": "country"},
                {"name": "age", "descending": True},
            ],
        }
    ]
    schema = {
        "columns": [
            {"name": "id", "dtype": "int"},
            {"name": "country", "dtype": "string"},
            {"name": "age", "dtype": "int"},
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
    assert [row["id"] for row in result] == [2, 1, 3, 4]


def test_summarise_supports_nunique_and_std() -> None:
    rows = [
        {"country": "BE", "city": "Brussels", "amount": 10.0},
        {"country": "BE", "city": "Ghent", "amount": 20.0},
        {"country": "BE", "city": "Ghent", "amount": 30.0},
        {"country": "NL", "city": "Amsterdam", "amount": 15.0},
        {"country": "NL", "city": "Rotterdam", "amount": 25.0},
    ]
    operations = [
        {"op": "group_by", "columns": ["country"]},
        {
            "op": "summarise",
            "aggregations": [
                {
                    "name": "city_count",
                    "function": "nunique",
                    "dtype": "int",
                    "expression": _name("city"),
                    "expression_source": "city",
                },
                {
                    "name": "amount_std",
                    "function": "std",
                    "dtype": "number",
                    "expression": _name("amount"),
                    "expression_source": "amount",
                },
            ],
        },
    ]
    schema = {
        "columns": [
            {"name": "country", "dtype": "string"},
            {"name": "city_count", "dtype": "int"},
            {"name": "amount_std", "dtype": "number"},
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
    be = next(row for row in result if row["country"] == "BE")
    nl = next(row for row in result if row["country"] == "NL")
    assert be["city_count"] == 2
    assert nl["city_count"] == 2
    assert be["amount_std"] == pytest.approx(10.0)
    assert nl["amount_std"] == pytest.approx(7.0710678118654755)


def test_join_with_expression_keys_and_suffixes() -> None:
    rows = [
        {"id": 1, "country_code": "BE", "region": "EU"},
        {"id": 2, "country_code": "US", "region": "NA"},
    ]
    join_rows = [
        {"code": "BE", "name": "Belgium", "region": "Europe"},
        {"code": "US", "name": "United States", "region": "North America"},
    ]
    operations = [
        {
            "op": "join",
            "join_target": "countries",
            "join_on": [],
            "join_how": "inner",
            "join_rows": join_rows,
            "join_schema": [
                {"name": "code", "dtype": "string"},
                {"name": "name", "dtype": "string"},
                {"name": "region", "dtype": "string"},
            ],
            "join_expressions": [
                {
                    "left_expression": _name("country_code"),
                    "right_expression": _name("code"),
                    "left_expression_source": "country_code",
                    "right_expression_source": "code",
                }
            ],
        },
        {"op": "select", "columns": ["id", "country_code", "name", "region", "region_right"]},
    ]
    schema = {
        "columns": [
            {"name": "id", "dtype": "int"},
            {"name": "country_code", "dtype": "string"},
            {"name": "name", "dtype": "string"},
            {"name": "region", "dtype": "string"},
            {"name": "region_right", "dtype": "string"},
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
        {"id": 1, "country_code": "BE", "name": "Belgium", "region": "EU", "region_right": "Europe"},
        {"id": 2, "country_code": "US", "name": "United States", "region": "NA", "region_right": "North America"},
    ]


def test_polars_backend_basic_operations() -> None:
    pytest.importorskip("polars")
    from namel3ss.codegen.backend.core.runtime.frames import (
        _PolarsFrameBackend,
        _normalize_order_columns,
        _normalize_select_columns,
    )

    rows = [
        {"id": 1, "age": 30, "active": True},
        {"id": 2, "age": 25, "active": False},
    ]
    backend = _PolarsFrameBackend.from_rows(
        rows,
        frame_name="users",
        context={},
        evaluate_expression=_evaluate_expression,
        runtime_truthy=_truthy,
    )
    predicate = _binary_expression(_name("active"), "==", _literal(True))
    backend = backend.filter(predicate, None)
    select_spec = _normalize_select_columns(
        [
            {"name": "uid", "source": "id"},
            {
                "name": "age_plus_one",
                "expression": _binary_expression(_name("age"), "+", _literal(1)),
                "expression_source": "age + 1",
            },
        ]
    )
    backend = backend.select(select_spec)
    backend = backend.order_by(_normalize_order_columns(["uid"], False))
    result = backend.to_rows(["uid", "age_plus_one"])
    assert result == [{"uid": 1, "age_plus_one": 31}]


def test_execute_pipeline_with_polars_backend() -> None:
    pytest.importorskip("polars")
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


def test_load_frame_file_source_csv(tmp_path: "PathLike[str]") -> None:
    csv_path = tmp_path / "frame.csv"
    csv_path.write_text("id,value\n1,alpha\n2,beta\n", encoding="utf-8")
    config = {
        "kind": "file",
        "path": str(csv_path),
        "format": "csv",
    }
    rows = load_frame_file_source(
        config,
        context={},
        resolve_placeholders=lambda value, _: value,
    )
    assert [int(row["id"]) for row in rows] == [1, 2]
    assert [row["value"] for row in rows] == ["alpha", "beta"]


def test_load_frame_file_source_parquet(tmp_path: "PathLike[str]") -> None:
    pl = pytest.importorskip("polars")
    path = tmp_path / "frame.parquet"
    pl.DataFrame({"id": [1, 2], "value": ["x", "y"]}).write_parquet(path)
    config = {
        "kind": "file",
        "path": str(path),
        "format": "parquet",
    }
    rows = load_frame_file_source(
        config,
        context={},
        resolve_placeholders=lambda value, _: value,
    )
    assert rows == [{"id": 1, "value": "x"}, {"id": 2, "value": "y"}]


@pytest.mark.asyncio
async def test_load_frame_sql_source_with_connection(tmp_path: "PathLike[str]") -> None:
    try:
        import sqlalchemy as sa
    except Exception:
        pytest.skip("sqlalchemy is not available in this environment")
    db_path = tmp_path / "frames.db"
    engine = sa.create_engine(f"sqlite:///{db_path}")
    try:
        with engine.begin() as conn:
            conn.execute(sa.text("CREATE TABLE items (id INTEGER, name TEXT)"))
            conn.execute(
                sa.text("INSERT INTO items (id, name) VALUES (:id, :name)"),
                [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
            )
        config = {
            "kind": "sql",
            "connection": str(engine.url),
            "table": "items",
        }
        rows = await load_frame_sql_source(
            config,
            session=None,
            context={},
            resolve_placeholders=lambda value, _: value,
        )
        assert rows == [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
    finally:
        engine.dispose()
