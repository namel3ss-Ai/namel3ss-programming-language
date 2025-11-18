from __future__ import annotations

from namel3ss.types import (
    ANY_TYPE,
    DatasetType,
    ExpressionType,
    FrameTypeRef,
    PromptIOTypes,
    ScalarKind,
    ScalarType,
    derive_group_schema,
    derive_select_schema,
    infer_scalar_type,
    is_assignable,
    is_compatible,
    lookup_column_type,
    stringify_type,
)
from namel3ss.types.core import ListType, MapType
from namel3ss.types.frame import FrameColumnType, N3FrameType


def _build_users_frame() -> FrameTypeRef:
    schema = N3FrameType(
        columns={
            "id": FrameColumnType(name="id", dtype="int", nullable=False),
            "age": FrameColumnType(name="age", dtype="int", nullable=True),
            "name": FrameColumnType(name="name", dtype="string", nullable=False),
            "country": FrameColumnType(name="country", dtype="string", nullable=False),
        },
        order=["id", "age", "name", "country"],
        key=["id"],
        splits={},
    )
    return FrameTypeRef(schema=schema, label="users")


def test_scalar_stringify_and_assignability() -> None:
    int_type = ScalarType(ScalarKind.INT)
    float_type = ScalarType(ScalarKind.FLOAT)
    bool_type = ScalarType(ScalarKind.BOOL)

    assert stringify_type(int_type) == "int"
    assert stringify_type(ListType(int_type)) == "list[int]"
    assert is_assignable(int_type, float_type)
    assert is_assignable(bool_type, int_type)
    assert not is_assignable(float_type, int_type)
    assert is_assignable(int_type, ANY_TYPE)


def test_collection_and_map_assignability() -> None:
    list_of_int = ListType(ScalarType(ScalarKind.INT))
    list_of_float = ListType(ScalarType(ScalarKind.FLOAT))
    assert is_assignable(list_of_int, list_of_float)
    assert not is_assignable(list_of_float, list_of_int)

    map_string_any = MapType(ScalarType(ScalarKind.STRING), ANY_TYPE)
    map_string_int = MapType(ScalarType(ScalarKind.STRING), ScalarType(ScalarKind.INT))
    assert is_assignable(map_string_int, map_string_any)
    assert is_compatible(map_string_any, map_string_int)


def test_frame_schema_helpers() -> None:
    frame = _build_users_frame()

    name_type = lookup_column_type(frame, "name")
    assert name_type == ScalarType(kind=ScalarKind.STRING, nullable=False)

    # Selecting columns should respect the requested order.
    projection = derive_select_schema(frame, ["name", "id"])
    assert projection.schema.order == ["name", "id"]

    grouped = derive_group_schema(
        frame,
        group_columns=["country"],
        aggregations={"avg_age": ScalarType(ScalarKind.FLOAT)},
    )
    assert grouped.schema.order == ["country", "avg_age"]
    assert lookup_column_type(grouped, "avg_age") == ScalarType(ScalarKind.FLOAT, nullable=True)


def test_dataset_and_expression_types() -> None:
    frame = _build_users_frame()
    dataset = DatasetType(frame=frame, source="users_dataset")
    assert stringify_type(dataset) == "users_dataset"
    assert is_assignable(dataset, DatasetType(frame=frame))

    expr = ExpressionType(value_type=ScalarType(ScalarKind.BOOL), is_predicate=True)
    assert "predicate" in stringify_type(expr)

    prompt_types = PromptIOTypes(
        inputs={"question": ScalarType(ScalarKind.STRING)},
        outputs={"answer": ScalarType(ScalarKind.STRING)},
    )
    assert "question" in prompt_types.inputs
    assert "answer" in prompt_types.outputs


def test_infer_scalar_type_aliases() -> None:
    dtype = infer_scalar_type("INTEGER", nullable=False)
    assert dtype == ScalarType(kind=ScalarKind.INT, nullable=False)
