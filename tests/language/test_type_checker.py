"""Tests for the static type checker integration."""

import pytest

from namel3ss.ast import (
    Action,
    App,
    AttributeRef,
    BinaryOp,
    ComputedColumnOp,
    Dataset,
    DatasetSchemaField,
    FilterOp,
    ForLoop,
    JoinOp,
    Literal,
    Module,
    NameRef,
    OrderByOp,
    Page,
    Prompt,
    PromptField,
    Program,
    RunPromptOperation,
    ShowTable,
    VariableAssignment,
)
from namel3ss.resolver import resolve_program
from namel3ss.types import N3TypeError, check_app


def _orders_dataset() -> Dataset:
    return Dataset(
        name="orders",
        source_type="table",
        source="orders",
        schema=[
            DatasetSchemaField(name="id", dtype="int", nullable=False),
            DatasetSchemaField(name="status", dtype="string"),
            DatasetSchemaField(name="amount", dtype="float"),
            DatasetSchemaField(name="subtotal", dtype="float"),
            DatasetSchemaField(name="tax", dtype="float"),
            DatasetSchemaField(name="region_id", dtype="int"),
        ],
    )


def _regions_dataset() -> Dataset:
    return Dataset(
        name="regions",
        source_type="table",
        source="regions",
        schema=[
            DatasetSchemaField(name="id", dtype="int", nullable=False),
            DatasetSchemaField(name="name", dtype="string"),
        ],
    )


def test_type_checker_detects_missing_dataset_column() -> None:
    dataset = _orders_dataset()
    dataset.operations.append(
        FilterOp(
            condition=BinaryOp(
                left=NameRef(name="unknown_column"),
                op="==",
                right=Literal(value="new"),
            )
        )
    )
    app = App(name="Demo", datasets=[dataset])

    with pytest.raises(N3TypeError, match="unknown_column"):
        check_app(app, path="demo.n3")


def test_type_checker_validates_prompt_arguments() -> None:
    dataset = _orders_dataset()
    prompt = Prompt(
        name="support_prompt",
        model="gpt",
        template="Respond helpfully",
        input_fields=[PromptField(name="question", field_type="text")],
    )
    action = Action(
        name="submit",
        trigger="onclick",
        operations=[
            RunPromptOperation(
                prompt_name="support_prompt",
                arguments={"question": Literal(value=42)},
            )
        ],
    )
    page = Page(name="Support", route="/support", statements=[action])
    app = App(name="Demo", datasets=[dataset], prompts=[prompt], pages=[page])

    with pytest.raises(N3TypeError, match="support_prompt"):
        check_app(app)


def test_type_checker_accepts_valid_program() -> None:
    dataset = _orders_dataset()
    page = Page(
        name="Orders",
        route="/orders",
        statements=[
            ForLoop(
                loop_var="order",
                source_kind="dataset",
                source_name="orders",
                body=[
                    VariableAssignment(
                        name="current_status",
                        value=AttributeRef(base="order", attr="status"),
                    ),
                ],
            ),
            ShowTable(title="Orders", source_type="dataset", source="orders"),
        ],
    )
    app = App(name="Demo", datasets=[dataset], pages=[page])

    # Should not raise
    check_app(app, path="demo.n3")


def test_computed_columns_are_type_checked() -> None:
    dataset = _orders_dataset()
    dataset.operations.extend(
        [
            ComputedColumnOp(
                name="total",
                expression=BinaryOp(left=NameRef("subtotal"), op="+", right=NameRef("tax")),
            ),
            FilterOp(
                condition=BinaryOp(left=NameRef("total"), op=">", right=Literal(value=100.0)),
            ),
        ]
    )
    app = App(name="Demo", datasets=[dataset])

    check_app(app)


def test_computed_column_requires_scalar_expression() -> None:
    dataset = _orders_dataset()
    dataset.operations.append(
        ComputedColumnOp(
            name="invalid",
            expression=NameRef("orders"),  # resolves to dataset type, not scalar
        )
    )
    app = App(name="Demo", datasets=[dataset])

    with pytest.raises(N3TypeError, match="scalar"):
        check_app(app)


def test_order_by_requires_known_columns() -> None:
    dataset = _orders_dataset()
    dataset.operations.append(OrderByOp(columns=["missing"]))
    app = App(name="Demo", datasets=[dataset])

    with pytest.raises(N3TypeError, match="missing"):
        check_app(app)


def test_dataset_join_merges_schema() -> None:
    orders = _orders_dataset()
    regions = _regions_dataset()
    orders.operations.append(
        JoinOp(
            target_type="dataset",
            target_name="regions",
            condition=BinaryOp(
                left=AttributeRef(base="orders", attr="region_id"),
                op="==",
                right=AttributeRef(base="regions", attr="id"),
            ),
        )
    )
    page = Page(
        name="Orders",
        route="/",
        statements=[
            ForLoop(
                loop_var="order",
                source_kind="dataset",
                source_name="orders",
                body=[
                    VariableAssignment(
                        name="region_name",
                        value=AttributeRef(base="order", attr="name"),
                    )
                ],
            )
        ],
    )
    app = App(name="Demo", datasets=[orders, regions], pages=[page])

    check_app(app)


def test_dataset_join_requires_known_target() -> None:
    orders = _orders_dataset()
    orders.operations.append(
        JoinOp(
            target_type="dataset",
            target_name="missing",
            condition=BinaryOp(left=NameRef("amount"), op="==", right=Literal(value=0)),
        )
    )
    app = App(name="Demo", datasets=[orders])

    with pytest.raises(N3TypeError, match="missing"):
        check_app(app)


def test_resolve_program_runs_type_checker() -> None:
    page = Page(
        name="Broken",
        route="/broken",
        statements=[ShowTable(title="Orders", source_type="dataset", source="missing")],
    )
    app = App(name="Demo", pages=[page])
    module = Module(name="main", path="demo.n3", body=[app], has_explicit_app=True)
    program = Program(modules=[module])

    with pytest.raises(N3TypeError, match="missing"):
        resolve_program(program, entry_path="demo.n3")
