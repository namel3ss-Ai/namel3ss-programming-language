from __future__ import annotations

import textwrap

from namel3ss.ast import App, Dataset, Frame
from namel3ss.ast.pages import ForLoop, IfBlock, ShowTable, ShowText
from namel3ss.lang.grammar import parse_module
from namel3ss.ast import FilterOp, GroupByOp


def test_parse_module_basic_blocks() -> None:
    source = textwrap.dedent(
        """
        module demo.app
        language_version "0.1.0"

        app "Demo App".

        dataset "monthly_sales" from table sales:
          filter by: region == "EU"
          group by: month

        frame "SalesFrame" from dataset monthly_sales:
          columns: month, total_revenue

        page "Home" at "/":
          show text "Welcome"
          show table "Sales" from dataset monthly_sales
        """
    )

    module = parse_module(source, path="examples/home.n3")

    assert module.name == "demo.app"
    assert module.language_version == "0.1.0"
    assert module.path == "examples/home.n3"
    assert module.imports == []
    assert module.has_explicit_app is True
    assert len(module.body) == 1

    app = module.body[0]
    assert isinstance(app, App)
    assert app.name == "Demo App"
    assert len(app.datasets) == 1
    dataset = app.datasets[0]
    assert isinstance(dataset, Dataset)
    assert dataset.name == "monthly_sales"
    assert dataset.source_type == "table"
    assert dataset.source == "sales"
    assert len(dataset.operations) == 2
    assert isinstance(dataset.operations[0], FilterOp)
    assert isinstance(dataset.operations[1], GroupByOp)

    assert len(app.frames) == 1
    frame = app.frames[0]
    assert isinstance(frame, Frame)
    assert frame.name == "SalesFrame"
    assert frame.source == "monthly_sales"
    assert [col.name for col in frame.columns] == ["month", "total_revenue"]

    assert len(app.pages) == 1
    page = app.pages[0]
    assert page.name == "Home"
    assert page.route == "/"
    assert len(page.statements) == 2
    assert isinstance(page.statements[0], ShowText)
    assert isinstance(page.statements[1], ShowTable)


def test_parse_module_control_flow() -> None:
    source = textwrap.dedent(
        """
        app "Control".

        page "Dashboard" at "/admin":
          if user.role == "admin":
            show text "Welcome, Admin"
          else:
            show text "Access denied"

          for order in dataset monthly_sales:
            show text "Order #{order.id}"
        """
    )

    module = parse_module(source)
    app = module.body[0]
    page = app.pages[0]

    assert len(page.statements) == 2
    first_stmt = page.statements[0]
    assert isinstance(first_stmt, IfBlock)
    assert isinstance(first_stmt.body[0], ShowText)
    assert first_stmt.else_body and isinstance(first_stmt.else_body[0], ShowText)

    second_stmt = page.statements[1]
    assert isinstance(second_stmt, ForLoop)
    assert second_stmt.loop_var == "order"
    assert second_stmt.source_kind == "dataset"
    assert second_stmt.source_name == "monthly_sales"
    assert isinstance(second_stmt.body[0], ShowText)
