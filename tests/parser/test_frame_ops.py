from __future__ import annotations

import pytest

from namel3ss.ast import (
    BinaryOp,
    FrameFilter,
    FrameGroupBy,
    FrameJoin,
    FrameOrderBy,
    FrameRef,
    FrameSelect,
    FrameSummarise,
)
from namel3ss.parser import Parser


def _parse_first_assignment(expression_line: str):
    source = (
        'app "Frames".\n'
        '\n'
        'frame users:\n'
        '  column id string\n'
        '  column name string\n'
        '  column active bool\n'
        '  column signup_at datetime\n'
        '\n'
        'frame daily_sales:\n'
        '  column country string\n'
        '  column revenue float\n'
        '\n'
        'frame customers:\n'
        '  column customer_id string\n'
        '\n'
        f'set derived = {expression_line}\n'
    )
    app = Parser(source).parse()
    assert app.variables, "Expected variable assignments to be parsed"
    return app.variables[0].value


def test_parse_filter_select_chain() -> None:
    expr = _parse_first_assignment('users.filter(active == true).select(id, name).order_by(signup_at, descending = true)')
    assert isinstance(expr, FrameOrderBy)
    assert expr.columns == ['signup_at']
    assert expr.descending is True

    select_expr = expr.source
    assert isinstance(select_expr, FrameSelect)
    assert select_expr.columns == ['id', 'name']

    filter_expr = select_expr.source
    assert isinstance(filter_expr, FrameFilter)
    assert isinstance(filter_expr.predicate, BinaryOp)
    assert isinstance(filter_expr.source, FrameRef)
    assert filter_expr.source.name == 'users'


def test_parse_group_by_summarise_chain() -> None:
    expr = _parse_first_assignment(
        'daily_sales.group_by(country).summarise(total_revenue = sum(revenue), avg_revenue = mean(revenue))'
    )
    assert isinstance(expr, FrameSummarise)
    assert set(expr.aggregations.keys()) == {'total_revenue', 'avg_revenue'}

    group_expr = expr.source
    assert isinstance(group_expr, FrameGroupBy)
    assert group_expr.columns == ['country']
    assert isinstance(group_expr.source, FrameRef)
    assert group_expr.source.name == 'daily_sales'


def test_parse_join_chain() -> None:
    expr = _parse_first_assignment('users.join(customers, on = customer_id, how = "left")')
    assert isinstance(expr, FrameJoin)
    assert expr.right == 'customers'
    assert expr.on == ['customer_id']
    assert expr.how == 'left'

    assert isinstance(expr.left, FrameRef)
    assert expr.left.name == 'users'


def test_invalid_unknown_operation_error() -> None:
    with pytest.raises(Exception):
        _parse_first_assignment('users.filter(active == true).foo(bar)')