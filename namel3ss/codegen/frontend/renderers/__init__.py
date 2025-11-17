"""Statement dispatch for frontend rendering."""

from __future__ import annotations

from typing import Optional, Sequence

from namel3ss.ast import (
    Action,
    ForLoop,
    IfBlock,
    PageStatement,
    ShowChart,
    ShowForm,
    ShowTable,
    ShowText,
)

from .actions import render_action
from .chart import render_chart
from .context import RenderContext
from .flow import render_for, render_if
from .forms import render_form
from .tables import render_table
from .text import render_text

_PAGE_COMPONENT_TYPES = (ShowText, ShowTable, ShowChart, ShowForm, Action)


def _next_component_index(ctx: RenderContext, stmt: PageStatement) -> Optional[int]:
    counts_for_component = ctx.scope == "page" and isinstance(stmt, _PAGE_COMPONENT_TYPES)
    if not counts_for_component:
        return None
    current = ctx.component_tracker.get("value", 0)
    ctx.component_tracker["value"] = current + 1
    return current


def render_statements(statements: Sequence[PageStatement], ctx: RenderContext) -> None:
    for stmt in statements:
        component_index = _next_component_index(ctx, stmt)
        if isinstance(stmt, ShowText):
            render_text(stmt, ctx, component_index)
        elif isinstance(stmt, ShowTable):
            render_table(stmt, ctx, component_index)
        elif isinstance(stmt, ShowChart):
            render_chart(stmt, ctx, component_index)
        elif isinstance(stmt, ShowForm):
            render_form(stmt, ctx, component_index)
        elif isinstance(stmt, Action):
            render_action(stmt, ctx, component_index)
        elif isinstance(stmt, IfBlock):
            render_if(stmt, ctx, component_index, render_statements)
        elif isinstance(stmt, ForLoop):
            render_for(stmt, ctx, component_index, render_statements)
        else:
            ctx.body_lines.append(f"<!-- Unknown statement type: {type(stmt).__name__} -->")
