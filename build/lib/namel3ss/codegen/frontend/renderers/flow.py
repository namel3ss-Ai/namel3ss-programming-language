"""Rendering helpers for control-flow statements."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from namel3ss.ast import ForLoop, IfBlock, PageStatement

from .context import RenderContext

RenderFn = Callable[[Sequence[PageStatement], RenderContext], None]


def render_if(stmt: IfBlock, ctx: RenderContext, component_index: Optional[int], render_fn: RenderFn) -> None:
    condition_text = getattr(stmt.condition, 'raw', repr(stmt.condition))
    ctx.body_lines.append(f"<!-- If condition: {condition_text} -->")
    ctx.body_lines.append(
        "<div class=\"if-block\" style=\"border-left: 3px solid #4CAF50; padding-left: 10px; margin: 10px 0;\">"
    )
    ctx.body_lines.append(f"<p><em>If {condition_text}:</em></p>")
    render_fn(stmt.body, ctx.nested())
    ctx.body_lines.append("</div>")

    if stmt.else_body:
        ctx.body_lines.append(
            "<div class=\"else-block\" style=\"border-left: 3px solid #FF9800; padding-left: 10px; margin: 10px 0;\">"
        )
        ctx.body_lines.append("<p><em>Else:</em></p>")
        render_fn(stmt.else_body, ctx.nested())
        ctx.body_lines.append("</div>")


def render_for(stmt: ForLoop, ctx: RenderContext, component_index: Optional[int], render_fn: RenderFn) -> None:
    ctx.body_lines.append(
        f"<!-- For loop: {stmt.loop_var} in {stmt.source_kind} {stmt.source_name} -->"
    )
    ctx.body_lines.append(
        "<div class=\"for-loop\" style=\"border-left: 3px solid #2196F3; padding-left: 10px; margin: 10px 0;\">"
    )
    ctx.body_lines.append(
        f"<p><em>For each {stmt.loop_var} in {stmt.source_kind} {stmt.source_name}:</em></p>"
    )
    for index in range(3):
        ctx.body_lines.append(
            "<div class=\"loop-iteration\" style=\"margin-left: 10px; margin-bottom: 5px;\">"
        )
        ctx.body_lines.append(f"<small>Iteration {index + 1}:</small>")
        render_fn(stmt.body, ctx.nested())
        ctx.body_lines.append("</div>")
    ctx.body_lines.append("</div>")
