"""Rendering helpers for form statements."""

from __future__ import annotations

import html
import json
from typing import Optional

from namel3ss.ast import GoToPageOperation, ShowForm, ToastOperation, UpdateOperation

from ..slugs import slugify_route
from ..theme import style_to_inline
from .context import RenderContext


def render_form(stmt: ShowForm, ctx: RenderContext, component_index: Optional[int]) -> None:
    form_id = f"form_{ctx.slug}_{ctx.counters.setdefault('form', 0)}"
    ctx.counters['form'] += 1
    styles = style_to_inline(stmt.styles)
    ctx.body_lines.append(f"<h3>{html.escape(stmt.title)}</h3>")
    ctx.body_lines.append(f"<form id=\"{form_id}\" style=\"{styles}\">")
    for field in stmt.fields:
        field_type = field.field_type or 'text'
        ctx.body_lines.append(
            f"  <label>{html.escape(field.name)}: <input name=\"{field.name}\" type=\"{field_type}\"></label><br>"
        )
    ctx.body_lines.append("  <button type=\"submit\">Submit</button>")
    ctx.body_lines.append("</form>")

    handler_lines = [
        f"var formEl = document.getElementById('{form_id}');",
        "if (formEl) {",
        "  formEl.addEventListener('submit', function(e) {",
        "    e.preventDefault();",
    ]
    for op in stmt.on_submit_ops:
        if isinstance(op, ToastOperation):
            handler_lines.append(f"    window.N3Widgets.showToast({json.dumps(op.message)});")
        elif isinstance(op, GoToPageOperation):
            target_slug = slugify_route(
                next((p.route for p in ctx.app.pages if p.name == op.page_name), op.page_name)
            )
            handler_lines.append(f"    window.location.href = '{target_slug}.html';")
        elif isinstance(op, UpdateOperation):
            handler_lines.append(
                f"    console.log('Update {op.table}: {op.set_expression} where {op.where_expression or ''}');"
            )
            handler_lines.append(f"    window.N3Widgets.showToast('Updated {op.table}');")
        else:
            handler_lines.append("    window.N3Widgets.showToast('Executed operation');")
    handler_lines.extend([
        "  });",
        "}",
    ])
    ctx.inline_scripts.append('\n'.join(handler_lines))
