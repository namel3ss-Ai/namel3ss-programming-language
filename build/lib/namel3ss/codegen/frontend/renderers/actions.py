"""Rendering helpers for action statements."""

from __future__ import annotations

import html
import json
import re
from typing import Optional

from namel3ss.ast import Action, GoToPageOperation, ToastOperation, UpdateOperation

from ..slugs import slugify_route
from .context import RenderContext


def render_action(stmt: Action, ctx: RenderContext, component_index: Optional[int]) -> None:
    button_label = stmt.name
    match = re.search(r'clicks\s+"([^"]+)"', stmt.trigger)
    if match:
        button_label = match.group(1)
    btn_id = f"action_btn_{ctx.slug}_{ctx.counters.setdefault('action', 0)}"
    ctx.counters['action'] += 1
    ctx.body_lines.append(f"<button id=\"{btn_id}\">{html.escape(button_label)}</button>")

    handler_lines = [
        f"var btnEl = document.getElementById('{btn_id}');",
        "if (btnEl) {",
        "  btnEl.addEventListener('click', function() {",
    ]
    for op in stmt.operations:
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
