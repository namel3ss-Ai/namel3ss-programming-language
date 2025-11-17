"""Rendering helpers for action statements."""

from __future__ import annotations

import html
import re
from typing import Dict, Optional

from namel3ss.ast import Action, ToastOperation

from .context import RenderContext


def _action_success_message(stmt: Action) -> Optional[str]:
    for op in stmt.operations:
        if isinstance(op, ToastOperation):
            return op.message
    return None


def render_action(stmt: Action, ctx: RenderContext, component_index: Optional[int]) -> None:
    button_label = stmt.name
    match = re.search(r'clicks\s+"([^"]+)"', stmt.trigger)
    if match:
        button_label = match.group(1)

    btn_id = f"action_btn_{ctx.slug}_{ctx.counters.setdefault('action', 0)}"
    ctx.counters['action'] += 1

    wrapper_attrs = ['class="n3-widget n3-widget-action"', 'data-n3-action-wrapper="true"']
    if component_index is not None:
        wrapper_attrs.append(f'data-n3-component-index="{component_index}"')
        wrapper_attrs.append(
            f'data-n3-endpoint="/api/pages/{ctx.backend_slug}/actions/{component_index}"'
        )

    ctx.body_lines.append(f"<section {' '.join(wrapper_attrs)}>")

    button_attrs = [
        f'id="{btn_id}"',
        'type="button"',
        'class="n3-action-button"',
        'data-n3-action="true"',
    ]
    if component_index is not None:
        button_attrs.append(f'data-n3-component-index="{component_index}"')
        button_attrs.append(
            f'data-n3-endpoint="/api/pages/{ctx.backend_slug}/actions/{component_index}"'
        )

    ctx.body_lines.append(f"  <button {' '.join(button_attrs)}>{html.escape(button_label)}</button>")
    ctx.body_lines.append(
        "  <div class=\"n3-widget-errors n3-widget-errors--hidden\" data-n3-error-slot></div>"
    )
    ctx.body_lines.append("</section>")

    action_def: Dict[str, object] = {
        "type": "action",
        "id": btn_id,
        "name": stmt.name,
        "label": button_label,
    }
    success_message = _action_success_message(stmt)
    if success_message:
        action_def["successMessage"] = success_message
    if component_index is not None:
        action_def["componentIndex"] = component_index
        action_def["endpoint"] = f"/api/pages/{ctx.backend_slug}/actions/{component_index}"

    ctx.widget_defs.append(action_def)
