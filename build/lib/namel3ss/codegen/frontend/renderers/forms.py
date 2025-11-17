"""Rendering helpers for form statements."""

from __future__ import annotations

import html
import json
from typing import Dict, List, Optional

from namel3ss.ast import ShowForm, ToastOperation

from ..theme import layout_to_payload, style_to_inline
from .context import RenderContext


def _form_success_message(stmt: ShowForm) -> Optional[str]:
    for op in stmt.on_submit_ops:
        if isinstance(op, ToastOperation):
            return op.message
    return None


def _form_preview_fields(stmt: ShowForm) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for field in stmt.fields:
        entries.append({
            "name": field.name,
            "type": field.field_type or 'text',
        })
    return entries


def render_form(stmt: ShowForm, ctx: RenderContext, component_index: Optional[int]) -> None:
    form_id = f"form_{ctx.slug}_{ctx.counters.setdefault('form', 0)}"
    ctx.counters['form'] += 1

    layout_payload = layout_to_payload(stmt.layout)
    styles = style_to_inline(stmt.styles)

    wrapper_classes = ['n3-widget', 'n3-widget-form']
    if layout_payload and layout_payload.get('variant'):
        wrapper_classes.append(f"n3-widget--{str(layout_payload['variant']).lower()}")
    emphasis = (layout_payload or {}).get('emphasis')
    if emphasis:
        wrapper_classes.append(f"n3-emphasis-{str(emphasis).lower()}")
    align = (layout_payload or {}).get('align')
    if align:
        wrapper_classes.append(f"n3-align-{str(align).lower()}")

    wrapper_attrs = [f'class="{ " ".join(wrapper_classes)}"']
    if layout_payload:
        wrapper_attrs.append(
            f'data-n3-layout="{html.escape(json.dumps(layout_payload), quote=True)}"'
        )
    if component_index is not None:
        wrapper_attrs.append(f'data-n3-component-index="{component_index}"')
        wrapper_attrs.append(
            f'data-n3-endpoint="/api/pages/{ctx.backend_slug}/forms/{component_index}"'
        )

    ctx.body_lines.append(f"<section {' '.join(wrapper_attrs)}>")
    ctx.body_lines.append(f"  <h3>{html.escape(stmt.title)}</h3>")

    form_attrs = [f'id="{form_id}"', 'class="n3-form"', 'data-n3-form="true"']
    if styles:
        form_attrs.append(f'style="{styles}"')
    if component_index is not None:
        form_attrs.append(f'data-n3-component-index="{component_index}"')
        form_attrs.append(
            f'data-n3-endpoint="/api/pages/{ctx.backend_slug}/forms/{component_index}"'
        )

    ctx.body_lines.append(f"  <form {' '.join(form_attrs)}>")

    for field in stmt.fields:
        field_name_attr = html.escape(field.name, quote=True)
        field_label = html.escape(field.name)
        input_type = html.escape(field.field_type or 'text', quote=True)
        ctx.body_lines.append(
            f"    <div class=\"n3-form-field\" data-n3-field=\"{field_name_attr}\">"
        )
        ctx.body_lines.append("      <label>")
        ctx.body_lines.append(f"        <span>{field_label}</span>")
        ctx.body_lines.append(
            f"        <input name=\"{field_name_attr}\" type=\"{input_type}\" required>"
        )
        ctx.body_lines.append("      </label>")
        ctx.body_lines.append(
            f"      <div class=\"n3-field-error\" data-n3-field-error=\"{field_name_attr}\"></div>"
        )
        ctx.body_lines.append("    </div>")

    ctx.body_lines.append(
        "    <div class=\"n3-widget-errors n3-widget-errors--hidden\" data-n3-error-slot></div>"
    )
    ctx.body_lines.append("    <button type=\"submit\">Submit</button>")
    ctx.body_lines.append("  </form>")
    ctx.body_lines.append("</section>")

    form_def: Dict[str, object] = {
        "type": "form",
        "id": form_id,
        "title": stmt.title,
        "fields": _form_preview_fields(stmt),
        "layout": layout_payload or {},
    }
    success_message = _form_success_message(stmt)
    if success_message:
        form_def["successMessage"] = success_message
    if component_index is not None:
        form_def["componentIndex"] = component_index
        form_def["endpoint"] = f"/api/pages/{ctx.backend_slug}/forms/{component_index}"

    ctx.widget_defs.append(form_def)
