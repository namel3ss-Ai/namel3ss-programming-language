"""Rendering helpers for table widgets."""

from __future__ import annotations

import html
import json
from typing import Any, Dict, Optional

from namel3ss.ast import ShowTable

from ..theme import layout_to_payload, style_to_inline
from .context import RenderContext


def render_table(stmt: ShowTable, ctx: RenderContext, component_index: Optional[int]) -> None:
    table_id = f"table_{ctx.slug}_{ctx.counters.setdefault('table', 0)}"
    ctx.counters['table'] += 1
    preview_payload = ctx.preview.table_preview(stmt)
    preview_data = preview_payload if isinstance(preview_payload, dict) else {}
    columns = preview_data.get("columns") or stmt.columns or ['Column 1', 'Column 2', 'Column 3']
    layout_payload = layout_to_payload(stmt.layout)
    variant = (layout_payload or {}).get('variant', '')
    style_inline = style_to_inline(stmt.style or {}) if stmt.style else ''

    wrapper_classes = ['n3-widget', 'n3-widget-table']
    if variant:
        wrapper_classes.append(f"n3-widget--{variant.lower()}")
    emphasis = (layout_payload or {}).get('emphasis')
    if emphasis:
        wrapper_classes.append(f"n3-emphasis-{emphasis.lower()}")
    align = (layout_payload or {}).get('align')
    if align:
        wrapper_classes.append(f"n3-align-{align.lower()}")

    wrapper_class_attr = " ".join(wrapper_classes)
    wrapper_attrs = [f'class="{wrapper_class_attr}"']
    if layout_payload:
        wrapper_attrs.append(
            f'data-n3-layout="{html.escape(json.dumps(layout_payload), quote=True)}"'
        )
    if style_inline:
        wrapper_attrs.append(f'style="{style_inline}"')
    if stmt.insight:
        wrapper_attrs.append(f'data-n3-insight="{html.escape(stmt.insight)}"')
    if component_index is not None:
        wrapper_attrs.append(f'data-n3-component-index="{component_index}"')
        wrapper_attrs.append(
            f'data-n3-endpoint="/api/pages/{ctx.backend_slug}/tables/{component_index}"'
        )

    table_classes = ['n3-table']
    if variant.lower() == 'dense':
        table_classes.append('n3-table-dense')

    ctx.body_lines.append(f"<section {' '.join(wrapper_attrs)}>")
    ctx.body_lines.append(f"  <h3>{html.escape(stmt.title)}</h3>")
    table_class_attr = " ".join(table_classes)
    ctx.body_lines.append(
        f"  <div class=\"n3-table-container\"><table id=\"{table_id}\" class=\"{table_class_attr}\"></table></div>"
    )
    ctx.body_lines.append("  <div class=\"n3-widget-errors n3-widget-errors--hidden\" data-n3-error-slot></div>")
    ctx.body_lines.append("</section>")

    table_data = {
        "title": stmt.title,
        "columns": columns,
        "rows": preview_data.get("rows", []),
        "insight": stmt.insight,
        "errors": preview_data.get("errors", []),
    }
    table_def: Dict[str, Any] = {
        "type": "table",
        "id": table_id,
        "data": table_data,
        "layout": layout_payload or {},
        "style": stmt.style or {},
        "insight": stmt.insight,
    }
    if component_index is not None:
        table_def["componentIndex"] = component_index
        table_def["endpoint"] = f"/api/pages/{ctx.backend_slug}/tables/{component_index}"
    ctx.widget_defs.append(table_def)
