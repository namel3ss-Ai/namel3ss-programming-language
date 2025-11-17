"""Rendering helpers for chart widgets."""

from __future__ import annotations

import html
import json
from typing import Any, Dict, Optional

from namel3ss.ast import ShowChart

from ..charts import build_chart_config
from ..theme import layout_to_payload, style_to_inline
from .context import RenderContext


def render_chart(stmt: ShowChart, ctx: RenderContext, component_index: Optional[int]) -> None:
    chart_id = f"chart_{ctx.slug}_{ctx.counters.setdefault('chart', 0)}"
    ctx.counters['chart'] += 1
    layout_payload = layout_to_payload(stmt.layout)
    style_inline = style_to_inline(stmt.style or {}) if stmt.style else ''

    wrapper_classes = ['n3-widget', 'n3-widget-chart']
    variant = (layout_payload or {}).get('variant')
    if variant:
        wrapper_classes.append(f"n3-widget--{variant.lower()}")
    align = (layout_payload or {}).get('align')
    if align:
        wrapper_classes.append(f"n3-align-{align.lower()}")
    emphasis = (layout_payload or {}).get('emphasis')
    if emphasis:
        wrapper_classes.append(f"n3-emphasis-{emphasis.lower()}")

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
            f'data-n3-endpoint="/api/pages/{ctx.backend_slug}/charts/{component_index}"'
        )

    ctx.body_lines.append(f"<section {' '.join(wrapper_attrs)}>")
    ctx.body_lines.append(f"  <h3>{html.escape(stmt.heading)}</h3>")
    ctx.body_lines.append(
        f"  <div class=\"n3-chart-container\"><canvas id=\"{chart_id}\" class=\"n3-chart\"></canvas></div>"
    )
    ctx.body_lines.append("  <div class=\"n3-widget-errors n3-widget-errors--hidden\" data-n3-error-slot></div>")
    ctx.body_lines.append("</section>")

    dataset_payload = ctx.preview.chart_preview(stmt)
    chart_config = build_chart_config(stmt, dataset_payload, theme=ctx.theme_mode)
    preview_errors = dataset_payload.get("errors", []) if isinstance(dataset_payload, dict) else []
    if stmt.insight:
        chart_config['insight'] = stmt.insight
    chart_def: Dict[str, Any] = {
        "type": "chart",
        "id": chart_id,
        "config": chart_config,
        "layout": layout_payload or {},
        "heading": stmt.heading,
        "style": stmt.style or {},
        "legend": stmt.legend or {},
        "title": stmt.title,
        "insight": stmt.insight,
        "errors": preview_errors,
    }
    if component_index is not None:
        chart_def["componentIndex"] = component_index
        chart_def["endpoint"] = f"/api/pages/{ctx.backend_slug}/charts/{component_index}"
    ctx.widget_defs.append(chart_def)
