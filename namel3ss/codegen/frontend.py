"""
Static frontend generator for Namel3ss.

This module provides functions to turn an :class:`namel3ss.ast.App` into
a set of static HTML, CSS and JavaScript files.  The aim is to
produce a self contained website that demonstrates the structure
defined in the N3 program.  It is not meant to be the final
production target but rather a convenient way to preview apps
without installing additional dependencies.

The generator writes the following files into a target directory:

* ``index.html`` – landing page linking to the first declared page.
* One HTML file per page (filename derived from the route).
* ``styles.css`` – basic styling derived from the theme and element
  defaults.
* ``scripts.js`` – JavaScript functions for actions, charts and
  simple UI behaviour.

Charts use Chart.js via a CDN (``https://cdn.jsdelivr.net/npm/chart.js``).
Tables are rendered with simple HTML.  Forms capture user input and
display a toast upon submission.  Actions are mapped to buttons and
produce alerts or page redirects depending on the operation.
"""

from __future__ import annotations

import copy
import os
import json
import textwrap
from pathlib import Path
import re
import html
from string import Template
from typing import Any, Dict, List, Optional, Tuple, Union

from ..ast import (
    App,
    Page,
    PageStatement,
    ShowText,
    ShowTable,
    ShowChart,
    ShowForm,
    Action,
    PredictStatement,
    VariableAssignment,
    UpdateOperation,
    ToastOperation,
    GoToPageOperation,
    IfBlock,
    ForLoop,
    LayoutMeta,
)

from .frontend import placeholders as placeholder_utils

_TEXT_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


_DEFAULT_CHART_COLORS: Dict[str, Tuple[str, str]] = {
    "bar": ("rgba(99, 102, 241, 0.6)", "rgba(99, 102, 241, 1)"),
    "line": ("rgba(16, 185, 129, 0.4)", "rgba(16, 185, 129, 1)"),
    "pie": ("rgba(249, 115, 22, 0.6)", "rgba(249, 115, 22, 1)"),
    "doughnut": ("rgba(249, 115, 22, 0.6)", "rgba(249, 115, 22, 1)"),
    "radar": ("rgba(236, 72, 153, 0.4)", "rgba(236, 72, 153, 1)"),
    "default": ("rgba(148, 163, 184, 0.6)", "rgba(148, 163, 184, 1)"),
}


_PAGE_COMPONENT_TYPES: Tuple[type, ...] = (
    ShowText,
    ShowTable,
    ShowChart,
    ShowForm,
    Action,
    VariableAssignment,
    IfBlock,
    ForLoop,
    PredictStatement,
)


def _theme_palette(theme: Optional[str]) -> Dict[str, str]:
    mode = (theme or "").lower()
    if mode in {"dark", "night", "dim"}:
        return {
            "text": "#f5f5f5",
            "grid": "rgba(255, 255, 255, 0.15)",
            "background": "#111827",
        }
    return {
        "text": "#1f2937",
        "grid": "rgba(15, 23, 42, 0.08)",
        "background": "#ffffff",
    }


def _layout_to_payload(layout: Optional[LayoutMeta]) -> Optional[Dict[str, Any]]:
    if not layout:
        return None
    payload: Dict[str, Any] = {}
    if layout.width is not None:
        payload["width"] = layout.width
    if layout.height is not None:
        payload["height"] = layout.height
    if layout.variant:
        payload["variant"] = layout.variant
    if layout.align:
        payload["align"] = layout.align
    if layout.emphasis:
        payload["emphasis"] = layout.emphasis
    if layout.extras:
        payload["extras"] = copy.deepcopy(layout.extras)
    return payload or None


def build_chart_config(
    chart_stmt: ShowChart,
    dataset_payload: Dict[str, Any],
    theme: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a resilient Chart.js configuration from AST metadata and dataset preview."""

    chart_type = (chart_stmt.chart_type or dataset_payload.get("type") or "bar").lower()
    labels = list(dataset_payload.get("labels") or [])
    raw_series = dataset_payload.get("datasets")
    if raw_series is None:
        raw_series = dataset_payload.get("series")
    if isinstance(raw_series, dict):
        raw_series = [raw_series]

    heading_label = chart_stmt.heading or chart_stmt.title or "Series"

    style_dict: Dict[str, Any] = copy.deepcopy(chart_stmt.style or {})
    title_style = style_dict.get("title") if isinstance(style_dict.get("title"), dict) else {}
    legend_style: Dict[str, Any] = {}
    if isinstance(style_dict.get("legend"), dict):
        legend_style.update(style_dict["legend"])
    if isinstance(chart_stmt.legend, dict):
        legend_style.update(chart_stmt.legend)
    colors_style = style_dict.get("colors") if isinstance(style_dict.get("colors"), dict) else {}
    axes_style = style_dict.get("axes") if isinstance(style_dict.get("axes"), dict) else {}

    normalised: List[Dict[str, Any]] = []
    if isinstance(raw_series, list):
        for entry in raw_series:
            if not isinstance(entry, dict):
                continue
            entry_copy: Dict[str, Any] = copy.deepcopy(entry)
            data = entry_copy.get("data")
            if not isinstance(data, list) or not data:
                continue
            bg_default, border_default = _DEFAULT_CHART_COLORS.get(chart_type, _DEFAULT_CHART_COLORS["default"])
            label = entry_copy.get("label") or heading_label
            entry_copy.setdefault("backgroundColor", chart_stmt.color or bg_default)
            entry_copy.setdefault("borderColor", chart_stmt.color or border_default)
            entry_copy.setdefault("borderWidth", 1)
            entry_copy.setdefault("label", label)
            if chart_type in {"line", "radar"}:
                entry_copy.setdefault("fill", False)
                entry_copy.setdefault("tension", 0.35)
            normalised.append(entry_copy)

    if not labels or not normalised:
        labels = ["No data"]
        bg_default, border_default = _DEFAULT_CHART_COLORS.get(chart_type, _DEFAULT_CHART_COLORS["default"])
        normalised = [
            {
                "label": heading_label,
                "data": [0],
                "backgroundColor": chart_stmt.color or bg_default,
                "borderColor": chart_stmt.color or border_default,
                "borderWidth": 1,
                "fill": False,
            }
        ]

    palette = _theme_palette(theme)
    
    def _to_number(value: Any) -> Optional[Union[int, float]]:
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.endswith("px"):
                cleaned = cleaned[:-2]
            try:
                return float(cleaned) if "." in cleaned else int(cleaned)
            except ValueError:
                return None
        return None

    def _apply_dataset_color(dataset: Dict[str, Any], color_value: Any) -> None:
        if not isinstance(color_value, str):
            return
        dataset["backgroundColor"] = color_value
        dataset["borderColor"] = color_value
        dataset.setdefault("borderWidth", 1)

    def _extract_axis_config(axis_key: str) -> Dict[str, Any]:
        if not isinstance(axes_style, dict):
            return {}
        candidates = [axis_key, axis_key.lower(), axis_key.upper(), f"{axis_key}_axis", f"{axis_key}-axis", f"{axis_key} axis"]
        for alias in candidates:
            if alias in axes_style:
                value = axes_style[alias]
                if isinstance(value, dict):
                    return value
                if value is not None:
                    return {"label": value}
        label_keys = [f"{axis_key}_label", f"{axis_key} label", f"{axis_key}-label"]
        for alias in label_keys:
            if alias in axes_style:
                value = axes_style[alias]
                if isinstance(value, dict):
                    return value
                if value is not None:
                    return {"label": value}
        direct = axes_style.get(axis_key[:1]) if axis_key not in axes_style and len(axis_key) > 1 else None
        if isinstance(direct, dict):
            return direct
        if direct is not None:
            return {"label": direct}
        return {}

    options: Dict[str, Any] = {
        "responsive": True,
        "maintainAspectRatio": False,
        "plugins": {
            "legend": {
                "display": True,
                "labels": {"color": palette["text"]},
            },
            "title": {
                "display": False,
                "text": "",
                "color": palette["text"],
            },
            "tooltip": {
                "backgroundColor": palette["background"],
                "titleColor": palette["text"],
                "bodyColor": palette["text"],
            },
        },
        "scales": {
            "x": {
                "ticks": {"color": palette["text"]},
                "grid": {"color": palette["grid"]},
            },
            "y": {
                "ticks": {"color": palette["text"]},
                "grid": {"color": palette["grid"]},
            },
        },
    }

    title_text = chart_stmt.title
    if not title_text:
        title_text = title_style.get("text") or title_style.get("label") or title_style.get("value")
    title_display = title_style.get("show")
    if title_display is None:
        title_display = bool(title_text)
    if title_display and not title_text:
        title_text = chart_stmt.heading
    raw_title_color = title_style.get("color")
    title_color = str(raw_title_color) if raw_title_color is not None else palette["text"]
    title_align = title_style.get("align")
    title_opts = options["plugins"]["title"]
    title_opts["display"] = bool(title_display)
    title_opts["text"] = str(title_text) if title_text is not None else ""
    title_opts["color"] = title_color
    if title_align:
        title_opts["align"] = title_align

    font_updates: Dict[str, Any] = {}
    for key, target in (("size", "size"), ("font_size", "size"), ("family", "family"), ("font_family", "family"), ("weight", "weight"), ("font_weight", "weight"), ("style", "style")):
        val = title_style.get(key)
        if val is None:
            continue
        if target == "size":
            number = _to_number(val)
            if number is not None:
                font_updates[target] = number
        else:
            font_updates[target] = val
    if font_updates:
        title_opts.setdefault("font", {}).update(font_updates)

    padding_value = title_style.get("padding")
    if padding_value is not None:
        number = _to_number(padding_value)
        title_opts["padding"] = number if number is not None else padding_value

    legend_opts = options["plugins"]["legend"]
    legend_display = legend_style.get("show")
    if legend_display is None:
        legend_display = bool(legend_style) or len(normalised) > 1
    legend_opts["display"] = bool(legend_display)
    legend_position = legend_style.get("position")
    if legend_position:
        legend_opts["position"] = str(legend_position)
    legend_align = legend_style.get("align")
    if legend_align:
        legend_opts["align"] = str(legend_align)
    legend_color = legend_style.get("color")
    if legend_color:
        legend_opts.setdefault("labels", {})["color"] = str(legend_color)
    legend_labels = legend_style.get("labels")
    if isinstance(legend_labels, dict):
        legend_opts.setdefault("labels", {}).update(legend_labels)

    def _apply_axis_customisation(axis_key: str) -> None:
        axis_cfg = _extract_axis_config(axis_key)
        if not axis_cfg:
            return
        axis_options = options["scales"].setdefault(axis_key, {})
        label_text_local = axis_cfg.get("label") or axis_cfg.get("text") or axis_cfg.get("title")
        if label_text_local:
            axis_options.setdefault("title", {})
            title_color = axis_cfg.get("title_color") or axis_cfg.get("color") or palette["text"]
            axis_options["title"].update(
                {
                    "display": True,
                    "text": str(label_text_local),
                    "color": str(title_color),
                }
            )
        tick_cfg = axis_cfg.get("ticks") if isinstance(axis_cfg.get("ticks"), dict) else {}
        if tick_cfg:
            axis_options.setdefault("ticks", {}).update(tick_cfg)
        tick_color = axis_cfg.get("tick_color") or (tick_cfg.get("color") if isinstance(tick_cfg, dict) else None) or axis_cfg.get("color")
        if tick_color:
            axis_options.setdefault("ticks", {})["color"] = str(tick_color)
        grid_cfg = axis_cfg.get("grid")
        if grid_cfg is not None:
            axis_options.setdefault("grid", {})
            if isinstance(grid_cfg, dict):
                axis_options["grid"].update(grid_cfg)
            elif isinstance(grid_cfg, bool):
                axis_options["grid"]["display"] = grid_cfg
            else:
                axis_options["grid"]["color"] = str(grid_cfg)
        show_axis = axis_cfg.get("show")
        if show_axis is not None:
            axis_options["display"] = bool(show_axis)
        options["scales"][axis_key] = axis_options

    _apply_axis_customisation("x")
    _apply_axis_customisation("y")

    series_colors = colors_style.get("series")
    if isinstance(series_colors, list) and series_colors:
        for idx, dataset in enumerate(normalised):
            color_value = series_colors[idx % len(series_colors)]
            _apply_dataset_color(dataset, color_value)
    elif isinstance(series_colors, dict):
        for idx, dataset in enumerate(normalised):
            color_value = None
            label = dataset.get("label")
            if label and label in series_colors:
                color_value = series_colors[label]
            elif str(idx) in series_colors:
                color_value = series_colors[str(idx)]
            elif idx in series_colors:
                color_value = series_colors[idx]
            _apply_dataset_color(dataset, color_value)
    elif isinstance(series_colors, str):
        for dataset in normalised:
            _apply_dataset_color(dataset, series_colors)

    tooltip_color = colors_style.get("tooltip")
    if isinstance(tooltip_color, str):
        options["plugins"]["tooltip"]["backgroundColor"] = tooltip_color

    grid_color = colors_style.get("grid")
    if isinstance(grid_color, str):
        for axis in ("x", "y"):
            options["scales"].setdefault(axis, {}).setdefault("grid", {})["color"] = grid_color

    layout = chart_stmt.layout
    if layout and layout.variant and layout.variant.lower() == "card":
        title_opts.setdefault("font", {})
        title_opts["display"] = True if title_style.get("show") is None else title_opts["display"]
        title_opts["font"].setdefault("size", 18)
        title_opts["font"].setdefault("weight", "600")
        legend_opts.setdefault("align", "center")
    if layout and layout.align:
        align_value = layout.align.lower()
        if align_value in {"start", "center", "end"}:
            legend_opts["align"] = align_value
    if layout and layout.emphasis:
        emphasis_value = layout.emphasis.lower()
        if emphasis_value in {"primary", "secondary"}:
            for dataset in normalised:
                if emphasis_value == "primary":
                    dataset["borderWidth"] = max(dataset.get("borderWidth", 1), 2)
                else:
                    dataset["borderWidth"] = max(dataset.get("borderWidth", 1), 1)

    config = {
        "type": chart_type,
        "data": {"labels": labels, "datasets": normalised},
        "options": options,
    }
    return config


def _infer_theme_mode(app: App, page: Page) -> Optional[str]:
    for key in ("theme", "mode", "appearance"):
        if isinstance(page.layout, dict):
            value = page.layout.get(key)
            if value:
                return str(value)
    for key in ("theme", "mode", "appearance"):
        value = app.theme.values.get(key)
        if value:
            return str(value)
    return None


def _style_to_inline(styles: Dict[str, str]) -> str:
    """Convert a style dictionary into an inline CSS string."""
    if not styles:
        return ''
    # Map some semantic names to CSS properties
    css_map = {
        'color': 'color',
        'background': 'background-color',
        'background colour': 'background-color',
        'size': 'font-size',
        'align': 'text-align',
        'weight': 'font-weight',
    }
    parts = []
    for key, value in styles.items():
        if isinstance(value, (dict, list, tuple)):
            continue
        if isinstance(value, bool):
            value = 'true' if value else 'false'
        value_str = str(value)
        prop = css_map.get(key.lower(), key.replace(' ', '-'))
        # convert common size words to CSS sizes
        if prop == 'font-size':
            lower_val = value_str.lower()
            if lower_val == 'small':
                value_str = '0.875rem'
            elif lower_val == 'medium':
                value_str = '1rem'
            elif lower_val == 'large':
                value_str = '1.25rem'
            elif lower_val in {'xlarge', 'xl'}:
                value_str = '1.5rem'
        parts.append(f"{prop}: {value_str};")
    return ' '.join(parts)


def _slugify_route(route: str) -> str:
    """Generate a safe filename from a page route."""
    if route == '/':
        return 'index'
    slug = route.strip('/')
    slug = slug.replace('/', '_')
    return slug or 'index'


def _slugify_page_name(name: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or f"page_{index}"


def _slugify_identifier(value: str, default: str = "insight") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or default


def _render_statements(
    statements: List[PageStatement],
    app: App,
    page: Page,
    slug: str,
    backend_slug: str,
    body_lines: List[str],
    inline_scripts: List[str],
    counters: Dict[str, int],
    widget_defs: List[Dict[str, Any]],
    *,
    theme_mode: Optional[str],
    component_tracker: Dict[str, int],
    scope: str = "page",
) -> None:
    """Render statements into HTML fragments and widget definitions."""

    for stmt in statements:
        counts_for_component = scope == "page" and isinstance(stmt, _PAGE_COMPONENT_TYPES)
        component_index: Optional[int] = None
        if counts_for_component:
            current = component_tracker.get("value", 0)
            component_index = current
            component_tracker["value"] = current + 1
        if isinstance(stmt, ShowText):
            styles = _style_to_inline(stmt.styles)
            template = stmt.text
            initial_text = _TEXT_PLACEHOLDER_RE.sub('', template)
            attrs = [f'data-n3-text-template="{html.escape(template, quote=True)}"']
            if styles:
                attrs.append(f'style="{styles}"')
            attr_str = ' '.join(attrs)
            body_lines.append(f"<p {attr_str}>{html.escape(initial_text)}</p>")

        elif isinstance(stmt, ShowTable):
            table_id = f"table_{slug}_{counters.setdefault('table', 0)}"
            counters['table'] += 1
            columns = stmt.columns or ['Column 1', 'Column 2', 'Column 3']
            table_placeholder = placeholder_utils.placeholder_table_rows(
                columns,
                real_rows=None,
            )
            placeholder_rows = list(table_placeholder.get('rows', []))
            layout_payload = _layout_to_payload(stmt.layout)
            variant = (layout_payload or {}).get('variant', '')
            style_inline = _style_to_inline(stmt.style or {}) if stmt.style else ''

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
                    f'data-n3-endpoint="/api/pages/{backend_slug}/tables/{component_index}"'
                )

            table_classes = ['n3-table']
            if variant.lower() == 'dense':
                table_classes.append('n3-table-dense')

            body_lines.append(f"<section {' '.join(wrapper_attrs)}>")
            body_lines.append(f"  <h3>{html.escape(stmt.title)}</h3>")
            table_class_attr = " ".join(table_classes)
            body_lines.append(
                f"  <div class=\"n3-table-container\"><table id=\"{table_id}\" class=\"{table_class_attr}\"></table></div>"
            )
            body_lines.append("</section>")

            table_data = {
                "title": stmt.title,
                "columns": columns,
                "rows": placeholder_rows,
                "insight": stmt.insight,
                "placeholder": table_placeholder,
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
                table_def["endpoint"] = f"/api/pages/{backend_slug}/tables/{component_index}"
            widget_defs.append(table_def)

        elif isinstance(stmt, ShowChart):
            chart_id = f"chart_{slug}_{counters.setdefault('chart', 0)}"
            counters['chart'] += 1
            layout_payload = _layout_to_payload(stmt.layout)
            style_inline = _style_to_inline(stmt.style or {}) if stmt.style else ''

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
                    f'data-n3-endpoint="/api/pages/{backend_slug}/charts/{component_index}"'
                )

            body_lines.append(f"<section {' '.join(wrapper_attrs)}>")
            body_lines.append(f"  <h3>{html.escape(stmt.heading)}</h3>")
            body_lines.append(
                f"  <div class=\"n3-chart-container\"><canvas id=\"{chart_id}\" class=\"n3-chart\"></canvas></div>"
            )
            body_lines.append("</section>")

            chart_kind = (stmt.chart_type or "bar").lower()
            chart_placeholder = placeholder_utils.build_placeholder_chart_payload(
                chart_kind,
                real_payload=None,
            )
            dataset_payload: Dict[str, Any] = {}
            if chart_placeholder.get("status") == "ok":
                dataset_payload = chart_placeholder["payload"]
            else:
                dataset_payload = {"labels": [], "datasets": []}
            chart_config = build_chart_config(stmt, dataset_payload, theme=theme_mode)
            if stmt.insight:
                chart_config['insight'] = stmt.insight
            chart_def: Dict[str, Any] = {
                "type": "chart",
                "id": chart_id,
                "config": chart_config,
                "placeholder": chart_placeholder,
                "layout": layout_payload or {},
                "heading": stmt.heading,
                "style": stmt.style or {},
                "legend": stmt.legend or {},
                "title": stmt.title,
                "insight": stmt.insight,
            }
            if component_index is not None:
                chart_def["componentIndex"] = component_index
                chart_def["endpoint"] = f"/api/pages/{backend_slug}/charts/{component_index}"
            widget_defs.append(chart_def)

        elif isinstance(stmt, ShowForm):
            form_id = f"form_{slug}_{counters.setdefault('form', 0)}"
            counters['form'] += 1
            styles = _style_to_inline(stmt.styles)
            body_lines.append(f"<h3>{html.escape(stmt.title)}</h3>")
            body_lines.append(f"<form id=\"{form_id}\" style=\"{styles}\">")
            for field in stmt.fields:
                field_type = field.field_type or 'text'
                body_lines.append(
                    f"  <label>{html.escape(field.name)}: <input name=\"{field.name}\" type=\"{field_type}\"></label><br>"
                )
            body_lines.append("  <button type=\"submit\">Submit</button>")
            body_lines.append("</form>")

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
                    target_slug = _slugify_route(
                        next((p.route for p in app.pages if p.name == op.page_name), op.page_name)
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
            inline_scripts.append('\n'.join(handler_lines))
        elif isinstance(stmt, Action):
            button_label = stmt.name
            match = re.search(r'clicks\s+"([^"]+)"', stmt.trigger)
            if match:
                button_label = match.group(1)
            btn_id = f"action_btn_{slug}_{counters.setdefault('action', 0)}"
            counters['action'] += 1
            body_lines.append(f"<button id=\"{btn_id}\">{html.escape(button_label)}</button>")

            handler_lines = [
                f"var btnEl = document.getElementById('{btn_id}');",
                "if (btnEl) {",
                "  btnEl.addEventListener('click', function() {",
            ]
            for op in stmt.operations:
                if isinstance(op, ToastOperation):
                    handler_lines.append(f"    window.N3Widgets.showToast({json.dumps(op.message)});")
                elif isinstance(op, GoToPageOperation):
                    target_slug = _slugify_route(
                        next((p.route for p in app.pages if p.name == op.page_name), op.page_name)
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
            inline_scripts.append('\n'.join(handler_lines))

        elif isinstance(stmt, IfBlock):
            condition_text = getattr(stmt.condition, 'raw', repr(stmt.condition))
            body_lines.append(f"<!-- If condition: {condition_text} -->")
            body_lines.append(
                "<div class=\"if-block\" style=\"border-left: 3px solid #4CAF50; padding-left: 10px; margin: 10px 0;\">"
            )
            body_lines.append(f"<p><em>If {condition_text}:</em></p>")
            _render_statements(
                stmt.body,
                app,
                page,
                slug,
                backend_slug,
                body_lines,
                inline_scripts,
                counters,
                widget_defs,
                theme_mode=theme_mode,
                component_tracker=component_tracker,
                scope="nested",
            )
            body_lines.append("</div>")

            if stmt.else_body:
                body_lines.append(
                    "<div class=\"else-block\" style=\"border-left: 3px solid #FF9800; padding-left: 10px; margin: 10px 0;\">"
                )
                body_lines.append("<p><em>Else:</em></p>")
                _render_statements(
                    stmt.else_body,
                    app,
                    page,
                    slug,
                    backend_slug,
                    body_lines,
                    inline_scripts,
                    counters,
                    widget_defs,
                    theme_mode=theme_mode,
                    component_tracker=component_tracker,
                    scope="nested",
                )
                body_lines.append("</div>")

        elif isinstance(stmt, ForLoop):
            body_lines.append(f"<!-- For loop: {stmt.loop_var} in {stmt.source_kind} {stmt.source_name} -->")
            body_lines.append(
                "<div class=\"for-loop\" style=\"border-left: 3px solid #2196F3; padding-left: 10px; margin: 10px 0;\">"
            )
            body_lines.append(f"<p><em>For each {stmt.loop_var} in {stmt.source_kind} {stmt.source_name}:</em></p>")
            for index in range(3):
                body_lines.append(
                    "<div class=\"loop-iteration\" style=\"margin-left: 10px; margin-bottom: 5px;\">"
                )
                body_lines.append(f"<small>Iteration {index + 1}:</small>")
                _render_statements(
                    stmt.body,
                    app,
                    page,
                    slug,
                    backend_slug,
                    body_lines,
                    inline_scripts,
                    counters,
                    widget_defs,
                    theme_mode=theme_mode,
                    component_tracker=component_tracker,
                    scope="nested",
                )
                body_lines.append("</div>")
            body_lines.append("</div>")

        else:
            body_lines.append(f"<!-- Unknown statement type: {type(stmt).__name__} -->")


def generate_site(app: App, output_dir: str, *, enable_realtime: bool = False) -> None:
    """Write a static representation of the app to ``output_dir``."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / 'styles.css').write_text(_generate_styles(app), encoding='utf-8')

    first_page_path: Optional[str] = None
    for idx, page in enumerate(app.pages):
        slug = _slugify_route(page.route)
        html_content = _generate_page_html(app, page, slug, idx, enable_realtime=enable_realtime)
        (out / f'{slug}.html').write_text(html_content, encoding='utf-8')
        if idx == 0:
            first_page_path = f'{slug}.html'

    if first_page_path is not None:
        index_html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>{html.escape(app.name)}</title>
    <link rel=\"stylesheet\" href=\"styles.css\">
    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
</head>
<body>
    <h1>{html.escape(app.name)}</h1>
    <p><a href=\"{first_page_path}\">Go to application</a></p>
</body>
</html>
"""
        (out / 'index.html').write_text(index_html, encoding='utf-8')

    (out / 'scripts.js').write_text(_generate_widget_library(), encoding='utf-8')


def _generate_styles(app: App) -> str:
    """Generate CSS from the app's theme.  Provides default styling."""
    theme_vars = [f"  --{key.replace(' ', '-')}: {value};" for key, value in app.theme.values.items()]
    # Build theme block separately to avoid backslash in f-string expressions
    if theme_vars:
        theme_block = ":root {\n" + "\n".join(theme_vars) + "\n}"
    else:
        theme_block = ''
    base_css = f"""
body {{
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 1rem;
  background-color: var(--background, #ffffff);
  color: var(--text, #333333);
}}
table {{
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 1rem;
}}
th, td {{
  border: 1px solid #ccc;
  padding: 0.5rem;
  text-align: left;
}}
.n3-widget {{
    margin-bottom: 1.5rem;
    padding: 1rem;
    border-radius: 0.75rem;
    border: 1px solid rgba(15, 23, 42, 0.08);
    background-color: var(--panel-bg, #ffffff);
    box-shadow: var(--widget-shadow, 0 12px 24px rgba(15, 23, 42, 0.05));
}}
.n3-widget--card {{
    box-shadow: 0 18px 32px rgba(15, 23, 42, 0.12);
}}
.n3-widget-chart,
.n3-widget-table {{
    background-color: var(--surface, #ffffff);
}}
.n3-align-center {{
    text-align: center;
}}
.n3-align-right {{
    text-align: right;
}}
.n3-emphasis-primary {{
    border-color: var(--primary, #2563eb);
}}
.n3-emphasis-secondary {{
    border-color: var(--secondary, #6366f1);
}}
.n3-table-container {{
    overflow-x: auto;
}}
.n3-table {{
    width: 100%;
    border-collapse: collapse;
}}
.n3-table-dense td,
.n3-table-dense th {{
    padding: 0.35rem;
}}
.n3-chart {{
    width: 100%;
    height: 320px;
    display: block;
}}
.toast {{
  position: fixed;
  bottom: 1rem;
  right: 1rem;
  background-color: #333;
  color: #fff;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  opacity: 0;
  transition: opacity 0.5s;
}}
.toast.show {{
  opacity: 1;
}}
.n3-insights {{
    margin-top: 2rem;
}}
.n3-insights > h3 {{
    margin-bottom: 0.75rem;
}}
.n3-insight-grid {{
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}}
.n3-insight-card {{
    border: 1px solid rgba(15, 23, 42, 0.12);
    border-radius: 0.75rem;
    padding: 1rem;
    background-color: var(--panel-bg, #ffffff);
    box-shadow: var(--widget-shadow, 0 12px 24px rgba(15, 23, 42, 0.04));
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}}
.n3-insight-card__header h4 {{
    margin: 0;
    font-size: 1.05rem;
    font-weight: 600;
}}
.n3-insight-metrics {{
    display: grid;
    gap: 0.5rem;
}}
.n3-insight-metric {{
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 0.5rem;
    padding: 0.75rem;
    background-color: var(--surface, #f9fafb);
}}
.n3-insight-metric__label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted, #6b7280);
    margin-bottom: 0.25rem;
}}
.n3-insight-metric__value {{
    font-size: 1.55rem;
    font-weight: 600;
    color: var(--text, #1f2937);
}}
.n3-insight-metric__trend,
.n3-insight-metric__status {{
    font-size: 0.85rem;
    color: var(--accent, #2563eb);
    margin-top: 0.35rem;
}}
.n3-insight-metric--alert {{
    border-color: var(--warning, #f97316);
}}
.n3-insight-narratives {{
    display: grid;
    gap: 0.5rem;
}}
.n3-insight-narrative {{
    border-radius: 0.5rem;
    padding: 0.75rem;
    border: 1px dashed rgba(15, 23, 42, 0.18);
    background-color: var(--surface-alt, #ffffff);
}}
.n3-insight-empty {{
    font-size: 0.85rem;
    color: var(--text-muted, #6b7280);
}}
.n3-insight-card--error {{
    border-color: var(--warning, #ef4444);
}}
.n3-insight-card--error::after {{
    content: attr(data-error);
    display: block;
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: var(--warning, #ef4444);
}}
"""
    return '\n'.join([theme_block, base_css]) if theme_block else base_css


def _generate_widget_library() -> str:
    """Return the shared widget runtime responsible for rendering charts and tables."""

    return textwrap.dedent(
        """
        (function(global) {
            'use strict';

            var widgets = global.N3Widgets || (global.N3Widgets = {});

            widgets.applyLayout = function(element, layout) {
                if (!element || !layout) {
                    return;
                }
                if (layout.variant) {
                    element.classList.add('n3-widget--' + String(layout.variant).toLowerCase());
                }
                if (layout.align) {
                    element.classList.add('n3-align-' + String(layout.align).toLowerCase());
                }
                if (layout.emphasis) {
                    element.classList.add('n3-emphasis-' + String(layout.emphasis).toLowerCase());
                }
            };

            widgets.resolvePath = function(path, data) {
                if (!path) {
                    return undefined;
                }
                var cleaned = String(path).trim();
                if (!cleaned) {
                    return undefined;
                }
                var target = data;
                var segments = cleaned.replace(/\\[(\\d+)\\]/g, '.$1').split('.');
                for (var i = 0; i < segments.length; i++) {
                    var key = segments[i];
                    if (!key) {
                        continue;
                    }
                    if (target == null) {
                        return undefined;
                    }
                    if (Object.prototype.hasOwnProperty.call(target, key)) {
                        target = target[key];
                        continue;
                    }
                    if (typeof target === 'object' && key in target) {
                        target = target[key];
                    } else {
                        return undefined;
                    }
                }
                return target;
            };

            widgets.interpolate = function(template, data) {
                if (!template) {
                    return '';
                }
                return String(template).replace(/\\{([^{}]+)\\}/g, function(match, token) {
                    var value = widgets.resolvePath(token, data);
                    return value == null ? '' : String(value);
                });
            };

            widgets.showToast = function(message, duration) {
                var toast = document.getElementById('toast');
                if (!toast) {
                    return;
                }
                toast.textContent = message;
                toast.classList.add('show');
                setTimeout(function() {
                    toast.classList.remove('show');
                }, duration || 3000);
            };

            if (!global.showToast) {
                global.showToast = widgets.showToast;
            }

            widgets.renderChart = function(canvasId, config, layout, insightName) {
                var canvas = document.getElementById(canvasId);
                if (!canvas || typeof Chart === 'undefined') {
                    return;
                }
                widgets.applyLayout(canvas, layout);
                var ctx = canvas.getContext('2d');
                if (!ctx) {
                    return;
                }
                if (canvas.__n3_chart__) {
                    canvas.__n3_chart__.destroy();
                }
                var insightRef = insightName || (config && config.insight);
                if (insightRef) {
                    canvas.setAttribute('data-n3-insight-ref', insightRef);
                }
                canvas.__n3_chart__ = new Chart(ctx, config);
            };

            widgets.renderTable = function(tableId, data, layout, insightName) {
                var table = document.getElementById(tableId);
                if (!table) {
                    return;
                }
                widgets.applyLayout(table, layout);
                var columns = Array.isArray(data && data.columns) ? data.columns.slice() : [];
                var rows = Array.isArray(data && data.rows) ? data.rows : [];
                table.innerHTML = '';

                if (insightName) {
                    table.setAttribute('data-n3-insight-ref', insightName);
                } else if (data && data.insight) {
                    table.setAttribute('data-n3-insight-ref', data.insight);
                }
                if (!columns.length && rows.length && typeof rows[0] === 'object' && rows[0] !== null) {
                    columns = Object.keys(rows[0]);
                }

                if (columns.length) {
                    var thead = document.createElement('thead');
                    var headerRow = document.createElement('tr');
                    columns.forEach(function(column) {
                        var th = document.createElement('th');
                        th.textContent = column;
                        headerRow.appendChild(th);
                    });
                    thead.appendChild(headerRow);
                    table.appendChild(thead);
                }

                var tbody = document.createElement('tbody');
                rows.forEach(function(row) {
                    var tr = document.createElement('tr');
                    columns.forEach(function(column, index) {
                        var td = document.createElement('td');
                        if (row && typeof row === 'object' && row !== null && Object.prototype.hasOwnProperty.call(row, column)) {
                            td.textContent = row[column];
                        } else if (Array.isArray(row)) {
                            td.textContent = row[index] != null ? row[index] : '';
                        } else {
                            td.textContent = '';
                        }
                        tr.appendChild(td);
                    });
                    tbody.appendChild(tr);
                });
                table.appendChild(tbody);
            };

            function getRegistry() {
                if (!widgets.__registry__) {
                    widgets.__registry__ = {};
                }
                return widgets.__registry__;
            }

            widgets.chartResponseToConfig = function(response, fallback) {
                if (!response) {
                    return fallback || {
                        type: 'line',
                        data: { labels: [], datasets: [] },
                        options: (fallback && fallback.options) || {},
                    };
                }
                if (response.type && response.data && Array.isArray(response.data.datasets)) {
                    return response;
                }
                var chartType = response.chart_type || (fallback && fallback.type) || 'line';
                var labels = Array.isArray(response.labels) ? response.labels.slice() : [];
                if (!labels.length && fallback && fallback.data && Array.isArray(fallback.data.labels)) {
                    labels = fallback.data.labels.slice();
                }
                var series = Array.isArray(response.series) ? response.series : [];
                var datasets = series.map(function(entry) {
                    var next = Object.assign({}, entry || {});
                    if (!Array.isArray(next.data)) {
                        next.data = [];
                    }
                    if (!next.label) {
                        next.label = response.title || 'Series';
                    }
                    return next;
                });
                if (!datasets.length && fallback && fallback.data && Array.isArray(fallback.data.datasets)) {
                    datasets = fallback.data.datasets.map(function(item) {
                        return Object.assign({}, item);
                    });
                }
                if (!datasets.length) {
                    datasets = [{ label: response.title || 'Series', data: [] }];
                }
                return {
                    type: chartType,
                    data: {
                        labels: labels,
                        datasets: datasets,
                    },
                    options: (fallback && fallback.options) || {},
                };
            };

            widgets.registerComponent = function(def) {
                if (!def || typeof def.componentIndex !== 'number') {
                    return;
                }
                var registry = getRegistry();
                var index = def.componentIndex;
                var entry = registry[index] || {};
                entry.type = def.type;
                entry.id = def.id;
                entry.layout = def.layout || {};
                entry.insight = def.insight || null;
                entry.initial = def;
                entry.lastSnapshot = entry.lastSnapshot || null;
                entry.previousSnapshot = entry.previousSnapshot || null;
                entry.render = function(payload) {
                    if (entry.type === 'chart') {
                        var config = widgets.chartResponseToConfig(payload, def.config || null);
                        widgets.renderChart(entry.id, config, entry.layout, entry.insight);
                    } else if (entry.type === 'table') {
                        var tableData = payload && payload.rows ? payload : (payload || def.data || {});
                        widgets.renderTable(entry.id, tableData, entry.layout, entry.insight);
                    }
                };
                registry[index] = entry;
            };

            widgets.rememberSnapshot = function(index, snapshot) {
                var registry = getRegistry();
                var entry = registry[index];
                if (!entry) {
                    return;
                }
                try {
                    entry.lastSnapshot = snapshot == null ? snapshot : JSON.parse(JSON.stringify(snapshot));
                } catch (err) {
                    entry.lastSnapshot = snapshot;
                }
            };

            widgets.updateComponent = function(index, payload, meta) {
                var registry = getRegistry();
                var entry = registry[index];
                if (!entry) {
                    return;
                }
                if (entry.lastSnapshot != null) {
                    try {
                        entry.previousSnapshot = JSON.parse(JSON.stringify(entry.lastSnapshot));
                    } catch (err) {
                        entry.previousSnapshot = entry.lastSnapshot;
                    }
                } else {
                    entry.previousSnapshot = null;
                }
                entry.lastSnapshot = payload;
                entry.render(payload);
                if (meta) {
                    entry.meta = meta;
                }
            };

            widgets.rollbackComponent = function(index) {
                var registry = getRegistry();
                var entry = registry[index];
                if (!entry) {
                    return;
                }
                var snapshot = entry.previousSnapshot;
                if (!snapshot) {
                    return;
                }
                entry.previousSnapshot = null;
                entry.lastSnapshot = snapshot;
                entry.render(snapshot);
            };

            widgets.applyRealtimeUpdate = function(event) {
                if (!event || typeof event !== 'object') {
                    return;
                }
                if (event.type === 'component') {
                    var componentIndex = typeof event.component_index === 'number'
                        ? event.component_index
                        : parseInt(event.component_index, 10);
                    if (!isNaN(componentIndex)) {
                        widgets.updateComponent(componentIndex, event.payload || {}, event.meta || {});
                    }
                    return;
                }
                if (event.type === 'rollback') {
                    var rollbackIndex = typeof event.component_index === 'number'
                        ? event.component_index
                        : parseInt(event.component_index, 10);
                    if (!isNaN(rollbackIndex)) {
                        widgets.rollbackComponent(rollbackIndex);
                    }
                    return;
                }
                if (event.type === 'snapshot' || event.type === 'hydration') {
                    if (event.payload && typeof event.payload === 'object') {
                        global.N3_PAGE_STATE = event.payload;
                    }
                }
            };

            widgets.populateMetrics = function(container, metrics) {
                if (!container) {
                    return;
                }
                container.innerHTML = '';
                if (!Array.isArray(metrics) || !metrics.length) {
                    var empty = document.createElement('div');
                    empty.className = 'n3-insight-empty';
                    empty.textContent = 'No metrics available.';
                    container.appendChild(empty);
                    return;
                }
                metrics.forEach(function(metric) {
                    if (!metric) {
                        return;
                    }
                    var card = document.createElement('div');
                    card.className = 'n3-insight-metric';
                    if (Array.isArray(metric.alerts) && metric.alerts.some(function(alert) { return alert && alert.triggered; })) {
                        card.classList.add('n3-insight-metric--alert');
                    }
                    var label = document.createElement('div');
                    label.className = 'n3-insight-metric__label';
                    label.textContent = metric.label || metric.name || 'Metric';
                    card.appendChild(label);

                    var value = document.createElement('div');
                    value.className = 'n3-insight-metric__value';
                    if (metric.formatted) {
                        value.textContent = metric.formatted;
                    } else if (metric.value != null) {
                        value.textContent = String(metric.value);
                    } else {
                        value.textContent = '—';
                    }
                    card.appendChild(value);

                    var trendText = '';
                    if (typeof metric.delta === 'number') {
                        trendText += (metric.delta >= 0 ? '+' : '') + metric.delta.toFixed(2);
                    }
                    if (typeof metric.delta_pct === 'number') {
                        if (trendText) {
                            trendText += ' (';
                        }
                        trendText += (metric.delta_pct >= 0 ? '+' : '') + metric.delta_pct.toFixed(1) + '%';
                        if (trendText.indexOf('(') !== -1) {
                            trendText += ')';
                        }
                    }
                    if (!trendText && metric.trend) {
                        trendText = metric.trend;
                    }
                    if (!trendText && metric.status) {
                        trendText = String(metric.status);
                    }
                    if (trendText) {
                        var trend = document.createElement('div');
                        trend.className = metric.trend ? 'n3-insight-metric__trend' : 'n3-insight-metric__status';
                        trend.textContent = trendText;
                        card.appendChild(trend);
                    }

                    container.appendChild(card);
                });
            };

            widgets.populateNarratives = function(container, narratives, templateData) {
                if (!container) {
                    return;
                }
                container.innerHTML = '';
                if (!Array.isArray(narratives) || !narratives.length) {
                    return;
                }
                narratives.forEach(function(narrative) {
                    if (!narrative) {
                        return;
                    }
                    var block = document.createElement('div');
                    block.className = 'n3-insight-narrative';
                    if (narrative.variant) {
                        block.classList.add('n3-insight-narrative--' + String(narrative.variant).toLowerCase());
                    }
                    if (narrative.style && typeof narrative.style === 'object') {
                        Object.keys(narrative.style).forEach(function(key) {
                            var cssKey = key.replace(/[A-Z]/g, function(match) {
                                return '-' + match.toLowerCase();
                            });
                            block.style.setProperty(cssKey, narrative.style[key]);
                        });
                    }
                    var text = narrative.text;
                    if (!text && narrative.template) {
                        text = widgets.interpolate(narrative.template, templateData);
                    }
                    block.textContent = text || '';
                    container.appendChild(block);
                });
            };

            widgets.renderInsight = function(containerId, spec) {
                var container = document.getElementById(containerId);
                if (!container) {
                    return;
                }
                widgets.applyLayout(container, spec && spec.layout);
                var slug = spec && spec.slug;
                var endpoint = spec && spec.endpoint;
                if (!endpoint && slug) {
                    endpoint = '/api/insights/' + slug;
                }
                if (!endpoint) {
                    container.classList.add('n3-insight-card--error');
                    container.setAttribute('data-error', 'Missing insight endpoint');
                    return;
                }
                fetch(endpoint, { headers: { 'Accept': 'application/json' } })
                    .then(function(response) {
                        if (!response.ok) {
                            throw new Error('Failed to load insight (' + response.status + ')');
                        }
                        return response.json();
                    })
                    .then(function(payload) {
                        var result = payload && payload.result ? payload.result : {};
                        var metrics = Array.isArray(result.metrics) ? result.metrics : [];
                        var metricMap = {};
                        metrics.forEach(function(metric) {
                            if (metric && metric.name) {
                                metricMap[metric.name] = metric;
                            }
                        });
                        var templateData = Object.assign({}, result, {
                            metrics: metricMap,
                            variables: result.variables || result.expose || result.expose_as || {},
                            alerts: result.alerts_list || [],
                            selection: result.selection || [],
                            events: result.events || [],
                        });
                        widgets.populateMetrics(container.querySelector('[data-n3-insight="metrics"]'), metrics);
                        widgets.populateNarratives(
                            container.querySelector('[data-n3-insight="narratives"]'),
                            Array.isArray(result.narratives) ? result.narratives : [],
                            templateData
                        );
                    })
                    .catch(function(err) {
                        container.classList.add('n3-insight-card--error');
                        container.setAttribute('data-error', err && err.message ? err.message : 'Insight failed');
                    });
            };

            widgets.bootstrap = function(definitions) {
                if (!Array.isArray(definitions)) {
                    return;
                }
                definitions.forEach(function(def) {
                    if (!def || !def.id) {
                        return;
                    }
                    var hasIndex = typeof def.componentIndex === 'number';
                    if (hasIndex) {
                        widgets.registerComponent(def);
                    }
                    if (def.type === 'chart') {
                        if (def.insight && def.config && typeof def.config === 'object') {
                            def.config.insight = def.insight;
                        }
                        widgets.renderChart(def.id, def.config || {}, def.layout || {}, def.insight || null);
                        if (hasIndex) {
                            widgets.rememberSnapshot(def.componentIndex, def.config || {});
                        }
                    } else if (def.type === 'table') {
                        if (def.insight && def.data && typeof def.data === 'object') {
                            def.data.insight = def.insight;
                        }
                        widgets.renderTable(def.id, def.data || {}, def.layout || {}, def.insight || null);
                        if (hasIndex) {
                            widgets.rememberSnapshot(def.componentIndex, def.data || {});
                        }
                    } else if (def.type === 'insight') {
                        widgets.renderInsight(def.id, def);
                    }
                });
            };
        })(window);

        (function(global) {
            'use strict';

            var realtime = global.N3Realtime || (global.N3Realtime = {});
            var connections = realtime.__connections || (realtime.__connections = {});
            var stateStore = realtime.__state || (realtime.__state = {});

            function toIntervalMs(value) {
                if (typeof value === 'number' && value > 0) {
                    return value * 1000;
                }
                var parsed = parseInt(value, 10);
                if (!isNaN(parsed) && parsed > 0) {
                    return parsed * 1000;
                }
                return null;
            }

            function dispatchEvent(name, detail) {
                if (typeof document === 'undefined' || typeof document.dispatchEvent !== 'function') {
                    return;
                }
                try {
                    var evt;
                    if (typeof CustomEvent === 'function') {
                        evt = new CustomEvent(name, { detail: detail });
                    } else {
                        evt = document.createEvent('CustomEvent');
                        evt.initCustomEvent(name, true, true, detail);
                    }
                    document.dispatchEvent(evt);
                } catch (err) {
                    console.warn('N3Realtime dispatch failed', err);
                }
            }

            function normalizePath(path) {
                if (typeof path !== 'string') {
                    return '';
                }
                if (!path) {
                    return '';
                }
                return path.charAt(0) === '/' ? path : '/' + path;
            }

            function buildWsUrl(slug, options) {
                if (!slug) {
                    return null;
                }
                var explicit = options && options.wsUrl;
                if (explicit) {
                    return explicit;
                }
                if (typeof window === 'undefined' || !window.location || !window.location.host) {
                    return null;
                }
                var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                var host = window.location.host;
                var base = protocol + '//' + host;
                var path = options && options.wsPath ? options.wsPath : '/ws/pages/' + encodeURIComponent(slug);
                return base.replace(/[/]$/, '') + normalizePath(path);
            }

            function buildPageUrl(slug, options) {
                var path = options && options.pageUrl ? options.pageUrl : '/api/pages/' + slug;
                var base = options && options.baseUrl;
                if (base) {
                    return base.replace(/[/]$/, '') + normalizePath(path);
                }
                return normalizePath(path);
            }

            function getConnection(slug) {
                var state = connections[slug];
                if (!state) {
                    state = connections[slug] = {
                        slug: slug,
                        active: false,
                        retries: 0,
                        fallbackIntervalMs: null,
                        reconnectTimer: null,
                        fallbackTimer: null,
                        websocket: null,
                        options: {},
                    };
                }
                return state;
            }

            function stopReconnect(state) {
                if (state.reconnectTimer) {
                    clearTimeout(state.reconnectTimer);
                    state.reconnectTimer = null;
                }
            }

            function stopFallback(state) {
                if (state.fallbackTimer) {
                    clearInterval(state.fallbackTimer);
                    state.fallbackTimer = null;
                }
            }

            function fetchSnapshot(slug, state, reason) {
                if (typeof fetch !== 'function') {
                    return;
                }
                var baseUrl = buildPageUrl(slug, state.options);
                if (!baseUrl) {
                    return;
                }
                var separator = baseUrl.indexOf('?') === -1 ? '?' : '&';
                var url = baseUrl + separator + '_ts=' + Date.now();
                fetch(url, { headers: { 'Accept': 'application/json' } })
                    .then(function(response) {
                        if (!response.ok) {
                            throw new Error('HTTP ' + response.status);
                        }
                        return response.json();
                    })
                    .then(function(payload) {
                        realtime.applyEvent(slug, {
                            type: 'snapshot',
                            slug: slug,
                            payload: payload,
                            meta: { source: reason || 'poll' },
                        });
                    })
                    .catch(function(err) {
                        console.warn('N3Realtime polling failed for ' + slug + ':', err);
                    });
            }

            function startFallback(slug, state, reason) {
                stopFallback(state);
                if (!state.fallbackIntervalMs) {
                    return;
                }
                fetchSnapshot(slug, state, reason || 'fallback-start');
                state.fallbackTimer = setInterval(function() {
                    fetchSnapshot(slug, state, 'fallback-tick');
                }, state.fallbackIntervalMs);
            }

            function applyEvent(slug, event) {
                if (!event || typeof event !== 'object') {
                    return;
                }
                if (!event.slug) {
                    event.slug = slug;
                }
                if (event.type === 'snapshot' || event.type === 'hydration') {
                    stateStore[slug] = event.payload || {};
                }
                if (global.N3Widgets && typeof global.N3Widgets.applyRealtimeUpdate === 'function') {
                    try {
                        global.N3Widgets.applyRealtimeUpdate(event);
                    } catch (err) {
                        console.error('N3Realtime failed to update widgets', err);
                    }
                }
                dispatchEvent('n3:realtime:' + (event.type || 'message'), {
                    slug: slug,
                    event: event,
                });
            }

            function handleMessage(slug, state, raw) {
                var data = raw;
                if (typeof raw === 'string') {
                    try {
                        data = JSON.parse(raw);
                    } catch (err) {
                        console.warn('N3Realtime received non-JSON message', raw);
                        return;
                    }
                }
                applyEvent(slug, data);
            }

            function scheduleReconnect(slug, state) {
                stopReconnect(state);
                if (!state.active) {
                    return;
                }
                state.retries += 1;
                // simple exponential backoff with an upper bound
                var delay = Math.min(30000, Math.pow(2, state.retries) * 250);
                state.reconnectTimer = setTimeout(function() {
                    openWebSocket(slug, state);
                }, delay);
                if (state.fallbackIntervalMs) {
                    startFallback(slug, state, 'reconnect-wait');
                }
            }

            function openWebSocket(slug, state) {
                stopReconnect(state);
                if (!state.active) {
                    return;
                }
                var url = buildWsUrl(slug, state.options);
                if (!url) {
                    if (state.fallbackIntervalMs) {
                        startFallback(slug, state, 'no-websocket');
                    }
                    return;
                }
                try {
                    var socket = new WebSocket(url);
                    state.websocket = socket;
                    socket.onopen = function() {
                        state.retries = 0;
                        stopFallback(state);
                        dispatchEvent('n3:realtime:connected', { slug: slug });
                    };
                    socket.onmessage = function(evt) {
                        handleMessage(slug, state, evt.data);
                    };
                    socket.onerror = function(err) {
                        console.warn('N3Realtime websocket error for ' + slug + ':', err);
                    };
                    socket.onclose = function() {
                        state.websocket = null;
                        dispatchEvent('n3:realtime:disconnected', { slug: slug });
                        if (state.active) {
                            scheduleReconnect(slug, state);
                        } else {
                            stopFallback(state);
                        }
                    };
                } catch (err) {
                    console.warn('N3Realtime failed to open websocket for ' + slug + ':', err);
                    if (state.fallbackIntervalMs) {
                        startFallback(slug, state, 'websocket-error');
                    }
                }
            }

            realtime.connectPage = function(slug, options) {
                if (!slug) {
                    return;
                }
                var state = getConnection(slug);
                state.active = true;
                state.options = options || {};
                state.fallbackIntervalMs = toIntervalMs(state.options.fallbackInterval);
                if (state.websocket && (state.websocket.readyState === 0 || state.websocket.readyState === 1)) {
                    return;
                }
                var skipWebSocket = typeof window !== 'undefined' && window.location && window.location.protocol === 'file:';
                if (skipWebSocket) {
                    startFallback(slug, state, 'file-protocol');
                    return;
                }
                openWebSocket(slug, state);
                if (!state.websocket && state.fallbackIntervalMs) {
                    startFallback(slug, state, 'websocket-unavailable');
                }
            };

            realtime.disconnectPage = function(slug) {
                var state = connections[slug];
                if (!state) {
                    return;
                }
                state.active = false;
                stopReconnect(state);
                stopFallback(state);
                if (state.websocket && state.websocket.readyState <= 1) {
                    try {
                        state.websocket.close();
                    } catch (err) {
                        // ignore close errors
                    }
                }
                state.websocket = null;
            };

            realtime.applyEvent = function(slug, event) {
                applyEvent(slug, event);
            };

            realtime.applySnapshot = function(slug, payload, meta) {
                applyEvent(slug, {
                    type: 'snapshot',
                    slug: slug,
                    payload: payload || {},
                    meta: meta || {},
                });
            };

            realtime.getState = function(slug) {
                return stateStore[slug];
            };
        })(window);
        """
    ).strip()


def _generate_page_html(
    app: App,
    page: Page,
    slug: str,
    page_index: int,
    *,
    enable_realtime: bool = False,
) -> str:
    body_lines: List[str] = []
    inline_scripts: List[str] = []
    widget_defs: List[Dict[str, Any]] = []
    counters: Dict[str, int] = {'chart': 0, 'action': 0, 'form': 0, 'table': 0}
    component_tracker: Dict[str, int] = {'value': 0}
    backend_slug = _slugify_page_name(page.name, page_index)
    theme_mode = _infer_theme_mode(app, page)

    body_lines.append(f"<h2>{html.escape(page.name)}</h2>")

    _render_statements(
        page.statements,
        app,
        page,
        slug,
        backend_slug,
        body_lines,
        inline_scripts,
        counters,
        widget_defs,
        theme_mode=theme_mode,
        component_tracker=component_tracker,
    )

    if app.insights:
        insights_section_id = f"insights_{slug}"
        body_lines.append(f'<section class="n3-insights" id="{insights_section_id}">')
        body_lines.append('  <h3>Insights</h3>')
        body_lines.append('  <div class="n3-insight-grid">')
        for idx, insight in enumerate(app.insights):
            widget_id = f"insight_{slug}_{idx}"
            insight_slug = _slugify_identifier(insight.name)
            body_lines.append(f'    <article class="n3-insight-card" id="{widget_id}">')
            body_lines.append('      <header class="n3-insight-card__header">')
            body_lines.append(f'        <h4>{html.escape(insight.name)}</h4>')
            body_lines.append('      </header>')
            body_lines.append('      <div class="n3-insight-metrics" data-n3-insight="metrics"></div>')
            body_lines.append('      <div class="n3-insight-narratives" data-n3-insight="narratives"></div>')
            body_lines.append('    </article>')
            widget_defs.append({
                "type": "insight",
                "id": widget_id,
                "slug": insight_slug,
                "title": insight.name,
                "endpoint": f"/api/insights/{insight_slug}",
            })
        body_lines.append('  </div>')
        body_lines.append('</section>')

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        "  <meta charset=\"UTF-8\">",
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">",
        f"  <title>{html.escape(page.name)} – {html.escape(app.name)}</title>",
        "  <link rel=\"stylesheet\" href=\"styles.css\">",
        "  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>",
        "</head>",
    ]
    body_attrs = [
        f'data-n3-page-slug="{backend_slug}"',
        f'data-n3-page-reactive="{"true" if page.reactive else "false"}"',
    ]
    if page.refresh_policy and getattr(page.refresh_policy, 'interval_seconds', None):
        body_attrs.append(
            f'data-n3-refresh-interval="{page.refresh_policy.interval_seconds}"'
        )

    html_parts.append(f"<body {' '.join(body_attrs)}>")

    if len(app.pages) > 1:
        html_parts.append("<nav><ul>")
        for p in app.pages:
            target_slug = _slugify_route(p.route)
            html_parts.append(f"  <li><a href=\"{target_slug}.html\">{html.escape(p.name)}</a></li>")
        html_parts.append("</ul></nav>")

    html_parts.extend(body_lines)
    html_parts.append('<div id="toast" class="toast"></div>')
    html_parts.append('<script src="scripts.js"></script>')

    bootstrap_template = Template(
        textwrap.dedent(
            """
            <script>
            (function() {
                var apiUrl = "$api_url";
                window.N3_VARS = window.N3_VARS || {};
                fetch(apiUrl, { headers: { 'Accept': 'application/json' } })
                    .then(function(response) {
                        if (!response.ok) {
                            throw new Error('Failed to load page data: ' + response.status);
                        }
                        return response.json();
                    })
                    .then(function(data) {
                        var vars = (data && data.vars) ? data.vars : {};
                        window.N3_VARS = vars;
                        document.querySelectorAll('[data-n3-text-template]').forEach(function(el) {
                            var tpl = el.getAttribute('data-n3-text-template') || '';
                            el.textContent = tpl.replace(/\\{([a-zA-Z_][a-zA-Z0-9_]*)\\}/g, function(match, name) {
                                return Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : '';
                            });
                        });
                        if (window.N3Realtime && window.N3Realtime.applySnapshot) {
                            window.N3Realtime.applySnapshot("$slug", data || {}, { source: 'bootstrap' });
                        }
                    })
                    .catch(function(err) {
                        console.error('Namel3ss frontend bootstrap failed:', err);
                    });
            })();
            </script>
            """
        )
    )
    html_parts.append(
        bootstrap_template.substitute(
            api_url=f"/api/pages/{backend_slug}",
            slug=backend_slug,
        ).strip()
    )

    runtime_lines: List[str] = []
    if widget_defs:
        runtime_lines.append(
            f"if (window.N3Widgets && window.N3Widgets.bootstrap) {{ window.N3Widgets.bootstrap({json.dumps(widget_defs)}); }}"
        )
    if enable_realtime and (page.reactive or page.refresh_policy):
        fallback_interval = None
        if page.refresh_policy and getattr(page.refresh_policy, 'interval_seconds', None):
            fallback_interval = page.refresh_policy.interval_seconds
        interval_literal = "null" if fallback_interval is None else str(fallback_interval)
        runtime_lines.append(
            f"if (window.N3Realtime && window.N3Realtime.connectPage) {{ window.N3Realtime.connectPage('{backend_slug}', {{ fallbackInterval: {interval_literal} }}); }}"
        )
    runtime_lines.extend(inline_scripts)
    if runtime_lines:
        html_parts.append("<script>")
        html_parts.append("document.addEventListener('DOMContentLoaded', function() {")
        for snippet in runtime_lines:
            for line in snippet.splitlines():
                html_parts.append(f"  {line}")
        html_parts.append("});")
        html_parts.append("</script>")

    html_parts.append("</body>")
    html_parts.append("</html>")
    return '\n'.join(html_parts)
