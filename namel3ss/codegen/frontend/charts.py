"""Chart configuration helpers."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

from namel3ss.ast import ShowChart

from .theme import theme_palette

DEFAULT_CHART_COLORS: Dict[str, Tuple[str, str]] = {
    "bar": ("rgba(99, 102, 241, 0.6)", "rgba(99, 102, 241, 1)"),
    "line": ("rgba(16, 185, 129, 0.4)", "rgba(16, 185, 129, 1)"),
    "pie": ("rgba(249, 115, 22, 0.6)", "rgba(249, 115, 22, 1)"),
    "doughnut": ("rgba(249, 115, 22, 0.6)", "rgba(249, 115, 22, 1)"),
    "radar": ("rgba(236, 72, 153, 0.4)", "rgba(236, 72, 153, 1)"),
    "default": ("rgba(148, 163, 184, 0.6)", "rgba(148, 163, 184, 1)"),
}

THEME_SERIES_PALETTES: Dict[str, Tuple[str, ...]] = {
    "light": (
        "#6366F1",
        "#10B981",
        "#F59E0B",
        "#F97316",
        "#EC4899",
        "#3B82F6",
        "#14B8A6",
        "#F87171",
    ),
    "dark": (
        "#A5B4FC",
        "#34D399",
        "#FBBF24",
        "#FB7185",
        "#F472B6",
        "#93C5FD",
        "#5EEAD4",
        "#FED7AA",
    ),
}


def _hex_to_rgb(value: str) -> Optional[Tuple[int, int, int]]:
    hex_value = value.strip().lstrip("#")
    if len(hex_value) not in {3, 6}:
        return None
    if len(hex_value) == 3:
        hex_value = "".join(ch * 2 for ch in hex_value)
    try:
        r = int(hex_value[0:2], 16)
        g = int(hex_value[2:4], 16)
        b = int(hex_value[4:6], 16)
    except ValueError:
        return None
    return (r, g, b)


def _rgb_to_rgba(rgb: Tuple[int, int, int], alpha: float) -> str:
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha:.3f})"


def _adjust_rgb(rgb: Tuple[int, int, int], lighten_factor: float) -> Tuple[int, int, int]:
    factor = max(-1.0, min(1.0, lighten_factor))
    if factor >= 0:
        return tuple(int(channel + (255 - channel) * factor) for channel in rgb)
    return tuple(int(channel * (1 + factor)) for channel in rgb)


def _build_dynamic_palette(count: int, chart_type: str, theme: Optional[str]) -> List[Tuple[str, str]]:
    if count <= 0:
        return []
    mode = (theme or "").lower()
    palette_key = "dark" if mode in {"dark", "night", "dim"} else "light"
    base_palette = THEME_SERIES_PALETTES.get(palette_key, THEME_SERIES_PALETTES["light"])
    results: List[Tuple[str, str]] = []
    for idx in range(count):
        base_color = base_palette[idx % len(base_palette)]
        rgb = _hex_to_rgb(base_color)
        if rgb is None:
            rgb = (99, 102, 241)
        cycle = idx // len(base_palette)
        if cycle:
            adjustment = min(0.35, 0.12 * cycle)
            rgb = _adjust_rgb(rgb, adjustment)
        border_color = _rgb_to_rgba(rgb, 0.95)
        if chart_type in {"line", "radar"}:
            fill_alpha = 0.25 + 0.08 * min(cycle, 2)
        else:
            fill_alpha = 0.65
        background_color = _rgb_to_rgba(rgb, fill_alpha)
        results.append((background_color, border_color))
    return results


def _deep_merge(target: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def build_chart_config(
    chart_stmt: ShowChart,
    dataset_payload: Dict[str, Any],
    theme: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a resilient Chart.js configuration from AST metadata."""

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
            label = entry_copy.get("label") or heading_label
            entry_copy.setdefault("label", label)
            entry_copy.setdefault("borderWidth", 1)
            if chart_type in {"line", "radar"}:
                entry_copy.setdefault("fill", False)
                entry_copy.setdefault("tension", 0.35)
            normalised.append(entry_copy)

    if not labels or not normalised:
        labels = list(labels) if labels else []
        normalised = []

    palette = theme_palette(theme)

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
        candidates = [
            axis_key,
            axis_key.lower(),
            axis_key.upper(),
            f"{axis_key}_axis",
            f"{axis_key}-axis",
            f"{axis_key} axis",
        ]
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

    auto_palette = _build_dynamic_palette(len(normalised), chart_type, theme)
    if auto_palette:
        for idx, dataset in enumerate(normalised):
            bg_auto, border_auto = auto_palette[idx]
            if chart_stmt.color:
                dataset.setdefault("backgroundColor", chart_stmt.color)
                dataset.setdefault("borderColor", chart_stmt.color)
            else:
                dataset.setdefault("backgroundColor", bg_auto)
                dataset.setdefault("borderColor", border_auto)
            dataset.setdefault("hoverBackgroundColor", dataset.get("backgroundColor"))
            dataset.setdefault("hoverBorderColor", dataset.get("borderColor"))

    dataset_style = style_dict.get("dataset")
    if isinstance(dataset_style, dict):
        for dataset in normalised:
            for key, value in dataset_style.items():
                dataset.setdefault(key, value)

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
    elif isinstance(colors_style.get("palette"), list):
        custom_palette = colors_style["palette"]
        for idx, dataset in enumerate(normalised):
            color_value = custom_palette[idx % len(custom_palette)]
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

    plugins_override = style_dict.get("plugins")
    if isinstance(plugins_override, dict):
        options.setdefault("plugins", {})
        _deep_merge(options["plugins"], plugins_override)

    elements_override = style_dict.get("elements")
    if isinstance(elements_override, dict):
        options.setdefault("elements", {})
        _deep_merge(options["elements"], elements_override)

    layout_override = style_dict.get("layout")
    if isinstance(layout_override, dict):
        options.setdefault("layout", {})
        _deep_merge(options["layout"], layout_override)

    scales_override = style_dict.get("scales")
    if isinstance(scales_override, dict):
        options.setdefault("scales", {})
        _deep_merge(options["scales"], scales_override)

    options_override = style_dict.get("options")
    if isinstance(options_override, dict):
        _deep_merge(options, options_override)

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

    background_override = style_dict.get("background") or style_dict.get("backgroundColor")
    if isinstance(background_override, str):
        options.setdefault("backgroundColor", background_override)

    config = {
        "type": chart_type,
        "data": {"labels": labels, "datasets": normalised},
        "options": options,
    }
    if isinstance(background_override, str):
        config["backgroundColor"] = background_override
    return config
