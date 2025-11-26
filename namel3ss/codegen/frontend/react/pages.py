"""
Page component generation and routing logic.

This module handles building page definitions from namel3ss App/Page structures
and generating React component files with routing configurations.
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

from namel3ss.ast import App, Page
from namel3ss.ast.pages import (
    AccordionLayout,
    GridLayout,
    ShowCard,
    ShowChart,
    ShowForm,
    ShowList,
    ShowTable,
    ShowText,
    SplitLayout,
    StackLayout,
    TabsLayout,
    ToastOperation,
    # Data display components
    ShowAvatarGroup,
    ShowDataChart,
    ShowDataList,
    ShowDataTable,
    ShowStatSummary,
    ShowTimeline,
)
from namel3ss.codegen.frontend.preview import PreviewDataResolver
from namel3ss.codegen.frontend.slugs import slugify_page_name, slugify_route
from .utils import normalize_route, make_component_name, write_file


class ReactPage:
    """
    Represents a React page component to be generated.
    
    Attributes:
        component_name: React component name (e.g., "IndexPage")
        file_name: Output filename without extension (e.g., "index")
        primary_route: Main route path (e.g., "/")
        extra_routes: Additional route aliases
        backend_slug: Backend page identifier
        definition: Page metadata and widget configuration
    """
    def __init__(
        self,
        component_name: str,
        file_name: str,
        primary_route: str,
        extra_routes: List[str],
        backend_slug: str,
        definition: Dict[str, Any],
    ):
        self.component_name = component_name
        self.file_name = file_name
        self.primary_route = primary_route
        self.extra_routes = extra_routes
        self.backend_slug = backend_slug
        self.definition = definition


def build_placeholder_page() -> ReactPage:
    """Generate a default placeholder page when no pages are defined."""
    definition = {
        "slug": "index",
        "route": "/",
        "title": "Welcome",
        "description": "Add pages to your .ai program to populate the React UI.",
        "reactive": False,
        "realtime": False,
        "widgets": [
            {
                "id": "text_1",
                "type": "text",
                "text": "Namel3ss generated this placeholder because no pages were defined.",
                "styles": {"align": "center"},
            }
        ],
        "preview": {},
    }
    return ReactPage(
        component_name="IndexPage",
        file_name="index",
        primary_route="/",
        extra_routes=[],
        backend_slug="index",
        definition=definition,
    )


def build_page_definition(
    app: App,
    page: Page,
    index: int,
    preview_provider: PreviewDataResolver,
    *,
    enable_realtime: bool,
) -> ReactPage:
    """
    Build a ReactPage from namel3ss Page structure.
    
    Args:
        app: Application instance
        page: Page to convert
        index: Page index in app.pages list
        preview_provider: Provider for widget preview data
        enable_realtime: Whether to enable WebSocket connections
    
    Returns:
        ReactPage with component metadata and widget definitions
    """
    backend_slug = slugify_page_name(page.name, index)
    raw_route = page.route or "/"
    route = normalize_route(raw_route)
    slug = slugify_route(raw_route)
    if index > 0 and route == "/":
        inferred = slug if slug != "index" else f"page_{index + 1}"
        route = f"/{inferred}"
        slug = inferred

    file_name = slug or ("page" + str(index + 1))
    if index == 0:
        file_name = "index"

    component_name = make_component_name(file_name, index)
    widgets, preview_map = collect_widgets(page, preview_provider)

    primary_route = "/"
    extra_routes: List[str] = []
    if route != "/":
        if index == 0:
            extra_routes.append(route)
        else:
            primary_route = route

    definition = {
        "slug": backend_slug,
        "route": route,
        "title": page.name,
        "description": page.layout.get("description") if isinstance(page.layout, dict) else None,
        "reactive": bool(page.reactive),
        "realtime": bool(enable_realtime and page.reactive),
        "widgets": widgets,
        "preview": preview_map,
    }

    return ReactPage(
        component_name=component_name,
        file_name=file_name,
        primary_route=primary_route,
        extra_routes=extra_routes,
        backend_slug=backend_slug,
        definition=definition,
    )


def collect_widgets(
    page: Page,
    preview_provider: PreviewDataResolver,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract widget configurations from page statements.
    
    Converts namel3ss ShowText, ShowTable, ShowChart, ShowForm statements
    into React widget configuration objects with preview data.
    
    Args:
        page: Page containing widget statements
        preview_provider: Provider for widget preview data
    
    Returns:
        Tuple of (widget_configs, preview_data_map)
    """
    widgets: List[Dict[str, Any]] = []
    preview_map: Dict[str, Any] = {}
    counters = {
        "text": 0,
        "table": 0,
        "chart": 0,
        "form": 0,
        "card": 0,
        "list": 0,
        "stack": 0,
        "grid": 0,
        "split": 0,
        "tabs": 0,
        "accordion": 0,
        # Data display components
        "data_table": 0,
        "data_list": 0,
        "stat_summary": 0,
        "timeline": 0,
        "avatar_group": 0,
        "data_chart": 0,
    }

    for statement in page.statements:
        if isinstance(statement, ShowText):
            counters["text"] += 1
            widget_id = f"text_{counters['text']}"
            widgets.append(
                {
                    "id": widget_id,
                    "type": "text",
                    "text": statement.text,
                    "styles": statement.styles or {},
                }
            )
        elif isinstance(statement, ShowTable):
            counters["table"] += 1
            widget_id = f"table_{counters['table']}"
            preview = preview_provider.table_preview(statement)
            preview_map[widget_id] = preview
            widgets.append(
                {
                    "id": widget_id,
                    "type": "table",
                    "title": statement.title,
                    "source": {
                        "kind": statement.source_type,
                        "name": statement.source,
                    },
                    "columns": statement.columns or preview.get("columns", []),
                }
            )
        elif isinstance(statement, ShowChart):
            counters["chart"] += 1
            widget_id = f"chart_{counters['chart']}"
            preview = preview_provider.chart_preview(statement)
            preview_map[widget_id] = preview
            widgets.append(
                {
                    "id": widget_id,
                    "type": "chart",
                    "title": statement.heading,
                    "chartType": statement.chart_type,
                    "source": {
                        "kind": statement.source_type,
                        "name": statement.source,
                    },
                    "x": statement.x,
                    "y": statement.y,
                }
            )
        elif isinstance(statement, ShowForm):
            counters["form"] += 1
            widget_id = f"form_{counters['form']}"
            success_message: str | None = None
            for op in statement.on_submit_ops:
                if isinstance(op, ToastOperation):
                    success_message = op.message
                    break
            preview_map[widget_id] = {
                "fields": [
                    {"name": field.name, "type": field.field_type}
                    for field in statement.fields
                ]
            }
            widgets.append(
                {
                    "id": widget_id,
                    "type": "form",
                    "title": statement.title,
                    "fields": [
                        {"name": field.name, "type": field.field_type}
                        for field in statement.fields
                    ],
                    "successMessage": success_message,
                }
            )
        elif isinstance(statement, ShowCard):
            counters["card"] += 1
            widget_id = f"card_{counters['card']}"
            preview = preview_provider.card_preview(statement)
            preview_map[widget_id] = preview
            widget_config = {
                "id": widget_id,
                "type": "card",
                "title": statement.title,
                "source": {
                    "kind": statement.source_type,
                    "name": statement.source,
                },
            }
            if statement.empty_state:
                widget_config["emptyState"] = serialize_empty_state(statement.empty_state)
            if statement.item_config:
                widget_config["itemConfig"] = serialize_item_config(statement.item_config)
            if statement.group_by:
                widget_config["groupBy"] = statement.group_by
            if statement.filter_by:
                widget_config["filterBy"] = statement.filter_by
            if statement.sort_by:
                widget_config["sortBy"] = statement.sort_by
            if statement.layout:
                widget_config["layout"] = statement.layout
            widgets.append(widget_config)
        elif isinstance(statement, ShowList):
            counters["list"] += 1
            widget_id = f"list_{counters['list']}"
            preview = preview_provider.list_preview(statement)
            preview_map[widget_id] = preview
            widget_config = {
                "id": widget_id,
                "type": "list",
                "title": statement.title,
                "source": {
                    "kind": statement.source_type,
                    "name": statement.source,
                },
            }
            if statement.list_type:
                widget_config["listType"] = statement.list_type
            if statement.empty_state:
                widget_config["emptyState"] = serialize_empty_state(statement.empty_state)
            if statement.item_config:
                widget_config["itemConfig"] = serialize_item_config(statement.item_config)
            if statement.enable_search:
                widget_config["enableSearch"] = statement.enable_search
            if statement.filters:
                widget_config["filters"] = statement.filters
            if statement.columns:
                widget_config["columns"] = statement.columns
            if statement.group_by:
                widget_config["groupBy"] = statement.group_by
            widgets.append(widget_config)
        elif isinstance(statement, StackLayout):
            counters["stack"] += 1
            widget_id = f"stack_{counters['stack']}"
            # Recursively collect children widgets
            child_widgets, child_preview = collect_widgets(
                Page(name=page.name, route=page.route, statements=statement.children),
                preview_provider
            )
            preview_map.update(child_preview)
            widget_config = {
                "id": widget_id,
                "type": "stack",
                "direction": statement.direction,
                "gap": statement.gap,
                "align": statement.align,
                "justify": statement.justify,
                "wrap": statement.wrap,
                "children": child_widgets,
            }
            if statement.style:
                widget_config["style"] = statement.style
            if statement.layout:
                widget_config["layout"] = statement.layout
            widgets.append(widget_config)
        elif isinstance(statement, GridLayout):
            counters["grid"] += 1
            widget_id = f"grid_{counters['grid']}"
            child_widgets, child_preview = collect_widgets(
                Page(name=page.name, route=page.route, statements=statement.children),
                preview_provider
            )
            preview_map.update(child_preview)
            widget_config = {
                "id": widget_id,
                "type": "grid",
                "columns": statement.columns,
                "gap": statement.gap,
                "responsive": statement.responsive,
                "children": child_widgets,
            }
            if statement.min_column_width:
                widget_config["minColumnWidth"] = statement.min_column_width
            if statement.style:
                widget_config["style"] = statement.style
            if statement.layout:
                widget_config["layout"] = statement.layout
            widgets.append(widget_config)
        elif isinstance(statement, SplitLayout):
            counters["split"] += 1
            widget_id = f"split_{counters['split']}"
            left_widgets, left_preview = collect_widgets(
                Page(name=page.name, route=page.route, statements=statement.left),
                preview_provider
            )
            right_widgets, right_preview = collect_widgets(
                Page(name=page.name, route=page.route, statements=statement.right),
                preview_provider
            )
            preview_map.update(left_preview)
            preview_map.update(right_preview)
            widget_config = {
                "id": widget_id,
                "type": "split",
                "ratio": statement.ratio,
                "resizable": statement.resizable,
                "orientation": statement.orientation,
                "left": left_widgets,
                "right": right_widgets,
            }
            if statement.style:
                widget_config["style"] = statement.style
            if statement.layout:
                widget_config["layout"] = statement.layout
            widgets.append(widget_config)
        elif isinstance(statement, TabsLayout):
            counters["tabs"] += 1
            widget_id = f"tabs_{counters['tabs']}"
            tabs_config = []
            for tab in statement.tabs:
                tab_widgets, tab_preview = collect_widgets(
                    Page(name=page.name, route=page.route, statements=tab.content),
                    preview_provider
                )
                preview_map.update(tab_preview)
                tab_config = {
                    "id": tab.id,
                    "label": tab.label,
                    "content": tab_widgets,
                }
                if tab.icon:
                    tab_config["icon"] = tab.icon
                if tab.badge:
                    tab_config["badge"] = tab.badge
                tabs_config.append(tab_config)
            widget_config = {
                "id": widget_id,
                "type": "tabs",
                "tabs": tabs_config,
                "persistState": statement.persist_state,
            }
            if statement.default_tab:
                widget_config["defaultTab"] = statement.default_tab
            if statement.style:
                widget_config["style"] = statement.style
            if statement.layout:
                widget_config["layout"] = statement.layout
            widgets.append(widget_config)
        elif isinstance(statement, AccordionLayout):
            counters["accordion"] += 1
            widget_id = f"accordion_{counters['accordion']}"
            items_config = []
            for item in statement.items:
                item_widgets, item_preview = collect_widgets(
                    Page(name=page.name, route=page.route, statements=item.content),
                    preview_provider
                )
                preview_map.update(item_preview)
                item_config = {
                    "id": item.id,
                    "title": item.title,
                    "defaultOpen": item.default_open,
                    "content": item_widgets,
                }
                if item.description:
                    item_config["description"] = item.description
                if item.icon:
                    item_config["icon"] = item.icon
                items_config.append(item_config)
            widget_config = {
                "id": widget_id,
                "type": "accordion",
                "items": items_config,
                "multiple": statement.multiple,
            }
            if statement.style:
                widget_config["style"] = statement.style
            if statement.layout:
                widget_config["layout"] = statement.layout
            widgets.append(widget_config)
        
        # Data display components
        elif isinstance(statement, ShowDataTable):
            counters["data_table"] += 1
            widget_id = f"data_table_{counters['data_table']}"
            preview = preview_provider.table_preview(statement) if hasattr(preview_provider, 'table_preview') else {}
            preview_map[widget_id] = preview
            widget_config = {
                "id": widget_id,
                "type": "data_table",
                "title": statement.title,
                "source": {
                    "kind": statement.source_type,
                    "name": statement.source,
                },
                "columns": [
                    {
                        "id": col.id,
                        "label": col.label,
                        "field": col.field,
                        "width": col.width,
                        "align": col.align,
                        "sortable": col.sortable,
                        "format": col.format,
                        "transform": col.transform,
                        "renderTemplate": col.render_template,
                    }
                    for col in statement.columns
                ] if statement.columns else [],
                "toolbar": {
                    "search": statement.toolbar.search,
                    "filters": statement.toolbar.filters,
                    "bulkActions": statement.toolbar.bulk_actions,
                    "actions": statement.toolbar.actions,
                } if statement.toolbar else None,
                "rowActions": statement.row_actions if statement.row_actions else [],
                "filterBy": statement.filter_by,
                "sortBy": statement.sort_by,
                "defaultSort": statement.default_sort,
                "pageSize": statement.page_size,
                "enablePagination": statement.enable_pagination,
                "emptyState": serialize_empty_state(statement.empty_state) if statement.empty_state else None,
                "style": statement.style,
                "layout": statement.layout,
            }
            widgets.append(widget_config)
        elif isinstance(statement, ShowDataList):
            counters["data_list"] += 1
            widget_id = f"data_list_{counters['data_list']}"
            preview = {}
            preview_map[widget_id] = preview
            widget_config = {
                "id": widget_id,
                "type": "data_list",
                "title": statement.title,
                "source": {
                    "kind": statement.source_type,
                    "name": statement.source,
                },
                "item": {
                    "avatar": statement.item.avatar,
                    "title": statement.item.title,
                    "subtitle": statement.item.subtitle,
                    "metadata": statement.item.metadata,
                    "actions": statement.item.actions,
                    "badge": statement.item.badge,
                    "icon": statement.item.icon,
                    "stateClass": statement.item.state_class,
                } if statement.item else None,
                "variant": statement.variant,
                "dividers": statement.dividers,
                "filterBy": statement.filter_by,
                "enableSearch": statement.enable_search,
                "searchPlaceholder": statement.search_placeholder,
                "pageSize": statement.page_size,
                "enablePagination": statement.enable_pagination,
                "emptyState": serialize_empty_state(statement.empty_state) if statement.empty_state else None,
                "style": statement.style,
                "layout": statement.layout,
            }
            widgets.append(widget_config)
        elif isinstance(statement, ShowStatSummary):
            counters["stat_summary"] += 1
            widget_id = f"stat_summary_{counters['stat_summary']}"
            preview = {}
            preview_map[widget_id] = preview
            widget_config = {
                "id": widget_id,
                "type": "stat_summary",
                "label": statement.label,
                "source": {
                    "kind": statement.source_type,
                    "name": statement.source,
                },
                "value": statement.value,
                "format": statement.format,
                "prefix": statement.prefix,
                "suffix": statement.suffix,
                "delta": statement.delta,
                "trend": statement.trend,
                "comparisonPeriod": statement.comparison_period,
                "sparkline": {
                    "dataSource": statement.sparkline.data_source,
                    "xField": statement.sparkline.x_field,
                    "yField": statement.sparkline.y_field,
                    "color": statement.sparkline.color,
                    "variant": statement.sparkline.variant,
                } if statement.sparkline else None,
                "color": statement.color,
                "icon": statement.icon,
                "style": statement.style,
                "layout": statement.layout,
            }
            widgets.append(widget_config)
        elif isinstance(statement, ShowTimeline):
            counters["timeline"] += 1
            widget_id = f"timeline_{counters['timeline']}"
            preview = {}
            preview_map[widget_id] = preview
            widget_config = {
                "id": widget_id,
                "type": "timeline",
                "title": statement.title,
                "source": {
                    "kind": statement.source_type,
                    "name": statement.source,
                },
                "item": {
                    "timestamp": statement.item.timestamp,
                    "title": statement.item.title,
                    "description": statement.item.description,
                    "icon": statement.item.icon,
                    "status": statement.item.status,
                    "color": statement.item.color,
                    "actions": statement.item.actions,
                } if statement.item else None,
                "variant": statement.variant,
                "showTimestamps": statement.show_timestamps,
                "groupByDate": statement.group_by_date,
                "filterBy": statement.filter_by,
                "sortBy": statement.sort_by,
                "pageSize": statement.page_size,
                "enablePagination": statement.enable_pagination,
                "emptyState": serialize_empty_state(statement.empty_state) if statement.empty_state else None,
                "style": statement.style,
                "layout": statement.layout,
            }
            widgets.append(widget_config)
        elif isinstance(statement, ShowAvatarGroup):
            counters["avatar_group"] += 1
            widget_id = f"avatar_group_{counters['avatar_group']}"
            preview = {}
            preview_map[widget_id] = preview
            widget_config = {
                "id": widget_id,
                "type": "avatar_group",
                "title": statement.title,
                "source": {
                    "kind": statement.source_type,
                    "name": statement.source,
                },
                "item": {
                    "name": statement.item.name,
                    "imageUrl": statement.item.image_url,
                    "initials": statement.item.initials,
                    "color": statement.item.color,
                    "status": statement.item.status,
                    "tooltip": statement.item.tooltip,
                } if statement.item else None,
                "maxVisible": statement.max_visible,
                "size": statement.size,
                "variant": statement.variant,
                "filterBy": statement.filter_by,
                "style": statement.style,
                "layout": statement.layout,
            }
            widgets.append(widget_config)
        elif isinstance(statement, ShowDataChart):
            counters["data_chart"] += 1
            widget_id = f"data_chart_{counters['data_chart']}"
            preview = preview_provider.chart_preview(statement) if hasattr(preview_provider, 'chart_preview') else {}
            preview_map[widget_id] = preview
            widget_config = {
                "id": widget_id,
                "type": "data_chart",
                "title": statement.title,
                "source": {
                    "kind": statement.source_type,
                    "name": statement.source,
                },
                "config": {
                    "variant": statement.config.variant,
                    "xField": statement.config.x_field,
                    "yFields": statement.config.y_fields,
                    "groupBy": statement.config.group_by,
                    "stacked": statement.config.stacked,
                    "smooth": statement.config.smooth,
                    "fill": statement.config.fill,
                    "legend": statement.config.legend,
                    "tooltip": statement.config.tooltip,
                    "xAxis": statement.config.x_axis,
                    "yAxis": statement.config.y_axis,
                    "colors": statement.config.colors,
                    "colorScheme": statement.config.color_scheme,
                } if statement.config else None,
                "filterBy": statement.filter_by,
                "sortBy": statement.sort_by,
                "height": statement.height,
                "emptyState": serialize_empty_state(statement.empty_state) if statement.empty_state else None,
                "style": statement.style,
                "layout": statement.layout,
            }
            widgets.append(widget_config)

    return widgets, preview_map


def serialize_empty_state(config: Any) -> Dict[str, Any]:
    """Convert EmptyStateConfig to JSON-serializable dict."""
    result = {}
    if hasattr(config, 'icon') and config.icon:
        result['icon'] = config.icon
    if hasattr(config, 'icon_size') and config.icon_size:
        result['iconSize'] = config.icon_size
    if hasattr(config, 'title') and config.title:
        result['title'] = config.title
    if hasattr(config, 'message') and config.message:
        result['message'] = config.message
    if hasattr(config, 'action_label') and config.action_label:
        result['actionLabel'] = config.action_label
    if hasattr(config, 'action_link') and config.action_link:
        result['actionLink'] = config.action_link
    return result


def serialize_item_config(config: Any) -> Dict[str, Any]:
    """Convert CardItemConfig to JSON-serializable dict."""
    result = {}
    if hasattr(config, 'type') and config.type:
        result['type'] = config.type
    if hasattr(config, 'style') and config.style:
        result['style'] = config.style
    if hasattr(config, 'state_class') and config.state_class:
        result['stateClass'] = config.state_class
    if hasattr(config, 'header') and config.header:
        result['header'] = serialize_card_header(config.header)
    if hasattr(config, 'sections') and config.sections:
        result['sections'] = [serialize_card_section(s) for s in config.sections]
    if hasattr(config, 'actions') and config.actions:
        result['actions'] = [serialize_conditional_action(a) for a in config.actions]
    if hasattr(config, 'footer') and config.footer:
        result['footer'] = serialize_card_footer(config.footer)
    return result


def serialize_card_header(header: Any) -> Dict[str, Any]:
    """Serialize CardHeader to dict."""
    result = {}
    if hasattr(header, 'title') and header.title:
        result['title'] = header.title
    if hasattr(header, 'subtitle') and header.subtitle:
        result['subtitle'] = header.subtitle
    if hasattr(header, 'avatar') and header.avatar:
        result['avatar'] = header.avatar
    if hasattr(header, 'badges') and header.badges:
        result['badges'] = [serialize_badge(b) for b in header.badges]
    return result


def serialize_badge(badge: Any) -> Dict[str, Any]:
    """Serialize BadgeConfig to dict."""
    result = {}
    if hasattr(badge, 'field') and badge.field:
        result['field'] = badge.field
    if hasattr(badge, 'text') and badge.text:
        result['text'] = badge.text
    if hasattr(badge, 'style') and badge.style:
        result['style'] = badge.style
    if hasattr(badge, 'transform') and badge.transform:
        result['transform'] = badge.transform
    if hasattr(badge, 'icon') and badge.icon:
        result['icon'] = badge.icon
    if hasattr(badge, 'condition') and badge.condition:
        result['condition'] = badge.condition
    return result


def serialize_card_section(section: Any) -> Dict[str, Any]:
    """Serialize CardSection to dict."""
    result = {}
    if hasattr(section, 'type') and section.type:
        result['type'] = section.type
    if hasattr(section, 'condition') and section.condition:
        result['condition'] = section.condition
    if hasattr(section, 'style') and section.style:
        result['style'] = section.style
    if hasattr(section, 'icon') and section.icon:
        result['icon'] = section.icon
    if hasattr(section, 'columns') and section.columns:
        result['columns'] = section.columns
    if hasattr(section, 'items') and section.items:
        result['items'] = [serialize_info_grid_item(i) for i in section.items]
    if hasattr(section, 'content') and section.content:
        result['content'] = section.content
    return result


def serialize_info_grid_item(item: Any) -> Dict[str, Any]:
    """Serialize InfoGridItem to dict."""
    result = {}
    if hasattr(item, 'icon') and item.icon:
        result['icon'] = item.icon
    if hasattr(item, 'label') and item.label:
        result['label'] = item.label
    if hasattr(item, 'field') and item.field:
        result['field'] = item.field
    if hasattr(item, 'values') and item.values:
        result['values'] = [serialize_field_value(v) for v in item.values]
    return result


def serialize_field_value(value: Any) -> Dict[str, Any]:
    """Serialize FieldValueConfig to dict."""
    result = {}
    if hasattr(value, 'field') and value.field:
        result['field'] = value.field
    if hasattr(value, 'text') and value.text:
        result['text'] = value.text
    if hasattr(value, 'format') and value.format:
        result['format'] = value.format
    if hasattr(value, 'style') and value.style:
        result['style'] = value.style
    if hasattr(value, 'transform') and value.transform:
        result['transform'] = value.transform
    return result


def serialize_conditional_action(action: Any) -> Dict[str, Any]:
    """Serialize ConditionalAction to dict."""
    result = {}
    if hasattr(action, 'label') and action.label:
        result['label'] = action.label
    if hasattr(action, 'icon') and action.icon:
        result['icon'] = action.icon
    if hasattr(action, 'style') and action.style:
        result['style'] = action.style
    if hasattr(action, 'action') and action.action:
        result['action'] = action.action
    if hasattr(action, 'link') and action.link:
        result['link'] = action.link
    if hasattr(action, 'params') and action.params:
        result['params'] = action.params
    if hasattr(action, 'condition') and action.condition:
        result['condition'] = action.condition
    return result


def serialize_card_footer(footer: Any) -> Dict[str, Any]:
    """Serialize CardFooter to dict."""
    result = {}
    if hasattr(footer, 'text') and footer.text:
        result['text'] = footer.text
    if hasattr(footer, 'condition') and footer.condition:
        result['condition'] = footer.condition
    if hasattr(footer, 'style') and footer.style:
        result['style'] = footer.style
    if hasattr(footer, 'left') and footer.left:
        result['left'] = footer.left
    if hasattr(footer, 'right') and footer.right:
        result['right'] = footer.right
    return result


def write_app_tsx(src_dir: Path, page_builds: List[ReactPage]) -> None:
    """
    Generate App.tsx with React Router configuration.
    
    Creates the main App component with BrowserRouter, ToastProvider,
    and Routes for all page components.
    
    Args:
        src_dir: Source directory for generated files
        page_builds: List of ReactPage configurations
    """
    imports = [f"import {build.component_name} from \"./pages/{build.file_name}\";" for build in page_builds]
    routes: List[str] = []
    for build in page_builds:
        routes.append(f"          <Route path=\"{build.primary_route}\" element={{<{build.component_name} />}} />")
        for extra in build.extra_routes:
            routes.append(f"          <Route path=\"{extra}\" element={{<{build.component_name} />}} />")
    routes.append("          <Route path=\"*\" element={<Navigate to=\"/\" replace />} />")

    template = textwrap.dedent(
        """
        import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
        import { ToastProvider } from "./components/Toast";
        __IMPORTS__

        export default function App() {
          return (
            <ToastProvider>
              <BrowserRouter>
                <Routes>
        __ROUTES__
                </Routes>
              </BrowserRouter>
            </ToastProvider>
          );
        }
        """
    ).strip()

    content = template.replace("__IMPORTS__", "\n".join(imports)).replace("__ROUTES__", "\n".join(routes)) + "\n"
    write_file(src_dir / "App.tsx", content)


def write_page_component(pages_dir: Path, build: ReactPage) -> None:
    """
    Generate individual page component file.
    
    Creates a React component that:
    - Fetches page data via usePageData hook
    - Establishes realtime connection if enabled
    - Renders widgets based on page definition
    - Handles loading and error states
    
    Args:
        pages_dir: Pages directory for component files
        build: ReactPage configuration
    """
    definition = json.dumps(build.definition, indent=2)
    template = textwrap.dedent(
        """
        import Layout from "../components/Layout";
        import ChartWidget from "../components/ChartWidget";
        import TableWidget from "../components/TableWidget";
        import FormWidget from "../components/FormWidget";
        import TextBlock from "../components/TextBlock";
        import CardWidget from "../components/CardWidget";
        import ListWidget from "../components/ListWidget";
        import { StackLayout, GridLayout, SplitLayout, TabsLayout, AccordionLayout } from "../components/LayoutComponents";
        import { DataTableWidget } from "../components/DataTableWidget";
        import { DataListWidget } from "../components/DataListWidget";
        import { StatSummaryWidget } from "../components/StatSummaryWidget";
        import { TimelineWidget } from "../components/TimelineWidget";
        import { AvatarGroupWidget } from "../components/AvatarGroupWidget";
        import { DataChartWidget } from "../components/DataChartWidget";
        import { NAV_LINKS } from "../lib/navigation";
        import { PageDefinition, resolveWidgetData, usePageData } from "../lib/n3Client";
        import { useRealtimePage } from "../lib/realtime";

        const PAGE_DEFINITION: PageDefinition = __DEFINITION__ as const;

        function renderWidget(widget: any, data: any): React.ReactNode {
          const widgetData = resolveWidgetData(widget.id, data) ?? PAGE_DEFINITION.preview[widget.id];
          
          switch (widget.type) {
            case "text":
              return <TextBlock key={widget.id} widget={widget} />;
            case "chart":
              return <ChartWidget key={widget.id} widget={widget} data={widgetData} />;
            case "table":
              return <TableWidget key={widget.id} widget={widget} data={widgetData} />;
            case "form":
              return <FormWidget key={widget.id} widget={widget} pageSlug={PAGE_DEFINITION.slug} />;
            case "card":
              return <CardWidget key={widget.id} widget={widget} data={widgetData} />;
            case "list":
              return <ListWidget key={widget.id} widget={widget} data={widgetData} />;
            
            // Data display components
            case "data_table":
              return <DataTableWidget key={widget.id} widget={widget} data={widgetData} />;
            case "data_list":
              return <DataListWidget key={widget.id} widget={widget} data={widgetData} />;
            case "stat_summary":
              return <StatSummaryWidget key={widget.id} widget={widget} data={widgetData} />;
            case "timeline":
              return <TimelineWidget key={widget.id} widget={widget} data={widgetData} />;
            case "avatar_group":
              return <AvatarGroupWidget key={widget.id} widget={widget} data={widgetData} />;
            case "data_chart":
              return <DataChartWidget key={widget.id} widget={widget} data={widgetData} />;
            
            case "stack":
              return (
                <StackLayout
                  key={widget.id}
                  direction={widget.direction}
                  gap={widget.gap}
                  align={widget.align}
                  justify={widget.justify}
                  wrap={widget.wrap}
                  style={widget.style}
                >
                  {widget.children?.map((child: any) => renderWidget(child, data)) || []}
                </StackLayout>
              );
            
            case "grid":
              return (
                <GridLayout
                  key={widget.id}
                  columns={widget.columns}
                  minColumnWidth={widget.minColumnWidth}
                  gap={widget.gap}
                  responsive={widget.responsive}
                  style={widget.style}
                >
                  {widget.children?.map((child: any) => renderWidget(child, data)) || []}
                </GridLayout>
              );
            
            case "split":
              return (
                <SplitLayout
                  key={widget.id}
                  left={widget.left?.map((child: any) => renderWidget(child, data)) || []}
                  right={widget.right?.map((child: any) => renderWidget(child, data)) || []}
                  ratio={widget.ratio}
                  resizable={widget.resizable}
                  orientation={widget.orientation}
                  style={widget.style}
                />
              );
            
            case "tabs":
              return (
                <TabsLayout
                  key={widget.id}
                  tabs={widget.tabs?.map((tab: any) => ({
                    id: tab.id,
                    label: tab.label,
                    icon: tab.icon,
                    badge: tab.badge,
                    content: tab.content?.map((child: any) => renderWidget(child, data)) || [],
                  })) || []}
                  defaultTab={widget.defaultTab}
                  persistState={widget.persistState}
                  style={widget.style}
                />
              );
            
            case "accordion":
              return (
                <AccordionLayout
                  key={widget.id}
                  items={widget.items?.map((item: any) => ({
                    id: item.id,
                    title: item.title,
                    description: item.description,
                    icon: item.icon,
                    defaultOpen: item.defaultOpen,
                    content: item.content?.map((child: any) => renderWidget(child, data)) || [],
                  })) || []}
                  multiple={widget.multiple}
                  style={widget.style}
                />
              );
            
            default:
              return null;
          }
        }

        export default function __COMPONENT__() {
          const { data, loading, error } = usePageData(PAGE_DEFINITION);
          useRealtimePage(PAGE_DEFINITION);

          return (
            <Layout title={PAGE_DEFINITION.title} description={PAGE_DEFINITION.description} navLinks={NAV_LINKS}>
              {loading ? (
                <p>Loading page data...</p>
              ) : error ? (
                <p role="alert">Failed to load page: {error}</p>
              ) : (
                <div style={{ display: "grid", gap: "1.25rem" }}>
                  {PAGE_DEFINITION.widgets.map((widget) => renderWidget(widget, data))}
                </div>
              )}
            </Layout>
          );
        }
        """
    ).strip()

    content = template.replace("__DEFINITION__", definition).replace("__COMPONENT__", build.component_name) + "\n"
    write_file(pages_dir / f"{build.file_name}.tsx", content)
