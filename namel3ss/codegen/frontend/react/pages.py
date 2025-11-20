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
from namel3ss.ast.pages import ShowChart, ShowForm, ShowTable, ShowText, ToastOperation
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
        "description": "Add pages to your .n3 program to populate the React UI.",
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

    return widgets, preview_map


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
        import { NAV_LINKS } from "../lib/navigation";
        import { PageDefinition, resolveWidgetData, usePageData } from "../lib/n3Client";
        import { useRealtimePage } from "../lib/realtime";

        const PAGE_DEFINITION: PageDefinition = __DEFINITION__ as const;

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
                  {PAGE_DEFINITION.widgets.map((widget) => {
                    const widgetData = resolveWidgetData(widget.id, data) ?? PAGE_DEFINITION.preview[widget.id];
                    if (widget.type === "text") {
                      return <TextBlock key={widget.id} widget={widget} />;
                    }
                    if (widget.type === "chart") {
                      return <ChartWidget key={widget.id} widget={widget} data={widgetData} />;
                    }
                    if (widget.type === "table") {
                      return <TableWidget key={widget.id} widget={widget} data={widgetData} />;
                    }
                    if (widget.type === "form") {
                      return <FormWidget key={widget.id} widget={widget} pageSlug={PAGE_DEFINITION.slug} />;
                    }
                    return null;
                  })}
                </div>
              )}
            </Layout>
          );
        }
        """
    ).strip()

    content = template.replace("__DEFINITION__", definition).replace("__COMPONENT__", build.component_name) + "\n"
    write_file(pages_dir / f"{build.file_name}.tsx", content)
