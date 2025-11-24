"""
Main orchestration module for React + Vite + TypeScript site generation.

This module coordinates the generation of a complete Vite frontend project
from a namel3ss App structure, calling all specialized modules to create
configuration files, components, pages, and supporting libraries.
"""

from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from namel3ss.ast import App
from namel3ss.codegen.frontend.preview import PreviewDataResolver

if TYPE_CHECKING:
    from namel3ss.ir import BackendIR

from .config import write_package_json, write_tsconfig, write_tsconfig_node, write_vite_config
from .scaffolding import write_index_html, write_main_tsx, write_index_css
from .components import (
    write_navigation,
    write_toast_component,
    write_layout_component,
    write_chart_widget,
    write_table_widget,
    write_form_widget,
    write_text_widget,
)
from .hooks import write_realtime_hook
from .client import write_client_lib
from .dataset_client import write_dataset_client_lib
from .pages import (
    ReactPage,
    build_placeholder_page,
    build_page_definition,
    write_app_tsx,
    write_page_component,
)


def generate_react_vite_site(
    app: App,
    output_dir: str,
    *,
    enable_realtime: bool = False,
    backend_ir: Optional["BackendIR"] = None,
) -> None:
    """
    Generate a complete Vite + React + TypeScript project for the given app.
    
    This is the main entry point that orchestrates the entire frontend generation
    process. It creates:
    - NPM configuration and TypeScript setup
    - HTML entry point and React main.tsx
    - Global CSS with theme integration
    - Navigation configuration
    - Reusable React components (Layout, Toast, widgets)
    - Custom hooks for realtime functionality
    - API client library with optimistic updates
    - App.tsx with React Router configuration
    - Individual page components
    
    Args:
        app: Namel3ss application to generate frontend for
        output_dir: Directory path for generated Vite project
        enable_realtime: Whether to enable WebSocket realtime connections
        backend_ir: Optional BackendIR for dataset client generation
    
    Generated structure:
        output_dir/
        ├── package.json
        ├── tsconfig.json
        ├── tsconfig.node.json
        ├── vite.config.ts
        ├── index.html
        └── src/
            ├── main.tsx
            ├── index.css
            ├── App.tsx
            ├── components/
            │   ├── Layout.tsx
            │   ├── Toast.tsx
            │   ├── ChartWidget.tsx
            │   ├── TableWidget.tsx
            │   ├── FormWidget.tsx
            │   └── TextBlock.tsx
            ├── lib/
            │   ├── navigation.ts
            │   ├── realtime.ts
            │   └── n3Client.ts
            └── pages/
                ├── index.tsx
                └── [other pages].tsx
    """
    out = Path(output_dir)
    src_dir = out / "src"
    components_dir = src_dir / "components"
    pages_dir = src_dir / "pages"
    lib_dir = src_dir / "lib"

    # Create directory structure
    for path in (out, src_dir, components_dir, pages_dir, lib_dir):
        path.mkdir(parents=True, exist_ok=True)

    preview_provider = PreviewDataResolver(app)

    # Build page definitions
    page_builds: List[ReactPage] = []
    nav_links: List[Dict[str, str]] = []

    if not app.pages:
        placeholder = build_placeholder_page()
        page_builds.append(placeholder)
        nav_links.append({"label": "Home", "path": "/"})
    else:
        for index, page in enumerate(app.pages):
            build = build_page_definition(
                app,
                page,
                index,
                preview_provider,
                enable_realtime=enable_realtime,
            )
            page_builds.append(build)
            nav_links.append({"label": page.name, "path": build.definition["route"]})

    # Generate configuration files
    write_package_json(out)
    write_tsconfig(out)
    write_tsconfig_node(out)
    write_vite_config(out)
    
    # Generate scaffolding
    write_index_html(out, app.name)
    write_main_tsx(src_dir)
    write_index_css(src_dir, app)
    
    # Generate navigation and components
    write_navigation(lib_dir, nav_links)
    write_toast_component(components_dir)
    write_layout_component(components_dir)
    write_chart_widget(components_dir)
    write_table_widget(components_dir)
    write_form_widget(components_dir)
    write_text_widget(components_dir)
    
    # Generate hooks and client library
    write_realtime_hook(lib_dir)
    write_client_lib(lib_dir)
    
    # Generate dataset client if backend_ir provided
    if backend_ir:
        write_dataset_client_lib(lib_dir, backend_ir)
    
    # Generate App.tsx with routing
    write_app_tsx(src_dir, page_builds)

    # Generate individual page components
    for build in page_builds:
        write_page_component(pages_dir, build)
