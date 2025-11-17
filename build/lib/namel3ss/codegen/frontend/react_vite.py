"""Generate a Vite + React + TypeScript frontend."""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from namel3ss.ast import App, Page
from namel3ss.ast.pages import ShowChart, ShowForm, ShowTable, ShowText, ToastOperation

from .assets import generate_styles
from .preview import PreviewDataResolver
from .slugs import slugify_page_name, slugify_route


@dataclass
class ReactPage:
    """Metadata required to render a React page component."""

    component_name: str
    file_name: str
    primary_route: str
    extra_routes: List[str]
    backend_slug: str
    definition: Dict[str, Any]


def generate_react_vite_site(app: App, output_dir: str, *, enable_realtime: bool = False) -> None:
    """Generate a Vite + React + TypeScript project for ``app``."""

    out = Path(output_dir)
    src_dir = out / "src"
    components_dir = src_dir / "components"
    pages_dir = src_dir / "pages"
    lib_dir = src_dir / "lib"

    for path in (out, src_dir, components_dir, pages_dir, lib_dir):
        path.mkdir(parents=True, exist_ok=True)

    preview_provider = PreviewDataResolver(app)

    page_builds: List[ReactPage] = []
    nav_links: List[Dict[str, str]] = []

    if not app.pages:
        placeholder = _build_placeholder_page()
        page_builds.append(placeholder)
        nav_links.append({"label": "Home", "path": "/"})
    else:
        for index, page in enumerate(app.pages):
            build = _build_page_definition(
                app,
                page,
                index,
                preview_provider,
                enable_realtime=enable_realtime,
            )
            page_builds.append(build)
            nav_links.append({"label": page.name, "path": build.definition["route"]})

    _write_package_json(out)
    _write_tsconfig(out)
    _write_tsconfig_node(out)
    _write_vite_config(out)
    _write_index_html(out, app.name)
    _write_main_tsx(src_dir)
    _write_index_css(src_dir, app)
    _write_navigation(lib_dir, nav_links)
    _write_toast_component(components_dir)
    _write_layout_component(components_dir)
    _write_chart_widget(components_dir)
    _write_table_widget(components_dir)
    _write_form_widget(components_dir)
    _write_text_widget(components_dir)
    _write_realtime_hook(lib_dir)
    _write_client_lib(lib_dir)
    _write_app_tsx(src_dir, page_builds)

    for build in page_builds:
        _write_page_component(pages_dir, build)


def _build_placeholder_page() -> ReactPage:
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


def _build_page_definition(
    app: App,
    page: Page,
    index: int,
    preview_provider: PreviewDataResolver,
    *,
    enable_realtime: bool,
) -> ReactPage:
    backend_slug = slugify_page_name(page.name, index)
    raw_route = page.route or "/"
    route = _normalize_route(raw_route)
    slug = slugify_route(raw_route)
    if index > 0 and route == "/":
        inferred = slug if slug != "index" else f"page_{index + 1}"
        route = f"/{inferred}"
        slug = inferred

    file_name = slug or ("page" + str(index + 1))
    if index == 0:
        file_name = "index"

    component_name = _make_component_name(file_name, index)
    widgets, preview_map = _collect_widgets(page, preview_provider)

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


def _collect_widgets(
    page: Page,
    preview_provider: PreviewDataResolver,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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


def _normalize_route(route: str | None) -> str:
    value = (route or "").strip()
    if not value:
        return "/"
    if not value.startswith("/"):
        value = "/" + value
    return value


def _make_component_name(base: str, index: int) -> str:
    cleaned = re.split(r"[^A-Za-z0-9]+", base)
    parts = [part for part in cleaned if part]
    if not parts:
        parts = ["Index"]
    name = "".join(part.capitalize() for part in parts)
    if not name or name.lower() == "index":
        name = "Index"
    suffix = "Page"
    if not name.endswith(suffix):
        name = f"{name}{suffix}"
    if index == 0:
        return "IndexPage"
    return name


def _write_package_json(out: Path) -> None:
    package = {
        "name": "namel3ss-react-frontend",
        "private": True,
        "version": "0.1.0",
        "type": "module",
        "scripts": {
            "dev": "vite",
            "build": "tsc && vite build",
            "preview": "vite preview",
        },
        "dependencies": {
            "react": "^18.3.1",
            "react-dom": "^18.3.1",
            "react-router-dom": "^6.28.0",
        },
        "devDependencies": {
            "@types/node": "^20.11.30",
            "@types/react": "^18.2.73",
            "@types/react-dom": "^18.2.24",
            "@vitejs/plugin-react": "^4.2.1",
            "typescript": "^5.4.5",
            "vite": "^5.3.1",
        },
    }
    _write_file(out / "package.json", json.dumps(package, indent=2) + "\n")


def _write_tsconfig(out: Path) -> None:
    tsconfig = {
        "compilerOptions": {
            "target": "ESNext",
            "useDefineForClassFields": True,
            "module": "ESNext",
            "moduleResolution": "Node",
            "strict": True,
            "jsx": "react-jsx",
            "resolveJsonModule": True,
            "isolatedModules": True,
            "esModuleInterop": True,
            "skipLibCheck": True,
        },
        "include": ["src"],
        "references": [{"path": "./tsconfig.node.json"}],
    }
    _write_file(out / "tsconfig.json", json.dumps(tsconfig, indent=2) + "\n")


def _write_tsconfig_node(out: Path) -> None:
    tsconfig_node = {
        "compilerOptions": {
            "composite": True,
            "module": "ESNext",
            "moduleResolution": "Node",
            "allowSyntheticDefaultImports": True,
        },
        "include": ["vite.config.ts"],
    }
    _write_file(out / "tsconfig.node.json", json.dumps(tsconfig_node, indent=2) + "\n")


def _write_vite_config(out: Path) -> None:
    content = textwrap.dedent(
        """
        import { defineConfig } from "vite";
        import react from "@vitejs/plugin-react";

        export default defineConfig({
          plugins: [react()],
          server: {
            proxy: {
              "/api": {
                target: "http://localhost:8000",
                changeOrigin: true,
              },
              "/ws": {
                target: "http://localhost:8000",
                ws: true,
              },
            },
          },
        });
        """
    ).strip() + "\n"
    _write_file(out / "vite.config.ts", content)


def _write_index_html(out: Path, title: str) -> None:
    content = textwrap.dedent(
        f"""
        <!doctype html>
        <html lang=\"en\">
          <head>
            <meta charset=\"UTF-8\" />
            <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
            <title>{title}</title>
          </head>
          <body>
            <div id=\"root\"></div>
            <script type=\"module\" src=\"/src/main.tsx\"></script>
          </body>
        </html>
        """
    ).strip() + "\n"
    _write_file(out / "index.html", content)


def _write_main_tsx(src_dir: Path) -> None:
    content = textwrap.dedent(
        """
        import React from "react";
        import ReactDOM from "react-dom/client";
        import App from "./App";
        import "./index.css";

        ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
          <React.StrictMode>
            <App />
          </React.StrictMode>
        );
        """
    ).strip() + "\n"
    _write_file(src_dir / "main.tsx", content)


def _write_index_css(src_dir: Path, app: App) -> None:
    styles = generate_styles(app)
    base = textwrap.dedent(
        """
        body {
          margin: 0;
          background-color: var(--background, #0f172a);
        }
        .n3-app {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          background-color: var(--background, #ffffff);
          color: var(--text, #0f172a);
        }
        .n3-main {
          flex: 1 1 auto;
          padding: 2rem clamp(1rem, 4vw, 4rem);
          max-width: 1200px;
          width: 100%;
          margin: 0 auto;
        }
        .n3-nav {
          display: flex;
          gap: 1rem;
          flex-wrap: wrap;
        }
        .n3-nav a {
          text-decoration: none;
          color: var(--text, #0f172a);
          font-weight: 600;
        }
        .n3-nav a.active {
          color: var(--primary, #2563eb);
        }
        .n3-widget {
          margin-bottom: 1.75rem;
          padding: 1.25rem;
          border-radius: 1rem;
          background-color: rgba(255, 255, 255, 0.95);
          box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
        }
        .n3-widget h3 {
          margin-top: 0;
        }
        .n3-toast {
          position: fixed;
          bottom: 1.5rem;
          right: 1.5rem;
          background: rgba(15, 23, 42, 0.92);
          color: #fff;
          padding: 0.75rem 1rem;
          border-radius: 0.75rem;
          box-shadow: 0 12px 32px rgba(15, 23, 42, 0.3);
        }
        table.n3-table {
          width: 100%;
          border-collapse: collapse;
        }
        table.n3-table th,
        table.n3-table td {
          border-bottom: 1px solid rgba(15, 23, 42, 0.08);
          padding: 0.65rem 0.75rem;
          text-align: left;
        }
        """
    ).strip()
    content = styles + "\n\n" + base + "\n"
    _write_file(src_dir / "index.css", content)


def _write_navigation(lib_dir: Path, nav_links: List[Dict[str, str]]) -> None:
    rendered = json.dumps(nav_links, indent=2)
    content = textwrap.dedent(
        f"""
        export interface NavLink {{
          label: string;
          path: string;
        }}

        export const NAV_LINKS: NavLink[] = {rendered} as const;
        """
    ).strip() + "\n"
    _write_file(lib_dir / "navigation.ts", content)


def _write_toast_component(components_dir: Path) -> None:
    content = textwrap.dedent(
        """
        import { createContext, PropsWithChildren, useCallback, useContext, useMemo, useState } from "react";

        interface ToastContextValue {
          show: (message: string, timeoutMs?: number) => void;
        }

        const ToastContext = createContext<ToastContextValue>({
          show: () => undefined,
        });

        export function ToastProvider({ children }: PropsWithChildren) {
          const [message, setMessage] = useState<string | null>(null);
          const [timer, setTimer] = useState<number | undefined>(undefined);

          const show = useCallback((nextMessage: string, timeoutMs = 2800) => {
            setMessage(nextMessage);
            if (timer) {
              window.clearTimeout(timer);
            }
            const id = window.setTimeout(() => setMessage(null), timeoutMs);
            setTimer(id);
          }, [timer]);

          const value = useMemo<ToastContextValue>(() => ({ show }), [show]);

          return (
            <ToastContext.Provider value={value}>
              {children}
              {message ? <div className="n3-toast" role="status">{message}</div> : null}
            </ToastContext.Provider>
          );
        }

        export function useToast() {
          return useContext(ToastContext);
        }
        """
    ).strip() + "\n"
    _write_file(components_dir / "Toast.tsx", content)


def _write_layout_component(components_dir: Path) -> None:
    content = textwrap.dedent(
        """
        import { NavLink } from "../lib/navigation";
        import { Link, useLocation } from "react-router-dom";
        import type { PropsWithChildren } from "react";

        interface LayoutProps {
          title: string;
          description?: string | null;
          navLinks: readonly NavLink[];
        }

        export default function Layout({ title, description, navLinks, children }: PropsWithChildren<LayoutProps>) {
          const location = useLocation();

          return (
            <div className="n3-app">
              <header style={{ padding: "1.25rem clamp(1rem, 4vw, 4rem)" }}>
                <h1 style={{ marginBottom: "0.25rem" }}>{title}</h1>
                {description ? <p style={{ marginTop: 0, color: "var(--text-muted, #475569)" }}>{description}</p> : null}
                <nav className="n3-nav">
                  {navLinks.map((link) => (
                    <Link key={link.path} to={link.path} className={location.pathname === link.path ? "active" : undefined}>
                      {link.label}
                    </Link>
                  ))}
                </nav>
              </header>
              <main className="n3-main">{children}</main>
            </div>
          );
        }
        """
    ).strip() + "\n"
    _write_file(components_dir / "Layout.tsx", content)


def _write_chart_widget(components_dir: Path) -> None:
    content = textwrap.dedent(
        """
        import type { ChartWidgetConfig } from "../lib/n3Client";
        import { ensureArray } from "../lib/n3Client";

        interface ChartWidgetProps {
          widget: ChartWidgetConfig;
          data: unknown;
        }

        export default function ChartWidget({ widget, data }: ChartWidgetProps) {
          const labels = Array.isArray((data as any)?.labels) ? (data as any).labels as string[] : [];
          const datasets = ensureArray<{ label?: string; data?: unknown[] }>((data as any)?.datasets);

          return (
            <section className="n3-widget">
              <h3>{widget.title}</h3>
              {labels.length && datasets.length ? (
                <div>
                  {datasets.map((dataset, index) => (
                    <div key={dataset.label ?? index} style={{ marginBottom: "0.75rem" }}>
                      <strong>{dataset.label ?? `Series ${index + 1}`}</strong>
                      <ul style={{ listStyle: "none", paddingLeft: 0 }}>
                        {labels.map((label, idx) => (
                          <li key={label + idx}>
                            <span style={{ fontWeight: 500 }}>{label}:</span> {Array.isArray(dataset.data) ? dataset.data[idx] : "n/a"}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              ) : (
                <pre style={{ marginTop: "0.75rem" }}>{JSON.stringify(data ?? widget, null, 2)}</pre>
              )}
            </section>
          );
        }
        """
    ).strip() + "\n"
    _write_file(components_dir / "ChartWidget.tsx", content)


def _write_table_widget(components_dir: Path) -> None:
    content = textwrap.dedent(
        """
        import type { TableWidgetConfig } from "../lib/n3Client";

        interface TableWidgetProps {
          widget: TableWidgetConfig;
          data: unknown;
        }

        export default function TableWidget({ widget, data }: TableWidgetProps) {
          const rows = Array.isArray((data as any)?.rows) ? (data as any).rows as Record<string, unknown>[] : [];
          const columns = widget.columns && widget.columns.length ? widget.columns : rows.length ? Object.keys(rows[0]) : [];

          return (
            <section className="n3-widget">
              <h3>{widget.title}</h3>
              {rows.length ? (
                <div style={{ overflowX: "auto" }}>
                  <table className="n3-table">
                    <thead>
                      <tr>
                        {columns.map((column) => (
                          <th key={column}>{column}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row, idx) => (
                        <tr key={idx}>
                          {columns.map((column) => (
                            <td key={column}>{String((row as any)[column] ?? "")}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <pre>{JSON.stringify(data ?? widget, null, 2)}</pre>
              )}
            </section>
          );
        }
        """
    ).strip() + "\n"
    _write_file(components_dir / "TableWidget.tsx", content)


def _write_form_widget(components_dir: Path) -> None:
    content = textwrap.dedent(
        """
        import { FormEvent, useState } from "react";
        import type { FormWidgetConfig } from "../lib/n3Client";
        import { useToast } from "./Toast";

        interface FormWidgetProps {
          widget: FormWidgetConfig;
          pageSlug: string;
        }

        export default function FormWidget({ widget, pageSlug }: FormWidgetProps) {
          const toast = useToast();
          const [submitting, setSubmitting] = useState(false);

          async function handleSubmit(event: FormEvent<HTMLFormElement>) {
            event.preventDefault();
            const form = new FormData(event.currentTarget);
            const payload: Record<string, unknown> = {};
            form.forEach((value, key) => {
              payload[key] = value;
            });
            try {
              setSubmitting(true);
              const response = await fetch(`/api/pages/${pageSlug}/forms/${widget.id}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
              });
              if (!response.ok) {
                throw new Error(`Request failed: ${response.status}`);
              }
              toast.show(widget.successMessage ?? "Form submitted");
            } catch (error) {
              console.warn("Form submission failed", error);
              toast.show("Unable to submit form right now");
            } finally {
              setSubmitting(false);
            }
          }

          return (
            <section className="n3-widget">
              <h3>{widget.title}</h3>
              <form onSubmit={handleSubmit} style={{ display: "grid", gap: "0.75rem", maxWidth: "420px" }}>
                {widget.fields.map((field) => (
                  <label key={field.name} style={{ display: "grid", gap: "0.35rem" }}>
                    <span style={{ fontWeight: 600 }}>{field.name}</span>
                    <input name={field.name} type={field.type ?? "text"} required style={{ padding: "0.55rem 0.75rem", borderRadius: "0.5rem", border: "1px solid rgba(15,23,42,0.18)" }} />
                  </label>
                ))}
                <button type="submit" disabled={submitting} style={{ padding: "0.65rem 1.25rem", borderRadius: "0.65rem", border: "none", background: "var(--primary, #2563eb)", color: "#fff", fontWeight: 600 }}>
                  {submitting ? "Submitting..." : "Submit"}
                </button>
              </form>
            </section>
          );
        }
        """
    ).strip() + "\n"
    _write_file(components_dir / "FormWidget.tsx", content)


def _write_text_widget(components_dir: Path) -> None:
    content = textwrap.dedent(
        """
        import type { CSSProperties } from "react";
        import type { TextWidgetConfig } from "../lib/n3Client";

        interface TextBlockProps {
          widget: TextWidgetConfig;
        }

        function normaliseStyles(styles: Record<string, string> | undefined): Record<string, string> {
          const result: Record<string, string> = {};
          if (!styles) {
            return result;
          }

          const sizeScale: Record<string, string> = {
            small: "0.875rem",
            medium: "1rem",
            large: "1.35rem",
            "x-large": "1.75rem",
            "xx-large": "2rem",
          };

          Object.entries(styles).forEach(([rawKey, rawValue]) => {
            const key = rawKey.toLowerCase();
            const value = rawValue;
            if (key === "align") {
              result.textAlign = value;
              return;
            }
            if (key === "weight") {
              const weight = value.toLowerCase();
              if (weight === "bold") {
                result.fontWeight = "700";
              } else if (weight === "light") {
                result.fontWeight = "300";
              } else if (weight === "normal") {
                result.fontWeight = "400";
              } else {
                result.fontWeight = value;
              }
              return;
            }
            if (key === "size") {
              const size = sizeScale[value.toLowerCase()] ?? value;
              result.fontSize = size;
              return;
            }
            const parts = rawKey.split(/[-_\\s]+/).filter(Boolean);
            if (!parts.length) {
              return;
            }
            const camel = parts[0] + parts.slice(1).map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1)).join("");
            result[camel] = value;
          });
          return result;
        }

        export default function TextBlock({ widget }: TextBlockProps) {
          return (
            <section className="n3-widget" style={normaliseStyles(widget.styles) as CSSProperties}>
              <p style={{ margin: 0 }}>{widget.text}</p>
            </section>
          );
        }
        """
    ).strip() + "\n"
    _write_file(components_dir / "TextBlock.tsx", content)


def _write_realtime_hook(lib_dir: Path) -> None:
    content = textwrap.dedent(
        """
        import { useEffect, useRef, useState } from "react";
        import type { PageDefinition } from "./n3Client";

        export interface RealtimeEvent {
          type?: string;
          slug: string;
          payload?: unknown;
          meta?: Record<string, unknown>;
        }

        export interface UseRealtimePageOptions {
          fallbackIntervalSeconds?: number;
          onEvent?: (event: RealtimeEvent) => void;
        }

        export interface UseRealtimePageState {
          connected: boolean;
          lastEvent: RealtimeEvent | null;
          lastError: string | null;
        }

        function getRuntime() {
          if (typeof window === "undefined") {
            return null;
          }
          const runtime = (window as any).N3Realtime;
          if (!runtime || typeof runtime.connectPage !== "function" || typeof runtime.disconnectPage !== "function") {
            return null;
          }
          return runtime;
        }

        function isMatchingSlug(event: Event, slug: string) {
          if (!(event instanceof CustomEvent)) {
            return false;
          }
          return Boolean(event.detail && event.detail.slug === slug);
        }

        function extractRealtimeEvent(event: Event, slug: string): RealtimeEvent | null {
          if (!(event instanceof CustomEvent)) {
            return null;
          }
          const detail = event.detail as { slug?: string; event?: RealtimeEvent } | undefined;
          if (!detail || detail.slug !== slug) {
            return null;
          }
          if (detail.event && typeof detail.event === "object") {
            const payload = detail.event;
            return {
              ...payload,
              slug: payload.slug ?? slug,
            };
          }
          return {
            slug,
            type: "message",
          };
        }

        export function useRealtimePage(definition: PageDefinition, options?: UseRealtimePageOptions): UseRealtimePageState {
          const [state, setState] = useState<UseRealtimePageState>({ connected: false, lastEvent: null, lastError: null });
          const optionsRef = useRef(options);
          optionsRef.current = options;

          const fallbackIntervalSeconds = typeof options?.fallbackIntervalSeconds === "number" && options.fallbackIntervalSeconds > 0
            ? options.fallbackIntervalSeconds
            : undefined;

          useEffect(() => {
            if (!definition.realtime || typeof document === "undefined") {
              setState({ connected: false, lastEvent: null, lastError: null });
              return;
            }
            const runtime = getRuntime();
            if (!runtime) {
              setState({ connected: false, lastEvent: null, lastError: "Realtime runtime unavailable." });
              return;
            }
            let alive = true;
            const slug = definition.slug;

            setState({ connected: false, lastEvent: null, lastError: null });

            const handleConnected: EventListener = (event) => {
              if (!alive || !isMatchingSlug(event, slug)) {
                return;
              }
              setState((prev) => ({ ...prev, connected: true, lastError: null }));
            };

            const handleDisconnected: EventListener = (event) => {
              if (!alive || !isMatchingSlug(event, slug)) {
                return;
              }
              setState((prev) => ({ ...prev, connected: false, lastError: "Connection lost. Retrying..." }));
            };

            const handleMessage: EventListener = (event) => {
              if (!alive) {
                return;
              }
              const realtimeEvent = extractRealtimeEvent(event, slug);
              if (!realtimeEvent) {
                return;
              }
              setState((prev) => ({
                connected: true,
                lastEvent: realtimeEvent,
                lastError: prev.lastError,
              }));
              optionsRef.current?.onEvent?.(realtimeEvent);
            };

            const listeners: Array<[string, EventListener]> = [
              ["n3:realtime:connected", handleConnected],
              ["n3:realtime:disconnected", handleDisconnected],
              ["n3:realtime:message", handleMessage],
              ["n3:realtime:snapshot", handleMessage],
              ["n3:realtime:hydration", handleMessage],
            ];

            listeners.forEach(([name, handler]) => document.addEventListener(name, handler));

            try {
              const connectOptions = fallbackIntervalSeconds ? { fallbackInterval: fallbackIntervalSeconds } : undefined;
              runtime.connectPage(slug, connectOptions);
            } catch (error) {
              console.warn("Failed to open realtime channel", error);
              setState((prev) => ({ ...prev, lastError: "Unable to open realtime channel." }));
            }

            return () => {
              alive = false;
              try {
                runtime.disconnectPage(slug);
              } catch (error) {
                console.warn("Failed to close realtime channel", error);
              }
              listeners.forEach(([name, handler]) => document.removeEventListener(name, handler));
            };
          }, [definition.realtime, definition.slug, fallbackIntervalSeconds]);

          return state;
        }
        """
    ).strip() + "\n"
    _write_file(lib_dir / "realtime.ts", content)


def _write_client_lib(lib_dir: Path) -> None:
    content = textwrap.dedent(
        """
        import { useCallback, useEffect, useRef, useState } from "react";

        export interface DataSourceRef {
          kind: string;
          name: string;
        }

        export interface TextWidgetConfig {
          id: string;
          type: "text";
          text: string;
          styles?: Record<string, string>;
        }

        export interface TableWidgetConfig {
          id: string;
          type: "table";
          title: string;
          columns: string[];
          source: DataSourceRef;
        }

        export interface ChartWidgetConfig {
          id: string;
          type: "chart";
          title: string;
          chartType?: string;
          source: DataSourceRef;
          x?: string | null;
          y?: string | null;
        }

        export interface FormWidgetField {
          name: string;
          type?: string;
        }

        export interface FormWidgetConfig {
          id: string;
          type: "form";
          title: string;
          fields: FormWidgetField[];
          successMessage?: string | null;
        }

        export type WidgetConfig =
          | TextWidgetConfig
          | TableWidgetConfig
          | ChartWidgetConfig
          | FormWidgetConfig;

        export interface PageDefinition {
          slug: string;
          route: string;
          title: string;
          description?: string | null;
          reactive: boolean;
          realtime: boolean;
          widgets: WidgetConfig[];
          preview: Record<string, unknown>;
        }

        export interface PageDataState {
          data: Record<string, unknown> | null;
          loading: boolean;
          error: string | null;
        }

        export interface PageDataReloadOptions {
          silent?: boolean;
        }

        export interface ApplyRealtimeOptions {
          replace?: boolean;
        }

        export interface UsePageDataResult extends PageDataState {
          reload: (options?: PageDataReloadOptions) => Promise<void>;
          applyRealtime: (payload: unknown, options?: ApplyRealtimeOptions) => void;
        }

        export interface MergeOptions {
          copy?: boolean;
        }

        function mergeHeaders(existing: HeadersInit | undefined, overrides: Record<string, string>): Record<string, string> {
          const result: Record<string, string> = {};
          if (existing) {
            if (Array.isArray(existing)) {
              existing.forEach((entry) => {
                if (!entry || entry.length < 2) {
                  return;
                }
                result[String(entry[0])] = String(entry[1]);
              });
            } else if (typeof Headers !== "undefined" && existing instanceof Headers) {
              existing.forEach((value, key) => {
                result[key] = value;
              });
            } else {
              Object.keys(existing as Record<string, string>).forEach((key) => {
                result[key] = String((existing as Record<string, string>)[key]);
              });
            }
          }
          Object.keys(overrides).forEach((key) => {
            result[key] = overrides[key];
          });
          return result;
        }

        async function requestJson<T = unknown>(path: string, init?: RequestInit): Promise<T | null> {
          const response = await fetch(path, init);
          if (!response.ok) {
            throw new Error(`Request failed: ${response.status}`);
          }
          const contentType = response.headers.get("content-type") ?? "";
          if (contentType.indexOf("application/json") === -1) {
            return null;
          }
          return (await response.json()) as T;
        }

        export async function fetchResource<T = unknown>(path: string, init?: RequestInit): Promise<T | null> {
          const headers = mergeHeaders(init?.headers, { Accept: "application/json" });
          return requestJson<T>(path, { ...init, headers });
        }

        export async function submitJson<T = unknown>(path: string, payload?: unknown, init?: RequestInit): Promise<T | null> {
          const headers = mergeHeaders(init?.headers, { Accept: "application/json", "Content-Type": "application/json" });
          const body = init?.body ?? JSON.stringify(payload ?? {});
          const method = init?.method ?? "POST";
          return requestJson<T>(path, { ...init, method, headers, body });
        }

        export function mergePartial<T extends Record<string, unknown>>(
          target: T | null | undefined,
          updates: Record<string, unknown> | null | undefined,
          options?: MergeOptions,
        ): T {
          const shouldCopy = options?.copy !== false;
          const base: Record<string, unknown> = target && typeof target === "object"
            ? (shouldCopy ? { ...(target as Record<string, unknown>) } : (target as Record<string, unknown>))
            : {};
          if (!updates || typeof updates !== "object") {
            return base as T;
          }
          Object.entries(updates).forEach(([key, value]) => {
            if (value && typeof value === "object" && !Array.isArray(value)) {
              const existing = base[key];
              const nextBase = existing && typeof existing === "object" && !Array.isArray(existing)
                ? (existing as Record<string, unknown>)
                : {};
              base[key] = mergePartial(nextBase, value as Record<string, unknown>, { copy: shouldCopy });
            } else {
              base[key] = value;
            }
          });
          return base as T;
        }

        export function usePageData(definition: PageDefinition): UsePageDataResult {
          const [state, setState] = useState<PageDataState>({ data: null, loading: true, error: null });
          const abortRef = useRef<AbortController | null>(null);
          const mountedRef = useRef(true);

          useEffect(() => {
            return () => {
              mountedRef.current = false;
              if (abortRef.current) {
                abortRef.current.abort();
              }
            };
          }, []);

          const fetchData = useCallback(async (options?: PageDataReloadOptions) => {
            abortRef.current?.abort();
            const controller = new AbortController();
            abortRef.current = controller;
            const silent = Boolean(options?.silent);
            setState((prev) => ({
              data: silent ? prev.data : null,
              loading: silent ? prev.loading : true,
              error: silent ? prev.error : null,
            }));
            try {
              const payload = await fetchResource<Record<string, unknown> | null>(`/api/pages/${definition.slug}`, {
                signal: controller.signal,
              });
              if (!controller.signal.aborted && mountedRef.current) {
                setState({ data: payload ?? {}, loading: false, error: null });
              }
            } catch (error) {
              if (controller.signal.aborted || !mountedRef.current) {
                return;
              }
              const message = error instanceof Error ? error.message : String(error);
              setState((prev) => ({
                data: silent ? prev.data : null,
                loading: false,
                error: message,
              }));
            }
          }, [definition.slug]);

          useEffect(() => {
            fetchData();
            return () => {
              abortRef.current?.abort();
            };
          }, [fetchData]);

          const reload = useCallback((options?: PageDataReloadOptions) => fetchData(options), [fetchData]);

          const applyRealtime = useCallback((payload: unknown, options?: ApplyRealtimeOptions) => {
            if (!payload || typeof payload !== "object" || !mountedRef.current) {
              return;
            }
            const replace = options?.replace ?? false;
            setState((prev) => ({
              data: replace
                ? (payload as Record<string, unknown>)
                : mergePartial(prev.data ?? {}, payload as Record<string, unknown>),
              loading: false,
              error: null,
            }));
          }, []);

          return { ...state, reload, applyRealtime };
        }

        export function resolveWidgetData(widgetId: string, pageData: Record<string, unknown> | null | undefined): unknown {
          if (!pageData || typeof pageData !== "object") {
            return undefined;
          }
          const buckets = [
            (pageData as any).widgets,
            (pageData as any).components,
            (pageData as any).data,
          ];
          for (const bucket of buckets) {
            if (bucket && typeof bucket === "object" && widgetId in bucket) {
              return (bucket as Record<string, unknown>)[widgetId];
            }
          }
          if (widgetId in (pageData as Record<string, unknown>)) {
            return (pageData as Record<string, unknown>)[widgetId];
          }
          return undefined;
        }

        export function ensureArray<T>(value: unknown): T[] {
          return Array.isArray(value) ? (value as T[]) : [];
        }
        """
    ).strip() + "\n"
    _write_file(lib_dir / "n3Client.ts", content)


def _write_app_tsx(src_dir: Path, page_builds: List[ReactPage]) -> None:
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
    _write_file(src_dir / "App.tsx", content)


def _write_page_component(pages_dir: Path, build: ReactPage) -> None:
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
          const { data, loading, error, reload, applyRealtime } = usePageData(PAGE_DEFINITION);
          const { connected: realtimeConnected, lastError: realtimeError } = useRealtimePage(PAGE_DEFINITION, {
            onEvent: (event) => {
              if (!event) {
                return;
              }
              if (event.payload && typeof event.payload === "object") {
                const replace = event.type === "snapshot" || event.type === "hydration";
                applyRealtime(event.payload, { replace });
                return;
              }
              if (event.type === "snapshot" || event.type === "hydration") {
                reload({ silent: true });
              }
              if (event.meta && typeof event.meta === "object") {
                const refresh = (event.meta as Record<string, unknown>).refresh;
                if (refresh === true || refresh === "page") {
                  reload({ silent: true });
                }
              }
            },
            fallbackIntervalSeconds: 15,
          });

          return (
            <Layout title={PAGE_DEFINITION.title} description={PAGE_DEFINITION.description} navLinks={NAV_LINKS}>
              {PAGE_DEFINITION.realtime ? (
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.85rem", marginBottom: "0.75rem", color: "var(--text-muted, #475569)" }}>
                  <span
                    aria-hidden="true"
                    style={{
                      width: "0.55rem",
                      height: "0.55rem",
                      borderRadius: "9999px",
                      backgroundColor: realtimeConnected ? "var(--success, #16a34a)" : "var(--warning, #dc2626)",
                      boxShadow: realtimeConnected ? "0 0 0 2px rgba(22, 163, 74, 0.25)" : "0 0 0 2px rgba(220, 38, 38, 0.25)",
                    }}
                  />
                  <span>{realtimeConnected ? "Live updates active" : "Waiting for live updates"}</span>
                  {realtimeError ? <span style={{ color: "var(--warning, #dc2626)" }}>• {realtimeError}</span> : null}
                </div>
              ) : null}
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
    _write_file(pages_dir / f"{build.file_name}.tsx", content)


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
