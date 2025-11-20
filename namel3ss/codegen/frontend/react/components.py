"""
React component generators for widgets and UI elements.

This module contains functions that generate React/TypeScript component files
for the Vite frontend, including navigation, layout, toast notifications,
and various widget types (charts, tables, forms, text blocks).
"""

import json
import textwrap
from pathlib import Path
from typing import Dict, List

from .utils import write_file


def write_navigation(lib_dir: Path, nav_links: List[Dict[str, str]]) -> None:
    """Generate navigation.ts with NavLink interface and constants."""
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
    write_file(lib_dir / "navigation.ts", content)


def write_toast_component(components_dir: Path) -> None:
    """Generate Toast.tsx with ToastProvider and useToast hook."""
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
    write_file(components_dir / "Toast.tsx", content)


def write_layout_component(components_dir: Path) -> None:
    """Generate Layout.tsx with header, navigation, and main content area."""
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
    write_file(components_dir / "Layout.tsx", content)


def write_chart_widget(components_dir: Path) -> None:
    """Generate ChartWidget.tsx for displaying chart data."""
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
    write_file(components_dir / "ChartWidget.tsx", content)


def write_table_widget(components_dir: Path) -> None:
    """Generate TableWidget.tsx for displaying tabular data."""
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
    write_file(components_dir / "TableWidget.tsx", content)


def write_form_widget(components_dir: Path) -> None:
    """Generate FormWidget.tsx for interactive form submission."""
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
    write_file(components_dir / "FormWidget.tsx", content)


def write_text_widget(components_dir: Path) -> None:
    """Generate TextBlock.tsx for displaying styled text content."""
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
            const parts = rawKey.split(/[-_\s]+/).filter(Boolean);
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
    write_file(components_dir / "TextBlock.tsx", content)
