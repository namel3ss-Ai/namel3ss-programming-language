"""Scaffolding generators: HTML entry point, main.tsx, and CSS styles."""

from __future__ import annotations

import textwrap
from pathlib import Path

from namel3ss.ast import App

from ..assets import generate_styles
from .utils import write_file


def write_index_html(out: Path, title: str) -> None:
    """Generate the HTML entry point for the Vite app."""
    content = textwrap.dedent(
        f"""
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>{title}</title>
          </head>
          <body>
            <div id="root"></div>
            <script type="module" src="/src/main.tsx"></script>
          </body>
        </html>
        """
    ).strip() + "\n"
    write_file(out / "index.html", content)


def write_main_tsx(src_dir: Path) -> None:
    """Generate the main React entry point (main.tsx)."""
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
    write_file(src_dir / "main.tsx", content)


def write_index_css(src_dir: Path, app: App) -> None:
    """Generate the global CSS file combining theme styles and React app styles."""
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
    write_file(src_dir / "index.css", content)
