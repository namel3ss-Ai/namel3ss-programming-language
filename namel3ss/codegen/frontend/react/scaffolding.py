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
        .ai-app {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          background-color: var(--background, #ffffff);
          color: var(--text, #0f172a);
        }
        .ai-main {
          flex: 1 1 auto;
          padding: 2rem clamp(1rem, 4vw, 4rem);
          max-width: 1200px;
          width: 100%;
          margin: 0 auto;
        }
        .ai-nav {
          display: flex;
          gap: 1rem;
          flex-wrap: wrap;
        }
        .ai-nav a {
          text-decoration: none;
          color: var(--text, #0f172a);
          font-weight: 600;
        }
        .ai-nav a.active {
          color: var(--primary, #2563eb);
        }
        .ai-widget {
          margin-bottom: 1.75rem;
          padding: 1.25rem;
          border-radius: 1rem;
          background-color: rgba(255, 255, 255, 0.95);
          box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
        }
        .ai-widget h3 {
          margin-top: 0;
        }
        .ai-toast {
          position: fixed;
          bottom: 1.5rem;
          right: 1.5rem;
          background: rgba(15, 23, 42, 0.92);
          color: #fff;
          padding: 0.75rem 1rem;
          border-radius: 0.75rem;
          box-shadow: 0 12px 32px rgba(15, 23, 42, 0.3);
        }
        table.ai-table {
          width: 100%;
          border-collapse: collapse;
        }
        table.ai-table th,
        table.ai-table td {
          border-bottom: 1px solid rgba(15, 23, 42, 0.08);
          padding: 0.65rem 0.75rem;
          text-align: left;
        }
        
        /* AI Semantic Components Styles */
        .chat-thread {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          padding: 1rem;
          background: var(--background, #ffffff);
          border-radius: 0.5rem;
        }
        .chat-message {
          display: flex;
          gap: 0.75rem;
          padding: 0.75rem;
          border-radius: 0.5rem;
        }
        .chat-message--user {
          background: var(--primary-light, #dbeafe);
        }
        .chat-message--assistant {
          background: var(--gray-light, #f1f5f9);
        }
        .chat-avatar {
          width: 2rem;
          height: 2rem;
          border-radius: 50%;
          background: var(--primary, #2563eb);
          color: white;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
          flex-shrink: 0;
        }
        .chat-message-header {
          display: flex;
          gap: 0.5rem;
          font-size: 0.875rem;
          color: var(--text-muted, #64748b);
          margin-bottom: 0.25rem;
        }
        .chat-copy-btn {
          margin-top: 0.5rem;
          padding: 0.25rem 0.5rem;
          font-size: 0.75rem;
          border: none;
          background: var(--gray, #e2e8f0);
          border-radius: 0.25rem;
          cursor: pointer;
        }
        
        .agent-panel {
          padding: 1.5rem;
          border-radius: 0.5rem;
          border: 1px solid var(--border, #e2e8f0);
          background: var(--background, #ffffff);
        }
        .agent-panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }
        .agent-status {
          padding: 0.25rem 0.75rem;
          border-radius: 1rem;
          font-size: 0.875rem;
          font-weight: 600;
        }
        .agent-status--running {
          background: #dcfce7;
          color: #166534;
        }
        .agent-status--idle {
          background: #e2e8f0;
          color: #475569;
        }
        .agent-metrics {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 1rem;
          margin: 1rem 0;
        }
        .agent-metric {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }
        .agent-metric-label {
          font-size: 0.875rem;
          color: var(--text-muted, #64748b);
        }
        .agent-metric-value {
          font-size: 1.5rem;
          font-weight: 700;
          color: var(--text, #0f172a);
        }
        
        .tool-call-view {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }
        .tool-call-item {
          border: 1px solid var(--border, #e2e8f0);
          border-radius: 0.5rem;
          padding: 1rem;
          background: var(--background, #ffffff);
        }
        .tool-call-header {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-weight: 600;
        }
        .tool-call-toggle {
          background: none;
          border: none;
          cursor: pointer;
          font-size: 0.875rem;
          padding: 0.25rem;
        }
        .tool-call-status {
          padding: 0.25rem 0.5rem;
          border-radius: 0.25rem;
          font-size: 0.75rem;
          font-weight: 600;
        }
        .tool-call-status--success {
          background: #dcfce7;
          color: #166534;
        }
        .tool-call-status--error {
          background: #fee2e2;
          color: #991b1b;
        }
        .tool-call-body pre {
          background: var(--gray-light, #f8fafc);
          padding: 0.75rem;
          border-radius: 0.25rem;
          overflow-x: auto;
          font-size: 0.875rem;
        }
        
        .log-view {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }
        .log-view-toolbar {
          display: flex;
          gap: 1rem;
          align-items: center;
          flex-wrap: wrap;
        }
        .log-search {
          flex: 1;
          min-width: 200px;
          padding: 0.5rem;
          border: 1px solid var(--border, #e2e8f0);
          border-radius: 0.25rem;
        }
        .log-level-filters {
          display: flex;
          gap: 0.5rem;
        }
        .log-level-filter {
          display: flex;
          align-items: center;
          gap: 0.25rem;
          font-size: 0.875rem;
        }
        .log-entries {
          border: 1px solid var(--border, #e2e8f0);
          border-radius: 0.5rem;
          background: var(--background, #ffffff);
        }
        .log-entry {
          padding: 0.5rem 1rem;
          border-bottom: 1px solid var(--border, #e2e8f0);
          display: flex;
          gap: 0.75rem;
          align-items: flex-start;
          font-family: monospace;
          font-size: 0.875rem;
        }
        .log-level {
          font-weight: 600;
          width: 5rem;
        }
        .log-level--error {
          color: #991b1b;
        }
        .log-level--warn {
          color: #ca8a04;
        }
        .log-level--info {
          color: #0284c7;
        }
        
        .evaluation-result {
          padding: 1.5rem;
          border: 1px solid var(--border, #e2e8f0);
          border-radius: 0.5rem;
        }
        .eval-metrics {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 1.5rem;
          margin: 1.5rem 0;
        }
        .eval-metric {
          padding: 1rem;
          border-radius: 0.5rem;
          background: var(--gray-light, #f8fafc);
        }
        .eval-metric--primary {
          border: 2px solid var(--primary, #2563eb);
        }
        .eval-metric-label {
          font-size: 0.875rem;
          color: var(--text-muted, #64748b);
          margin-bottom: 0.5rem;
        }
        .eval-metric-value {
          font-size: 2rem;
          font-weight: 700;
          color: var(--text, #0f172a);
        }
        .eval-errors-table {
          width: 100%;
          border-collapse: collapse;
          margin-top: 1rem;
        }
        .eval-errors-table th,
        .eval-errors-table td {
          border: 1px solid var(--border, #e2e8f0);
          padding: 0.75rem;
          text-align: left;
        }
        .eval-errors-table pre {
          margin: 0;
          white-space: pre-wrap;
          font-size: 0.75rem;
        }
        
        .diff-view {
          border: 1px solid var(--border, #e2e8f0);
          border-radius: 0.5rem;
          background: var(--background, #ffffff);
        }
        .diff-legend {
          display: flex;
          gap: 1rem;
          padding: 0.75rem 1rem;
          background: var(--gray-light, #f8fafc);
          border-bottom: 1px solid var(--border, #e2e8f0);
          font-size: 0.875rem;
        }
        .diff-legend-item--added {
          color: #166534;
        }
        .diff-legend-item--removed {
          color: #991b1b;
        }
        .diff-toolbar {
          display: flex;
          gap: 0.5rem;
          padding: 0.75rem 1rem;
          border-bottom: 1px solid var(--border, #e2e8f0);
        }
        .diff-toolbar button {
          padding: 0.5rem 1rem;
          border: 1px solid var(--border, #e2e8f0);
          border-radius: 0.25rem;
          background: white;
          cursor: pointer;
        }
        .diff-split {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1px;
          background: var(--border, #e2e8f0);
        }
        .diff-pane {
          background: var(--background, #ffffff);
          overflow-x: auto;
        }
        .diff-line {
          display: flex;
          font-family: monospace;
          font-size: 0.875rem;
          line-height: 1.5;
          padding: 0.25rem 0.5rem;
        }
        .diff-line--added {
          background: #dcfce7;
        }
        .diff-line--removed {
          background: #fee2e2;
        }
        .diff-line-number {
          width: 3rem;
          color: var(--text-muted, #94a3b8);
          text-align: right;
          padding-right: 0.75rem;
          user-select: none;
        }
        .diff-line-content {
          flex: 1;
        }
        """
    ).strip()
    content = styles + "\n\n" + base + "\n"
    write_file(src_dir / "index.css", content)
