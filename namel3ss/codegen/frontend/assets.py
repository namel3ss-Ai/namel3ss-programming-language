"""Static assets emitted by the frontend generator."""

from __future__ import annotations

import textwrap
from typing import List

from namel3ss.ast import App

# Import widget library generator for modularization
# The massive JavaScript will be extracted to separate template files
try:
    from . import widget_library as _widget_lib
    _USE_MODULAR_WIDGETS = True
except ImportError:
    _USE_MODULAR_WIDGETS = False


def _generate_theme_variables(app: App) -> str:
    """Generate CSS custom properties from app theme."""
    theme_vars = [f"  --{key.replace(' ', '-')}: {value};" for key, value in app.theme.values.items()]
    if not theme_vars:
        return ""
    return ":root {\n" + "\n".join(theme_vars) + "\n}"


def _generate_base_styles() -> str:
    """Generate base typography and layout styles."""
    return """
body {
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 1rem;
  background-color: var(--background, #ffffff);
  color: var(--text, #333333);
}
table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 1rem;
}
th, td {
  border: 1px solid #ccc;
  padding: 0.5rem;
  text-align: left;
}
.toast {
  position: fixed;
  bottom: 1rem;
  right: 1rem;
  background-color: #333;
  color: #fff;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  opacity: 0;
  transition: opacity 0.5s;
}
.toast.show {
  opacity: 1;
}
"""


def _generate_widget_styles() -> str:
    """Generate widget container and component styles."""
    return """
.n3-widget {
    margin-bottom: 1.5rem;
    padding: 1rem;
    border-radius: 0.75rem;
    border: 1px solid rgba(15, 23, 42, 0.08);
    background-color: var(--panel-bg, #ffffff);
    box-shadow: var(--widget-shadow, 0 12px 24px rgba(15, 23, 42, 0.05));
}
.n3-widget--card {
    box-shadow: 0 18px 32px rgba(15, 23, 42, 0.12);
}
.n3-widget-chart,
.n3-widget-table {
    background-color: var(--surface, #ffffff);
}
.n3-align-center {
    text-align: center;
}
.n3-align-right {
    text-align: right;
}
.n3-emphasis-primary {
    border-color: var(--primary, #2563eb);
}
.n3-emphasis-secondary {
    border-color: var(--secondary, #6366f1);
}
.n3-table-container {
    overflow-x: auto;
}
.n3-table {
    width: 100%;
    border-collapse: collapse;
}
.n3-table-dense td,
.n3-table-dense th {
    padding: 0.35rem;
}
.n3-chart {
    width: 100%;
    height: 320px;
    display: block;
}
"""


def _generate_insight_styles() -> str:
    """Generate insight card, metric, and narrative styles."""
    return """
.n3-insights {
    margin-top: 2rem;
}
.n3-insights > h3 {
    margin-bottom: 0.75rem;
}
.n3-insight-grid {
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}
.n3-insight-card {
    border: 1px solid rgba(15, 23, 42, 0.12);
    border-radius: 0.75rem;
    padding: 1rem;
    background-color: var(--panel-bg, #ffffff);
    box-shadow: var(--widget-shadow, 0 12px 24px rgba(15, 23, 42, 0.04));
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}
.n3-insight-card__header h4 {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 600;
}
.n3-insight-metrics {
    display: grid;
    gap: 0.5rem;
}
.n3-insight-metric {
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 0.5rem;
    padding: 0.75rem;
    background-color: var(--surface, #f9fafb);
}
.n3-insight-metric__label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted, #6b7280);
    margin-bottom: 0.25rem;
}
.n3-insight-metric__value {
    font-size: 1.55rem;
    font-weight: 600;
    color: var(--text, #1f2937);
}
.n3-insight-metric__trend,
.n3-insight-metric__status {
    font-size: 0.85rem;
    color: var(--accent, #2563eb);
    margin-top: 0.35rem;
}
.n3-insight-metric--alert {
    border-color: var(--warning, #f97316);
}
.n3-insight-narratives {
    display: grid;
    gap: 0.5rem;
}
.n3-insight-narrative {
    border-radius: 0.5rem;
    padding: 0.75rem;
    border: 1px dashed rgba(15, 23, 42, 0.18);
    background-color: var(--surface-alt, #ffffff);
}
.n3-insight-empty {
    font-size: 0.85rem;
    color: var(--text-muted, #6b7280);
}
.n3-insight-card--error {
    border-color: var(--warning, #ef4444);
}
.n3-insight-card--error::after {
    content: attr(data-error);
    display: block;
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: var(--warning, #ef4444);
}
"""


def _generate_error_styles() -> str:
    """Generate error display and validation styles."""
    return """
.n3-widget-errors {
    margin-top: 0.75rem;
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid transparent;
    background-color: rgba(15, 23, 42, 0.05);
    color: var(--text, #1f2937);
    font-size: 0.9rem;
    line-height: 1.4;
}
.n3-widget-errors--hidden {
    display: none;
}
.n3-widget-errors--severity-info {
    border-left-color: var(--info, #2563eb);
    background-color: rgba(37, 99, 235, 0.08);
    color: var(--info-text, #1e3a8a);
}
.n3-widget-errors--severity-warning {
    border-left-color: var(--warning, #f97316);
    background-color: rgba(249, 115, 22, 0.12);
    color: var(--warning-text, #7c2d12);
}
.n3-widget-errors--severity-error {
    border-left-color: var(--warning, #ef4444);
    background-color: rgba(239, 68, 68, 0.1);
    color: var(--warning-text, #991b1b);
}
.n3-widget-errors--severity-debug {
    border-left-color: rgba(107, 114, 128, 0.6);
    background-color: rgba(107, 114, 128, 0.08);
    color: rgba(55, 65, 81, 0.9);
}
.n3-widget-error {
    display: block;
}
.n3-widget-error + .n3-widget-error {
    margin-top: 0.5rem;
}
.n3-widget-error__code {
    display: inline-block;
    font-weight: 600;
    text-transform: uppercase;
    margin-right: 0.5rem;
}
.n3-widget-error__message {
    font-weight: 500;
}
.n3-widget-error__detail {
    display: block;
    margin-top: 0.25rem;
    font-size: 0.82rem;
    opacity: 0.85;
}
.n3-field-error {
    display: none;
    margin-top: 0.35rem;
    font-size: 0.82rem;
    color: var(--warning, #dc2626);
}
.n3-field-error--visible {
    display: block;
}
.n3-input-error,
.n3-input-error:focus {
    border-color: var(--warning, #dc2626);
    outline-color: var(--warning, #dc2626);
}
"""


def _generate_form_styles() -> str:
    """Generate form, button, and interaction styles."""
    return """
.n3-form-field {
    margin-bottom: 1rem;
}
.n3-form--submitting {
    opacity: 0.85;
}
.n3-button--pending {
    opacity: 0.7;
    pointer-events: none;
}
.n3-action--pending {
    opacity: 0.7;
    pointer-events: none;
}
.n3-page-errors {
    margin: 1rem 0;
}
"""


def generate_styles(app: App) -> str:
    """
    Generate CSS from the app's theme and provide default styling.
    
    Assembles modular CSS sections:
    - Theme variables (CSS custom properties from app.theme)
    - Base styles (typography, layout)
    - Widget styles (containers, cards, charts, tables)
    - Insight styles (metrics, narratives, cards)
    - Error styles (validation, severity levels)
    - Form styles (inputs, buttons, interactions)
    
    Returns:
        Complete CSS stylesheet as a string
    """
    sections = [
        _generate_theme_variables(app),
        _generate_base_styles(),
        _generate_widget_styles(),
        _generate_insight_styles(),
        _generate_error_styles(),
        _generate_form_styles(),
    ]
    
    # Filter out empty sections and join with newlines
    return "\n".join(section for section in sections if section)


def generate_widget_library() -> str:
    """
    Return the shared widget runtime responsible for rendering interactive widgets.
    
    This function loads the widget library from modular JavaScript templates:
    - widget-core.js: Core utilities, CRUD operations, error handling
    - widget-rendering.js: Widget rendering, component registry, form/action handling
    - widget-realtime.js: WebSocket connections, fallback polling, state management
    
    Returns:
        Complete JavaScript code for N3Widgets, N3Crud, and N3Realtime namespaces
    """
    return _widget_lib.generate_widget_library()
