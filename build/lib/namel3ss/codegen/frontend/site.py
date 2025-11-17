"""Site generation entry points."""

from __future__ import annotations

import html
import json
import textwrap
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

from namel3ss.ast import App, Page

from .assets import generate_styles, generate_widget_library
from .renderers import render_statements
from .renderers.context import RenderContext
from .preview import PreviewDataResolver
from .slugs import slugify_identifier, slugify_page_name, slugify_route
from .theme import infer_theme_mode


def generate_site(
    app: App,
    output_dir: str,
    *,
    enable_realtime: bool = False,
    target: str = "static",
) -> None:
    """Generate a frontend project in ``output_dir`` for the provided ``app``.

    Parameters
    ----------
    app:
        Parsed application AST.
    output_dir:
        Destination directory for generated assets.
    enable_realtime:
        When ``True`` the generated frontend will attach realtime hooks where
        supported.
    target:
        Frontend flavour to emit. ``"static"`` preserves the legacy
        HTML/CSS output. ``"react-vite"`` produces a Vite + React + TypeScript
        project scaffold.
    """

    if target == "static":
        generate_static_site(app, output_dir, enable_realtime=enable_realtime)
        return

    if target == "react-vite":
        from .react_vite import generate_react_vite_site

        generate_react_vite_site(app, output_dir, enable_realtime=enable_realtime)
        return

    raise ValueError(f"Unsupported frontend target '{target}'")


def generate_static_site(app: App, output_dir: str, *, enable_realtime: bool = False) -> None:
    """Write a static representation of the app to ``output_dir``."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / 'styles.css').write_text(generate_styles(app), encoding='utf-8')

    first_page_path: Optional[str] = None
    for idx, page in enumerate(app.pages):
        slug = slugify_route(page.route)
        html_content = _generate_page_html(app, page, slug, idx, enable_realtime=enable_realtime)
        (out / f'{slug}.html').write_text(html_content, encoding='utf-8')
        if idx == 0:
            first_page_path = f'{slug}.html'

    models_page = _generate_models_page(app)
    if models_page:
        (out / 'models.html').write_text(models_page, encoding='utf-8')

    experiments_page = _generate_experiments_page(app)
    if experiments_page:
        (out / 'experiments.html').write_text(experiments_page, encoding='utf-8')

    index_links: List[Dict[str, str]] = []
    if first_page_path is not None:
        index_links.append({"label": "Application", "href": first_page_path})
    if app.models:
        index_links.append({"label": "Model Dashboard", "href": 'models.html'})
    if app.experiments:
        index_links.append({"label": "Experiment Metrics", "href": 'experiments.html'})

    index_items = '\n'.join(
        f"        <li><a href=\"{link['href']}\">{html.escape(link['label'])}</a></li>"
        for link in index_links
    ) or "        <li>No pages defined yet.</li>"

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
    <ul>
{index_items}
    </ul>
</body>
</html>
"""
    (out / 'index.html').write_text(index_html, encoding='utf-8')

    (out / 'scripts.js').write_text(generate_widget_library(), encoding='utf-8')


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
    backend_slug = slugify_page_name(page.name, page_index)
    theme_mode = infer_theme_mode(app, page)

    body_lines.append(f"<h2>{html.escape(page.name)}</h2>")
    body_lines.append(
        '<div class="n3-page-errors n3-widget-errors n3-widget-errors--hidden" data-n3-page-errors></div>'
    )

    preview_provider = PreviewDataResolver(app)

    ctx = RenderContext(
        app=app,
        page=page,
        slug=slug,
        backend_slug=backend_slug,
        body_lines=body_lines,
        inline_scripts=inline_scripts,
        counters=counters,
        widget_defs=widget_defs,
        theme_mode=theme_mode,
        component_tracker=component_tracker,
        preview=preview_provider,
    )
    render_statements(page.statements, ctx)

    if app.insights:
        insights_section_id = f"insights_{slug}"
        body_lines.append(f'<section class="n3-insights" id="{insights_section_id}">')
        body_lines.append('  <h3>Insights</h3>')
        body_lines.append('  <div class="n3-insight-grid">')
        for idx, insight in enumerate(app.insights):
            widget_id = f"insight_{slug}_{idx}"
            insight_slug = slugify_identifier(insight.name)
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

    nav_block = _build_nav_block(app, f"{slug}.html")
    if nav_block:
        html_parts.append(nav_block)

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
                        if (window.N3Widgets && typeof window.N3Widgets.hydratePage === 'function') {
                            window.N3Widgets.hydratePage("$slug", data || {});
                        }
                        document.querySelectorAll('[data-n3-text-template]').forEach(function(el) {
                            var tpl = el.getAttribute('data-n3-text-template') || '';
                            el.textContent = tpl.replace(/[{]([a-zA-Z_][a-zA-Z0-9_]*)[}]/g, function(match, name) {
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


def _build_nav_block(app: App, current_href: str) -> str:
        items: List[str] = []
        for page in app.pages:
                target_slug = f"{slugify_route(page.route)}.html"
                active = ' class="active"' if target_slug == current_href else ''
                items.append(f"  <li><a href=\"{target_slug}\"{active}>{html.escape(page.name)}</a></li>")
        if app.models:
                active = ' class="active"' if current_href == 'models.html' else ''
                items.append(f"  <li><a href=\"models.html\"{active}>Models</a></li>")
        if app.experiments:
                active = ' class="active"' if current_href == 'experiments.html' else ''
                items.append(f"  <li><a href=\"experiments.html\"{active}>Experiments</a></li>")
        if not items:
                return ""
        return "<nav><ul>\n" + "\n".join(items) + "\n</ul></nav>"


def _generate_models_page(app: App) -> Optional[str]:
        if not getattr(app, 'models', None):
                return None

        rows = []
        for model in app.models:
                slug = slugify_identifier(model.name, default="model")
                registry = getattr(model, 'registry', None)
                version = getattr(registry, 'version', '') if registry else ''
                accuracy = getattr(registry, 'accuracy', '') if registry else ''
                rows.append(
                        f"      <tr><td>{html.escape(model.name)}</td><td>{html.escape(model.model_type)}</td>"
                        f"<td>{html.escape(model.engine or '')}</td><td>{html.escape(str(version) or '')}</td>"
                        f"<td>{html.escape(str(accuracy) or '')}</td><td><code>{slug}</code></td></tr>"
                )
        tbody = "\n".join(rows) or "      <tr><td colspan=\"6\">No models defined.</td></tr>"
        nav_block = _build_nav_block(app, 'models.html')

        template = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>{html.escape(app.name)} – Models</title>
    <link rel=\"stylesheet\" href=\"styles.css\">
    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
</head>
<body data-n3-page-slug=\"models\">
    <h2>Registered Models</h2>
    {nav_block}
    <table class=\"n3-table\">
        <thead>
            <tr><th>Name</th><th>Type</th><th>Engine</th><th>Version</th><th>Accuracy</th><th>Slug</th></tr>
        </thead>
        <tbody>
{tbody}
        </tbody>
    </table>
    <section>
        <h3>Runtime Metrics</h3>
        <p>Data comes from <code>/api/pages/model/metrics</code>.</p>
        <pre id=\"model-metrics\">Loading metrics…</pre>
    </section>
    <section>
        <h3>Prediction Stub</h3>
        <p>POST JSON payloads to <code>/api/models/&lt;slug&gt;/predict</code> to exercise the deterministic
        backend helper.</p>
    </section>
    <script>
    (function() {{
        fetch('/api/pages/model/metrics', {{ headers: {{ 'Accept': 'application/json' }} }})
            .then(function(response) {{
                if (!response.ok) {{ throw new Error('request failed: ' + response.status); }}
                return response.json();
            }})
            .then(function(payload) {{
                var target = document.getElementById('model-metrics');
                target.textContent = JSON.stringify(payload.rows || payload, null, 2);
            }})
            .catch(function(err) {{
                document.getElementById('model-metrics').textContent = 'Unable to load metrics: ' + err.message;
            }});
    }})();
    </script>
    <script src=\"scripts.js\"></script>
</body>
</html>
"""
        return textwrap.dedent(template)


def _generate_experiments_page(app: App) -> Optional[str]:
        experiments = getattr(app, 'experiments', None)
        if not experiments:
                return None

        nav_block = _build_nav_block(app, 'experiments.html')
        experiments_payload = [
                {
                        "name": exp.name,
                        "slug": slugify_identifier(exp.name, default="experiment"),
                        "description": getattr(exp, 'description', None) or "",
                }
                for exp in experiments
        ]
        experiments_json = json.dumps(experiments_payload)

        template = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>{html.escape(app.name)} – Experiments</title>
    <link rel=\"stylesheet\" href=\"styles.css\">
    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
</head>
<body data-n3-page-slug=\"experiments\">
    <h2>Experiments</h2>
    {nav_block}
    <p>Each experiment fetches results from <code>/api/experiments/&lt;slug&gt;/metrics</code>.</p>
    <div id=\"n3-experiments\" class=\"n3-experiment-grid\"></div>
    <script>
    (function() {{
        var experiments = {experiments_json};
        var container = document.getElementById('n3-experiments');
        experiments.forEach(function(exp) {{
            var card = document.createElement('article');
            card.className = 'n3-experiment-card';
            var title = document.createElement('h3');
            title.textContent = exp.name;
            var desc = document.createElement('p');
            desc.textContent = exp.description || 'No description provided.';
            var metrics = document.createElement('pre');
            metrics.className = 'n3-experiment-metrics';
            metrics.textContent = 'Loading metrics…';
            card.appendChild(title);
            card.appendChild(desc);
            card.appendChild(metrics);
            container.appendChild(card);
            fetch('/api/experiments/' + exp.slug + '/metrics', {{ headers: {{ 'Accept': 'application/json' }} }})
                .then(function(response) {{
                    if (!response.ok) {{ throw new Error('request failed: ' + response.status); }}
                    return response.json();
                }})
                .then(function(payload) {{
                    metrics.textContent = JSON.stringify(payload, null, 2);
                }})
                .catch(function(err) {{
                    metrics.textContent = 'Unable to load metrics: ' + err.message;
                }});
        }});
    }})();
    </script>
    <script src=\"scripts.js\"></script>
</body>
</html>
"""
        return textwrap.dedent(template)
