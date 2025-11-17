"""Page rendering helpers for the generated runtime module."""

from __future__ import annotations

from typing import Any, Dict, List

from namel3ss.codegen.backend.state import PageSpec, _component_to_serializable
from ..utils import _format_literal


def _page_handlers_block(entries: List[str]) -> str:
    if entries:
        handler_lines = [
            "PAGE_HANDLERS: Dict[str, Callable[[Optional[AsyncSession]], Awaitable[Dict[str, Any]]]] = {"
        ]
        handler_lines.extend(entries)
        handler_lines.append("}")
        return "\n".join(handler_lines)
    return "PAGE_HANDLERS: Dict[str, Callable[[Optional[AsyncSession]], Awaitable[Dict[str, Any]]]] = {}"


def _render_page_function(page: PageSpec) -> List[str]:
    lines: List[str] = []
    func_name = f"page_{page.slug}_{page.index}"
    instructions = [_component_to_serializable(component) for component in page.components]
    lines.append(f"async def {func_name}(session: Optional[AsyncSession] = None) -> Dict[str, Any]:")
    lines.append(f"    context = build_context({page.slug!r})")
    lines.append("    scope = ScopeFrame()")
    lines.append("    scope.set('context', context)")
    lines.append(f"    instructions = {_format_literal(instructions)}")
    lines.append("    components = await render_statements(instructions, context, scope, session)")
    lines.append(f"    base_api_path = {page.api_path!r}")
    lines.append("    components = await prepare_page_components({'api_path': base_api_path, 'slug': %r}, components, context, session)" % page.slug)
    slug_lower = page.slug.lower()
    page_scope = f"page:{page.slug}"
    page_scope_lower = page_scope.lower()
    page_dot_scope_lower = f"page.{page.slug}".lower()
    lines.append("    page_errors: List[Dict[str, Any]] = []")
    lines.append("    for entry in _collect_runtime_errors(context):")
    lines.append("        if not isinstance(entry, dict):")
    lines.append("            continue")
    lines.append("        scope_value = entry.get('scope')")
    lines.append("        normalized_scope = str(scope_value).strip().lower() if scope_value is not None else ''")
    lines.append(f"        if normalized_scope in {{'', {slug_lower!r}, {page_scope_lower!r}, {page_dot_scope_lower!r}, 'page'}}:")
    lines.append(f"            entry['scope'] = {page_scope!r}")
    lines.append("        page_errors.append(entry)")
    lines.append("    return {")
    lines.append(f"        'name': {page.name!r},")
    lines.append(f"        'route': {page.route!r},")
    lines.append(f"        'slug': {page.slug!r},")
    lines.append("        'api_path': base_api_path,")
    lines.append(f"        'reactive': {page.reactive!r},")
    lines.append(f"        'refresh_policy': {_format_literal(page.refresh_policy)},")
    lines.append("        'components': components,")
    lines.append("        'errors': page_errors,")
    lines.append(f"        'layout': {_format_literal(page.layout)},")
    lines.append("    }")
    return lines


def _page_to_dict(page: PageSpec) -> Dict[str, Any]:
    return {
        "name": page.name,
        "route": page.route,
        "slug": page.slug,
        "index": page.index,
        "api_path": page.api_path,
        "reactive": page.reactive,
        "refresh_policy": page.refresh_policy,
        "layout": page.layout,
        "components": [component.__dict__ for component in page.components],
    }


__all__ = ["_render_page_function", "_page_to_dict", "_page_handlers_block"]
