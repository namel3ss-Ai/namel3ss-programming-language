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
    lines.append("    return {")
    lines.append(f"        'name': {page.name!r},")
    lines.append(f"        'route': {page.route!r},")
    lines.append(f"        'slug': {page.slug!r},")
    lines.append(f"        'reactive': {page.reactive!r},")
    lines.append(f"        'refresh_policy': {_format_literal(page.refresh_policy)},")
    lines.append("        'components': components,")
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
