"""Page and layout encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, List, Set, TYPE_CHECKING

from .expressions import _encode_value, _expression_to_runtime
from .utils import _page_api_path, _slugify_page_name, _slugify_route

if TYPE_CHECKING:
    from ....ast import LayoutMeta, LayoutSpec, Page, Prompt
    from .classes import PageSpec


def _encode_page(
    page: "Page",
    env_keys: Set[str],
    prompt_lookup: Dict[str, "Prompt"],
) -> "PageSpec":
    """Encode a page definition for backend state."""
    # Import here to avoid circular dependency
    from .statements import _encode_statement
    from .classes import PageSpec
    
    components: List[Any] = []
    for statement in page.body:
        component = _encode_statement(statement, env_keys, prompt_lookup)
        if component is not None:
            components.append(component)
    
    layout = _encode_layout_meta(page.layout_meta)
    api_path = _page_api_path(page.route or page.name)
    metadata_encoded = _encode_value(page.metadata or {}, env_keys)
    if not isinstance(metadata_encoded, dict):
        metadata_encoded = {"value": metadata_encoded} if metadata_encoded else {}
    
    return PageSpec(
        name=page.name,
        slug=_slugify_page_name(page.name),
        route=_slugify_route(page.route or page.name),
        index=0,  # Will be set by caller
        api_path=api_path,
        components=components,
        layout=layout,
    )


def _encode_layout_meta(layout: "LayoutMeta") -> Dict[str, Any]:
    """Encode page layout metadata."""
    if layout is None:
        return {"direction": "column", "spacing": "medium", "extras": {}}
    return {
        "direction": layout.direction or "column",
        "spacing": layout.spacing or "medium",
        "extras": dict(layout.extras or {}),
    }


def _encode_layout_spec(layout: "LayoutSpec") -> Dict[str, Any]:
    """Encode component layout specification."""
    if layout is None:
        return {}
    return {
        "width": layout.width,
        "height": layout.height,
        "variant": layout.variant,
        "order": layout.order,
        "area": layout.area,
        "breakpoint": layout.breakpoint,
        "props": dict(layout.props or {}),
    }
