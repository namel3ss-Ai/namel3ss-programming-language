"""Assemble the generated backend runtime module."""

from __future__ import annotations

from typing import Any, Dict, List

from namel3ss.ml import get_default_model_registry, load_model_registry

from namel3ss.codegen.backend.state import BackendState
from .context import render_context_registry_block
from .header import render_runtime_header
from .llm_init import render_llm_initialization_block
from .tool_init import render_tool_initialization_block
from .prompt_init import render_prompt_initialization_block
from .pages import _page_handlers_block, _page_to_dict, _render_page_function
from .realtime import render_broadcast_block
from .registries import render_registries_block
from .sections import collect_runtime_sections
from .symbolic_evaluator import evaluate_expression_tree

__all__ = [
    "_render_runtime_module", 
    "_render_page_function", 
    "_page_to_dict",
    "evaluate_expression_tree"
]


def _render_runtime_module(
    state: BackendState,
    embed_insights: bool,
    enable_realtime: bool,
    connector_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the runtime module by stitching together all required blocks."""
    parts: List[str] = []
    page_handler_entries: List[str] = []
    configured_model_registry = load_model_registry() or get_default_model_registry()

    parts.append(render_runtime_header(enable_realtime, connector_config))
    parts.append(render_context_registry_block())
    parts.append(render_broadcast_block(enable_realtime))
    parts.append(
        render_registries_block(
            state,
            configured_model_registry or {},
            embed_insights=embed_insights,
            enable_realtime=enable_realtime,
        )
    )
    parts.append(render_llm_initialization_block(state))
    parts.append(render_tool_initialization_block(state))
    parts.append(render_prompt_initialization_block(state))
    parts.extend(collect_runtime_sections())

    for page in state.pages:
        page_lines = _render_page_function(page)
        if not page_lines:
            continue
        func_name = f"page_{page.slug}_{page.index}"
        page_handler_entries.append(f"    {page.slug!r}: {func_name},")
        parts.append("\n".join(page_lines))

    parts.append(_page_handlers_block(page_handler_entries))
    return "\n\n".join(part for part in parts if part).strip() + "\n"
