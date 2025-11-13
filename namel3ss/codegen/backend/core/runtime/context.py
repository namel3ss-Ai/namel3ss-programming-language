"""Context registry utilities for the generated runtime."""

from __future__ import annotations

import textwrap


def render_context_registry_block() -> str:
    """Return the context registry class used by the runtime module."""
    context_runtime = '''
class ContextRegistry:
    """Simple registry for runtime context values."""

    def __init__(self) -> None:
        self._global: Dict[str, Any] = {}
        self._pages: Dict[str, Dict[str, Any]] = {}

    def set_global(self, values: Dict[str, Any]) -> None:
        self._global = dict(values)

    def set_page(self, slug: str, values: Dict[str, Any]) -> None:
        self._pages[slug] = dict(values)

    def build(self, slug: Optional[str]) -> Dict[str, Any]:
        base = dict(self._global)
        if slug and slug in self._pages:
            base.update(self._pages[slug])
        return base


CONTEXT = ContextRegistry()
'''
    return textwrap.dedent(context_runtime).strip()


__all__ = ["render_context_registry_block"]
