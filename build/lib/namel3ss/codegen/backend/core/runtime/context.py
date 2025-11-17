"""Context registry utilities for the generated runtime."""

from __future__ import annotations

import textwrap


def render_context_registry_block() -> str:
    """Return the context registry class used by the runtime module."""
    context_runtime = '''
from contextvars import ContextVar


_REQUEST_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("namel3ss_request_context", default={})
_REQUEST_CONTEXT_CACHE: Dict[str, Any] = {}


def set_request_context(values: Optional[Dict[str, Any]]) -> None:
    """Store request-scoped context for downstream runtime helpers."""

    global _REQUEST_CONTEXT_CACHE
    data = dict(values) if isinstance(values, dict) else {}
    _REQUEST_CONTEXT.set(data)
    _REQUEST_CONTEXT_CACHE = dict(data)


def get_request_context(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the current request context (if any)."""

    current = _REQUEST_CONTEXT.get()
    if not isinstance(current, dict) or not current:
        if _REQUEST_CONTEXT_CACHE:
            return dict(_REQUEST_CONTEXT_CACHE)
        return dict(default or {})
    return dict(current)


def clear_request_context() -> None:
    """Reset the request context to an empty mapping."""

    global _REQUEST_CONTEXT_CACHE
    _REQUEST_CONTEXT.set({})
    _REQUEST_CONTEXT_CACHE = {}


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
        request_context = get_request_context({})
        if request_context:
            base.setdefault("request", {}).update(request_context)
            tenant_value = request_context.get("tenant")
            if tenant_value is not None and "tenant" not in base:
                base["tenant"] = tenant_value
        return base


CONTEXT = ContextRegistry()
'''
    return textwrap.dedent(context_runtime).strip()


__all__ = ["render_context_registry_block"]
