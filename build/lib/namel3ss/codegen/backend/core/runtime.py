"""Render the generated runtime module."""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional

from namel3ss.ml import get_default_model_registry, load_model_registry

from ..state import BackendState, PageSpec, _component_to_serializable
from .runtime_sections import (
    ACTIONS_SECTION,
    CONFIG_SECTION,
    CONNECTORS_SECTION,
    CONTEXT_SECTION,
    DATASET_SECTION,
    INSIGHTS_SECTION,
    LLM_SECTION,
    MODELS_SECTION,
    PUBSUB_SECTION,
    PREDICTION_SECTION,
    REGISTRY_SECTION,
    RENDERING_SECTION,
    STREAMS_SECTION,
)
from .utils import _assign_literal, _format_literal

__all__ = ["_render_runtime_module"]


def _render_runtime_module(
    state: BackendState,
    embed_insights: bool,
    enable_realtime: bool,
) -> str:
    parts: List[str] = []
    page_handler_entries: List[str] = []
    configured_model_registry = load_model_registry() or get_default_model_registry()

    parts.append(_runtime_header(enable_realtime))
    parts.append(_context_registry_block())
    parts.append(_broadcast_block(enable_realtime))
    parts.append(
        _registries_block(
            state,
            configured_model_registry or {},
            embed_insights=embed_insights,
            enable_realtime=enable_realtime,
        )
    )
    parts.extend(_runtime_sections())

    for page in state.pages:
        page_lines = _render_page_function(page)
        if not page_lines:
            continue
        func_name = f"page_{page.slug}_{page.index}"
        page_handler_entries.append(f"    {page.slug!r}: {func_name},")
        parts.append("\n".join(page_lines))

    parts.append(_page_handlers_block(page_handler_entries))
    return "\n\n".join(part for part in parts if part).strip() + "\n"


def _runtime_header(enable_realtime: bool) -> str:
    lines = [
        '"""Generated runtime primitives for Namel3ss (N3)."""',
        "",
        "from __future__ import annotations",
        "",
        "import asyncio",
        "import ast",
        "import contextlib",
        "import copy",
        "import csv",
        "import functools",
        "import hashlib",
        "import inspect",
        "import importlib",
        "import importlib.util",
        "import json",
        "import logging",
        "import math",
        "import os",
        "import pickle",
        "import re",
        "import sys",
        "import threading",
        "import time",
        "from collections import defaultdict, OrderedDict, deque",
        "from types import SimpleNamespace",
    "from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union",
        "",
    "try:",
    "    import httpx",
    "except ImportError:",
    "    httpx = None  # type: ignore",
    "try:",
    "    from fastapi import HTTPException",
    "except ImportError:",
    "    class HTTPException(Exception):",
    "        def __init__(self, status_code: int, detail: Any = None) -> None:",
    "            super().__init__(detail)",
    "            self.status_code = status_code",
    "            self.detail = detail",
    "try:",
    "    from fastapi.responses import StreamingResponse",
    "except ImportError:",
    "    class StreamingResponse:",
    "        def __init__(self, content: Any, media_type: str = 'application/json') -> None:",
    "            self.content = content",
    "            self.media_type = media_type",
    "        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:",
    "            raise RuntimeError('StreamingResponse requires FastAPI installed')",
    "from starlette.concurrency import run_in_threadpool",
    ]
    if enable_realtime:
        lines.extend(
            [
                "try:",
                "    from fastapi import WebSocket, WebSocketDisconnect",
                "except ImportError:  # pragma: no cover - FastAPI <0.65 fallback",
                "    from fastapi.websockets import WebSocket, WebSocketDisconnect",
            ]
        )
    lines.extend(
        [
            "try:",
            "    from sqlalchemy import MetaData, text, bindparam, update",
            "    from sqlalchemy.sql import Select, table as sql_table, column",
            "    _HAS_SQLA_UPDATE = True",
            "except ImportError:  # pragma: no cover - optional dependency",
            "    from sqlalchemy import MetaData, text",
            "    from sqlalchemy.sql import Select",
            "    bindparam = None  # type: ignore",
            "    update = None  # type: ignore",
            "    sql_table = None  # type: ignore",
            "    column = None  # type: ignore",
            "    _HAS_SQLA_UPDATE = False",
            "from sqlalchemy.ext.asyncio import AsyncSession",
            "from pathlib import Path",
            "",
            "from .schemas import (",
            "    ActionResponse,",
            "    ChartResponse,",
            "    FormResponse,",
            "    InsightResponse,",
            "    PredictionResponse,",
            "    ExperimentResult,",
            "    TableResponse,",
            ")",
            "",
            "logger = logging.getLogger(__name__)",
            "if httpx is None:",
            "    class _MissingHTTPXAsyncClient:",
            "        def __init__(self, *args: Any, **kwargs: Any) -> None:",
            "            raise RuntimeError('httpx is required for HTTP connectors')",
            "        async def __aenter__(self) -> '_MissingHTTPXAsyncClient':",
            "            return self",
            "        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:",
            "            return None",
            "    _HTTPX_CLIENT_CLS = _MissingHTTPXAsyncClient",
            "else:",
            "    _HTTPX_CLIENT_CLS = httpx.AsyncClient",
            'CONTEXT_MARKER_KEY = "__context__"',
            '_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")',
            '_AGGREGATE_ALIAS_PATTERN = re.compile(r"\\s+as\\s+", flags=re.IGNORECASE)',
            '_UPDATE_ASSIGNMENT_PATTERN = re.compile(r"^\\s*([A-Za-z_][A-Za-z0-9_\\.]*?)\\s*=\\s*(.+)$")',
            '_WHERE_CONDITION_PATTERN = re.compile(r"^\\s*([A-Za-z_][A-Za-z0-9_\\.]*?)\\s*=\\s*(.+)$")',
        ]
    )
    lines.extend(
        [
            "try:",
            "    from sqlalchemy.orm import Session",
            "except ImportError:",
            "    Session = None  # type: ignore",
        ]
    )
    lines.extend(
        [
            "",
            "RUNTIME_SETTINGS: Dict[str, Any] = {}",
            "CACHE_CONFIG: Dict[str, Any] = {}",
            "STREAM_CONFIG: Dict[str, Any] = {}",
            "ASYNC_RUNTIME_ENABLED: bool = os.getenv('NAMEL3SS_RUNTIME_ASYNC', '0') in {'1', 'true', 'True'}",
            "DEFAULT_CACHE_MAX_ENTRIES: int = 256",
            "DEFAULT_CACHE_TTL: Optional[int] = None",
            "REDIS_URL: Optional[str] = os.getenv('NAMEL3SS_REDIS_URL')",
        "USE_REDIS_CACHE: bool = bool(REDIS_URL)",
        "USE_REDIS_PUBSUB: bool = os.getenv('NAMEL3SS_REDIS_PUBSUB', '0') in {'1', 'true', 'True'}",
        "REDIS_PUBSUB_CHANNEL_PREFIX: str = os.getenv('NAMEL3SS_REDIS_CHANNEL', 'namel3ss')",
        "CACHE_BACKEND: Optional['BaseCache'] = None",
        "REDIS_PUBSUB_PUBLISHER: Optional[Any] = None",
        "REDIS_PUBSUB_SUBSCRIBER: Optional[Any] = None",
        "REDIS_PUBSUB_TASK: Optional[asyncio.Task] = None",
        "REDIS_PUBSUB_LOCK = asyncio.Lock()",
            "DATASET_SUBSCRIBERS: Dict[str, Set[str]] = defaultdict(set)",
            "DATASET_CACHE_INDEX: Dict[str, Set[str]] = defaultdict(set)",
            "REFRESH_TASKS: Dict[str, asyncio.Task] = {}",
            "REFRESH_LOCK = asyncio.Lock()",
        ]
    )
    return "\n".join(lines).strip()


def _context_registry_block() -> str:
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


def _broadcast_block(enable_realtime: bool) -> str:
    if enable_realtime:
        broadcast = '''
class PageBroadcastManager:
    """Manage WebSocket connections for reactive pages."""

    def __init__(self) -> None:
        self._connections: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, slug: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.setdefault(slug, []).append(websocket)

    async def disconnect(self, slug: str, websocket: WebSocket) -> None:
        async with self._lock:
            connections = self._connections.get(slug)
            if not connections:
                return
            if websocket in connections:
                connections.remove(websocket)
            if not connections:
                self._connections.pop(slug, None)

    async def broadcast(self, slug: str, message: Dict[str, Any]) -> None:
        async with self._lock:
            connections = list(self._connections.get(slug, []))
        if not connections:
            return
        stale: List[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception:
                stale.append(connection)
        if stale:
            async with self._lock:
                current = self._connections.get(slug)
                if not current:
                    return
                for connection in stale:
                    if connection in current:
                        current.remove(connection)
                if not current:
                    self._connections.pop(slug, None)

    async def has_listeners(self, slug: str) -> bool:
        async with self._lock:
            return bool(self._connections.get(slug))


BROADCAST = PageBroadcastManager()
'''
        return textwrap.dedent(broadcast).strip()

    fallback = '''
class PageBroadcastManager:
    """No-op broadcast manager when realtime is disabled."""

    async def connect(self, slug: str, websocket: Any) -> None:  # pragma: no cover - noop
        return None

    async def disconnect(self, slug: str, websocket: Any) -> None:  # pragma: no cover - noop
        return None

    async def broadcast(self, slug: str, message: Dict[str, Any]) -> None:
        return None

    async def has_listeners(self, slug: str) -> bool:
        return False


BROADCAST = PageBroadcastManager()
'''
    return textwrap.dedent(fallback).strip()


def _registries_block(
    state: BackendState,
    configured_model_registry: Dict[str, Any],
    *,
    embed_insights: bool,
    enable_realtime: bool,
) -> str:
    registries = [
        _assign_literal("APP", "Dict[str, Any]", state.app),
        _assign_literal("DATASETS", "Dict[str, Dict[str, Any]]", state.datasets),
        _assign_literal("CONNECTORS", "Dict[str, Dict[str, Any]]", state.connectors),
        _assign_literal("AI_CONNECTORS", "Dict[str, Dict[str, Any]]", state.ai_connectors),
        _assign_literal("AI_TEMPLATES", "Dict[str, Dict[str, Any]]", state.templates),
        _assign_literal("AI_CHAINS", "Dict[str, Dict[str, Any]]", state.chains),
        _assign_literal("AI_EXPERIMENTS", "Dict[str, Dict[str, Any]]", state.experiments),
        _assign_literal("INSIGHTS", "Dict[str, Dict[str, Any]]", state.insights),
        _assign_literal(
            "MODEL_REGISTRY",
            "Dict[str, Dict[str, Any]]",
            configured_model_registry,
        ),
        "MODEL_CACHE: Dict[str, Any] = {}",
        "MODEL_LOADERS: Dict[str, Callable[[str, Dict[str, Any]], Any]] = {}",
        "MODEL_RUNNERS: Dict[str, Callable[[str, Any, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {}",
        "MODEL_EXPLAINERS: Dict[str, Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {}",
        _assign_literal("MODELS", "Dict[str, Dict[str, Any]]", state.models),
        _assign_literal(
            "PAGES", "List[Dict[str, Any]]", [_page_to_dict(page) for page in state.pages]
        ),
        "PAGE_SPEC_BY_SLUG: Dict[str, Dict[str, Any]] = {page['slug']: page for page in PAGES}",
        _assign_literal("ENV_KEYS", "List[str]", state.env_keys),
        f"EMBED_INSIGHTS: bool = {'True' if embed_insights else 'False'}",
        f"REALTIME_ENABLED: bool = {'True' if enable_realtime else 'False'}",
        "CONNECTOR_DRIVERS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[List[Dict[str, Any]]]]] = {}",
        "DATASET_TRANSFORMS: Dict[str, Callable[[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]], List[Dict[str, Any]]]] = {}",
    ]
    return "\n".join(registries).rstrip()


def _runtime_sections() -> List[str]:
    return [
        CONFIG_SECTION,
        PUBSUB_SECTION,
        STREAMS_SECTION,
        CONTEXT_SECTION,
        DATASET_SECTION,
        ACTIONS_SECTION,
        RENDERING_SECTION,
        REGISTRY_SECTION,
        LLM_SECTION,
        MODELS_SECTION,
        CONNECTORS_SECTION,
        PREDICTION_SECTION,
        INSIGHTS_SECTION,
    ]


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
