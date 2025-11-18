"""Utilities for rendering the runtime module header."""

from __future__ import annotations

from typing import List


def render_runtime_header(enable_realtime: bool) -> str:
    """Produce import and constant definitions for the generated runtime module."""
    lines: List[str] = [
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
        "import hmac",
        "import inspect",
        "import importlib",
        "import importlib.util",
        "import json",
        "import logging",
        "import math",
        "import os",
        "import pickle",
        "import re",
        "import secrets",
        "import uuid",
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
            "from namel3ss.plugins.registry import get_plugin, PluginRegistryError",
            "",
            "from namel3ss.plugins.base import PLUGIN_CATEGORY_EVALUATOR",
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
            "CONTEXT_MARKER_KEY = \"__context__\"",
            "_IDENTIFIER_RE = re.compile(r\"^[A-Za-z_][A-Za-z0-9_]*$\")",
            "_AGGREGATE_ALIAS_PATTERN = re.compile(r\"\\s+as\\s+\", flags=re.IGNORECASE)",
            "_UPDATE_ASSIGNMENT_PATTERN = re.compile(r\"^\\s*([A-Za-z_][A-Za-z0-9_\\.]*?)\\s*=\\s*(.+)$\")",
            "_WHERE_CONDITION_PATTERN = re.compile(r\"^\\s*([A-Za-z_][A-Za-z0-9_\\.]*?)\\s*=\\s*(.+)$\")",
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


__all__ = ["render_runtime_header"]
