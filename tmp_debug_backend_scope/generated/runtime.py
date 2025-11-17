"""Generated runtime primitives for Namel3ss (N3)."""

from __future__ import annotations

import asyncio
import ast
import contextlib
import copy
import csv
import functools
import hashlib
import inspect
import importlib
import importlib.util
import json
import logging
import math
import os
import pickle
import re
import sys
import threading
import time
from collections import defaultdict, OrderedDict, deque
from types import SimpleNamespace
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore
try:
    from fastapi import HTTPException
except ImportError:
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: Any = None) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail
try:
    from fastapi.responses import StreamingResponse
except ImportError:
    class StreamingResponse:
        def __init__(self, content: Any, media_type: str = 'application/json') -> None:
            self.content = content
            self.media_type = media_type
        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            raise RuntimeError('StreamingResponse requires FastAPI installed')
from starlette.concurrency import run_in_threadpool
try:
    from sqlalchemy import MetaData, text, bindparam, update
    from sqlalchemy.sql import Select, table as sql_table, column
    _HAS_SQLA_UPDATE = True
except ImportError:  # pragma: no cover - optional dependency
    from sqlalchemy import MetaData, text
    from sqlalchemy.sql import Select
    bindparam = None  # type: ignore
    update = None  # type: ignore
    sql_table = None  # type: ignore
    column = None  # type: ignore
    _HAS_SQLA_UPDATE = False
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path

from .schemas import (
    ActionResponse,
    ChartResponse,
    FormResponse,
    InsightResponse,
    PredictionResponse,
    ExperimentResult,
    TableResponse,
)

logger = logging.getLogger(__name__)
if httpx is None:
    class _MissingHTTPXAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError('httpx is required for HTTP connectors')
        async def __aenter__(self) -> '_MissingHTTPXAsyncClient':
            return self
        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None
    _HTTPX_CLIENT_CLS = _MissingHTTPXAsyncClient
else:
    _HTTPX_CLIENT_CLS = httpx.AsyncClient
CONTEXT_MARKER_KEY = "__context__"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_AGGREGATE_ALIAS_PATTERN = re.compile(r"\s+as\s+", flags=re.IGNORECASE)
_UPDATE_ASSIGNMENT_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_\.]*?)\s*=\s*(.+)$")
_WHERE_CONDITION_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_\.]*?)\s*=\s*(.+)$")
try:
    from sqlalchemy.orm import Session
except ImportError:
    Session = None  # type: ignore

RUNTIME_SETTINGS: Dict[str, Any] = {}
CACHE_CONFIG: Dict[str, Any] = {}
STREAM_CONFIG: Dict[str, Any] = {}
ASYNC_RUNTIME_ENABLED: bool = os.getenv('NAMEL3SS_RUNTIME_ASYNC', '0') in {'1', 'true', 'True'}
DEFAULT_CACHE_MAX_ENTRIES: int = 256
DEFAULT_CACHE_TTL: Optional[int] = None
REDIS_URL: Optional[str] = os.getenv('NAMEL3SS_REDIS_URL')
USE_REDIS_CACHE: bool = bool(REDIS_URL)
USE_REDIS_PUBSUB: bool = os.getenv('NAMEL3SS_REDIS_PUBSUB', '0') in {'1', 'true', 'True'}
REDIS_PUBSUB_CHANNEL_PREFIX: str = os.getenv('NAMEL3SS_REDIS_CHANNEL', 'namel3ss')
CACHE_BACKEND: Optional['BaseCache'] = None
REDIS_PUBSUB_PUBLISHER: Optional[Any] = None
REDIS_PUBSUB_SUBSCRIBER: Optional[Any] = None
REDIS_PUBSUB_TASK: Optional[asyncio.Task] = None
REDIS_PUBSUB_LOCK = asyncio.Lock()
DATASET_SUBSCRIBERS: Dict[str, Set[str]] = defaultdict(set)
DATASET_CACHE_INDEX: Dict[str, Set[str]] = defaultdict(set)
REFRESH_TASKS: Dict[str, asyncio.Task] = {}
REFRESH_LOCK = asyncio.Lock()

from contextvars import ContextVar


_REQUEST_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("namel3ss_request_context", default={})


def set_request_context(values: Optional[Dict[str, Any]]) -> None:
    """Store request-scoped context for downstream runtime helpers."""

    _REQUEST_CONTEXT.set(dict(values) if isinstance(values, dict) else {})


def get_request_context(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the current request context (if any)."""

    current = _REQUEST_CONTEXT.get()
    if not isinstance(current, dict) or not current:
        return dict(default or {})
    return dict(current)


def clear_request_context() -> None:
    """Reset the request context to an empty mapping."""

    _REQUEST_CONTEXT.set({})


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

APP: Dict[str, Any] = {'name': 'StreamApp', 'database': None, 'theme': {}, 'variables': []}
DATASETS: Dict[str, Dict[str, Any]] = {'orders': {'name': 'orders',
            'source_type': 'table',
            'source': 'orders',
            'operations': [],
            'transforms': [],
            'schema': [],
            'features': [],
            'targets': [],
            'quality_checks': [],
            'profile': None,
            'connector': {'type': 'table', 'name': None, 'options': {}},
            'reactive': False,
            'refresh_policy': None,
            'cache_policy': None,
            'pagination': None,
            'streaming': None,
            'metadata': {},
            'lineage': {},
            'tags': [],
            'sample_rows': [{'id': 1, 'value': 10},
                            {'id': 2, 'value': 20},
                            {'id': 3, 'value': 30}]}}
CONNECTORS: Dict[str, Dict[str, Any]] = {'orders': {'type': 'table', 'name': None, 'options': {}}}
AI_CONNECTORS: Dict[str, Dict[str, Any]] = {}
AI_TEMPLATES: Dict[str, Dict[str, Any]] = {}
AI_CHAINS: Dict[str, Dict[str, Any]] = {}
AI_EXPERIMENTS: Dict[str, Dict[str, Any]] = {}
INSIGHTS: Dict[str, Dict[str, Any]] = {}
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {'churn_classifier': {'type': 'sklearn',
                      'framework': 'scikit-learn',
                      'version': 'v1',
                      'metrics': {'accuracy': 0.91},
                      'metadata': {'owner': 'team_data',
                                   'description': 'Customer churn predictor',
                                   'coefficients': {'tenure': -0.15,
                                                    'spend': -0.05,
                                                    'support_calls': 0.25},
                                   'intercept': -0.2,
                                   'loader': 'namel3ss.ml.hooks:load_churn_classifier',
                                   'runner': 'namel3ss.ml.hooks:run_churn_classifier',
                                   'explainer': 'namel3ss.ml.hooks:explain_churn_classifier',
                                   'trainer': 'namel3ss.ml.hooks:train_churn_classifier',
                                   'deployer': 'namel3ss.ml.hooks:deploy_churn_classifier'}},
 'image_classifier': {'type': 'deep_learning',
                      'framework': 'pytorch',
                      'version': 'v1',
                      'metrics': {'accuracy': 0.94, 'loss': 0.12},
                      'metadata': {'input_shape': [224, 224, 3],
                                   'model_file': 'models/image_classifier.pt',
                                   'feature_order': ['feature_a', 'feature_b'],
                                   'weights': [0.7, 0.3],
                                   'bias': 0.05,
                                   'threshold': 0.5,
                                   'loader': 'namel3ss.ml.hooks:load_image_classifier',
                                   'runner': 'namel3ss.ml.hooks:run_image_classifier',
                                   'explainer': 'namel3ss.ml.hooks:explain_image_classifier',
                                   'trainer': 'namel3ss.ml.hooks:train_image_classifier',
                                   'deployer': 'namel3ss.ml.hooks:deploy_image_classifier'}}}
MODEL_CACHE: Dict[str, Any] = {}
MODEL_LOADERS: Dict[str, Callable[[str, Dict[str, Any]], Any]] = {}
MODEL_RUNNERS: Dict[str, Callable[[str, Any, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {}
MODEL_EXPLAINERS: Dict[str, Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {}
MODELS: Dict[str, Dict[str, Any]] = {}
PAGES: List[Dict[str, Any]] = [{'name': 'Dashboard',
  'route': '/',
  'slug': 'dashboard',
  'index': 0,
  'api_path': '/api/pages/root',
  'reactive': False,
  'refresh_policy': None,
  'layout': {},
  'components': [{'type': 'table',
                  'payload': {'title': 'Orders',
                              'source_type': 'dataset',
                              'source': 'orders',
                              'columns': [],
                              'filter': None,
                              'sort': None,
                              'style': {},
                              'layout': None,
                              'insight': None,
                              'dynamic_columns': None}}]}]
PAGE_SPEC_BY_SLUG: Dict[str, Dict[str, Any]] = {page['slug']: page for page in PAGES}
ENV_KEYS: List[str] = []
EMBED_INSIGHTS: bool = False
REALTIME_ENABLED: bool = False
CONNECTOR_DRIVERS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[List[Dict[str, Any]]]]] = {}
DATASET_TRANSFORMS: Dict[str, Callable[[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]], List[Dict[str, Any]]]] = {}

def _runtime_setting(key: str, default: Any = None) -> Any:
    if key in RUNTIME_SETTINGS:
        return RUNTIME_SETTINGS[key]
    return default


def is_async_mode() -> bool:
    configured = _runtime_setting("async_mode")
    if configured is None:
        return bool(ASYNC_RUNTIME_ENABLED)
    return bool(configured)


def use_redis_cache() -> bool:
    configured = _runtime_setting("use_redis_cache")
    if configured is None:
        return bool(USE_REDIS_CACHE)
    return bool(configured)


def use_redis_pubsub() -> bool:
    configured = _runtime_setting("use_redis_pubsub")
    if configured is None:
        return bool(USE_REDIS_PUBSUB)
    return bool(configured)


def _resolve_stream_settings() -> Dict[str, Any]:
    runtime_settings = _runtime_setting("stream", {})
    if isinstance(runtime_settings, dict):
        return dict(runtime_settings)
    return {}


def _resolve_cache_settings() -> Dict[str, Any]:
    settings = dict(CACHE_CONFIG)
    runtime_settings = _runtime_setting("cache", {})
    if isinstance(runtime_settings, dict):
        settings.update(runtime_settings)
    return settings


class BaseCache:
    """Abstract cache interface supporting sync and async access."""

    async def aget(self, key: str) -> Any:
        return self.get(key)

    async def aset(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self.set(key, value, ttl)

    async def adelete(self, key: str) -> None:
        self.delete(key)

    def get(self, key: str) -> Any:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class LRUCache(BaseCache):
    """In-memory cache with LRU eviction and optional TTL handling."""

    def __init__(self, max_entries: int = DEFAULT_CACHE_MAX_ENTRIES, ttl: Optional[int] = DEFAULT_CACHE_TTL) -> None:
        self.max_entries = max(1, int(max_entries or DEFAULT_CACHE_MAX_ENTRIES))
        self.default_ttl = ttl
        self._store: "OrderedDict[str, Tuple[Any, Optional[float]]]" = OrderedDict()
        self._lock = threading.RLock()

    def _is_expired(self, expires_at: Optional[float]) -> bool:
        return expires_at is not None and expires_at <= time.time()

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            value, expires_at = entry
            if self._is_expired(expires_at):
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return copy.deepcopy(value)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expires_at = None
        duration = ttl if ttl is not None else self.default_ttl
        if duration:
            expires_at = time.time() + max(int(duration), 1)
        with self._lock:
            self._store[key] = (copy.deepcopy(value), expires_at)
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class RedisCache(BaseCache):
    """Redis cache wrapper used when redis.asyncio is available."""

    def __init__(self, url: str, namespace: str = "namel3ss") -> None:
        spec = importlib.util.find_spec("redis.asyncio")
        if spec is None:
            raise RuntimeError("redis.asyncio is required for RedisCache")
        redis_module = importlib.import_module("redis.asyncio")
        self._namespace = namespace.rstrip(":") + ":"
        self._client = redis_module.from_url(url)

    def _make_key(self, key: str) -> str:
        return f"{self._namespace}{key}"

    async def aget(self, key: str) -> Any:
        payload = await self._client.get(self._make_key(key))
        if payload is None:
            return None
        try:
            return json.loads(payload)
        except Exception:
            return None

    async def aset(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        payload = json.dumps(value)
        if ttl:
            await self._client.set(self._make_key(key), payload, ex=max(int(ttl), 1))
        else:
            await self._client.set(self._make_key(key), payload)

    async def adelete(self, key: str) -> None:
        await self._client.delete(self._make_key(key))

    def get(self, key: str) -> Any:
        raise RuntimeError("Use async methods with RedisCache")

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        raise RuntimeError("Use async methods with RedisCache")

    def delete(self, key: str) -> None:
        raise RuntimeError("Use async methods with RedisCache")

    def clear(self) -> None:
        raise RuntimeError("Use async methods with RedisCache")


def _create_cache_backend(settings: Optional[Dict[str, Any]] = None) -> BaseCache:
    configuration = _resolve_cache_settings()
    if settings:
        configuration.update(settings)
    backend = str(configuration.get("backend") or ("redis" if use_redis_cache() else "lru")).lower()
    if backend == "redis" and (use_redis_cache() or configuration.get("url")):
        url = configuration.get("url") or REDIS_URL
        if not url:
            raise RuntimeError("Redis cache requested but no redis URL configured")
        try:
            return RedisCache(url, namespace=configuration.get("namespace", "namel3ss"))
        except RuntimeError as exc:
            logger.warning("Falling back to LRU cache: %s", exc)
    max_entries = configuration.get("max_entries", DEFAULT_CACHE_MAX_ENTRIES)
    ttl = configuration.get("ttl", DEFAULT_CACHE_TTL)
    return LRUCache(max_entries=max_entries, ttl=ttl)


async def _cache_get(key: str) -> Any:
    global CACHE_BACKEND
    if CACHE_BACKEND is None:
        CACHE_BACKEND = _create_cache_backend()
    return await CACHE_BACKEND.aget(key)


async def _cache_set(key: str, value: Any, ttl: Optional[int] = None) -> None:
    global CACHE_BACKEND
    if CACHE_BACKEND is None:
        CACHE_BACKEND = _create_cache_backend()
    await CACHE_BACKEND.aset(key, value, ttl)


async def _cache_delete(key: str) -> None:
    global CACHE_BACKEND
    if CACHE_BACKEND is None:
        CACHE_BACKEND = _create_cache_backend()
    await CACHE_BACKEND.adelete(key)


def _require_dependency(module: str, extra: str) -> None:
    if not module:
        return
    if importlib.util.find_spec(module) is None:
        raise ImportError(f"Missing {module}; install 'namel3ss[{extra}]'")

_TOPIC_DELIMITER = "::"


def _normalize_topic(topic: Optional[str]) -> str:
    text = str(topic or "global").strip()
    return text or "global"


def _expand_topic_routes(topic: str) -> List[str]:
    if not topic:
        return ["global", "*"]
    segments = topic.split(_TOPIC_DELIMITER)
    routes: List[str] = []
    for index in range(len(segments), 0, -1):
        candidate = _TOPIC_DELIMITER.join(segments[:index])
        if candidate and candidate not in routes:
            routes.append(candidate)
    if "*" not in routes:
        routes.append("*")
    return routes


def _stream_setting(topic: str, key: str, default: Any) -> Any:
    config = STREAM_CONFIG.get(topic)
    if isinstance(config, dict) and key in config:
        return config[key]
    stream_settings = _runtime_setting("stream", {})
    if isinstance(stream_settings, dict) and key in stream_settings:
        return stream_settings[key]
    return default


def _next_stream_sequence(topic: str) -> int:
    state = STREAM_CONFIG.setdefault(topic, {})
    current = int(state.get("sequence") or 0)
    current += 1
    state["sequence"] = current
    return current


def _stream_channel_prefix() -> str:
    settings = _resolve_stream_settings()
    prefix = None
    if isinstance(settings, dict):
        prefix = settings.get("channel_prefix") or settings.get("namespace")
    base = prefix or REDIS_PUBSUB_CHANNEL_PREFIX or "namel3ss"
    return str(base).strip(":")


def _stream_channel_pattern() -> str:
    prefix = _stream_channel_prefix()
    if prefix:
        return f"{prefix}{_TOPIC_DELIMITER}*"
    return "*"


def _channel_for_topic(topic: str) -> str:
    prefix = _stream_channel_prefix()
    normalized = _normalize_topic(topic)
    if prefix:
        return f"{prefix}{_TOPIC_DELIMITER}{normalized}"
    return normalized


def _topic_from_channel(channel: Optional[str]) -> str:
    if not channel:
        return "global"
    text = str(channel)
    prefix = _stream_channel_prefix()
    prefix_token = f"{prefix}{_TOPIC_DELIMITER}" if prefix else ""
    if prefix_token and text.startswith(prefix_token):
        return text[len(prefix_token):]
    return text


PUBSUB_TOPICS: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
PUBSUB_LOCK = asyncio.Lock()


async def subscribe_topic(
    topic: str,
    *,
    queue_size: Optional[int] = None,
    replay_last_event: bool = True,
) -> asyncio.Queue:
    normalized = _normalize_topic(topic)
    maxsize = queue_size if queue_size is not None else int(_stream_setting(normalized, "queue_size", 128))
    if maxsize <= 0:
        maxsize = 1
    queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    async with PUBSUB_LOCK:
        PUBSUB_TOPICS[normalized].add(queue)
    if replay_last_event:
        last_event = STREAM_CONFIG.get(normalized, {}).get("last_event") if isinstance(STREAM_CONFIG.get(normalized), dict) else None
        if isinstance(last_event, dict):
            try:
                queue.put_nowait(dict(last_event))
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(dict(last_event))
                except asyncio.QueueFull:
                    logger.warning("Queue for topic '%s' is full; skipping replay", normalized)
    if use_redis_pubsub():
        await start_pubsub_listener()
    return queue


async def unsubscribe_topic(topic: str, queue: asyncio.Queue) -> None:
    normalized = _normalize_topic(topic)
    async with PUBSUB_LOCK:
        subscribers = PUBSUB_TOPICS.get(normalized)
        if not subscribers:
            return
        subscribers.discard(queue)
        if not subscribers:
            PUBSUB_TOPICS.pop(normalized, None)


async def publish_event(
    topic: str,
    payload: Dict[str, Any],
    *,
    propagate: bool = True,
) -> None:
    normalized = _normalize_topic(topic)
    routes = _expand_topic_routes(normalized)
    message = dict(payload)
    if not message.get("topic"):
        message["topic"] = normalized
    else:
        message["topic"] = _normalize_topic(message["topic"])
    if "id" not in message:
        message["id"] = _next_stream_sequence(normalized)
    timestamped = _with_timestamp(message)
    stream_state = STREAM_CONFIG.setdefault(normalized, {})
    stream_state["last_event"] = dict(timestamped)
    queues: Set[asyncio.Queue] = set()
    async with PUBSUB_LOCK:
        for route in routes:
            queues.update(PUBSUB_TOPICS.get(route, set()))
    if queues:
        for queue in queues:
            try:
                queue.put_nowait(dict(timestamped))
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(dict(timestamped))
                except asyncio.QueueFull:
                    logger.warning("Dropping pubsub message for topic '%s' due to slow consumer", normalized)
    if propagate and use_redis_pubsub():
        await _redis_publish_event(normalized, timestamped)


async def publish_dataset_event(dataset: str, payload: Dict[str, Any]) -> None:
    await publish_event(f"dataset::{dataset}", payload)


async def _redis_publish_event(topic: str, payload: Dict[str, Any]) -> None:
    try:
        redis_module = importlib.import_module("redis.asyncio")
    except Exception:
        logger.warning("Redis pub/sub requested but redis.asyncio is unavailable")
        return
    settings = _resolve_stream_settings()
    redis_settings = settings.get("redis") if isinstance(settings, dict) else {}
    url = None
    if isinstance(redis_settings, dict):
        url = redis_settings.get("url") or redis_settings.get("publisher_url")
    if not url:
        url = settings.get("redis_url") if isinstance(settings, dict) else None
    if not url:
        url = REDIS_URL
    if not url:
        logger.warning("Redis pub/sub enabled but no redis URL configured")
        return
    channel = _channel_for_topic(topic)
    client = None
    try:
        client = redis_module.from_url(url, decode_responses=True)
        await client.publish(channel, json.dumps(payload, default=str))
    except Exception:
        logger.exception("Failed to publish redis event for topic '%s'", topic)
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass


async def start_pubsub_listener(pattern: Optional[str] = None) -> None:
    if not use_redis_pubsub():
        return
    actual_pattern = pattern or _stream_channel_pattern()
    async with REDIS_PUBSUB_LOCK:
        if REDIS_PUBSUB_TASK and not REDIS_PUBSUB_TASK.done():
            return
        REDIS_PUBSUB_TASK = asyncio.create_task(_run_pubsub_listener(actual_pattern))


async def stop_pubsub_listener() -> None:
    async with REDIS_PUBSUB_LOCK:
        task = REDIS_PUBSUB_TASK
        REDIS_PUBSUB_TASK = None
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:  # pragma: no cover - normal shutdown
            pass


async def _run_pubsub_listener(pattern: str) -> None:
    try:
        redis_module = importlib.import_module("redis.asyncio")
    except Exception:
        logger.warning("Redis pub/sub listener unavailable: redis.asyncio missing")
        return
    settings = _resolve_stream_settings()
    redis_settings = settings.get("redis") if isinstance(settings, dict) else {}
    url = None
    if isinstance(redis_settings, dict):
        url = redis_settings.get("url") or redis_settings.get("subscriber_url")
    if not url:
        url = settings.get("redis_url") if isinstance(settings, dict) else None
    if not url:
        url = REDIS_URL
    if not url:
        logger.warning("Redis pub/sub listener cannot start without redis URL")
        return
    client = None
    pubsub = None
    try:
        client = redis_module.from_url(url, decode_responses=True)
        pubsub = client.pubsub(ignore_subscribe_messages=True)
    except Exception:
        logger.exception("Failed to initialise redis pub/sub listener")
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass
        return
    try:
        await pubsub.psubscribe(pattern)
        async for message in pubsub.listen():
            if not message:
                continue
            message_type = message.get("type")
            if message_type not in {"message", "pmessage"}:
                continue
            data = message.get("data")
            if data is None:
                continue
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            try:
                payload = json.loads(data)
            except Exception:
                logger.debug("Ignoring non-JSON pubsub payload for pattern '%s'", pattern)
                continue
            topic = payload.get("topic")
            if not topic:
                channel = message.get("channel") or message.get("pattern")
                topic = _topic_from_channel(channel)
            await publish_event(topic or "global", payload, propagate=False)
    except asyncio.CancelledError:  # pragma: no cover - shutdown
        raise
    except Exception:
        logger.exception("Redis pub/sub listener failed")
    finally:
        if pubsub is not None:
            try:
                await pubsub.reset()
            except Exception:
                pass
            try:
                await pubsub.close()
            except Exception:
                pass
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass

async def stream_topic(
    topic: str,
    heartbeat: Optional[int] = None,
    *,
    replay_last_event: bool = True,
) -> StreamingResponse:
    normalized = _normalize_topic(topic)
    queue = await subscribe_topic(normalized, replay_last_event=replay_last_event)
    interval = heartbeat if heartbeat is not None else int(_stream_setting(normalized, "heartbeat", 30))
    if interval <= 0:
        interval = 30
    pytest_mode = bool(os.getenv("PYTEST_CURRENT_TEST"))

    async def event_source() -> AsyncIterator[bytes]:
        idle_heartbeats = 0
        delivered_payload = False
        try:
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=interval)
                except asyncio.TimeoutError:
                    if pytest_mode and delivered_payload:
                        break
                    yield b": heartbeat\n\n"
                    if pytest_mode:
                        idle_heartbeats += 1
                        if idle_heartbeats >= 1:
                            break
                    continue
                data = json.dumps(message, default=str)
                yield f"data: {data}\n\n".encode("utf-8")
                delivered_payload = True
                idle_heartbeats = 0
                if pytest_mode and queue.empty():
                    break
        finally:
            await unsubscribe_topic(normalized, queue)

    response = StreamingResponse(event_source(), media_type="text/event-stream")
    response.content = event_source
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


async def stream_dataset(dataset: str, heartbeat: Optional[int] = None) -> StreamingResponse:
    base_topic = f"dataset::{dataset}"
    normalized = _normalize_topic(base_topic)
    context = build_context(None)
    dataset_spec = DATASETS.get(dataset)
    scope = context.get("page") or "global"
    if dataset_spec is not None:
        resolved_scope, _, _ = _dataset_cache_settings(dataset_spec, context)
        scope = resolved_scope or scope
    subscription_key = _make_dataset_cache_key(dataset, scope, context)
    queue = await subscribe_topic(normalized, replay_last_event=True)
    interval = heartbeat if heartbeat is not None else int(_stream_setting(normalized, "heartbeat", 30))
    if interval <= 0:
        interval = 30
    pytest_mode = bool(os.getenv("PYTEST_CURRENT_TEST"))

    async def event_source() -> AsyncIterator[bytes]:
        idle_heartbeats = 0
        delivered_payload = False
        try:
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=interval)
                except asyncio.TimeoutError:
                    if pytest_mode and delivered_payload:
                        break
                    yield b": heartbeat\n\n"
                    if pytest_mode:
                        idle_heartbeats += 1
                        if idle_heartbeats >= 1:
                            break
                    continue
                recipient = message.get("recipient") if isinstance(message, dict) else None
                if recipient and recipient != subscription_key:
                    continue
                data = json.dumps(message, default=str)
                yield f"data: {data}\n\n".encode("utf-8")
                delivered_payload = True
                idle_heartbeats = 0
                if pytest_mode and queue.empty():
                    break
        finally:
            await unsubscribe_topic(normalized, queue)

    response = StreamingResponse(event_source(), media_type="text/event-stream")
    response.content = event_source
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


async def stream_page(slug: str, heartbeat: Optional[int] = None) -> StreamingResponse:
    return await stream_topic(f"page::{slug}", heartbeat=heartbeat)


async def emit_page_event(slug: str, payload: Dict[str, Any]) -> None:
    await publish_event(f"page::{slug}", payload)


async def emit_global_event(payload: Dict[str, Any]) -> None:
    await publish_event("global", payload)

def build_context(page_slug: Optional[str]) -> Dict[str, Any]:
    base = CONTEXT.build(page_slug)
    context: Dict[str, Any] = dict(base)
    context.setdefault("vars", {})
    for variable in APP.get("variables", []):
        context["vars"][variable["name"]] = _resolve_placeholders(
            variable.get("value"), context
        )
    env_values = {name: os.getenv(name) for name in ENV_KEYS}
    context.setdefault("env", {}).update(env_values)
    if page_slug:
        context.setdefault("page", page_slug)
    context.setdefault("app", APP)
    context.setdefault("models", MODEL_REGISTRY)
    context.setdefault("connectors", AI_CONNECTORS)
    context.setdefault("templates", AI_TEMPLATES)
    context.setdefault("chains", AI_CHAINS)
    context.setdefault("experiments", AI_EXPERIMENTS)
    context.setdefault("call_python_model", call_python_model)
    context.setdefault("call_llm_connector", call_llm_connector)
    context.setdefault("run_chain", run_chain)
    context.setdefault("evaluate_experiment", evaluate_experiment)
    context.setdefault("predict", predict)
    context.setdefault("datasets", DATASETS)
    return context


def _resolve_placeholders(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if CONTEXT_MARKER_KEY in value:
            marker = value[CONTEXT_MARKER_KEY]
            scope = marker.get("scope")
            path = marker.get("path", [])
            default = marker.get("default")
            return _resolve_context_scope(scope, path, context, default)
        return {key: _resolve_placeholders(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(item, context) for item in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        return os.getenv(env_name, value)
    return value


def _resolve_context_scope(
    scope: Optional[str],
    path: Iterable[str],
    context: Dict[str, Any],
    default: Any = None,
) -> Any:
    parts = list(path)
    if scope in (None, "ctx", "context"):
        return _resolve_context_path(context, parts, default)
    if scope == "env":
        if not parts:
            return default
        return os.getenv(parts[0], default)
    if scope == "vars":
        return _resolve_context_path(context.get("vars", {}), parts, default)
    target = context.get(scope)
    return _resolve_context_path(target, parts, default)


def _resolve_context_path(
    context: Optional[Dict[str, Any]],
    path: Iterable[str],
    default: Any = None,
) -> Any:
    current: Any = context
    for segment in path:
        if isinstance(current, dict):
            current = current.get(segment)
        else:
            current = None
        if current is None:
            return default
    return current


TEMPLATE_PATTERN = re.compile(r"{([^{}]+)}")


def _render_template_value(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            token = match.group(1).strip()
            if not token:
                return match.group(0)
            if token.startswith("$"):
                return os.getenv(token[1:], "")
            if ":" in token:
                scope, _, path = token.partition(":")
                parts = [segment for segment in path.split(".") if segment]
                resolved = _resolve_context_scope(scope, parts, context, "")
                return "" if resolved is None else str(resolved)
            value = _resolve_context_path(context.get("vars"), [token], None)
            if value is not None:
                return str(value)
            resolved = _resolve_context_path(context, [token], "")
            return "" if resolved is None else str(resolved)
        return TEMPLATE_PATTERN.sub(_replace, value)
    if isinstance(value, list):
        return [_render_template_value(item, context) for item in value]
    if isinstance(value, dict):
        return {
            key: _render_template_value(item, context) for key, item in value.items()
        }
    return value


def register_connector_driver(
    connector_type: str,
    handler: Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[Any]],
) -> None:
    if not connector_type or handler is None:
        return
    CONNECTOR_DRIVERS[connector_type.lower()] = handler


def register_dataset_transform(
    transform_type: str,
    handler: Callable[[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]], List[Dict[str, Any]]],
) -> None:
    if not transform_type or handler is None:
        return
    DATASET_TRANSFORMS[transform_type.lower()] = handler


class BreakFlow(Exception):
    """Internal signal to indicate a ``break`` statement was encountered."""

    def __init__(self) -> None:
        super().__init__()
        self.components: List[Dict[str, Any]] = []

    def extend(self, items: Iterable[Dict[str, Any]]) -> None:
        if not items:
            return
        self.components.extend(items)


class ContinueFlow(Exception):
    """Internal signal to indicate a ``continue`` statement was encountered."""

    def __init__(self) -> None:
        super().__init__()
        self.components: List[Dict[str, Any]] = []

    def extend(self, items: Iterable[Dict[str, Any]]) -> None:
        if not items:
            return
        self.components.extend(items)


class ScopeFrame:
    """Hierarchical scope for storing variables during page rendering."""

    def __init__(self, parent: Optional['ScopeFrame'] = None) -> None:
        self.parent = parent
        self._values: Dict[str, Any] = {}

    def child(self) -> 'ScopeFrame':
        return ScopeFrame(self)

    def contains(self, name: str) -> bool:
        if name in self._values:
            return True
        if self.parent is not None:
            return self.parent.contains(name)
        return False

    def get(self, name: str, default: Any = None) -> Any:
        if name in self._values:
            return self._values[name]
        if self.parent is not None:
            return self.parent.get(name, default)
        return default

    def assign(self, name: str, value: Any) -> None:
        if name in self._values:
            self._values[name] = value
            return
        if self.parent is not None and self.parent.contains(name):
            self.parent.assign(name, value)
            return
        self._values[name] = value

    def set(self, name: str, value: Any) -> None:
        self.assign(name, value)

    def bind(self, name: str, value: Any) -> None:
        self._values[name] = value

    def remove_local(self, name: str) -> None:
        self._values.pop(name, None)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.parent is not None:
            data.update(self.parent.to_dict())
        data.update(self._values)
        return data


_MISSING = object()

_RUNTIME_CALLABLES: Dict[str, Callable[..., Any]] = {
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "sorted": sorted,
    "abs": abs,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "range": range,
}


def _assign_variable(scope: ScopeFrame, context: Dict[str, Any], name: str, value: Any) -> None:
    scope.assign(name, value)
    context.setdefault("vars", {})[name] = value


def _bind_variable(scope: ScopeFrame, context: Dict[str, Any], name: str, value: Any) -> None:
    scope.bind(name, value)
    context.setdefault("vars", {})[name] = value


def _restore_variable(context: Dict[str, Any], name: str, previous: Any) -> None:
    vars_map = context.setdefault("vars", {})
    if previous is _MISSING:
        vars_map.pop(name, None)
    else:
        vars_map[name] = previous


def _runtime_truthy(value: Any) -> bool:
    return bool(value)


def _clone_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def _ensure_numeric(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _resolve_option_dict(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(raw or {})

from namel3ss.codegen.backend.core.runtime.datasets import (
    fetch_dataset_rows as _fetch_dataset_rows_impl,
    execute_sql as _execute_sql_impl,
    load_dataset_source as _load_dataset_source_impl,
    execute_dataset_pipeline as _execute_dataset_pipeline_impl,
    resolve_connector as _resolve_connector_impl,
)


class DatasetExpressionError(RuntimeError):
    """Raised when a dataset expression violates sandbox restrictions."""


class _SandboxGuard(dict):
    __slots__ = ()

    def __setitem__(self, key, value):  # pragma: no cover - defensive guard
        raise PermissionError("sandbox scope is read-only")


def _sandbox_validate(expression: str) -> None:
    prohibited = {"__import__", "open", "exec", "eval", "compile", "globals", "locals"}
    for keyword in prohibited:
        if keyword in expression:
            raise DatasetExpressionError(f"Use of '{keyword}' is not permitted in dataset expressions")


def _sandbox_eval_expression(expression: str, scope: Dict[str, Any]) -> Any:
    _sandbox_validate(expression)
    compiled = compile(expression, "<dataset_expr>", "eval")
    return eval(compiled, {"__builtins__": {}}, _SandboxGuard(scope))


async def fetch_dataset_rows(
    key: str,
    session: AsyncSession,
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return await _fetch_dataset_rows_impl(
        key,
        session,
        context,
        datasets=DATASETS,
        resolve_connector=_resolve_connector,
        dataset_cache_settings=_dataset_cache_settings,
        make_dataset_cache_key=_make_dataset_cache_key,
        dataset_cache_index=DATASET_CACHE_INDEX,
        cache_get=_cache_get,
        clone_rows=_clone_rows,
        load_dataset_source=_load_dataset_source,
        execute_dataset_pipeline=_execute_dataset_pipeline,
        cache_set=_cache_set,
        broadcast_dataset_refresh=_broadcast_dataset_refresh,
        schedule_dataset_refresh=_schedule_dataset_refresh,
    )


async def _execute_sql(session: Optional[AsyncSession], query: Any) -> Any:
    return await _execute_sql_impl(
        session,
        query,
        async_session_type=AsyncSession,
        sync_session_type=Session,
        run_in_threadpool=run_in_threadpool,
    )


async def _load_dataset_source(
    dataset: Dict[str, Any],
    connector: Dict[str, Any],
    session: AsyncSession,
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return await _load_dataset_source_impl(
        dataset,
        connector,
        session,
        context,
        connector_drivers=CONNECTOR_DRIVERS,
        httpx_client_cls=_HTTPX_CLIENT_CLS,
        normalize_connector_rows=_normalize_connector_rows,
        execute_sql=_execute_sql,
        logger=logger,
        fetch_dataset_rows_fn=fetch_dataset_rows,
    )


async def _execute_dataset_pipeline(
    dataset: Dict[str, Any],
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return await _execute_dataset_pipeline_impl(
        dataset,
        rows,
        context,
        clone_rows=_clone_rows,
        apply_filter=_apply_filter,
        apply_computed_column=_apply_computed_column,
        apply_order=_apply_order,
        apply_window_operation=_apply_window_operation,
        apply_group_aggregate=_apply_group_aggregate,
        apply_transforms=_apply_transforms,
        evaluate_quality_checks=_evaluate_quality_checks,
    )


def _resolve_connector(dataset: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    return _resolve_connector_impl(
        dataset,
        context,
        deepcopy=copy.deepcopy,
        resolve_placeholders=_resolve_placeholders,
    )


def _parse_aggregate_expression(expression: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if expression is None:
        return None, None
    text = expression.strip()
    if not text:
        return None, None
    parts = _AGGREGATE_ALIAS_PATTERN.split(text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip() or None
    return text, None


def _aggregate_result_key(function: str, expression: Optional[str], alias: Optional[str]) -> str:
    if alias:
        return alias
    base = f"{function}_{expression or 'value'}"
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", base).strip("_")
    return sanitized or f"{function}_value"


def _evaluate_dataset_expression(
    expression: Optional[str],
    row: Dict[str, Any],
    context: Dict[str, Any],
    rows: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    if expression is None:
        return None
    expr = expression.strip()
    if not expr:
        return None
    scope: Dict[str, Any] = {
        "row": row,
        "rows": rows or [],
        "context": context,
        "math": math,
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }
    scope.update(row)
    try:
        return _sandbox_eval_expression(expr, scope)
    except DatasetExpressionError as exc:
        logger.warning("Disallowed dataset expression '%s': %s", expression, exc)
    except Exception:
        logger.debug("Failed to evaluate dataset expression '%s'", expression)
    return None


def _apply_filter(rows: List[Dict[str, Any]], condition: Optional[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not rows or not condition:
        return rows
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        value = _evaluate_dataset_expression(condition, row, context, rows)
        if _runtime_truthy(value):
            filtered.append(row)
    return filtered


def _apply_computed_column(
    rows: List[Dict[str, Any]],
    name: str,
    expression: Optional[str],
    context: Dict[str, Any],
) -> None:
    if not name:
        return
    for row in rows:
        row[name] = _evaluate_dataset_expression(expression, row, context, rows)


def _apply_group_aggregate(
    rows: List[Dict[str, Any]],
    columns: Sequence[str],
    aggregates: Sequence[Tuple[str, str]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not columns:
        key = tuple()
        grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {key: rows}
    else:
        grouped = defaultdict(list)
        for row in rows:
            key = tuple(row.get(column) for column in columns)
            grouped[key].append(row)
    results: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        base: Dict[str, Any] = {}
        for idx, column in enumerate(columns):
            base[column] = key[idx] if idx < len(key) else None
        for function, expression in aggregates:
            expr_source, alias = _parse_aggregate_expression(expression)
            result_key = _aggregate_result_key(function or "agg", expr_source, alias)
            values = []
            if expr_source:
                for row in items:
                    values.append(_evaluate_dataset_expression(expr_source, row, context, items))
            numeric_values = [_ensure_numeric(value) for value in values if value is not None]
            func_lower = str(function or "").lower()
            if func_lower == "sum":
                base[result_key] = sum(numeric_values) if numeric_values else 0
            elif func_lower == "count":
                if values:
                    base[result_key] = sum(1 for value in values if _runtime_truthy(value))
                else:
                    base[result_key] = len(items)
            elif func_lower == "avg":
                base[result_key] = sum(numeric_values) / len(numeric_values) if numeric_values else 0
            elif func_lower == "min":
                base[result_key] = min(numeric_values) if numeric_values else None
            elif func_lower == "max":
                base[result_key] = max(numeric_values) if numeric_values else None
            else:
                base[result_key] = values
        results.append(base)
    return results


def _apply_order(rows: List[Dict[str, Any]], columns: Sequence[str]) -> List[Dict[str, Any]]:
    if not columns:
        return rows

    def sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
        values: List[Any] = []
        for column in columns:
            column_name = column.lstrip("-")
            values.append(row.get(column_name))
        return tuple(values)

    descending_flags = [col.startswith("-") for col in columns]
    ordered = sorted(rows, key=sort_key, reverse=descending_flags[0] if descending_flags else False)
    if any(descending_flags[1:]):
        for index in range(1, len(columns)):
            column = columns[index]
            if not column.startswith("-"):
                continue
            column_name = column.lstrip("-")
            ordered = sorted(ordered, key=lambda item: item.get(column_name), reverse=True)
    return ordered


def _apply_window_operation(
    rows: List[Dict[str, Any]],
    name: str,
    function: str,
    target: Optional[str],
) -> None:
    if not name:
        return
    func_lower = (function or "").lower()
    values = [row.get(target) if target else None for row in rows]
    if func_lower == "rank":
        sorted_pairs = sorted(enumerate(values), key=lambda item: item[1])
        ranks = {index: rank + 1 for rank, (index, _) in enumerate(sorted_pairs)}
        for idx, row in enumerate(rows):
            row[name] = ranks.get(idx, idx + 1)
        return
    running_total = 0.0
    for index, row in enumerate(rows):
        value = _ensure_numeric(row.get(target)) if target else 0.0
        if func_lower in {"cumsum", "running_sum", "sum"}:
            running_total += value
            row[name] = running_total
        elif func_lower == "avg":
            running_total += value
            row[name] = running_total / (index + 1)
        else:
            row[name] = value


def _apply_transforms(
    rows: List[Dict[str, Any]],
    transforms: Sequence[Dict[str, Any]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    current = _clone_rows(rows)
    for step in transforms:
        transform_type = str(step.get("type") or "").lower()
        handler = DATASET_TRANSFORMS.get(transform_type)
        if handler is None:
            continue
        options = _resolve_option_dict(step.get("options") if isinstance(step, dict) else {})
        try:
            current = handler(current, options, context)
        except Exception:
            logger.exception("Dataset transform '%s' failed", transform_type)
    return current


def _evaluate_quality_checks(
    rows: List[Dict[str, Any]],
    checks: Sequence[Dict[str, Any]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for check in checks:
        name = check.get("name")
        condition = check.get("condition")
        metric = check.get("metric")
        threshold = check.get("threshold")
        severity = check.get("severity", "error")
        message = check.get("message")
        extras = dict(check.get("extras") or {})
        passed = True
        metric_value = None
        if metric:
            metric_lower = metric.lower()
            if metric_lower == "row_count":
                metric_value = len(rows)
            elif metric_lower == "null_ratio" and threshold:
                target_column = extras.get("column")
                if target_column:
                    nulls = sum(1 for row in rows if row.get(target_column) in {None, ""})
                    metric_value = nulls / max(len(rows), 1)
            else:
                metric_value = sum(_ensure_numeric(row.get(metric)) for row in rows)
        if condition:
            violations = [
                row for row in rows
                if not _runtime_truthy(_evaluate_dataset_expression(condition, row, context, rows))
            ]
            passed = not violations
        elif isinstance(threshold, (int, float)) and metric_value is not None:
            passed = metric_value >= threshold
        results.append(
            {
                "name": name,
                "passed": passed,
                "severity": severity,
                "message": message,
                "metric": metric,
                "value": metric_value,
                "threshold": threshold,
                "extras": extras,
            }
        )
    return results


def _dataset_cache_settings(dataset: Dict[str, Any], context: Dict[str, Any]) -> Tuple[str, bool, Optional[int]]:
    policy = dataset.get("cache_policy") or {}
    if not isinstance(policy, dict):
        policy = {}
    runtime_cache_toggle = _runtime_setting("cache_enabled")
    dataset_runtime_settings = _runtime_setting("datasets")
    dataset_runtime_cache: Optional[bool] = None
    default_scope = context.get("page") or "global"
    default_ttl: Optional[int] = None
    if isinstance(dataset_runtime_settings, dict):
        dataset_runtime_cache = dataset_runtime_settings.get("cache_enabled") if isinstance(dataset_runtime_settings.get("cache_enabled"), bool) else None
        runtime_scope = dataset_runtime_settings.get("default_scope")
        if isinstance(runtime_scope, str) and runtime_scope.strip():
            default_scope = runtime_scope.strip()
        runtime_ttl = dataset_runtime_settings.get("ttl")
        if isinstance(runtime_ttl, int) and runtime_ttl > 0:
            default_ttl = runtime_ttl
    strategy = str(policy.get("strategy") or "auto").lower()
    enabled = strategy not in {"none", "off", "disabled"}
    ttl_value = policy.get("ttl_seconds")
    ttl = ttl_value if isinstance(ttl_value, int) and ttl_value > 0 else default_ttl
    scope = policy.get("scope") or default_scope
    if isinstance(dataset_runtime_cache, bool):
        enabled = dataset_runtime_cache
    if isinstance(runtime_cache_toggle, bool):
        enabled = enabled and runtime_cache_toggle
    return scope, enabled, ttl


def _normalize_identity_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_identity_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_identity_value(item) for item in value]
    if isinstance(value, set):
        return sorted(str(item) for item in value)
    if isinstance(value, tuple):
        return [_normalize_identity_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _dataset_identity_signature(context: Dict[str, Any]) -> Tuple[str, str]:
    vars_snapshot = context.get("vars") if isinstance(context.get("vars"), dict) else {}
    request_ctx = context.get("request") if isinstance(context.get("request"), dict) else {}
    tenant_value = context.get("tenant") or request_ctx.get("tenant") or "global"
    subject_value = context.get("subject") or request_ctx.get("subject")
    scopes_value = request_ctx.get("scopes")
    if isinstance(scopes_value, (list, tuple, set)):
        scopes_list = sorted({str(item).strip() for item in scopes_value if str(item).strip()})
    elif isinstance(scopes_value, str):
        scopes_list = sorted({segment.strip() for segment in scopes_value.replace(",", " ").split() if segment.strip()})
    else:
        scopes_list = []

    identity = {
        "vars": _normalize_identity_value(vars_snapshot),
        "tenant": str(tenant_value).strip() if tenant_value is not None else "global",
        "subject": str(subject_value).strip() if subject_value else None,
        "scopes": scopes_list,
    }
    serialised = json.dumps(identity, sort_keys=True)
    digest = hashlib.sha1(serialised.encode("utf-8")).hexdigest()
    tenant_fragment = identity["tenant"] or "global"
    tenant_fragment = re.sub(r"[^0-9A-Za-z_.-]+", "_", tenant_fragment) or "global"
    return tenant_fragment, digest


def _make_dataset_cache_key(dataset_name: str, scope: str, context: Dict[str, Any]) -> str:
    tenant_fragment, digest = _dataset_identity_signature(context)
    return f"dataset::{scope}::{tenant_fragment}::{dataset_name}::{digest}"


def _coerce_dataset_names(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        names: List[str] = []
        for key, enabled in value.items():
            if enabled:
                names.append(str(key))
        return names
    if isinstance(value, (list, tuple, set)):
        names = [str(item) for item in value if isinstance(item, str) and item]
        return names
    return []


async def _invalidate_dataset(dataset_name: str) -> None:
    if not dataset_name:
        return
    tracked_keys = list(DATASET_CACHE_INDEX.get(dataset_name, set()))
    if tracked_keys:
        for cache_key in tracked_keys:
            try:
                await _cache_delete(cache_key)
            except Exception:
                logger.exception("Failed to delete cache entry '%s' for dataset '%s'", cache_key, dataset_name)
    DATASET_CACHE_INDEX.pop(dataset_name, None)
    STREAM_CONFIG.setdefault(dataset_name, {})["last_invalidate_ts"] = time.time()
    message = {"type": "dataset.invalidate", "dataset": dataset_name}
    await publish_dataset_event(dataset_name, message)
    if tracked_keys:
        for cache_key in tracked_keys:
            targeted = dict(message)
            targeted["recipient"] = cache_key
            await publish_dataset_event(dataset_name, targeted)
    if REALTIME_ENABLED:
        timestamped = _with_timestamp(dict(message))
        subscribers = list(DATASET_SUBSCRIBERS.get(dataset_name, set()))
        if subscribers:
            for slug in subscribers:
                payload = dict(timestamped)
                payload["slug"] = slug
                await BROADCAST.broadcast(slug, payload)
        else:
            await BROADCAST.broadcast(dataset_name, timestamped)


async def _invalidate_datasets(dataset_names: Iterable[str]) -> None:
    unique = []
    seen: Set[str] = set()
    for name in dataset_names:
        if not name or name in seen:
            continue
        if name not in DATASETS:
            continue
        seen.add(name)
        unique.append(name)
    if not unique:
        return
    await asyncio.gather(*[_invalidate_dataset(name) for name in unique])


async def _broadcast_dataset_refresh(
    slug: Optional[str],
    dataset_name: str,
    payload: List[Dict[str, Any]],
    context: Dict[str, Any],
    cache_key: str,
) -> None:
    message = {
        "type": "dataset",
        "slug": slug,
        "dataset": dataset_name,
        "rows": payload,
        "meta": _page_meta(slug) if slug else {},
    }
    message["recipient"] = cache_key
    await publish_dataset_event(dataset_name, message)
    if REALTIME_ENABLED:
        await BROADCAST.broadcast(slug or dataset_name, _with_timestamp(dict(message)))


async def _schedule_dataset_refresh(dataset_name: str, dataset: Dict[str, Any], session: Optional[AsyncSession], context: Dict[str, Any]) -> None:
    policy = dataset.get("refresh_policy") or {}
    if not isinstance(policy, dict):
        return
    interval = policy.get("interval_seconds") or policy.get("interval")
    try:
        interval_value = int(interval)
    except (TypeError, ValueError):
        return
    if interval_value <= 0:
        return
    slug = context.get("page") if isinstance(context.get("page"), str) else None
    if slug:
        DATASET_SUBSCRIBERS[dataset_name].add(slug)
    STREAM_CONFIG.setdefault(dataset_name, {})["refresh_interval"] = interval_value
    context.setdefault("refresh_schedule", {})[dataset_name] = interval_value
    policy_message = {
        "type": "refresh_policy",
        "slug": slug,
        "dataset": dataset_name,
        "interval": interval_value,
    }
    await publish_dataset_event(dataset_name, policy_message)
    if REALTIME_ENABLED and slug:
        await BROADCAST.broadcast(slug, _with_timestamp(dict(policy_message)))

async def _execute_action_operation(
    operation: Dict[str, Any],
    context: Dict[str, Any],
    scope: ScopeFrame,
) -> Optional[Dict[str, Any]]:
    otype = str(operation.get("type") or "").lower()
    result: Optional[Dict[str, Any]] = None
    if otype == "toast":
        message = _render_template_value(operation.get("message"), context)
        result = {"type": "toast", "message": message}
    elif otype == "python_call":
        module = operation.get("module")
        method = operation.get("method")
        arguments_raw = operation.get("arguments") or {}
        arguments = _resolve_placeholders(arguments_raw, context)
        payload = call_python_model(module, method, arguments)
        result = {"type": "python_call", "result": payload}
    elif otype == "connector_call":
        name = operation.get("name")
        arguments_raw = operation.get("arguments") or {}
        arguments = _resolve_placeholders(arguments_raw, context)
        payload = call_llm_connector(name, arguments)
        result = {"type": "connector_call", "result": payload}
    elif otype == "chain_run":
        name = operation.get("name")
        inputs_raw = operation.get("inputs") or {}
        inputs = _resolve_placeholders(inputs_raw, context)
        payload = run_chain(name, inputs)
        result = {"type": "chain_run", "result": payload}
    elif otype == "navigate":
        target_page = operation.get("page_name") or operation.get("page") or operation.get("target")
        resolved = _resolve_page_reference(target_page)
        if resolved is None:
            result = {
                "type": "navigate",
                "status": "not_found",
                "target": target_page,
            }
        else:
            result = {
                "type": "navigate",
                "status": "ok",
                "page": resolved.get("name"),
                "page_slug": resolved.get("slug"),
                "page_route": resolved.get("route"),
            }
    elif otype == "update":
        table = operation.get("table")
        set_expression = operation.get("set_expression")
        where_expression = operation.get("where_expression")
        session: Optional[AsyncSession] = context.get("session")
        if session is None:
            result = {"type": "update", "status": "no_session"}
        else:
            updated = await _execute_update(table, set_expression, where_expression, session, context)
            result = {"type": "update", "status": "ok", "rows": updated}
    else:
        return None

    if result is not None:
        await _handle_post_action_effects(operation, context, result, otype)
    return result


async def _execute_update(
    table: Optional[str],
    set_expression: Optional[str],
    where_expression: Optional[str],
    session: AsyncSession,
    context: Optional[Dict[str, Any]],
) -> int:
    if not table or not set_expression:
        return 0
    ctx = context or {}
    try:
        sanitized_table = _sanitize_table_reference(table)
        assignments, assignment_params = _parse_update_assignments(set_expression, ctx)
        conditions, where_params = _parse_where_expression(where_expression, ctx)
    except ValueError:
        logger.warning("Rejected unsafe update targeting table '%s'", table)
        return 0
    if not assignments:
        return 0
    parameters: Dict[str, Any] = {}
    parameters.update(assignment_params)
    parameters.update(where_params)
    try:
        if _HAS_SQLA_UPDATE and update is not None and sql_table is not None and column is not None and bindparam is not None:  # type: ignore[truthy-function]
            statement = _build_update_statement(sanitized_table, assignments, conditions)
            if statement is None:
                return 0
            result = await session.execute(statement, parameters)
        else:
            query_text = _build_fallback_update_sql(sanitized_table, assignments, conditions)
            result = await session.execute(text(query_text), parameters)
        await session.commit()
        rowcount = getattr(result, "rowcount", 0) or 0
        return rowcount
    except Exception:
        await session.rollback()
        logger.exception("Failed to execute update on table '%s'", sanitized_table)
        return 0


def _build_update_statement(
    table_name: str,
    assignments: Sequence[Tuple[str, str, Any]],
    conditions: Sequence[Tuple[str, str, Any]],
) -> Optional[Any]:
    table_clause = _create_table_clause(table_name, assignments, conditions)
    if table_clause is None:
        return None
    stmt = update(table_clause)  # type: ignore[arg-type]
    values_mapping: Dict[str, Any] = {}
    for column_name, param_name, _ in assignments:
        values_mapping[column_name] = bindparam(param_name)  # type: ignore[arg-type]
    if values_mapping:
        stmt = stmt.values(**values_mapping)
    condition_expr: Optional[Any] = None
    for column_name, param_name, _ in conditions:
        clause = table_clause.c[column_name] == bindparam(param_name)  # type: ignore[index,operator]
        condition_expr = clause if condition_expr is None else condition_expr & clause
    if condition_expr is not None:
        stmt = stmt.where(condition_expr)
    return stmt


def _build_fallback_update_sql(
    table_name: str,
    assignments: Sequence[Tuple[str, str, Any]],
    conditions: Sequence[Tuple[str, str, Any]],
) -> str:
    set_clause = ", ".join(f"{column} = :{param}" for column, param, _ in assignments)
    if conditions:
        where_clause = " AND ".join(f"{column} = :{param}" for column, param, _ in conditions)
        return f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
    return f"UPDATE {table_name} SET {set_clause}"


def _create_table_clause(
    table_name: str,
    assignments: Sequence[Tuple[str, str, Any]],
    conditions: Sequence[Tuple[str, str, Any]],
) -> Optional[Any]:
    if sql_table is None or column is None:
        return None
    segments = table_name.split(".")
    name = segments[-1]
    schema = ".".join(segments[:-1]) if len(segments) > 1 else None
    seen: Dict[str, None] = {}
    column_order: List[str] = []
    for column_name, _, _ in list(assignments) + list(conditions):
        if column_name not in seen:
            seen[column_name] = None
            column_order.append(column_name)
    columns = [column(column_name) for column_name in column_order]  # type: ignore[arg-type]
    table_clause = sql_table(name, *columns, schema=schema)  # type: ignore[arg-type]
    return table_clause


def _sanitize_table_reference(value: str) -> str:
    return _sanitize_identifier_path(value)


def _sanitize_identifier_path(raw: str) -> str:
    segments = [segment.strip() for segment in (raw or "").split(".") if segment and segment.strip()]
    if not segments:
        raise ValueError("Empty identifier path")
    cleaned: List[str] = []
    for segment in segments:
        if not _IDENTIFIER_RE.match(segment):
            raise ValueError(f"Invalid identifier segment '{segment}'")
        cleaned.append(segment)
    return ".".join(cleaned)


def _parse_update_assignments(expression: str, context: Dict[str, Any]) -> Tuple[List[Tuple[str, str, Any]], Dict[str, Any]]:
    assignments = _split_assignment_list(expression)
    if not assignments:
        raise ValueError("No assignments provided")
    entries: List[Tuple[str, str, Any]] = []
    params: Dict[str, Any] = {}
    for index, assignment in enumerate(assignments):
        match = _UPDATE_ASSIGNMENT_PATTERN.match(assignment)
        if not match:
            raise ValueError(f"Unsupported assignment '{assignment}'")
        column = _sanitize_identifier_path(match.group(1))
        value_expr = match.group(2).strip()
        value = _evaluate_update_value(value_expr, context)
        param_name = f"set_{index}"
        entries.append((column, param_name, value))
        params[param_name] = value
    return entries, params


def _parse_where_expression(expression: Optional[str], context: Dict[str, Any]) -> Tuple[List[Tuple[str, str, Any]], Dict[str, Any]]:
    if not expression:
        return [], {}
    conditions = _split_conditions(str(expression))
    if not conditions:
        raise ValueError("Unable to parse WHERE expression")
    entries: List[Tuple[str, str, Any]] = []
    params: Dict[str, Any] = {}
    for index, condition in enumerate(conditions):
        match = _WHERE_CONDITION_PATTERN.match(condition)
        if not match:
            raise ValueError(f"Unsupported WHERE clause '{condition}'")
        column = _sanitize_identifier_path(match.group(1))
        value_expr = match.group(2).strip()
        value = _evaluate_update_value(value_expr, context)
        param_name = f"where_{index}"
        entries.append((column, param_name, value))
        params[param_name] = value
    return entries, params


def _split_assignment_list(expression: str) -> List[str]:
    if not expression:
        return []
    parts: List[str] = []
    current: List[str] = []
    quote: Optional[str] = None
    escape = False
    depth = 0
    for char in expression:
        if escape:
            current.append(char)
            escape = False
            continue
        if quote:
            current.append(char)
            if char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
            current.append(char)
            continue
        if char in ")]}":
            if depth:
                depth -= 1
            current.append(char)
            continue
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _split_conditions(expression: str) -> List[str]:
    expr = expression.strip()
    if not expr:
        return []
    parts: List[str] = []
    current: List[str] = []
    quote: Optional[str] = None
    escape = False
    depth = 0
    i = 0
    length = len(expr)
    while i < length:
        char = expr[i]
        if escape:
            current.append(char)
            escape = False
            i += 1
            continue
        if quote:
            current.append(char)
            if char == "\\":
                escape = True
            elif char == quote:
                quote = None
            i += 1
            continue
        if char in {'"', "'"}:
            quote = char
            current.append(char)
            i += 1
            continue
        if char in "([{":
            depth += 1
            current.append(char)
            i += 1
            continue
        if char in ")]}":
            if depth:
                depth -= 1
            current.append(char)
            i += 1
            continue
        if depth == 0 and expr[i:].lower().startswith("and") and (i == 0 or expr[i - 1].isspace()) and (i + 3 >= length or expr[i + 3].isspace() or expr[i + 3] in ")]}"):
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            i += 3
            while i < length and expr[i].isspace():
                i += 1
            continue
        current.append(char)
        i += 1
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _evaluate_update_value(expression: str, context: Dict[str, Any]) -> Any:
    value_expr = expression.strip()
    if not value_expr:
        return None
    lowered = value_expr.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(value_expr)
    except Exception:
        pass
    result = _evaluate_dataset_expression(value_expr, {}, context, [])
    if result is None and lowered not in {"null", "none"}:
        try:
            if "." in value_expr:
                return float(value_expr)
            return int(value_expr)
        except Exception:
            return value_expr
    return result


async def _handle_post_action_effects(
    operation: Dict[str, Any],
    context: Dict[str, Any],
    result: Dict[str, Any],
    operation_type: str,
) -> None:
    if not result:
        return
    status_value = result.get("status")
    if isinstance(status_value, str) and status_value.lower() in {"error", "failed", "no_session"}:
        return
    dataset_targets = _coerce_dataset_names(
        operation.get("invalidate_datasets")
        or operation.get("refresh_datasets")
        or operation.get("datasets")
    )
    if dataset_targets:
        await _invalidate_datasets(dataset_targets)
    event_topic = operation.get("publish_event") or operation.get("emit_event")
    if event_topic:
        event_payload = _resolve_placeholders(operation.get("event_payload"), context)
        message: Dict[str, Any] = {
            "type": "action",
            "operation": operation_type,
            "result": result,
            "datasets": dataset_targets,
        }
        if event_payload is not None:
            message["payload"] = event_payload
        await publish_event(str(event_topic), message)


def _resolve_page_reference(target: Optional[str]) -> Optional[Dict[str, Any]]:
    if not target:
        return None
    lowered = str(target).strip().lower()
    if not lowered:
        return None
    for page in PAGES:
        name = str(page.get("name") or "").strip()
        slug = str(page.get("slug") or "").strip()
        route = str(page.get("route") or "").strip()
        candidates = {
            name.lower(),
            slug.lower(),
            route.lower(),
        }
        if lowered in candidates:
            return {
                "name": name or page.get("name"),
                "slug": slug or page.get("slug"),
                "route": route or page.get("route"),
            }
    return None


def _get_component_spec(slug: str, component_index: int) -> Dict[str, Any]:
    page_spec = PAGE_SPEC_BY_SLUG.get(slug)
    if not page_spec:
        raise KeyError(f"Unknown page '{slug}'")
    components = page_spec.get("components") or []
    if component_index < 0 or component_index >= len(components):
        raise IndexError(f"Component index {component_index} out of range for '{slug}'")
    component = components[component_index]
    if not isinstance(component, dict):
        raise ValueError("Invalid component payload")
    return component


async def _execute_component_interaction(
    slug: str,
    component_index: int,
    payload: Optional[Dict[str, Any]],
    session: Optional[AsyncSession],
    *,
    kind: str,
) -> Dict[str, Any]:
    component = _get_component_spec(slug, component_index)
    component_type = str(component.get("type") or "").lower()
    if component_type != kind.lower():
        raise ValueError(f"Component {component_index} on '{slug}' is not of type '{kind}'")

    submitted: Dict[str, Any] = dict(payload or {})
    context = build_context(slug)
    if session is not None:
        context["session"] = session
    context.setdefault("vars", {})
    for key, value in submitted.items():
        if isinstance(key, str):
            context["vars"].setdefault(key, value)
    context.setdefault("payload", {}).update(submitted)
    if kind == "form":
        context.setdefault("form", {}).update(submitted)

    scope = ScopeFrame()
    scope.set("context", context)
    scope.bind("payload", submitted)
    if kind == "form":
        scope.bind("form", submitted)

    operations = component.get("operations") or []
    results: List[Dict[str, Any]] = []
    for operation in operations:
        outcome = await _execute_action_operation(operation, context, scope)
        if outcome:
            results.append(outcome)

    response: Dict[str, Any] = {
        "status": "ok",
        "slug": slug,
        "component_index": component_index,
        "type": kind,
        "results": results,
    }
    if kind == "form":
        response["accepted"] = submitted
    if results:
        response["effects"] = results
    return response


async def submit_form(
    slug: str,
    component_index: int,
    payload: Optional[Dict[str, Any]],
    *,
    session: Optional[AsyncSession] = None,
) -> Dict[str, Any]:
    return await _execute_component_interaction(
        slug,
        component_index,
        payload,
        session,
        kind="form",
    )


async def trigger_action(
    slug: str,
    component_index: int,
    payload: Optional[Dict[str, Any]],
    *,
    session: Optional[AsyncSession] = None,
) -> Dict[str, Any]:
    return await _execute_component_interaction(
        slug,
        component_index,
        payload,
        session,
        kind="action",
    )

async def render_statements(
    statements: Optional[Iterable[Dict[str, Any]]],
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    return await _render_block(statements, context, scope, session, allow_control=False)


async def _render_block(
    statements: Optional[Iterable[Dict[str, Any]]],
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
    *,
    allow_control: bool,
) -> List[Dict[str, Any]]:
    components: List[Dict[str, Any]] = []
    if not statements:
        return components
    for statement in statements:
        try:
            result = await _render_statement(statement, context, scope, session)
        except (BreakFlow, ContinueFlow) as exc:
            if components:
                exc.extend(components)
            if allow_control:
                raise exc


            logger.warning("Loop control statement encountered outside of a loop")
            break
        if result:
            components.extend(result)
    return components


async def _render_statement(
    statement: Dict[str, Any],


    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    stype = statement.get("type")
    if stype == "variable":
        name = statement.get("name")
        expr = statement.get("value_expr")
        if expr is not None:
            value = await _evaluate_runtime_expression(expr, context, scope)


        else:
            value = _resolve_placeholders(statement.get("value"), context)
        if name:
            _assign_variable(scope, context, name, value)
        return []
    if stype == "if":
        return await _render_if(statement, context, scope, session)
    if stype == "for_loop":
        return await _render_for_loop(statement, context, scope, session)
    if stype == "while_loop":
        return await _render_while_loop(statement, context, scope, session)
    if stype == "break":
        raise BreakFlow()
    if stype == "continue":
        raise ContinueFlow()

    payload = copy.deepcopy(statement)
    payload_type = payload.pop("type", None)
    if payload_type == "text":
        rendered_text = _render_template_value(payload.get("text"), context)
        styles = copy.deepcopy(payload.get("styles", {}))
        return [{"type": "text", "text": rendered_text, "styles": styles}]
    if payload_type == "table":
        resolved = _resolve_placeholders(payload, context)
        if "title" in resolved:
            resolved["title"] = _render_template_value(resolved.get("title"), context)
        if "filter" in resolved:
            resolved["filter"] = _render_template_value(resolved.get("filter"), context)
        if "sort" in resolved:
            resolved["sort"] = _render_template_value(resolved.get("sort"), context)
        return [{"type": "table", **resolved}]
    if payload_type == "chart":
        resolved = _resolve_placeholders(payload, context)
        if "heading" in resolved:
            resolved["heading"] = _render_template_value(resolved.get("heading"), context)
        if "title" in resolved:
            resolved["title"] = _render_template_value(resolved.get("title"), context)
        return [{"type": "chart", **resolved}]
    if payload_type == "form":
        resolved = _resolve_placeholders(payload, context)
        if "title" in resolved:


            resolved["title"] = _render_template_value(resolved.get("title"), context)
        return [{"type": "form", **resolved}]
    if payload_type == "action":
        resolved = _resolve_placeholders(payload, context)
        if "name" in resolved:
            resolved["name"] = _render_template_value(resolved.get("name"), context)
        if "trigger" in resolved:
            resolved["trigger"] = _render_template_value(resolved.get("trigger"), context)
        operations = resolved.get("operations") or []
        results: List[Dict[str, Any]] = []
        for operation in operations:
            outcome = await _execute_action_operation(operation, context, scope)
            if outcome:
                results.append(outcome)
        if results:
            return results
        return [{"type": "action", **resolved}]
    if payload_type == "predict":
        resolved = _resolve_placeholders(payload, context)
        return [{"type": "predict", **resolved}]
    if payload_type:
        resolved = _resolve_placeholders(payload, context)
        return [{"type": payload_type, **resolved}]
    return []


async def _render_if(
    statement: Dict[str, Any],
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    condition = statement.get("condition")


    if _runtime_truthy(await _evaluate_runtime_expression(condition, context, scope)):
        return await _render_block(statement.get("body"), context, scope, session, allow_control=True)
    for branch in statement.get("elifs", []) or []:
        branch_condition = branch.get("condition")
        if _runtime_truthy(await _evaluate_runtime_expression(branch_condition, context, scope)):
            return await _render_block(branch.get("body"), context, scope, session, allow_control=True)
    else_body = statement.get("else_body")
    if else_body:
        return await _render_block(else_body, context, scope, session, allow_control=True)
    return []


async def _render_for_loop(
    statement: Dict[str, Any],
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    loop_var = statement.get("loop_var")
    source_kind = statement.get("source_kind")
    source_name = statement.get("source_name")
    if not loop_var or not source_kind or not source_name:
        return []
    items = await _resolve_loop_iterable(source_kind, source_name, context, scope, session)
    if not items:
        return []
    components: List[Dict[str, Any]] = []


    vars_map = context.setdefault("vars", {})
    for item in items:
        previous = vars_map.get(loop_var, _MISSING)
        loop_scope = scope.child()
        _bind_variable(loop_scope, context, loop_var, item)
        try:
            rendered = await _render_block(statement.get("body"), context, loop_scope, session, allow_control=True)
            if rendered:
                components.extend(rendered)
        except ContinueFlow as exc:
            if exc.components:
                components.extend(exc.components)
            continue
        except BreakFlow as exc:
            if exc.components:
                components.extend(exc.components)
            break


        finally:
            _restore_variable(context, loop_var, previous)
    return components


async def _render_while_loop(
    statement: Dict[str, Any],
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    condition = statement.get("condition")
    components: List[Dict[str, Any]] = []
    loop_scope = scope.child()
    iterations = 0
    while _runtime_truthy(await _evaluate_runtime_expression(condition, context, loop_scope)):
        iterations += 1
        if iterations > 1000:
            logger.warning("Aborting while loop after 1000 iterations for safety")
            break
        try:
            rendered = await _render_block(statement.get("body"), context, loop_scope, session, allow_control=True)
            if rendered:
                components.extend(rendered)
        except ContinueFlow as exc:
            if exc.components:
                components.extend(exc.components)
            continue
        except BreakFlow as exc:
            if exc.components:
                components.extend(exc.components)
            break
    return components


async def _resolve_loop_iterable(
    source_kind: str,
    source_name: str,
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Any]:
    if source_kind == "dataset":
        if session is not None:
            try:
                return await fetch_dataset_rows(source_name, session, context)
            except Exception:  # pragma: no cover - runtime fetch failure
                logger.exception("Failed to fetch dataset rows for %s", source_name)
        dataset_spec = DATASETS.get(source_name)
        if dataset_spec:
            return list(dataset_spec.get("sample_rows", []))


        return []
    if source_kind == "table":
        tables = context.get("tables")
        if isinstance(tables, dict) and source_name in tables:
            table_value = tables[source_name]
            if isinstance(table_value, list):
                return table_value
        dataset_spec = DATASETS.get(source_name)
        if dataset_spec:
            return list(dataset_spec.get("sample_rows", []))
        return []
    items = scope.get(source_name)
    if isinstance(items, list):
        return items
    return []


async def _evaluate_runtime_expression(
    expression: Optional[Dict[str, Any]],
    context: Dict[str, Any],
    scope: ScopeFrame,
) -> Any:
    if expression is None:
        return None
    etype = expression.get("type")
    if etype == "literal":
        return _resolve_placeholders(expression.get("value"), context)
    if etype == "name":
        name = expression.get("name")
        if not isinstance(name, str):
            return None
        value = scope.get(name, _MISSING)
        if value is not _MISSING:
            return value
        vars_map = context.get("vars", {})
        if isinstance(vars_map, dict) and name in vars_map:
            return vars_map[name]
        if name in _RUNTIME_CALLABLES:
            return _RUNTIME_CALLABLES[name]
        return context.get(name)
    if etype == "attribute":
        path = list(expression.get("path") or [])
        if not path:
            return None
        base_name = path[0]
        base_value = await _evaluate_runtime_expression({"type": "name", "name": base_name}, context, scope)
        if base_value is None:
            return None
        return _traverse_attribute_path(base_value, path[1:])
    if etype == "context":
        return _resolve_context_scope(
            expression.get("scope"),
            expression.get("path", []),
            context,
            expression.get("default"),
        )
    if etype == "binary":
        op = expression.get("op")
        left = await _evaluate_runtime_expression(expression.get("left"), context, scope)
        right = await _evaluate_runtime_expression(expression.get("right"), context, scope)
        try:
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right if right not in (0, None) else None
            if op == "and":
                return _runtime_truthy(left) and _runtime_truthy(right)
            if op == "or":
                return _runtime_truthy(left) or _runtime_truthy(right)
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == "in":
                return left in right if right is not None else False
            if op == "not in":
                return left not in right if right is not None else True
        except Exception:  # pragma: no cover - operator failure
            logger.exception("Failed to evaluate binary operation '%s'", op)
            return None
        return None
    if etype == "unary":
        operand = await _evaluate_runtime_expression(expression.get("operand"), context, scope)
        op = expression.get("op")
        try:
            if op == "not":
                return not _runtime_truthy(operand)
            if op == "-":
                return -operand
            if op == "+":
                return +operand
        except Exception:  # pragma: no cover - unary failure
            logger.exception("Failed to evaluate unary operation '%s'", op)
            return None
        return None
    if etype == "call":
        function_expr = expression.get("function")
        func = await _evaluate_runtime_expression(function_expr, context, scope)
        if func is None:
            return None
        arguments = []
        for arg in expression.get("arguments", []) or []:
            arguments.append(await _evaluate_runtime_expression(arg, context, scope))
        try:
            if inspect.iscoroutinefunction(func):
                return await func(*arguments)
            result = func(*arguments) if callable(func) else None
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception:  # pragma: no cover - call failure
            logger.exception("Failed to execute call expression")
            return None
    return None


def _traverse_attribute_path(value: Any, path: Iterable[str]) -> Any:
    current = value
    for segment in path:
        if current is None:
            return None
        current = _resolve_path_segment(current, segment)
    return current


def _resolve_path_segment(value: Any, segment: str) -> Any:
    if isinstance(value, dict):
        return value.get(segment)
    if isinstance(value, (list, tuple)):
        try:
            index = int(segment)
        except (TypeError, ValueError):
            return None
        if 0 <= index < len(value):
            return value[index]
        return None
    if hasattr(value, segment):
        return getattr(value, segment)
    return None

def _page_meta(slug: str) -> Dict[str, Any]:
    spec = PAGE_SPEC_BY_SLUG.get(slug, {})
    return {
        "reactive": bool(spec.get("reactive")),
        "refresh_policy": spec.get("refresh_policy"),
    }


def _model_to_payload(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    if isinstance(model, dict):
        return dict(model)
    if hasattr(model, "__dict__"):
        return {
            key: value
            for key, value in model.__dict__.items()
            if not key.startswith("_")
        }
    return {"value": model}


def _with_timestamp(payload: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(payload)
    enriched.setdefault("ts", time.time())
    return enriched


def register_model_loader(framework: str, loader: Callable[[str, Dict[str, Any]], Any]) -> None:
    if not framework or loader is None:
        return
    MODEL_LOADERS[framework.lower()] = loader


def register_model_runner(
    framework: str,
    runner: Callable[[str, Any, Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
) -> None:
    if not framework or runner is None:
        return
    MODEL_RUNNERS[framework.lower()] = runner


def register_model_explainer(
    framework: str,
    explainer: Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
) -> None:
    if not framework or explainer is None:
        return
    MODEL_EXPLAINERS[framework.lower()] = explainer


def _resolve_model_artifact_path(model_spec: Dict[str, Any]) -> Optional[str]:
    metadata_obj = model_spec.get("metadata")
    metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
    path = metadata.get("model_file") or metadata.get("artifact_path")
    if not path:
        return None
    root = os.getenv("NAMEL3SS_MODEL_ROOT")
    if root and not os.path.isabs(path):
        return os.path.join(root, path)
    return path


def _load_python_callable(import_path: str) -> Optional[Callable[..., Any]]:
    module_path, _, attr = import_path.rpartition(":")
    if not module_path or not attr:
        return None
    module = importlib.import_module(module_path)
    return getattr(module, attr, None)


def _import_python_module(module: str) -> Optional[Any]:
    if not module:
        return None
    try:
        if module.endswith(".py"):
            path = Path(module)
            if not path.is_absolute():
                base = os.getenv("NAMEL3SS_APP_ROOT")
                if base:
                    path = Path(base) / path
                else:
                    path = Path.cwd() / path
            if not path.exists():
                return None
            module_name = path.stem
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec and spec.loader:
                module_obj = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module_obj)  # type: ignore[attr-defined]
                    sys.modules.setdefault(module_name, module_obj)
                    return module_obj
                except Exception:  # pragma: no cover - user module failure
                    logger.exception("Failed to import python module from %s", path)
                    return None
            return None
        return importlib.import_module(module)
    except Exception:  # pragma: no cover - import failure
        logger.exception("Failed to import python module %s", module)
        return None


def _is_truthy_env(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _trim_traceback(limit: int = 5, max_chars: int = 3000) -> str:
    import traceback

    try:
        formatted = traceback.format_exc(limit=limit)
    except Exception:  # pragma: no cover - safety guard
        return ""
    if not formatted:
        return ""
    text = formatted.strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _short_error(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def call_python_model(
    module: str,
    method: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Invoke a Python callable and return structured status details.

    Args:
        module: Fully qualified module path or file path that can be imported.
        method: Attribute name to resolve on the imported module; defaults to ``predict`` when empty.
        arguments: Keyword arguments passed to the callable when invoked.

    Returns:
        A dictionary containing ``status`` alongside contextual fields:

        * ``status = 'ok'`` includes the callable result.
        * ``status = 'error'`` reports structured failure details and a trimmed traceback when stubs are disabled.
        * ``status = 'stub'`` mirrors legacy stub behaviour when ``NAMEL3SS_ALLOW_STUBS`` is truthy.
    """

    args = dict(arguments or {})
    attr_name = method or "predict"
    allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")

    try:
        module_obj = _import_python_module(module)
        if module_obj is None:
            raise ImportError(f"Module '{module}' could not be imported")

        callable_obj = getattr(module_obj, attr_name)
        if not callable(callable_obj):
            raise TypeError(f"Attribute '{attr_name}' on module '{module}' is not callable")

        result = callable_obj(**args)
        return {
            "status": "ok",
            "result": result,
            "inputs": args,
            "module": module,
            "method": attr_name,
        }
    except Exception as exc:  # pragma: no cover - user callable failure
        logger.exception("Python callable %s.%s raised an error", module, attr_name)
        error_message = _short_error(exc)
        if allow_stubs:
            return {
                "status": "stub",
                "result": "stub_prediction",
                "inputs": args,
                "module": module,
                "method": attr_name,
                "error": error_message,
            }

        response = {
            "status": "error",
            "inputs": args,
            "module": module,
            "method": attr_name,
            "error": error_message,
        }

        traceback_text = _trim_traceback()
        if traceback_text:
            response["traceback"] = traceback_text

        return response


def _ensure_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _as_path_segments(path: Any) -> List[str]:
    if path is None:
        return []
    if isinstance(path, (list, tuple)):
        segments: List[str] = []
        for item in path:
            segments.extend(_as_path_segments(item))
        return segments
    text = str(path).strip()
    if not text:
        return []
    normalized = text.replace("[", ".").replace("]", ".")
    return [segment for segment in normalized.split(".") if segment]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        multiplier = 1.0
        if text.endswith("%"):
            multiplier = 0.01
            text = text[:-1]
        try:
            return float(text) * multiplier
        except ValueError:
            return None
    return None


def _aggregate_numeric_samples(samples: List[float], aggregation: str) -> Optional[float]:
    if not samples:
        return None
    key = aggregation.lower()
    if key in ("avg", "average", "mean"):
        return sum(samples) / len(samples)
    if key in ("sum", "total"):
        return sum(samples)
    if key in ("min", "minimum"):
        return min(samples)
    if key in ("max", "maximum"):
        return max(samples)
    if key == "median":
        ordered = sorted(samples)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[midpoint]
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2
    if key in ("p95", "percentile95", "p_95"):
        ordered = sorted(samples)
        index = int(round(0.95 * (len(ordered) - 1)))
        index = max(0, min(index, len(ordered) - 1))
        return ordered[index]
    if key in ("p90", "percentile90", "p_90"):
        ordered = sorted(samples)
        index = int(round(0.90 * (len(ordered) - 1)))
        index = max(0, min(index, len(ordered) - 1))
        return ordered[index]
    if key in ("last", "latest"):
        return samples[-1]
    if key == "first":
        return samples[0]
    return sum(samples) / len(samples)


def _collect_metric_samples(
    metric: Dict[str, Any],
    variants: List[Dict[str, Any]],
    args: Dict[str, Any],
) -> List[float]:
    metadata = _ensure_dict(metric.get("metadata"))
    source_kind = str(metric.get("source_kind") or metadata.get("source") or "score").lower()
    path = metadata.get("path") or metadata.get("result_path") or metadata.get("field")
    segments = _as_path_segments(path)
    samples: List[float] = []

    if source_kind in ("score", "variant", "variant_score"):
        for variant in variants:
            value: Any = variant.get("score")
            if value is None and segments:
                value = _traverse_attribute_path(variant, segments)
            sample = _safe_float(value)
            if sample is not None:
                samples.append(sample)
    elif source_kind in ("result", "output", "prediction"):
        target_segments = segments or ["result", "output", "score"]
        for variant in variants:
            value = _traverse_attribute_path(variant, target_segments)
            sample = _safe_float(value)
            if sample is not None:
                samples.append(sample)
    elif source_kind in ("payload", "input", "request"):
        value = _traverse_attribute_path(args, segments) if segments else args.get("input")
        sample = _safe_float(value)
        if sample is not None:
            samples.append(sample)
    elif source_kind in ("manual", "provided", "static"):
        values = metadata.get("samples") or metadata.get("values") or []
        if isinstance(values, (list, tuple)):
            for entry in values:
                sample = _safe_float(entry)
                if sample is not None:
                    samples.append(sample)
    else:
        target_segments = segments or ["result", "output", "score"]
        for variant in variants:
            value = _traverse_attribute_path(variant, target_segments)
            sample = _safe_float(value)
            if sample is not None:
                samples.append(sample)

    extras = metadata.get("include")
    if isinstance(extras, (list, tuple)):
        for entry in extras:
            sample = _safe_float(entry)
            if sample is not None:
                samples.append(sample)

    return samples


def _evaluate_experiment_metrics(
    spec: Dict[str, Any],
    variants: List[Dict[str, Any]],
    args: Dict[str, Any],
) -> List[Dict[str, Any]]:
    metrics_result: List[Dict[str, Any]] = []
    for index, metric in enumerate(spec.get("metrics", []), start=1):
        metadata = _ensure_dict(metric.get("metadata"))
        aggregation = str(metadata.get("aggregation") or metadata.get("aggregate") or "max")
        samples = _collect_metric_samples(metric, variants, args)
        aggregated_value = _aggregate_numeric_samples(samples, aggregation) if samples else None
        if aggregated_value is None:
            calculated = float(round(0.72 + 0.045 * index, 3))
        else:
            calculated = float(aggregated_value)
        precision = metadata.get("precision") if isinstance(metadata.get("precision"), int) else metadata.get("round")
        if isinstance(precision, int):
            rounded_value = round(calculated, max(int(precision), 0))
        else:
            rounded_value = round(calculated, 4)

        goal_value = _safe_float(metric.get("goal"))
        direction = str(metadata.get("goal_operator") or metadata.get("direction") or "gte").lower()
        achieved_goal: Optional[bool] = None
        if goal_value is not None:
            if direction in ("lte", "le", "max", "lower", "down"):
                achieved_goal = rounded_value <= goal_value
            else:
                achieved_goal = rounded_value >= goal_value

        if samples:
            metadata.setdefault("aggregation", aggregation)
            metadata.setdefault("samples", len(samples))
            metadata.setdefault(
                "summary",
                {
                    "min": min(samples),
                    "max": max(samples),
                    "mean": round(sum(samples) / len(samples), 6),
                },
            )

        metrics_result.append(
            {
                "name": metric.get("name"),
                "value": rounded_value,
                "goal": metric.get("goal"),
                "source_kind": metric.get("source_kind"),
                "source_name": metric.get("source_name"),
                "metadata": metadata,
                "achieved_goal": achieved_goal,
            }
        )

    return metrics_result

import json
import os
import time
from typing import Any, Dict, List, Optional


def _stringify_prompt_value(name: str, value: Any) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, separators=(",", ":"))
        except Exception:
            return str(value)
    if value is None:
        return ""
    return str(value).strip()


def _format_error_message(name: str, prompt_text: Optional[str], reason: str) -> str:
    """Compose a concise diagnostic string for logging failures."""

    title = f"llm:{name}" if name else "llm:unknown"
    cause = reason.strip() or "unspecified error"
    return f"{title} failed: {cause}"


def _default_llm_endpoint(provider: str, config: Dict[str, Any]) -> Optional[str]:
    endpoint = config.get("endpoint") or config.get("url")
    if endpoint:
        return str(endpoint)
    provider_key = provider.lower()
    base_url = str(config.get("api_base") or config.get("base_url") or "")
    if provider_key == "openai":
        api_base = base_url or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
        deployment = config.get("deployment")
        if deployment:
            version = str(config.get("api_version") or os.getenv("OPENAI_API_VERSION") or "2024-02-01")
            return f"{api_base.rstrip('/')}/deployments/{deployment}/chat/completions?api-version={version}"
        return f"{api_base.rstrip('/')}/chat/completions"
    if provider_key == "anthropic":
        api_base = base_url or "https://api.anthropic.com/v1"
        return f"{api_base.rstrip('/')}/messages"
    return None


def _build_llm_request(
    provider: str,
    model_name: str,
    prompt_text: str,
    config: Dict[str, Any],
    args: Dict[str, Any],
) -> Dict[str, Any]:
    if not prompt_text:
        raise ValueError("Prompt text is required for LLM requests")

    provider_key = provider.lower().strip()
    method = str(config.get("method") or "post").upper()
    timeout_value = config.get("timeout", 30.0)
    try:
        timeout = max(float(timeout_value), 1.0)
    except Exception:
        timeout = 30.0

    headers = _ensure_dict(config.get("headers"))
    params = _ensure_dict(config.get("params"))
    payload = config.get("payload")
    body = dict(payload) if isinstance(payload, dict) else {}

    mode = str(config.get("mode") or ("chat" if provider_key in {"openai", "anthropic"} else "completion")).lower()
    if mode == "chat":
        messages: List[Dict[str, Any]] = []
        system_prompt = config.get("system") or config.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        extra_messages = config.get("messages")
        if isinstance(extra_messages, list):
            for message in extra_messages:
                if isinstance(message, dict):
                    messages.append(dict(message))
        user_role = str(config.get("user_role") or "user")
        messages.append({"role": user_role, "content": prompt_text})
        body.setdefault("messages", messages)
        body.setdefault("model", model_name)
    else:
        prompt_field = str(config.get("prompt_field") or "prompt")
        body.setdefault("model", model_name)
        body.setdefault(prompt_field, prompt_text)

    payload_from_args = config.get("payload_from_args")
    if isinstance(payload_from_args, (list, tuple)):
        for key in payload_from_args:
            key_str = str(key)
            if key_str in args and key_str not in body:
                body[key_str] = args[key_str]

    endpoint = _default_llm_endpoint(provider_key, config)
    if not endpoint:
        raise ValueError(f"Endpoint is not configured for provider '{provider}'")

    api_key = config.get("api_key")
    if not api_key:
        env_key = config.get("api_key_env")
        if isinstance(env_key, str):
            api_key = os.getenv(env_key) or api_key

    if provider_key == "openai":
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is missing")
        bearer = str(api_key)
        if not bearer.lower().startswith("bearer "):
            bearer = f"Bearer {bearer}"
        headers.setdefault("Authorization", bearer)
        headers.setdefault("Content-Type", "application/json")
    elif provider_key == "anthropic":
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key is missing")
        headers.setdefault("x-api-key", str(api_key))
        headers.setdefault("content-type", "application/json")
        headers.setdefault("anthropic-version", str(config.get("api_version") or "2023-06-01"))
    elif api_key:
        header_name = str(config.get("api_key_header") or "Authorization")
        if header_name.lower() == "authorization" and not str(api_key).lower().startswith("bearer "):
            headers.setdefault("Authorization", f"Bearer {api_key}")
        else:
            headers.setdefault(header_name, str(api_key))

    normalized_headers: Dict[str, str] = {
        str(key): str(value)
        for key, value in headers.items()
    }
    normalized_params: Dict[str, Any] = {
        str(key): value
        for key, value in params.items()
    }

    normalized_headers.setdefault("Content-Type", "application/json")

    return {
        "method": method,
        "url": str(endpoint),
        "headers": normalized_headers,
        "params": normalized_params,
        "body": body,
        "timeout": timeout,
        "provider": provider,
    }


def _is_truthy_env(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _redact_config(config: Dict[str, Any]) -> Dict[str, Any]:
    sensitive_keys = {
        "api_key",
        "authorization",
        "token",
        "secret",
        "password",
        "x-api-key",
        "api-key",
    }

    def _sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): ("***" if str(key).strip().lower() in sensitive_keys else _sanitize(val))
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [_sanitize(item) for item in value]
        return value

    return _sanitize(dict(config)) if isinstance(config, dict) else {}


def _http_post_json(
    url: str,
    data: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
) -> tuple[int, str, Optional[Any]]:
    import json as _json
    import urllib.request

    payload_bytes = _json.dumps(data).encode("utf-8")
    request_headers = {
        str(key): str(value)
        for key, value in headers.items()
    }
    request_headers.setdefault("Content-Type", "application/json")

    request = urllib.request.Request(url, data=payload_bytes, headers=request_headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
        status_code = getattr(response, "status", None)
        if status_code is None:
            status_code = response.getcode()
        raw_bytes = response.read()

    text = raw_bytes.decode("utf-8", "replace")
    try:
        parsed = _json.loads(text)
    except Exception:
        parsed = None
    return int(status_code or 0), text, parsed


def _extract_llm_text(
    provider: str,
    payload: Any,
    config: Dict[str, Any],
) -> str:
    response_path = config.get("response_path") or config.get("result_path")
    if response_path:
        value = _traverse_attribute_path(payload, _as_path_segments(response_path))
        if value is not None:
            return str(value)
    provider_key = provider.lower()
    if isinstance(payload, dict):
        if provider_key == "openai":
            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, list):
                            segments: List[str] = []
                            for entry in content:
                                if isinstance(entry, dict) and "text" in entry:
                                    segments.append(str(entry["text"]))
                                else:
                                    segments.append(str(entry))
                            joined = "\n".join(segment for segment in segments if segment)
                            if joined:
                                return joined
                        if content is not None:
                            return str(content)
                    if "text" in first_choice:
                        return str(first_choice["text"])
        if provider_key == "anthropic":
            content = payload.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    if "text" in first:
                        return str(first["text"])
                    if "value" in first:
                        return str(first["value"])
            if isinstance(content, str):
                return content
        if "result" in payload and isinstance(payload["result"], str):
            return payload["result"]
    if isinstance(payload, str):
        return payload
    return json.dumps(payload)


def call_llm_connector(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute an LLM connector and return structured response details."""

    import traceback
    import urllib.error
    import urllib.parse

    start_time = time.time()
    spec = AI_CONNECTORS.get(name, {})
    args = dict(payload or {})
    context_stub = {
        "env": {key: os.getenv(key) for key in ENV_KEYS},
        "vars": {},
        "app": APP,
    }
    config_raw = spec.get("config", {})
    config_resolved = _resolve_placeholders(config_raw, context_stub)
    if not isinstance(config_resolved, dict):
        config_resolved = config_raw if isinstance(config_raw, dict) else {}

    provider = str(config_resolved.get("provider") or spec.get("type") or name or "").strip()
    model_name = str(config_resolved.get("model") or "").strip()
    allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")

    prompt_value = args.get("prompt") or args.get("input")
    prompt_text = _stringify_prompt_value(name, prompt_value) if prompt_value is not None else ""
    redacted_config = _redact_config(config_resolved)

    def _elapsed_ms() -> float:
        return float(round((time.time() - start_time) * 1000.0, 3))

    def _stub_response(reason: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.warning(_format_error_message(name, prompt_text, reason))
        response: Dict[str, Any] = {
            "status": "stub",
            "provider": provider or "unknown",
            "model": model_name,
            "inputs": args,
            "result": {"text": "[stub: llm call failed]"},
            "config": redacted_config,
            "error": reason,
        }
        if metadata:
            response["metadata"] = metadata
        return response

    def _error_response(reason: str, metadata: Optional[Dict[str, Any]] = None, tb_text: str = "") -> Dict[str, Any]:
        logger.error(_format_error_message(name, prompt_text, reason))
        response: Dict[str, Any] = {
            "status": "error",
            "provider": provider or "unknown",
            "model": model_name,
            "inputs": args,
            "error": reason,
        }
        if metadata:
            response["metadata"] = metadata
        if tb_text:
            response["traceback"] = tb_text
        return response

    if not provider or not model_name:
        reason = "LLM provider or model is not configured"
        meta = {"elapsed_ms": _elapsed_ms()}
        return _stub_response(reason, meta) if allow_stubs else _error_response(reason, meta)

    try:
        request_spec = _build_llm_request(provider, model_name, prompt_text, config_resolved, args)
        method = str(request_spec.get("method") or "POST").upper()
        if method != "POST":
            raise ValueError(f"Unsupported HTTP method '{method}' for LLM connector")

        url = str(request_spec.get("url"))
        params = request_spec.get("params") or {}
        if params:
            query = urllib.parse.urlencode(params, doseq=True)
            url = f"{url}&{query}" if "?" in url else f"{url}?{query}"

        timeout = float(request_spec.get("timeout") or 30.0)
        status_code, raw_text, parsed_json = _http_post_json(
            url,
            request_spec.get("body") or {},
            request_spec.get("headers") or {},
            timeout,
        )

        result_payload: Dict[str, Any] = {}
        if parsed_json is not None:
            result_payload["json"] = parsed_json
            extracted = _extract_llm_text(provider, parsed_json, config_resolved)
            if extracted:
                result_payload["text"] = str(extracted)
        if raw_text:
            if "text" not in result_payload:
                result_payload["text"] = raw_text
            result_payload.setdefault("raw", raw_text)

        metadata = {
            "http_status": status_code,
            "elapsed_ms": _elapsed_ms(),
        }

        return {
            "status": "ok",
            "provider": provider,
            "model": model_name,
            "inputs": args,
            "result": result_payload,
            "metadata": metadata,
        }
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", "replace") if exc.fp else ""
        reason = f"HTTP {exc.code}: {exc.reason or 'request failed'}"
        meta = {
            "http_status": exc.code,
            "elapsed_ms": _elapsed_ms(),
            "response": error_body[:1024] if error_body else None,
        }
        if meta.get("response") is None:
            meta.pop("response", None)
        return _stub_response(reason, meta) if allow_stubs else _error_response(reason, meta)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        meta = {"elapsed_ms": _elapsed_ms()}
        tb_text = ""
        try:
            tb_text = traceback.format_exc(limit=5).strip()
        except Exception:  # pragma: no cover
            tb_text = ""
        if tb_text and len(tb_text) > 3000:
            tb_text = tb_text[:3000]
        return _stub_response(reason, meta) if allow_stubs else _error_response(reason, meta, tb_text)


def run_chain(name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute a configured AI chain and return detailed step results."""

    start_time = time.time()
    args = dict(payload or {})
    allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")
    spec = AI_CHAINS.get(name)

    if not spec:
        response: Dict[str, Any] = {
            "status": "not_found",
            "result": None,
            "steps": [],
            "inputs": args,
            "metadata": {"elapsed_ms": float(round((time.time() - start_time) * 1000.0, 3))},
        }
        if not allow_stubs:
            response["error"] = f"Chain '{name}' is not defined"
        return response

    input_key = spec.get("input_key", "input")
    working: Any = args.get(input_key, args)
    steps_history: List[Dict[str, Any]] = []
    result_value: Any = None
    status: str = "partial"

    for index, step in enumerate(spec.get("steps", []), start=1):
        kind = (step.get("kind") or "").lower()
        target = step.get("target") or ""
        options = step.get("options") or {}
        stop_on_error = bool(step.get("stop_on_error", True))

        entry: Dict[str, Any] = {
            "step": index,
            "kind": kind,
            "name": target,
            "inputs": None,
            "output": None,
            "status": "partial",
        }
        steps_history.append(entry)

        if kind == "template":
            template = AI_TEMPLATES.get(target) or {}
            prompt = template.get("prompt", "")
            context = {
                "input": working,
                "vars": args,
                "payload": args,
            }
            entry["inputs"] = context
            try:
                rendered = _render_template_value(prompt, context)
                entry["output"] = rendered
                entry["status"] = "ok"
                working = rendered
                result_value = rendered
                status = "ok"
            except Exception as exc:  # pragma: no cover
                entry["output"] = {"error": str(exc)}
                entry["status"] = "error"
                status = "error"
                if stop_on_error:
                    break
        elif kind == "connector":
            connector_payload = dict(args)
            connector_payload.setdefault("prompt", working)
            entry["inputs"] = connector_payload
            response = call_llm_connector(target, connector_payload)
            entry["output"] = response
            entry["status"] = response.get("status", "partial")
            step_status = response.get("status")
            if step_status == "ok":
                working = response.get("result")
                result_value = working
                status = "ok"
            elif step_status == "error":
                status = "error"
                result_value = response
                if stop_on_error:
                    break
            elif step_status == "stub" and status != "error":
                result_value = result_value or response
        elif kind == "python":
            module_name = options.get("module") or target or ""
            method_name = options.get("method") or "predict"
            python_args = args
            provided_args = options.get("arguments")
            if isinstance(provided_args, dict):
                merged_args = dict(args)
                merged_args.update(provided_args)
                python_args = merged_args
            entry["inputs"] = python_args
            response = call_python_model(module_name, method_name, python_args)
            entry["output"] = response
            entry["status"] = response.get("status", "partial")
            step_status = response.get("status")
            if step_status == "ok":
                working = response.get("result")
                result_value = working
                status = "ok"
            elif step_status == "error":
                status = "error"
                result_value = response
                if stop_on_error:
                    break
            elif step_status == "stub" and status != "error":
                result_value = result_value or response
        else:
            entry["output"] = {"error": f"Unsupported step kind '{kind}'"}
            entry["status"] = "error"
            status = "error"
            if stop_on_error:
                break

    elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))
    if status != "error" and result_value is None:
        status = "partial"

    return {
        "status": status if steps_history else "partial",
        "result": result_value,
        "steps": steps_history,
        "inputs": args,
        "metadata": {"elapsed_ms": elapsed_ms},
    }

def evaluate_experiment(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate an experiment by executing each variant and computing metrics.

    Parameters
    ----------
    name:
        Experiment identifier registered in ``AI_EXPERIMENTS``.
    payload:
        Optional dictionary containing request inputs. Should include prediction
        payloads and, when required, ground-truth targets (for example
        ``{"y_true": [...], "y_pred": [...]}`` or ``{"examples": [{"y_true": .., "y_pred": ..}]}``).

    Returns
    -------
    Dict[str, Any]
        Structured evaluation report containing keys ``experiment``,
        ``variants`` (per-variant outputs and metric values), ``metrics``
        (experiment-level summaries), ``metric_definitions`` (metric settings),
        ``leaderboard`` (sorted variant summaries), ``winner`` (top variant
        name), ``inputs`` (echoed payload), ``metadata`` and a ``status`` flag
        (``"ok"`` or ``"not_found"``).

        Variants include a ``status`` field set to ``"ok"`` on success or
        ``"error"`` when execution failed. When a variant fails, the error is
        captured without aborting the experiment.
    """

    spec = AI_EXPERIMENTS.get(name)
    args = dict(payload or {})
    if not spec:
        return {
            "status": "error",
            "error": "experiment_not_found",
            "detail": f"Experiment '{name}' is not registered.",
            "experiment": name,
            "variants": [],
            "metrics": [],
            "metric_definitions": [],
            "leaderboard": [],
            "winner": None,
            "inputs": args,
            "metadata": {},
        }

    metric_configs = _normalise_metric_configs(spec.get("metrics", []))
    primary_metric = _resolve_primary_metric(metric_configs)

    variants_result: List[Dict[str, Any]] = []
    for variant in spec.get("variants", []):
        variant_result = _evaluate_experiment_variant(variant, args)
        if metric_configs:
            variant_metrics = _compute_variant_metrics(variant_result, metric_configs, args)
            variant_result["metrics"] = variant_metrics
        variants_result.append(variant_result)

    experiment_metrics = _summarise_experiment_metrics(metric_configs, variants_result)
    leaderboard, winner = _build_experiment_leaderboard(variants_result, primary_metric)

    status = "ok"
    if variants_result and all(entry.get("status") == "error" for entry in variants_result):
        status = "error"

    result = {
        "experiment": spec.get("name", name),
        "slug": spec.get("slug") or spec.get("name", name),
        "variants": variants_result,
        "metrics": experiment_metrics,
        "metric_definitions": metric_configs,
        "leaderboard": leaderboard,
        "winner": winner,
        "inputs": args,
        "metadata": spec.get("metadata", {}),
        "status": status,
    }
    if status == "error":
        result.setdefault("error", "experiment_failed")
        result.setdefault("detail", "All experiment variants failed to execute.")
    return result


def run_prediction(model_name: str, payload: Optional[Dict[str, Any]] = None) -> PredictionResponse:
    model_key = str(model_name)
    if model_key not in MODELS and model_key not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' is not registered")
    request_payload = dict(payload or {})
    result = predict(model_key, request_payload)
    response_payload = {
        "model": result.get("model", model_key),
        "version": result.get("version"),
        "framework": result.get("framework"),
        "input": result.get("input") or {},
        "output": result.get("output") or {},
        "explanations": result.get("explanations") or {},
        "metadata": result.get("spec_metadata") or result.get("metadata") or {},
    }
    return PredictionResponse(**response_payload)


async def predict_model(model_name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience helper used by the generated API and tests."""

    response = run_prediction(model_name, payload)
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return dict(response)


def run_experiment(slug: str, payload: Optional[Dict[str, Any]] = None) -> ExperimentResult:
    experiment_key = str(slug)
    if experiment_key not in AI_EXPERIMENTS:
        raise HTTPException(status_code=404, detail=f"Experiment '{slug}' is not defined")
    request_payload = dict(payload or {})
    result = evaluate_experiment(experiment_key, request_payload)
    return ExperimentResult(**result)


async def experiment_metrics(slug: str) -> Dict[str, Any]:
    """Return experiment metrics snapshot as a plain dictionary."""

    response = run_experiment(slug, {})
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return dict(response)


async def run_experiment_endpoint(
    slug: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute an experiment with an optional payload and return a dict."""

    response = run_experiment(slug, payload or {})
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return dict(response)


def _normalise_metric_configs(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    explicit_primary = False
    for metric in metrics or []:
        metadata_raw = metric.get("metadata") or {}
        metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {"value": metadata_raw}
        entry: Dict[str, Any] = {
            "name": metric.get("name"),
            "slug": metric.get("slug"),
            "goal": metric.get("goal"),
            "source_kind": metric.get("source_kind"),
            "source_name": metric.get("source_name"),
            "direction": _determine_metric_direction(metric.get("name"), metadata),
            "metadata": metadata,
        }
        primary_marker = metadata.get("primary") if isinstance(metadata, dict) else None
        if primary_marker is None:
            primary_marker = metric.get("primary")
        if isinstance(primary_marker, str):
            primary_marker = primary_marker.lower() in {"true", "1", "yes"}
        if isinstance(primary_marker, bool) and primary_marker:
            entry["primary"] = True
            explicit_primary = True
        configs.append(entry)
    if configs and not explicit_primary:
        configs[0]["primary"] = True
    return configs


def _determine_metric_direction(name: Optional[str], metadata: Dict[str, Any]) -> str:
    direction_value = metadata.get("direction") or metadata.get("goal_direction")
    if direction_value is None:
        direction_value = metadata.get("optimize") or metadata.get("optimise")
    if isinstance(direction_value, str):
        normalized = direction_value.lower()
        if normalized in {"max", "maximize", "maximise", "higher", "increase", "asc"}:
            return "maximize"
        if normalized in {"min", "minimize", "minimise", "lower", "decrease", "desc"}:
            return "minimize"
    goal_operator = metadata.get("goal_operator") or metadata.get("operator")
    if isinstance(goal_operator, str):
        text = goal_operator.strip().lower()
        if any(symbol in text for symbol in (">", "gt")):
            return "maximize"
        if any(symbol in text for symbol in ("<", "lt")):
            return "minimize"
    name_text = str(name or "").lower()
    if any(keyword in name_text for keyword in ("latency", "delay", "duration", "time", "error", "loss", "mse", "rmse", "mae", "cost")):
        return "minimize"
    return "maximize"


def _resolve_primary_metric(metrics: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not metrics:
        return None
    for metric in metrics:
        marker = metric.get("primary")
        if isinstance(marker, str):
            marker = marker.lower() in {"true", "1", "yes"}
        if marker:
            return metric
    return metrics[0]


def _evaluate_experiment_variant(variant: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    target_type = str(variant.get("target_type") or "model").lower()
    target_name = str(variant.get("target_name") or "")
    config = variant.get("config") or {}
    result: Dict[str, Any] = {
        "name": variant.get("name"),
        "slug": variant.get("slug"),
        "target_type": target_type,
        "target_name": target_name,
        "config": config,
        "status": "ok",
        "raw_output": None,
        "result": None,
        "metrics": {},
    }
    try:
        if target_type == "model" and target_name:
            model_input = _resolve_variant_input(args, config)
            payload = predict(target_name, model_input)
        elif target_type == "chain" and target_name:
            chain_args = _resolve_chain_arguments(args, config)
            payload = run_chain(target_name, chain_args)
        elif target_type == "python":
            module_name = config.get("module") or target_name
            method_name = config.get("method") or config.get("callable") or "predict"
            python_args = _resolve_python_arguments(args, config)
            if not module_name:
                raise ValueError("python variant requires a module in 'target_name' or config['module']")
            payload = call_python_model(module_name, method_name, python_args)
        else:
            raise ValueError(f"Unsupported target type '{target_type}' for variant '{variant.get('name')}'")
        result["raw_output"] = payload
        result["result"] = payload
        status_value = _extract_status_from_payload(payload)
        if status_value == "error":
            result["status"] = "error"
            error_text = _extract_error_from_payload(payload)
            if error_text:
                result["error"] = error_text
        elif status_value == "partial":
            result["status"] = "partial"
    except Exception as exc:  # pragma: no cover - defensive guard
        result["status"] = "error"
        result["error"] = str(exc)
    return result


def _extract_status_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        status_value = payload.get("status")
        if isinstance(status_value, str):
            normalized = status_value.lower()
            if normalized in {"ok", "success"}:
                return "ok"
            if normalized in {"error", "failed", "failure"}:
                return "error"
            if normalized in {"partial"}:
                return "partial"
    return "ok"


def _extract_error_from_payload(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for key in ("error", "detail", "message"):
            value = payload.get(key)
            if value:
                return str(value)
    return None


def _resolve_variant_input(args: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(config, dict):
        override = config.get("payload") or config.get("input") or config.get("data")
        if isinstance(override, dict):
            return dict(override)
        if isinstance(override, str) and override in args:
            candidate = args.get(override)
            if isinstance(candidate, dict):
                return dict(candidate)
            if candidate is not None:
                return {"value": candidate}
    model_input = args.get("input") or args.get("payload") or {}
    if isinstance(model_input, dict):
        return dict(model_input)
    if isinstance(model_input, (list, tuple)):
        return {"values": list(model_input)}
    if model_input is not None:
        return {"value": model_input}
    return {}


def _resolve_chain_arguments(args: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    chain_args = dict(args)
    if isinstance(config, dict):
        overrides = config.get("inputs") or config.get("arguments") or {}
        if isinstance(overrides, dict):
            chain_args.update(overrides)
        input_override = config.get("input")
        if isinstance(input_override, dict):
            chain_args["input"] = dict(input_override)
        elif isinstance(input_override, str) and input_override in args:
            chain_args["input"] = args[input_override]
    chain_args.setdefault("input", args.get("input", args))
    return chain_args


def _resolve_python_arguments(args: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    python_args: Dict[str, Any] = {}
    if isinstance(config, dict):
        for key in ("arguments", "inputs", "kwargs"):
            value = config.get(key)
            if isinstance(value, dict):
                python_args.update(value)
    if not python_args and isinstance(args, dict):
        python_args = {"payload": args}
    return python_args


def _compute_variant_metrics(
    variant_result: Dict[str, Any],
    metric_configs: List[Dict[str, Any]],
    args: Dict[str, Any],
) -> Dict[str, float]:
    if variant_result.get("status") != "ok":
        return {}
    metric_inputs = _resolve_metric_inputs(variant_result, args)
    metrics: Dict[str, float] = {}
    for metric in metric_configs:
        value = _evaluate_metric_value(metric, metric_inputs)
        if value is not None:
            metrics[metric["name"]] = value
    return metrics


def _summarise_experiment_metrics(
    metric_configs: List[Dict[str, Any]],
    variants: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for index, config in enumerate(metric_configs, start=1):
        name = config.get("name")
        metadata_source = config.get("metadata") or {}
        metadata = dict(metadata_source) if isinstance(metadata_source, dict) else {}
        values: List[float] = []
        if name:
            for variant in variants:
                variant_metrics = variant.get("metrics") or {}
                number = _to_float(variant_metrics.get(name))
                if number is not None:
                    values.append(number)
        if not values:
            for variant in variants:
                number = _extract_score_from_variant(variant)
                if number is not None:
                    values.append(number)
        aggregation = metadata.get("aggregation") or metadata.get("aggregate") or "mean"
        aggregation_text = str(aggregation).lower()
        aggregated_value = _aggregate_metric_values(values, aggregation_text)
        if aggregated_value is None:
            if not values:
                continue
            aggregated_value = float(sum(values) / len(values))
        else:
            aggregated_value = float(aggregated_value)
        metadata["aggregation"] = aggregation_text
        samples = len(values)
        metadata["samples"] = samples
        if values:
            metadata["summary"] = {
                "min": min(values),
                "max": max(values),
                "mean": round(sum(values) / len(values), 6),
            }
        precision = metadata.get("precision") if isinstance(metadata.get("precision"), int) else metadata.get("round")
        if isinstance(precision, int):
            aggregated_value = round(aggregated_value, max(int(precision), 0))
        else:
            aggregated_value = round(aggregated_value, 4)
        goal_value = _to_float(config.get("goal"))
        direction = config.get("direction")
        achieved_goal: Optional[bool] = None
        if goal_value is not None:
            minimize = str(direction).lower() == "minimize"
            achieved_goal = aggregated_value <= goal_value if minimize else aggregated_value >= goal_value
        summaries.append(
            {
                "name": name,
                "value": aggregated_value,
                "goal": config.get("goal"),
                "direction": direction,
                "source_kind": config.get("source_kind"),
                "source_name": config.get("source_name"),
                "metadata": metadata,
                "achieved_goal": achieved_goal,
                "primary": bool(config.get("primary")),
            }
        )
    return summaries


def _aggregate_metric_values(values: List[float], aggregation: Any) -> Optional[float]:
    if not values:
        return None
    key = str(aggregation or "mean").lower()
    if key in {"max", "maximum", "highest"}:
        return max(values)
    if key in {"min", "minimum", "lowest"}:
        return min(values)
    if key in {"sum", "total"}:
        return sum(values)
    if key in {"avg", "average", "mean"}:
        return sum(values) / len(values)
    if key in {"median", "p50"}:
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 0:
            return (ordered[mid - 1] + ordered[mid]) / 2.0
        return ordered[mid]
    if key in {"p95", "percentile95"}:
        return _percentile(values, 95)
    if key in {"p90", "percentile90"}:
        return _percentile(values, 90)
    if key in {"p80", "percentile80"}:
        return _percentile(values, 80)
    if key in {"p20", "percentile20"}:
        return _percentile(values, 20)
    if key in {"p10", "percentile10"}:
        return _percentile(values, 10)
    return sum(values) / len(values)


def _percentile(values: List[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    rank = (percentile / 100) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _resolve_metric_inputs(variant_result: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Optional[List[Any]]]:
    y_true = _normalise_sequence(args.get("y_true"))
    y_pred = _normalise_sequence(args.get("y_pred"))
    y_score = _normalise_numeric_series(args.get("y_score"))

    examples = args.get("examples")
    if isinstance(examples, list):
        if not y_true:
            y_true = [item.get("y_true") for item in examples if isinstance(item, dict) and item.get("y_true") is not None]
        if not y_pred:
            y_pred = [item.get("y_pred") for item in examples if isinstance(item, dict) and item.get("y_pred") is not None]

    if not y_true:
        for key in ("labels", "targets", "ground_truth", "actuals"):
            series = _normalise_sequence(args.get(key))
            if series:
                y_true = series
                break

    raw_output = variant_result.get("raw_output")
    if not y_pred:
        y_pred = _extract_predictions_from_raw(raw_output)
    if not y_true:
        y_true = _extract_ground_truth(args)
    if not y_score:
        y_score = _extract_numeric_scores(raw_output)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def _normalise_sequence(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict) and "values" in value:
        values = value.get("values")
        if isinstance(values, (list, tuple)):
            return list(values)
    return [value]


def _normalise_numeric_series(value: Any) -> Optional[List[float]]:
    sequence = _normalise_sequence(value)
    if not sequence:
        return None
    numeric: List[float] = []
    for item in sequence:
        number = _to_float(item)
        if number is None:
            return None
        numeric.append(number)
    return numeric


def _extract_predictions_from_raw(payload: Any) -> Optional[List[Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("predictions"), (list, tuple)):
            return list(payload["predictions"])
        output = payload.get("output")
        if isinstance(output, dict):
            if isinstance(output.get("predictions"), (list, tuple)):
                return list(output["predictions"])
            if isinstance(output.get("labels"), (list, tuple)):
                return list(output["labels"])
            if "label" in output:
                return [output.get("label")]
        if "result" in payload:
            result_value = payload.get("result")
            if isinstance(result_value, (list, tuple)):
                return list(result_value)
            return [result_value]
        if "value" in payload and not isinstance(payload.get("value"), dict):
            return [payload.get("value")]
    if isinstance(payload, (list, tuple)):
        return list(payload)
    if payload is not None:
        return [payload]
    return None


def _extract_numeric_scores(payload: Any) -> Optional[List[float]]:
    if isinstance(payload, dict):
        for key in ("score", "value"):
            number = _to_float(payload.get(key))
            if number is not None:
                return [number]
        output = payload.get("output")
        if isinstance(output, dict):
            number = _to_float(output.get("score"))
            if number is not None:
                return [number]
            if isinstance(output.get("scores"), (list, tuple)):
                numeric = [_to_float(item) for item in output.get("scores", [])]
                filtered = [item for item in numeric if item is not None]
                if filtered:
                    return [float(item) for item in filtered]
        result_value = payload.get("result")
        if isinstance(result_value, dict):
            numeric_nested = _extract_numeric_scores(result_value)
            if numeric_nested:
                return numeric_nested
        number = _to_float(payload.get("result"))
        if number is not None:
            return [number]
    if isinstance(payload, (list, tuple)):
        numeric = [_to_float(item) for item in payload]
        filtered = [item for item in numeric if item is not None]
        if filtered:
            return [float(item) for item in filtered]
    number = _to_float(payload)
    if number is not None:
        return [number]
    return None


def _extract_ground_truth(args: Dict[str, Any]) -> Optional[List[Any]]:
    for key in ("ground_truth", "truth", "labels", "targets", "actuals", "y_true"):
        series = _normalise_sequence(args.get(key))
        if series:
            return series
    examples = args.get("examples")
    if isinstance(examples, list):
        values = [item.get("y_true") for item in examples if isinstance(item, dict) and item.get("y_true") is not None]
        if values:
            return values
    return None


def _evaluate_metric_value(metric: Dict[str, Any], inputs: Dict[str, Optional[List[Any]]]) -> Optional[float]:
    name = str(metric.get("name") or "").lower()
    y_true = inputs.get("y_true")
    y_pred = inputs.get("y_pred")
    y_score = inputs.get("y_score")
    metadata = metric.get("metadata") or {}

    if name in {"accuracy", "acc"}:
        pairs = _label_pairs(y_true, y_pred)
        return _compute_accuracy(pairs)
    if name in {"precision"}:
        pairs = _label_pairs(y_true, y_pred)
        return _compute_precision(pairs, _safe_positive_label(metadata, pairs))
    if name in {"recall"}:
        pairs = _label_pairs(y_true, y_pred)
        return _compute_recall(pairs, _safe_positive_label(metadata, pairs))
    if name in {"f1", "f1_score"}:
        pairs = _label_pairs(y_true, y_pred)
        positive_label = _safe_positive_label(metadata, pairs)
        return _compute_f1(pairs, positive_label)
    if name in {"mse", "mean_squared_error"}:
        pairs = _numeric_pairs(y_true, y_pred or y_score)
        return _compute_mse(pairs)
    if name in {"rmse"}:
        pairs = _numeric_pairs(y_true, y_pred or y_score)
        return _compute_rmse(pairs)
    if name in {"mae", "mean_absolute_error"}:
        pairs = _numeric_pairs(y_true, y_pred or y_score)
        return _compute_mae(pairs)
    return None


def _label_pairs(y_true: Optional[List[Any]], y_pred: Optional[List[Any]]) -> List[Any]:
    if not y_true or not y_pred:
        return []
    pairs: List[Any] = []
    for truth, pred in zip(y_true, y_pred):
        if truth is None or pred is None:
            continue
        pairs.append((truth, pred))
    return pairs


def _numeric_pairs(y_true: Optional[List[Any]], y_pred: Optional[List[Any]]) -> List[Any]:
    if not y_true or not y_pred:
        return []
    pairs: List[Any] = []
    for truth, pred in zip(y_true, y_pred):
        truth_val = _to_float(truth)
        pred_val = _to_float(pred)
        if truth_val is None or pred_val is None:
            continue
        pairs.append((truth_val, pred_val))
    return pairs


def _labels_equal(left: Any, right: Any) -> bool:
    if left == right:
        return True
    try:
        return float(left) == float(right)
    except Exception:
        pass
    return str(left).strip().lower() == str(right).strip().lower()


def _labels_match(value: Any, positive_label: Any) -> bool:
    if isinstance(positive_label, bool):
        return bool(value) is bool(positive_label) and bool(value) == positive_label
    if isinstance(positive_label, (int, float)) and not isinstance(positive_label, bool):
        try:
            return float(value) == float(positive_label)
        except Exception:
            return False
    return str(value).strip().lower() == str(positive_label).strip().lower()


def _safe_positive_label(metadata: Dict[str, Any], pairs: List[Any]) -> Any:
    label = metadata.get("positive_label") or metadata.get("positive_class")
    if label is not None:
        return label
    candidates = [truth for truth, _ in pairs if truth is not None]
    for candidate in candidates:
        text = str(candidate).strip().lower()
        if text in {"positive", "pos"}:
            return candidate
    for candidate in candidates:
        if isinstance(candidate, bool):
            return True
    for candidate in candidates:
        numeric = _to_float(candidate)
        if numeric == 1.0:
            return candidate
    return True


def _compute_accuracy(pairs: List[Any]) -> Optional[float]:
    if not pairs:
        return None
    matches = sum(1 for truth, pred in pairs if _labels_equal(truth, pred))
    return matches / len(pairs)


def _compute_precision(pairs: List[Any], positive_label: Any) -> Optional[float]:
    if not pairs:
        return None
    tp = fp = 0
    for truth, pred in pairs:
        if _labels_match(pred, positive_label):
            if _labels_match(truth, positive_label):
                tp += 1
            else:
                fp += 1
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def _compute_recall(pairs: List[Any], positive_label: Any) -> Optional[float]:
    if not pairs:
        return None
    tp = fn = 0
    for truth, pred in pairs:
        if _labels_match(truth, positive_label):
            if _labels_match(pred, positive_label):
                tp += 1
            else:
                fn += 1
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def _compute_f1(pairs: List[Any], positive_label: Any) -> Optional[float]:
    if not pairs:
        return None
    precision = _compute_precision(pairs, positive_label) or 0.0
    recall = _compute_recall(pairs, positive_label) or 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _compute_mse(pairs: List[Any]) -> Optional[float]:
    if not pairs:
        return None
    errors = [(pred - truth) ** 2 for truth, pred in pairs]
    return sum(errors) / len(errors)


def _compute_rmse(pairs: List[Any]) -> Optional[float]:
    mse = _compute_mse(pairs)
    if mse is None:
        return None
    return mse ** 0.5


def _compute_mae(pairs: List[Any]) -> Optional[float]:
    if not pairs:
        return None
    errors = [abs(pred - truth) for truth, pred in pairs]
    return sum(errors) / len(errors)


def _build_experiment_leaderboard(
    variants: List[Dict[str, Any]],
    primary_metric: Optional[Dict[str, Any]],
):
    if not variants:
        return [], None
    metric_name = (primary_metric or {}).get("name")
    direction = (primary_metric or {}).get("direction", "maximize")
    maximize = str(direction).lower() != "minimize"

    entries: List[Dict[str, Any]] = []
    for variant in variants:
        metrics = variant.get("metrics") or {}
        score_value: Optional[float] = None
        if metric_name and metric_name in metrics:
            score_value = _to_float(metrics.get(metric_name))
        if score_value is None:
            score_value = _extract_score_from_variant(variant)
        variant["score"] = score_value
        if score_value is not None:
            entries.append({"name": variant.get("name"), "score": float(score_value)})

    if not entries:
        return [], None

    def sort_key(item: Dict[str, Any]) -> float:
        value = item.get("score")
        return -value if maximize else value

    leaderboard = sorted(entries, key=sort_key)
    winner = leaderboard[0]["name"] if leaderboard else None
    return leaderboard, winner


def _extract_score_from_variant(variant: Dict[str, Any]) -> Optional[float]:
    direct = _to_float(variant.get("score"))
    if direct is not None:
        return direct
    raw_output = variant.get("raw_output") or variant.get("result")
    if isinstance(raw_output, dict):
        for key in ("score",):
            number = _to_float(raw_output.get(key))
            if number is not None:
                return number
        output = raw_output.get("output")
        if isinstance(output, dict):
            number = _to_float(output.get("score"))
            if number is not None:
                return number
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None

def _generic_loader(model_name: str, model_spec: Dict[str, Any]) -> Any:
    metadata_obj = model_spec.get("metadata")
    metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
    loader_path = metadata.get("loader")
    if not loader_path:
        return None
    try:
        callable_loader = _load_python_callable(loader_path)
        if callable_loader is None:
            return None
        return callable_loader(model_name, model_spec)
    except Exception:  # pragma: no cover - user loader failure
        logger.exception("Generic loader failed for model %s", model_name)
        return None


def _pytorch_loader(model_name: str, model_spec: Dict[str, Any]) -> Any:
    path = _resolve_model_artifact_path(model_spec)
    if not path or not Path(path).exists():
        return None
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        logger.debug("PyTorch not available for model %s", model_name)
        return None
    try:
        if path.endswith((".pt", ".pth")):
            try:
                return torch.jit.load(path, map_location="cpu")
            except Exception:
                return torch.load(path, map_location="cpu")
        return torch.load(path, map_location="cpu")
    except Exception:  # pragma: no cover - IO/runtime failure
        logger.exception("Failed to load PyTorch model %s from %s", model_name, path)
        return None


def _tensorflow_loader(model_name: str, model_spec: Dict[str, Any]) -> Any:
    path = _resolve_model_artifact_path(model_spec)
    if not path or not Path(path).exists():
        return None
    try:
        import tensorflow as tf  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        logger.debug("TensorFlow not available for model %s", model_name)
        return None
    try:
        return tf.saved_model.load(path)
    except Exception:  # pragma: no cover - IO/runtime failure
        logger.exception("Failed to load TensorFlow model %s from %s", model_name, path)
        return None


def _onnx_loader(model_name: str, model_spec: Dict[str, Any]) -> Any:
    path = _resolve_model_artifact_path(model_spec)
    if not path or not Path(path).exists():
        return None
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        logger.debug("ONNX Runtime not available for model %s", model_name)
        return None
    try:
        return ort.InferenceSession(path)
    except Exception:  # pragma: no cover - IO/runtime failure
        logger.exception("Failed to load ONNX model %s from %s", model_name, path)
        return None


def _sklearn_loader(model_name: str, model_spec: Dict[str, Any]) -> Any:
    path = _resolve_model_artifact_path(model_spec)
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        logger.debug("scikit-learn artifact for %s not found at %s", model_name, file_path)
        return None
    try:
        try:
            import joblib  # type: ignore
        except Exception:
            joblib = None  # type: ignore
        if joblib is not None:
            return joblib.load(file_path)
    except Exception:
        logger.exception("joblib failed to load model %s", model_name)
    try:
        with file_path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        logger.exception("Failed to load pickled model %s from %s", model_name, file_path)
    return None


def _coerce_numeric_payload(payload: Any) -> Optional[List[float]]:
    if isinstance(payload, dict):
        values: List[float] = []
        for key in sorted(payload):
            value = payload[key]
            if isinstance(value, (int, float)):
                values.append(float(value))
            elif isinstance(value, (list, tuple)):
                try:
                    values.extend(float(item) for item in value)
                except Exception:
                    return None
            else:
                return None
        return values
    if isinstance(payload, (list, tuple)):
        try:
            return [float(item) for item in payload]
        except Exception:
            return None
    if isinstance(payload, (int, float)):
        return [float(payload)]
    return None


def _generic_runner(
    model_name: str,
    model_instance: Any,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    metadata = model_spec.get("metadata") or {}
    runner_path = metadata.get("runner")
    if runner_path:
        try:
            callable_runner = _load_python_callable(runner_path)
            if callable_runner is not None:
                result = callable_runner(model_instance, payload, model_spec)
                if isinstance(result, dict):
                    return result
        except Exception:  # pragma: no cover - user runner failure
            logger.exception("Custom runner failed for model %s", model_name)
    if callable(model_instance):
        try:
            result = model_instance(payload)
            if isinstance(result, dict):
                return result
            if isinstance(result, (list, tuple)) and result:
                score = float(result[0])
                label = "Positive" if score >= 0 else "Negative"
                return {"score": score, "label": label}
        except Exception:
            logger.debug("Callable runner failed for model %s", model_name)
    return None


def _pytorch_runner(
    model_name: str,
    model_instance: Any,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None
    if model_instance is None:
        return None
    try:
        if hasattr(model_instance, "eval"):
            model_instance.eval()
        values = _coerce_numeric_payload(payload)
        if values is None:
            return None
        input_tensor = torch.tensor(values, dtype=torch.float32)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = model_instance(input_tensor)
        if isinstance(output, torch.Tensor):
            flattened = output.detach().cpu().view(-1).tolist()
            score = float(flattened[0]) if flattened else 0.0
        elif isinstance(output, (list, tuple)) and output:
            score = float(output[0])
        else:
            return None
        label = "Positive" if score >= 0 else "Negative"
        return {"score": score, "label": label}
    except Exception:  # pragma: no cover - runtime failure
        logger.exception("PyTorch runner failed for model %s", model_name)
        return None


def _tensorflow_runner(
    model_name: str,
    model_instance: Any,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        import tensorflow as tf  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None
    if model_instance is None:
        return None
    try:
        values = _coerce_numeric_payload(payload)
        if values is None:
            return None
        input_tensor = tf.convert_to_tensor([values], dtype=tf.float32)
        output = model_instance(input_tensor)
        if hasattr(output, "numpy"):
            score = float(output.numpy().reshape(-1)[0])
        elif isinstance(output, (list, tuple)) and output:
            first = output[0]
            if hasattr(first, "numpy"):
                score = float(first.numpy().reshape(-1)[0])
            else:
                score = float(first)
        else:
            return None
        label = "Positive" if score >= 0 else "Negative"
        return {"score": score, "label": label}
    except Exception:  # pragma: no cover - runtime failure
        logger.exception("TensorFlow runner failed for model %s", model_name)
        return None


def _onnx_runner(
    model_name: str,
    model_instance: Any,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if model_instance is None:
        return None
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None
    try:
        values = _coerce_numeric_payload(payload)
        if values is None:
            return None
        input_name = model_instance.get_inputs()[0].name  # type: ignore[attr-defined]
        array = np.array([values], dtype=np.float32)
        output = model_instance.run(None, {input_name: array})  # type: ignore[call-arg]
        if not output:
            return None
        first = output[0]
        if hasattr(first, "reshape"):
            score = float(first.reshape(-1)[0])
        elif isinstance(first, (list, tuple)) and first:
            score = float(first[0])
        else:
            score = float(first)
        label = "Positive" if score >= 0 else "Negative"
        return {"score": score, "label": label}
    except Exception:  # pragma: no cover - runtime failure
        logger.exception("ONNX runner failed for model %s", model_name)
        return None


def _sklearn_runner(
    model_name: str,
    model_instance: Any,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if model_instance is None:
        return None
    values = _coerce_numeric_payload(payload)
    if values is None:
        return None
    sample = [values]
    try:
        score: Optional[float] = None
        if hasattr(model_instance, "predict_proba"):
            probabilities = model_instance.predict_proba(sample)  # type: ignore[attr-defined]
            if hasattr(probabilities, "tolist"):
                probabilities = probabilities.tolist()
            if isinstance(probabilities, list) and probabilities:
                first = probabilities[0]
                if isinstance(first, list) and first:
                    score = float(first[-1])
                elif isinstance(first, (int, float)):
                    score = float(first)
        if score is None and hasattr(model_instance, "predict"):
            prediction = model_instance.predict(sample)  # type: ignore[attr-defined]
            if hasattr(prediction, "tolist"):
                prediction = prediction.tolist()
            if isinstance(prediction, list) and prediction:
                first_value = prediction[0]
                score = float(first_value)
            elif isinstance(prediction, (int, float)):
                score = float(prediction)
        if score is None:
            return None
        metadata = model_spec.get("metadata") or {}
        threshold = float(metadata.get("threshold", 0.5))
        label = "Positive" if score >= threshold else "Negative"
        return {"score": score, "label": label}
    except Exception:
        logger.exception("scikit-learn runner failed for model %s", model_name)
        return None


def _default_explainer(
    model_name: str,
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    return _default_explanations(model_name, payload, prediction)


def _register_default_model_hooks() -> None:
    register_model_loader("generic", _generic_loader)
    register_model_loader("python", _generic_loader)
    register_model_loader("pytorch", _pytorch_loader)
    register_model_loader("torch", _pytorch_loader)
    register_model_loader("sklearn", _sklearn_loader)
    register_model_loader("scikit-learn", _sklearn_loader)
    register_model_loader("tensorflow", _tensorflow_loader)
    register_model_loader("tf", _tensorflow_loader)
    register_model_loader("onnx", _onnx_loader)
    register_model_loader("onnxruntime", _onnx_loader)

    register_model_runner("generic", _generic_runner)
    register_model_runner("callable", _generic_runner)
    register_model_runner("pytorch", _pytorch_runner)
    register_model_runner("torch", _pytorch_runner)
    register_model_runner("sklearn", _sklearn_runner)
    register_model_runner("scikit-learn", _sklearn_runner)
    register_model_runner("tensorflow", _tensorflow_runner)
    register_model_runner("tf", _tensorflow_runner)
    register_model_runner("onnx", _onnx_runner)
    register_model_runner("onnxruntime", _onnx_runner)

    register_model_explainer("generic", _default_explainer)
    register_model_explainer("pytorch", _default_explainer)
    register_model_explainer("torch", _default_explainer)
    register_model_explainer("tensorflow", _default_explainer)
    register_model_explainer("tf", _default_explainer)
    register_model_explainer("onnx", _default_explainer)
    register_model_explainer("onnxruntime", _default_explainer)


_register_default_model_hooks()

import importlib
import inspect
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    httpx = None  # type: ignore

from sqlalchemy.ext.asyncio import AsyncSession


def _normalize_connector_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        rows: List[Dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                rows.append(dict(item))
            else:
                rows.append({"value": item})
        return rows
    if isinstance(payload, dict):
        return [dict(payload)]
    return []


def _extract_rows_from_connector_response(payload: Any) -> List[Dict[str, Any]]:
    """Return normalized rows from a connector driver response."""

    if isinstance(payload, dict):
        if "result" in payload:
            return _normalize_connector_rows(payload.get("result"))
        if "batch" in payload:
            return _normalize_connector_rows(payload.get("batch"))
        return []
    return _normalize_connector_rows(payload)


def _is_truthy_env(name: str) -> bool:
    """Return True when the named environment variable is set to a truthy value."""

    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _redact_secrets(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of *data* with secret-like keys masked."""

    secret_keys = {"api_key", "authorization", "token", "secret", "password", "x-api-key"}

    def _sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): ("***" if str(key).strip().lower() in secret_keys else _sanitize(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [_sanitize(item) for item in value]
        return value

    if not isinstance(data, dict):
        return {}
    return _sanitize(dict(data))


def _now_ms() -> float:
    """Return the current wall-clock time in milliseconds with millisecond precision."""

    return float(round(time.time() * 1000.0, 3))


async def _default_sql_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not connector:
        return []
    try:
        _require_dependency("sqlalchemy", "sql")
    except ImportError as exc:
        raise ImportError(str(exc)) from exc
    query = connector.get("options", {}).get("query")
    if not query:
        table_name = connector.get("options", {}).get("table") or connector.get("name")
        if not table_name:
            return []
        query = f"SELECT * FROM {table_name}"
    session: Optional[AsyncSession] = context.get("session")
    if session is None:
        return []
    try:
        result = await session.execute(text(query))
        return [dict(row) for row in result.mappings()]
    except Exception:
        logger.exception("Default SQL driver failed for query '%s'", query)
        return []


async def _default_rest_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    endpoint = connector.get("options", {}).get("endpoint") if connector else None
    if not endpoint:
        return []
    if httpx is None:
        logger.warning("REST connector '%s' requires httpx to be installed", connector.get("name"))
        return []
    method = str(connector.get("options", {}).get("method") or "get").lower()
    payload = _resolve_placeholders(connector.get("options", {}).get("payload"), context)
    headers = _resolve_placeholders(connector.get("options", {}).get("headers"), context)
    timeout_value = connector.get("options", {}).get("timeout_ms")
    try:
        timeout = float(timeout_value) / 1000.0 if timeout_value is not None else 10.0
    except Exception:
        timeout = 10.0
    retries_value = connector.get("options", {}).get("max_retries")
    try:
        retries = max(int(retries_value), 0) if retries_value is not None else 1
    except Exception:
        retries = 1
    client_kwargs: Dict[str, Any] = {}
    timeout_config: Optional[Any]
    if httpx is not None:
        try:
            timeout_config = httpx.Timeout(timeout)
        except Exception:
            timeout_config = timeout
    else:
        timeout_config = timeout
    if timeout_config is not None:
        client_kwargs["timeout"] = timeout_config
    async with _HTTPX_CLIENT_CLS(**client_kwargs) as client:
        try:
            request_method = getattr(client, method, client.get)
            attempt = 0
            while True:
                attempt += 1
                start = _now_ms()
                try:
                    response = await request_method(
                        endpoint,
                        json=payload if isinstance(payload, dict) else None,
                        headers=headers if isinstance(headers, dict) else None,
                    )
                    response.raise_for_status()
                    data = response.json()
                    rows = _normalize_connector_rows(data)
                    logger.info(
                        "REST connector '%s' succeeded in %.2f ms",
                        connector.get("name"),
                        _now_ms() - start,
                    )
                    if rows:
                        return rows
                    break
                except ((httpx.HTTPError, httpx.TimeoutException) if httpx is not None else (Exception,)) as exc:
                    logger.warning(
                        "REST connector '%s' attempt %d/%d failed: %s",
                        connector.get("name"),
                        attempt,
                        retries,
                        exc,
                    )
                    if attempt >= retries:
                        raise
                except Exception:
                    logger.exception("Default REST driver failed for endpoint '%s'", endpoint)
                    break
        except Exception:
            logger.error("REST connector '%s' exhausted retries", connector.get("name"))
    return []


async def _default_graphql_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    options = connector.get("options", {}) if connector else {}
    endpoint = options.get("endpoint") or options.get("url") or connector.get("name")
    query = options.get("query")
    if not endpoint or not query:
        return []
    if httpx is None:
        logger.warning("GraphQL connector '%s' requires httpx to be installed", connector.get("name"))
        return []
    variables = _resolve_placeholders(options.get("variables"), context)
    headers = _resolve_placeholders(options.get("headers"), context)
    root_field = options.get("root")
    timeout_value = options.get("timeout_ms")
    try:
        timeout = float(timeout_value) / 1000.0 if timeout_value is not None else 10.0
    except Exception:
        timeout = 10.0
    retries_value = options.get("max_retries")
    try:
        retries = max(int(retries_value), 0) if retries_value is not None else 1
    except Exception:
        retries = 1
    client_kwargs: Dict[str, Any] = {}
    if httpx is not None:
        try:
            client_kwargs["timeout"] = httpx.Timeout(timeout)
        except Exception:
            client_kwargs["timeout"] = timeout
    else:
        client_kwargs["timeout"] = timeout
    async with _HTTPX_CLIENT_CLS(**client_kwargs) as client:
        try:
            attempt = 0
            while True:
                attempt += 1
                start = _now_ms()
                try:
                    response = await client.post(
                        endpoint,
                        json={"query": query, "variables": variables or {}},
                        headers=headers if isinstance(headers, dict) else None,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    logger.info(
                        "GraphQL connector '%s' succeeded in %.2f ms",
                        connector.get("name"),
                        _now_ms() - start,
                    )
                    break
                except ((httpx.HTTPError, httpx.TimeoutException) if httpx is not None else (Exception,)) as exc:
                    logger.warning(
                        "GraphQL connector '%s' attempt %d/%d failed: %s",
                        connector.get("name"),
                        attempt,
                        retries,
                        exc,
                    )
                    if attempt >= retries:
                        raise
                except Exception:
                    logger.exception("Default GraphQL driver failed for endpoint '%s'", endpoint)
                    raise
        except Exception:
            logger.error("GraphQL connector '%s' exhausted retries", connector.get("name"))
            return []
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict):
        target: Any
        if root_field:
            target = data.get(root_field)
        else:
            target = next(iter(data.values())) if data else None
        rows = _normalize_connector_rows(target)
        if rows:
            return rows
    return []


async def _default_grpc_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a gRPC connector via an optional pluggable driver."""

    start_ms = _now_ms()
    connector_obj = connector or {}
    raw_options = connector_obj.get("options") or {}
    resolved_options = _resolve_placeholders(raw_options, context) if raw_options else {}
    options = resolved_options if isinstance(resolved_options, dict) else (raw_options if isinstance(raw_options, dict) else {})
    inputs = {"connector": connector_obj.get("name")}
    redacted_config = _redact_secrets(options)

    allow_demo = _is_truthy_env("NAMEL3SS_ALLOW_STUBS") or bool(options.get("demo"))

    host = str(options.get("host") or "").strip()
    service = str(options.get("service") or connector_obj.get("name") or "").strip()
    method = str(options.get("method") or "").strip()

    def _elapsed() -> float:
        return max(_now_ms() - start_ms, 0.0)

    if not host or not service or not method:
        message = "Missing gRPC configuration (host/service/method)"
        logger.warning("gRPC connector '%s' missing configuration", connector_obj.get("name"))
        return {
            "status": "not_configured",
            "result": None,
            "error": message,
            "traceback": None,
            "config": redacted_config,
            "inputs": inputs,
            "metadata": {"elapsed_ms": _elapsed()},
        }

    port_value = options.get("port")
    try:
        port = int(port_value) if port_value is not None else 443
    except Exception:
        port = 443

    tls = bool(options.get("tls", True))
    deadline_value = options.get("deadline_ms")
    try:
        deadline_ms = int(deadline_value) if deadline_value is not None else None
    except Exception:
        deadline_ms = None

    metadata_option = options.get("metadata")
    metadata_headers = metadata_option if isinstance(metadata_option, dict) else None
    payload_option = options.get("payload")
    payload_dict = payload_option if isinstance(payload_option, dict) else {}

    driver_path = options.get("driver")

    def _import_driver(path: str) -> Callable[..., Any]:
        candidate = _load_python_callable(path) if ":" in path else None
        if candidate is None and path:
            module_name, _, attr = path.rpartition(".")
            if not module_name or not attr:
                raise ImportError(f"Invalid driver path '{path}'")
            module = importlib.import_module(module_name)
            candidate = getattr(module, attr)
        if candidate is None or not callable(candidate):
            raise TypeError(f"Driver '{path}' is not callable")
        return candidate

    if isinstance(driver_path, str) and driver_path.strip():
        try:
            driver_callable = _import_driver(driver_path.strip())
        except Exception as exc:
            logger.error("Failed to import gRPC driver '%s' for connector '%s'", driver_path, connector_obj.get("name"))
            return {
                "status": "error",
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": _trim_traceback(),
                "config": redacted_config,
                "inputs": inputs,
                "metadata": {"elapsed_ms": _elapsed()},
            }

        try:
            response = driver_callable(
                host=host,
                service=service,
                method=method,
                payload=payload_dict,
                port=port,
                metadata=metadata_headers,
                tls=tls,
                deadline_ms=deadline_ms,
            )
            if inspect.isawaitable(response):
                response = await response
        except Exception as exc:
            logger.error(
                "gRPC driver '%s' raised an error for connector '%s'",
                driver_path,
                connector_obj.get("name"),
            )
            return {
                "status": "error",
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": _trim_traceback(),
                "config": redacted_config,
                "inputs": inputs,
                "metadata": {"elapsed_ms": _elapsed()},
            }

        return {
            "status": "ok",
            "result": response,
            "error": None,
            "traceback": None,
            "config": None,
            "inputs": inputs,
            "metadata": {
                "elapsed_ms": _elapsed(),
                "endpoint": f"{host}:{port}",
                "service": service,
                "method": method,
            },
        }

    if allow_demo:
        logger.info("gRPC connector '%s' running in demo mode", connector_obj.get("name"))
        return {
            "status": "demo",
            "result": [
                {
                    "service": service,
                    "method": method,
                    "note": "demo mode – no gRPC client configured",
                }
            ],
            "error": None,
            "traceback": None,
            "config": redacted_config,
            "inputs": inputs,
            "metadata": {
                "elapsed_ms": _elapsed(),
                "endpoint": f"{host}:{port}",
                "service": service,
                "method": method,
            },
        }

    message = "No gRPC driver configured. Set 'driver' to a callable implementation or enable demo mode."
    logger.warning("gRPC connector '%s' has no driver configured", connector_obj.get("name"))
    return {
        "status": "not_configured",
        "result": None,
        "error": message,
        "traceback": None,
        "config": redacted_config,
        "inputs": inputs,
        "metadata": {"elapsed_ms": _elapsed()},
    }


async def _default_streaming_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Produce streaming batches from configured sources without fabricating data by default."""

    start_ms = _now_ms()
    connector_obj = connector or {}
    raw_options = connector_obj.get("options") or {}
    resolved_options = _resolve_placeholders(raw_options, context) if raw_options else {}
    options = resolved_options if isinstance(resolved_options, dict) else (raw_options if isinstance(raw_options, dict) else {})

    def _elapsed() -> float:
        return max(_now_ms() - start_ms, 0.0)

    redacted_config = _redact_secrets(options)
    allow_demo = _is_truthy_env("NAMEL3SS_ALLOW_STUBS") or bool(options.get("demo"))

    batch_size_value = options.get("batch_size", 100)
    try:
        batch_size = max(int(batch_size_value), 1)
    except Exception:
        batch_size = 100

    max_rows_value = options.get("max_rows")
    try:
        max_rows = int(max_rows_value) if max_rows_value is not None else None
    except Exception:
        max_rows = None

    connector_name = str(connector_obj.get("name") or options.get("stream") or "default")
    cursors = context.setdefault("stream_cursors", {}) if isinstance(context, dict) else {}
    cursor_state = cursors.setdefault(connector_name, {})

    seed_rows_raw = options.get("seed_rows") or options.get("rows") or options.get("sample")
    seed_rows_resolved = _resolve_placeholders(seed_rows_raw, context) if seed_rows_raw else None
    seed_rows = _normalize_connector_rows(seed_rows_resolved) if seed_rows_resolved is not None else []

    source_spec = options.get("source") if isinstance(options.get("source"), dict) else {}
    source_type = str(source_spec.get("type") or "").lower()

    def _normalize_batch(items: Iterable[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                normalized.append(dict(item))
            else:
                normalized.append({"value": item})
        return normalized[:batch_size]

    def _error_result(message: str, trace: Optional[str] = None, source: Optional[str] = None) -> Dict[str, Any]:
        logger.error("Streaming connector '%s' failed: %s", connector_name, message)
        metadata = {"elapsed_ms": _elapsed(), "source": source, "exhausted": False}
        return {
            "status": "error",
            "batch": None,
            "error": message,
            "traceback": trace,
            "config": redacted_config,
            "metadata": metadata,
        }

    if source_type == "python":
        driver_path = source_spec.get("driver") or source_spec.get("callable")
        if not isinstance(driver_path, str) or not driver_path.strip():
            return {
                "status": "not_configured",
                "batch": None,
                "error": "Python streaming source requires a 'driver' callable",
                "traceback": None,
                "config": redacted_config,
                "metadata": {"elapsed_ms": _elapsed(), "source": "python", "exhausted": False},
            }
        try:
            driver_callable = _load_python_callable(driver_path.strip())
            if driver_callable is None:
                module_name, _, attr = driver_path.strip().rpartition(".")
                if not module_name or not attr:
                    raise ImportError(f"Invalid driver path '{driver_path}'")
                module = importlib.import_module(module_name)
                driver_callable = getattr(module, attr)
            if not callable(driver_callable):
                raise TypeError(f"Driver '{driver_path}' is not callable")
        except Exception as exc:
            return _error_result(f"{type(exc).__name__}: {exc}", _trim_traceback(), "python")

        kwargs = source_spec.get("kwargs") if isinstance(source_spec.get("kwargs"), dict) else {}
        iterator = cursor_state.get("iterator")
        if iterator is None:
            try:
                produced = driver_callable(**kwargs) if kwargs else driver_callable()
            except Exception as exc:
                return _error_result(f"{type(exc).__name__}: {exc}", _trim_traceback(), "python")
            iterator = iter(produced)
            cursor_state["iterator"] = iterator
            cursor_state["exhausted"] = False

        batch: List[Dict[str, Any]] = []
        while len(batch) < batch_size:
            try:
                item = next(iterator)
            except StopIteration:
                cursor_state["exhausted"] = True
                break
            if item is None:
                continue
            batch.append(item if isinstance(item, dict) else {"value": item})

        metadata = {
            "elapsed_ms": _elapsed(),
            "source": "python",
            "exhausted": bool(cursor_state.get("exhausted") and not batch),
        }
        return {
            "status": "ok",
            "batch": batch,
            "error": None,
            "traceback": None,
            "config": None,
            "metadata": metadata,
        }

    if source_type == "http":
        url = source_spec.get("url")
        if not isinstance(url, str) or not url.strip():
            return {
                "status": "not_configured",
                "batch": None,
                "error": "HTTP streaming source requires a 'url'",
                "traceback": None,
                "config": redacted_config,
                "metadata": {"elapsed_ms": _elapsed(), "source": "http", "exhausted": False},
            }
        method = str(source_spec.get("method") or "GET").upper()
        headers = source_spec.get("headers") if isinstance(source_spec.get("headers"), dict) else {}
        body = source_spec.get("body") if isinstance(source_spec.get("body"), (dict, list)) else None
        timeout_value = source_spec.get("timeout", 10.0)
        try:
            timeout = float(timeout_value)
        except Exception:
            timeout = 10.0

        import json as _json
        import urllib.error
        import urllib.request

        data_bytes = None
        request_headers = {str(key): str(value) for key, value in headers.items()}
        if body is not None and method in {"POST", "PUT", "PATCH"}:
            data_bytes = _json.dumps(body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")

        request = urllib.request.Request(url, data=data_bytes, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw_bytes = response.read()
        except urllib.error.URLError as exc:  # pragma: no cover - network failures
            return _error_result(f"{type(exc).__name__}: {exc}", _trim_traceback(), "http")

        text = raw_bytes.decode("utf-8", "replace") if raw_bytes else ""
        try:
            payload = _json.loads(text) if text else []
        except Exception:
            payload = text

        if isinstance(payload, list):
            batch = _normalize_batch(payload)
        elif isinstance(payload, dict):
            batch = _normalize_batch([payload])
        else:
            batch = []

        metadata = {"elapsed_ms": _elapsed(), "source": "http", "exhausted": len(batch) < batch_size}
        return {
            "status": "ok",
            "batch": batch,
            "error": None,
            "traceback": None,
            "config": None,
            "metadata": metadata,
        }

    if source_type == "file":
        path = source_spec.get("path")
        if not isinstance(path, str) or not path:
            return {
                "status": "not_configured",
                "batch": None,
                "error": "File streaming source requires a 'path'",
                "traceback": None,
                "config": redacted_config,
                "metadata": {"elapsed_ms": _elapsed(), "source": "file", "exhausted": False},
            }
        fmt = str(source_spec.get("format") or "jsonl").lower()

        import json as _json

        if fmt == "jsonl":
            offset = int(cursor_state.get("offset", 0))
            batch: List[Dict[str, Any]] = []
            exhausted = False
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    handle.seek(offset)
                    for _ in range(batch_size):
                        line = handle.readline()
                        if not line:
                            exhausted = True
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            batch.append(_json.loads(line))
                        except Exception:
                            batch.append({"value": line})
                    cursor_state["offset"] = handle.tell()
            except FileNotFoundError:
                return _error_result("File source not found", None, "file")

            metadata = {"elapsed_ms": _elapsed(), "source": "file", "exhausted": exhausted and not batch}
            return {
                "status": "ok",
                "batch": _normalize_batch(batch),
                "error": None,
                "traceback": None,
                "config": None,
                "metadata": metadata,
            }

        records = cursor_state.get("records")
        if records is None:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = _json.load(handle)
            except FileNotFoundError:
                return _error_result("File source not found", None, "file")
            except Exception as exc:
                return _error_result(f"{type(exc).__name__}: {exc}", _trim_traceback(), "file")
            if isinstance(payload, list):
                records = payload
            elif isinstance(payload, dict):
                records = [payload]
            else:
                records = []
            cursor_state["records"] = records

        index = int(cursor_state.get("index", 0))
        chunk = records[index:index + batch_size]
        cursor_state["index"] = index + len(chunk)
        exhausted = cursor_state["index"] >= len(records)
        metadata = {"elapsed_ms": _elapsed(), "source": "file", "exhausted": exhausted}
        return {
            "status": "ok",
            "batch": _normalize_batch(chunk),
            "error": None,
            "traceback": None,
            "config": None,
            "metadata": metadata,
        }

    if seed_rows and not source_type:
        stored_rows = cursor_state.setdefault("seed_rows", seed_rows)
        index = int(cursor_state.get("index", 0))
        slice_rows = stored_rows[index:index + batch_size]
        cursor_state["index"] = index + len(slice_rows)
        exhausted = cursor_state["index"] >= len(stored_rows)
        metadata = {"elapsed_ms": _elapsed(), "source": None, "exhausted": exhausted}
        return {
            "status": "ok",
            "batch": _normalize_batch(slice_rows),
            "error": None,
            "traceback": None,
            "config": None,
            "metadata": metadata,
        }

    if not seed_rows and not source_type:
        if allow_demo:
            start_index = int(cursor_state.get("demo_index", 0))
            end_index = start_index + batch_size
            if max_rows is not None:
                end_index = min(end_index, max_rows)
            batch = [
                {"demo": True, "sequence": seq}
                for seq in range(start_index, end_index)
            ]
            cursor_state["demo_index"] = end_index
            exhausted = max_rows is not None and end_index >= max_rows
            metadata = {"elapsed_ms": _elapsed(), "source": None, "exhausted": exhausted}
            return {
                "status": "demo",
                "batch": batch,
                "error": None,
                "traceback": None,
                "config": redacted_config,
                "metadata": metadata,
            }

        message = "No streaming source configured. Provide 'source' or 'seed_rows', or enable demo mode."
        logger.warning("Streaming connector '%s' has no configured source", connector_name)
        return {
            "status": "not_configured",
            "batch": None,
            "error": message,
            "traceback": None,
            "config": redacted_config,
            "metadata": {"elapsed_ms": _elapsed(), "source": None, "exhausted": False},
        }

    message = f"Unsupported streaming source type '{source_type or 'unknown'}'"
    logger.warning("Streaming connector '%s' has unsupported source type '%s'", connector_name, source_type)
    return {
        "status": "not_configured",
        "batch": None,
        "error": message,
        "traceback": None,
        "config": redacted_config,
        "metadata": {"elapsed_ms": _elapsed(), "source": source_type or None, "exhausted": False},
    }


register_connector_driver("sql", _default_sql_driver)
register_connector_driver("rest", _default_rest_driver)
register_connector_driver("graphql", _default_graphql_driver)
register_connector_driver("grpc", _default_grpc_driver)
register_connector_driver("stream", _default_streaming_driver)
register_connector_driver("streaming", _default_streaming_driver)


def _transform_take(rows: List[Dict[str, Any]], options: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    limit = options.get("limit") or options.get("count") or 10
    try:
        limit_int = int(limit)
    except Exception:
        limit_int = 10
    return rows[: max(limit_int, 0)]


def _transform_select_columns(rows: List[Dict[str, Any]], options: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    columns = options.get("columns") or options.get("fields")
    if not columns:
        return rows
    if isinstance(columns, str):
        columns = [segment.strip() for segment in columns.split(",") if segment.strip()]
    selected: List[Dict[str, Any]] = []
    for row in rows:
        entry = {column: row.get(column) for column in columns}
        selected.append(entry)
    return selected


register_dataset_transform("take", _transform_take)
register_dataset_transform("limit", _transform_take)
register_dataset_transform("select", _transform_select_columns)

def get_model_spec(model_name: str) -> Dict[str, Any]:
    base = copy.deepcopy(MODEL_REGISTRY.get(model_name) or {})
    generated = MODELS.get(model_name) or {}
    registry_info = generated.get("registry") or {}

    if not base:
        base = {
            "type": generated.get("type", "unknown"),
            "framework": generated.get("engine") or generated.get("framework") or "unknown",
            "version": registry_info.get("version", "v1"),
            "metrics": registry_info.get("metrics", {}),
            "metadata": registry_info.get("metadata", {}),
        }
    else:
        base.setdefault("type", generated.get("type") or base.get("type") or "unknown")
        base.setdefault("framework", generated.get("engine") or base.get("framework") or "unknown")
        base.setdefault("version", registry_info.get("version") or base.get("version") or "v1")
        merged_metrics = dict(base.get("metrics") or {})
        merged_metrics.update(registry_info.get("metrics") or {})
        base["metrics"] = merged_metrics
        merged_metadata = dict(base.get("metadata") or {})
        merged_metadata.update(registry_info.get("metadata") or {})
        base["metadata"] = merged_metadata

    base.setdefault("metrics", {})
    base.setdefault("metadata", {})
    base["type"] = base.get("type") or generated.get("type") or "custom"
    base["framework"] = base.get("framework") or "unknown"
    base["version"] = base.get("version") or "v1"
    return base


def _load_model_instance(model_name: str, model_spec: Dict[str, Any]) -> Any:
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    framework = str(model_spec.get("framework") or "").lower()
    model_type = str(model_spec.get("type") or "").lower()
    metadata = model_spec.get("metadata") or {}
    custom_loader = None
    loader_path = metadata.get("loader")
    if loader_path:
        try:
            custom_loader = _load_python_callable(loader_path)
        except Exception as exc:  # pragma: no cover - loader import failure
            logger.exception("Failed to import custom loader for %s", model_name)
            raise RuntimeError(f"Custom loader for '{model_name}' could not be imported") from exc
    loader = (
        custom_loader
        or MODEL_LOADERS.get(framework)
        or MODEL_LOADERS.get(model_type)
        or MODEL_LOADERS.get("generic")
    )
    if loader is None:
        raise RuntimeError(
            f"No loader registered for model '{model_name}' (framework='{framework or 'unknown'}', type='{model_type or 'unknown'}')"
        )
    def _invoke_loader(func: Callable[..., Any]) -> Any:
        attempts: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = [
            ((model_name, model_spec), {}),
            ((model_spec,), {}),
        ]
        if isinstance(metadata, dict) and metadata:
            attempts.append(((metadata,), {}))
            attempts.append(((), {"config": metadata}))
        attempts.append(((), {}))
        attempts.append(((), {"model_name": model_name, "model_spec": model_spec}))

        last_signature_error: Optional[BaseException] = None
        for args, kwargs in attempts:
            try:
                return func(*args, **kwargs)
            except TypeError as exc:
                message = str(exc).lower()
                signature_issue = any(token in message for token in ("positional", "keyword", "argument"))
                if signature_issue:
                    last_signature_error = exc
                    continue
                raise
        if last_signature_error is not None:
            raise last_signature_error
        raise RuntimeError(f"Loader for '{model_name}' could not be invoked with available signatures")

    try:
        instance = _invoke_loader(loader)
    except Exception as exc:  # pragma: no cover - loader failure
        logger.exception("Model loader failed for %s", model_name)
        raise RuntimeError(f"Loader for '{model_name}' raised an error") from exc
    if isinstance(instance, dict) and instance.get("status") == "error":
        error_detail = instance.get("detail") or instance.get("error") or "unknown loader error"
        raise RuntimeError(f"Loader for '{model_name}' reported an error: {error_detail}")
    MODEL_CACHE[model_name] = instance
    return instance


def _coerce_output(raw_output: Any) -> Dict[str, Any]:
    """Normalise a raw runner payload into the canonical prediction schema."""

    result: Dict[str, Any] = {
        "score": None,
        "label": None,
        "scores": None,
        "raw": raw_output,
        "status": "ok",
        "error": None,
    }

    if raw_output is None:
        result["status"] = "partial"
        return result

    if isinstance(raw_output, Exception):
        result["status"] = "error"
        result["error"] = f"{type(raw_output).__name__}: {raw_output}"
        return result

    if isinstance(raw_output, dict):
        status_value = raw_output.get("status")
        if isinstance(status_value, str):
            normalized = status_value.lower()
            if normalized in {"error", "failed", "failure"}:
                result["status"] = "error"
            elif normalized in {"partial", "incomplete"}:
                result["status"] = "partial"
        error_value = raw_output.get("error") or raw_output.get("detail") or raw_output.get("message")
        if error_value:
            result["error"] = str(error_value)
            if result["status"] == "ok":
                result["status"] = "partial"

        candidates: List[Dict[str, Any]] = [raw_output]
        output_section = raw_output.get("output")
        if isinstance(output_section, dict):
            candidates.append(output_section)
        prediction_section = raw_output.get("prediction")
        if isinstance(prediction_section, dict):
            candidates.append(prediction_section)

        for candidate in candidates:
            if result["score"] is None:
                for key in ("score", "prob", "probability", "probabilities", "proba", "confidence"):
                    value = candidate.get(key)
                    coerced = _to_float(value)
                    if coerced is not None:
                        result["score"] = coerced
                        break
            if result["label"] is None:
                for key in ("label", "prediction", "class", "value"):
                    if key in candidate:
                        value = candidate[key]
                        if isinstance(value, (str, int, float)):
                            result["label"] = value
                            break
                        if isinstance(value, (list, tuple)) and value:
                            first = value[0]
                            if isinstance(first, (str, int, float)):
                                result["label"] = first
                                break
            if result["scores"] is None:
                for key in ("scores", "probabilities", "logits", "distribution"):
                    value = candidate.get(key)
                    if isinstance(value, (dict, list, tuple)):
                        result["scores"] = value
                        break

        if result["score"] is None and isinstance(result["scores"], dict):
            for key, value in result["scores"].items():
                numeric = _to_float(value)
                if numeric is not None:
                    result["score"] = numeric
                    if result["label"] is None and isinstance(key, (str, int)):
                        result["label"] = key
                    break

        if result["score"] is None and isinstance(result["scores"], (list, tuple)):
            for value in result["scores"]:
                numeric = _to_float(value)
                if numeric is not None:
                    result["score"] = numeric
                    break

        if result["label"] is None:
            label_candidate = raw_output.get("label") if isinstance(raw_output, dict) else None
            if isinstance(label_candidate, (str, int, float)):
                result["label"] = label_candidate

        if result["score"] is None and result["label"] is None and result["scores"] is None and result["error"] is None:
            result["status"] = "partial"

        return result

    if isinstance(raw_output, (list, tuple)):
        numeric_values = [_to_float(value) for value in raw_output]
        filtered = [value for value in numeric_values if value is not None]
        if filtered:
            result["score"] = filtered[0]
            result["scores"] = filtered
        else:
            result["scores"] = list(raw_output)
        result["status"] = "partial" if result["score"] is None else "ok"
        return result

    if isinstance(raw_output, (int, float)):
        result["score"] = float(raw_output)
        return result

    if isinstance(raw_output, str):
        result["label"] = raw_output
        result["status"] = "partial"
        return result

    result["status"] = "partial"
    return result


def _safe_numeric_explanations(
    model_name: str,
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Generate lightweight perturbation-based explanations when possible.

    The helper inspects numeric features in the input payload, perturbs each by
    ±ε (where ε = max(|value| * 0.01, 0.01)) and measures the local impact on
    the primary scalar prediction. It falls back gracefully when inputs are not
    numeric or when the runtime cannot be re-executed.
    """

    if not isinstance(payload, dict) or not payload:
        return None

    context = prediction.get("_explanation_context") if isinstance(prediction, dict) else None
    if not isinstance(context, dict):
        return None

    runner = context.get("runner")
    model_instance = context.get("model_instance")
    model_spec = context.get("model_spec")
    coerce = context.get("coerce")
    base_output = prediction.get("output") if isinstance(prediction, dict) else None

    if runner is None or model_instance is None or model_spec is None or coerce is None:
        return None
    if not isinstance(base_output, dict):
        return None

    numeric_features: Dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            numeric_features[key] = float(value)
    if not numeric_features:
        return None

    feature_names = sorted(numeric_features.keys())[:16]

    reference = _select_scalar_reference(base_output)
    if reference is None:
        return None

    local_impacts: List[Dict[str, Any]] = []
    global_importances: Dict[str, float] = {}

    def _invoke(adjusted_payload: Dict[str, Any]) -> Optional[float]:
        try:
            raw = runner(model_name, model_instance, adjusted_payload, model_spec)
        except Exception:
            logger.exception("Model runner failed during explanation for %s", model_name)
            return None
        coerced = coerce(raw)
        if coerced.get("status") == "error":
            return None
        return _extract_scalar_from_output(coerced, reference)

    for feature in feature_names:
        base_value = numeric_features[feature]
        epsilon = max(abs(base_value) * 0.01, 0.01)

        decreased_payload = dict(payload)
        increased_payload = dict(payload)
        decreased_payload[feature] = base_value - epsilon
        increased_payload[feature] = base_value + epsilon

        lower = _invoke(decreased_payload)
        upper = _invoke(increased_payload)
        if lower is None or upper is None:
            continue

        impact = (upper - lower) / (2 * epsilon)
        local_impacts.append({"feature": feature, "impact": impact})
        global_importances[feature] = abs(impact)

    if not local_impacts:
        return None

    return {
        "global_importances": global_importances,
        "local_explanations": local_impacts,
        "visualizations": {},
    }


def _default_stub_explanations(
    model_name: str,
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    output = prediction.get("output") if isinstance(prediction, dict) else None
    features: Dict[str, float] = {}
    if isinstance(output, dict):
        feature_map = output.get("features")
        if isinstance(feature_map, dict):
            for key, value in feature_map.items():
                if isinstance(value, (int, float)):
                    features[str(key)] = float(value)
    if not features and isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                features[str(key)] = float(value)
    if not features:
        features = {"feature_a": 0.5, "feature_b": 0.25}
    total = sum(abs(value) for value in features.values()) or 1.0
    global_importances = {
        key: round(abs(value) / total, 4)
        for key, value in features.items()
    }
    local_explanations = [
        {"feature": key, "impact": round(value, 4)}
        for key, value in features.items()
    ]
    return {
        "global_importances": global_importances,
        "local_explanations": local_explanations,
        "visualizations": {
            "saliency_map": "base64://image_classifier_saliency",
            "grad_cam": "base64://image_classifier_grad_cam",
            "attention": "base64://demo_attention_heatmap",
        },
    }


def _select_scalar_reference(output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    score = _to_float(output.get("score"))
    if score is not None:
        return {"kind": "score"}
    scores = output.get("scores")
    if isinstance(scores, dict):
        numeric_items = [
            (key, _to_float(value))
            for key, value in scores.items()
            if _to_float(value) is not None
        ]
        if numeric_items:
            key, _ = max(numeric_items, key=lambda item: item[1])
            return {"kind": "scores_dict", "key": key}
    if isinstance(scores, (list, tuple)):
        numeric_values = [_to_float(value) for value in scores]
        filtered = [value for value in numeric_values if value is not None]
        if filtered:
            index = numeric_values.index(filtered[0])
            return {"kind": "scores_index", "index": index}
    label = output.get("label")
    if isinstance(label, (str, int, float)):
        return {"kind": "label"}
    return None


def _extract_scalar_from_output(output: Dict[str, Any], reference: Dict[str, Any]) -> Optional[float]:
    kind = reference.get("kind") if isinstance(reference, dict) else None
    if kind == "score":
        return _to_float(output.get("score"))
    if kind == "scores_dict":
        scores = output.get("scores")
        key = reference.get("key")
        if isinstance(scores, dict) and key in scores:
            return _to_float(scores.get(key))
    if kind == "scores_index":
        scores = output.get("scores")
        index = reference.get("index")
        if isinstance(scores, (list, tuple)) and isinstance(index, int) and 0 <= index < len(scores):
            return _to_float(scores[index])
    if kind == "label":
        label_value = output.get("label")
        return _to_float(label_value)
    return _to_float(output.get("score"))


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _deterministic_stub_output(
    model_name: str,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Dict[str, Any]:
    numeric_inputs = {
        str(key): float(value)
        for key, value in (payload or {}).items()
        if isinstance(value, (int, float))
    }
    if not numeric_inputs:
        numeric_inputs = {"feature_a": 0.5, "feature_b": 0.25}
    weights = [0.3, 0.15, 0.1, 0.05, 0.03]
    baseline = 0.75
    score = baseline
    for index, (key, value) in enumerate(sorted(numeric_inputs.items())):
        weight = weights[index % len(weights)]
        score += value * weight
    score = round(score, 2)
    label = "Positive" if score >= 0 else "Negative"
    features = {key: round(value, 4) for key, value in numeric_inputs.items()}
    confidence = round(min(max(score / (abs(score) + 1.0), 0.0), 1.0), 4)
    return {
        "score": score,
        "label": label,
        "features": features,
        "confidence": confidence,
        "status": "ok",
    }


def explain_prediction(
    model_name: str,
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    framework = str(prediction.get("framework") or "").lower()
    model_name_key = str(prediction.get("model") or "").lower()
    metadata = prediction.get("spec_metadata") or {}
    custom_explainer = None
    explainer_path = metadata.get("explainer") if isinstance(metadata, dict) else None
    if explainer_path:
        try:
            custom_explainer = _load_python_callable(explainer_path)
        except Exception:  # pragma: no cover - explainer import failure
            logger.exception("Failed to import custom explainer for %s", model_name)
            custom_explainer = None
    explainer = (
        custom_explainer
        or MODEL_EXPLAINERS.get(framework)
        or MODEL_EXPLAINERS.get(model_name_key)
        or MODEL_EXPLAINERS.get("generic")
    )
    if explainer:
        try:
            value = explainer(model_name, payload, prediction)
            if isinstance(value, dict) and value:
                return value
        except Exception:  # pragma: no cover - explainer failure
            logger.exception("Model explainer failed for %s", model_name)
    fallback = _safe_numeric_explanations(model_name, payload, prediction)
    if fallback:
        return fallback
    try:
        allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")  # type: ignore[name-defined]
    except NameError:  # pragma: no cover - defensive when section loaded independently
        allow_stubs = False
    if allow_stubs:
        return _default_stub_explanations(model_name, payload, prediction)
    return None


def predict(model_name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute a model runner, normalise its output, and attach viable explanations.

    The function preserves the public signature while ensuring that no synthetic
    predictions are fabricated. Runner outputs are coerced into a canonical
    structure, and perturbation-based explanations are only produced for numeric
    inputs when no custom explainer is available.
    """

    from time import perf_counter

    inputs: Dict[str, Any]
    if isinstance(payload, dict):
        inputs = dict(payload)
    elif payload is None:
        inputs = {}
    else:
        inputs = {"value": payload}

    model_spec = get_model_spec(model_name)
    framework = model_spec.get("framework", "unknown")
    version = model_spec.get("version", "v1")

    metadata_payload: Dict[str, Any] = {
        "framework": framework,
        "version": version,
    }

    try:
        model_instance = _load_model_instance(model_name, model_spec)
    except Exception as exc:
        logger.exception("Failed to load model instance for %s", model_name)
        error_message = f"{type(exc).__name__}: {exc}"
        return {
            "model": model_name,
            "version": version,
            "framework": framework,
            "inputs": inputs,
            "input": inputs,
            "output": {
                "score": None,
                "label": None,
                "scores": None,
                "raw": None,
                "status": "error",
                "error": error_message,
            },
            "spec_metadata": model_spec.get("metadata") or {},
            "metadata": metadata_payload,
            "status": "error",
        }
    framework_key = framework.lower()
    model_type = str(model_spec.get("type") or "").lower()
    metadata = model_spec.get("metadata") or {}
    runner_callable = None
    runner_path = metadata.get("runner")
    if runner_path:
        try:
            runner_callable = _load_python_callable(runner_path)
        except Exception:  # pragma: no cover - runner import failure
            logger.exception("Failed to import custom runner for %s", model_name)
            runner_callable = None

    runner = (
        runner_callable
        or MODEL_RUNNERS.get(framework_key)
        or MODEL_RUNNERS.get(model_type)
        or MODEL_RUNNERS.get("generic")
    )

    runner_name = getattr(runner, "__name__", None) if runner else None
    if not runner_name and runner_callable is not None:
        runner_name = getattr(runner_callable, "__name__", None)
    if runner_name:
        metadata_payload["runner"] = runner_name

    overall_status = "ok"
    raw_output: Any = None
    runner_error: Optional[BaseException] = None
    timing_ms: Optional[float] = None

    if runner is None:
        overall_status = "error"
        output = {
            "score": None,
            "label": None,
            "scores": None,
            "raw": None,
            "status": "error",
            "error": "Runner not registered for model",
        }
    elif model_instance is None:
        overall_status = "error"
        output = {
            "score": None,
            "label": None,
            "scores": None,
            "raw": None,
            "status": "error",
            "error": "Model loader returned no instance",
        }
    else:
        start = perf_counter()
        try:
            raw_output = runner(model_name, model_instance, inputs, model_spec)
        except Exception as exc:  # pragma: no cover - runner failure guard
            runner_error = exc
            logger.exception("Model runner failed for %s", model_name)
        finally:
            elapsed = perf_counter() - start
            timing_ms = round(elapsed * 1000, 4)
            metadata_payload["timing_ms"] = timing_ms

        if runner_error is not None:
            overall_status = "error"
            output = {
                "score": None,
                "label": None,
                "scores": None,
                "raw": None,
                "status": "error",
                "error": f"{type(runner_error).__name__}: {runner_error}",
            }
        else:
            output = _coerce_output(raw_output)
            overall_status = output.get("status", "ok")

    explanations = None
    if isinstance(raw_output, dict):
        for container in (raw_output, raw_output.get("output") if isinstance(raw_output.get("output"), dict) else None):
            if not isinstance(container, dict):
                continue
            candidate = container.get("explanations")
            if isinstance(candidate, dict) and candidate:
                explanations = candidate
                break
            visuals = container.get("visualizations")
            if isinstance(visuals, dict) and visuals:
                explanations = {"visualizations": visuals}
                break

    result = {
        "model": model_name,
        "version": version,
        "framework": framework,
        "inputs": inputs,
        "input": inputs,
        "output": output,
        "spec_metadata": metadata,
        "metadata": {key: value for key, value in metadata_payload.items() if value is not None},
        "status": overall_status,
    }

    if explanations is None and runner_error is None and runner is not None and overall_status != "error":
        result["_explanation_context"] = {
            "runner": runner,
            "model_instance": model_instance,
            "model_spec": model_spec,
            "coerce": _coerce_output,
        }
        custom_explanations = explain_prediction(model_name, inputs, result)
        result.pop("_explanation_context", None)
        if isinstance(custom_explanations, dict) and custom_explanations:
            explanations = custom_explanations
    else:
        result.pop("_explanation_context", None)

    if explanations:
        result["explanations"] = explanations

    if overall_status == "error" and output.get("error") is None:
        output["error"] = "Prediction failed"

    return result


async def broadcast_page_snapshot(slug: str, payload: Dict[str, Any]) -> None:
    if not REALTIME_ENABLED:
        return
    message = {
        "type": "snapshot",
        "slug": slug,
        "payload": payload,
        "meta": _page_meta(slug),
    }
    await BROADCAST.broadcast(slug, _with_timestamp(message))


async def broadcast_component_update(
    slug: str,
    component_type: str,
    component_index: int,
    model: Any,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    if not REALTIME_ENABLED:
        return
    payload = _model_to_payload(model)
    message: Dict[str, Any] = {
        "type": "component",
        "slug": slug,
        "component_type": component_type,
        "component_index": component_index,
        "payload": payload,
        "meta": {"page": _page_meta(slug)},
    }
    if meta:
        message["meta"].update(meta)
    await BROADCAST.broadcast(slug, _with_timestamp(message))


async def broadcast_rollback(slug: str, component_index: int) -> None:
    if not REALTIME_ENABLED:
        return
    message = {
        "type": "rollback",
        "slug": slug,
        "component_index": component_index,
        "meta": {"page": _page_meta(slug)},
    }
    await BROADCAST.broadcast(slug, _with_timestamp(message))

from namel3ss.codegen.backend.core.runtime.insights import (
    evaluate_insights_for_dataset as _evaluate_insights_for_dataset_impl,
    run_insight as _run_insight_impl,
    evaluate_expression as _evaluate_expression_impl,
    resolve_expression_path as _resolve_expression_path_impl,
)
from namel3ss.codegen.backend.core.sql_compiler import compile_dataset_to_sql


def evaluate_insights_for_dataset(
    name: str,
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    return _evaluate_insights_for_dataset_impl(
        name,
        rows,
        context,
        insights=INSIGHTS,
        run_insight=_run_insight,
    )


def _run_insight(
    spec: Dict[str, Any],
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    return _run_insight_impl(
        spec,
        rows,
        context,
        model_registry=MODEL_REGISTRY,
        predict_callable=predict,
        evaluate_expression=_evaluate_expression,
        resolve_expression_path=_resolve_expression_path,
        render_template_value=_render_template_value,
    )


def _evaluate_expression(
    expression: Optional[str],
    rows: List[Dict[str, Any]],
    scope: Dict[str, Any],
) -> Any:
    return _evaluate_expression_impl(
        expression,
        rows,
        scope,
        resolve_expression_path=_resolve_expression_path,
    )


def _resolve_expression_path(
    expression: str,
    rows: List[Dict[str, Any]],
    scope: Dict[str, Any],
) -> Any:
    return _resolve_expression_path_impl(expression, rows, scope)



def evaluate_insight(slug: str, context: Optional[Dict[str, Any]] = None) -> InsightResponse:
    spec = INSIGHTS.get(slug)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Insight '{slug}' is not defined")
    ctx = dict(context or build_context(None))
    rows: List[Dict[str, Any]] = []
    result = evaluate_insights_for_dataset(slug, rows, ctx)
    dataset = result.get("dataset") or spec.get("source_dataset") or slug
    return InsightResponse(name=slug, dataset=dataset, result=result)

async def page_dashboard_0(session: Optional[AsyncSession] = None) -> Dict[str, Any]:
    context = build_context('dashboard')
    scope = ScopeFrame()
    scope.set('context', context)
    instructions = [{'type': 'table',
  'title': 'Orders',
  'source_type': 'dataset',
  'source': 'orders',
  'columns': [],
  'filter': None,
  'sort': None,
  'style': {},
  'layout': None,
  'insight': None,
  'dynamic_columns': None}]
    components = await render_statements(instructions, context, scope, session)
    return {
        'name': 'Dashboard',
        'route': '/',
        'slug': 'dashboard',
        'reactive': False,
        'refresh_policy': None,
        'components': components,
        'layout': {},
    }

PAGE_HANDLERS: Dict[str, Callable[[Optional[AsyncSession]], Awaitable[Dict[str, Any]]]] = {
    'dashboard': page_dashboard_0,
}
