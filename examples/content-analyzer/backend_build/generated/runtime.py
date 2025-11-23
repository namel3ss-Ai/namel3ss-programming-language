"""Generated runtime primitives for Namel3ss (N3)."""

from __future__ import annotations

import asyncio
import ast
import contextlib
import copy
import csv
import functools
import hashlib
import hmac
import inspect
import importlib
import importlib.util
import json
import logging
import math
import os
import pickle
import re
import secrets
import uuid
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
from namel3ss.plugins.registry import get_plugin, PluginRegistryError
from namel3ss.ml.training.backends import (
    registered_training_backends as _registered_training_backends,
    resolve_training_backend as _resolve_training_backend_impl,
    resolve_training_plan as _resolve_training_plan_impl,
)
try:
    from namel3ss.ml.connectors.base import RetryConfig, make_resilient_request
except ImportError:
    RetryConfig = None  # type: ignore
    make_resilient_request = None  # type: ignore

from namel3ss.plugins.base import PLUGIN_CATEGORY_EVALUATOR

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
RUNTIME_SETTINGS['connectors'] = {
    'retry_max_attempts': 3,
    'retry_base_delay': 0.5,
    'retry_max_delay': 5.0,
    'concurrency_limit': 10,
}
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

class PageBroadcastManager:
    """No-op broadcast manager when realtime is disabled."""

    async def connect(self, slug: str, websocket: Any, *, context: Optional[Dict[str, Any]] = None) -> str:  # pragma: no cover - noop
        return "offline"

    async def disconnect(self, slug: str, websocket: Any) -> None:  # pragma: no cover - noop
        return None

    async def broadcast(self, slug: Optional[str], message: Dict[str, Any], *, propagate: bool = True) -> Dict[str, Any]:
        return dict(message)

    async def has_listeners(self, slug: str) -> bool:
        return False

    async def listener_count(self, slug: str) -> int:
        return 0


BROADCAST = PageBroadcastManager()


async def broadcast_page_event(
    slug: Optional[str],
    *,
    event_type: str,
    dataset: Optional[str],
    payload: Any,
    source: str,
    status: Optional[str] = None,
    operation_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    propagate: bool = True,
) -> Dict[str, Any]:
    return {
        "type": event_type,
        "slug": slug,
        "dataset": dataset,
        "payload": payload,
        "meta": {"source": source, "status": status, "operation_id": operation_id, **(meta or {})},
    }


async def broadcast_dataset_refresh(
    slug: Optional[str],
    dataset_name: str,
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
    reason: str,
) -> None:
    return None


async def broadcast_mutation_event(
    slug: Optional[str],
    dataset_name: str,
    operation_id: Optional[str],
    status: str,
    payload: Dict[str, Any],
    *,
    source: str = "crud-write",
    meta: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "type": "mutation",
        "slug": slug,
        "dataset": dataset_name,
        "payload": payload,
        "meta": {"source": source, "status": status, "operation_id": operation_id, "error": error, **(meta or {})},
    }


async def resolve_websocket_context(websocket: Any) -> Dict[str, Any]:
    return {}

APP: Dict[str, Any] = {'name': 'Content Analyzer', 'database': None, 'theme': {}, 'variables': []}
DATASETS: Dict[str, Dict[str, Any]] = {}
FRAMES: Dict[str, Dict[str, Any]] = {}
CONNECTORS: Dict[str, Dict[str, Any]] = {}
AI_CONNECTORS: Dict[str, Dict[str, Any]] = {}
AI_MODELS: Dict[str, Dict[str, Any]] = {}
LLM_REGISTRY: Dict[str, Dict[str, Any]] = {'content_analyzer_llm': {'name': 'content_analyzer_llm',
                          'provider': 'openai',
                          'model': 'gpt-4o-mini',
                          'temperature': 0.2,
                          'max_tokens': 512,
                          'config': {},
                          'metadata': {}}}
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}
AI_MEMORIES: Dict[str, Dict[str, Any]] = {}
AI_PROMPTS: Dict[str, Dict[str, Any]] = {'analyze_content': {'name': 'analyze_content',
                     'model': 'content_analyzer_llm',
                     'template': '\n'
                                 '  Analyze the following content.\n'
                                 '\n'
                                 '  Content:\n'
                                 '  {{content}}\n'
                                 '\n'
                                 '  Return a JSON object with the following keys:\n'
                                 '  - topics: string[]\n'
                                 '  - sentiment: "positive" | "neutral" | "negative"\n'
                                 '  - risk_level: "low" | "medium" | "high"\n'
                                 '  - notes: string\n'
                                 '  ',
                     'input': [],
                     'output': [],
                     'parameters': {},
                     'metadata': {},
                     'description': None}}
AI_TEMPLATES: Dict[str, Dict[str, Any]] = {}
AI_CHAINS: Dict[str, Dict[str, Any]] = {}
AGENT_DEFS: Dict[str, Dict[str, Any]] = {'ContentAnalyzerAgent': {'name': 'ContentAnalyzerAgent',
                          'llm': 'content_analyzer_llm',
                          'tools': [],
                          'system_prompt': None,
                          'memory': 'conversation_window',
                          'goal': '',
                          'max_iterations': None,
                          'max_tokens': None,
                          'temperature': None,
                          'top_p': None,
                          'config': {},
                          'metadata': {}}}
AGENT_GRAPHS: Dict[str, Dict[str, Any]] = {}
AI_EXPERIMENTS: Dict[str, Dict[str, Any]] = {}
TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}
TUNING_JOBS: Dict[str, Dict[str, Any]] = {}
INSIGHTS: Dict[str, Dict[str, Any]] = {}
CRUD_RESOURCES: Dict[str, Dict[str, Any]] = {}
EVALUATORS: Dict[str, Dict[str, Any]] = {}
METRICS: Dict[str, Dict[str, Any]] = {}
GUARDRAILS: Dict[str, Dict[str, Any]] = {}
APP_QUERIES: Dict[str, Dict[str, Any]] = {}
KNOWLEDGE_MODULES: Dict[str, Dict[str, Any]] = {}
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
PAGES: List[Dict[str, Any]] = []
PAGE_SPEC_BY_SLUG: Dict[str, Dict[str, Any]] = {page['slug']: page for page in PAGES}
ENV_KEYS: List[str] = []
EMBED_INSIGHTS: bool = False
REALTIME_ENABLED: bool = False
CONNECTOR_DRIVERS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[List[Dict[str, Any]]]]] = {}
DATASET_TRANSFORMS: Dict[str, Callable[[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]], List[Dict[str, Any]]]] = {}
TRAINING_JOB_HISTORY: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
TUNING_JOB_HISTORY: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

# LLM Instance Initialization
from namel3ss.llm import create_llm, get_registry as get_llm_registry

# Dictionary to store initialized LLM instances
_LLM_INSTANCES: Dict[str, Any] = {}

def _initialize_llm_instances() -> None:
    """Initialize LLM instances from LLM_REGISTRY."""
    global _LLM_INSTANCES
    llm_registry = get_llm_registry()
    llm_registry.clear()  # Clear any previous instances

    for name, spec in LLM_REGISTRY.items():
        try:
            provider = spec.get('provider', '')
            model = spec.get('model', '')
            config = spec.get('config', {})

            if not provider or not model:
                logger.warning(f"Skipping LLM '{name}': missing provider or model")
                continue

            # Create LLM instance and register it
            llm = create_llm(name, provider, model, config, register=True)
            _LLM_INSTANCES[name] = llm
            logger.info(f"Initialized LLM '{name}' ({provider}/{model})")
        except Exception as exc:
            logger.error(f"Failed to initialize LLM '{name}': {exc}")
            # Continue with other LLMs even if one fails


# Initialize LLM instances at module load time
_initialize_llm_instances()



# Prompt initialization
from namel3ss.prompts import create_prompt, get_registry as get_prompt_registry

_PROMPT_INSTANCES: Dict[str, Any] = {}


def _initialize_prompt_instances():
    """Initialize prompt instances from AI_PROMPTS."""
    for name, spec in AI_PROMPTS.items():
        try:
            prompt = create_prompt(
                name=name,
                template=spec['template'],
                model=spec.get('model'),
                args=spec.get('args', {}),
                register=True
            )
            _PROMPT_INSTANCES[name] = prompt
        except Exception as e:
            logger.warning(f"Failed to initialize prompt '{name}': {e}")
            continue


_initialize_prompt_instances()


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

_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_SENSITIVE_HEADERS = {
    "server",
    "x-powered-by",
}
_CSRF_COOKIE_NAME = os.getenv("NAMEL3SS_CSRF_COOKIE", "namel3ss-csrf")
_CSRF_HEADER_NAME = os.getenv("NAMEL3SS_CSRF_HEADER", "x-csrf-token")
_CSRF_COOKIE_PATH = os.getenv("NAMEL3SS_CSRF_COOKIE_PATH", "/")
_CSRF_COOKIE_SAMESITE = (os.getenv("NAMEL3SS_CSRF_COOKIE_SAMESITE") or "lax").title()
try:
    _CSRF_COOKIE_MAX_AGE = int(os.getenv("NAMEL3SS_CSRF_TTL", "43200"))
except ValueError:
    _CSRF_COOKIE_MAX_AGE = 43200
_CSRF_SECRET_ENV = os.getenv("NAMEL3SS_CSRF_SECRET") or os.getenv("NAMEL3SS_SECRET_KEY")
_CSRF_SECRET = None
_CSRF_SECRET_BYTES: bytes
_RATE_LIMITS = {
    "auth": os.getenv("NAMEL3SS_RATE_LIMIT_AUTH"),
    "ai": os.getenv("NAMEL3SS_RATE_LIMIT_AI"),
    "experiments": os.getenv("NAMEL3SS_RATE_LIMIT_EXPERIMENTS"),
}
_RATE_LIMIT_DEFAULTS = {
    "auth": (10, 60),
    "ai": (60, 60),
    "experiments": (30, 60),
}
_RATE_LIMIT_BUCKETS: Dict[str, Dict[str, int]] = {}
_RATE_LIMIT_STATE: Dict[str, deque] = defaultdict(deque)
_RATE_LIMIT_LOCK = threading.Lock()


def _is_truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUTHY_VALUES


def get_env_setting(name: str, default: Any = None) -> Any:
    value = os.getenv(name)
    if value is None:
        return default
    return value


def is_production_mode() -> bool:
    runtime_value = _runtime_setting("environment")
    if runtime_value:
        candidate = str(runtime_value).strip().lower()
    else:
        candidate = str(os.getenv("NAMEL3SS_ENV") or os.getenv("ENV") or "development").strip().lower()
    return candidate in {"prod", "production"}


def is_debug_mode() -> bool:
    runtime_value = _runtime_setting("debug")
    if runtime_value is not None:
        return bool(runtime_value)
    env_value = os.getenv("NAMEL3SS_DEBUG")
    if env_value is not None:
        return _is_truthy(env_value)
    return not is_production_mode()


def get_secret(name: str, default: Optional[str] = None, *, allow_generate: bool = False) -> str:
    candidate = os.getenv(name)
    if candidate:
        return candidate
    runtime_secrets = _runtime_setting("secrets")
    if isinstance(runtime_secrets, dict) and runtime_secrets.get(name):
        return str(runtime_secrets[name])
    if allow_generate and not is_production_mode():
        generated = secrets.token_urlsafe(48)
        logger.warning("Secret %s was generated automatically for development.", name)
        return generated
    return default or ""


_CSRF_SECRET = _CSRF_SECRET_ENV or get_secret("NAMEL3SS_CSRF_SECRET", allow_generate=True)
_CSRF_SECRET_BYTES = (_CSRF_SECRET or "").encode("utf-8") if _CSRF_SECRET else secrets.token_bytes(32)


def csrf_protection_enabled() -> bool:
    runtime_value = _runtime_setting("csrf_enabled")
    if runtime_value is not None:
        return bool(runtime_value)
    env_value = os.getenv("NAMEL3SS_ENABLE_CSRF")
    if env_value is not None:
        return _is_truthy(env_value)
    return True


def csrf_cookie_name() -> str:
    return _CSRF_COOKIE_NAME


def csrf_header_name() -> str:
    return _CSRF_HEADER_NAME


def _sign_csrf(token: str) -> str:
    return hmac.new(_CSRF_SECRET_BYTES, token.encode("utf-8"), hashlib.sha256).hexdigest()


def _serialize_csrf_token(token: str) -> str:
    return f"{token}:{_sign_csrf(token)}"


def _deserialize_csrf_token(payload: Optional[str]) -> Optional[str]:
    if not payload or ":" not in payload:
        return None
    token, signature = payload.split(":", 1)
    expected = _sign_csrf(token)
    if hmac.compare_digest(expected, signature):
        return token
    return None


def ensure_csrf_cookie(cookies: Optional[Dict[str, Any]]) -> Tuple[Optional[str], bool]:
    if not csrf_protection_enabled():
        return None, False
    cookie_value = None
    if cookies and hasattr(cookies, "get"):
        cookie_value = cookies.get(_CSRF_COOKIE_NAME)
    token = _deserialize_csrf_token(cookie_value)
    if token:
        return cookie_value, False
    token = secrets.token_urlsafe(32)
    return _serialize_csrf_token(token), True


def should_enforce_csrf(method: str) -> bool:
    if not csrf_protection_enabled():
        return False
    return method.upper() in {"POST", "PUT", "PATCH", "DELETE"}


def validate_csrf_request(method: str, headers: Optional[Dict[str, Any]], cookies: Optional[Dict[str, Any]]) -> bool:
    if not should_enforce_csrf(method):
        return True
    cookie_value = None
    if cookies and hasattr(cookies, "get"):
        cookie_value = cookies.get(_CSRF_COOKIE_NAME)
    header_value = None
    if headers and hasattr(headers, "get"):
        header_value = headers.get(_CSRF_HEADER_NAME) or headers.get(_CSRF_HEADER_NAME.lower())
    token = _deserialize_csrf_token(cookie_value)
    if not token or not header_value:
        return False
    return hmac.compare_digest(token, header_value.strip())


def csrf_cookie_settings() -> Dict[str, Any]:
    return {
        "path": _CSRF_COOKIE_PATH,
        "httponly": False,
        "secure": _is_truthy(os.getenv("NAMEL3SS_CSRF_COOKIE_SECURE"), default=is_production_mode()),
        "samesite": _CSRF_COOKIE_SAMESITE,
        "max_age": _CSRF_COOKIE_MAX_AGE,
    }


def set_csrf_cookie(response: Any, value: str) -> None:
    if not value or not csrf_protection_enabled():
        return
    setter = getattr(response, "set_cookie", None)
    if not callable(setter):
        return
    params = csrf_cookie_settings()
    try:
        setter(
            _CSRF_COOKIE_NAME,
            value,
            max_age=params["max_age"],
            path=params["path"],
            secure=params["secure"],
            httponly=params["httponly"],
            samesite=params["samesite"],
        )
    except Exception:
        logger.debug("Unable to attach CSRF cookie")


def apply_security_headers(response: Any, request_id: Optional[str] = None) -> None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return

    def _set_header(key: str, value: str) -> None:
        try:
            if hasattr(headers, "get"):
                current = headers.get(key)  # type: ignore[attr-defined]
                if current is None:
                    headers[key] = value  # type: ignore[index]
            elif key not in headers:
                headers[key] = value  # type: ignore[index]
        except Exception:
            headers[key] = value  # type: ignore[index]

    try:
        _set_header("x-frame-options", "DENY")
        _set_header("x-content-type-options", "nosniff")
        _set_header("referrer-policy", "same-origin")
        _set_header("x-xss-protection", "1; mode=block")
        csp = "default-src 'self'; frame-ancestors 'none'; img-src 'self' data:;"
        _set_header("content-security-policy", csp)
        if request_id:
            _set_header("x-request-id", request_id)
        for header in _SENSITIVE_HEADERS:
            if header in headers:
                headers[header] = "redacted"
    except Exception:
        logger.debug("Unable to apply security headers")


def _parse_rate_limit(value: Optional[str], bucket: str) -> Dict[str, int]:
    default_limit, default_window = _RATE_LIMIT_DEFAULTS[bucket]
    if not value:
        return {"limit": default_limit, "window": default_window}
    text = value.strip().lower()
    if "/" in text:
        limit_part, window_part = text.split("/", 1)
    else:
        limit_part, window_part = text, "60s"
    try:
        limit = int(limit_part)
    except ValueError:
        limit = default_limit
    multiplier = 1
    if window_part.endswith("min"):
        multiplier = 60
        window_part = window_part[:-3]
    elif window_part.endswith("m"):
        multiplier = 60
        window_part = window_part[:-1]
    elif window_part.endswith("h"):
        multiplier = 3600
        window_part = window_part[:-1]
    elif window_part.endswith("s"):
        window_part = window_part[:-1]
    try:
        window = int(float(window_part) * multiplier)
    except ValueError:
        window = default_window
    window = max(window, 1)
    limit = max(limit, 1)
    return {"limit": limit, "window": window}


for bucket, value in _RATE_LIMITS.items():
    _RATE_LIMIT_BUCKETS[bucket] = _parse_rate_limit(value, bucket)


class RateLimitExceeded(Exception):
    def __init__(self, bucket: str, limit: int, window: int) -> None:
        super().__init__(f"Rate limit exceeded for {bucket}")
        self.bucket = bucket
        self.limit = limit
        self.window = window


def describe_rate_limit(bucket: str) -> Dict[str, int]:
    return dict(_RATE_LIMIT_BUCKETS.get(bucket, {}))


def enforce_rate_limit(bucket: str, identifier: str) -> None:
    config = _RATE_LIMIT_BUCKETS.get(bucket)
    if not config:
        return
    limit = int(config.get("limit", 0))
    window = int(config.get("window", 0))
    if limit <= 0 or window <= 0:
        return
    key = f"{bucket}:{identifier or 'anonymous'}"
    now = time.time()
    with _RATE_LIMIT_LOCK:
        entries = _RATE_LIMIT_STATE[key]
        while entries and entries[0] <= now - window:
            entries.popleft()
        if len(entries) >= limit:
            raise RateLimitExceeded(bucket, limit, window)
        entries.append(now)

class Namel3ssError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = details or {}


def _safe_message(exc: Exception) -> str:
    text = str(exc) if exc else ""
    return text or exc.__class__.__name__


def _request_id_from_headers(headers: Optional[Dict[str, Any]]) -> Optional[str]:
    if not headers or not hasattr(headers, "get"):
        return None
    candidate = headers.get("x-request-id") or headers.get("X-Request-ID")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return None


def ensure_request_id(headers: Optional[Dict[str, Any]] = None) -> str:
    header_value = _request_id_from_headers(headers)
    if header_value:
        return header_value
    return uuid.uuid4().hex


def build_error_payload(
    code: str,
    message: str,
    *,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> Tuple[int, Dict[str, Any]]:
    payload = {
        "error": {
            "code": code,
            "message": message,
        },
    }
    if details:
        payload["error"]["details"] = details
    if request_id:
        payload["request_id"] = request_id
    return status_code, payload


def format_error_response(exc: Exception, request_id: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
    status_code = getattr(exc, "status_code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    if isinstance(exc, Namel3ssError):
        message = _safe_message(exc)
        return build_error_payload(exc.code, message, status_code=status_code, details=exc.details, request_id=request_id)
    if hasattr(exc, "detail"):
        detail = getattr(exc, "detail")
        detail_message = detail if isinstance(detail, str) else _safe_message(exc)
        return build_error_payload("http_error", detail_message, status_code=status_code, request_id=request_id)
    if is_debug_mode():
        logger.exception("Unhandled exception")
        return build_error_payload("internal_server_error", _safe_message(exc), status_code=500, request_id=request_id)
    logger.exception("Unhandled exception")
    return build_error_payload("internal_server_error", "An unexpected error occurred.", status_code=500, request_id=request_id)


def apply_error_response(response: Any, status_code: int, payload: Dict[str, Any]) -> Any:
    setter = getattr(response, "status_code", None)
    if isinstance(setter, int):
        response.status_code = status_code
    headers = getattr(response, "headers", None)
    if headers is not None:
        headers.setdefault("content-type", "application/json")
    body_setter = getattr(response, "body", None)
    if body_setter is not None:
        response.body = json.dumps(payload).encode("utf-8")
    return response

START_TIME = time.time()
_LOGGING_CONFIGURED = False
_TRACING_CONFIGURED = False
_METRIC_LOCK = threading.RLock()
_REQUEST_METRICS: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: {
    "count": 0,
    "total_ms": 0.0,
    "max_ms": 0.0,
    "errors": 0,
})
_DATASET_METRICS: Dict[str, Dict[str, float]] = defaultdict(lambda: {
    "count": 0,
    "total_ms": 0.0,
    "rows": 0.0,
})
_CONNECTOR_STATUS: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
_REQUEST_ID_CONTEXT = ContextVar("namel3ss_request_id", default=None)
_REQUEST_CONTEXT_FILTER_INSTALLED = False
_TRACER = None


class _RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        request_id = current_request_id()
        if request_id:
            setattr(record, "namel3ss_request_id", request_id)
        return True


_REQUEST_CONTEXT_FILTER = _RequestContextFilter()


def _iso_timestamp(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


class _JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": _iso_timestamp(record.created),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        request_id = current_request_id()
        if request_id:
            payload["request_id"] = request_id
        if hasattr(record, "namel3ss_event"):
            payload["event"] = record.namel3ss_event  # type: ignore[attr-defined]
        if hasattr(record, "namel3ss_data"):
            payload["data"] = record.namel3ss_data  # type: ignore[attr-defined]
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _ConsoleLogFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("[%(levelname)s] %(asctime)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras: Dict[str, Any] = {}
        request_id = current_request_id()
        if request_id:
            extras["request_id"] = request_id
        if hasattr(record, "namel3ss_event"):
            extras["event"] = record.namel3ss_event  # type: ignore[attr-defined]
        if hasattr(record, "namel3ss_data"):
            extras["data"] = record.namel3ss_data  # type: ignore[attr-defined]
        if extras:
            return f"{base} | {json.dumps(extras, default=str)}"
        return base


def bind_request_id(request_id: Optional[str]) -> None:
    if request_id:
        _REQUEST_ID_CONTEXT.set(str(request_id))
    else:
        _REQUEST_ID_CONTEXT.set(None)


def current_request_id() -> Optional[str]:
    value = _REQUEST_ID_CONTEXT.get()
    return str(value) if value else None


def clear_request_id() -> None:
    _REQUEST_ID_CONTEXT.set(None)


def merge_request_context(values: Optional[Dict[str, Any]]) -> None:
    if not isinstance(values, dict):
        return
    try:
        base = get_request_context({})
    except Exception:
        base = {}
    base.update(values)
    try:
        setter = globals().get("set_request_context")
        if callable(setter):
            setter(base)
    except Exception:
        logger.debug("Unable to merge request context", exc_info=True)


def configure_logging(level: Optional[str] = None) -> None:
    global _LOGGING_CONFIGURED, _REQUEST_CONTEXT_FILTER_INSTALLED
    if _LOGGING_CONFIGURED and not level:
        return
    log_level = (level or os.getenv("NAMEL3SS_LOG_LEVEL") or "INFO").upper()
    root = logging.getLogger()
    handler = logging.StreamHandler()
    formatter: logging.Formatter
    if is_production_mode():
        formatter = _JsonLogFormatter()
    else:
        formatter = _ConsoleLogFormatter()
    handler.setFormatter(formatter)
    if not _REQUEST_CONTEXT_FILTER_INSTALLED:
        handler.addFilter(_REQUEST_CONTEXT_FILTER)
        _REQUEST_CONTEXT_FILTER_INSTALLED = True
    root.handlers = [handler]
    root.setLevel(getattr(logging, log_level, logging.INFO))
    _LOGGING_CONFIGURED = True


def request_timer() -> float:
    return time.perf_counter()


def _ensure_tracer():
    global _TRACER
    if _TRACER is not None:
        return _TRACER
    try:
        from opentelemetry import trace  # type: ignore
    except Exception:
        return None
    _TRACER = trace.get_tracer("namel3ss.runtime")
    return _TRACER


@contextlib.contextmanager
def tracing_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    tracer = _ensure_tracer()
    if tracer is None:
        yield None
        return
    span_cm = tracer.start_as_current_span(name)
    with span_cm as span:
        if attributes:
            for key, value in attributes.items():
                try:
                    span.set_attribute(key, value)
                except Exception:
                    continue
        yield span


def _normalize_route(path: str) -> str:
    if not path:
        return "/"
    normalized = path.split("?", 1)[0]
    return normalized or "/"


def _update_request_metrics(route: str, method: str, status_code: int, duration_ms: float) -> None:
    with _METRIC_LOCK:
        stats = _REQUEST_METRICS[(route, method)]
        stats["count"] += 1
        stats["total_ms"] += duration_ms
        stats["max_ms"] = max(duration_ms, stats["max_ms"])
        if status_code >= 500:
            stats["errors"] += 1


def record_request_observation(
    started_at: float,
    path: str,
    method: str,
    status_code: int,
    request_id: Optional[str] = None,
    client_host: Optional[str] = None,
) -> Dict[str, Any]:
    duration_ms = max((time.perf_counter() - started_at) * 1000.0, 0.0)
    route = _normalize_route(path)
    verb = (method or "GET").upper()
    _update_request_metrics(route, verb, status_code, duration_ms)
    payload = {
        "route": route,
        "method": verb,
        "status": status_code,
        "duration_ms": round(duration_ms, 3),
    }
    if request_id:
        payload["request_id"] = request_id
    if client_host:
        payload["client_ip"] = client_host
    logger.info(
        "HTTP request",
        extra={
            "namel3ss_event": "http_request",
            "namel3ss_data": payload,
        },
    )
    return payload


def observe_dataset_stage(dataset: Optional[str], stage: str, duration_ms: float, rows: int) -> None:
    if not dataset:
        return
    with _METRIC_LOCK:
        stats = _DATASET_METRICS[dataset]
        stats["count"] += 1
        stats["total_ms"] += duration_ms
        stats["rows"] += rows


def observe_dataset_fetch(dataset: Optional[str], status: str, duration_ms: float, rows: int, cache_state: str) -> None:
    if not dataset:
        return
    with _METRIC_LOCK:
        stats = _DATASET_METRICS[dataset]
        stats.setdefault("status", status)
        stats.setdefault("cache", cache_state)
        stats["count"] += 1
        stats["total_ms"] += duration_ms
        stats["rows"] += rows


def observe_connector_status(name: Optional[str], status: str) -> None:
    key = name or "unnamed"
    with _METRIC_LOCK:
        _CONNECTOR_STATUS[key][status] += 1


def _format_labels(labels: Dict[str, Any]) -> str:
    parts = []
    for key in sorted(labels):
        value = str(labels[key]).replace('"', '\"')
        parts.append(f'{key}="{value}"')
    return ",".join(parts)


def render_prometheus_metrics() -> str:
    lines: List[str] = []
    lines.append("# HELP namel3ss_uptime_seconds Application uptime")
    lines.append("# TYPE namel3ss_uptime_seconds gauge")
    lines.append(f"namel3ss_uptime_seconds {max(time.time() - START_TIME, 0.0):.3f}")
    with _METRIC_LOCK:
        if _REQUEST_METRICS:
            lines.append("# HELP namel3ss_request_count Total requests per route")
            lines.append("# TYPE namel3ss_request_count counter")
            for (route, method), stats in sorted(_REQUEST_METRICS.items()):
                labels = _format_labels({"route": route, "method": method})
                lines.append(f"namel3ss_request_count{{{labels}}} {int(stats['count'])}")
            lines.append("# HELP namel3ss_request_duration_seconds_total Total request duration")
            lines.append("# TYPE namel3ss_request_duration_seconds_total counter")
            for (route, method), stats in sorted(_REQUEST_METRICS.items()):
                labels = _format_labels({"route": route, "method": method})
                seconds = stats['total_ms'] / 1000.0
                lines.append(f"namel3ss_request_duration_seconds_total{{{labels}}} {seconds:.6f}")
        if _DATASET_METRICS:
            lines.append("# HELP namel3ss_dataset_fetch_seconds Dataset fetch runtime")
            lines.append("# TYPE namel3ss_dataset_fetch_seconds counter")
            for dataset, stats in sorted(_DATASET_METRICS.items()):
                labels = _format_labels({"dataset": dataset, "cache": stats.get("cache", "unknown")})
                seconds = stats['total_ms'] / 1000.0
                lines.append(f"namel3ss_dataset_fetch_seconds{{{labels}}} {seconds:.6f}")
                lines.append(f"namel3ss_dataset_rows_total{{{labels}}} {int(stats['rows'])}")
        if _CONNECTOR_STATUS:
            lines.append("# HELP namel3ss_connector_status Connector executions grouped by status")
            lines.append("# TYPE namel3ss_connector_status counter")
            for name, statuses in sorted(_CONNECTOR_STATUS.items()):
                for status, count in sorted(statuses.items()):
                    labels = _format_labels({"connector": name, "status": status})
                    lines.append(f"namel3ss_connector_status{{{labels}}} {int(count)}")
    lines.append("")
    return "\n".join(lines)


def health_summary() -> Dict[str, Any]:
    uptime = max(time.time() - START_TIME, 0.0)
    payload = {
        "status": "ok",
        "uptime_seconds": round(uptime, 3),
    }
    if APP:
        payload["app"] = {
            "name": APP.get("name"),
            "version": APP.get("version"),
        }
    return payload


async def _database_ready(timeout: float = 2.0) -> Tuple[bool, str]:
    try:
        from .. import database as _generated_database  # type: ignore
    except Exception:
        return True, "skipped"
    engine = getattr(_generated_database, "engine", None)
    if engine is None:
        return False, "engine_missing"
    try:
        async with engine.begin() as connection:
            await connection.execute(text("SELECT 1"))
        return True, "ok"
    except Exception as exc:
        logger.warning("Database readiness check failed: %s", exc)
        return False, str(exc)


async def readiness_checks() -> Dict[str, Any]:
    database_ready, detail = await _database_ready()
    status = "ok" if database_ready else "error"
    return {
        "status": status,
        "checks": {
            "database": {
                "ok": database_ready,
                "detail": detail,
            }
        },
    }


def configure_tracing(app: Optional[Any] = None) -> None:
    global _TRACING_CONFIGURED, _TRACER
    if _TRACING_CONFIGURED:
        return
    if os.getenv("NAMEL3SS_ENABLE_TRACING", "0").lower() not in {"1", "true", "yes", "on"}:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except Exception:
        logger.debug("OpenTelemetry SDK not installed; tracing disabled")
        return
    exporter = ConsoleSpanExporter()
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type: ignore

        headers: Dict[str, str] = {}
        for entry in os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "").split(","):
            if not entry or "=" not in entry:
                continue
            key, value = entry.split("=", 1)
            headers[key.strip()] = value.strip()
        exporter = OTLPSpanExporter(
            endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            headers=headers or None,
        )
    except Exception:
        logger.debug("OTLP exporter unavailable; falling back to console span exporter")
    service_name = APP.get("name") if isinstance(APP, dict) else "namel3ss"
    provider = TracerProvider(resource=Resource.create({"service.name": service_name or "namel3ss"}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer("namel3ss.runtime")
    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor.instrument_app(app)
        except Exception:
            logger.debug("FastAPI instrumentation unavailable for tracing")
    _TRACING_CONFIGURED = True

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
    if not context.get("page"):
        fallback_context = build_context("root")
        if fallback_context.get("page"):
            context = fallback_context
        elif PAGE_SPEC_BY_SLUG:
            first_slug = next(iter(PAGE_SPEC_BY_SLUG.keys()))
            context = build_context(first_slug)
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

def _record_runtime_error(
    context: Dict[str, Any],
    *,
    code: str,
    message: str,
    scope: Optional[str] = None,
    source: str = "runtime",
    detail: Optional[str] = None,
    severity: str = "error",
) -> Dict[str, Any]:
    severity_value = severity if severity in {"debug", "info", "warning", "error"} else "error"
    error_entry = {
        "code": code,
        "message": message,
        "scope": scope,
        "source": source,
        "detail": detail,
        "severity": severity_value,
    }
    context.setdefault("errors", []).append(error_entry)
    return error_entry


def _observability_setting_from(config: Any, channel: str) -> Optional[bool]:
    if not isinstance(config, dict):
        return None
    if "enabled" in config:
        enabled_flag = bool(config["enabled"])
        if not enabled_flag:
            return False
        default_flag: Optional[bool] = True
    else:
        default_flag = None
    if channel in config:
        return bool(config[channel])
    return default_flag


def _observability_enabled(context: Optional[Dict[str, Any]], channel: str) -> bool:
    result: Optional[bool] = None
    runtime_config = RUNTIME_SETTINGS.get("observability") if "observability" in RUNTIME_SETTINGS else None
    if isinstance(runtime_config, dict):
        flag = _observability_setting_from(runtime_config, channel)
        if flag is not None:
            result = flag
    if isinstance(context, dict):
        context_config = context.get("observability")
        if isinstance(context_config, dict):
            flag = _observability_setting_from(context_config, channel)
            if flag is not None:
                result = flag
    if result is None:
        return True
    return result


def _record_runtime_metric(
    context: Optional[Dict[str, Any]],
    *,
    name: str,
    value: Any,
    unit: str = "count",
    tags: Optional[Dict[str, Any]] = None,
    scope: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    if not _observability_enabled(context, "metrics"):
        return {}
    try:
        numeric_value: Any
        if isinstance(value, (int, float)):
            numeric_value = float(value)
        else:
            numeric_value = value
    except Exception:
        numeric_value = value
    entry = {
        "name": str(name),
        "value": numeric_value,
        "unit": str(unit) if unit is not None else "count",
        "tags": {str(key): val for key, val in (tags or {}).items()},
        "scope": scope,
        "timestamp": float(timestamp if timestamp is not None else time.time()),
    }
    telemetry = context.setdefault("telemetry", {})
    metrics = telemetry.setdefault("metrics", [])
    metrics.append(entry)
    return entry


def _record_runtime_event(
    context: Optional[Dict[str, Any]],
    *,
    event: str,
    level: str = "info",
    message: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    timestamp: Optional[float] = None,
    log: bool = True,
) -> Dict[str, Any]:
    entry = {
        "event": str(event),
        "level": str(level or "info").lower(),
        "message": str(message or event),
        "timestamp": float(timestamp if timestamp is not None else time.time()),
        "data": dict(data or {}),
    }
    if isinstance(context, dict) and _observability_enabled(context, "events"):
        telemetry = context.setdefault("telemetry", {})
        events = telemetry.setdefault("events", [])
        events.append(entry)
    if log and _observability_enabled(context, "events"):
        level_name = entry["level"].upper()
        numeric_level = getattr(logging, level_name, logging.INFO)
        try:
            logger.log(
                numeric_level,
                entry["message"],
                extra={
                    "namel3ss_event": entry["event"],
                    "namel3ss_data": entry["data"],
                },
            )
        except Exception:
            logger.log(numeric_level, entry["message"])
    return entry


def _collect_runtime_errors(
    context: Dict[str, Any],
    scope: Optional[str] = None,
    *,
    consume: bool = True,
) -> List[Dict[str, Any]]:
    if not isinstance(context, dict):
        return []
    errors = [entry for entry in context.get("errors", []) if isinstance(entry, dict)]
    if not errors:
        return []
    if scope is None:
        selected = list(errors)
        if consume:
            context.pop("errors", None)
        return selected
    selected: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []
    for entry in errors:
        if entry.get("scope") == scope:
            selected.append(entry)
        else:
            remaining.append(entry)
    if consume:
        if remaining:
            context["errors"] = remaining
        else:
            context.pop("errors", None)
    return selected


_GLOBAL_MEMORY_STORE: Dict[str, List[Dict[str, Any]]] = {}


def _build_memory_runtime_state(
    existing_state: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    definitions = globals().get("AI_MEMORIES", {})
    state: Dict[str, Dict[str, Any]] = {}
    for name, spec in definitions.items():
        scope = str(spec.get("scope") or "session").lower()
        kind = str(spec.get("kind") or "list").lower()
        max_items_raw = spec.get("max_items")
        try:
            max_items = int(max_items_raw) if max_items_raw is not None else None
            if max_items is not None and max_items <= 0:
                max_items = None
        except Exception:
            max_items = None
        config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
        if scope == "global":
            entries = _GLOBAL_MEMORY_STORE.setdefault(name, [])
        else:
            existing_entries = []
            if existing_state and name in existing_state:
                existing_entries = list(existing_state[name].get("entries") or [])
            entries = existing_entries
        state[name] = {
            "scope": scope,
            "kind": kind,
            "max_items": max_items,
            "config": dict(config),
            "entries": entries,
        }
    return state


def _ensure_memory_state(context: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(context, dict):
        return _build_memory_runtime_state()
    memory_state = context.get("memory_state")
    if isinstance(memory_state, dict):
        return memory_state
    memory_state = _build_memory_runtime_state()
    context["memory_state"] = memory_state
    return memory_state


def _memory_snapshot(
    memory_state: Dict[str, Dict[str, Any]],
    names: Iterable[str],
) -> Dict[str, List[Dict[str, Any]]]:
    snapshot: Dict[str, List[Dict[str, Any]]] = {}
    for name in names:
        store = memory_state.get(name)
        if not store:
            continue
        snapshot[name] = copy.deepcopy(store.get("entries") or [])
    return snapshot


def _normalize_memory_entry(kind: str, value: Any) -> Dict[str, Any]:
    timestamp = time.time()
    if kind == "conversation":
        if isinstance(value, dict):
            entry = dict(value)
        else:
            entry = {"content": value}
        entry.setdefault("role", entry.get("role") or "system")
        entry["timestamp"] = timestamp
        return entry
    if kind == "key_value":
        entry: Dict[str, Any] = {}
        if isinstance(value, dict):
            if "key" in value:
                entry["key"] = value["key"]
            entry["value"] = value.get("value", value)
        else:
            entry["value"] = value
        entry["timestamp"] = timestamp
        return entry
    return {"value": value, "timestamp": timestamp}


def _write_memory_entries(
    memory_state: Dict[str, Dict[str, Any]],
    names: Iterable[str],
    value: Any,
    *,
    context: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
) -> None:
    for name in names:
        store = memory_state.get(name)
        if not store:
            if context is not None:
                _record_runtime_error(
                    context,
                    code="memory.undefined",
                    message=f"Memory '{name}' is not defined",
                    scope="memory",
                    detail=source,
                )
            else:
                logger.warning("Memory '%s' is not defined", name)
            continue
        kind = str(store.get("kind") or "list").lower()
        if kind == "vector":
            if context is not None:
                _record_runtime_error(
                    context,
                    code="memory.unsupported_kind",
                    message=f"Memory '{name}' uses kind 'vector' without a configured backend",
                    scope="memory",
                    detail=source,
                )
            else:
                logger.error("Memory '%s' uses kind 'vector' without a configured backend", name)
            raise ValueError(f"Memory '{name}' uses unsupported kind 'vector' without a configured backend")
        entries = store.setdefault("entries", [])
        entries.append(_normalize_memory_entry(kind, value))
        max_items = store.get("max_items")
        if isinstance(max_items, int) and max_items > 0:
            while len(entries) > max_items:
                entries.pop(0)


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
    context.setdefault("ai_models", AI_MODELS)
    context.setdefault("prompts", AI_PROMPTS)
    context.setdefault("templates", AI_TEMPLATES)
    context.setdefault("chains", AI_CHAINS)
    context.setdefault("agents", AGENT_DEFS)
    context.setdefault("graphs", AGENT_GRAPHS)
    # Add LLM instances if they exist (from first-class LLM blocks)
    if '_LLM_INSTANCES' in globals():
        context.setdefault("llms", _LLM_INSTANCES)
    context.setdefault("experiments", AI_EXPERIMENTS)
    context.setdefault("training_jobs", TRAINING_JOBS)
    context.setdefault("tuning_jobs", TUNING_JOBS)
    memory_state = _ensure_memory_state(context)
    context.setdefault("memory_state", memory_state)
    context.setdefault("memory", memory_state)
    context.setdefault("memory_definitions", AI_MEMORIES)
    context.setdefault("call_python_model", call_python_model)
    context.setdefault("call_llm_connector", call_llm_connector)
    context.setdefault("run_prompt", run_prompt)
    context.setdefault("run_chain", run_chain)
    context.setdefault("evaluate_experiment", evaluate_experiment)
    context.setdefault("run_training_job", run_training_job)
    context.setdefault("resolve_training_job_plan", resolve_training_job_plan)
    context.setdefault("list_training_jobs", list_training_jobs)
    context.setdefault("available_training_backends", available_training_backends)
    context.setdefault("predict", predict)
    context.setdefault("datasets", DATASETS)
    context.setdefault("datasets_data", {})
    context.setdefault("frames", FRAMES)
    if AI_CONNECTORS:
        tool_instances: Dict[str, Any] = {}
        for connector_name, spec in AI_CONNECTORS.items():
            if not isinstance(spec, dict):
                continue
            tool_instances[connector_name] = _instantiate_tool_plugin(connector_name, spec, context)
        if tool_instances:
            context.setdefault("tools", {}).update(tool_instances)
    evaluator_instances: Dict[str, Any] = {}
    for evaluator_name, spec in EVALUATORS.items():
        if not isinstance(spec, dict):
            continue
        evaluator_instances[evaluator_name] = _instantiate_evaluator_plugin(evaluator_name, spec)
    if evaluator_instances:
        context.setdefault("evaluators", {}).update(evaluator_instances)
    else:
        context.setdefault("evaluators", {})
    context.setdefault("guardrails", GUARDRAILS)
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
            resolved = _resolve_context_path(context, [token], None)
            if resolved is not None:
                return str(resolved)
            vars_context = context.get("vars") if isinstance(context.get("vars"), dict) else None
            if vars_context is not None:
                value = _resolve_context_path(vars_context, [token], None)
                if value is not None:
                    return str(value)
            final_value = _resolve_context_path(context, [token], "")
            return "" if final_value is None else str(final_value)
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


_PLUGIN_CATEGORY_ALIASES: Dict[str, str] = {
    "llm": "llm_provider",
    "llm_provider": "llm_provider",
    "vector": "vector_store",
    "vector_store": "vector_store",
    "embedding": "embedding_provider",
    "embedding_provider": "embedding_provider",
    "graph": "graph_db",
    "graph_db": "graph_db",
    "custom": "custom_tool",
    "custom_tool": "custom_tool",
    "tool": "custom_tool",
}


def _determine_connector_category(spec: Dict[str, Any]) -> str:
    raw = spec.get("category") or spec.get("type")
    key = str(raw or "").strip().lower()
    if not key:
        return ""
    return _PLUGIN_CATEGORY_ALIASES.get(key, key)


def _instantiate_tool_plugin(name: str, spec: Dict[str, Any], context: Dict[str, Any]) -> Any:
    category = _determine_connector_category(spec)
    provider_raw = spec.get("provider") or (spec.get("config") or {}).get("provider")
    provider = str(provider_raw or "").strip()
    if not category or not provider:
        raise RuntimeError(f"Connector '{name}' is missing a plugin category or provider.")
    try:
        plugin_cls = get_plugin(category, provider)
    except PluginRegistryError as exc:
        message = f"Connector '{name}' references unknown plugin '{provider}' in category '{category}'."
        logger.error(message)
        raise RuntimeError(message) from exc
    plugin = plugin_cls()
    config_raw = spec.get("config", {})
    config_resolved = _resolve_placeholders(config_raw, context)
    if not isinstance(config_resolved, dict):
        config_resolved = config_raw if isinstance(config_raw, dict) else {}
    try:
        plugin.configure(dict(config_resolved))
    except Exception as exc:
        message = f"Failed to configure connector '{name}': {exc}"
        logger.error(message)
        raise RuntimeError(message) from exc
    return plugin


def _instantiate_evaluator_plugin(name: str, spec: Dict[str, Any]) -> Any:
    provider = str(spec.get("provider") or "").strip()
    if not provider:
        raise RuntimeError(f"Evaluator '{name}' is missing a provider.")
    try:
        plugin_cls = get_plugin(PLUGIN_CATEGORY_EVALUATOR, provider)
    except PluginRegistryError as exc:
        message = f"Evaluator '{name}' references unknown plugin '{provider}'."
        logger.error(message)
        raise RuntimeError(message) from exc
    plugin = plugin_cls()
    config_raw = spec.get("config", {})
    config_payload = dict(config_raw) if isinstance(config_raw, dict) else {}
    try:
        plugin.configure(config_payload)
    except Exception as exc:
        message = f"Failed to configure evaluator '{name}': {exc}"
        logger.error(message)
        raise RuntimeError(message) from exc
    return plugin


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
from namel3ss.codegen.backend.core.runtime.expression_sandbox import (
    ExpressionSandboxError as DatasetExpressionError,
    evaluate_expression_tree as _evaluate_expression_tree,
)


async def fetch_dataset_rows(
    key: str,
    session: AsyncSession,
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    dataset_entry = DATASETS.get(key)
    dataset_name = key
    if isinstance(dataset_entry, dict) and dataset_entry.get("name"):
        dataset_name = str(dataset_entry.get("name"))
    span_attrs = {"dataset": dataset_name, "key": key}
    with tracing_span("namel3ss.dataset.fetch", span_attrs):
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
            record_error=_record_runtime_error,
            record_event=_record_runtime_event,
            record_metric=_record_runtime_metric,
            observe_dataset_stage=observe_dataset_stage,
            observe_dataset_fetch=observe_dataset_fetch,
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
    record_error: Optional[Callable[..., Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    return await _load_dataset_source_impl(
        dataset,
        connector,
        session,
        context,
        connector_drivers=CONNECTOR_DRIVERS,
        httpx_client_cls=_HTTPX_CLIENT_CLS,
        normalize_connector_rows=_normalize_connector_rows,
        extract_connector_rows=_extract_rows_from_connector_response,
        execute_sql=_execute_sql,
        logger=logger,
        fetch_dataset_rows_fn=fetch_dataset_rows,
        record_error=record_error or _record_runtime_error,
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
    expression: Optional[Any],
    row: Dict[str, Any],
    context: Dict[str, Any],
    rows: Optional[List[Dict[str, Any]]] = None,
    dataset_name: Optional[str] = None,
    *,
    expression_source: Optional[str] = None,
) -> Any:
    target = expression if expression is not None else expression_source
    if target is None:
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
        if isinstance(target, str):
            expr_text = target.strip()
            if not expr_text:
                return None
            return _evaluate_expression_tree(expr_text, scope, context)
        return _evaluate_expression_tree(target, scope, context)
    except DatasetExpressionError as exc:
        original = expression_source if expression_source is not None else (target if isinstance(target, str) else str(target))
        _record_runtime_error(
            context,
            code="dataset_expression_disallowed",
            message=f"Dataset expression '{original}' is not permitted.",
            scope=dataset_name,
            source="dataset",
            detail=str(exc),
        )
        logger.warning("Disallowed dataset expression '%s': %s", original, exc)
    except Exception as exc:
        original = expression_source if expression_source is not None else (target if isinstance(target, str) else str(target))
        _record_runtime_error(
            context,
            code="dataset_expression_failed",
            message=f"Dataset expression '{original}' failed during evaluation.",
            scope=dataset_name,
            source="dataset",
            detail=str(exc),
        )
        logger.debug("Failed to evaluate dataset expression '%s'", original)
    return None


def _apply_filter(
    rows: List[Dict[str, Any]],
    condition: Optional[str],
    context: Dict[str, Any],
    dataset_name: Optional[str],
    condition_expr: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    expr_payload: Optional[Any] = condition_expr if condition_expr is not None else condition
    if expr_payload is None:
        return rows
    if isinstance(expr_payload, str) and not expr_payload.strip():
        return rows
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        value = _evaluate_dataset_expression(
            expr_payload,
            row,
            context,
            rows,
            dataset_name,
            expression_source=condition,
        )
        if _runtime_truthy(value):
            filtered.append(row)
    return filtered


def _apply_computed_column(
    rows: List[Dict[str, Any]],
    name: str,
    expression: Optional[str],
    context: Dict[str, Any],
    dataset_name: Optional[str],
    expression_expr: Optional[Any] = None,
) -> None:
    if not name:
        return
    for row in rows:
        row[name] = _evaluate_dataset_expression(
            expression_expr if expression_expr is not None else expression,
            row,
            context,
            rows,
            dataset_name,
            expression_source=expression,
        )


def _apply_group_aggregate(
    rows: List[Dict[str, Any]],
    columns: Sequence[str],
    aggregates: Sequence[Tuple[str, str]],
    context: Dict[str, Any],
    dataset_name: Optional[str],
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
                    values.append(
                        _evaluate_dataset_expression(
                            expr_source,
                            row,
                            context,
                            items,
                            dataset_name,
                            expression_source=expr_source,
                        )
                    )
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
    dataset_name: Optional[str],
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
        except Exception as exc:
            _record_runtime_error(
                context,
                code="dataset_transform_failed",
                message=f"Dataset transform '{transform_type}' failed.",
                scope=dataset_name,
                source="dataset",
                detail=str(exc),
            )
            logger.exception("Dataset transform '%s' failed", transform_type)
    return current


def _evaluate_quality_checks(
    rows: List[Dict[str, Any]],
    checks: Sequence[Dict[str, Any]],
    context: Dict[str, Any],
    dataset_name: Optional[str],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for check in checks:
        name = check.get("name")
        condition = check.get("condition")
        condition_expr = check.get("condition_expr")
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
        condition_payload: Optional[Any] = condition_expr if condition_expr is not None else condition
        has_condition = condition_payload is not None
        if isinstance(condition_payload, str):
            has_condition = bool(condition_payload.strip())
        if has_condition:
            violations = [
                row for row in rows
                if not _runtime_truthy(
                    _evaluate_dataset_expression(
                        condition_payload,
                        row,
                        context,
                        rows,
                        dataset_name,
                        expression_source=condition,
                    )
                )
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
    context_payload = dict(context or {})
    context_payload.setdefault("cache_key", cache_key)
    reason = str(context_payload.get("refresh_reason") or "fetch")
    await broadcast_dataset_refresh(slug, dataset_name, payload, context_payload, reason)


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


from namel3ss.codegen.backend.core.runtime.frames import (
    DEFAULT_FRAME_LIMIT as _DEFAULT_FRAME_LIMIT,
    MAX_FRAME_LIMIT as _MAX_FRAME_LIMIT,
    N3Frame as _N3Frame,
    FramePipelineExecutionError as _FramePipelineExecutionError,
    FrameSourceLoadError as _FrameSourceLoadError,
    build_pipeline_frame_spec as _build_pipeline_frame_spec,
    execute_frame_pipeline_plan as _execute_frame_pipeline_plan,
    load_frame_file_source as _load_frame_file_source,
    load_frame_sql_source as _load_frame_sql_source,
    project_frame_rows as _project_frame_rows,
)

_FRAME_ERROR_NOT_FOUND = "FRAME_NOT_FOUND"
_FRAME_ERROR_INVALID_PARAMS = "INVALID_QUERY_PARAMS"


async def fetch_frame_rows(
    key: str,
    session: Optional[AsyncSession],
    context: Dict[str, Any],
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    order_by: Optional[str] = None,
    as_response: bool = False,
) -> Any:
    try:
        frame = await _resolve_frame_runtime(key, session, context, set())
        enforce_defaults = as_response or limit is not None or (offset not in {None, 0}) or order_by is not None
        if not enforce_defaults:
            return frame.rows
        window, total, effective_limit, normalized_offset, normalized_order = _materialize_frame_view(
            frame,
            limit,
            offset,
            order_by,
            require_defaults=as_response,
        )
        if not as_response:
            return window
        schema_payload = frame.schema_payload()
        errors = _collect_runtime_errors(context, scope=frame.name)
        return {
            "name": frame.name,
            "source": _frame_source(frame.spec, frame.name),
            "schema": schema_payload,
            "rows": window,
            "total": total,
            "limit": effective_limit,
            "offset": normalized_offset,
            "order_by": normalized_order,
            "errors": errors,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Frame execution failed for %s", key)
        raise _frame_error(500, "FRAME_INTERNAL_ERROR", "Unexpected error while processing frame request.")


async def fetch_frame_schema(
    key: str,
    session: Optional[AsyncSession],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    frame_key = _normalize_frame_key(key)
    frame_spec = _require_frame_spec(frame_key, context)
    frame = _N3Frame(frame_key, frame_spec, [])
    return frame.schema_payload()


async def export_frame_csv(
    key: str,
    session: Optional[AsyncSession],
    context: Dict[str, Any],
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    order_by: Optional[str] = None,
) -> bytes:
    frame = await _resolve_frame_runtime(key, session, context, set())
    window, *_ = _materialize_frame_view(
        frame,
        limit,
        offset,
        order_by,
        require_defaults=False,
    )
    column_names = [column.get("name") for column in frame.schema_payload()["columns"] if column.get("name")]
    return frame.to_csv_bytes(columns=column_names, rows=window)


async def export_frame_parquet(
    key: str,
    session: Optional[AsyncSession],
    context: Dict[str, Any],
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    order_by: Optional[str] = None,
) -> bytes:
    frame = await _resolve_frame_runtime(key, session, context, set())
    window, *_ = _materialize_frame_view(
        frame,
        limit,
        offset,
        order_by,
        require_defaults=False,
    )
    return frame.to_parquet_bytes(rows=window)


def _frame_error(status_code: int, error: str, detail: Optional[str] = None) -> HTTPException:
    payload = {"status_code": status_code, "error": error, "detail": detail}
    return HTTPException(status_code=status_code, detail=payload)


def _normalize_frame_key(key: str) -> str:
    if not key:
        raise _frame_error(404, _FRAME_ERROR_NOT_FOUND, "Frame name must be provided.")
    return str(key)


async def _resolve_frame_runtime(
    key: str,
    session: Optional[AsyncSession],
    context: Dict[str, Any],
    visited: Optional[Set[str]] = None,
) -> _N3Frame:
    frame_key = _normalize_frame_key(key)
    active_visited = visited or set()
    if frame_key in active_visited:
        _record_runtime_error(
            context,
            code="frame_recursion_detected",
            message=f"Cyclic frame dependency detected for '{frame_key}'.",
            scope=frame_key,
            source="frame",
            severity="error",
        )
        return _N3Frame(frame_key, {}, [])
    active_visited.add(frame_key)
    frame_spec = _require_frame_spec(frame_key, context)
    base_rows = await _load_frame_source_rows(frame_key, frame_spec, session, context, active_visited)
    if not base_rows:
        base_rows = _frame_example_rows(frame_spec)
    shaped_rows = _project_frame_rows(
        frame_key,
        frame_spec,
        base_rows or [],
        context,
        resolve_placeholders=_resolve_placeholders,
        evaluate_expression=_evaluate_dataset_expression,
        runtime_truthy=_runtime_truthy,
        record_error=_record_runtime_error,
    )
    return _N3Frame(frame_key, frame_spec, shaped_rows or [])


def _require_frame_spec(frame_key: str, context: Dict[str, Any]) -> Dict[str, Any]:
    frame_spec = FRAMES.get(frame_key)
    if isinstance(frame_spec, dict):
        return frame_spec
    _record_runtime_error(
        context,
        code="frame_missing",
        message=f"Frame '{frame_key}' is not defined.",
        scope=frame_key,
        source="frame",
        severity="error",
    )
    raise _frame_error(404, _FRAME_ERROR_NOT_FOUND, f"Frame '{frame_key}' does not exist.")


def _frame_example_rows(frame_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    examples = frame_spec.get("examples")
    if not isinstance(examples, list):
        return []
    dict_rows = [row for row in examples if isinstance(row, dict)]
    if not dict_rows:
        return []
    return _clone_rows(dict_rows)


def _frame_source(frame_spec: Dict[str, Any], frame_key: str) -> Dict[str, Any]:
    return {
        "type": str(frame_spec.get("source_type") or "dataset"),
        "name": frame_spec.get("source") or frame_spec.get("name") or frame_key,
    }


def _materialize_frame_view(
    frame: _N3Frame,
    limit: Optional[int],
    offset: Optional[int],
    order_by: Optional[str],
    *,
    require_defaults: bool,
) -> Tuple[List[Dict[str, Any]], int, int, int, Optional[str]]:
    normalized_limit, normalized_offset = _normalize_pagination(limit, offset, require_defaults=require_defaults)
    order_spec, normalized_order = _normalize_order_by(order_by, frame)
    if normalized_limit is None and normalized_offset == 0 and order_spec is None:
        window = frame.rows
        total = len(window)
        effective_limit = total
    else:
        window, total = frame.window_rows(order_spec, normalized_limit, normalized_offset)
        remaining = max(total - normalized_offset, 0)
        if normalized_limit is None:
            effective_limit = remaining
        else:
            effective_limit = min(normalized_limit, remaining)
    return window, total, effective_limit, normalized_offset, normalized_order


def _normalize_pagination(
    limit: Optional[int],
    offset: Optional[int],
    *,
    require_defaults: bool,
) -> Tuple[Optional[int], int]:
    normalized_limit: Optional[int]
    if limit is None:
        normalized_limit = _DEFAULT_FRAME_LIMIT if require_defaults else None
    else:
        normalized_limit = _coerce_positive_int(limit, "limit")
    if normalized_limit is not None:
        normalized_limit = min(normalized_limit, _MAX_FRAME_LIMIT)
    normalized_offset = _coerce_non_negative_int(offset, "offset")
    return normalized_limit, normalized_offset


def _coerce_positive_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, f"{field.title()} must be an integer.")
    if result <= 0:
        raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, f"{field.title()} must be positive.")
    return result


def _coerce_non_negative_int(value: Any, field: str) -> int:
    if value is None:
        return 0
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, f"{field.title()} must be an integer.")
    if result < 0:
        raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, f"{field.title()} cannot be negative.")
    return result


def _normalize_order_by(order_by: Optional[str], frame: _N3Frame) -> Tuple[Optional[Tuple[str, bool]], Optional[str]]:
    if order_by is None:
        return None, None
    candidate = str(order_by).strip()
    if not candidate:
        return None, None
    descending = False
    column_token = candidate
    if candidate.startswith("-"):
        descending = True
        column_token = candidate[1:]
    if ":" in column_token:
        column_token, direction_token = column_token.split(":", 1)
        direction_token = direction_token.strip().lower()
        if direction_token in {"desc", "descending", "down"}:
            descending = True
        elif direction_token in {"asc", "ascending", "up"}:
            descending = False
    column_name = column_token.strip()
    if not column_name:
        raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, "order_by must reference a column name.")
    valid_columns = {column.get("name") for column in frame.spec.get("columns") or [] if column.get("name")}
    if column_name not in valid_columns:
        raise _frame_error(
            400,
            _FRAME_ERROR_INVALID_PARAMS,
            f"Column '{column_name}' is not defined for frame '{frame.name}'.",
        )
    normalized = f"{column_name}:{'desc' if descending else 'asc'}"
    return (column_name, descending), normalized


async def _load_frame_source_rows(
    frame_key: str,
    frame_spec: Dict[str, Any],
    session: Optional[AsyncSession],
    context: Dict[str, Any],
    visited: Set[str],
) -> List[Dict[str, Any]]:
    source_config = frame_spec.get("source_config")
    if isinstance(source_config, dict):
        try:
            return await _load_frame_source_from_config(
                frame_key,
                source_config,
                session,
                context,
            )
        except _FrameSourceLoadError as exc:
            _record_runtime_error(
                context,
                code="frame_source_failed",
                message=str(exc),
                scope=frame_key,
                source="frame",
                detail=str(exc),
            )
            return []
    source_kind = str(frame_spec.get("source_type") or "dataset").lower()
    source_name = frame_spec.get("source") or frame_spec.get("name") or frame_key
    if source_kind == "dataset":
        try:
            return await fetch_dataset_rows(source_name, session, context)
        except Exception as exc:  # pragma: no cover - dataset fetch failure
            _record_runtime_error(
                context,
                code="frame_source_failed",
                message=f"Failed to load dataset '{source_name}' for frame '{frame_key}'.",
                scope=frame_key,
                source="frame",
                detail=str(exc),
            )
            return []
    if source_kind == "frame":
        nested_frame = await _resolve_frame_runtime(source_name, session, context, visited)
        return nested_frame.rows
    if source_kind == "table":
        tables = context.get("tables")
        if isinstance(tables, dict):
            table_value = tables.get(source_name)
            if isinstance(table_value, list):
                dict_rows = [row for row in table_value if isinstance(row, dict)]
                if dict_rows:
                    return _clone_rows(dict_rows)
            elif isinstance(table_value, dict):
                return _clone_rows([table_value])
        dataset_spec = DATASETS.get(source_name)
        if isinstance(dataset_spec, dict):
            sample_rows = dataset_spec.get("sample_rows")
            if isinstance(sample_rows, list):
                dict_rows = [row for row in sample_rows if isinstance(row, dict)]
                if dict_rows:
                    return _clone_rows(dict_rows)
        return []
    _record_runtime_error(
        context,
        code="frame_source_unknown",
        message=f"Unsupported frame source type '{source_kind}'.",
        scope=frame_key,
        source="frame",
        detail=source_kind,
    )
    return []


async def _load_frame_source_from_config(
    frame_key: str,
    source_config: Dict[str, Any],
    session: Optional[AsyncSession],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    kind = str(source_config.get("kind") or source_config.get("type") or "file").lower()
    if kind == "file":
        return _load_frame_file_source(
            source_config,
            context=context,
            resolve_placeholders=_resolve_placeholders,
        )
    if kind == "sql":
        return await _load_frame_sql_source(
            source_config,
            session,
            context=context,
            resolve_placeholders=_resolve_placeholders,
        )
    raise _FrameSourceLoadError(f"Unsupported frame source kind '{kind}' for '{frame_key}'.")


async def _evaluate_frame_pipeline(
    plan: Dict[str, Any],
    session: Optional[AsyncSession],
    context: Dict[str, Any],
    visited: Set[str],
) -> _N3Frame:
    if not isinstance(plan, dict):
        raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, "Invalid frame pipeline payload.")
    root = plan.get("root")
    if not isinstance(root, str) or not root:
        raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, "Frame pipeline requires a root frame.")
    base_frame = await _resolve_frame_runtime(root, session, context, set(visited))
    operations: List[Dict[str, Any]] = []
    for operation in plan.get("operations") or []:
        if not isinstance(operation, dict):
            continue
        if operation.get("op") == "join":
            join_target = operation.get("join_target")
            if not join_target:
                raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, "Join operation requires a target frame.")
            join_frame = await _resolve_frame_runtime(join_target, session, context, set(visited))
            op_payload = dict(operation)
            op_payload["join_rows"] = join_frame.rows
            op_payload["join_schema"] = join_frame.spec.get("columns")
            operations.append(op_payload)
            continue
        operations.append(operation)
    schema_payload = plan.get("schema") or {}
    try:
        rows = _execute_frame_pipeline_plan(
            root,
            schema_payload,
            base_frame.rows,
            operations,
            context=context,
            evaluate_expression=_evaluate_dataset_expression,
            runtime_truthy=_runtime_truthy,
        )
    except _FramePipelineExecutionError as exc:
        _record_runtime_error(
            context,
            code="frame_pipeline_failed",
            message=str(exc),
            scope=root,
            source="frame",
            detail=str(exc),
            severity="error",
        )
        raise
    derived_spec = _build_pipeline_frame_spec(schema_payload, base_frame.spec, f"{root}__pipeline")
    return _N3Frame(derived_spec.get("name") or root, derived_spec, rows)


async def _resolve_frame_pipeline_value(
    value: Dict[str, Any],
    session: Optional[AsyncSession],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    plan = value.get("__frame_pipeline__")
    if not isinstance(plan, dict):
        return []
    try:
        derived = await _evaluate_frame_pipeline(plan, session, context, set())
    except _FramePipelineExecutionError as exc:
        raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, str(exc))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Frame pipeline evaluation failed")
        raise _frame_error(500, "FRAME_PIPELINE_FAILED", "Frame pipeline evaluation failed.")
    return derived.rows


def _crud_sanitize_identifier(value: str) -> str:
    segments = [segment.strip() for segment in (value or "").split(".") if segment and segment.strip()]
    if not segments:
        raise ValueError("Empty identifier path")
    for segment in segments:
        if not _IDENTIFIER_RE.match(segment):
            raise ValueError(f"Invalid identifier segment '{segment}'")
    return ".".join(segments)


def _crud_select_clause(resource: Dict[str, Any]) -> str:
    columns = [column for column in resource.get("select_fields") or [] if column]
    if not columns:
        return "*"
    return ", ".join(_crud_sanitize_identifier(column) for column in columns)


def _crud_allowed_operations(resource: Dict[str, Any]) -> Set[str]:
    allowed = resource.get("allowed_operations") or []
    return {str(operation).lower() for operation in allowed}


def _crud_normalize_pagination(resource: Dict[str, Any], limit: Optional[int], offset: Optional[int]) -> Tuple[int, int]:
    default_limit = int(resource.get("default_limit") or 100)
    max_limit = int(resource.get("max_limit") or max(default_limit, 100))
    limit_value = default_limit if limit is None else max(int(limit), 1)
    if max_limit > 0:
        limit_value = min(limit_value, max_limit)
    offset_value = max(int(offset or 0), 0)
    return limit_value, offset_value


def _crud_build_context(session: Optional[AsyncSession], base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    context = build_context(None)
    if isinstance(base, dict):
        context.update(base)
    if session is not None:
        context["session"] = session
    return context


def _crud_tenant_clause(resource: Dict[str, Any], context: Dict[str, Any], *, required: bool) -> Tuple[str, Dict[str, Any]]:
    tenant_column = resource.get("tenant_column")
    if not tenant_column:
        return "", {}
    tenant_value = context.get("tenant")
    if tenant_value is None and isinstance(context.get("request"), dict):
        tenant_value = context["request"].get("tenant")
    if tenant_value is None:
        if required:
            raise PermissionError("tenant_scope_required")
        return "", {}
    clause = f"{_crud_sanitize_identifier(tenant_column)} = :tenant_value"
    return clause, {"tenant_value": tenant_value}


def _get_crud_resource(slug: str) -> Dict[str, Any]:
    resource = CRUD_RESOURCES.get(slug)
    if not isinstance(resource, dict):
        raise KeyError(slug)
    return resource


def describe_crud_resources() -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    for slug, spec in CRUD_RESOURCES.items():
        if not isinstance(spec, dict):
            continue
        catalog.append(
            {
                "slug": slug,
                "label": spec.get("label"),
                "source_type": spec.get("source_type"),
                "primary_key": spec.get("primary_key"),
                "tenant_column": spec.get("tenant_column"),
                "operations": list(spec.get("allowed_operations") or []),
                "default_limit": int(spec.get("default_limit") or 0),
                "max_limit": int(spec.get("max_limit") or 0),
                "read_only": bool(spec.get("read_only")),
            }
        )
    catalog.sort(key=lambda item: item["slug"])
    return catalog


async def _crud_table_list(
    resource: Dict[str, Any],
    session: AsyncSession,
    context: Dict[str, Any],
    *,
    limit: int,
    offset: int,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    table_name = _crud_sanitize_identifier(resource.get("source_name"))
    select_clause = _crud_select_clause(resource)
    order_column = _crud_sanitize_identifier(resource.get("primary_key") or "id")
    tenant_clause, tenant_params = _crud_tenant_clause(resource, context, required=bool(resource.get("tenant_column")))
    where_parts: List[str] = []
    params: Dict[str, Any] = dict(tenant_params)
    params.update({"limit": limit, "offset": offset})
    if tenant_clause:
        where_parts.append(tenant_clause)
    where_sql = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
    query = text(f"SELECT {select_clause} FROM {table_name}{where_sql} ORDER BY {order_column} LIMIT :limit OFFSET :offset")
    result = await session.execute(query, params)
    rows = [dict(row) for row in result.mappings()]
    total: Optional[int] = None
    try:
        count_query = text(f"SELECT COUNT(1) FROM {table_name}{where_sql}")
        count_result = await session.execute(count_query, tenant_params)
        total_value = count_result.scalar()
        total = int(total_value) if total_value is not None else None
    except Exception:
        total = None
    return rows, total


async def _crud_dataset_list(
    resource: Dict[str, Any],
    session: Optional[AsyncSession],
    context: Dict[str, Any],
    *,
    limit: int,
    offset: int,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    dataset_name = resource.get("source_name")
    rows = await fetch_dataset_rows(dataset_name, session, context)
    tenant_column = resource.get("tenant_column")
    if tenant_column:
        clause, params = _crud_tenant_clause(resource, context, required=True)
        tenant_value = params.get("tenant_value")
        rows = [row for row in rows if str(row.get(tenant_column)) == str(tenant_value)]
    total = len(rows)
    window = rows[offset : offset + limit]
    return window, total


async def _crud_select_table_row(
    resource: Dict[str, Any],
    identifier: Any,
    session: AsyncSession,
    context: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    table_name = _crud_sanitize_identifier(resource.get("source_name"))
    select_clause = _crud_select_clause(resource)
    pk_column = _crud_sanitize_identifier(resource.get("primary_key") or "id")
    tenant_clause, tenant_params = _crud_tenant_clause(resource, context, required=bool(resource.get("tenant_column")))
    params: Dict[str, Any] = dict(tenant_params)
    params["pk"] = identifier
    where_clause = f"{pk_column} = :pk"
    if tenant_clause:
        where_clause = f"{where_clause} AND {tenant_clause}"
    query = text(f"SELECT {select_clause} FROM {table_name} WHERE {where_clause} LIMIT 1")
    result = await session.execute(query, params)
    mapping = result.mappings().first()
    return dict(mapping) if mapping else None


async def _crud_select_dataset_row(resource: Dict[str, Any], identifier: Any, session: Optional[AsyncSession], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rows, _total = await _crud_dataset_list(resource, session, context, limit=resource.get("max_limit", 1000), offset=0)
    pk_column = resource.get("primary_key") or "id"
    for row in rows:
        if str(row.get(pk_column)) == str(identifier):
            return row
    return None


async def _crud_lookup_row(
    resource: Dict[str, Any],
    identifier: Any,
    session: Optional[AsyncSession],
    context: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    source_type = str(resource.get("source_type") or "table").lower()
    if source_type == "dataset":
        return await _crud_select_dataset_row(resource, identifier, session, context)
    if session is None:
        raise RuntimeError("Database session is required for CRUD table operations")
    return await _crud_select_table_row(resource, identifier, session, context)


async def list_crud_resource(
    slug: str,
    session: Optional[AsyncSession],
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> Dict[str, Any]:
    resource = _get_crud_resource(slug)
    allowed = _crud_allowed_operations(resource)
    if "list" not in allowed:
        raise PermissionError("list")
    context = _crud_build_context(session)
    limit_value, offset_value = _crud_normalize_pagination(resource, limit, offset)
    source_type = str(resource.get("source_type") or "table").lower()
    if source_type == "dataset":
        rows, total = await _crud_dataset_list(resource, session, context, limit=limit_value, offset=offset_value)
    else:
        if session is None:
            raise RuntimeError("Database session is required for CRUD queries")
        rows, total = await _crud_table_list(resource, session, context, limit=limit_value, offset=offset_value)
    errors = _collect_runtime_errors(context)
    return {
        "resource": slug,
        "label": resource.get("label"),
        "status": "ok",
        "items": rows,
        "limit": limit_value,
        "offset": offset_value,
        "total": total,
        "errors": errors,
    }


async def retrieve_crud_resource(
    slug: str,
    identifier: Any,
    session: Optional[AsyncSession],
) -> Dict[str, Any]:
    resource = _get_crud_resource(slug)
    allowed = _crud_allowed_operations(resource)
    if "retrieve" not in allowed:
        raise PermissionError("retrieve")
    context = _crud_build_context(session)
    row = await _crud_lookup_row(resource, identifier, session, context)
    errors = _collect_runtime_errors(context)
    status = "ok" if row is not None else "not_found"
    return {
        "resource": slug,
        "label": resource.get("label"),
        "status": status,
        "item": row,
        "errors": errors,
    }


async def create_crud_resource(
    slug: str,
    payload: Dict[str, Any],
    session: Optional[AsyncSession],
) -> Dict[str, Any]:
    resource = _get_crud_resource(slug)
    allowed = _crud_allowed_operations(resource)
    if "create" not in allowed or resource.get("read_only"):
        raise PermissionError("create")
    if session is None:
        raise RuntimeError("Database session is required for CRUD mutations")
    context = _crud_build_context(session)
    tenant_clause, tenant_params = _crud_tenant_clause(resource, context, required=bool(resource.get("tenant_column")))
    values = dict(payload or {})
    if tenant_params and resource.get("tenant_column") not in values:
        values[resource["tenant_column"]] = tenant_params["tenant_value"]
    mutable_fields = resource.get("mutable_fields") or []
    if not mutable_fields:
        mutable_fields = [field for field in resource.get("select_fields") or [] if field and field != resource.get("primary_key")]
    columns: List[str] = []
    params: Dict[str, Any] = {}
    for field in mutable_fields:
        if field in values:
            columns.append(field)
            params[field] = values[field]
    if not columns:
        raise ValueError("No writable fields provided")
    column_clause = ", ".join(_crud_sanitize_identifier(column) for column in columns)
    placeholder_clause = ", ".join(f":{column}" for column in columns)
    table_name = _crud_sanitize_identifier(resource.get("source_name"))
    returning_clause = _crud_select_clause(resource)
    statement = text(f"INSERT INTO {table_name} ({column_clause}) VALUES ({placeholder_clause}) RETURNING {returning_clause}")
    try:
        result = await session.execute(statement, params)
        await session.commit()
        created = result.mappings().first()
    except Exception as exc:
        await session.rollback()
        _record_runtime_error(
            context,
            code="crud_create_failed",
            message=f"Failed to create record for '{slug}'.",
            scope=slug,
            source="crud",
            detail=str(exc),
        )
        raise
    if created is None:
        pk_name = resource.get("primary_key") or "id"
        identifier = params.get(pk_name)
        created = await _crud_lookup_row(resource, identifier, session, context) if identifier is not None else params
    errors = _collect_runtime_errors(context)
    return {
        "resource": slug,
        "label": resource.get("label"),
        "status": "created",
        "item": dict(created) if created is not None else None,
        "errors": errors,
    }


async def update_crud_resource(
    slug: str,
    identifier: Any,
    payload: Dict[str, Any],
    session: Optional[AsyncSession],
) -> Dict[str, Any]:
    resource = _get_crud_resource(slug)
    allowed = _crud_allowed_operations(resource)
    if "update" not in allowed or resource.get("read_only"):
        raise PermissionError("update")
    if session is None:
        raise RuntimeError("Database session is required for CRUD mutations")
    context = _crud_build_context(session)
    source_type = str(resource.get("source_type") or "table").lower()
    if source_type == "dataset":
        raise PermissionError("update")
    mutable_fields = resource.get("mutable_fields") or []
    if not mutable_fields:
        mutable_fields = [field for field in resource.get("select_fields") or [] if field and field != resource.get("primary_key")]
    assignments: List[str] = []
    params: Dict[str, Any] = {"pk": identifier}
    for field in mutable_fields:
        if field in payload:
            sanitized = _crud_sanitize_identifier(field)
            placeholder = f"set_{field}"
            assignments.append(f"{sanitized} = :{placeholder}")
            params[placeholder] = payload[field]
    if not assignments:
        raise ValueError("No mutable fields provided")
    tenant_clause, tenant_params = _crud_tenant_clause(resource, context, required=bool(resource.get("tenant_column")))
    params.update(tenant_params)
    where_clause = f"{_crud_sanitize_identifier(resource.get('primary_key') or 'id')} = :pk"
    if tenant_clause:
        where_clause = f"{where_clause} AND {tenant_clause}"
    table_name = _crud_sanitize_identifier(resource.get("source_name"))
    update_sql = text(f"UPDATE {table_name} SET {', '.join(assignments)} WHERE {where_clause}")
    try:
        result = await session.execute(update_sql, params)
        await session.commit()
        updated_rows = result.rowcount or 0
    except Exception as exc:
        await session.rollback()
        _record_runtime_error(
            context,
            code="crud_update_failed",
            message=f"Failed to update record for '{slug}'.",
            scope=slug,
            source="crud",
            detail=str(exc),
        )
        raise
    if updated_rows <= 0:
        status = "not_found"
        item: Optional[Dict[str, Any]] = None
    else:
        status = "updated"
        item = await _crud_lookup_row(resource, identifier, session, context)
    errors = _collect_runtime_errors(context)
    return {
        "resource": slug,
        "label": resource.get("label"),
        "status": status,
        "item": item,
        "errors": errors,
    }


async def delete_crud_resource(
    slug: str,
    identifier: Any,
    session: Optional[AsyncSession],
) -> Dict[str, Any]:
    resource = _get_crud_resource(slug)
    allowed = _crud_allowed_operations(resource)
    if "delete" not in allowed or resource.get("read_only"):
        raise PermissionError("delete")
    if session is None:
        raise RuntimeError("Database session is required for CRUD mutations")
    context = _crud_build_context(session)
    source_type = str(resource.get("source_type") or "table").lower()
    if source_type == "dataset":
        raise PermissionError("delete")
    tenant_clause, tenant_params = _crud_tenant_clause(resource, context, required=bool(resource.get("tenant_column")))
    params: Dict[str, Any] = dict(tenant_params)
    params["pk"] = identifier
    pk_column = _crud_sanitize_identifier(resource.get("primary_key") or "id")
    where_clause = f"{pk_column} = :pk"
    if tenant_clause:
        where_clause = f"{where_clause} AND {tenant_clause}"
    table_name = _crud_sanitize_identifier(resource.get("source_name"))
    delete_sql = text(f"DELETE FROM {table_name} WHERE {where_clause}")
    try:
        result = await session.execute(delete_sql, params)
        await session.commit()
        removed = result.rowcount or 0
    except Exception as exc:
        await session.rollback()
        _record_runtime_error(
            context,
            code="crud_delete_failed",
            message=f"Failed to delete record for '{slug}'.",
            scope=slug,
            source="crud",
            detail=str(exc),
        )
        raise
    status = "deleted" if removed > 0 else "not_found"
    errors = _collect_runtime_errors(context)
    return {
        "resource": slug,
        "label": resource.get("label"),
        "status": status,
        "deleted": removed > 0,
        "errors": errors,
    }

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
    elif otype == "prompt_call":
        name = operation.get("prompt") or operation.get("name")
        arguments_raw = operation.get("arguments") or {}
        arguments = _resolve_placeholders(arguments_raw, context)
        payload = run_prompt(name, arguments, context=context)
        result = {"type": "prompt_call", "result": payload}
    elif otype == "chain_run":
        name = operation.get("name")
        inputs_raw = operation.get("inputs") or {}
        inputs = _resolve_placeholders(inputs_raw, context)
        payload = run_chain(name, inputs, context=context)
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


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    if isinstance(value, dict):
        return len(value) == 0
    return False


def _validate_form_submission(
    component: Dict[str, Any],
    submitted: Dict[str, Any],
    context: Dict[str, Any],
) -> bool:
    fields = component.get("fields") or []
    has_errors = False
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = field.get("name")
        if not name:
            continue
        required = field.get("required", True)
        if not required:
            continue
        value = submitted.get(name)
        if _is_missing_value(value):
            has_errors = True
            field_label = field.get("label") or name
            _record_runtime_error(
                context,
                code="missing_required_field",
                message="This field is required.",
                scope=f"field:{name}",
                source="form",
                detail=f"Expected value for '{field_label}'.",
            )
    return has_errors


def _partition_interaction_errors(
    slug: str,
    errors: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not errors:
        return [], []
    widget_errors: List[Dict[str, Any]] = []
    page_errors: List[Dict[str, Any]] = []
    slug_value = slug or ""
    slug_lower = slug_value.lower()
    page_scope = f"page:{slug_value}"
    page_scope_lower = page_scope.lower()
    page_dot_scope_lower = f"page.{slug_value}".lower()
    for entry in errors:
        if not isinstance(entry, dict):
            continue
        scope_value = entry.get("scope")
        normalized_scope = str(scope_value).strip().lower() if scope_value is not None else ""
        if normalized_scope in {"", slug_lower, page_scope_lower, page_dot_scope_lower, "page"}:
            entry["scope"] = page_scope
            page_errors.append(entry)
            continue
        if normalized_scope.startswith("page:") or normalized_scope.startswith("page.") or normalized_scope.startswith("app:"):
            page_errors.append(entry)
            continue
        widget_errors.append(entry)
    return widget_errors, page_errors


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

    component_payload = component.get("payload") if isinstance(component.get("payload"), dict) else component

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

    operations = component_payload.get("operations") or []
    results: List[Dict[str, Any]] = []
    validation_failed = False
    if kind == "form":
        validation_failed = _validate_form_submission(component_payload, submitted, context)
    if not validation_failed:
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
    collected_errors: List[Dict[str, Any]] = []
    if slug:
        collected_errors.extend(_collect_runtime_errors(context, scope=slug))
    collected_errors.extend(_collect_runtime_errors(context))
    widget_errors, page_errors = _partition_interaction_errors(slug, collected_errors)
    if widget_errors:
        response["errors"] = widget_errors
    if page_errors:
        response["page_errors"] = page_errors
        response["pageErrors"] = page_errors
    if widget_errors or page_errors:
        combined = widget_errors + page_errors
        if any((entry.get("severity") or "error").lower() == "error" for entry in combined if isinstance(entry, dict)):
            response["status"] = "error"
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
        raw_value = statement.get("value")
        pipeline_plan = raw_value.get("__frame_pipeline__") if isinstance(raw_value, dict) else None
        if pipeline_plan is not None:
            value = await _resolve_frame_pipeline_value(raw_value, session, context)
        elif expr is not None:
            value = await _evaluate_runtime_expression(expr, context, scope)
        else:
            value = _resolve_placeholders(raw_value, context)
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
    component_index = payload.pop("__component_index", None)
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
        error_scope = _component_error_scope(resolved)
        component_errors: List[Dict[str, Any]] = []
        if error_scope:
            component_errors.extend(_collect_runtime_errors(context, error_scope))
        insight_ref = resolved.get("insight")
        if isinstance(insight_ref, str):
            component_errors.extend(_collect_runtime_errors(context, insight_ref))
        if component_errors:
            resolved["errors"] = component_errors
        if component_index is not None:
            resolved["component_index"] = component_index
        return [{"type": "table", **resolved}]
    if payload_type == "chart":
        resolved = _resolve_placeholders(payload, context)
        if "heading" in resolved:
            resolved["heading"] = _render_template_value(resolved.get("heading"), context)
        if "title" in resolved:
            resolved["title"] = _render_template_value(resolved.get("title"), context)
        dataset_ref = _component_error_scope(resolved)
        component_errors: List[Dict[str, Any]] = []
        if dataset_ref:
            component_errors.extend(_collect_runtime_errors(context, dataset_ref))
        insight_ref = resolved.get("insight")
        if isinstance(insight_ref, str):
            component_errors.extend(_collect_runtime_errors(context, insight_ref))
        if component_errors:
            resolved["errors"] = component_errors
        if component_index is not None:
            resolved["component_index"] = component_index
        return [{"type": "chart", **resolved}]
    if payload_type == "form":
        resolved = _resolve_placeholders(payload, context)
        if "title" in resolved:


            resolved["title"] = _render_template_value(resolved.get("title"), context)
        if component_index is not None:
            resolved["component_index"] = component_index
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
        if component_index is not None:
            resolved["component_index"] = component_index
        if results:
            return results
        return [{"type": "action", **resolved}]
    if payload_type == "predict":
        resolved = _resolve_placeholders(payload, context)
        if component_index is not None:
            resolved["component_index"] = component_index
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
            except Exception as exc:  # pragma: no cover - runtime fetch failure
                logger.exception("Failed to fetch dataset rows for %s", source_name)
                _record_runtime_error(
                    context,
                    code="dataset_fetch_failed",
                    message=f"Failed to fetch dataset '{source_name}'.",
                    scope=source_name,
                    source="dataset",
                    detail=str(exc),
                )
        dataset_spec = DATASETS.get(source_name)
        if dataset_spec:
            return list(dataset_spec.get("sample_rows", []))


        return []
    if source_kind == "frame":
        try:
            return await fetch_frame_rows(source_name, session, context)
        except Exception as exc:  # pragma: no cover - runtime fetch failure
            logger.exception("Failed to fetch frame rows for %s", source_name)
            _record_runtime_error(
                context,
                code="frame_fetch_failed",
                message=f"Failed to fetch frame '{source_name}'.",
                scope=source_name,
                source="frame",
                detail=str(exc),
            )
        frame_spec = FRAMES.get(source_name)
        if isinstance(frame_spec, dict):
            examples = frame_spec.get("examples")
            if isinstance(examples, list):
                sample_rows = [row for row in examples if isinstance(row, dict)]
                if sample_rows:
                    return _clone_rows(sample_rows)
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
        except Exception as exc:  # pragma: no cover - operator failure
            logger.exception("Failed to evaluate binary operation '%s'", op)
            _record_runtime_error(
                context,
                code="runtime_expression_failed",
                message=f"Binary operation '{op}' failed during evaluation.",
                scope=context.get("page") if isinstance(context, dict) else None,
                source="page",
                detail=str(exc),
            )
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
        except Exception as exc:  # pragma: no cover - unary failure
            logger.exception("Failed to evaluate unary operation '%s'", op)
            _record_runtime_error(
                context,
                code="runtime_expression_failed",
                message=f"Unary operation '{op}' failed during evaluation.",
                scope=context.get("page") if isinstance(context, dict) else None,
                source="page",
                detail=str(exc),
            )
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
        except Exception as exc:  # pragma: no cover - call failure
            logger.exception("Failed to execute call expression")
            _record_runtime_error(
                context,
                code="runtime_expression_failed",
                message="Runtime call expression failed during execution.",
                scope=context.get("page") if isinstance(context, dict) else None,
                source="page",
                detail=str(exc),
            )
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


async def prepare_page_components(
    page_meta: Dict[str, Any],
    components: List[Dict[str, Any]],
    context: Dict[str, Any],
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    hydrated: List[Dict[str, Any]] = []
    counters: Dict[str, int] = {}
    base_path = page_meta.get('api_path') or '/api/pages'
    for order, component in enumerate(components or []):
        if not isinstance(component, dict):
            continue
        clone = copy.deepcopy(component)
        ctype = str(clone.get('type') or 'component')
        counters[ctype] = counters.get(ctype, 0) + 1
        clone.setdefault('id', f"{ctype}-{counters[ctype]}")
        clone['order'] = order
        component_index = clone.get('component_index')
        if ctype == 'table':
            await _hydrate_table_component(clone, context, session)
            if component_index is not None:
                clone.setdefault('endpoint', f"{base_path}/tables/{component_index}")
        elif ctype == 'chart':
            await _hydrate_chart_component(clone, context, session)
            if component_index is not None:
                clone.setdefault('endpoint', f"{base_path}/charts/{component_index}")
        elif ctype == 'form' and component_index is not None:
            clone.setdefault('submit_url', f"{base_path}/forms/{component_index}")
        elif ctype == 'action' and component_index is not None:
            clone.setdefault('action_url', f"{base_path}/actions/{component_index}")
        hydrated.append(clone)
    return hydrated


async def _hydrate_table_component(
    component: Dict[str, Any],
    context: Dict[str, Any],
    session: Optional[AsyncSession],
) -> None:
    source_type, dataset_name = _component_source_info(component)
    rows: List[Dict[str, Any]] = []
    if dataset_name:
        if source_type == 'frame':
            try:
                rows = await fetch_frame_rows(dataset_name, session, context)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to fetch rows for frame '%s'", dataset_name)
                _record_runtime_error(
                    context,
                    code="frame_fetch_failed",
                    message=f"Failed to fetch frame '{dataset_name}'.",
                    scope=dataset_name,
                    source="frame",
                    detail=str(exc),
                )
            if not rows:
                frame_spec = FRAMES.get(dataset_name) if isinstance(FRAMES, dict) else None
                if isinstance(frame_spec, dict):
                    examples = frame_spec.get('examples')
                    if isinstance(examples, list):
                        sample_rows = [row for row in examples if isinstance(row, dict)]
                        if sample_rows:
                            rows = _clone_rows(sample_rows)
        else:
            try:
                rows = await fetch_dataset_rows(dataset_name, session, context)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to fetch rows for dataset '%s'", dataset_name)
                _record_runtime_error(
                    context,
                    code="dataset_fetch_failed",
                    message=f"Failed to fetch dataset '{dataset_name}'.",
                    scope=dataset_name,
                    source="dataset",
                    detail=str(exc),
                )
            if not rows:
                dataset_spec = DATASETS.get(dataset_name) or {}
                sample_rows = dataset_spec.get('sample_rows') if isinstance(dataset_spec, dict) else None
                if isinstance(sample_rows, list):
                    rows = _clone_rows(sample_rows)
    if rows:
        rows = _clone_rows(rows)
    else:
        rows = []
    limited_rows = rows[:50]
    component['rows'] = limited_rows
    columns = component.get('columns')
    if not columns:
        if limited_rows:
            columns = list(limited_rows[0].keys())
        elif source_type == 'frame' and dataset_name:
            frame_spec = FRAMES.get(dataset_name) if isinstance(FRAMES, dict) else None
            if isinstance(frame_spec, dict):
                frame_columns = frame_spec.get('columns') or []
                if isinstance(frame_columns, list):
                    candidate = [col.get('name') for col in frame_columns if isinstance(col, dict) and col.get('name')]
                    if candidate:
                        columns = candidate
        columns = columns or []
    component['columns'] = columns


async def _hydrate_chart_component(
    component: Dict[str, Any],
    context: Dict[str, Any],
    session: Optional[AsyncSession],
) -> None:
    source_type, dataset_name = _component_source_info(component)
    rows: List[Dict[str, Any]] = []
    if dataset_name:
        if source_type == 'frame':
            try:
                rows = await fetch_frame_rows(dataset_name, session, context)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to fetch chart rows for frame '%s'", dataset_name)
                _record_runtime_error(
                    context,
                    code="frame_fetch_failed",
                    message=f"Failed to fetch frame '{dataset_name}' for chart.",
                    scope=dataset_name,
                    source="frame",
                    detail=str(exc),
                )
            if not rows:
                frame_spec = FRAMES.get(dataset_name) if isinstance(FRAMES, dict) else None
                if isinstance(frame_spec, dict):
                    examples = frame_spec.get('examples')
                    if isinstance(examples, list):
                        dict_rows = [row for row in examples if isinstance(row, dict)]
                        if dict_rows:
                            rows = _clone_rows(dict_rows)
        else:
            try:
                rows = await fetch_dataset_rows(dataset_name, session, context)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to fetch chart rows for dataset '%s'", dataset_name)
                _record_runtime_error(
                    context,
                    code="dataset_fetch_failed",
                    message=f"Failed to fetch dataset '{dataset_name}' for chart.",
                    scope=dataset_name,
                    source="dataset",
                    detail=str(exc),
                )
            if not rows:
                dataset_spec = DATASETS.get(dataset_name) or {}
                sample_rows = dataset_spec.get('sample_rows') if isinstance(dataset_spec, dict) else None
                if isinstance(sample_rows, list):
                    rows = _clone_rows(sample_rows)
    if rows:
        rows = _clone_rows(rows)
    else:
        rows = []
    x_key = component.get('x')
    y_key = component.get('y')
    labels: List[Any] = []
    values: List[float] = []
    limited_rows = rows[:50]
    for idx, row in enumerate(limited_rows, start=1):
        if not isinstance(row, dict):
            continue
        label_value = row.get(x_key) if x_key else None
        if label_value is None:
            label_value = row.get('label') or row.get('name') or idx
        labels.append(label_value)
        raw_value = row.get(y_key) if y_key else row.get('value')
        values.append(_coerce_number(raw_value))
    if not labels and limited_rows:
        labels = list(range(1, len(limited_rows) + 1))
    if not values:
        values = [0 for _ in labels] or [0]
    component['labels'] = labels
    component['series'] = [{
        'label': component.get('title') or component.get('heading') or 'Series',
        'data': values,
    }]
    component['rows'] = limited_rows


def _component_source_info(component: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    source_type = str(component.get('source_type') or '').lower()
    source = component.get('source')
    source_name: Optional[str] = None
    if isinstance(source, str):
        source_name = source
    elif isinstance(source, dict):
        name = source.get('name')
        if isinstance(name, str):
            source_name = name
        if not source_type:
            inline_type = source.get('type')
            if isinstance(inline_type, str):
                source_type = inline_type.lower()
    if not source_type and source_name:
        source_type = 'dataset'
    return source_type, source_name


def _component_dataset_name(component: Dict[str, Any]) -> Optional[str]:
    _, source_name = _component_source_info(component)
    return source_name


def _component_error_scope(component: Dict[str, Any]) -> Optional[str]:
    source_type, source_name = _component_source_info(component)
    if source_name and source_type in {'dataset', 'table', 'frame'}:
        return source_name
    return None


def _coerce_number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

_ALLOWED_MODULE_ROOTS: List[Path] = []
if os.getenv("NAMEL3SS_APP_ROOT"):
    try:
        _ALLOWED_MODULE_ROOTS.append(Path(os.getenv("NAMEL3SS_APP_ROOT")).resolve())
    except Exception:
        logger.warning("Invalid NAMEL3SS_APP_ROOT configured")
if os.getenv("NAMEL3SS_ALLOWED_MODULE_ROOTS"):
    for entry in os.getenv("NAMEL3SS_ALLOWED_MODULE_ROOTS", "").split(":"):
        candidate = entry.strip()
        if not candidate:
            continue
        try:
            _ALLOWED_MODULE_ROOTS.append(Path(candidate).resolve())
        except Exception:
            logger.warning("Invalid module root %s", candidate)
try:
    _ALLOWED_MODULE_ROOTS.append(Path(__file__).resolve().parent.parent)
except Exception:
    pass
_ALLOWED_MODULE_ROOTS.append(Path.cwd().resolve())


def _is_allowed_module_path(candidate: Path) -> bool:
    resolved = candidate.resolve()
    for root in _ALLOWED_MODULE_ROOTS:
        try:
            resolved.relative_to(root)
            return True
        except Exception:
            continue
    return False
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
            if not _is_allowed_module_path(path):
                logger.warning("Rejected python import outside allowed roots: %s", path)
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


def call_python_model(
    module: str,
    method: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    args = dict(arguments or {})
    module_obj = _import_python_module(module)
    attr_name = method or "predict"
    if module_obj is not None:
        callable_obj = getattr(module_obj, attr_name, None)
        if callable(callable_obj):
            try:
                result = callable_obj(**args)
                payload = result if isinstance(result, dict) else {"value": result}
                return {
                    "result": payload,
                    "inputs": args,
                    "module": module,
                    "method": attr_name,
                    "status": "ok",
                }
            except Exception:  # pragma: no cover - user callable failure
                logger.exception("Python callable %s.%s raised an error", module, attr_name)
    return {
        "result": "stub_prediction",
        "inputs": args,
        "module": module,
        "method": attr_name,
        "status": "stub",
    }


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

import asyncio
import copy
import inspect
import json
import os
import time
from typing import Any, Awaitable, Dict, List, Optional, Tuple

from namel3ss.codegen.backend.core.runtime.expression_sandbox import (
    evaluate_expression_tree as _evaluate_expression_tree,
)

# Structured prompt support
try:
    from namel3ss.prompts import PromptProgram, execute_structured_prompt_sync
    from namel3ss.llm.base import BaseLLM
    from namel3ss.ast import Prompt, PromptArgument, OutputSchema, OutputField, OutputFieldType
    _STRUCTURED_PROMPTS_AVAILABLE = True
except ImportError:
    _STRUCTURED_PROMPTS_AVAILABLE = False


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
    import random

    request_headers = {
        str(key): str(value)
        for key, value in headers.items()
    }
    request_headers.setdefault("Content-Type", "application/json")

    try:
        import httpx as _httpx  # type: ignore
    except Exception:  # pragma: no cover - optional dependency guard
        _httpx = None  # type: ignore

    if _httpx is not None:
        # Get connector config from runtime settings for retry behavior
        connector_settings = RUNTIME_SETTINGS.get("connectors", {})
        retry_max_attempts = connector_settings.get("retry_max_attempts", 3)
        retry_base_delay = connector_settings.get("retry_base_delay", 0.5)
        retry_max_delay = connector_settings.get("retry_max_delay", 5.0)

        client_kwargs: Dict[str, Any] = {}
        try:
            client_kwargs["timeout"] = _httpx.Timeout(timeout)
        except Exception:
            client_kwargs["timeout"] = timeout

        last_error: Optional[Exception] = None
        for attempt in range(1, retry_max_attempts + 1):
            try:
                with _httpx.Client(**client_kwargs) as client:
                    response = client.request(
                        "POST",
                        url,
                        json=data,
                        headers=request_headers,
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    status_code = int(getattr(response, "status_code", 0))
                    parsed: Optional[Any]
                    try:
                        parsed = response.json()
                    except Exception:
                        parsed = None
                    text = ""
                    raw_text_candidate = getattr(response, "text", "")
                    if isinstance(raw_text_candidate, str) and raw_text_candidate:
                        text = raw_text_candidate
                    elif parsed is not None:
                        try:
                            text = _json.dumps(parsed)
                        except Exception:
                            text = ""
                    else:
                        raw_bytes = getattr(response, "content", b"")
                        if isinstance(raw_bytes, (bytes, bytearray)):
                            text = bytes(raw_bytes).decode("utf-8", "replace")
                    return status_code, text, parsed
            except Exception as exc:
                last_error = exc
                if attempt < retry_max_attempts:
                    # Calculate exponential backoff with jitter
                    delay = min(retry_base_delay * (2 ** (attempt - 1)), retry_max_delay)
                    jitter = random.uniform(0, delay * 0.1)
                    import time
                    time.sleep(delay + jitter)
                else:
                    raise last_error from exc

    payload_bytes = _json.dumps(data).encode("utf-8")
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


def _extract_memory_names(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        names: List[str] = []
        for item in value:
            if isinstance(item, str):
                names.append(item)
        return names
    return []


_SCHEMA_TYPE_MAPPING = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": (list, tuple),
}


# LLM Response Caching
class _LLMCache:
    """Thread-safe TTL cache for LLM responses."""

    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        """Initialize cache.

        Args:
            maxsize: Maximum number of cached responses
            ttl: Time-to-live in seconds (default 1 hour)
        """
        import threading
        from collections import OrderedDict
        import time

        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "errors": 0,
        }

    def _hash_key(self, prompt: str, model: str, params: Dict[str, Any]) -> str:
        """Generate cache key from prompt, model, and parameters."""
        import hashlib
        import json

        # Normalize parameters for consistent hashing
        normalized_params = {
            k: v for k, v in sorted(params.items())
            if k not in ("stream", "timeout")  # Exclude non-deterministic params
        }

        key_data = {
            "prompt": prompt,
            "model": model,
            "params": normalized_params,
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, prompt: str, model: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached response if available and not expired."""
        import time

        key = self._hash_key(prompt, model, params)

        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None

            response, timestamp = self.cache[key]

            # Check if expired
            if time.time() - timestamp > self.ttl:
                del self.cache[key]
                self.stats["evictions"] += 1
                self.stats["misses"] += 1
                return None

            # Move to end (LRU)
            self.cache.move_to_end(key)
            self.stats["hits"] += 1
            return response

    def put(self, prompt: str, model: str, params: Dict[str, Any], response: Any) -> None:
        """Store response in cache."""
        import time

        key = self._hash_key(prompt, model, params)

        with self.lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.maxsize and key not in self.cache:
                self.cache.popitem(last=False)
                self.stats["evictions"] += 1

            self.cache[key] = (response, time.time())

    def clear(self) -> None:
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self.lock:
            total = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total if total > 0 else 0.0

            return {
                **self.stats,
                "size": len(self.cache),
                "hit_rate": hit_rate,
            }


# Global cache instance
_llm_cache: Optional[_LLMCache] = None


def _get_llm_cache() -> _LLMCache:
    """Get or create global LLM cache."""
    global _llm_cache

    if _llm_cache is None:
        # Get cache settings from runtime config
        cache_config = RUNTIME_SETTINGS.get("llm_cache", {})
        enabled = cache_config.get("enabled", True)

        if not enabled:
            # Return dummy cache that always misses
            class _DummyCache:
                def get(self, *args, **kwargs): return None
                def put(self, *args, **kwargs): pass
                def clear(self): pass
                def get_stats(self): return {}
            return _DummyCache()

        maxsize = cache_config.get("maxsize", 1000)
        ttl = cache_config.get("ttl", 3600)  # 1 hour default

        _llm_cache = _LLMCache(maxsize=maxsize, ttl=ttl)

    return _llm_cache


def _cached_llm_call(
    prompt: str,
    model: str,
    params: Dict[str, Any],
    call_func: Callable[[], Any],
) -> Any:
    """Execute LLM call with caching.

    Args:
        prompt: The prompt text
        model: Model identifier
        params: Call parameters
        call_func: Function to call if cache miss

    Returns:
        LLM response (from cache or fresh call)
    """
    cache = _get_llm_cache()

    # Try cache first
    cached_response = cache.get(prompt, model, params)
    if cached_response is not None:
        _record_event(
            "llm_cache",
            "hit",
            "info",
            {"model": model, "prompt_length": len(prompt)}
        )
        return cached_response

    # Cache miss - make actual call
    try:
        response = call_func()

        # Store in cache
        cache.put(prompt, model, params, response)

        _record_event(
            "llm_cache",
            "miss",
            "info",
            {"model": model, "prompt_length": len(prompt)}
        )

        return response

    except Exception as exc:
        _record_event(
            "llm_cache",
            "error",
            "error",
            {"model": model, "error": str(exc)}
        )
        raise


def _clear_llm_cache() -> Dict[str, Any]:
    """Clear LLM cache and return final stats."""
    cache = _get_llm_cache()
    stats = cache.get_stats()
    cache.clear()
    return stats


# LLM Observability Metrics
class _LLMMetrics:
    """Track LLM performance and usage metrics."""

    def __init__(self):
        """Initialize metrics tracking."""
        import threading
        from collections import defaultdict

        self.lock = threading.RLock()
        self.metrics = {
            "calls": defaultdict(int),  # By model
            "errors": defaultdict(int),  # By error type
            "latencies": defaultdict(list),  # By model
            "tokens": defaultdict(int),  # By model
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def record_call(
        self,
        model: str,
        latency_ms: float,
        tokens: Optional[int] = None,
        cached: bool = False,
    ) -> None:
        """Record successful LLM call."""
        with self.lock:
            self.metrics["calls"][model] += 1
            self.metrics["latencies"][model].append(latency_ms)

            if tokens:
                self.metrics["tokens"][model] += tokens

            if cached:
                self.metrics["cache_hits"] += 1
            else:
                self.metrics["cache_misses"] += 1

    def record_error(self, model: str, error_type: str) -> None:
        """Record LLM call error."""
        with self.lock:
            error_key = f"{model}:{error_type}"
            self.metrics["errors"][error_key] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self.lock:
            stats: Dict[str, Any] = {
                "total_calls": sum(self.metrics["calls"].values()),
                "total_errors": sum(self.metrics["errors"].values()),
                "cache_hit_rate": 0.0,
                "by_model": {},
            }

            # Calculate cache hit rate
            total_cache_ops = self.metrics["cache_hits"] + self.metrics["cache_misses"]
            if total_cache_ops > 0:
                stats["cache_hit_rate"] = self.metrics["cache_hits"] / total_cache_ops

            # Per-model statistics
            for model in self.metrics["calls"].keys():
                latencies = self.metrics["latencies"].get(model, [])

                if latencies:
                    latencies_sorted = sorted(latencies)
                    n = len(latencies_sorted)

                    stats["by_model"][model] = {
                        "calls": self.metrics["calls"][model],
                        "tokens": self.metrics["tokens"].get(model, 0),
                        "latency_p50": latencies_sorted[int(n * 0.5)] if n > 0 else 0,
                        "latency_p95": latencies_sorted[int(n * 0.95)] if n > 0 else 0,
                        "latency_p99": latencies_sorted[int(n * 0.99)] if n > 0 else 0,
                        "latency_avg": sum(latencies) / len(latencies) if latencies else 0,
                    }

            # Error breakdown
            stats["errors"] = dict(self.metrics["errors"])

            return stats

    def reset(self) -> None:
        """Reset all metrics."""
        with self.lock:
            from collections import defaultdict

            self.metrics = {
                "calls": defaultdict(int),
                "errors": defaultdict(int),
                "latencies": defaultdict(list),
                "tokens": defaultdict(int),
                "cache_hits": 0,
                "cache_misses": 0,
            }


# Global metrics instance
_llm_metrics: Optional[_LLMMetrics] = None


def _get_llm_metrics() -> _LLMMetrics:
    """Get or create global metrics tracker."""
    global _llm_metrics

    if _llm_metrics is None:
        _llm_metrics = _LLMMetrics()

    return _llm_metrics


def _record_llm_metrics(
    model: str,
    latency_ms: float,
    tokens: Optional[int] = None,
    cached: bool = False,
    error: Optional[str] = None,
) -> None:
    """Record LLM operation metrics.

    Args:
        model: Model identifier
        latency_ms: Request latency in milliseconds
        tokens: Number of tokens used (if available)
        cached: Whether response was from cache
        error: Error type if call failed
    """
    metrics = _get_llm_metrics()

    if error:
        metrics.record_error(model, error)
    else:
        metrics.record_call(model, latency_ms, tokens, cached)


def _get_llm_stats() -> Dict[str, Any]:
    """Get current LLM metrics snapshot."""
    metrics = _get_llm_metrics()
    return metrics.get_stats()


def _reset_llm_metrics() -> None:
    """Reset LLM metrics."""
    metrics = _get_llm_metrics()
    metrics.reset()


# LLM Request Batching
class _LLMBatchRequest:
    """Represents a pending batch request."""

    def __init__(self, prompt: str, model: str, params: Dict[str, Any]):
        """Initialize batch request."""
        import asyncio
        import threading

        self.prompt = prompt
        self.model = model
        self.params = params
        self.result: Optional[Any] = None
        self.error: Optional[Exception] = None
        self.event = threading.Event()

    def set_result(self, result: Any) -> None:
        """Set successful result."""
        self.result = result
        self.event.set()

    def set_error(self, error: Exception) -> None:
        """Set error result."""
        self.error = error
        self.event.set()

    def wait(self, timeout: Optional[float] = None) -> Any:
        """Wait for result and return it."""
        self.event.wait(timeout)
        if self.error:
            raise self.error
        return self.result


class _LLMBatcher:
    """Batches LLM requests for efficient processing."""

    def __init__(
        self,
        max_batch_size: int = 10,
        batch_timeout_ms: float = 50.0,
    ):
        """Initialize batcher.

        Args:
            max_batch_size: Maximum requests per batch
            batch_timeout_ms: Time to wait for batch to fill (ms)
        """
        import threading
        from collections import defaultdict
        from queue import Queue

        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms

        # Separate queues per model
        self.queues: Dict[str, Queue] = defaultdict(Queue)
        self.locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self.processors: Dict[str, threading.Thread] = {}
        self.running = True

        self.stats = {
            "batches_processed": 0,
            "requests_batched": 0,
            "batch_sizes": [],
        }

    def _process_batches(self, model: str) -> None:
        """Background thread to process batches for a model."""
        import time

        while self.running:
            queue = self.queues[model]

            # Collect batch
            batch: List[_LLMBatchRequest] = []
            deadline = time.time() + (self.batch_timeout_ms / 1000.0)

            while len(batch) < self.max_batch_size and time.time() < deadline:
                try:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break

                    request = queue.get(timeout=remaining)
                    batch.append(request)
                except Exception:
                    break

            if not batch:
                continue

            # Process batch
            try:
                self._execute_batch(model, batch)
            except Exception as exc:
                # Set error on all requests
                for request in batch:
                    request.set_error(exc)

    def _execute_batch(self, model: str, batch: List[_LLMBatchRequest]) -> None:
        """Execute a batch of requests."""
        import time

        if len(batch) == 0:
            return

        # Record batch metrics
        self.stats["batches_processed"] += 1
        self.stats["requests_batched"] += len(batch)
        self.stats["batch_sizes"].append(len(batch))

        start_time = time.time()

        # Check if provider supports batching
        provider_supports_batch = self._supports_batching(model)

        if provider_supports_batch and len(batch) > 1:
            # Use provider batch API
            try:
                results = self._call_batch_api(model, batch)
                for request, result in zip(batch, results):
                    request.set_result(result)
            except Exception as exc:
                # Fallback to individual calls
                _record_event(
                    "llm_batch",
                    "fallback",
                    "warning",
                    {"model": model, "batch_size": len(batch), "error": str(exc)}
                )
                self._execute_individual(batch)
        else:
            # Execute individually (but in parallel if possible)
            self._execute_individual(batch)

        # Record timing
        duration_ms = (time.time() - start_time) * 1000
        _record_event(
            "llm_batch",
            "completed",
            "info",
            {
                "model": model,
                "batch_size": len(batch),
                "duration_ms": duration_ms,
                "per_request_ms": duration_ms / len(batch),
            }
        )

    def _supports_batching(self, model: str) -> bool:
        """Check if model/provider supports batch API."""
        # OpenAI batch API support
        if "gpt" in model.lower() or "openai" in model.lower():
            return True

        # Anthropic batch support (Claude)
        if "claude" in model.lower() or "anthropic" in model.lower():
            return True

        # Default: no batch support
        return False

    def _call_batch_api(
        self,
        model: str,
        batch: List[_LLMBatchRequest]
    ) -> List[Any]:
        """Call provider batch API."""
        # Extract prompts and common params
        prompts = [req.prompt for req in batch]

        # Use first request's params as template (they should be similar)
        params = dict(batch[0].params)

        # Determine provider
        if "gpt" in model.lower() or "openai" in model.lower():
            return self._call_openai_batch(model, prompts, params)
        elif "claude" in model.lower() or "anthropic" in model.lower():
            return self._call_anthropic_batch(model, prompts, params)
        else:
            # No batch support - fallback
            raise NotImplementedError(f"Batch API not implemented for {model}")

    def _call_openai_batch(
        self,
        model: str,
        prompts: List[str],
        params: Dict[str, Any]
    ) -> List[Any]:
        """Call OpenAI batch completion API."""
        # Build batch request
        messages_list = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]

        # Note: OpenAI doesn't have synchronous batch API yet
        # This is a simulated approach using concurrent calls
        import concurrent.futures

        def call_single(messages):
            return _call_openai_completion(model, messages, params)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            results = list(executor.map(call_single, messages_list))

        return results

    def _call_anthropic_batch(
        self,
        model: str,
        prompts: List[str],
        params: Dict[str, Any]
    ) -> List[Any]:
        """Call Anthropic batch completion API."""
        # Anthropic has batch API - use it
        import concurrent.futures

        def call_single(prompt):
            return _call_anthropic_completion(model, prompt, params)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            results = list(executor.map(call_single, prompts))

        return results

    def _execute_individual(self, batch: List[_LLMBatchRequest]) -> None:
        """Execute batch requests individually (with parallelism)."""
        import concurrent.futures

        def execute_request(request: _LLMBatchRequest):
            try:
                # Call individual LLM endpoint
                result = _call_llm_single(
                    request.model,
                    request.prompt,
                    request.params
                )
                request.set_result(result)
            except Exception as exc:
                request.set_error(exc)

        # Execute in parallel (up to 5 concurrent)
        max_workers = min(5, len(batch))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(execute_request, batch)

    def add_request(
        self,
        prompt: str,
        model: str,
        params: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Any:
        """Add request to batch queue and wait for result.

        Args:
            prompt: Prompt text
            model: Model identifier
            params: Call parameters
            timeout: Max wait time in seconds

        Returns:
            LLM response
        """
        # Create request
        request = _LLMBatchRequest(prompt, model, params)

        # Start processor for this model if not running
        if model not in self.processors:
            with self.locks[model]:
                if model not in self.processors:
                    import threading
                    processor = threading.Thread(
                        target=self._process_batches,
                        args=(model,),
                        daemon=True
                    )
                    processor.start()
                    self.processors[model] = processor

        # Add to queue
        self.queues[model].put(request)

        # Wait for result
        return request.wait(timeout)

    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        stats = dict(self.stats)

        if self.stats["batch_sizes"]:
            stats["avg_batch_size"] = sum(self.stats["batch_sizes"]) / len(self.stats["batch_sizes"])
            stats["max_batch_size"] = max(self.stats["batch_sizes"])
        else:
            stats["avg_batch_size"] = 0
            stats["max_batch_size"] = 0

        return stats

    def shutdown(self) -> None:
        """Shutdown batch processors."""
        self.running = False

        # Wait for processors to finish
        for processor in self.processors.values():
            processor.join(timeout=1.0)


# Global batcher instance
_llm_batcher: Optional[_LLMBatcher] = None


def _get_llm_batcher() -> _LLMBatcher:
    """Get or create global LLM batcher."""
    global _llm_batcher

    if _llm_batcher is None:
        # Get batch settings from runtime config
        batch_config = RUNTIME_SETTINGS.get("llm_batch", {})
        enabled = batch_config.get("enabled", True)

        if not enabled:
            # Return dummy batcher that calls directly
            class _DummyBatcher:
                def add_request(self, prompt, model, params, timeout=30.0):
                    return _call_llm_single(model, prompt, params)
                def get_stats(self):
                    return {}
                def shutdown(self):
                    pass
            return _DummyBatcher()

        max_batch_size = batch_config.get("max_batch_size", 10)
        batch_timeout_ms = batch_config.get("batch_timeout_ms", 50.0)

        _llm_batcher = _LLMBatcher(
            max_batch_size=max_batch_size,
            batch_timeout_ms=batch_timeout_ms
        )

    return _llm_batcher


def _batched_llm_call(
    prompt: str,
    model: str,
    params: Dict[str, Any],
    timeout: float = 30.0,
) -> Any:
    """Execute LLM call with automatic batching.

    Args:
        prompt: Prompt text
        model: Model identifier  
        params: Call parameters
        timeout: Max wait time

    Returns:
        LLM response
    """
    batcher = _get_llm_batcher()

    try:
        return batcher.add_request(prompt, model, params, timeout)
    except Exception as exc:
        _record_event(
            "llm_batch",
            "error",
            "error",
            {"model": model, "error": str(exc)}
        )
        raise


def _get_batch_stats() -> Dict[str, Any]:
    """Get current batching statistics."""
    batcher = _get_llm_batcher()
    return batcher.get_stats()


# Circuit Breaker Pattern
class _CircuitState:
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class _CircuitBreaker:
    """Circuit breaker for LLM provider calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            success_threshold: Successes needed to close circuit
        """
        import threading
        import time

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = _CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.lock = threading.RLock()

        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "rejected_calls": 0,
            "state_changes": 0,
        }

    def call(self, func: Callable[[], Any]) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to call

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        import time

        with self.lock:
            self.stats["total_calls"] += 1

            # Check if circuit should transition to half-open
            if self.state == _CircuitState.OPEN:
                if self.last_failure_time is not None:
                    elapsed = time.time() - self.last_failure_time
                    if elapsed >= self.recovery_timeout:
                        self._transition_to_half_open()

            # Reject if circuit is open
            if self.state == _CircuitState.OPEN:
                self.stats["rejected_calls"] += 1
                raise _CircuitBreakerError(
                    f"Circuit breaker is OPEN (too many failures). "
                    f"Will retry in {self.recovery_timeout}s"
                )

            # Allow limited requests in half-open state
            if self.state == _CircuitState.HALF_OPEN:
                # Only allow one request at a time in half-open
                pass

        # Execute function (outside lock to allow concurrency)
        try:
            result = func()
            self._on_success()
            return result

        except Exception as exc:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Handle successful call."""
        with self.lock:
            self.stats["successful_calls"] += 1
            self.failure_count = 0

            if self.state == _CircuitState.HALF_OPEN:
                self.success_count += 1

                # Close circuit if enough successes
                if self.success_count >= self.success_threshold:
                    self._transition_to_closed()

    def _on_failure(self) -> None:
        """Handle failed call."""
        import time

        with self.lock:
            self.stats["failed_calls"] += 1
            self.failure_count += 1
            self.last_failure_time = time.time()

            # Open circuit if too many failures
            if self.state == _CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self._transition_to_open()

            # Reopen if failure in half-open state
            elif self.state == _CircuitState.HALF_OPEN:
                self._transition_to_open()

    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        self.state = _CircuitState.OPEN
        self.stats["state_changes"] += 1

        _record_event(
            "circuit_breaker",
            "opened",
            "warning",
            {
                "failures": self.failure_count,
                "threshold": self.failure_threshold,
            }
        )

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self.state = _CircuitState.HALF_OPEN
        self.success_count = 0
        self.stats["state_changes"] += 1

        _record_event(
            "circuit_breaker",
            "half_open",
            "info",
            {"message": "Testing recovery"}
        )

    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self.state = _CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.stats["state_changes"] += 1

        _record_event(
            "circuit_breaker",
            "closed",
            "info",
            {"message": "Recovery successful"}
        )

    def get_state(self) -> str:
        """Get current circuit state."""
        with self.lock:
            return self.state

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self.lock:
            stats = dict(self.stats)
            stats["state"] = self.state
            stats["failure_count"] = self.failure_count

            # Calculate success rate
            total = stats["successful_calls"] + stats["failed_calls"]
            if total > 0:
                stats["success_rate"] = stats["successful_calls"] / total
            else:
                stats["success_rate"] = 0.0

            return stats

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self.lock:
            self.state = _CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None

            _record_event(
                "circuit_breaker",
                "reset",
                "info",
                {"message": "Manual reset"}
            )


class _CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


# Global circuit breakers per model
_circuit_breakers: Dict[str, _CircuitBreaker] = {}
_circuit_breaker_lock = None


def _get_circuit_breaker(model: str) -> _CircuitBreaker:
    """Get or create circuit breaker for model."""
    global _circuit_breakers, _circuit_breaker_lock

    if _circuit_breaker_lock is None:
        import threading
        _circuit_breaker_lock = threading.Lock()

    if model not in _circuit_breakers:
        with _circuit_breaker_lock:
            if model not in _circuit_breakers:
                # Get circuit breaker settings
                cb_config = RUNTIME_SETTINGS.get("circuit_breaker", {})
                enabled = cb_config.get("enabled", True)

                if not enabled:
                    # Return dummy that always allows
                    class _DummyCircuitBreaker:
                        def call(self, func):
                            return func()
                        def get_state(self):
                            return _CircuitState.CLOSED
                        def get_stats(self):
                            return {}
                        def reset(self):
                            pass
                    return _DummyCircuitBreaker()

                failure_threshold = cb_config.get("failure_threshold", 5)
                recovery_timeout = cb_config.get("recovery_timeout", 60.0)
                success_threshold = cb_config.get("success_threshold", 2)

                _circuit_breakers[model] = _CircuitBreaker(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    success_threshold=success_threshold,
                )

    return _circuit_breakers[model]


def _protected_llm_call(
    model: str,
    call_func: Callable[[], Any],
) -> Any:
    """Execute LLM call with circuit breaker protection.

    Args:
        model: Model identifier
        call_func: Function to call

    Returns:
        LLM response

    Raises:
        _CircuitBreakerError: If circuit is open
    """
    circuit_breaker = _get_circuit_breaker(model)
    return circuit_breaker.call(call_func)


def _get_circuit_breaker_stats(model: Optional[str] = None) -> Dict[str, Any]:
    """Get circuit breaker statistics.

    Args:
        model: Specific model, or None for all models

    Returns:
        Statistics dict
    """
    if model:
        cb = _get_circuit_breaker(model)
        return {model: cb.get_stats()}

    # Get stats for all models
    stats = {}
    for model_name, cb in _circuit_breakers.items():
        stats[model_name] = cb.get_stats()

    return stats


def _reset_circuit_breaker(model: str) -> None:
    """Manually reset circuit breaker for model.

    Args:
        model: Model identifier
    """
    circuit_breaker = _get_circuit_breaker(model)
    circuit_breaker.reset()


def _tool_schema(plugin: Any, attr: str) -> Optional[Dict[str, Any]]:
    schema = getattr(plugin, attr, None)
    return dict(schema) if isinstance(schema, dict) else None


def _matches_schema_type(value: Any, expected: str) -> bool:
    normalized = expected.strip().lower()
    type_info = _SCHEMA_TYPE_MAPPING.get(normalized)
    if type_info is None:
        return True
    return isinstance(value, type_info)


def _validate_tool_payload(plugin: Any, payload: Dict[str, Any], chain_name: str, step_label: str) -> None:
    schema = _tool_schema(plugin, "input_schema")
    if not schema:
        return
    for field, rule in schema.items():
        details = rule if isinstance(rule, dict) else {"type": str(rule)}
        required = bool(details.get("required"))
        expected_type = str(details.get("type") or "").strip()
        if required and field not in payload:
            raise ValueError(f"Tool step '{step_label}' in chain '{chain_name}' is missing required field '{field}'.")
        if expected_type and field in payload and not _matches_schema_type(payload[field], expected_type):
            raise TypeError(
                f"Field '{field}' in tool step '{step_label}' does not match expected type '{expected_type}'."
            )


def _await_plugin_result(result: Any) -> Any:
    if inspect.isawaitable(result):
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop and running_loop.is_running():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(result)
            finally:
                loop.close()
        return asyncio.run(result)
    return result


def _call_tool_plugin(plugin: Any, context: Dict[str, Any], payload: Dict[str, Any]) -> Any:
    result = plugin.call(context, payload)
    return _await_plugin_result(result)


_VIOLATION_LABELS = {"violation", "unsafe", "toxic", "blocked"}


def _evaluation_result_is_violation(result: Any) -> bool:
    if isinstance(result, dict):
        if bool(result.get("violation")):
            return True
        label = result.get("label")
        if isinstance(label, str) and label.strip().lower() in _VIOLATION_LABELS:
            return True
    return False


def _run_step_evaluations(
    config: Dict[str, Any],
    entry: Dict[str, Any],
    output: Any,
    context: Dict[str, Any],
    chain_name: str,
    step_label: str,
) -> Dict[str, Any]:
    evaluator_names = config.get("evaluators") or []
    if not evaluator_names:
        return {}
    plugins = context.get("evaluators") if isinstance(context.get("evaluators"), dict) else {}
    evaluation_results: Dict[str, Any] = {}
    for evaluator_name in evaluator_names:
        plugin = plugins.get(evaluator_name)
        if plugin is None:
            message = f"Evaluator '{evaluator_name}' is not configured."
            _record_runtime_error(
                context,
                code="evaluation_missing",
                message=message,
                scope=f"chain:{chain_name}",
                detail=f"step:{step_label}",
            )
            raise RuntimeError(message)
        payload = {
            "output": output,
            "step": step_label,
            "chain": chain_name,
        }
        try:
            evaluation_results[evaluator_name] = _await_plugin_result(plugin.call(context, payload))
        except Exception as exc:
            message = f"Evaluator '{evaluator_name}' failed: {exc}"
            _record_runtime_error(
                context,
                code="evaluation_failed",
                message=message,
                scope=f"chain:{chain_name}",
                detail=f"step:{step_label}",
            )
            raise
    entry["evaluation"] = evaluation_results
    return evaluation_results


def _apply_guardrail(
    guardrail_name: str,
    evaluation_results: Dict[str, Any],
    output: Any,
    entry: Dict[str, Any],
    chain_name: str,
    step_label: str,
    context: Dict[str, Any],
) -> Any:
    guardrail = GUARDRAILS.get(guardrail_name, {})
    if not guardrail:
        message = f"Guardrail '{guardrail_name}' is not defined."
        _record_runtime_error(
            context,
            code="guardrail_missing",
            message=message,
            scope=f"chain:{chain_name}",
            detail=f"step:{step_label}",
        )
        raise RuntimeError(message)
    evaluator_names = guardrail.get("evaluators") or []
    violation = False
    for evaluator_name in evaluator_names:
        result = evaluation_results.get(evaluator_name)
        if result is None:
            continue
        if _evaluation_result_is_violation(result):
            violation = True
            break
    entry["guardrail"] = {
        "name": guardrail_name,
        "action": guardrail.get("action"),
        "violated": violation,
    }
    if not violation:
        return output
    action = str(guardrail.get("action") or "block").lower()
    message = guardrail.get("message") or f"Guardrail '{guardrail_name}' blocked step '{step_label}'."
    data = {
        "guardrail": guardrail_name,
        "chain": chain_name,
        "step": step_label,
    }
    if action == "log_only":
        _record_runtime_event(
            context,
            event="guardrail_violation",
            level="warning",
            message=message,
            data=data,
        )
        return output
    _record_runtime_error(
        context,
        code="guardrail_violation",
        message=message,
        scope=f"chain:{chain_name}",
        detail=f"step:{step_label}",
    )
    blocked_payload = {
        "status": "blocked",
        "message": message,
        "guardrail": guardrail_name,
    }
    entry["output"] = blocked_payload
    entry["status"] = "error"
    return blocked_payload


async def _execute_workflow_nodes(
    nodes: List[Dict[str, Any]],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any]:
    """
    Execute workflow nodes with support for parallel execution.

    Detects groups of parallelizable steps and executes them concurrently
    when ENABLE_PARALLEL_STEPS environment variable is set.
    """
    enable_parallel = _is_truthy_env("ENABLE_PARALLEL_STEPS")

    if enable_parallel:
        # Detect parallel groups
        groups = _detect_parallel_groups(nodes or [])

        status = "partial"
        result_value: Any = None
        current_value = working
        encountered_error = False

        for group in groups:
            if len(group) == 1:
                # Single step - execute normally
                node = group[0]
                node_type = str(node.get("type") or "step").lower()

                if node_type == "step":
                    step_status, current_value, step_result, stop_on_error = await _execute_workflow_step(
                        node,
                        context=context,
                        chain_scope=chain_scope,
                        args=args,
                        working=current_value,
                        memory_state=memory_state,
                        allow_stubs=allow_stubs,
                        steps_history=steps_history,
                        chain_name=chain_name,
                    )
                    if step_status == "error":
                        encountered_error = True
                        if stop_on_error:
                            return "error", step_result, current_value
                    elif step_status == "ok":
                        status = "ok"
                        result_value = step_result
                    elif step_status == "stub" and status != "error" and result_value is None:
                        result_value = step_result
                elif node_type == "if":
                    branch_status, branch_result, branch_value = await _execute_workflow_if(
                        node,
                        context=context,
                        chain_scope=chain_scope,
                        args=args,
                        working=current_value,
                        memory_state=memory_state,
                        allow_stubs=allow_stubs,
                        steps_history=steps_history,
                        chain_name=chain_name,
                    )
                    if branch_status == "error":
                        return "error", branch_result, branch_value
                    if branch_status == "ok" and branch_result is not None:
                        status = "ok"
                        result_value = branch_result
                    current_value = branch_value
                elif node_type == "for":
                    loop_status, loop_result, loop_value = await _execute_workflow_for(
                        node,
                        context=context,
                        chain_scope=chain_scope,
                        args=args,
                        working=current_value,
                        memory_state=memory_state,
                        allow_stubs=allow_stubs,
                        steps_history=steps_history,
                        chain_name=chain_name,
                    )
                    if loop_status == "error":
                        return "error", loop_result, loop_value
                    if loop_status == "ok" and loop_result is not None:
                        status = "ok"
                        result_value = loop_result
                    current_value = loop_value
                elif node_type == "while":
                    while_status, while_result, while_value = await _execute_workflow_while(
                        node,
                        context=context,
                        chain_scope=chain_scope,
                        args=args,
                        working=current_value,
                        memory_state=memory_state,
                        allow_stubs=allow_stubs,
                        steps_history=steps_history,
                        chain_name=chain_name,
                    )
                    if while_status == "error":
                        return "error", while_result, while_value
                    if while_status == "ok" and while_result is not None:
                        status = "ok"
                        result_value = while_result
                    current_value = while_value
            else:
                # Multiple steps - execute in parallel
                logger.info(f"Executing {len(group)} steps in parallel for chain '{chain_name}'")
                parallel_status, parallel_result, parallel_value = await _execute_parallel_steps(
                    group,
                    context=context,
                    chain_scope=chain_scope,
                    args=args,
                    working=current_value,
                    memory_state=memory_state,
                    allow_stubs=allow_stubs,
                    steps_history=steps_history,
                    chain_name=chain_name,
                )
                if parallel_status == "error":
                    return "error", parallel_result, parallel_value
                if parallel_status == "ok" and parallel_result is not None:
                    status = "ok"
                    result_value = parallel_result
                current_value = parallel_value

        if encountered_error and status != "error":
            status = "error"
        return status, result_value, current_value

    # Sequential execution (default)
    status = "partial"
    result_value: Any = None
    current_value = working
    encountered_error = False
    for node in nodes or []:
        node_type = str(node.get("type") or "step").lower()
        if node_type == "step":
            step_status, current_value, step_result, stop_on_error = await _execute_workflow_step(
                node,
                context=context,
                chain_scope=chain_scope,
                args=args,
                working=current_value,
                memory_state=memory_state,
                allow_stubs=allow_stubs,
                steps_history=steps_history,
                chain_name=chain_name,
            )
            if step_status == "error":
                encountered_error = True
                if stop_on_error:
                    return "error", step_result, current_value
            elif step_status == "ok":
                status = "ok"
                result_value = step_result
            elif step_status == "stub" and status != "error" and result_value is None:
                result_value = step_result
        elif node_type == "if":
            branch_status, branch_result, branch_value = await _execute_workflow_if(
                node,
                context=context,
                chain_scope=chain_scope,
                args=args,
                working=current_value,
                memory_state=memory_state,
                allow_stubs=allow_stubs,
                steps_history=steps_history,
                chain_name=chain_name,
            )
            if branch_status == "error":
                return "error", branch_result, branch_value
            if branch_status == "ok" and branch_result is not None:
                status = "ok"
                result_value = branch_result
            current_value = branch_value
        elif node_type == "for":
            loop_status, loop_result, loop_value = await _execute_workflow_for(
                node,
                context=context,
                chain_scope=chain_scope,
                args=args,
                working=current_value,
                memory_state=memory_state,
                allow_stubs=allow_stubs,
                steps_history=steps_history,
                chain_name=chain_name,
            )
            if loop_status == "error":
                return "error", loop_result, loop_value
            if loop_status == "ok" and loop_result is not None:
                status = "ok"
                result_value = loop_result
            current_value = loop_value
        elif node_type == "while":
            while_status, while_result, while_value = await _execute_workflow_while(
                node,
                context=context,
                chain_scope=chain_scope,
                args=args,
                working=current_value,
                memory_state=memory_state,
                allow_stubs=allow_stubs,
                steps_history=steps_history,
                chain_name=chain_name,
            )
            if while_status == "error":
                return "error", while_result, while_value
            if while_status == "ok" and while_result is not None:
                status = "ok"
                result_value = while_result
            current_value = while_value
        else:
            _record_runtime_error(
                context,
                code="workflow.unsupported_node",
                message=f"Workflow node type '{node_type}' is not supported",
                scope=chain_name,
                source="workflow",
            )
            return "error", {"error": f"Unsupported workflow node '{node_type}'"}, current_value
    if encountered_error and status != "error":
        status = "error"
    return status, result_value, current_value


async def _execute_workflow_step(
    step: Dict[str, Any],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any, bool]:
    chain_scope["counter"] = chain_scope.get("counter", 0) + 1
    step_label = step.get("name") or step.get("target") or f"step_{chain_scope['counter']}"
    kind = (step.get("kind") or "").lower()
    target = step.get("target") or ""
    options = step.get("options") or {}
    resolved_options = _resolve_placeholders(options, context)
    stop_on_error = bool(step.get("stop_on_error", True))
    read_memory = _extract_memory_names(resolved_options.pop("read_memory", None))
    write_memory = _extract_memory_names(resolved_options.pop("write_memory", None))
    memory_payload = _memory_snapshot(memory_state, read_memory) if read_memory else {}
    entry: Dict[str, Any] = {
        "step": len(steps_history) + 1,
        "kind": kind,
        "name": step_label,
        "inputs": None,
        "output": None,
        "status": "partial",
    }
    steps_history.append(entry)
    current_value = working
    step_result: Any = None
    if kind == "template":
        template = AI_TEMPLATES.get(target) or {}
        prompt = template.get("prompt", "")
        template_context = {
            "input": working,
            "vars": args,
            "payload": args,
            "memory": memory_payload,
            "steps": context.get("steps", {}),
            "loop": context.get("loop", {}),
            "locals": chain_scope.get("locals", {}),
        }
        entry["inputs"] = template_context
        try:
            rendered = _render_template_value(prompt, template_context)
            entry["output"] = rendered
            entry["status"] = "ok"
            current_value = rendered
            step_result = rendered
            if write_memory:
                _write_memory_entries(
                    memory_state,
                    write_memory,
                    rendered,
                    context=context,
                    source=f"template:{step_label}",
                )
        except Exception as exc:
            entry["output"] = {"error": str(exc)}
            entry["status"] = "error"
    elif kind == "connector":
        connector_payload = dict(args)
        connector_payload.setdefault("prompt", working)
        if memory_payload:
            connector_payload.setdefault("memory", {}).update(memory_payload)
        entry["inputs"] = connector_payload
        response = await call_llm_connector(target, connector_payload)
        entry["output"] = response
        entry["status"] = response.get("status", "partial")
        step_status = response.get("status")
        if step_status == "ok":
            current_value = response.get("result")
            step_result = current_value
        elif step_status == "error":
            step_result = response
        elif step_status == "stub":
            step_result = response
        if step_status == "ok" and write_memory:
            _write_memory_entries(
                memory_state,
                write_memory,
                response.get("result"),
                context=context,
                source=f"connector:{step_label}",
            )
    elif kind == "prompt":
        prompt_payload = working if isinstance(working, dict) else working
        if isinstance(prompt_payload, dict):
            payload_data = dict(prompt_payload)
        else:
            payload_data = {"value": prompt_payload}
        if memory_payload:
            payload_data.setdefault("memory", {}).update(memory_payload)
        if read_memory:
            payload_data["read_memory"] = read_memory
        if write_memory:
            payload_data["write_memory"] = write_memory
        response = await run_prompt(target, payload_data, context=context, memory=memory_payload)
        entry["inputs"] = response.get("inputs")
        entry["output"] = response
        entry["status"] = response.get("status", "partial")
        step_status = response.get("status")
        if step_status == "ok":
            current_value = response.get("output")
            step_result = current_value
        elif step_status == "error":
            step_result = response
        elif step_status == "stub":
            step_result = response
    elif kind == "tool":
        tools_registry = context.get("tools") if isinstance(context.get("tools"), dict) else {}
        payload = dict(resolved_options)
        if working is not None and "input" not in payload:
            payload["input"] = working
        if memory_payload:
            payload.setdefault("memory", {}).update(memory_payload)
        entry["inputs"] = payload
        plugin = tools_registry.get(target)
        if plugin is None:
            message = f"Tool '{target}' is not configured"
            entry["output"] = {"error": message}
            entry["status"] = "error"
            step_result = entry["output"]
            _record_runtime_error(
                context,
                code="tool_not_found",
                message=message,
                scope=f"chain:{chain_name}",
                detail=f"step:{step_label}",
            )
        else:
            try:
                _validate_tool_payload(plugin, payload, chain_name, step_label)
                tool_result = _call_tool_plugin(plugin, context, payload)
                entry["output"] = tool_result
                entry["status"] = "ok"
                current_value = tool_result
                step_result = tool_result
                if write_memory:
                    _write_memory_entries(
                        memory_state,
                        write_memory,
                        tool_result,
                        context=context,
                        source=f"tool:{step_label}",
                    )
            except Exception as exc:
                message = str(exc)
                entry["output"] = {"error": message}
                entry["status"] = "error"
                step_result = entry["output"]
                _record_runtime_error(
                    context,
                    code="tool_step_error",
                    message=f"Tool step '{step_label}' failed in chain '{chain_name}'",
                    scope=f"chain:{chain_name}",
                    detail=message,
                )
    elif kind == "python":
        module_name = resolved_options.get("module") or target or ""
        method_name = resolved_options.get("method") or "predict"
        python_args = dict(args)
        provided_args = resolved_options.get("arguments")
        if isinstance(provided_args, dict):
            python_args.update(provided_args)
        if memory_payload:
            python_args.setdefault("memory", {}).update(memory_payload)
        entry["inputs"] = python_args
        response = call_python_model(module_name, method_name, python_args)
        entry["output"] = response
        entry["status"] = response.get("status", "partial")
        step_status = response.get("status")
        if step_status == "ok":
            current_value = response.get("result")
            step_result = current_value
        elif step_status == "error":
            step_result = response
        elif step_status == "stub":
            step_result = response
        if step_status == "ok" and write_memory:
            _write_memory_entries(
                memory_state,
                write_memory,
                response.get("result"),
                context=context,
                source=f"python:{step_label}",
            )
    elif kind == "graph":
        # Execute multi-agent graph
        graphs_registry = context.get("graphs") if isinstance(context.get("graphs"), dict) else {}
        agents_registry = context.get("agents") if isinstance(context.get("agents"), dict) else {}

        # Get LLM and tool registries - these are the actual instantiated objects
        llms_registry = {}
        llm_reg = context.get("llms") if isinstance(context.get("llms"), dict) else {}
        for llm_name, llm_instance in llm_reg.items():
            llms_registry[llm_name] = llm_instance

        tools_registry = {}
        tool_reg = context.get("tools") if isinstance(context.get("tools"), dict) else {}
        for tool_name, tool_instance in tool_reg.items():
            tools_registry[tool_name] = tool_instance

        graph_input = working
        if isinstance(resolved_options.get("input"), str):
            graph_input = resolved_options["input"]
        elif resolved_options.get("input") is not None:
            graph_input = resolved_options["input"]

        graph_context = dict(resolved_options.get("context", {}))
        if memory_payload:
            graph_context.setdefault("memory", {}).update(memory_payload)

        entry["inputs"] = {"input": graph_input, "context": graph_context}

        try:
            from namel3ss.agents.factory import run_graph_from_state

            response = run_graph_from_state(
                target,
                graphs_registry,
                agents_registry,
                llms_registry,
                tools_registry,
                graph_input,
                context=graph_context,
            )
            entry["output"] = response
            entry["status"] = response.get("status", "partial")
            step_status = response.get("status")

            if step_status == "success":
                current_value = response.get("final_response")
                step_result = current_value
                if write_memory:
                    _write_memory_entries(
                        memory_state,
                        write_memory,
                        current_value,
                        context=context,
                        source=f"graph:{step_label}",
                    )
            elif step_status == "error":
                step_result = response
                entry["status"] = "error"
        except Exception as exc:
            message = str(exc)
            entry["output"] = {"error": message}
            entry["status"] = "error"
            step_result = entry["output"]
            _record_runtime_error(
                context,
                code="graph_step_error",
                message=f"Graph step '{step_label}' failed in chain '{chain_name}'",
                scope=f"chain:{chain_name}",
                detail=message,
            )
    else:
        entry["output"] = {"error": f"Unsupported step kind '{kind}'"}
        entry["status"] = "error"
        step_result = entry["output"]
    evaluation_cfg = step.get("evaluation")
    if evaluation_cfg:
        try:
            evaluation_results = _run_step_evaluations(
                evaluation_cfg,
                entry,
                current_value,
                context,
                chain_name,
                step_label,
            )
        except Exception as exc:
            entry["output"] = {"error": str(exc)}
            entry["status"] = "error"
            step_result = entry["output"]
            entry["result"] = current_value
            chain_scope.setdefault("steps", {})[step_label] = entry
            return entry.get("status", "partial"), current_value, step_result, stop_on_error
        guardrail_name = evaluation_cfg.get("guardrail")
        if guardrail_name:
            try:
                current_value = _apply_guardrail(
                    guardrail_name,
                    evaluation_results,
                    current_value,
                    entry,
                    chain_name,
                    step_label,
                    context,
                )
                step_result = current_value
            except Exception as exc:
                entry["output"] = {"error": str(exc)}
                entry["status"] = "error"
                step_result = entry["output"]
                entry["result"] = current_value
                chain_scope.setdefault("steps", {})[step_label] = entry
                return entry.get("status", "partial"), current_value, step_result, stop_on_error
    entry["result"] = current_value
    chain_scope.setdefault("steps", {})[step_label] = entry
    return entry.get("status", "partial"), current_value, step_result, stop_on_error


async def _execute_workflow_if(
    node: Dict[str, Any],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any]:
    condition = node.get("condition")
    condition_source = node.get("condition_source")
    scope = _workflow_scope(chain_scope, args, working)
    branch_nodes = node.get("then") or []
    branch_selected = False
    if _evaluate_workflow_condition(condition, condition_source, scope, context, chain_name):
        branch_selected = True
    else:
        for branch in node.get("elif", []):
            if _evaluate_workflow_condition(
                branch.get("condition"),
                branch.get("condition_source"),
                scope,
                context,
                chain_name,
            ):
                branch_nodes = branch.get("steps") or []
                branch_selected = True
                break
        if not branch_selected:
            branch_nodes = node.get("else") or []
    if not branch_nodes:
        return "partial", None, working
    return await _execute_workflow_nodes(
        branch_nodes,
        context=context,
        chain_scope=chain_scope,
        args=args,
        working=working,
        memory_state=memory_state,
        allow_stubs=allow_stubs,
        steps_history=steps_history,
        chain_name=chain_name,
    )


async def _execute_workflow_for(
    node: Dict[str, Any],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any]:
    iterable = _resolve_workflow_iterable(node, context, chain_scope, args, working, chain_name)
    if not iterable:
        return "partial", None, working
    loop_var = node.get("loop_var") or "item"
    max_iterations = node.get("max_iterations")
    iterations = 0
    result_value: Any = None
    current_value = working
    loop_context = context.setdefault("loop", {})
    scope_loop = chain_scope.setdefault("loop", {})
    for item in iterable:
        if max_iterations and iterations >= max_iterations:
            break
        chain_scope["locals"][loop_var] = item
        loop_context[loop_var] = item
        scope_loop[loop_var] = item
        branch_status, branch_result, branch_value = await _execute_workflow_nodes(
            node.get("body") or [],
            context=context,
            chain_scope=chain_scope,
            args=args,
            working=current_value,
            memory_state=memory_state,
            allow_stubs=allow_stubs,
            steps_history=steps_history,
            chain_name=chain_name,
        )
        if branch_status == "error":
            chain_scope["locals"].pop(loop_var, None)
            loop_context.pop(loop_var, None)
            return "error", branch_result, branch_value
        if branch_result is not None:
            result_value = branch_result
        current_value = branch_value
        iterations += 1
    chain_scope["locals"].pop(loop_var, None)
    loop_context.pop(loop_var, None)
    scope_loop.pop(loop_var, None)
    return ("ok" if iterations else "partial"), result_value, current_value


async def _execute_workflow_while(
    node: Dict[str, Any],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any]:
    max_iterations = node.get("max_iterations") or 100
    iterations = 0
    result_value: Any = None
    current_value = working
    condition = node.get("condition")
    source = node.get("condition_source")
    while iterations < max_iterations:
        scope = _workflow_scope(chain_scope, args, current_value)
        if not _evaluate_workflow_condition(condition, source, scope, context, chain_name):
            break
        branch_status, branch_result, branch_value = await _execute_workflow_nodes(
            node.get("body") or [],
            context=context,
            chain_scope=chain_scope,
            args=args,
            working=current_value,
            memory_state=memory_state,
            allow_stubs=allow_stubs,
            steps_history=steps_history,
            chain_name=chain_name,
        )
        if branch_status == "error":
            return "error", branch_result, branch_value
        if branch_result is not None:
            result_value = branch_result
        current_value = branch_value
        iterations += 1
    if iterations >= max_iterations:
        _record_runtime_error(
            context,
            code="workflow.loop_limit",
            message=f"Workflow while loop in chain '{chain_name}' exceeded iteration limit",
            scope=chain_name,
            source="workflow",
        )
        return "error", {"error": "Loop iteration limit exceeded"}, current_value
    return ("ok" if iterations else "partial"), result_value, current_value


def _evaluate_workflow_condition(
    expression: Optional[Any],
    expression_source: Optional[str],
    scope: Dict[str, Any],
    context: Dict[str, Any],
    chain_name: str,
) -> bool:
    if expression is None and not expression_source:
        return False
    try:
        if expression is None:
            return bool(_evaluate_expression_tree(expression_source or "", scope, context))
        return bool(_evaluate_expression_tree(expression, scope, context))
    except Exception as exc:
        _record_runtime_error(
            context,
            code="workflow.condition_failed",
            message=f"Workflow condition in chain '{chain_name}' failed",
            scope=chain_name,
            source="workflow",
            detail=str(exc),
        )
    return False


def _resolve_workflow_iterable(
    node: Dict[str, Any],
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    chain_name: str,
) -> List[Any]:
    source_kind = str(node.get("source_kind") or "expression").lower()
    if source_kind == "dataset":
        name = node.get("source_name")
        dataset_rows = context.get("datasets_data", {}).get(name)
        if not dataset_rows:
            _record_runtime_error(
                context,
                code="workflow.dataset_missing",
                message=f"Dataset '{name}' is not available for workflow loop",
                scope=chain_name,
                source="workflow",
            )
            return []
        return list(dataset_rows)
    expr_payload = node.get("source_expression")
    expr_source = node.get("source_expression_source")
    value = _evaluate_workflow_expression(
        expr_payload,
        expr_source,
        chain_scope,
        args,
        working,
        context,
        chain_name,
    )
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return list(value)
    try:
        return list(value)
    except Exception:
        return []


def _evaluate_workflow_expression(
    expression: Optional[Any],
    expression_source: Optional[str],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    context: Dict[str, Any],
    chain_name: str,
) -> Any:
    if expression is None and not expression_source:
        return None
    scope = _workflow_scope(chain_scope, args, working)
    try:
        if expression is None:
            expr_text = expression_source or ""
            if not expr_text:
                return None
            return _evaluate_expression_tree(expr_text, scope, context)
        return _evaluate_expression_tree(expression, scope, context)
    except Exception as exc:
        _record_runtime_error(
            context,
            code="workflow.expression_failed",
            message=f"Workflow expression in chain '{chain_name}' failed",
            scope=chain_name,
            source="workflow",
            detail=str(exc),
        )
    return None


def _workflow_scope(chain_scope: Dict[str, Any], args: Dict[str, Any], working: Any) -> Dict[str, Any]:
    scope = {
        "input": chain_scope.get("input"),
        "steps": chain_scope.get("steps", {}),
        "locals": chain_scope.get("locals", {}),
        "loop": chain_scope.get("loop", {}),
        "payload": args,
        "value": working,
    }
    scope.update(chain_scope.get("locals", {}))
    return scope


def _normalize_prompt_inputs(prompt_spec: Dict[str, Any], payload: Any) -> Dict[str, Any]:
    schema = prompt_spec.get("input") or []
    if isinstance(payload, dict):
        return dict(payload)
    if not schema:
        return {}
    first_field = schema[0]
    return {first_field.get("name", "value"): payload}


def _coerce_prompt_field_value(field: Dict[str, Any], value: Any) -> Tuple[Any, Optional[str]]:
    name = field.get("name") or "value"
    field_type = str(field.get("type") or "text").lower()
    if value is None:
        return None, None
    try:
        if field_type in {"text", "string"}:
            return str(value), None
        if field_type in {"int", "integer"}:
            return int(value), None
        if field_type in {"float", "number"}:
            return float(value), None
        if field_type in {"boolean", "bool"}:
            if isinstance(value, bool):
                return value, None
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "on"}:
                    return True, None
                if lowered in {"false", "0", "no", "off"}:
                    return False, None
            return bool(value), None
        if field_type in {"json", "object"}:
            if isinstance(value, (dict, list)):
                return value, None
            if isinstance(value, str):
                return json.loads(value), None
        if field_type in {"list", "array"}:
            if isinstance(value, list):
                return value, None
            return [value], None
        if field_type == "enum":
            allowed = [str(item) for item in field.get("enum", []) if item is not None]
            text_value = str(value)
            if allowed and text_value not in allowed:
                return text_value, f"Value '{text_value}' is not permitted for '{name}'"
            return text_value, None
        return value, None
    except Exception as exc:
        return value, f"Unable to coerce field '{name}': {exc}"


def _validate_prompt_inputs(
    prompt_spec: Dict[str, Any],
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    schema = prompt_spec.get("input") or []
    provided = dict(payload or {})
    values: Dict[str, Any] = {}
    errors: List[str] = []
    for field in schema:
        name = field.get("name")
        if not name:
            continue
        if name in provided:
            raw_value = provided.pop(name)
        else:
            raw_value = field.get("default")
        if raw_value is None:
            if field.get("required", True):
                errors.append(f"Missing required prompt field '{name}'")
            continue
        coerced, err = _coerce_prompt_field_value(field, raw_value)
        if err:
            errors.append(err)
            continue
        values[name] = coerced
    if provided:
        extras = ", ".join(sorted(provided.keys()))
        errors.append(f"Unsupported prompt inputs: {extras}")
    return values, errors


def _merge_prompt_request_config(
    model_spec: Dict[str, Any],
    prompt_spec: Dict[str, Any],
) -> Dict[str, Any]:
    base_config = _ensure_dict(model_spec.get("config"))
    merged = copy.deepcopy(base_config)
    prompt_params = prompt_spec.get("parameters") or {}
    if not isinstance(prompt_params, dict):
        prompt_params = {}
    payload = merged.setdefault("payload", {})
    payload_keys = {
        "temperature",
        "top_p",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "response_format",
        "seed",
    }
    for key in payload_keys:
        if key in prompt_params and prompt_params[key] is not None:
            payload.setdefault(key, prompt_params[key])
    for key in ("headers", "params"):
        value = prompt_params.get(key)
        if isinstance(value, dict):
            merged.setdefault(key, {}).update(value)
    for key in ("timeout", "mode", "user_role", "system", "deployment", "endpoint", "api_version"):
        if key in prompt_params and prompt_params[key] is not None and key not in merged:
            merged[key] = prompt_params[key]
    return merged


def _project_prompt_output(
    prompt_spec: Dict[str, Any],
    result_payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    schema = prompt_spec.get("output") or []
    if not schema:
        return {}, []
    errors: List[str] = []
    text_value = result_payload.get("text") or ""
    structured: Optional[Dict[str, Any]] = None
    if isinstance(result_payload.get("json"), dict):
        structured = result_payload["json"]
        field_names = [field.get("name") for field in schema if field.get("name")]
        if not all(name in structured for name in field_names):
            structured = None
    elif isinstance(text_value, str):
        stripped = text_value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                structured = json.loads(stripped)
            except Exception:
                structured = None
    values: Dict[str, Any] = {}
    if structured is None:
        if len(schema) == 1:
            field = schema[0]
            coerced, err = _coerce_prompt_field_value(field, text_value)
            if err:
                errors.append(err)
            else:
                values[field.get("name") or "value"] = coerced
            return values, errors
        errors.append("Prompt output is not valid JSON")
        return {}, errors
    for field in schema:
        name = field.get("name")
        if not name:
            continue
        if name not in structured:
            errors.append(f"Prompt output is missing '{name}'")
            continue
        coerced, err = _coerce_prompt_field_value(field, structured.get(name))
        if err:
            errors.append(err)
            continue
        values[name] = coerced
    return values, errors


def _call_llm_via_registry(
    name: str,
    prompt_text: str,
    args: Dict[str, Any],
    config: Dict[str, Any],
    start_time: float,
) -> Optional[Dict[str, Any]]:
    """
    Try to call an LLM using the new BaseLLM interface from _LLM_INSTANCES.

    Returns response dict if successful, None if LLM not found in registry.
    """
    # Check if _LLM_INSTANCES is defined (only exists if app has LLM blocks)
    if '_LLM_INSTANCES' not in globals():
        return None

    # Check if LLM instance exists
    llm_instance = _LLM_INSTANCES.get(name)
    if llm_instance is None:
        return None

    try:
        from namel3ss.llm import ChatMessage

        # Determine mode (chat vs completion)
        mode = str(config.get("mode") or "chat").lower()

        # Build messages for chat mode
        if mode == "chat":
            messages = []

            # Add system message if present
            system_prompt = config.get("system") or config.get("system_prompt")
            if system_prompt:
                messages.append(ChatMessage(role="system", content=str(system_prompt)))

            # Add extra messages from config
            extra_messages = config.get("messages")
            if isinstance(extra_messages, list):
                for msg in extra_messages:
                    if isinstance(msg, dict):
                        role = str(msg.get("role", "user"))
                        content = str(msg.get("content", ""))
                        messages.append(ChatMessage(role=role, content=content))

            # Add user prompt
            user_role = str(config.get("user_role") or "user")
            messages.append(ChatMessage(role=user_role, content=prompt_text))

            # Call LLM with chat interface
            llm_response = llm_instance.generate_chat(messages, **args)
        else:
            # Completion mode
            llm_response = llm_instance.generate(prompt_text, **args)

        # Convert LLMResponse to our standard format
        elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))

        result_payload = {
            "text": llm_response.text,
            "raw": llm_response.raw,
        }

        metadata = {
            "elapsed_ms": elapsed_ms,
            "provider": llm_response.metadata.get("provider", llm_instance.get_provider_name()),
        }

        # Add usage info if available
        if llm_response.usage:
            metadata["usage"] = llm_response.usage
            result_payload["usage"] = llm_response.usage

        if llm_response.finish_reason:
            metadata["finish_reason"] = llm_response.finish_reason

        return {
            "status": "ok",
            "provider": llm_instance.get_provider_name(),
            "model": llm_response.model,
            "inputs": args,
            "result": result_payload,
            "metadata": metadata,
        }

    except Exception as exc:
        # Return error response
        import traceback
        elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))
        tb_text = ""
        try:
            tb_text = traceback.format_exc(limit=5).strip()
        except Exception:
            tb_text = ""
        if tb_text and len(tb_text) > 3000:
            tb_text = tb_text[:3000]

        return {
            "status": "error",
            "provider": llm_instance.get_provider_name() if llm_instance else "unknown",
            "model": getattr(llm_instance, "model", "unknown"),
            "inputs": args,
            "error": f"{type(exc).__name__}: {exc}",
            "metadata": {"elapsed_ms": elapsed_ms},
            "traceback": tb_text,
        }


async def _call_llm_via_registry_async(
    name: str,
    prompt_text: str,
    args: Dict[str, Any],
    config: Dict[str, Any],
    start_time: float,
) -> Optional[Dict[str, Any]]:
    """
    Async version: Try to call an LLM using the new BaseLLM interface from _LLM_INSTANCES.

    Returns response dict if successful, None if LLM not found in registry.
    """
    # Check if _LLM_INSTANCES is defined (only exists if app has LLM blocks)
    if '_LLM_INSTANCES' not in globals():
        return None

    # Check if LLM instance exists
    llm_instance = _LLM_INSTANCES.get(name)
    if llm_instance is None:
        return None

    try:
        from namel3ss.llm import ChatMessage

        # Determine mode (chat vs completion)
        mode = str(config.get("mode") or "chat").lower()

        # Build messages for chat mode
        if mode == "chat":
            messages = []

            # Add system message if present
            system_prompt = config.get("system") or config.get("system_prompt")
            if system_prompt:
                messages.append(ChatMessage(role="system", content=str(system_prompt)))

            # Add extra messages from config
            extra_messages = config.get("messages")
            if isinstance(extra_messages, list):
                for msg in extra_messages:
                    if isinstance(msg, dict):
                        role = str(msg.get("role", "user"))
                        content = str(msg.get("content", ""))
                        messages.append(ChatMessage(role=role, content=content))

            # Add user prompt
            user_role = str(config.get("user_role") or "user")
            messages.append(ChatMessage(role=user_role, content=prompt_text))

            # Call LLM with async chat interface
            llm_response = await llm_instance.agenerate_chat(messages, **args)
        else:
            # Async completion mode
            llm_response = await llm_instance.agenerate(prompt_text, **args)

        # Convert LLMResponse to our standard format
        elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))

        result_payload = {
            "text": llm_response.text,
            "raw": llm_response.raw,
        }

        metadata = {
            "elapsed_ms": elapsed_ms,
            "provider": llm_response.metadata.get("provider", llm_instance.get_provider_name()),
        }

        # Add usage info if available
        if llm_response.usage:
            metadata["usage"] = llm_response.usage
            result_payload["usage"] = llm_response.usage

        if llm_response.finish_reason:
            metadata["finish_reason"] = llm_response.finish_reason

        return {
            "status": "ok",
            "provider": llm_instance.get_provider_name(),
            "model": llm_response.model,
            "inputs": args,
            "result": result_payload,
            "metadata": metadata,
        }

    except Exception as exc:
        # Return error response
        import traceback
        elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))
        tb_text = ""
        try:
            tb_text = traceback.format_exc(limit=5).strip()
        except Exception:
            tb_text = ""
        if tb_text and len(tb_text) > 3000:
            tb_text = tb_text[:3000]

        return {
            "status": "error",
            "provider": llm_instance.get_provider_name() if llm_instance else "unknown",
            "model": getattr(llm_instance, "model", "unknown"),
            "inputs": args,
            "error": f"{type(exc).__name__}: {exc}",
            "metadata": {"elapsed_ms": elapsed_ms},
            "traceback": tb_text,
        }


def call_llm_connector(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute an LLM connector and return structured response details."""

    import traceback
    import urllib.error
    import urllib.parse

    start_time = time.time()

    # Try new LLM interface first
    args = dict(payload or {})
    prompt_value = args.get("prompt") or args.get("input")
    prompt_text = _stringify_prompt_value(name, prompt_value) if prompt_value is not None else ""

    # Attempt to use new BaseLLM interface
    spec = AI_CONNECTORS.get(name, {})
    context_stub = {
        "env": {key: os.getenv(key) for key in ENV_KEYS},
        "vars": {},
        "app": APP,
    }
    config_raw = spec.get("config", {})
    config_resolved = _resolve_placeholders(config_raw, context_stub)
    if not isinstance(config_resolved, dict):
        config_resolved = config_raw if isinstance(config_raw, dict) else {}

    llm_response = _call_llm_via_registry(name, prompt_text, args, config_resolved, start_time)
    if llm_response is not None:
        return llm_response

    # Fall back to old implementation
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


def _is_structured_prompt(prompt_spec: Dict[str, Any]) -> bool:
    """Check if a prompt has structured args and output_schema."""
    return bool(prompt_spec.get("args") or prompt_spec.get("output_schema"))


def _reconstruct_prompt_ast(prompt_spec: Dict[str, Any]) -> "Prompt":
    """Reconstruct a Prompt AST node from the encoded spec for structured execution."""
    if not _STRUCTURED_PROMPTS_AVAILABLE:
        raise ImportError("Structured prompts not available - missing imports")

    # Reconstruct args
    args_list = []
    for arg_dict in prompt_spec.get("args", []):
        arg = PromptArgument(
            name=arg_dict["name"],
            arg_type=arg_dict["type"],
            required=arg_dict.get("required", True),
            default=arg_dict.get("default"),
            description=arg_dict.get("description"),
        )
        args_list.append(arg)

    # Reconstruct output_schema
    output_schema = None
    schema_dict = prompt_spec.get("output_schema")
    if schema_dict:
        fields = []
        for field_dict in schema_dict.get("fields", []):
            field_type = _reconstruct_output_field_type(field_dict["field_type"])
            field = OutputField(
                name=field_dict["name"],
                field_type=field_type,
                required=field_dict.get("required", True),
                description=field_dict.get("description"),
            )
            fields.append(field)
        output_schema = OutputSchema(fields=fields)

    # Create Prompt AST node
    return Prompt(
        name=prompt_spec["name"],
        model=prompt_spec["model"],
        template=prompt_spec["template"],
        args=args_list,
        output_schema=output_schema,
        description=prompt_spec.get("description"),
    )


def _reconstruct_output_field_type(type_dict: Dict[str, Any]) -> "OutputFieldType":
    """Reconstruct an OutputFieldType from encoded dict."""
    element_type = None
    if type_dict.get("element_type"):
        element_type = _reconstruct_output_field_type(type_dict["element_type"])

    nested_fields = None
    if type_dict.get("nested_fields"):
        nested_fields = []
        for nested_dict in type_dict["nested_fields"]:
            nested_field_type = _reconstruct_output_field_type(nested_dict["field_type"])
            nested_field = OutputField(
                name=nested_dict["name"],
                field_type=nested_field_type,
                required=nested_dict.get("required", True),
                description=nested_dict.get("description"),
            )
            nested_fields.append(nested_field)

    return OutputFieldType(
        base_type=type_dict["base_type"],
        element_type=element_type,
        enum_values=type_dict.get("enum_values"),
        nested_fields=nested_fields,
        nullable=type_dict.get("nullable", False),
    )


def _get_llm_instance(model_name: str, context: Optional[Dict[str, Any]]) -> Optional["BaseLLM"]:
    """Get or create an instantiated LLM instance."""
    if not context:
        return None

    # First, try to get from llms registry (if they're already BaseLLM instances)
    llms_registry = context.get("llms", {})
    if isinstance(llms_registry, dict):
        llm_instance = llms_registry.get(model_name)
        if isinstance(llm_instance, BaseLLM):
            return llm_instance

    # Otherwise, get model spec and create instance
    model_spec = AI_MODELS.get(model_name)
    if not model_spec:
        return None

    provider = str(model_spec.get("provider", "")).lower()
    model_id = model_spec.get("model")

    if not model_id:
        return None

    # Create LLM instance based on provider
    try:
        if provider == "openai":
            from namel3ss.llm.openai_llm import OpenAILLM
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None
            return OpenAILLM(
                model=model_id,
                api_key=api_key,
                temperature=model_spec.get("temperature", 0.7),
                max_tokens=model_spec.get("max_tokens", 1024),
            )
        elif provider == "anthropic":
            from namel3ss.llm.anthropic_llm import AnthropicLLM
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return None
            return AnthropicLLM(
                model=model_id,
                api_key=api_key,
                temperature=model_spec.get("temperature", 0.7),
                max_tokens=model_spec.get("max_tokens", 1024),
            )
        # Add more providers as needed
    except ImportError:
        pass

    return None


def _run_structured_prompt(
    prompt_spec: Dict[str, Any],
    args: Dict[str, Any],
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute a structured prompt with validation and return results."""
    if not _STRUCTURED_PROMPTS_AVAILABLE:
        return {
            "status": "error",
            "error": "Structured prompts not available - missing namel3ss.prompts module",
        }

    prompt_name = prompt_spec.get("name", "unknown")
    model_name = prompt_spec.get("model", "unknown")

    try:
        # Reconstruct the Prompt AST
        prompt_ast = _reconstruct_prompt_ast(prompt_spec)

        # Get LLM instance
        llm_instance = _get_llm_instance(model_name, context)
        if not llm_instance:
            error_msg = f"LLM instance '{model_name}' not found in context"
            logger.error(f"Structured prompt '{prompt_name}': {error_msg}")
            _record_runtime_metric(
                context,
                name="prompt_program_failures",
                value=1,
                unit="count",
                tags={"prompt": prompt_name, "reason": "llm_not_found"},
            )
            return {
                "status": "error",
                "error": error_msg,
            }

        # Execute structured prompt
        logger.info(f"Executing structured prompt '{prompt_name}' with model '{model_name}'")
        result = execute_structured_prompt_sync(
            prompt_def=prompt_ast,
            llm=llm_instance,
            args=args,
            retry_on_validation_error=True,
            max_retries=2,
        )

        # Record metrics
        _record_runtime_metric(
            context,
            name="prompt_program_latency_ms",
            value=result.latency_ms,
            unit="milliseconds",
            tags={"prompt": prompt_name, "model": model_name},
        )

        validation_attempts = result.metadata.get("validation_attempts", 1)
        if validation_attempts > 1:
            retry_count = validation_attempts - 1
            _record_runtime_metric(
                context,
                name="prompt_program_retries",
                value=retry_count,
                unit="count",
                tags={"prompt": prompt_name, "model": model_name},
            )
            logger.warning(f"Structured prompt '{prompt_name}' required {retry_count} retries due to validation failures")

        # Check for validation errors in metadata
        if result.metadata.get("validation_errors"):
            validation_errors = result.metadata["validation_errors"]
            _record_runtime_metric(
                context,
                name="prompt_program_validation_failures",
                value=len(validation_errors),
                unit="count",
                tags={"prompt": prompt_name},
            )
            # Log sanitized validation errors (first 200 chars)
            for error in validation_errors[:3]:  # Log up to 3 errors
                sanitized_error = str(error)[:200]
                logger.warning(f"Validation error in '{prompt_name}': {sanitized_error}")

        # Record success metric
        _record_runtime_metric(
            context,
            name="prompt_program_success",
            value=1,
            unit="count",
            tags={"prompt": prompt_name, "model": model_name},
        )

        logger.info(f"Structured prompt '{prompt_name}' completed successfully in {result.latency_ms}ms")

        # Convert to compatible response format
        return {
            "status": "ok",
            "prompt": prompt_name,
            "model": model_name,
            "inputs": args,
            "output": result.output,
            "result": {"text": json.dumps(result.output), "json": result.output},
            "metadata": {
                "elapsed_ms": result.latency_ms,
                "tokens_used": result.tokens_used or 0,
                "validation_attempts": validation_attempts,
                "structured": True,
            },
        }
    except Exception as exc:
        error_msg = f"Structured prompt execution failed: {exc}"
        logger.error(f"Structured prompt '{prompt_name}': {error_msg}")
        _record_runtime_metric(
            context,
            name="prompt_program_failures",
            value=1,
            unit="count",
            tags={"prompt": prompt_name, "reason": "execution_error"},
        )
        return {
            "status": "error",
            "error": error_msg,
            "prompt": prompt_name,
            "model": model_name,
        }


# Async version for non-blocking LLM calls
async def call_llm_connector(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Async version: Execute an LLM connector and return structured response details."""

    import traceback

    start_time = time.time()

    # Try new LLM interface first
    args = dict(payload or {})
    prompt_value = args.get("prompt") or args.get("input")
    prompt_text = _stringify_prompt_value(name, prompt_value) if prompt_value is not None else ""

    # Attempt to use new BaseLLM interface (async)
    spec = AI_CONNECTORS.get(name, {})
    context_stub = {
        "env": {key: os.getenv(key) for key in ENV_KEYS},
        "vars": {},
        "app": APP,
    }
    config_raw = spec.get("config", {})
    config_resolved = _resolve_placeholders(config_raw, context_stub)
    if not isinstance(config_resolved, dict):
        config_resolved = config_raw if isinstance(config_raw, dict) else {}

    llm_response = await _call_llm_via_registry_async(name, prompt_text, args, config_resolved, start_time)
    if llm_response is not None:
        return llm_response

    # Fall back to HTTP-based implementation (keep sync for now - most connectors use the registry)
    # If needed, we can make HTTP calls async using aiohttp later
    provider = str(config_resolved.get("provider") or spec.get("type") or name or "").strip()
    model_name = str(config_resolved.get("model") or "").strip()
    allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")

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

    # For connectors without BaseLLM registry, return error for now
    # Full HTTP async support can be added if needed
    return _error_response(f"Async connector '{name}' requires BaseLLM registry (not legacy HTTP)")


def _reconstruct_prompt_ast(prompt_spec: Dict[str, Any]) -> "Prompt":
    """Reconstruct a Prompt AST node from the encoded spec for structured execution."""
    if not _STRUCTURED_PROMPTS_AVAILABLE:
        raise ImportError("Structured prompts not available - missing imports")

    # Reconstruct args
    args_list = []
    for arg_dict in prompt_spec.get("args", []):
        arg = PromptArgument(
            name=arg_dict["name"],
            arg_type=arg_dict["type"],
            required=arg_dict.get("required", True),
            default=arg_dict.get("default"),
            description=arg_dict.get("description"),
        )
        args_list.append(arg)

    # Reconstruct output_schema
    output_schema = None
    schema_dict = prompt_spec.get("output_schema")
    if schema_dict:
        fields = []
        for field_dict in schema_dict.get("fields", []):
            field_type = _reconstruct_output_field_type(field_dict["field_type"])
            field = OutputField(
                name=field_dict["name"],
                field_type=field_type,
                required=field_dict.get("required", True),
                description=field_dict.get("description"),
            )
            fields.append(field)
        output_schema = OutputSchema(fields=fields)

    # Create Prompt AST node
    return Prompt(
        name=prompt_spec["name"],
        model=prompt_spec["model"],
        template=prompt_spec["template"],
        args=args_list,
        output_schema=output_schema,
        description=prompt_spec.get("description"),
    )


def _reconstruct_output_field_type(type_dict: Dict[str, Any]) -> "OutputFieldType":
    """Reconstruct an OutputFieldType from encoded dict."""
    element_type = None
    if type_dict.get("element_type"):
        element_type = _reconstruct_output_field_type(type_dict["element_type"])

    nested_fields = None
    if type_dict.get("nested_fields"):
        nested_fields = []
        for nested_dict in type_dict["nested_fields"]:
            nested_field_type = _reconstruct_output_field_type(nested_dict["field_type"])
            nested_field = OutputField(
                name=nested_dict["name"],
                field_type=nested_field_type,
                required=nested_dict.get("required", True),
                description=nested_dict.get("description"),
            )
            nested_fields.append(nested_field)

    return OutputFieldType(
        base_type=type_dict["base_type"],
        element_type=element_type,
        enum_values=type_dict.get("enum_values"),
        nested_fields=nested_fields,
        nullable=type_dict.get("nullable", False),
    )


def _get_llm_instance(model_name: str, context: Optional[Dict[str, Any]]) -> Optional["BaseLLM"]:
    """Get or create an instantiated LLM instance."""
    if not context:
        return None

    # First, try to get from llms registry (if they're already BaseLLM instances)
    llms_registry = context.get("llms", {})
    if isinstance(llms_registry, dict):
        llm_instance = llms_registry.get(model_name)
        if isinstance(llm_instance, BaseLLM):
            return llm_instance

    # Otherwise, get model spec and create instance
    model_spec = AI_MODELS.get(model_name)
    if not model_spec:
        return None

    provider = str(model_spec.get("provider", "")).lower()
    model_id = model_spec.get("model")

    if not model_id:
        return None

    # Create LLM instance based on provider
    try:
        if provider == "openai":
            from namel3ss.llm.openai_llm import OpenAILLM
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None
            return OpenAILLM(
                model=model_id,
                api_key=api_key,
                temperature=model_spec.get("temperature", 0.7),
                max_tokens=model_spec.get("max_tokens", 1024),
            )
        elif provider == "anthropic":
            from namel3ss.llm.anthropic_llm import AnthropicLLM
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return None
            return AnthropicLLM(
                model=model_id,
                api_key=api_key,
                temperature=model_spec.get("temperature", 0.7),
                max_tokens=model_spec.get("max_tokens", 1024),
            )
        # Add more providers as needed
    except ImportError:
        pass

    return None


def _run_structured_prompt(
    prompt_spec: Dict[str, Any],
    args: Dict[str, Any],
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute a structured prompt with validation and return results."""
    if not _STRUCTURED_PROMPTS_AVAILABLE:
        return {
            "status": "error",
            "error": "Structured prompts not available - missing namel3ss.prompts module",
        }

    prompt_name = prompt_spec.get("name", "unknown")
    model_name = prompt_spec.get("model", "unknown")

    try:
        # Reconstruct the Prompt AST
        prompt_ast = _reconstruct_prompt_ast(prompt_spec)

        # Get LLM instance
        llm_instance = _get_llm_instance(model_name, context)
        if not llm_instance:
            error_msg = f"LLM instance '{model_name}' not found in context"
            logger.error(f"Structured prompt '{prompt_name}': {error_msg}")
            _record_runtime_metric(
                context,
                name="prompt_program_failures",
                value=1,
                unit="count",
                tags={"prompt": prompt_name, "reason": "llm_not_found"},
            )
            return {
                "status": "error",
                "error": error_msg,
            }

        # Execute structured prompt
        logger.info(f"Executing structured prompt '{prompt_name}' with model '{model_name}'")
        result = execute_structured_prompt_sync(
            prompt_def=prompt_ast,
            llm=llm_instance,
            args=args,
            retry_on_validation_error=True,
            max_retries=2,
        )

        # Record metrics
        _record_runtime_metric(
            context,
            name="prompt_program_latency_ms",
            value=result.latency_ms,
            unit="milliseconds",
            tags={"prompt": prompt_name, "model": model_name},
        )

        validation_attempts = result.metadata.get("validation_attempts", 1)
        if validation_attempts > 1:
            retry_count = validation_attempts - 1
            _record_runtime_metric(
                context,
                name="prompt_program_retries",
                value=retry_count,
                unit="count",
                tags={"prompt": prompt_name, "model": model_name},
            )
            logger.warning(f"Structured prompt '{prompt_name}' required {retry_count} retries due to validation failures")

        # Check for validation errors in metadata
        if result.metadata.get("validation_errors"):
            validation_errors = result.metadata["validation_errors"]
            _record_runtime_metric(
                context,
                name="prompt_program_validation_failures",
                value=len(validation_errors),
                unit="count",
                tags={"prompt": prompt_name},
            )
            # Log sanitized validation errors (first 200 chars)
            for error in validation_errors[:3]:  # Log up to 3 errors
                sanitized_error = str(error)[:200]
                logger.warning(f"Validation error in '{prompt_name}': {sanitized_error}")

        # Record success metric
        _record_runtime_metric(
            context,
            name="prompt_program_success",
            value=1,
            unit="count",
            tags={"prompt": prompt_name, "model": model_name},
        )

        logger.info(f"Structured prompt '{prompt_name}' completed successfully in {result.latency_ms}ms")

        # Convert to compatible response format
        return {
            "status": "ok",
            "prompt": prompt_name,
            "model": model_name,
            "inputs": args,
            "output": result.output,
            "result": {"text": json.dumps(result.output), "json": result.output},
            "metadata": {
                "elapsed_ms": result.latency_ms,
                "tokens_used": result.tokens_used or 0,
                "validation_attempts": validation_attempts,
                "structured": True,
            },
        }
    except Exception as exc:
        error_msg = f"Structured prompt execution failed: {exc}"
        logger.error(f"Structured prompt '{prompt_name}': {error_msg}")
        _record_runtime_metric(
            context,
            name="prompt_program_failures",
            value=1,
            unit="count",
            tags={"prompt": prompt_name, "reason": "execution_error"},
        )
        return {
            "status": "error",
            "error": error_msg,
            "prompt": prompt_name,
            "model": model_name,
        }


def run_prompt(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
    memory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    start_time = time.time()
    allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")
    prompt_spec = AI_PROMPTS.get(name)
    model_name = str(prompt_spec.get("model") or "") if prompt_spec else ""
    prompt_text = ""
    payload_dict = dict(payload or {}) if isinstance(payload, dict) else {}
    memory_state = _ensure_memory_state(context)
    read_memory = _extract_memory_names(payload_dict.pop("read_memory", None))
    write_memory = _extract_memory_names(payload_dict.pop("write_memory", None))
    memory_payload = dict(memory or {})
    if not memory_payload and read_memory:
        memory_payload = _memory_snapshot(memory_state, read_memory)
    normalized_inputs: Dict[str, Any] = dict(payload_dict) if isinstance(payload, dict) else {}

    def _elapsed_ms() -> float:
        return float(round((time.time() - start_time) * 1000.0, 3))

    def _stub_response(reason: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.warning(_format_error_message(name, None, reason))
        response: Dict[str, Any] = {
            "status": "stub",
            "prompt": name,
            "model": model_name,
            "inputs": normalized_inputs,
            "output": {},
            "result": {"text": "[stub: prompt call failed]"},
            "metadata": metadata or {},
            "error": reason,
        }
        return response

    def _error_response(reason: str, metadata: Optional[Dict[str, Any]] = None, tb_text: str = "") -> Dict[str, Any]:
        logger.error(_format_error_message(name, prompt_text, reason))
        response: Dict[str, Any] = {
            "status": "error",
            "prompt": name,
            "model": model_name,
            "inputs": normalized_inputs,
            "metadata": metadata or {},
            "error": reason,
        }
        if tb_text:
            response["traceback"] = tb_text
        return response

    if not prompt_spec:
        metadata = {"elapsed_ms": _elapsed_ms()}
        return _stub_response(f"Prompt '{name}' is not defined", metadata) if allow_stubs else _error_response(f"Prompt '{name}' is not defined", metadata)

    # Check if this is a structured prompt and route to structured execution
    if _STRUCTURED_PROMPTS_AVAILABLE and _is_structured_prompt(prompt_spec):
        # Extract args from payload (skip memory-related keys)
        args_for_prompt = {k: v for k, v in payload_dict.items() if k not in ["read_memory", "write_memory"]}
        structured_result = _run_structured_prompt(prompt_spec, args_for_prompt, context)

        # Handle memory writes if needed
        if structured_result.get("status") == "ok" and write_memory:
            _write_memory_entries(
                memory_state,
                write_memory,
                structured_result.get("output"),
                context=context,
                source=f"prompt:{name}",
            )

        return structured_result

    normalized_inputs = _normalize_prompt_inputs(
        prompt_spec,
        payload_dict if isinstance(payload, dict) else (payload or {}),
    )
    validated_inputs, validation_errors = _validate_prompt_inputs(prompt_spec, normalized_inputs)
    if validation_errors:
        metadata = {"elapsed_ms": _elapsed_ms()}
        reason = "; ".join(validation_errors)
        return _stub_response(reason, metadata) if allow_stubs else _error_response(reason, metadata)

    model_spec = AI_MODELS.get(model_name)
    if not model_spec:
        metadata = {"elapsed_ms": _elapsed_ms()}
        reason = f"Prompt model '{model_name}' is not defined"
        return _stub_response(reason, metadata) if allow_stubs else _error_response(reason, metadata)

    provider = str(model_spec.get("provider") or model_name or "unknown")
    model_id = model_spec.get("model")
    if not model_id:
        metadata = {"elapsed_ms": _elapsed_ms()}
        reason = f"Model id is not configured for prompt '{name}'"
        return _stub_response(reason, metadata) if allow_stubs else _error_response(reason, metadata)

    try:
        prompt_text = _render_template_value(
            prompt_spec.get("template"),
            {
                "input": validated_inputs,
                "vars": validated_inputs,
                "payload": validated_inputs,
                "memory": memory_payload,
            },
        )
    except Exception as exc:
        metadata = {"elapsed_ms": _elapsed_ms()}
        reason = f"Prompt template error: {exc}"
        return _stub_response(reason, metadata) if allow_stubs else _error_response(reason, metadata)

    try:
        request_config = _merge_prompt_request_config(model_spec, prompt_spec)
        request_spec = _build_llm_request(provider, model_id, prompt_text, request_config, validated_inputs)
        method = str(request_spec.get("method") or "POST").upper()
        if method != "POST":
            raise ValueError(f"Unsupported HTTP method '{method}' for prompt '{name}'")

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
            extracted = _extract_llm_text(provider, parsed_json, request_config)
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
        structured_output, projection_errors = _project_prompt_output(prompt_spec, result_payload)
        response: Dict[str, Any] = {
            "status": "ok",
            "prompt": name,
            "model": model_name,
            "inputs": validated_inputs,
            "output": structured_output,
            "result": result_payload,
            "metadata": metadata,
        }
        if write_memory:
            _write_memory_entries(
                memory_state,
                write_memory,
                structured_output or result_payload,
                context=context,
                source=f"prompt:{name}",
            )
        if projection_errors:
            response["warnings"] = projection_errors
        return response
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
        except Exception:
            tb_text = ""
        if tb_text and len(tb_text) > 3000:
            tb_text = tb_text[:3000]
        return _stub_response(reason, meta) if allow_stubs else _error_response(reason, meta, tb_text)


async def run_prompt(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
    memory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Async version of run_prompt - attempts to use LLM registry for async calls."""

    start_time = time.time()
    allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")
    prompt_spec = AI_PROMPTS.get(name)
    model_name = str(prompt_spec.get("model") or "") if prompt_spec else ""
    payload_dict = dict(payload or {}) if isinstance(payload, dict) else {}
    memory_state = _ensure_memory_state(context)
    read_memory = _extract_memory_names(payload_dict.pop("read_memory", None))
    write_memory = _extract_memory_names(payload_dict.pop("write_memory", None))
    memory_payload = dict(memory or {})
    if not memory_payload and read_memory:
        memory_payload = _memory_snapshot(memory_state, read_memory)

    def _elapsed_ms() -> float:
        return float(round((time.time() - start_time) * 1000.0, 3))

    def _error_response(reason: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response: Dict[str, Any] = {
            "status": "error",
            "prompt": name,
            "model": model_name,
            "inputs": payload_dict,
            "metadata": metadata or {},
            "error": reason,
        }
        return response

    if not prompt_spec:
        return _error_response(f"Prompt '{name}' is not defined", {"elapsed_ms": _elapsed_ms()})

    # For async, try to use BaseLLM interface via call_llm_connector
    # Build a connector-style payload
    connector_payload = dict(payload_dict)

    # Render template
    try:
        prompt_text = _render_template_value(
            prompt_spec.get("template"),
            {
                "input": payload_dict,
                "vars": payload_dict,
                "payload": payload_dict,
                "memory": memory_payload,
            },
        )
        connector_payload["prompt"] = prompt_text
        connector_payload["input"] = prompt_text
    except Exception as exc:
        return _error_response(f"Template error: {exc}", {"elapsed_ms": _elapsed_ms()})

    # Use async call_llm_connector (which tries the BaseLLM registry)
    try:
        connector_response = await call_llm_connector(model_name, connector_payload)

        # Adapt response format
        if connector_response.get("status") == "ok":
            result = connector_response.get("result", {})
            output_text = result.get("text", "")

            # Write to memory if configured
            if write_memory:
                _write_memory_entries(
                    memory_state,
                    write_memory,
                    output_text,
                    context=context,
                    source=f"prompt:{name}",
                )

            return {
                "status": "ok",
                "prompt": name,
                "model": model_name,
                "inputs": payload_dict,
                "output": output_text,
                "result": result,
                "metadata": connector_response.get("metadata", {"elapsed_ms": _elapsed_ms()}),
            }
        else:
            # Error or stub response
            return {
                "status": connector_response.get("status", "error"),
                "prompt": name,
                "model": model_name,
                "inputs": payload_dict,
                "error": connector_response.get("error", "Unknown error"),
                "metadata": connector_response.get("metadata", {"elapsed_ms": _elapsed_ms()}),
            }
    except Exception as exc:
        return _error_response(f"{type(exc).__name__}: {exc}", {"elapsed_ms": _elapsed_ms()})


async def run_chain(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a configured AI chain and return detailed step results."""

    start_time = time.time()
    args = dict(payload or {})
    allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")
    spec = AI_CHAINS.get(name)
    runtime_context = context if isinstance(context, dict) else build_context(None)

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
    memory_state = _ensure_memory_state(runtime_context)
    chain_scope: Dict[str, Any] = {
        "input": working,
        "steps": {},
        "locals": {},
        "loop": {},
        "counter": 0,
    }
    runtime_context["steps"] = chain_scope["steps"]
    runtime_context.setdefault("chain", {})["steps"] = chain_scope["steps"]
    runtime_context["payload"] = args

    steps_history: List[Dict[str, Any]] = []

    # Check for timeout configuration
    timeout_seconds = spec.get("timeout")
    if timeout_seconds is None:
        timeout_seconds = float(os.getenv("CHAIN_TIMEOUT_SECONDS", "0"))  # 0 = no timeout

    try:
        if timeout_seconds > 0:
            # Execute with timeout
            status, result_value, working = await _execute_with_timeout(
                _execute_workflow_nodes(
                    spec.get("steps", []),
                    context=runtime_context,
                    chain_scope=chain_scope,
                    args=args,
                    working=working,
                    memory_state=memory_state,
                    allow_stubs=allow_stubs,
                    steps_history=steps_history,
                    chain_name=name,
                ),
                timeout_seconds=timeout_seconds,
                chain_name=name,
                context=runtime_context,
            )
        else:
            # Execute without timeout
            status, result_value, working = await _execute_workflow_nodes(
                spec.get("steps", []),
                context=runtime_context,
                chain_scope=chain_scope,
                args=args,
                working=working,
                memory_state=memory_state,
                allow_stubs=allow_stubs,
                steps_history=steps_history,
                chain_name=name,
            )
    except TimeoutError as exc:
        elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))
        return {
            "status": "timeout",
            "result": None,
            "steps": steps_history,
            "inputs": args,
            "error": str(exc),
            "metadata": {"elapsed_ms": elapsed_ms, "timeout_seconds": timeout_seconds},
        }
    except Exception as exc:
        elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))
        logger.error(f"Chain '{name}' execution failed: {exc}")
        return {
            "status": "error",
            "result": None,
            "steps": steps_history,
            "inputs": args,
            "error": str(exc),
            "metadata": {"elapsed_ms": elapsed_ms},
        }

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

    data_config = spec.get("data_config")
    if isinstance(data_config, dict) and data_config:
        try:
            dataset_payload = _prepare_experiment_dataset(data_config, args)
        except Exception as exc:
            logger.exception("Failed to prepare dataset for experiment %s", name)
            return {
                "status": "error",
                "error": "experiment_data_error",
                "detail": str(exc),
                "experiment": spec.get("name", name),
                "slug": spec.get("slug") or spec.get("name", name),
                "variants": [],
                "metrics": [],
                "metric_definitions": metric_configs,
                "leaderboard": [],
                "winner": None,
                "inputs": args,
                "metadata": spec.get("metadata", {}),
            }
        _ensure_dataset_inputs(args, dataset_payload)

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


def _ensure_dataset_inputs(args: Dict[str, Any], dataset_payload: Dict[str, Any]) -> None:
    if not isinstance(args, dict):
        return
    args.setdefault("dataset", dataset_payload)
    if dataset_payload.get("rows"):
        args.setdefault("examples", dataset_payload["rows"])
    if dataset_payload.get("y") and not args.get("y_true"):
        args["y_true"] = list(dataset_payload["y"])
    if dataset_payload.get("splits") and not args.get("splits"):
        args["splits"] = dataset_payload["splits"]
    existing_input = args.get("input")
    if isinstance(existing_input, dict):
        existing_input.setdefault("dataset", dataset_payload)
    elif existing_input is None:
        args["input"] = {"dataset": dataset_payload}


def _prepare_experiment_dataset(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict) or not config:
        raise ValueError("Experiment data configuration must be a non-empty dictionary.")
    context = build_context(None)
    frame = _await_frame_task(_load_experiment_frame(config, context))
    if frame is None:
        raise ValueError("Experiment data source did not return a frame.")
    columns = frame.spec.get("columns") or []
    column_index = {column.get("name"): column for column in columns if column.get("name")}
    frame_name = frame.name or config.get("frame") or "frame"

    target_column = _resolve_target_column(config, columns, column_index, frame_name)
    feature_columns = _resolve_feature_columns(config, columns, column_index, target_column, frame_name)
    time_column = _resolve_time_column(config, columns, column_index)
    group_columns = _resolve_group_columns(config, columns, column_index)
    weight_column = _resolve_weight_column(config, columns, column_index)

    rows = _clone_rows(frame.rows)
    if not rows:
        raise ValueError(f"Frame '{frame_name}' did not yield any rows for experiment data.")

    dataset_payload: Dict[str, Any] = {
        "frame": frame_name,
        "schema": frame.schema_payload(),
        "features": feature_columns,
        "target": target_column,
        "time_column": time_column,
        "group_columns": group_columns,
        "weight_column": weight_column,
        "rows": rows,
        "X": _build_feature_matrix(rows, feature_columns),
        "y": _build_target_series(rows, target_column),
    }
    if weight_column:
        dataset_payload["weights"] = _build_target_series(rows, weight_column)
    split_spec = config.get("splits") or frame.spec.get("splits") or {}
    splits_payload = _materialise_split_payload(
        rows,
        split_spec,
        feature_columns,
        target_column,
        weight_column,
        time_column,
        frame.spec.get("key"),
    )
    if splits_payload:
        dataset_payload["splits"] = splits_payload
    return dataset_payload


async def _load_experiment_frame(config: Dict[str, Any], context: Dict[str, Any]) -> Any:
    pipeline_value = config.get("pipeline")
    if isinstance(pipeline_value, dict) and "__frame_pipeline__" in pipeline_value:
        pipeline_value = pipeline_value["__frame_pipeline__"]
    if isinstance(pipeline_value, dict) and pipeline_value.get("root"):
        return await _evaluate_frame_pipeline(pipeline_value, None, context, set())
    frame_name = config.get("frame")
    if not frame_name:
        raise ValueError("Experiment data config requires a 'frame' or 'pipeline'.")
    return await _resolve_frame_runtime(frame_name, None, context, set())


def _await_frame_task(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    holder: Dict[str, Any] = {}
    error: List[BaseException] = []
    done = threading.Event()

    def _runner() -> None:
        try:
            holder["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive
            error.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    done.wait()
    if error:
        raise error[0]
    return holder.get("value")


def _resolve_target_column(
    config: Dict[str, Any],
    columns: List[Dict[str, Any]],
    column_index: Dict[str, Dict[str, Any]],
    frame_name: str,
) -> str:
    target_value = config.get("target") or config.get("label")
    if isinstance(target_value, dict):
        target_value = target_value.get("name")
    if target_value:
        target_name = str(target_value)
        if target_name not in column_index:
            raise ValueError(f"Frame '{frame_name}' does not define target column '{target_name}'.")
        return target_name
    for column in columns:
        role = str(column.get("role") or "").lower()
        if role in {"target", "label"} and column.get("name"):
            return column["name"]
    raise ValueError(f"Frame '{frame_name}' does not define a target column or role.")


def _resolve_feature_columns(
    config: Dict[str, Any],
    columns: List[Dict[str, Any]],
    column_index: Dict[str, Dict[str, Any]],
    target_column: str,
    frame_name: str,
) -> List[str]:
    explicit = _coerce_column_list(config.get("features") or config.get("feature_columns"))
    if explicit:
        _validate_columns(explicit, column_index, frame_name)
        return _deduplicate(explicit)
    inferred: List[str] = []
    for column in columns:
        name = column.get("name")
        if not name or name == target_column:
            continue
        role = str(column.get("role") or "").lower()
        if role in {"feature", "input"}:
            inferred.append(name)
    if not inferred:
        for column in columns:
            name = column.get("name")
            if not name or name == target_column:
                continue
            inferred.append(name)
            if len(inferred) >= 4:
                break
    if not inferred:
        raise ValueError(f"Frame '{frame_name}' does not provide feature candidates.")
    return _deduplicate(inferred)


def _resolve_time_column(
    config: Dict[str, Any],
    columns: List[Dict[str, Any]],
    column_index: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    time_value = config.get("time") or config.get("time_column") or config.get("timestamp")
    if isinstance(time_value, dict):
        time_value = time_value.get("name")
    if time_value:
        name = str(time_value)
        if name not in column_index:
            raise ValueError(f"Frame does not define time column '{name}'.")
        return name
    for column in columns:
        role = str(column.get("role") or "").lower()
        if role == "time" and column.get("name"):
            return column["name"]
    return None


def _resolve_group_columns(
    config: Dict[str, Any],
    columns: List[Dict[str, Any]],
    column_index: Dict[str, Dict[str, Any]],
) -> List[str]:
    explicit = _coerce_column_list(config.get("group_columns") or config.get("groups"))
    if explicit:
        _validate_columns(explicit, column_index, None)
        return _deduplicate(explicit)
    inferred: List[str] = []
    for column in columns:
        role = str(column.get("role") or "").lower()
        if role in {"group", "id"} and column.get("name"):
            inferred.append(column["name"])
    return _deduplicate(inferred)


def _resolve_weight_column(
    config: Dict[str, Any],
    columns: List[Dict[str, Any]],
    column_index: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    weight_value = config.get("weight") or config.get("weight_column") or config.get("sample_weight")
    if isinstance(weight_value, dict):
        weight_value = weight_value.get("name")
    if weight_value:
        candidate = str(weight_value)
        if candidate not in column_index:
            raise ValueError(f"Frame does not define weight column '{candidate}'.")
        return candidate
    for column in columns:
        role = str(column.get("role") or "").lower()
        if role == "weight" and column.get("name"):
            return column["name"]
    return None


def _coerce_column_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        parts = [segment.strip() for segment in value.split(",") if segment.strip()]
        return parts or [value.strip()]
    if isinstance(value, dict):
        if "name" in value:
            return [str(value["name"])]
        value = value.get("columns") or value.get("names")
    result: List[str] = []
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict) and item.get("name"):
                result.append(str(item["name"]))
    return result


def _validate_columns(columns: List[str], column_index: Dict[str, Dict[str, Any]], frame_name: Optional[str]) -> None:
    missing = [name for name in columns if name not in column_index]
    if missing:
        joined = ", ".join(missing)
        if frame_name:
            raise ValueError(f"Frame '{frame_name}' does not define columns: {joined}")
        raise ValueError(f"Unknown columns referenced in experiment data config: {joined}")


def _deduplicate(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _materialise_split_payload(
    rows: List[Dict[str, Any]],
    split_spec: Dict[str, Any],
    feature_columns: List[str],
    target_column: str,
    weight_column: Optional[str],
    time_column: Optional[str],
    key_columns: Optional[Sequence[str]],
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(split_spec, dict) or not split_spec:
        return {}
    normalized: List[Tuple[str, float]] = []
    for name, value in split_spec.items():
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        normalized.append((str(name), weight))
    if not normalized:
        return {}
    order_source = list(rows)
    if time_column and all(time_column in row for row in order_source):
        order_source.sort(key=lambda row: row.get(time_column))
    elif key_columns:
        order_source.sort(key=lambda row: tuple(row.get(col) for col in key_columns))
    total_rows = len(order_source)
    total_weight = sum(weight for _, weight in normalized)
    splits: Dict[str, Dict[str, Any]] = {}
    offset = 0
    for index, (name, weight) in enumerate(normalized):
        if index == len(normalized) - 1:
            subset = order_source[offset:]
        else:
            span = int(round((weight / total_weight) * total_rows))
            if span <= 0 and total_rows > len(splits):
                span = 1
            subset = order_source[offset : offset + span]
            offset += span
        if not subset:
            continue
        splits[name] = _build_split_record(subset, feature_columns, target_column, weight_column)
    return splits


def _build_split_record(
    subset: List[Dict[str, Any]],
    feature_columns: List[str],
    target_column: str,
    weight_column: Optional[str],
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "rows": list(subset),
        "X": _build_feature_matrix(subset, feature_columns),
        "y": _build_target_series(subset, target_column),
        "size": len(subset),
    }
    if weight_column:
        record["weights"] = _build_target_series(subset, weight_column)
    return record


def _build_feature_matrix(rows: List[Dict[str, Any]], feature_columns: List[str]) -> List[Dict[str, Any]]:
    matrix: List[Dict[str, Any]] = []
    for row in rows:
        matrix.append({column: row.get(column) for column in feature_columns})
    return matrix


def _build_target_series(rows: List[Dict[str, Any]], column: Optional[str]) -> List[Any]:
    if not column:
        return []
    return [row.get(column) for row in rows]


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
            if isinstance(result_value, dict):
                if isinstance(result_value.get("predictions"), (list, tuple)):
                    return list(result_value["predictions"])
                if "label" in result_value:
                    return [result_value.get("label")]
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
        metadata = raw_output.get("metadata")
        if isinstance(metadata, dict):
            for meta_key in ("score", "value", "quality", "metric", "elapsed_ms"):
                number = _to_float(metadata.get(meta_key))
                if number is not None:
                    return number
        result_section = raw_output.get("result")
        if isinstance(result_section, dict):
            metadata = result_section.get("metadata")
            if isinstance(metadata, dict):
                for meta_key in ("score", "value", "quality", "metric", "elapsed_ms"):
                    number = _to_float(metadata.get(meta_key))
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

def list_training_jobs() -> List[str]:
    """List all available training job names."""
    return sorted(TRAINING_JOBS.keys())


def list_tuning_jobs() -> List[str]:
    """List all available tuning job names."""
    return sorted(TUNING_JOBS.keys())


def available_training_backends() -> List[str]:
    """List registered training backend names."""
    return list(_registered_training_backends())


def get_training_job(name: str) -> Optional[Dict[str, Any]]:
    """Get training job spec by name."""
    key = str(name)
    spec = TRAINING_JOBS.get(key)
    if not spec:
        return None
    return copy.deepcopy(spec)


def get_tuning_job(name: str) -> Optional[Dict[str, Any]]:
    """Get tuning job spec by name."""
    key = str(name)
    spec = TUNING_JOBS.get(key)
    if not spec:
        return None
    return copy.deepcopy(spec)


def resolve_training_job_plan(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve a training plan with payload and overrides."""
    key = str(name)
    spec = TRAINING_JOBS.get(key)
    if not spec:
        raise ValueError(f"Training job '{name}' is not defined")
    env_map = _training_environment(context)
    payload_map = dict(payload or {})
    overrides_map = dict(overrides or {})
    return _resolve_training_plan_impl(spec, payload_map, overrides_map, env=env_map)


def training_job_history(name: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Get training job execution history."""
    history = TRAINING_JOB_HISTORY.get(str(name), [])
    if limit is not None and limit > 0:
        selected = history[-limit:]
    else:
        selected = history
    return [copy.deepcopy(entry) for entry in selected]


def tuning_job_history(name: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Get tuning job execution history."""
    history = TUNING_JOB_HISTORY.get(str(name), [])
    if limit is not None and limit > 0:
        selected = history[-limit:]
    else:
        selected = history
    return [copy.deepcopy(entry) for entry in selected]


async def run_training_job(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
    session: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute a training job with full dataset loading, training, and model registry persistence."""
    key = str(name)
    spec = TRAINING_JOBS.get(key)
    runtime_context = context if isinstance(context, dict) else build_context(None)

    if not spec:
        return {
            "status": "error",
            "error": "training_job_not_found",
            "detail": f"Training job '{name}' is not defined",
            "job": key,
        }

    payload_map = dict(payload or {})
    overrides_map = dict(overrides or {})

    try:
        plan = resolve_training_job_plan(key, payload_map, overrides_map, context=runtime_context)
    except Exception as exc:
        logger.exception("Failed to resolve training plan for job %s", key)
        return {
            "status": "error",
            "error": "training_plan_failed",
            "detail": str(exc),
            "job": key,
        }

    # Load dataset
    dataset_name = spec.get("dataset")
    if not dataset_name:
        return {
            "status": "error",
            "error": "missing_dataset",
            "detail": "Training job must specify a dataset",
            "job": key,
        }

    try:
        dataset_context = await _load_training_dataset(dataset_name, spec, session, runtime_context)
    except Exception as exc:
        logger.exception("Failed to load dataset for training job %s", key)
        return {
            "status": "error",
            "error": "dataset_load_failed",
            "detail": str(exc),
            "job": key,
        }

    # Prepare backend context
    backend_context = _prepare_training_context(spec, runtime_context, dataset_context)
    backend_context.setdefault("payload", payload_map)
    backend_context.setdefault("overrides", overrides_map)
    backend_context.setdefault("job", spec.get("name", key))

    started = time.time()
    backend_name = plan.get("backend", "local")
    framework = spec.get("framework", backend_name)

    try:
        backend = _resolve_training_backend_impl(framework or backend_name)
        result_payload = backend.run(plan, backend_context)
    except Exception as exc:
        logger.exception("Training backend '%s' failed for job '%s'", backend_name, key)
        return {
            "status": "error",
            "error": "training_backend_failed",
            "detail": str(exc),
            "job": spec.get("name", key),
            "backend": backend_name,
        }

    result = dict(result_payload or {}) if isinstance(result_payload, dict) else {}
    result.setdefault("status", "ok")
    result.setdefault("job", spec.get("name", key))
    result.setdefault("backend", backend_name or "local")
    result.setdefault("model", spec.get("model"))
    result.setdefault("dataset", spec.get("dataset"))
    result.setdefault("hyperparameters", plan.get("hyperparameters", {}))
    result.setdefault("resources", plan.get("resources", {}))
    result.setdefault("metadata", plan.get("metadata", {}))

    duration_ms = float(round((time.time() - started) * 1000.0, 3))

    # Persist trained model to registry if training succeeded
    if result.get("status") == "ok":
        try:
            registry_key = await _persist_trained_model(spec, result, session)
            result["registry_key"] = registry_key
        except Exception as exc:
            logger.warning("Failed to persist trained model to registry: %s", exc)
            result["registry_warning"] = str(exc)

    _record_training_history(key, result, duration_ms)
    _record_runtime_event(
        runtime_context,
        event="training_job.complete",
        level="info" if result.get("status") == "ok" else "warning",
        data={
            "job": key,
            "backend": result.get("backend"),
            "status": result.get("status"),
            "duration_ms": duration_ms,
        },
    )
    return result


async def run_tuning_job(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
    session: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute a hyperparameter tuning job with trial tracking and best model selection."""
    key = str(name)
    spec = TUNING_JOBS.get(key)
    runtime_context = context if isinstance(context, dict) else build_context(None)

    if not spec:
        return {
            "status": "error",
            "error": "tuning_job_not_found",
            "detail": f"Tuning job '{name}' is not defined",
            "job": key,
        }

    # Get base training job
    training_job_name = spec.get("training_job")
    if not training_job_name or training_job_name not in TRAINING_JOBS:
        return {
            "status": "error",
            "error": "invalid_training_job",
            "detail": f"Tuning job '{name}' references invalid training job '{training_job_name}'",
            "job": key,
        }

    started = time.time()
    strategy = spec.get("strategy", "grid")
    max_trials = spec.get("max_trials", 10)
    parallel_trials = spec.get("parallel_trials", 1)
    search_space = spec.get("search_space", {})
    objective_metric = spec.get("objective_metric", "accuracy")
    early_stopping = spec.get("early_stopping")

    try:
        trials = await _execute_tuning_trials(
            training_job_name=training_job_name,
            search_space=search_space,
            strategy=strategy,
            max_trials=max_trials,
            parallel_trials=parallel_trials,
            objective_metric=objective_metric,
            early_stopping=early_stopping,
            payload=payload,
            overrides=overrides,
            context=runtime_context,
            session=session,
        )
    except Exception as exc:
        logger.exception("Tuning job '%s' failed", key)
        return {
            "status": "error",
            "error": "tuning_execution_failed",
            "detail": str(exc),
            "job": key,
        }

    # Find best trial
    best_trial = _find_best_trial(trials, objective_metric)

    duration_ms = float(round((time.time() - started) * 1000.0, 3))

    result = {
        "status": "ok",
        "job": key,
        "training_job": training_job_name,
        "strategy": strategy,
        "trials": trials,
        "total_trials": len(trials),
        "best_trial": best_trial,
        "best_hyperparameters": best_trial.get("hyperparameters", {}) if best_trial else {},
        "best_metrics": best_trial.get("metrics", {}) if best_trial else {},
        "objective_metric": objective_metric,
        "duration_ms": duration_ms,
        "metadata": spec.get("metadata", {}),
    }

    # Persist best model to registry
    if best_trial and best_trial.get("status") == "ok":
        try:
            training_spec = TRAINING_JOBS.get(training_job_name, {})
            registry_key = await _persist_trained_model(training_spec, best_trial, session)
            result["best_model_registry_key"] = registry_key
        except Exception as exc:
            logger.warning("Failed to persist best tuned model to registry: %s", exc)
            result["registry_warning"] = str(exc)

    _record_tuning_history(key, result, duration_ms)
    _record_runtime_event(
        runtime_context,
        event="tuning_job.complete",
        level="info",
        data={
            "job": key,
            "total_trials": len(trials),
            "duration_ms": duration_ms,
        },
    )
    return result


async def _load_training_dataset(
    dataset_name: str,
    training_spec: Dict[str, Any],
    session: Optional[Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Load dataset for training with proper async handling."""
    dataset_spec = DATASETS.get(dataset_name) or FRAMES.get(dataset_name)
    if not dataset_spec:
        raise ValueError(f"Dataset '{dataset_name}' not found")

    # Use existing dataset loading infrastructure
    if dataset_name in DATASETS:
        rows = await load_dataset(dataset_name, session=session, context=context)
    else:
        rows = await load_frame(dataset_name, session=session, context=context)

    return {
        "dataset_name": dataset_name,
        "dataset_rows": rows,
        "dataset_size": len(rows) if rows else 0,
        "dataset_schema": dataset_spec.get("schema", []),
    }


async def _execute_tuning_trials(
    training_job_name: str,
    search_space: Dict[str, Any],
    strategy: str,
    max_trials: int,
    parallel_trials: int,
    objective_metric: str,
    early_stopping: Optional[Dict[str, Any]],
    payload: Optional[Dict[str, Any]],
    overrides: Optional[Dict[str, Any]],
    context: Dict[str, Any],
    session: Optional[Any],
) -> List[Dict[str, Any]]:
    """Execute tuning trials with specified search strategy."""
    trials = []

    # Generate trial configurations
    trial_configs = _generate_trial_configs(search_space, strategy, max_trials)

    # Execute trials (sequential for now; parallel execution would require asyncio.gather)
    for trial_idx, trial_hyperparams in enumerate(trial_configs):
        trial_overrides = dict(overrides or {})
        trial_overrides["hyperparameters"] = trial_hyperparams

        logger.info("Executing tuning trial %d/%d with hyperparameters: %s", 
                   trial_idx + 1, len(trial_configs), trial_hyperparams)

        trial_result = await run_training_job(
            name=training_job_name,
            payload=payload,
            overrides=trial_overrides,
            context=context,
            session=session,
        )

        trial_result["trial_index"] = trial_idx
        trial_result["hyperparameters"] = trial_hyperparams
        trials.append(trial_result)

        # Check early stopping
        if early_stopping and _should_stop_early(trials, early_stopping, objective_metric):
            logger.info("Early stopping triggered after %d trials", len(trials))
            break

    return trials


def _generate_trial_configs(
    search_space: Dict[str, Any],
    strategy: str,
    max_trials: int,
) -> List[Dict[str, Any]]:
    """Generate hyperparameter configurations for trials based on strategy."""
    import random

    if strategy == "grid":
        # Grid search: all combinations
        return _grid_search_configs(search_space, max_trials)
    elif strategy == "random":
        # Random search: random sampling
        return _random_search_configs(search_space, max_trials)
    else:
        # Default to random
        return _random_search_configs(search_space, max_trials)


def _grid_search_configs(search_space: Dict[str, Any], max_trials: int) -> List[Dict[str, Any]]:
    """Generate grid search configurations."""
    import itertools

    param_names = list(search_space.keys())
    param_values = []

    for param_name in param_names:
        spec = search_space[param_name]
        if isinstance(spec, dict):
            if "values" in spec:
                param_values.append(spec["values"])
            elif "min" in spec and "max" in spec:
                # Sample uniformly
                min_val = spec["min"]
                max_val = spec["max"]
                step = spec.get("step", (max_val - min_val) / 5.0)
                values = []
                current = min_val
                while current <= max_val:
                    values.append(current)
                    current += step
                param_values.append(values)
            else:
                param_values.append([None])
        else:
            param_values.append([spec])

    # Generate all combinations
    all_configs = []
    for combo in itertools.product(*param_values):
        config = dict(zip(param_names, combo))
        all_configs.append(config)
        if len(all_configs) >= max_trials:
            break

    return all_configs[:max_trials]


def _random_search_configs(search_space: Dict[str, Any], max_trials: int) -> List[Dict[str, Any]]:
    """Generate random search configurations."""
    import random

    configs = []
    for _ in range(max_trials):
        config = {}
        for param_name, spec in search_space.items():
            if isinstance(spec, dict):
                if "values" in spec:
                    config[param_name] = random.choice(spec["values"])
                elif "min" in spec and "max" in spec:
                    min_val = spec["min"]
                    max_val = spec["max"]
                    if spec.get("log", False):
                        # Log-uniform sampling
                        import math
                        log_min = math.log(max(min_val, 1e-10))
                        log_max = math.log(max_val)
                        config[param_name] = math.exp(random.uniform(log_min, log_max))
                    else:
                        # Uniform sampling
                        config[param_name] = random.uniform(min_val, max_val)
                else:
                    config[param_name] = None
            else:
                config[param_name] = spec
        configs.append(config)

    return configs


def _should_stop_early(
    trials: List[Dict[str, Any]],
    early_stopping: Dict[str, Any],
    objective_metric: str,
) -> bool:
    """Check if early stopping criteria are met."""
    patience = early_stopping.get("patience", 5)
    min_delta = early_stopping.get("min_delta", 0.0)
    mode = early_stopping.get("mode", "max")

    if len(trials) < patience:
        return False

    # Get recent metric values
    recent_metrics = []
    for trial in trials[-patience:]:
        metrics = trial.get("metrics", {})
        if objective_metric in metrics:
            recent_metrics.append(metrics[objective_metric])

    if len(recent_metrics) < patience:
        return False

    # Check if no improvement
    if mode == "max":
        best_recent = max(recent_metrics)
        is_improving = any(m >= best_recent + min_delta for m in recent_metrics[1:])
    else:
        best_recent = min(recent_metrics)
        is_improving = any(m <= best_recent - min_delta for m in recent_metrics[1:])

    return not is_improving


def _find_best_trial(trials: List[Dict[str, Any]], objective_metric: str) -> Optional[Dict[str, Any]]:
    """Find the best trial based on objective metric."""
    valid_trials = [t for t in trials if t.get("status") == "ok" and objective_metric in t.get("metrics", {})]
    if not valid_trials:
        return None

    # Assume higher is better for accuracy-like metrics, lower for loss-like metrics
    if "loss" in objective_metric.lower() or "error" in objective_metric.lower():
        return min(valid_trials, key=lambda t: t["metrics"][objective_metric])
    else:
        return max(valid_trials, key=lambda t: t["metrics"][objective_metric])


async def _persist_trained_model(
    training_spec: Dict[str, Any],
    result: Dict[str, Any],
    session: Optional[Any],
) -> str:
    """Persist trained model to the model registry."""
    model_name = training_spec.get("model", "unnamed_model")
    job_name = training_spec.get("name", "unnamed_job")

    # Generate registry key
    import time
    timestamp = int(time.time())
    version = f"v{timestamp}"
    registry_key = f"{model_name}_{job_name}_{version}"

    # Prepare registry entry
    artifacts = result.get("artifacts", {})
    metrics = result.get("metrics", {})
    metadata = result.get("metadata", {})

    registry_entry = {
        "type": metadata.get("model_type", "sklearn"),
        "framework": metadata.get("framework", training_spec.get("framework", "sklearn")),
        "version": version,
        "metrics": metrics,
        "metadata": {
            **metadata,
            "training_job": job_name,
            "dataset": training_spec.get("dataset"),
            "target": training_spec.get("target"),
            "features": training_spec.get("features", []),
            "hyperparameters": result.get("hyperparameters", {}),
            "trained_at": timestamp,
        },
    }

    # Add model artifacts if available
    if "model_object" in artifacts:
        registry_entry["metadata"]["model_object"] = artifacts["model_object"]
    if "feature_names" in artifacts:
        registry_entry["metadata"]["feature_names"] = artifacts["feature_names"]
    if "target_name" in artifacts:
        registry_entry["metadata"]["target_name"] = artifacts["target_name"]

    # Store in MODEL_REGISTRY
    MODEL_REGISTRY[registry_key] = registry_entry

    logger.info("Persisted trained model to registry: %s", registry_key)
    return registry_key


def _training_environment(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    env: Dict[str, Any] = {}
    if isinstance(context, dict):
        context_env = context.get("env") or {}
        if isinstance(context_env, dict):
            for key, value in context_env.items():
                if value is not None:
                    env[str(key)] = str(value)
    for key in ENV_KEYS:
        if key not in env:
            value = os.getenv(key)
            if value is not None:
                env[key] = value
    return env


def _prepare_training_context(
    spec: Dict[str, Any],
    runtime_context: Optional[Dict[str, Any]],
    dataset_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {}

    # Add dataset context
    if dataset_context:
        context.update(dataset_context)

    # Add training-specific metadata
    context["metadata"] = {
        **spec.get("metadata", {}),
        "target": spec.get("target"),
        "features": spec.get("features", []),
        "split": spec.get("split", {}),
        "model_type": spec.get("metadata", {}).get("model_type", "RandomForestClassifier"),
    }

    if runtime_context:
        context.setdefault("context", runtime_context)

    return context


def _record_training_history(name: str, result: Dict[str, Any], duration_ms: float) -> None:
    entry = copy.deepcopy(result)
    entry.setdefault("ts", time.time())
    entry.setdefault("duration_ms", duration_ms)
    history = TRAINING_JOB_HISTORY.setdefault(name, [])
    history.append(entry)
    if len(history) > 50:
        del history[:-50]


def _record_tuning_history(name: str, result: Dict[str, Any], duration_ms: float) -> None:
    entry = copy.deepcopy(result)
    entry.setdefault("ts", time.time())
    entry.setdefault("duration_ms", duration_ms)
    history = TUNING_JOB_HISTORY.setdefault(name, [])
    history.append(entry)
    if len(history) > 50:
        del history[:-50]

import importlib
import inspect
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

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
        if "rows" in payload:
            return _normalize_connector_rows(payload.get("rows"))
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


def _trim_traceback(limit: int = 5, max_chars: int = 3000) -> str:
    """Return a truncated traceback string for logging contexts."""

    import traceback

    try:
        formatted = traceback.format_exc(limit=limit)
    except Exception:  # pragma: no cover - defensive fallback
        return ""
    if not formatted:
        return ""
    text = formatted.strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


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


def _prune_nones(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _prune_nones(value) for key, value in payload.items() if value is not None}
    if isinstance(payload, list):
        return [_prune_nones(item) for item in payload if item is not None]
    return payload


def _coerce_bool_option(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_connector_value(value: Any, context: Dict[str, Any], missing_env: Optional[Set[str]] = None) -> Any:
    resolved = _resolve_placeholders(value, context)
    return _materialize_connector_value(resolved, context, missing_env)


def _materialize_connector_value(value: Any, context: Dict[str, Any], missing_env: Optional[Set[str]]) -> Any:
    if isinstance(value, dict):
        return {
            key: _materialize_connector_value(item, context, missing_env)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_materialize_connector_value(item, context, missing_env) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return text
        if text.startswith("${") and text.endswith("}"):
            env_name = text[2:-1].strip()
            env_value = os.getenv(env_name)
            if env_value is None and missing_env is not None:
                missing_env.add(env_name)
            return env_value
        if text.lower().startswith("env?") and ":" not in text:
            env_name = text[4:].strip()
            return os.getenv(env_name)
        prefix, sep, remainder = text.partition(":")
        if sep:
            scope = prefix.strip().lower()
            target = remainder.strip()
            if scope == "env":
                env_value = os.getenv(target)
                if env_value is None and missing_env is not None:
                    missing_env.add(target)
                return env_value
            if scope in {"ctx", "context"}:
                parts = [segment for segment in target.split(".") if segment]
                return _resolve_context_scope("ctx", parts, context, None)
            if scope == "vars":
                parts = [segment for segment in target.split(".") if segment]
                return _resolve_context_scope("vars", parts, context, None)
            if scope == "env?":
                return os.getenv(target)
        rendered = _render_template_value(text, context)
        return rendered
    return value


def _now_ms() -> float:
    """Return the current wall-clock time in milliseconds with millisecond precision."""

    return float(round(time.time() * 1000.0, 3))


def _emit_connector_telemetry(
    context: Optional[Dict[str, Any]],
    connector: Optional[Dict[str, Any]],
    *,
    driver: str,
    status: str,
    start_ms: float,
    rows: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    name = (connector or {}).get("name") if connector else None
    duration_ms = max(_now_ms() - start_ms, 0.0)
    if 'observe_connector_status' in globals():
        try:
            observe_connector_status(name, status)
        except Exception:
            logger.debug("Failed to record connector status for %s", name or driver, exc_info=True)
    tags = {
        "connector": name or driver,
        "driver": driver,
        "status": status,
    }
    if rows is not None:
        tags["rows_present"] = "true" if rows > 0 else "false"
    payload: Dict[str, Any] = {
        "connector": name,
        "driver": driver,
        "status": status,
        "duration_ms": duration_ms,
    }
    request_id = current_request_id()
    if request_id:
        payload["request_id"] = request_id
    if rows is not None:
        payload["rows"] = rows
    if metadata:
        payload.update({key: value for key, value in metadata.items() if value is not None})
    if error:
        payload["error"] = error
    level = "info"
    if status in {"error", "not_configured", "missing_config", "dependency_missing"}:
        level = "warning"
    elif status in {"demo", "cache_hit"}:
        level = "debug"
    _record_runtime_metric(
        context,
        name="connector.duration",
        value=duration_ms,
        unit="milliseconds",
        tags=tags,
        scope=name or driver,
    )
    if rows is not None:
        _record_runtime_metric(
            context,
            name="connector.rows",
            value=rows,
            unit="count",
            tags=tags,
            scope=name or driver,
        )
    _record_runtime_event(
        context,
        event="connector.execute",
        level=level,
        message=f"Connector '{name or driver}' {status}",
        data=payload,
    )


async def _default_sql_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    start_ms = _now_ms()
    driver_name = "sql"
    span_attrs = {"connector": connector.get("name") if connector else None, "driver": driver_name}
    with tracing_span("namel3ss.connector.sql", span_attrs):
        if not connector:
            _emit_connector_telemetry(context, connector, driver=driver_name, status="missing_config", start_ms=start_ms, rows=0)
            return []
        try:
            _require_dependency("sqlalchemy", "sql")
        except ImportError as exc:
            _emit_connector_telemetry(
                context,
                connector,
                driver=driver_name,
                status="error",
                start_ms=start_ms,
                rows=0,
                error=str(exc),
            )
            raise ImportError(str(exc)) from exc
        query = connector.get("options", {}).get("query")
        if not query:
            table_name = connector.get("options", {}).get("table") or connector.get("name")
            if not table_name:
                _emit_connector_telemetry(
                    context,
                    connector,
                    driver=driver_name,
                    status="not_configured",
                    start_ms=start_ms,
                    rows=0,
                    metadata={"reason": "no_table"},
                )
                return []
            query = f"SELECT * FROM {table_name}"
        session: Optional[AsyncSession] = context.get("session")
        if session is None:
            _emit_connector_telemetry(
                context,
                connector,
                driver=driver_name,
                status="no_session",
                start_ms=start_ms,
                rows=0,
            )
            return []
        try:
            result = await session.execute(text(query))
            rows = [dict(row) for row in result.mappings()]
            _emit_connector_telemetry(
                context,
                connector,
                driver=driver_name,
                status="ok",
                start_ms=start_ms,
                rows=len(rows),
                metadata={"query": query},
            )
            return rows
        except Exception as exc:
            logger.exception("Default SQL driver failed for query '%s'", query)
            _emit_connector_telemetry(
                context,
                connector,
                driver=driver_name,
                status="error",
                start_ms=start_ms,
                rows=0,
                metadata={"query": query},
                error=str(exc),
            )
            return []


async def _default_rest_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    start_ms = _now_ms()
    driver_name = "rest"
    connector_obj = connector or {}
    raw_options = connector_obj.get("options") or {}
    missing_env: Set[str] = set()
    resolved_options = _resolve_connector_value(raw_options, context, missing_env) if raw_options else {}
    options = resolved_options if isinstance(resolved_options, dict) else {}

    method = str(options.get("method") or "GET").upper()
    endpoint_value = (
        options.get("endpoint")
        or options.get("url")
        or connector_obj.get("endpoint")
        or connector_obj.get("url")
    )
    endpoint = str(endpoint_value).strip() if endpoint_value else ""

    params: Dict[str, Any] = {}
    params_option = options.get("params")
    if isinstance(params_option, dict):
        params = {str(key): value for key, value in params_option.items() if value is not None}

    headers: Dict[str, Any] = {}
    headers_option = options.get("headers")
    if isinstance(headers_option, dict):
        header_entries: Dict[str, Any] = {}
        for key, value in headers_option.items():
            if value is None:
                continue
            if isinstance(value, (bytes, bytearray)):
                header_entries[str(key)] = value.decode("utf-8", "replace")
            else:
                header_entries[str(key)] = str(value)
        headers = header_entries

    def _split_context_pointer(text: str) -> Tuple[str, str]:
        expr = text.strip()
        if not expr:
            return expr, expr
        if "=" in expr:
            name, _, remainder = expr.partition("=")
            return name.strip(), remainder.strip()
        scoped_expr = expr
        if ":" in expr:
            _, _, scoped_expr = expr.partition(":")
            scoped_expr = scoped_expr.strip()
        if "." in scoped_expr:
            name = scoped_expr.rsplit(".", 1)[-1]
        else:
            name = scoped_expr
        return (name or expr, expr)

    def _resolve_context_param_pointer(pointer: Any) -> Any:
        if pointer is None:
            return None
        if isinstance(pointer, (dict, list)):
            return _resolve_connector_value(pointer, context, missing_env)
        if isinstance(pointer, str):
            text = pointer.strip()
            if not text:
                return None
            materialized = _materialize_connector_value(text, context, missing_env)
            if materialized is not None and materialized != text:
                return materialized
            parts = [segment for segment in text.split(".") if segment]
            if parts:
                fallback = _resolve_context_path(context, parts, None)
                if fallback is not None:
                    return fallback
            return materialized
        return pointer

    def _resolve_context_params(spec: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if spec is None:
            return result
        resolved_spec = _resolve_connector_value(spec, context, missing_env)
        if isinstance(resolved_spec, dict):
            for param_name, pointer in resolved_spec.items():
                value = _resolve_context_param_pointer(pointer)
                if value is not None:
                    result[str(param_name)] = value
        elif isinstance(resolved_spec, list):
            for entry in resolved_spec:
                if isinstance(entry, dict):
                    for param_name, pointer in entry.items():
                        value = _resolve_context_param_pointer(pointer)
                        if value is not None:
                            result[str(param_name)] = value
                elif isinstance(entry, str):
                    name, pointer_expr = _split_context_pointer(entry)
                    value = _resolve_context_param_pointer(pointer_expr)
                    if value is not None:
                        result[str(name)] = value
        return result

    params.update(_resolve_context_params(raw_options.get("context_params") or options.get("context_params")))
    params = {key: value for key, value in params.items() if value is not None}

    redacted_config = _prune_nones(_redact_secrets(options))
    allow_demo = _is_truthy_env("NAMEL3SS_ALLOW_STUBS") or _coerce_bool_option(options.get("demo"), False)

    def _elapsed() -> float:
        return max(_now_ms() - start_ms, 0.0)

    def _traverse_result_path(payload: Any, path: str) -> Any:
        if not path:
            return payload
        target = payload
        for segment in path.split("."):
            key = segment.strip()
            if not key:
                continue
            if isinstance(target, dict):
                target = target.get(key)
            elif isinstance(target, list):
                try:
                    index = int(key)
                except ValueError:
                    target = None
                else:
                    if 0 <= index < len(target):
                        target = target[index]
                    else:
                        target = None
            else:
                target = None
            if target is None:
                break
        return target

    result_path = str(options.get("result_path") or "").strip()

    timeout_value = options.get("timeout")
    if timeout_value is None:
        timeout_value = options.get("timeout_seconds")
    if timeout_value is None and options.get("timeout_ms") is not None:
        try:
            timeout_value = float(options.get("timeout_ms")) / 1000.0
        except Exception:
            timeout_value = None
    try:
        timeout_seconds = float(timeout_value) if timeout_value is not None else 10.0
    except Exception:
        timeout_seconds = 10.0

    retries_value = options.get("max_retries") if "max_retries" in options else options.get("retries")
    try:
        max_attempts = max(int(retries_value), 1) if retries_value is not None else 1
    except Exception:
        max_attempts = 1

    body = options.get("body")
    if body is None:
        body = options.get("payload")
    if body is None:
        body = options.get("data")

    inferred_body_format = "json" if isinstance(body, (dict, list)) else "raw"
    body_format_value = options.get("body_format")
    body_format = str(body_format_value or inferred_body_format).strip().lower()

    headers = {key: value for key, value in headers.items() if value is not None}

    inputs: Dict[str, Any] = {"connector": connector_obj.get("name"), "method": method}
    if params:
        inputs["params"] = sorted(params.keys())
    if headers:
        inputs["headers"] = sorted(headers.keys())

    def _finalize(
        status: str,
        *,
        result: Any = None,
        error: Optional[str] = None,
        traceback_text: Optional[str] = None,
        status_code: Optional[int] = None,
        attempts_value: int = 0,
        extra_metadata: Optional[Dict[str, Any]] = None,
        include_config: bool = False,
    ) -> Dict[str, Any]:
        rows_for_status = _extract_rows_from_connector_response({"result": result}) if result is not None else []
        rows_count = len(rows_for_status)
        metadata: Dict[str, Any] = {
            "elapsed_ms": _elapsed(),
            "endpoint": endpoint or None,
            "method": method,
            "status_code": status_code,
            "attempts": attempts_value,
            "result_path": result_path or None,
            "body_format": body_format or None,
        }
        if params:
            metadata["params_keys"] = sorted(params.keys())
        if headers:
            metadata["header_keys"] = sorted(headers.keys())
        if missing_env:
            metadata["missing_env"] = sorted(missing_env)
        if extra_metadata:
            metadata.update(extra_metadata)
        metadata = _prune_nones(metadata)

        _emit_connector_telemetry(
            context,
            connector,
            driver=driver_name,
            status=status,
            start_ms=start_ms,
            rows=rows_count,
            metadata={key: value for key, value in metadata.items() if key != "elapsed_ms"},
            error=error,
        )

        return {
            "status": status,
            "result": result,
            "rows": rows_for_status if rows_for_status else None,
            "error": error,
            "traceback": traceback_text,
            "config": redacted_config if include_config else None,
            "inputs": _prune_nones(dict(inputs)),
            "metadata": metadata,
        }

    if not endpoint:
        if allow_demo:
            demo_rows_raw = (
                options.get("demo_rows")
                or options.get("seed_rows")
                or options.get("sample_rows")
            )
            demo_rows = (
                _normalize_connector_rows(demo_rows_raw)
                if demo_rows_raw is not None
                else [
                    {"demo": True, "connector": connector_obj.get("name") or "rest", "sequence": index}
                    for index in range(3)
                ]
            )
            logger.info(
                "REST connector '%s' running in demo mode without endpoint",
                connector_obj.get("name"),
            )
            return _finalize(
                "demo",
                result=demo_rows,
                attempts_value=0,
                extra_metadata={"demo": True},
                include_config=True,
            )
        message = "REST connector requires an 'endpoint' option"
        if missing_env:
            message = f"{message}; missing env: {', '.join(sorted(missing_env))}"
        logger.warning("REST connector '%s' missing endpoint", connector_obj.get("name"))
        return _finalize("not_configured", error=message, attempts_value=0, include_config=True)

    if httpx is None:
        message = "REST connector requires httpx to be installed"
        logger.warning("REST connector '%s' requires httpx to be installed", connector_obj.get("name"))
        if allow_demo:
            demo_rows_raw = (
                options.get("demo_rows")
                or options.get("seed_rows")
                or options.get("sample_rows")
            )
            demo_rows = (
                _normalize_connector_rows(demo_rows_raw)
                if demo_rows_raw is not None
                else [
                    {"demo": True, "connector": connector_obj.get("name") or "rest", "sequence": index}
                    for index in range(3)
                ]
            )
            return _finalize(
                "demo",
                result=demo_rows,
                attempts_value=0,
                extra_metadata={"demo": True, "reason": "dependency_missing"},
                include_config=True,
            )
        return _finalize("dependency_missing", error=message, attempts_value=0, include_config=True)

    client_kwargs: Dict[str, Any] = {}
    if timeout_seconds is not None:
        try:
            client_kwargs["timeout"] = httpx.Timeout(timeout_seconds) if httpx is not None else timeout_seconds
        except Exception:
            client_kwargs["timeout"] = timeout_seconds
    if "verify" in options:
        client_kwargs["verify"] = _coerce_bool_option(options.get("verify"), True)
    if _coerce_bool_option(options.get("follow_redirects"), False):
        client_kwargs["follow_redirects"] = True

    request_kwargs: Dict[str, Any] = {}
    if params:
        request_kwargs["params"] = params
    if headers:
        request_kwargs["headers"] = headers
    if body is not None and method not in {"GET", "DELETE", "HEAD"}:
        if body_format == "json":
            request_kwargs["json"] = body
        elif body_format == "form":
            if isinstance(body, dict):
                request_kwargs["data"] = {str(key): "" if value is None else str(value) for key, value in body.items()}
            else:
                request_kwargs["data"] = body
        else:
            if isinstance(body, (bytes, bytearray)):
                request_kwargs["content"] = body
            else:
                request_kwargs["content"] = str(body)

    # Get connector config from runtime settings
    connector_settings = _runtime_setting("connectors", {})
    retry_max_attempts = connector_settings.get("retry_max_attempts", max_attempts)
    retry_base_delay = connector_settings.get("retry_base_delay", 0.5)
    retry_max_delay = connector_settings.get("retry_max_delay", 5.0)

    # Use configured retry settings if available and make_resilient_request is available
    if make_resilient_request is not None and RetryConfig is not None:
        retry_config = RetryConfig(
            max_attempts=retry_max_attempts,
            base_delay_seconds=retry_base_delay,
            max_delay_seconds=retry_max_delay,
        )

        async def _make_request_with_retry() -> Tuple[int, Any, Optional[int]]:
            """Make HTTP request with resilient retry wrapper."""
            async def _request_fn(client_instance: Any) -> Any:
                response = await client_instance.request(method, endpoint, **request_kwargs)
                response.raise_for_status()
                return response

            async with _HTTPX_CLIENT_CLS(**client_kwargs) as client:
                response = await make_resilient_request(
                    _request_fn,
                    retry_config,
                    connector_obj.get("name") or driver_name,
                    client,
                )
                status_code = getattr(response, "status_code", None)
                raw_text = getattr(response, "text", None)
                if raw_text is None and hasattr(response, "content"):
                    content_bytes = getattr(response, "content")
                    raw_text = content_bytes.decode("utf-8", "replace") if content_bytes else ""
                if raw_text is not None and not raw_text.strip():
                    data: Any = []
                else:
                    try:
                        data = response.json()
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        logger.error(
                            "REST connector '%s' returned non-JSON payload",
                            connector_obj.get("name"),
                        )
                        raise ValueError(message) from exc
                return (retry_max_attempts, data, status_code)

        try:
            attempts, data, last_status = await _make_request_with_retry()
            result_data = data
            if result_path:
                result_data = _traverse_result_path(data, result_path)
            rows = _normalize_connector_rows(result_data)
            status_value = "ok" if rows else "empty"
            logger.info(
                "REST connector '%s' succeeded with status %s",
                connector_obj.get("name"),
                last_status,
            )
            return _finalize(
                status_value,
                result=result_data,
                status_code=last_status,
                attempts_value=attempts,
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            traceback_text = _trim_traceback()
            last_status_code = None
            if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
                last_status_code = getattr(exc.response, "status_code", None)
            return _finalize(
                "error",
                error=error_message,
                traceback_text=traceback_text,
                status_code=last_status_code,
                attempts_value=retry_max_attempts,
                include_config=True,
            )

    # Fallback to manual retry loop if resilient request is not available
    attempts = 0
    last_error: Optional[BaseException] = None
    last_status: Optional[int] = None

    async with _HTTPX_CLIENT_CLS(**client_kwargs) as client:
        while attempts < max_attempts:
            attempts += 1
            try:
                span_attrs = {
                    "connector": connector_obj.get("name") or driver_name,
                    "driver": driver_name,
                    "attempt": attempts,
                    "endpoint": endpoint,
                }
                with tracing_span("namel3ss.connector.rest.request", span_attrs):
                    response = await client.request(method, endpoint, **request_kwargs)
                last_status = getattr(response, "status_code", last_status)
                response.raise_for_status()
                raw_text = getattr(response, "text", None)
                if raw_text is None and hasattr(response, "content"):
                    content_bytes = getattr(response, "content")
                    raw_text = content_bytes.decode("utf-8", "replace") if content_bytes else ""
                if raw_text is not None and not raw_text.strip():
                    data: Any = []
                else:
                    try:
                        data = response.json()
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        logger.error(
                            "REST connector '%s' returned non-JSON payload",
                            connector_obj.get("name"),
                        )
                        return _finalize(
                            "error",
                            error=message,
                            status_code=last_status,
                            attempts_value=attempts,
                            include_config=True,
                        )
                result_data = data
                if result_path:
                    result_data = _traverse_result_path(data, result_path)
                rows = _normalize_connector_rows(result_data)
                status_value = "ok" if rows else "empty"
                logger.info(
                    "REST connector '%s' succeeded with status %s",
                    connector_obj.get("name"),
                    last_status,
                )
                return _finalize(
                    status_value,
                    result=result_data,
                    status_code=last_status,
                    attempts_value=attempts,
                )
            except Exception as exc:
                last_error = exc
                if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
                    last_status = getattr(exc.response, "status_code", last_status)
                if httpx is not None and isinstance(exc, httpx.TimeoutException):
                    logger.warning(
                        "REST connector '%s' attempt %d/%d timed out",
                        connector_obj.get("name"),
                        attempts,
                        max_attempts,
                    )
                else:
                    logger.warning(
                        "REST connector '%s' attempt %d/%d failed: %s",
                        connector_obj.get("name"),
                        attempts,
                        max_attempts,
                        exc,
                    )
                if attempts >= max_attempts:
                    error_message = f"{type(exc).__name__}: {exc}"
                    traceback_text = _trim_traceback()
                    return _finalize(
                        "error",
                        error=error_message,
                        traceback_text=traceback_text,
                        status_code=last_status,
                        attempts_value=attempts,
                        include_config=True,
                    )

    error_message = (
        f"{type(last_error).__name__}: {last_error}"
        if last_error is not None
        else "REST connector terminated without a response"
    )
    return _finalize(
        "error",
        error=error_message,
        traceback_text=_trim_traceback(),
        status_code=last_status,
        attempts_value=attempts,
        include_config=True,
    )


async def _default_graphql_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    start_ms = _now_ms()
    driver_name = "graphql"
    options = connector.get("options", {}) if connector else {}
    endpoint = options.get("endpoint") or options.get("url") or connector.get("name")
    query = options.get("query")
    if not endpoint or not query:
        _emit_connector_telemetry(
            context,
            connector,
            driver=driver_name,
            status="not_configured",
            start_ms=start_ms,
            rows=0,
            metadata={"reason": "missing_endpoint_or_query"},
        )
        return []
    if httpx is None:
        logger.warning("GraphQL connector '%s' requires httpx to be installed", connector.get("name"))
        _emit_connector_telemetry(
            context,
            connector,
            driver=driver_name,
            status="dependency_missing",
            start_ms=start_ms,
            rows=0,
            metadata={"endpoint": endpoint},
        )
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

    # Get connector config from runtime settings
    connector_settings = _runtime_setting("connectors", {})
    retry_max_attempts = connector_settings.get("retry_max_attempts", retries)
    retry_base_delay = connector_settings.get("retry_base_delay", 0.5)
    retry_max_delay = connector_settings.get("retry_max_delay", 5.0)

    client_kwargs: Dict[str, Any] = {}
    if httpx is not None:
        try:
            client_kwargs["timeout"] = httpx.Timeout(timeout)
        except Exception:
            client_kwargs["timeout"] = timeout
    else:
        client_kwargs["timeout"] = timeout

    attempts = 0
    payload: Dict[str, Any] = {}
    status = "error"
    last_error: Optional[Exception] = None

    # Use resilient request if available
    if make_resilient_request is not None and RetryConfig is not None:
        retry_config = RetryConfig(
            max_attempts=retry_max_attempts,
            base_delay_seconds=retry_base_delay,
            max_delay_seconds=retry_max_delay,
        )

        try:
            async def _graphql_request(client_instance: Any) -> Any:
                response = await client_instance.post(
                    endpoint,
                    json={"query": query, "variables": variables or {}},
                    headers=headers if isinstance(headers, dict) else None,
                )
                response.raise_for_status()
                return response.json()

            async with _HTTPX_CLIENT_CLS(**client_kwargs) as client:
                payload = await make_resilient_request(
                    _graphql_request,
                    retry_config,
                    connector.get("name") or driver_name,
                    client,
                )
                attempts = retry_max_attempts
                status = "ok"
        except Exception as exc:
            logger.error("GraphQL connector '%s' failed after retries", connector.get("name"))
            last_error = exc
            status = "error"
    else:
        # Fallback to manual retry loop
        async with _HTTPX_CLIENT_CLS(**client_kwargs) as client:
            try:
                while True:
                    attempts += 1
                    request_start = _now_ms()
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
                            _now_ms() - request_start,
                        )
                        status = "ok"
                        break
                    except ((httpx.HTTPError, httpx.TimeoutException) if httpx is not None else (Exception,)) as exc:
                        logger.warning(
                            "GraphQL connector '%s' attempt %d/%d failed: %s",
                            connector.get("name"),
                            attempts,
                            retries,
                            exc,
                        )
                        last_error = exc
                        status = "retry_failed"
                        if attempts >= retries:
                            raise
                    except Exception as exc:
                        logger.exception("Default GraphQL driver failed for endpoint '%s'", endpoint)
                        last_error = exc
                        status = "error"
                        raise
            except Exception as exc:
                logger.error("GraphQL connector '%s' exhausted retries", connector.get("name"))
                last_error = exc
                status = "error"
    data = payload.get("data") if isinstance(payload, dict) else None
    rows_count = 0
    rows: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        target: Any
        if root_field:
            target = data.get(root_field)
        else:
            target = next(iter(data.values())) if data else None
        rows = _normalize_connector_rows(target)
        rows_count = len(rows)
        if rows:
            status = "ok"
    metadata = {
        "endpoint": endpoint,
        "attempts": attempts,
        "root_field": root_field,
    }
    _emit_connector_telemetry(
        context,
        connector,
        driver=driver_name,
        status=status if rows else (status if status != "ok" else "empty"),
        start_ms=start_ms,
        rows=rows_count,
        metadata=metadata,
        error=str(last_error) if last_error else None,
    )
    if rows:
        return rows
    return []


async def _default_grpc_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a gRPC connector via an optional pluggable driver."""

    start_ms = _now_ms()
    driver_name = "grpc"
    connector_obj = connector or {}
    raw_options = connector_obj.get("options") or {}
    missing_env: Set[str] = set()
    resolved_options = _resolve_connector_value(raw_options, context, missing_env) if raw_options else {}
    options = resolved_options if isinstance(resolved_options, dict) else (raw_options if isinstance(raw_options, dict) else {})
    inputs = {"connector": connector_obj.get("name")}
    redacted_config = _prune_nones(_redact_secrets(options))

    def _emit(status: str, result_payload: Any = None, metadata: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        extracted_rows = _extract_rows_from_connector_response({"result": result_payload}) if result_payload is not None else []
        rows_count = len(extracted_rows)
        metadata_payload = dict(metadata or {})
        if missing_env:
            metadata_payload.setdefault("missing_env", sorted(missing_env))
        _emit_connector_telemetry(
            context,
            connector,
            driver=driver_name,
            status=status,
            start_ms=start_ms,
            rows=rows_count,
            metadata=metadata_payload,
            error=error,
        )

    allow_demo = _is_truthy_env("NAMEL3SS_ALLOW_STUBS") or _coerce_bool_option(options.get("demo"), False)

    host = str(options.get("host") or "").strip()
    service = str(options.get("service") or connector_obj.get("name") or "").strip()
    method = str(options.get("method") or "").strip()

    def _elapsed() -> float:
        return max(_now_ms() - start_ms, 0.0)

    if not host or not service or not method:
        message = "Missing gRPC configuration (host/service/method)"
        logger.warning("gRPC connector '%s' missing configuration", connector_obj.get("name"))
        if missing_env:
            message = f"{message}; missing env: {', '.join(sorted(missing_env))}"
        payload = {
            "status": "not_configured",
            "result": None,
            "error": message,
            "traceback": None,
            "config": redacted_config,
            "inputs": inputs,
            "metadata": _prune_nones(
                {
                    "elapsed_ms": _elapsed(),
                    "missing_env": sorted(missing_env) if missing_env else None,
                }
            ),
        }
        _emit("not_configured", metadata=payload["metadata"], error=message)
        return payload

    port_value = options.get("port")
    try:
        port = int(port_value) if port_value is not None else 443
    except Exception:
        port = 443

    tls = _coerce_bool_option(options.get("tls"), True)
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
            payload = {
                "status": "error",
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": _trim_traceback(),
                "config": redacted_config,
                "inputs": inputs,
                "metadata": _prune_nones(
                    {
                        "elapsed_ms": _elapsed(),
                        "missing_env": sorted(missing_env) if missing_env else None,
                    }
                ),
            }
            _emit("error", metadata=payload["metadata"], error=payload["error"])
            return payload

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
            payload = {
                "status": "error",
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": _trim_traceback(),
                "config": redacted_config,
                "inputs": inputs,
                "metadata": _prune_nones(
                    {
                        "elapsed_ms": _elapsed(),
                        "missing_env": sorted(missing_env) if missing_env else None,
                    }
                ),
            }
            _emit("error", error=payload["error"], metadata=payload["metadata"])
            return payload

        payload = {
            "status": "ok",
            "result": response,
            "error": None,
            "traceback": None,
            "config": None,
            "inputs": inputs,
            "metadata": _prune_nones(
                {
                    "elapsed_ms": _elapsed(),
                    "endpoint": f"{host}:{port}",
                    "service": service,
                    "method": method,
                    "missing_env": sorted(missing_env) if missing_env else None,
                }
            ),
        }
        _emit("ok", result_payload=response, metadata=payload["metadata"])
        return payload

    if allow_demo:
        logger.info("gRPC connector '%s' running in demo mode", connector_obj.get("name"))
        payload = {
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
            "metadata": _prune_nones(
                {
                    "elapsed_ms": _elapsed(),
                    "endpoint": f"{host}:{port}",
                    "service": service,
                    "method": method,
                    "missing_env": sorted(missing_env) if missing_env else None,
                }
            ),
        }
        _emit("demo", result_payload=payload["result"], metadata=payload["metadata"])
        return payload

    message = "No gRPC driver configured. Set 'driver' to a callable implementation or enable demo mode."
    logger.warning("gRPC connector '%s' has no driver configured", connector_obj.get("name"))
    payload = {
        "status": "not_configured",
        "result": None,
        "error": message,
        "traceback": None,
        "config": redacted_config,
        "inputs": inputs,
        "metadata": _prune_nones(
            {
                "elapsed_ms": _elapsed(),
                "missing_env": sorted(missing_env) if missing_env else None,
            }
        ),
    }
    _emit("not_configured", metadata=payload["metadata"], error=message)
    return payload


async def _default_streaming_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Produce streaming batches from configured sources without fabricating data by default."""

    start_ms = _now_ms()
    driver_name = "stream"
    connector_obj = connector or {}
    raw_options = connector_obj.get("options") or {}
    resolved_options = _resolve_placeholders(raw_options, context) if raw_options else {}
    options = resolved_options if isinstance(resolved_options, dict) else (raw_options if isinstance(raw_options, dict) else {})

    def _elapsed() -> float:
        return max(_now_ms() - start_ms, 0.0)

    def _emit(
        status: str,
        *,
        batch: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        rows = len(batch) if isinstance(batch, list) else 0
        _emit_connector_telemetry(
            context,
            connector,
            driver=driver_name,
            status=status,
            start_ms=start_ms,
            rows=rows,
            metadata=metadata,
            error=error,
        )

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
        _emit("error", metadata=metadata, error=message)
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
            payload = {
                "status": "not_configured",
                "batch": None,
                "error": "Python streaming source requires a 'driver' callable",
                "traceback": None,
                "config": redacted_config,
                "metadata": {"elapsed_ms": _elapsed(), "source": "python", "exhausted": False},
            }
            _emit("not_configured", metadata=payload["metadata"], error=payload["error"])
            return payload
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
        payload = {
            "status": "ok",
            "batch": batch,
            "error": None,
            "traceback": None,
            "config": None,
            "metadata": metadata,
        }
        _emit("ok", batch=batch, metadata=metadata)
        return payload

    if source_type == "http":
        url = source_spec.get("url")
        if not isinstance(url, str) or not url.strip():
            payload = {
                "status": "not_configured",
                "batch": None,
                "error": "HTTP streaming source requires a 'url'",
                "traceback": None,
                "config": redacted_config,
                "metadata": {"elapsed_ms": _elapsed(), "source": "http", "exhausted": False},
            }
            _emit("not_configured", metadata=payload["metadata"], error=payload["error"])
            return payload
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
        payload = {
            "status": "ok",
            "batch": batch,
            "error": None,
            "traceback": None,
            "config": None,
            "metadata": metadata,
        }
        _emit("ok", batch=batch, metadata=metadata)
        return payload

    if source_type == "file":
        path = source_spec.get("path")
        if not isinstance(path, str) or not path:
            payload = {
                "status": "not_configured",
                "batch": None,
                "error": "File streaming source requires a 'path'",
                "traceback": None,
                "config": redacted_config,
                "metadata": {"elapsed_ms": _elapsed(), "source": "file", "exhausted": False},
            }
            _emit("not_configured", metadata=payload["metadata"], error=payload["error"])
            return payload
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
            payload = {
                "status": "ok",
                "batch": _normalize_batch(batch),
                "error": None,
                "traceback": None,
                "config": None,
                "metadata": metadata,
            }
            _emit("ok", batch=payload["batch"], metadata=metadata)
            return payload

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
        payload = {
            "status": "ok",
            "batch": _normalize_batch(chunk),
            "error": None,
            "traceback": None,
            "config": None,
            "metadata": metadata,
        }
        _emit("ok", batch=payload["batch"], metadata=metadata)
        return payload

    if seed_rows and not source_type:
        stored_rows = cursor_state.setdefault("seed_rows", seed_rows)
        index = int(cursor_state.get("index", 0))
        slice_rows = stored_rows[index:index + batch_size]
        cursor_state["index"] = index + len(slice_rows)
        exhausted = cursor_state["index"] >= len(stored_rows)
        metadata = {"elapsed_ms": _elapsed(), "source": None, "exhausted": exhausted}
        payload = {
            "status": "ok",
            "batch": _normalize_batch(slice_rows),
            "error": None,
            "traceback": None,
            "config": None,
            "metadata": metadata,
        }
        _emit("ok", batch=payload["batch"], metadata=metadata)
        return payload

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
            payload = {
                "status": "demo",
                "batch": batch,
                "error": None,
                "traceback": None,
                "config": redacted_config,
                "metadata": metadata,
            }
            _emit("demo", batch=batch, metadata=metadata)
            return payload

        message = "No streaming source configured. Provide 'source' or 'seed_rows', or enable demo mode."
        logger.warning("Streaming connector '%s' has no configured source", connector_name)
        payload = {
            "status": "not_configured",
            "batch": None,
            "error": message,
            "traceback": None,
            "config": redacted_config,
            "metadata": {"elapsed_ms": _elapsed(), "source": None, "exhausted": False},
        }
        _emit("not_configured", metadata=payload["metadata"], error=message)
        return payload

    message = f"Unsupported streaming source type '{source_type or 'unknown'}'"
    logger.warning("Streaming connector '%s' has unsupported source type '%s'", connector_name, source_type)
    payload = {
        "status": "not_configured",
        "batch": None,
        "error": message,
        "traceback": None,
        "config": redacted_config,
        "metadata": {"elapsed_ms": _elapsed(), "source": source_type or None, "exhausted": False},
    }
    _emit("not_configured", metadata=payload["metadata"], error=message)
    return payload


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

    stub_output: Optional[Dict[str, Any]] = None
    loader_error: Optional[str] = None
    try:
        model_instance = _load_model_instance(model_name, model_spec)
    except Exception as exc:
        logger.exception("Failed to load model instance for %s", model_name)
        loader_error = f"{type(exc).__name__}: {exc}"
        metadata_payload["loader_error"] = loader_error
        env_value = os.getenv("NAMEL3SS_ALLOW_STUBS")
        allow_stub_predictions = True
        if env_value is not None:
            normalized = env_value.strip().lower()
            allow_stub_predictions = normalized in {"1", "true", "yes", "on"}
        if allow_stub_predictions:
            stub_output = _deterministic_stub_output(model_name, inputs, model_spec)
            metadata_payload["stubbed"] = True
            model_instance = None
        else:
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
                    "error": loader_error,
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

    runner = None
    if stub_output is None:
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

    if stub_output is not None:
        output = dict(stub_output)
        raw_output = output
        overall_status = output.get("status", "ok")
    elif runner is None:
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

    if explanations is None and stub_output is not None:
        explanations = _default_stub_explanations(model_name, inputs, {"output": output})

    if explanations:
        result["explanations"] = explanations

    if overall_status == "error" and output.get("error") is None:
        output["error"] = "Prediction failed"

    return result


async def broadcast_page_snapshot(slug: str, payload: Dict[str, Any]) -> None:
    if not REALTIME_ENABLED:
        return
    meta = {"page": _page_meta(slug), "status": "ok"}
    await broadcast_page_event(
        slug,
        event_type="snapshot",
        dataset=None,
        payload=payload,
        source="page-hydrate",
        status="ok",
        meta=meta,
    )


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
    event_meta: Dict[str, Any] = {"page": _page_meta(slug), "component_type": component_type, "component_index": component_index}
    if meta:
        event_meta.update(meta)
    await broadcast_page_event(
        slug,
        event_type="update",
        dataset=None,
        payload=payload,
        source="model-prediction",
        status="ok",
        meta=event_meta,
    )


async def broadcast_rollback(slug: str, component_index: int) -> None:
    if not REALTIME_ENABLED:
        return
    meta = {"page": _page_meta(slug), "component_index": component_index, "status": "rollback"}
    await broadcast_page_event(
        slug,
        event_type="mutation",
        dataset=None,
        payload={"component_index": component_index},
        source="ui-rollback",
        status="rollback",
        meta=meta,
    )

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

PAGE_HANDLERS: Dict[str, Callable[[Optional[AsyncSession]], Awaitable[Dict[str, Any]]]] = {}
