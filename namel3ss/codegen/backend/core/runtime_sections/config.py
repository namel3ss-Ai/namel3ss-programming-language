from __future__ import annotations

from textwrap import dedent

CONFIG_SECTION = dedent(
    '''


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

    '''
).strip()

__all__ = ["CONFIG_SECTION"]
