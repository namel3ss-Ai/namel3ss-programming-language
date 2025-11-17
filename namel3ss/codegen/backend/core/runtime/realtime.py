"""Realtime broadcast manager blocks for the runtime module."""

from __future__ import annotations

import textwrap


def render_broadcast_block(enable_realtime: bool) -> str:
    """Render the WebSocket broadcast helpers used by reactive pages."""
    if enable_realtime:
        broadcast = '''
_RUNTIME_NODE_ID = uuid.uuid4().hex


def _build_event_meta(source: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = {"source": source, "ts": time.time(), "node": _RUNTIME_NODE_ID}
    if extra:
        meta.update({key: value for key, value in extra.items() if value is not None})
    return meta


def _make_page_event(
    slug: Optional[str],
    *,
    event_type: str,
    dataset: Optional[str],
    payload: Any,
    source: str,
    status: Optional[str] = None,
    operation_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    event: Dict[str, Any] = {
        "type": event_type,
        "slug": slug,
        "dataset": dataset,
        "payload": payload,
        "meta": _build_event_meta(source, meta),
        "event_id": uuid.uuid4().hex,
    }
    if status:
        event["meta"]["status"] = status
    if operation_id:
        event["meta"]["operation_id"] = operation_id
    return event


class PageBroadcastManager:
    """Manage WebSocket connections for reactive pages with optional pub/sub fan-out."""

    def __init__(self) -> None:
        self._connections: Dict[str, Dict[WebSocket, Dict[str, Any]]] = {}
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, slug: str, websocket: WebSocket, *, context: Optional[Dict[str, Any]] = None) -> str:
        await websocket.accept()
        connection_context = dict(context or {})
        connection_id = uuid.uuid4().hex
        connection_info = {
            "id": connection_id,
            "context": connection_context,
            "tenant": connection_context.get("tenant"),
            "connected_at": time.time(),
        }
        subscribe = False
        async with self._lock:
            bucket = self._connections.setdefault(slug, {})
            bucket[websocket] = connection_info
            if use_redis_pubsub() and slug not in self._subscriptions:
                subscribe = True
        if subscribe:
            await self._ensure_subscription(slug)
        return connection_id

    async def disconnect(self, slug: str, websocket: WebSocket) -> None:
        cleanup = False
        async with self._lock:
            bucket = self._connections.get(slug)
            if not bucket:
                return
            bucket.pop(websocket, None)
            if not bucket:
                self._connections.pop(slug, None)
                cleanup = True
        if cleanup:
            await self._drop_subscription(slug)

    async def broadcast(
        self,
        slug: Optional[str],
        message: Dict[str, Any],
        *,
        propagate: bool = True,
    ) -> Dict[str, Any]:
        target_slug = slug or message.get("slug")
        dataset_name = message.get("dataset")
        enriched = dict(message)
        meta = enriched.setdefault("meta", {})
        meta.setdefault("node", _RUNTIME_NODE_ID)
        if target_slug is not None:
            enriched["slug"] = target_slug
        elif dataset_name is not None:
            enriched["slug"] = dataset_name
        if "event_id" not in enriched:
            enriched["event_id"] = uuid.uuid4().hex
        enriched = _with_timestamp(enriched)

        listeners: Dict[WebSocket, Dict[str, Any]] = {}
        async with self._lock:
            if target_slug is not None and target_slug in self._connections:
                listeners = dict(self._connections[target_slug])
        meta["listeners"] = len(listeners)
        stale: List[WebSocket] = []
        for websocket in listeners:
            try:
                await websocket.send_json(enriched)
            except Exception:
                stale.append(websocket)
        if stale:
            await self._remove_stale(target_slug, stale)

        if propagate and target_slug:
            try:
                await publish_event(f"page::{target_slug}", enriched)
            except Exception:
                logger.exception("Failed to propagate realtime event for page '%s'", target_slug)
        return enriched

    async def has_listeners(self, slug: str) -> bool:
        async with self._lock:
            listeners = self._connections.get(slug)
            return bool(listeners)

    async def listener_count(self, slug: str) -> int:
        async with self._lock:
            listeners = self._connections.get(slug)
            return len(listeners) if listeners else 0

    async def _remove_stale(self, slug: Optional[str], websockets: Sequence[WebSocket]) -> None:
        if slug is None or not websockets:
            return
        cleanup = False
        async with self._lock:
            bucket = self._connections.get(slug)
            if not bucket:
                return
            for websocket in websockets:
                bucket.pop(websocket, None)
            if not bucket:
                self._connections.pop(slug, None)
                cleanup = True
        if cleanup:
            await self._drop_subscription(slug)

    async def _ensure_subscription(self, slug: str) -> None:
        if not use_redis_pubsub():
            return
        topic = f"page::{slug}"
        try:
            queue = await subscribe_topic(topic, queue_size=256, replay_last_event=True)
        except Exception:
            logger.exception("Failed to subscribe to realtime topic '%s'", topic)
            return
        relay = asyncio.create_task(self._relay_subscription(slug, topic, queue))
        async with self._lock:
            self._subscriptions[slug] = {"topic": topic, "queue": queue, "task": relay}

    async def _drop_subscription(self, slug: str) -> None:
        handle: Optional[Dict[str, Any]] = None
        async with self._lock:
            handle = self._subscriptions.pop(slug, None)
        if not handle:
            return
        topic = handle.get("topic")
        queue = handle.get("queue")
        task = handle.get("task")
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:  # pragma: no cover - expected on shutdown
                pass
        if topic and queue:
            try:
                await unsubscribe_topic(topic, queue)
            except Exception:
                logger.exception("Failed to unsubscribe from realtime topic '%s'", topic)

    async def _relay_subscription(self, slug: str, topic: str, queue: asyncio.Queue) -> None:
        try:
            while True:
                message = await queue.get()
                if not isinstance(message, dict):
                    continue
                await self.broadcast(slug, message, propagate=False)
        except asyncio.CancelledError:  # pragma: no cover - expected during shutdown
            raise
        except Exception:
            logger.exception("Realtime subscription relay for '%s' failed", topic)


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
    event = _make_page_event(
        slug,
        event_type=event_type,
        dataset=dataset,
        payload=payload,
        source=source,
        status=status,
        operation_id=operation_id,
        meta=meta,
    )
    await BROADCAST.broadcast(slug, event, propagate=propagate)
    return event


async def broadcast_dataset_refresh(
    slug: Optional[str],
    dataset_name: str,
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
    reason: str,
) -> None:
    targets: Set[str] = set()
    if slug:
        targets.add(slug)
    subscribers = DATASET_SUBSCRIBERS.get(dataset_name, set())
    if subscribers:
        targets.update(subscribers)
    cache_key = context.get("cache_key") if isinstance(context, dict) else None
    dataset_meta = {"reason": reason, "rows": len(rows)}
    if cache_key is not None:
        dataset_meta["cache_key"] = cache_key
    dataset_event = {
        "type": "dataset.snapshot",
        "dataset": dataset_name,
        "slug": slug,
        "rows": rows,
        "meta": _build_event_meta("dataset-refresh", dataset_meta),
    }
    await publish_dataset_event(dataset_name, _with_timestamp(dataset_event))
    if not targets:
        return
    for target_slug in sorted(targets):
        expanded_meta = dict(dataset_meta)
        expanded_meta["slug"] = target_slug
        await broadcast_page_event(
            target_slug,
            event_type="snapshot",
            dataset=dataset_name,
            payload={"rows": rows},
            source="dataset-refresh",
            status="ok",
            meta=expanded_meta,
        )


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
    extra_meta = dict(meta or {})
    extra_meta["status"] = status
    if error:
        extra_meta["error"] = error
    event = await broadcast_page_event(
        slug,
        event_type="mutation",
        dataset=dataset_name,
        payload=payload,
        source=source,
        status=status,
        operation_id=operation_id,
        meta=extra_meta,
    )
    dataset_event = {
        "type": "dataset.mutation",
        "dataset": dataset_name,
        "slug": slug,
        "payload": payload,
        "meta": _build_event_meta(source, {"status": status, "operation_id": operation_id, "error": error}),
    }
    await publish_dataset_event(dataset_name, _with_timestamp(dataset_event))
    return event


async def resolve_websocket_context(websocket: WebSocket) -> Dict[str, Any]:
    context = get_request_context({})
    headers = getattr(websocket, "headers", {}) or {}
    api_key = os.getenv("NAMEL3SS_API_KEY")
    provided_key: Optional[str] = None
    if hasattr(headers, "get"):
        provided_key = headers.get("x-api-key") or headers.get("X-Api-Key")
    if not provided_key and hasattr(headers, "get"):
        auth_header = headers.get("authorization") or headers.get("Authorization")
        if isinstance(auth_header, str) and auth_header.lower().startswith("bearer "):
            provided_key = auth_header[7:].strip()
    if api_key and provided_key != api_key:
        raise WebSocketDisconnect(code=4401)
    if not context and hasattr(websocket, "query_params"):
        try:
            tenant_hint = websocket.query_params.get("tenant")  # type: ignore[attr-defined]
        except Exception:
            tenant_hint = None
        if tenant_hint and isinstance(tenant_hint, str):
            context = {"tenant": tenant_hint}
    return context
'''
        return textwrap.dedent(broadcast).strip()

    fallback = '''
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
'''
    return textwrap.dedent(fallback).strip()
