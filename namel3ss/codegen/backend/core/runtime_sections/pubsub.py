from __future__ import annotations

from textwrap import dedent

PUBSUB_SECTION = dedent(
    '''

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

    '''
).strip()

__all__ = ["PUBSUB_SECTION"]
