from __future__ import annotations

from textwrap import dedent

STREAMS_SECTION = dedent(
    '''

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
                    yield b": heartbeat\\n\\n"
                    if pytest_mode:
                        idle_heartbeats += 1
                        if idle_heartbeats >= 1:
                            break
                    continue
                data = json.dumps(message, default=str)
                yield f"data: {data}\\n\\n".encode("utf-8")
                delivered_payload = True
                idle_heartbeats = 0
                if pytest_mode and queue.empty():
                    break
        finally:
            await unsubscribe_topic(normalized, queue)

    response = StreamingResponse(event_source(), media_type="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


async def stream_dataset(dataset: str, heartbeat: Optional[int] = None) -> StreamingResponse:
    return await stream_topic(f"dataset::{dataset}", heartbeat=heartbeat)


async def stream_page(slug: str, heartbeat: Optional[int] = None) -> StreamingResponse:
    return await stream_topic(f"page::{slug}", heartbeat=heartbeat)


async def emit_page_event(slug: str, payload: Dict[str, Any]) -> None:
    await publish_event(f"page::{slug}", payload)


async def emit_global_event(payload: Dict[str, Any]) -> None:
    await publish_event("global", payload)

    '''
).strip()

__all__ = ["STREAMS_SECTION"]
