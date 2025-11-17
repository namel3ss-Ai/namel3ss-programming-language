"""Realtime broadcast manager blocks for the runtime module."""

from __future__ import annotations

import textwrap


def render_broadcast_block(enable_realtime: bool) -> str:
    """Render the WebSocket broadcast helpers used by reactive pages."""
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


__all__ = ["render_broadcast_block"]
