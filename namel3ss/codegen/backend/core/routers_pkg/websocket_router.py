"""Generate WebSocket router for realtime dataset subscriptions."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from namel3ss.ir import BackendIR


def _render_websocket_router_module(backend_ir: "BackendIR", enable_realtime: bool = False) -> str:
    """Generate WebSocket router for realtime dataset subscriptions.
    
    This router provides WebSocket endpoints for:
    - Real-time dataset change subscriptions
    - Live updates for create/update/delete operations
    - Automatic reconnection support
    
    Only generated when enable_realtime=True.
    """
    
    if not enable_realtime:
        # Return minimal no-op router when realtime is disabled
        template = '''
"""WebSocket router (disabled - realtime not enabled)."""

from fastapi import APIRouter

router = APIRouter(prefix="/ws", tags=["websocket"])

# Realtime WebSocket support is disabled
# To enable, set enable_realtime=True in backend generation
# or install the [realtime] extra: pip install namel3ss[realtime]

__all__ = ["router"]
'''
        return textwrap.dedent(template).strip() + "\n"
    
    # Get realtime-enabled datasets
    realtime_datasets = [ds for ds in backend_ir.datasets if ds.realtime_enabled]
    
    # Generate subscription endpoints
    subscription_endpoints = []
    for dataset in realtime_datasets:
        endpoint_code = _generate_subscription_endpoint(dataset)
        subscription_endpoints.append(endpoint_code)
    
    endpoints_code = "\n\n".join(subscription_endpoints) if subscription_endpoints else "# No realtime datasets configured"
    
    template = '''
"""WebSocket router for realtime dataset subscriptions."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from ...database import get_session
from .. import runtime
from ..helpers import router_dependencies

router = APIRouter(prefix="/ws", tags=["websocket"])
logger = logging.getLogger(__name__)


class DatasetSubscriptionManager:
    """Manages WebSocket connections for dataset subscriptions."""
    
    def __init__(self) -> None:
        self._connections: Dict[str, Dict[str, WebSocket]] = {{}}
        self._lock = asyncio.Lock()
    
    async def connect(self, dataset_name: str, connection_id: str, websocket: WebSocket) -> None:
        """Register a new WebSocket connection for a dataset."""
        await websocket.accept()
        async with self._lock:
            if dataset_name not in self._connections:
                self._connections[dataset_name] = {{}}
            self._connections[dataset_name][connection_id] = websocket
    
    async def disconnect(self, dataset_name: str, connection_id: str) -> None:
        """Unregister a WebSocket connection."""
        async with self._lock:
            if dataset_name in self._connections:
                self._connections[dataset_name].pop(connection_id, None)
                if not self._connections[dataset_name]:
                    del self._connections[dataset_name]
    
    async def broadcast(self, dataset_name: str, message: Dict[str, Any]) -> int:
        """Broadcast a message to all subscribers of a dataset.
        
        Returns the number of active connections that received the message.
        """
        sent_count = 0
        stale_connections = []
        
        async with self._lock:
            connections = self._connections.get(dataset_name, {{}}).copy()
        
        for connection_id, websocket in connections.items():
            try:
                await websocket.send_json(message)
                sent_count += 1
            except Exception as e:
                logger.debug(f"Failed to send to connection {{connection_id}}: {{e}}")
                stale_connections.append((dataset_name, connection_id))
        
        # Clean up stale connections
        if stale_connections:
            async with self._lock:
                for ds_name, conn_id in stale_connections:
                    if ds_name in self._connections:
                        self._connections[ds_name].pop(conn_id, None)
        
        return sent_count
    
    def get_connection_count(self, dataset_name: str) -> int:
        """Get the number of active connections for a dataset."""
        return len(self._connections.get(dataset_name, {{}}))


# Global subscription manager
_subscription_manager = DatasetSubscriptionManager()


async def _handle_dataset_subscription(
    dataset_name: str,
    websocket: WebSocket,
    connection_id: str,
) -> None:
    """Handle a WebSocket subscription to a dataset.
    
    This coroutine manages the lifecycle of a subscription:
    1. Accepts the connection
    2. Subscribes to Redis pub/sub for the dataset
    3. Relays messages to the client
    4. Handles disconnection and cleanup
    """
    await _subscription_manager.connect(dataset_name, connection_id, websocket)
    
    try:
        # Subscribe to Redis pub/sub channel
        topic = f"dataset:{{dataset_name}}:changes"
        
        # Check if Redis pub/sub is available
        use_redis = getattr(runtime, "use_redis_pubsub", lambda: False)()
        
        if use_redis:
            # Subscribe to Redis channel
            subscribe_func = getattr(runtime, "subscribe_topic", None)
            if subscribe_func:
                queue = await subscribe_func(topic, queue_size=256, replay_last_event=False)
                
                # Relay messages from Redis to WebSocket
                try:
                    while True:
                        message = await queue.get()
                        if isinstance(message, dict):
                            await websocket.send_json(message)
                except asyncio.CancelledError:
                    pass
                finally:
                    # Unsubscribe from Redis
                    unsubscribe_func = getattr(runtime, "unsubscribe_topic", None)
                    if unsubscribe_func:
                        await unsubscribe_func(topic, queue)
        else:
            # Fallback: Keep connection alive for local broadcasts
            while True:
                # Receive ping messages to keep connection alive
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    if data == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    # Send heartbeat
                    await websocket.send_json({{"type": "heartbeat"}})
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"WebSocket error for dataset {{dataset_name}}: {{e}}")
    finally:
        await _subscription_manager.disconnect(dataset_name, connection_id)


{endpoints_code}


__all__ = ["router"]
'''
    
    return textwrap.dedent(template).strip().format(endpoints_code=endpoints_code) + "\n"


def _generate_subscription_endpoint(dataset_spec) -> str:
    """Generate a WebSocket subscription endpoint for a dataset."""
    
    dataset_name = dataset_spec.name
    func_name = f"subscribe_to_{dataset_name}"
    
    template = '''
@router.websocket("/{dataset_name}")
async def {func_name}(websocket: WebSocket):
    """WebSocket endpoint for {dataset_name} realtime updates.
    
    Clients connect to this endpoint to receive real-time notifications
    when records are created, updated, or deleted in the {dataset_name} dataset.
    
    Message format:
    {{
        "type": "dataset.create|update|delete",
        "dataset": "{dataset_name}",
        "payload": {{...record data...}},
        "meta": {{
            "event_type": "create|update|delete",
            "ts": 1234567890.123,
            "node": "abc123..."
        }}
    }}
    """
    import uuid
    connection_id = uuid.uuid4().hex
    await _handle_dataset_subscription("{dataset_name}", websocket, connection_id)
'''
    
    return textwrap.dedent(template).strip().format(
        dataset_name=dataset_name,
        func_name=func_name,
    )
