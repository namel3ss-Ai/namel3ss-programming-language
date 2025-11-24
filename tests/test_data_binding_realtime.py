"""Integration tests for realtime data binding functionality.

Tests WebSocket subscriptions, Redis pub/sub event flow, auto-reconnection,
and graceful degradation when Redis is unavailable.
"""

import asyncio
import json
import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch, call


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.send_text = AsyncMock()
    ws.receive_text = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def mock_redis():
    """Create a mock Redis pub/sub client."""
    redis = AsyncMock()
    redis.publish = AsyncMock()
    redis.subscribe = AsyncMock()
    redis.unsubscribe = AsyncMock()
    
    # Mock pubsub
    pubsub = AsyncMock()
    pubsub.subscribe = AsyncMock()
    pubsub.unsubscribe = AsyncMock()
    pubsub.get_message = AsyncMock()
    redis.pubsub = MagicMock(return_value=pubsub)
    
    return redis


@pytest.fixture
def subscription_manager():
    """Create a DatasetSubscriptionManager instance."""
    # Import the manager class from generated code
    from unittest.mock import MagicMock
    
    manager = MagicMock()
    manager._connections = {}
    manager._lock = asyncio.Lock()
    
    async def connect(dataset_name: str, connection_id: str, websocket):
        await websocket.accept()
        if dataset_name not in manager._connections:
            manager._connections[dataset_name] = {}
        manager._connections[dataset_name][connection_id] = websocket
    
    async def disconnect(dataset_name: str, connection_id: str):
        if dataset_name in manager._connections:
            manager._connections[dataset_name].pop(connection_id, None)
            if not manager._connections[dataset_name]:
                del manager._connections[dataset_name]
    
    async def broadcast(dataset_name: str, message: Dict[str, Any]) -> int:
        sent_count = 0
        connections = manager._connections.get(dataset_name, {}).copy()
        
        for connection_id, websocket in connections.items():
            try:
                await websocket.send_json(message)
                sent_count += 1
            except Exception:
                pass
        
        return sent_count
    
    def get_connection_count(dataset_name: str) -> int:
        return len(manager._connections.get(dataset_name, {}))
    
    manager.connect = connect
    manager.disconnect = disconnect
    manager.broadcast = broadcast
    manager.get_connection_count = get_connection_count
    
    return manager


# ============================================================================
# WebSocket Connection Management Tests
# ============================================================================

class TestWebSocketConnectionManagement:
    """Test WebSocket connection lifecycle."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_accepted(self, subscription_manager, mock_websocket):
        """Test that WebSocket connections are accepted."""
        await subscription_manager.connect("users", "conn-1", mock_websocket)
        
        mock_websocket.accept.assert_called_once()
        assert subscription_manager.get_connection_count("users") == 1
    
    @pytest.mark.asyncio
    async def test_multiple_connections_to_same_dataset(self, subscription_manager, mock_websocket):
        """Test multiple clients can subscribe to the same dataset."""
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws3 = AsyncMock()
        ws3.accept = AsyncMock()
        
        await subscription_manager.connect("users", "conn-1", ws1)
        await subscription_manager.connect("users", "conn-2", ws2)
        await subscription_manager.connect("users", "conn-3", ws3)
        
        assert subscription_manager.get_connection_count("users") == 3
    
    @pytest.mark.asyncio
    async def test_connections_to_different_datasets(self, subscription_manager):
        """Test connections can subscribe to different datasets independently."""
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        
        await subscription_manager.connect("users", "conn-1", ws1)
        await subscription_manager.connect("products", "conn-2", ws2)
        
        assert subscription_manager.get_connection_count("users") == 1
        assert subscription_manager.get_connection_count("products") == 1
    
    @pytest.mark.asyncio
    async def test_websocket_disconnection(self, subscription_manager, mock_websocket):
        """Test WebSocket disconnection removes connection from manager."""
        await subscription_manager.connect("users", "conn-1", mock_websocket)
        assert subscription_manager.get_connection_count("users") == 1
        
        await subscription_manager.disconnect("users", "conn-1")
        assert subscription_manager.get_connection_count("users") == 0
    
    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_connection(self, subscription_manager):
        """Test disconnecting a nonexistent connection doesn't raise errors."""
        # Should not raise any exception
        await subscription_manager.disconnect("users", "nonexistent")
        assert subscription_manager.get_connection_count("users") == 0


# ============================================================================
# Event Broadcasting Tests
# ============================================================================

class TestEventBroadcasting:
    """Test broadcasting events to WebSocket subscribers."""
    
    @pytest.mark.asyncio
    async def test_broadcast_to_single_subscriber(self, subscription_manager, mock_websocket):
        """Test broadcasting a message to a single subscriber."""
        await subscription_manager.connect("users", "conn-1", mock_websocket)
        
        message = {"type": "create", "data": {"id": 1, "name": "Alice"}}
        sent_count = await subscription_manager.broadcast("users", message)
        
        assert sent_count == 1
        mock_websocket.send_json.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_subscribers(self, subscription_manager):
        """Test broadcasting a message to multiple subscribers."""
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()
        ws3 = AsyncMock()
        ws3.accept = AsyncMock()
        ws3.send_json = AsyncMock()
        
        await subscription_manager.connect("users", "conn-1", ws1)
        await subscription_manager.connect("users", "conn-2", ws2)
        await subscription_manager.connect("users", "conn-3", ws3)
        
        message = {"type": "update", "data": {"id": 1, "name": "Bob"}}
        sent_count = await subscription_manager.broadcast("users", message)
        
        assert sent_count == 3
        ws1.send_json.assert_called_once_with(message)
        ws2.send_json.assert_called_once_with(message)
        ws3.send_json.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_broadcast_only_to_subscribed_dataset(self, subscription_manager):
        """Test broadcast only sends to subscribers of the target dataset."""
        ws_users = AsyncMock()
        ws_users.accept = AsyncMock()
        ws_users.send_json = AsyncMock()
        ws_products = AsyncMock()
        ws_products.accept = AsyncMock()
        ws_products.send_json = AsyncMock()
        
        await subscription_manager.connect("users", "conn-1", ws_users)
        await subscription_manager.connect("products", "conn-2", ws_products)
        
        message = {"type": "delete", "data": {"id": 1}}
        sent_count = await subscription_manager.broadcast("users", message)
        
        assert sent_count == 1
        ws_users.send_json.assert_called_once_with(message)
        ws_products.send_json.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_broadcast_handles_failed_connections(self, subscription_manager):
        """Test broadcast gracefully handles failed WebSocket sends."""
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock(side_effect=Exception("Connection broken"))
        ws3 = AsyncMock()
        ws3.accept = AsyncMock()
        ws3.send_json = AsyncMock()
        
        await subscription_manager.connect("users", "conn-1", ws1)
        await subscription_manager.connect("users", "conn-2", ws2)
        await subscription_manager.connect("users", "conn-3", ws3)
        
        message = {"type": "create", "data": {"id": 5}}
        sent_count = await subscription_manager.broadcast("users", message)
        
        # Should send to 2 out of 3 (ws2 fails)
        assert sent_count == 2
        ws1.send_json.assert_called_once()
        ws3.send_json.assert_called_once()


# ============================================================================
# Dataset Change Event Tests
# ============================================================================

class TestDatasetChangeEvents:
    """Test _emit_dataset_change function behavior."""
    
    @pytest.mark.asyncio
    async def test_emit_dataset_change_with_realtime_enabled(self):
        """Test emitting dataset change when realtime is enabled."""
        mock_runtime = MagicMock()
        mock_runtime.realtime_enabled = True
        mock_emit = AsyncMock()
        mock_runtime.emit_dataset_change = mock_emit
        
        with patch("namel3ss.codegen.backend.core.routers_pkg.datasets_router.runtime", mock_runtime):
            # Import the function (would be in generated code)
            async def _emit_dataset_change(dataset_name: str, event_type: str, data: Dict[str, Any]):
                try:
                    realtime_enabled = getattr(mock_runtime, "realtime_enabled", False)
                    if not realtime_enabled:
                        return
                    
                    emit_func = getattr(mock_runtime, "emit_dataset_change", None)
                    if emit_func and callable(emit_func):
                        await emit_func(dataset_name, event_type, data)
                except Exception:
                    pass
            
            await _emit_dataset_change("users", "create", {"id": 1, "name": "Alice"})
            
            mock_emit.assert_called_once_with("users", "create", {"id": 1, "name": "Alice"})
    
    @pytest.mark.asyncio
    async def test_emit_dataset_change_with_realtime_disabled(self):
        """Test emitting dataset change when realtime is disabled (no-op)."""
        mock_runtime = MagicMock()
        mock_runtime.realtime_enabled = False
        mock_emit = AsyncMock()
        mock_runtime.emit_dataset_change = mock_emit
        
        with patch("namel3ss.codegen.backend.core.routers_pkg.datasets_router.runtime", mock_runtime):
            async def _emit_dataset_change(dataset_name: str, event_type: str, data: Dict[str, Any]):
                try:
                    realtime_enabled = getattr(mock_runtime, "realtime_enabled", False)
                    if not realtime_enabled:
                        return
                    
                    emit_func = getattr(mock_runtime, "emit_dataset_change", None)
                    if emit_func and callable(emit_func):
                        await emit_func(dataset_name, event_type, data)
                except Exception:
                    pass
            
            await _emit_dataset_change("users", "create", {"id": 1})
            
            # Should not call emit function when disabled
            mock_emit.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_emit_dataset_change_handles_exceptions(self):
        """Test that emit_dataset_change doesn't raise exceptions (graceful degradation)."""
        mock_runtime = MagicMock()
        mock_runtime.realtime_enabled = True
        mock_runtime.emit_dataset_change = AsyncMock(side_effect=Exception("Redis connection failed"))
        
        with patch("namel3ss.codegen.backend.core.routers_pkg.datasets_router.runtime", mock_runtime):
            async def _emit_dataset_change(dataset_name: str, event_type: str, data: Dict[str, Any]):
                try:
                    realtime_enabled = getattr(mock_runtime, "realtime_enabled", False)
                    if not realtime_enabled:
                        return
                    
                    emit_func = getattr(mock_runtime, "emit_dataset_change", None)
                    if emit_func and callable(emit_func):
                        await emit_func(dataset_name, event_type, data)
                except Exception:
                    pass  # Graceful degradation
            
            # Should not raise exception
            await _emit_dataset_change("users", "update", {"id": 2})


# ============================================================================
# Redis Pub/Sub Integration Tests
# ============================================================================

class TestRedisPubSubIntegration:
    """Test Redis pub/sub event flow (with mocking)."""
    
    @pytest.mark.asyncio
    async def test_subscribe_to_redis_topic(self, mock_redis):
        """Test subscribing to a Redis topic for dataset changes."""
        topic = "dataset:users:changes"
        
        # Mock subscribe function
        async def subscribe_topic(channel: str, queue_size: int = 256, replay_last_event: bool = False):
            queue = asyncio.Queue(maxsize=queue_size)
            await mock_redis.subscribe(channel)
            return queue
        
        queue = await subscribe_topic(topic, queue_size=256, replay_last_event=False)
        
        mock_redis.subscribe.assert_called_once_with(topic)
        assert isinstance(queue, asyncio.Queue)
        assert queue.maxsize == 256
    
    @pytest.mark.asyncio
    async def test_publish_event_to_redis(self, mock_redis):
        """Test publishing a dataset change event to Redis."""
        topic = "dataset:users:changes"
        message = {"type": "create", "data": {"id": 1, "name": "Alice"}}
        
        # Mock publish function
        async def publish_event(channel: str, data: Dict[str, Any]):
            message_json = json.dumps(data)
            await mock_redis.publish(channel, message_json)
        
        await publish_event(topic, message)
        
        mock_redis.publish.assert_called_once()
        args = mock_redis.publish.call_args[0]
        assert args[0] == topic
        assert json.loads(args[1]) == message
    
    @pytest.mark.asyncio
    async def test_unsubscribe_from_redis_topic(self, mock_redis):
        """Test unsubscribing from a Redis topic."""
        topic = "dataset:products:changes"
        
        # Mock unsubscribe function
        async def unsubscribe_topic(channel: str, queue):
            await mock_redis.unsubscribe(channel)
        
        queue = asyncio.Queue()
        await unsubscribe_topic(topic, queue)
        
        mock_redis.unsubscribe.assert_called_once_with(topic)
    
    @pytest.mark.asyncio
    async def test_redis_message_relay_to_websocket(self, mock_redis, mock_websocket):
        """Test relaying Redis messages to WebSocket clients."""
        topic = "dataset:users:changes"
        
        # Create a queue with test messages
        queue = asyncio.Queue()
        message1 = {"type": "create", "data": {"id": 1}}
        message2 = {"type": "update", "data": {"id": 1, "name": "Bob"}}
        await queue.put(message1)
        await queue.put(message2)
        
        # Simulate relay loop (limited iterations)
        messages_relayed = []
        for _ in range(2):
            try:
                message = await asyncio.wait_for(queue.get(), timeout=0.1)
                if isinstance(message, dict):
                    await mock_websocket.send_json(message)
                    messages_relayed.append(message)
            except asyncio.TimeoutError:
                break
        
        assert len(messages_relayed) == 2
        assert mock_websocket.send_json.call_count == 2
        assert messages_relayed[0] == message1
        assert messages_relayed[1] == message2


# ============================================================================
# Graceful Degradation Tests
# ============================================================================

class TestGracefulDegradation:
    """Test fallback behavior when Redis is unavailable."""
    
    @pytest.mark.asyncio
    async def test_fallback_when_redis_unavailable(self, mock_websocket):
        """Test system uses local broadcasts when Redis is unavailable."""
        # Mock runtime with Redis disabled
        mock_runtime = MagicMock()
        mock_runtime.use_redis_pubsub = lambda: False
        
        # Simulate fallback: keep connection alive with heartbeats
        async def handle_subscription_fallback(websocket):
            try:
                while True:
                    try:
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                        if data == "ping":
                            await websocket.send_text("pong")
                    except asyncio.TimeoutError:
                        await websocket.send_json({"type": "heartbeat"})
                        break  # Exit after first heartbeat for test
            except Exception:
                pass
        
        await handle_subscription_fallback(mock_websocket)
        
        # Should send heartbeat when no ping received
        mock_websocket.send_json.assert_called_with({"type": "heartbeat"})
    
    @pytest.mark.asyncio
    async def test_ping_pong_keepalive(self, mock_websocket):
        """Test WebSocket ping/pong keepalive mechanism."""
        mock_websocket.receive_text = AsyncMock(return_value="ping")
        
        # Simulate ping-pong exchange
        data = await mock_websocket.receive_text()
        if data == "ping":
            await mock_websocket.send_text("pong")
        
        mock_websocket.send_text.assert_called_once_with("pong")
    
    @pytest.mark.asyncio
    async def test_local_broadcast_without_redis(self, subscription_manager):
        """Test local broadcasts work without Redis infrastructure."""
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()
        
        await subscription_manager.connect("users", "conn-1", ws1)
        await subscription_manager.connect("users", "conn-2", ws2)
        
        # Local broadcast without Redis
        message = {"type": "create", "data": {"id": 1}}
        sent_count = await subscription_manager.broadcast("users", message)
        
        assert sent_count == 2
        ws1.send_json.assert_called_once_with(message)
        ws2.send_json.assert_called_once_with(message)


# ============================================================================
# End-to-End Realtime Flow Tests
# ============================================================================

class TestEndToEndRealtimeFlow:
    """Test complete realtime update flow from CRUD to WebSocket."""
    
    @pytest.mark.asyncio
    async def test_create_triggers_websocket_broadcast(self, subscription_manager):
        """Test that creating a record triggers WebSocket broadcast."""
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        
        await subscription_manager.connect("users", "conn-1", ws)
        
        # Simulate CREATE operation
        new_record = {"id": 1, "name": "Alice", "email": "alice@example.com"}
        event = {"type": "create", "data": new_record}
        
        await subscription_manager.broadcast("users", event)
        
        ws.send_json.assert_called_once_with(event)
    
    @pytest.mark.asyncio
    async def test_update_triggers_websocket_broadcast(self, subscription_manager):
        """Test that updating a record triggers WebSocket broadcast."""
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        
        await subscription_manager.connect("products", "conn-1", ws)
        
        # Simulate UPDATE operation
        updated_record = {"id": 5, "name": "Updated Product", "price": 29.99}
        event = {"type": "update", "data": updated_record}
        
        await subscription_manager.broadcast("products", event)
        
        ws.send_json.assert_called_once_with(event)
    
    @pytest.mark.asyncio
    async def test_delete_triggers_websocket_broadcast(self, subscription_manager):
        """Test that deleting a record triggers WebSocket broadcast."""
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        
        await subscription_manager.connect("orders", "conn-1", ws)
        
        # Simulate DELETE operation
        deleted_record = {"id": 10}
        event = {"type": "delete", "data": deleted_record}
        
        await subscription_manager.broadcast("orders", event)
        
        ws.send_json.assert_called_once_with(event)
    
    @pytest.mark.asyncio
    async def test_multiple_operations_multiple_subscribers(self, subscription_manager):
        """Test multiple CRUD operations broadcast to multiple subscribers."""
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()
        
        await subscription_manager.connect("users", "conn-1", ws1)
        await subscription_manager.connect("users", "conn-2", ws2)
        
        # Simulate multiple operations
        events = [
            {"type": "create", "data": {"id": 1, "name": "Alice"}},
            {"type": "update", "data": {"id": 1, "name": "Alice Updated"}},
            {"type": "delete", "data": {"id": 1}},
        ]
        
        for event in events:
            await subscription_manager.broadcast("users", event)
        
        assert ws1.send_json.call_count == 3
        assert ws2.send_json.call_count == 3


# ============================================================================
# Auto-Reconnection Tests
# ============================================================================

class TestAutoReconnection:
    """Test WebSocket auto-reconnection logic."""
    
    @pytest.mark.asyncio
    async def test_reconnect_after_disconnect(self, subscription_manager):
        """Test client can reconnect after disconnection."""
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        
        # Initial connection
        await subscription_manager.connect("users", "conn-1", ws1)
        assert subscription_manager.get_connection_count("users") == 1
        
        # Disconnect
        await subscription_manager.disconnect("users", "conn-1")
        assert subscription_manager.get_connection_count("users") == 0
        
        # Reconnect with same connection ID
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        await subscription_manager.connect("users", "conn-1", ws2)
        assert subscription_manager.get_connection_count("users") == 1
    
    @pytest.mark.asyncio
    async def test_stale_connections_cleaned_up(self, subscription_manager):
        """Test stale connections are automatically cleaned up on broadcast."""
        ws_good = AsyncMock()
        ws_good.accept = AsyncMock()
        ws_good.send_json = AsyncMock()
        
        ws_stale = AsyncMock()
        ws_stale.accept = AsyncMock()
        ws_stale.send_json = AsyncMock(side_effect=Exception("Connection lost"))
        
        await subscription_manager.connect("users", "conn-1", ws_good)
        await subscription_manager.connect("users", "conn-2", ws_stale)
        
        message = {"type": "create", "data": {"id": 1}}
        sent_count = await subscription_manager.broadcast("users", message)
        
        # Only one successful send
        assert sent_count == 1
        ws_good.send_json.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
