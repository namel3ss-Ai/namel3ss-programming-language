"""
Real-time data broadcasting using Redis pub/sub for dataset changes.

This module provides infrastructure for broadcasting dataset changes
to connected clients in real-time using Redis as the message broker.
"""

import json
import asyncio
import logging
from typing import Any, Dict, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    try:
        import aioredis
    except ImportError:
        pass

try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Global Redis connection pool
_redis_pool: Optional[Any] = None


async def init_redis(redis_url: str = "redis://localhost:6379") -> None:
    """Initialize Redis connection pool for broadcasting."""
    global _redis_pool
    
    if not REDIS_AVAILABLE:
        logger.warning("Redis not available - realtime features will be disabled")
        return
    
    try:
        _redis_pool = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
        
        # Test the connection
        await _redis_pool.ping()
        logger.info("Redis connection initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
        _redis_pool = None


async def close_redis() -> None:
    """Close Redis connection pool."""
    global _redis_pool
    
    if _redis_pool:
        try:
            await _redis_pool.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
        finally:
            _redis_pool = None


async def broadcast_dataset_change(
    dataset_name: str,
    event_type: str,
    data: Dict[str, Any],
    channel_prefix: str = "dataset_updates",
) -> bool:
    """
    Broadcast a dataset change event to all subscribed clients.
    
    Args:
        dataset_name: Name of the dataset that changed
        event_type: Type of change ('create', 'update', 'delete')
        data: The data that changed
        channel_prefix: Redis channel prefix
        
    Returns:
        True if broadcast was successful, False otherwise
    """
    if not _redis_pool:
        logger.debug("Redis not available - skipping broadcast")
        return False
    
    with tracer.start_as_current_span("broadcast_dataset_change") as span:
        try:
            span.set_attribute("dataset.name", dataset_name)
            span.set_attribute("event.type", event_type)
            
            # Prepare message payload
            message = {
                "dataset": dataset_name,
                "event_type": event_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            # Broadcast to multiple channels for flexible subscription patterns
            channels = [
                f"{channel_prefix}:all",  # All dataset updates
                f"{channel_prefix}:{dataset_name}",  # Specific dataset updates
                f"{channel_prefix}:{dataset_name}:{event_type}",  # Specific dataset + event type
            ]
            
            published_count = 0
            for channel in channels:
                try:
                    result = await _redis_pool.publish(channel, json.dumps(message))
                    published_count += result
                    span.set_attribute(f"channel.{channel}.subscribers", result)
                except Exception as channel_error:
                    logger.warning(f"Failed to publish to channel {channel}: {channel_error}")
            
            span.set_attribute("broadcast.total_subscribers", published_count)
            logger.debug(
                f"Broadcasted {event_type} event for {dataset_name} to {published_count} subscribers"
            )
            
            return published_count > 0
            
        except Exception as e:
            logger.error(f"Failed to broadcast dataset change: {e}")
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            return False


async def subscribe_to_dataset_changes(
    dataset_name: Optional[str] = None,
    event_types: Optional[list] = None,
    channel_prefix: str = "dataset_updates",
) -> Any:
    """
    Subscribe to dataset change events.
    
    Args:
        dataset_name: Specific dataset to subscribe to (None for all)
        event_types: List of event types to subscribe to
        channel_prefix: Redis channel prefix
        
    Returns:
        Redis PubSub object for receiving messages
        
    Raises:
        RuntimeError: If Redis is not available
    """
    if not _redis_pool:
        raise RuntimeError("Redis not available for subscriptions")
    
    pubsub = _redis_pool.pubsub()
    
    # Determine channels to subscribe to
    if dataset_name is None:
        # Subscribe to all dataset changes
        await pubsub.subscribe(f"{channel_prefix}:all")
    elif event_types is None:
        # Subscribe to all events for specific dataset
        await pubsub.subscribe(f"{channel_prefix}:{dataset_name}")
    else:
        # Subscribe to specific dataset + event type combinations
        for event_type in event_types:
            await pubsub.subscribe(f"{channel_prefix}:{dataset_name}:{event_type}")
    
    return pubsub


class DatasetChangeHandler:
    """Handler for processing dataset change events."""
    
    def __init__(self, dataset_name: Optional[str] = None):
        self.dataset_name = dataset_name
        self._pubsub: Optional[Any] = None
        self._running = False
    
    async def start_listening(self) -> None:
        """Start listening for dataset changes."""
        if not _redis_pool:
            logger.warning("Redis not available - dataset change handler disabled")
            return
        
        try:
            self._pubsub = await subscribe_to_dataset_changes(self.dataset_name)
            self._running = True
            
            logger.info(f"Started listening for dataset changes (dataset: {self.dataset_name or 'all'})")
            
            async for message in self._pubsub.listen():
                if not self._running:
                    break
                
                if message["type"] == "message":
                    await self._handle_message(message["data"])
                    
        except Exception as e:
            logger.error(f"Error in dataset change handler: {e}")
            await self.stop_listening()
    
    async def stop_listening(self) -> None:
        """Stop listening for dataset changes."""
        self._running = False
        
        if self._pubsub:
            try:
                await self._pubsub.unsubscribe()
                await self._pubsub.close()
            except Exception as e:
                logger.error(f"Error closing pubsub: {e}")
            finally:
                self._pubsub = None
        
        logger.info("Stopped listening for dataset changes")
    
    async def _handle_message(self, message_data: str) -> None:
        """Handle a dataset change message."""
        try:
            message = json.loads(message_data)
            
            dataset_name = message.get("dataset")
            event_type = message.get("event_type")
            data = message.get("data", {})
            timestamp = message.get("timestamp")
            
            logger.debug(f"Received {event_type} event for dataset {dataset_name}")
            
            # Override this method in subclasses to handle specific events
            await self.on_dataset_change(
                dataset_name=dataset_name,
                event_type=event_type,
                data=data,
                timestamp=timestamp,
            )
            
        except Exception as e:
            logger.error(f"Error handling dataset change message: {e}")
    
    async def on_dataset_change(
        self,
        dataset_name: str,
        event_type: str,
        data: Dict[str, Any],
        timestamp: str,
    ) -> None:
        """
        Override this method to handle dataset changes.
        
        Args:
            dataset_name: Name of the dataset that changed
            event_type: Type of change ('create', 'update', 'delete')
            data: The data that changed
            timestamp: ISO timestamp of the change
        """
        pass


async def get_redis_connection() -> Optional[Any]:
    """Get the current Redis connection."""
    return _redis_pool


def is_redis_available() -> bool:
    """Check if Redis is available and connected."""
    return _redis_pool is not None