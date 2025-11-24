"""
Event-Driven Runtime System for Namel3ss.

This module implements reactive event-driven execution patterns with support for:
- Event triggers and reactive workflows
- WebSocket-based real-time event streaming
- Event routing and filtering
- Async event handling and processing
- Integration with parallel and distributed execution

Enables building reactive, event-driven AI applications with Namel3ss.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    websockets = None
    WebSocketServerProtocol = None

try:
    import redis.asyncio as redis
except ImportError:
    redis = None


# =============================================================================
# Core Event System Data Structures
# =============================================================================

class EventType(Enum):
    """Types of events in the system."""
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    USER_INPUT = "user_input"
    SYSTEM_ALERT = "system_alert"
    CUSTOM = "custom"


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """Represents a system event."""
    event_id: str
    event_type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'source': self.source,
            'data': self.data,
            'timestamp': self.timestamp,
            'priority': self.priority.value,
            'correlation_id': self.correlation_id,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        return cls(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            source=data['source'],
            data=data['data'],
            timestamp=data['timestamp'],
            priority=EventPriority(data['priority']),
            correlation_id=data.get('correlation_id'),
            metadata=data.get('metadata', {}),
        )


@dataclass
class EventHandler:
    """Represents an event handler."""
    handler_id: str
    event_types: Set[EventType]
    handler_func: Callable[[Event], Any]
    priority: int = 0
    filters: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    max_concurrency: int = 10
    timeout_seconds: Optional[float] = None


@dataclass
class EventSubscription:
    """Represents an event subscription."""
    subscription_id: str
    subscriber_id: str
    event_types: Set[EventType]
    filters: Dict[str, Any] = field(default_factory=dict)
    delivery_mode: str = "async"  # async, sync, batch
    active: bool = True
    created_at: float = field(default_factory=time.time)


# =============================================================================
# Event Bus and Router
# =============================================================================

class EventBus(ABC):
    """Abstract event bus for publishing and subscribing to events."""
    
    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish an event."""
        pass
    
    @abstractmethod
    async def subscribe(
        self, 
        event_types: Set[EventType], 
        handler: Callable[[Event], Any]
    ) -> str:
        """Subscribe to events."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events."""
        pass


class MemoryEventBus(EventBus):
    """In-memory event bus implementation."""
    
    def __init__(self):
        """Initialize memory event bus."""
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.handlers: Dict[str, EventHandler] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_failed': 0,
            'active_subscriptions': 0,
        }
    
    async def start(self) -> None:
        """Start the event bus."""
        if not self.processing_task:
            self.processing_task = asyncio.create_task(self._process_events())
            logger.info("Memory event bus started")
    
    async def stop(self) -> None:
        """Stop the event bus."""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None
            logger.info("Memory event bus stopped")
    
    async def publish(self, event: Event) -> None:
        """Publish an event to the bus."""
        await self.event_queue.put(event)
        self.stats['events_published'] += 1
        logger.debug(f"Published event {event.event_id} of type {event.event_type.value}")
    
    async def subscribe(
        self, 
        event_types: Set[EventType], 
        handler: Callable[[Event], Any],
        filters: Optional[Dict[str, Any]] = None,
        handler_id: Optional[str] = None,
    ) -> str:
        """Subscribe to specific event types."""
        handler_id = handler_id or str(uuid.uuid4())
        subscription_id = str(uuid.uuid4())
        
        # Create handler
        event_handler = EventHandler(
            handler_id=handler_id,
            event_types=event_types,
            handler_func=handler,
            filters=filters or {},
        )
        self.handlers[handler_id] = event_handler
        
        # Create subscription
        subscription = EventSubscription(
            subscription_id=subscription_id,
            subscriber_id=handler_id,
            event_types=event_types,
            filters=filters or {},
        )
        self.subscriptions[subscription_id] = subscription
        
        self.stats['active_subscriptions'] += 1
        
        logger.info(f"Created subscription {subscription_id} for events {[t.value for t in event_types]}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> None:
        """Remove a subscription."""
        if subscription_id in self.subscriptions:
            subscription = self.subscriptions[subscription_id]
            handler_id = subscription.subscriber_id
            
            # Remove subscription
            del self.subscriptions[subscription_id]
            
            # Remove handler if no more subscriptions
            if handler_id in self.handlers:
                has_other_subscriptions = any(
                    sub.subscriber_id == handler_id 
                    for sub in self.subscriptions.values()
                )
                if not has_other_subscriptions:
                    del self.handlers[handler_id]
            
            self.stats['active_subscriptions'] -= 1
            logger.info(f"Removed subscription {subscription_id}")
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        while True:
            try:
                # Get event from queue
                event = await self.event_queue.get()
                
                # Find matching handlers
                matching_handlers = []
                for handler in self.handlers.values():
                    if self._matches_handler(event, handler):
                        matching_handlers.append(handler)
                
                # Execute handlers concurrently
                if matching_handlers:
                    tasks = [
                        self._execute_handler(handler, event)
                        for handler in matching_handlers
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                self.stats['events_processed'] += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                self.stats['events_failed'] += 1
    
    def _matches_handler(self, event: Event, handler: EventHandler) -> bool:
        """Check if event matches handler criteria."""
        # Check if handler is active
        if not handler.active:
            return False
        
        # Check event type
        if event.event_type not in handler.event_types:
            return False
        
        # Check filters
        for filter_key, filter_value in handler.filters.items():
            if filter_key in event.data:
                if event.data[filter_key] != filter_value:
                    return False
            elif filter_key in event.metadata:
                if event.metadata[filter_key] != filter_value:
                    return False
            else:
                return False
        
        return True
    
    async def _execute_handler(self, handler: EventHandler, event: Event) -> None:
        """Execute an event handler."""
        try:
            if handler.timeout_seconds:
                await asyncio.wait_for(
                    handler.handler_func(event),
                    timeout=handler.timeout_seconds
                )
            else:
                await handler.handler_func(event)
                
            logger.debug(f"Handler {handler.handler_id} processed event {event.event_id}")
            
        except asyncio.TimeoutError:
            logger.warning(f"Handler {handler.handler_id} timed out for event {event.event_id}")
        except Exception as e:
            logger.error(f"Handler {handler.handler_id} failed for event {event.event_id}: {e}")


class RedisEventBus(EventBus):
    """Redis-based distributed event bus."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis event bus."""
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.PubSub] = None
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.handlers: Dict[str, EventHandler] = {}
        self.channel_prefix = "namel3ss_events"
        
        if redis is None:
            raise ImportError("redis package is required for RedisEventBus")
    
    async def connect(self) -> None:
        """Connect to Redis."""
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()
        await self.redis_client.ping()
        logger.info(f"Connected to Redis event bus at {self.redis_url}")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.pubsub:
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Disconnected from Redis event bus")
    
    async def publish(self, event: Event) -> None:
        """Publish event to Redis channel."""
        if not self.redis_client:
            raise RuntimeError("Not connected to Redis")
        
        channel = f"{self.channel_prefix}:{event.event_type.value}"
        event_data = json.dumps(event.to_dict())
        
        await self.redis_client.publish(channel, event_data)
        logger.debug(f"Published event {event.event_id} to Redis channel {channel}")
    
    async def subscribe(
        self, 
        event_types: Set[EventType], 
        handler: Callable[[Event], Any],
        **kwargs
    ) -> str:
        """Subscribe to Redis channels."""
        if not self.pubsub:
            raise RuntimeError("Not connected to Redis")
        
        subscription_id = str(uuid.uuid4())
        handler_id = kwargs.get('handler_id', str(uuid.uuid4()))
        
        # Subscribe to channels
        channels = [f"{self.channel_prefix}:{event_type.value}" for event_type in event_types]
        await self.pubsub.subscribe(*channels)
        
        # Store subscription and handler
        subscription = EventSubscription(
            subscription_id=subscription_id,
            subscriber_id=handler_id,
            event_types=event_types,
            filters=kwargs.get('filters', {}),
        )
        self.subscriptions[subscription_id] = subscription
        
        event_handler = EventHandler(
            handler_id=handler_id,
            event_types=event_types,
            handler_func=handler,
            filters=kwargs.get('filters', {}),
        )
        self.handlers[handler_id] = event_handler
        
        # Start listening task if not already started
        if not hasattr(self, '_listener_task'):
            self._listener_task = asyncio.create_task(self._listen_for_events())
        
        logger.info(f"Subscribed to Redis channels: {channels}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from Redis channels."""
        if subscription_id in self.subscriptions:
            subscription = self.subscriptions[subscription_id]
            
            # Unsubscribe from channels
            channels = [f"{self.channel_prefix}:{event_type.value}" for event_type in subscription.event_types]
            await self.pubsub.unsubscribe(*channels)
            
            # Remove subscription and handler
            del self.subscriptions[subscription_id]
            if subscription.subscriber_id in self.handlers:
                del self.handlers[subscription.subscriber_id]
            
            logger.info(f"Unsubscribed from Redis channels: {channels}")
    
    async def _listen_for_events(self) -> None:
        """Listen for events from Redis."""
        if not self.pubsub:
            return
        
        try:
            async for message in self.pubsub.listen():
                if message['type'] == 'message':
                    try:
                        event_data = json.loads(message['data'])
                        event = Event.from_dict(event_data)
                        
                        # Find matching handlers
                        for handler in self.handlers.values():
                            if event.event_type in handler.event_types:
                                asyncio.create_task(self._execute_handler(handler, event))
                                
                    except Exception as e:
                        logger.error(f"Error processing Redis event: {e}")
        except asyncio.CancelledError:
            pass
    
    async def _execute_handler(self, handler: EventHandler, event: Event) -> None:
        """Execute event handler."""
        try:
            await handler.handler_func(event)
        except Exception as e:
            logger.error(f"Handler {handler.handler_id} failed: {e}")


# =============================================================================
# Event-Driven Execution Runtime
# =============================================================================

class EventDrivenExecutor:
    """
    Event-driven execution engine for Namel3ss.
    
    Features:
    - Event-triggered workflows
    - Reactive execution patterns
    - WebSocket event streaming
    - Integration with parallel/distributed execution
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        enable_websocket: bool = True,
        websocket_host: str = "localhost",
        websocket_port: int = 8765,
    ):
        """Initialize event-driven executor."""
        self.event_bus = event_bus or MemoryEventBus()
        self.enable_websocket = enable_websocket and websockets is not None
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        
        # WebSocket management
        self.websocket_server = None
        self.websocket_clients: Set[WebSocketServerProtocol] = set()
        
        # Event-driven workflows
        self.event_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_instances: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            'workflows_registered': 0,
            'workflows_triggered': 0,
            'events_streamed': 0,
            'websocket_connections': 0,
        }
        
        logger.info(f"EventDrivenExecutor initialized with WebSocket: {self.enable_websocket}")
    
    async def start(self) -> None:
        """Start the event-driven executor."""
        # Start event bus
        if hasattr(self.event_bus, 'start'):
            await self.event_bus.start()
        elif hasattr(self.event_bus, 'connect'):
            await self.event_bus.connect()
        
        # Start WebSocket server
        if self.enable_websocket:
            await self._start_websocket_server()
        
        logger.info("EventDrivenExecutor started")
    
    async def stop(self) -> None:
        """Stop the event-driven executor."""
        # Stop WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        # Stop event bus
        if hasattr(self.event_bus, 'stop'):
            await self.event_bus.stop()
        elif hasattr(self.event_bus, 'disconnect'):
            await self.event_bus.disconnect()
        
        logger.info("EventDrivenExecutor stopped")
    
    async def publish_event(
        self,
        event_type: Union[EventType, str],
        source: str,
        data: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: Optional[str] = None,
    ) -> str:
        """Publish an event."""
        if isinstance(event_type, str):
            try:
                event_type = EventType(event_type)
            except ValueError:
                event_type = EventType.CUSTOM
        
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source=source,
            data=data,
            priority=priority,
            correlation_id=correlation_id,
        )
        
        await self.event_bus.publish(event)
        
        # Stream to WebSocket clients
        if self.enable_websocket:
            await self._stream_event_to_websockets(event)
        
        return event.event_id
    
    async def register_event_handler(
        self,
        event_types: List[Union[EventType, str]],
        handler_func: Callable[[Event], Any],
        filters: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> str:
        """Register an event handler."""
        # Convert string event types to EventType enum
        parsed_event_types = set()
        for event_type in event_types:
            if isinstance(event_type, str):
                try:
                    parsed_event_types.add(EventType(event_type))
                except ValueError:
                    parsed_event_types.add(EventType.CUSTOM)
            else:
                parsed_event_types.add(event_type)
        
        subscription_id = await self.event_bus.subscribe(
            event_types=parsed_event_types,
            handler=handler_func,
            filters=filters,
        )
        
        logger.info(f"Registered event handler for {[t.value for t in parsed_event_types]}")
        return subscription_id
    
    async def register_workflow_trigger(
        self,
        workflow_name: str,
        trigger_event_type: Union[EventType, str],
        workflow_config: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an event-triggered workflow."""
        if isinstance(trigger_event_type, str):
            try:
                trigger_event_type = EventType(trigger_event_type)
            except ValueError:
                trigger_event_type = EventType.CUSTOM
        
        # Store workflow configuration
        self.event_workflows[workflow_name] = {
            'trigger_event_type': trigger_event_type,
            'workflow_config': workflow_config,
            'filters': filters or {},
        }
        
        # Register event handler for workflow trigger
        async def workflow_trigger_handler(event: Event) -> None:
            await self._execute_event_workflow(workflow_name, event)
        
        await self.register_event_handler(
            event_types=[trigger_event_type],
            handler_func=workflow_trigger_handler,
            filters=filters,
        )
        
        self.stats['workflows_registered'] += 1
        logger.info(f"Registered workflow trigger: {workflow_name} -> {trigger_event_type.value}")
    
    async def _execute_event_workflow(self, workflow_name: str, trigger_event: Event) -> None:
        """Execute an event-triggered workflow."""
        if workflow_name not in self.event_workflows:
            logger.error(f"Unknown workflow: {workflow_name}")
            return
        
        workflow_config = self.event_workflows[workflow_name]['workflow_config']
        instance_id = str(uuid.uuid4())
        
        logger.info(f"Executing event workflow: {workflow_name} (instance: {instance_id})")
        
        try:
            # Store workflow instance
            self.workflow_instances[instance_id] = {
                'workflow_name': workflow_name,
                'trigger_event': trigger_event,
                'started_at': time.time(),
                'status': 'running',
            }
            
            # Import here to avoid circular imports
            from .coordinator import get_distributed_executor
            
            # Execute workflow using distributed executor
            executor = await get_distributed_executor()
            
            # Convert workflow config to parallel block
            parallel_block = {
                'name': f"event_workflow_{workflow_name}",
                'steps': workflow_config.get('steps', []),
                'strategy': workflow_config.get('strategy', 'all'),
                'max_concurrency': workflow_config.get('max_concurrency', 5),
                'timeout_seconds': workflow_config.get('timeout_seconds', 300),
            }
            
            # Create workflow context with trigger event data
            context = {
                'trigger_event': trigger_event.to_dict(),
                'workflow_instance_id': instance_id,
                **workflow_config.get('context', {})
            }
            
            # Execute workflow
            result = await executor.execute_parallel_block(
                parallel_block=parallel_block,
                step_executor=self._default_step_executor,
                context=context,
            )
            
            # Update workflow instance
            self.workflow_instances[instance_id].update({
                'status': 'completed' if result.overall_status == 'completed' else 'failed',
                'completed_at': time.time(),
                'result': result,
            })
            
            # Publish workflow completion event
            await self.publish_event(
                event_type=EventType.WORKFLOW_COMPLETED,
                source=f"workflow_{workflow_name}",
                data={
                    'workflow_name': workflow_name,
                    'instance_id': instance_id,
                    'trigger_event_id': trigger_event.event_id,
                    'status': result.overall_status,
                    'duration_ms': result.total_duration_ms,
                },
                correlation_id=trigger_event.correlation_id,
            )
            
            self.stats['workflows_triggered'] += 1
            
            logger.info(f"Completed workflow {workflow_name} (instance: {instance_id})")
            
        except Exception as e:
            # Update workflow instance with error
            self.workflow_instances[instance_id].update({
                'status': 'failed',
                'completed_at': time.time(),
                'error': str(e),
            })
            
            # Publish workflow failure event
            await self.publish_event(
                event_type=EventType.WORKFLOW_COMPLETED,  # Use same event type but with error status
                source=f"workflow_{workflow_name}",
                data={
                    'workflow_name': workflow_name,
                    'instance_id': instance_id,
                    'trigger_event_id': trigger_event.event_id,
                    'status': 'failed',
                    'error': str(e),
                },
                correlation_id=trigger_event.correlation_id,
            )
            
            logger.error(f"Workflow {workflow_name} (instance: {instance_id}) failed: {e}")
    
    async def _default_step_executor(self, step: Any, context: Dict[str, Any]) -> Any:
        """Default step executor for event workflows."""
        # This is a placeholder - in real implementation, you'd integrate
        # with the actual Namel3ss step execution system
        
        if isinstance(step, dict):
            step_type = step.get('type', 'unknown')
            step_data = step.get('data', {})
            
            logger.info(f"Executing step of type: {step_type}")
            
            # Simulate step execution
            await asyncio.sleep(0.1)
            
            return {
                'step_type': step_type,
                'step_data': step_data,
                'context_keys': list(context.keys()),
                'executed_at': time.time(),
            }
        else:
            # Handle other step types
            return {'step': str(step), 'executed_at': time.time()}
    
    async def _start_websocket_server(self) -> None:
        """Start WebSocket server for event streaming."""
        if not websockets:
            logger.warning("WebSocket support not available")
            return
        
        async def handle_websocket(websocket: WebSocketServerProtocol, path: str) -> None:
            """Handle WebSocket connection."""
            self.websocket_clients.add(websocket)
            self.stats['websocket_connections'] += 1
            
            logger.info(f"WebSocket client connected: {websocket.remote_address}")
            
            try:
                # Send welcome message
                welcome_msg = {
                    'type': 'welcome',
                    'message': 'Connected to Namel3ss Event Stream',
                    'timestamp': time.time(),
                }
                await websocket.send(json.dumps(welcome_msg))
                
                # Keep connection alive
                await websocket.wait_closed()
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_clients.discard(websocket)
                self.stats['websocket_connections'] -= 1
                logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
        
        self.websocket_server = await websockets.serve(
            handle_websocket,
            self.websocket_host,
            self.websocket_port
        )
        
        logger.info(f"WebSocket server started on {self.websocket_host}:{self.websocket_port}")
    
    async def _stream_event_to_websockets(self, event: Event) -> None:
        """Stream event to all connected WebSocket clients."""
        if not self.websocket_clients:
            return
        
        event_message = {
            'type': 'event',
            'event': event.to_dict(),
        }
        
        # Send to all connected clients
        disconnected_clients = set()
        for client in self.websocket_clients:
            try:
                await client.send(json.dumps(event_message))
                self.stats['events_streamed'] += 1
            except Exception as e:
                logger.warning(f"Failed to send event to WebSocket client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event system statistics."""
        bus_stats = {}
        if hasattr(self.event_bus, 'stats'):
            bus_stats = self.event_bus.stats
        
        return {
            'event_driven_executor': self.stats.copy(),
            'event_bus': bus_stats,
            'workflow_instances': len(self.workflow_instances),
            'registered_workflows': len(self.event_workflows),
        }


# =============================================================================
# Convenience Functions and Global Instance
# =============================================================================

# Global event-driven executor instance
_global_event_executor: Optional[EventDrivenExecutor] = None


async def get_event_executor() -> EventDrivenExecutor:
    """Get global event-driven executor instance."""
    global _global_event_executor
    
    if _global_event_executor is None:
        _global_event_executor = EventDrivenExecutor()
        await _global_event_executor.start()
    
    return _global_event_executor


async def publish_event(
    event_type: Union[EventType, str],
    source: str,
    data: Dict[str, Any],
    **kwargs
) -> str:
    """Convenience function to publish an event."""
    executor = await get_event_executor()
    return await executor.publish_event(event_type, source, data, **kwargs)


async def register_event_workflow(
    workflow_name: str,
    trigger_event_type: Union[EventType, str],
    workflow_config: Dict[str, Any],
    **kwargs
) -> None:
    """Convenience function to register an event-triggered workflow."""
    executor = await get_event_executor()
    await executor.register_workflow_trigger(
        workflow_name, trigger_event_type, workflow_config, **kwargs
    )


# Example usage and patterns
async def create_reactive_workflow_example():
    """Example of creating a reactive workflow."""
    executor = await get_event_executor()
    
    # Register a workflow that responds to task completion
    await executor.register_workflow_trigger(
        workflow_name="post_processing_pipeline",
        trigger_event_type=EventType.TASK_COMPLETED,
        workflow_config={
            'steps': [
                {'type': 'validate_result', 'data': {}},
                {'type': 'store_result', 'data': {}},
                {'type': 'notify_completion', 'data': {}},
            ],
            'strategy': 'all',
            'max_concurrency': 3,
        },
        filters={'task_type': 'analysis'}  # Only trigger for analysis tasks
    )
    
    # Simulate task completion
    await executor.publish_event(
        event_type=EventType.TASK_COMPLETED,
        source="analysis_engine",
        data={
            'task_type': 'analysis',
            'task_id': 'analysis_123',
            'result': {'status': 'success', 'data': 'processed_output'},
        }
    )
    
    logger.info("Reactive workflow example created and triggered")