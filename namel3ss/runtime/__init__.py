"""
Namel3ss Runtime Module.

This module provides the core runtime components for executing Namel3ss programs,
including parallel execution, distributed computing, and event-driven patterns.

Components:
- parallel: Parallel execution engine with multiple strategies
- distributed: Distributed task queues and worker management
- coordinator: Distributed parallel execution coordinator
- events: Event-driven runtime and reactive workflows
"""

# Core parallel execution
from .parallel import (
    ParallelExecutor,
    ParallelStrategy,
    ParallelExecutionResult,
    ParallelExecutionContext,
    ParallelTaskResult,
    execute_parallel_steps,
    get_current_parallel_context,
)

# Distributed execution
from .distributed import (
    DistributedTaskQueue,
    DistributedTask,
    TaskStatus,
    WorkerNode,
    WorkerStatus,
    BrokerType,
    MessageBroker,
    RedisMessageBroker,
    RabbitMQMessageBroker,
    MemoryMessageBroker,
    create_message_broker,
    create_distributed_queue,
)

# Distributed coordination
from .coordinator import (
    DistributedParallelExecutor,
    create_distributed_executor,
    get_distributed_executor,
    execute_distributed_parallel,
)

# Event-driven runtime
from .events import (
    EventDrivenExecutor,
    Event,
    EventType,
    EventPriority,
    EventHandler,
    EventSubscription,
    EventBus,
    MemoryEventBus,
    RedisEventBus,
    get_event_executor,
    publish_event,
    register_event_workflow,
)

__all__ = [
    # Parallel execution
    'ParallelExecutor',
    'ParallelStrategy',
    'ParallelExecutionResult',
    'ParallelExecutionContext',
    'ParallelTaskResult',
    'execute_parallel_steps',
    'get_current_parallel_context',
    
    # Distributed execution
    'DistributedTaskQueue',
    'DistributedTask',
    'TaskStatus',
    'WorkerNode',
    'WorkerStatus',
    'BrokerType',
    'MessageBroker',
    'RedisMessageBroker',
    'RabbitMQMessageBroker',
    'MemoryMessageBroker',
    'create_message_broker',
    'create_distributed_queue',
    
    # Distributed coordination
    'DistributedParallelExecutor',
    'create_distributed_executor',
    'get_distributed_executor',
    'execute_distributed_parallel',
    
    # Event-driven runtime
    'EventDrivenExecutor',
    'Event',
    'EventType',
    'EventPriority',
    'EventHandler',
    'EventSubscription',
    'EventBus',
    'MemoryEventBus',
    'RedisEventBus',
    'get_event_executor',
    'publish_event',
    'register_event_workflow',
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "Namel3ss Team"
__description__ = "Production-grade parallel and distributed execution runtime for Namel3ss"