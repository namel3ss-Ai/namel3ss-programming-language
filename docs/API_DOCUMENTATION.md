# Namel3ss Parallel & Distributed Execution API Documentation

## 📚 Complete API Reference for Parallel and Distributed Execution

This comprehensive API documentation covers all components of the Namel3ss parallel and distributed execution system.

## 📋 Table of Contents

1. [Core API Overview](#core-api-overview)
2. [Parallel Execution API](#parallel-execution-api)
3. [Distributed Execution API](#distributed-execution-api)
4. [Event-Driven Execution API](#event-driven-execution-api)
5. [Security API](#security-api)
6. [Observability API](#observability-api)
7. [Configuration API](#configuration-api)
8. [Error Handling](#error-handling)
9. [Examples & Usage Patterns](#examples--usage-patterns)
10. [SDK & Client Libraries](#sdk--client-libraries)

---

## Core API Overview

### System Architecture

```python
from namel3ss.runtime.parallel import ParallelExecutor
from namel3ss.runtime.distributed import DistributedTaskQueue
from namel3ss.runtime.coordinator import DistributedParallelExecutor
from namel3ss.runtime.events import EventDrivenExecutor
from namel3ss.runtime.security import SecurityManager
from namel3ss.runtime.observability import ObservabilityManager
```

### Base Classes and Interfaces

#### ExecutorInterface
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class ExecutorInterface(ABC):
    """Base interface for all execution engines."""
    
    @abstractmethod
    async def execute(self, task: Any, context: Optional[Dict] = None) -> Any:
        """Execute a task with optional context."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        pass
```

---

## Parallel Execution API

### ParallelExecutor

The `ParallelExecutor` provides high-performance parallel execution with multiple strategies.

#### Class Definition
```python
class ParallelExecutor:
    def __init__(
        self,
        default_max_concurrency: int = 10,
        enable_security: bool = False,
        security_manager: Optional[SecurityManager] = None,
        enable_observability: bool = False,
        observability_manager: Optional[ObservabilityManager] = None
    ):
        """
        Initialize parallel executor.
        
        Args:
            default_max_concurrency: Default maximum concurrent tasks
            enable_security: Enable security validation
            security_manager: Security manager instance
            enable_observability: Enable metrics and tracing
            observability_manager: Observability manager instance
        """
```

#### Core Methods

##### execute_parallel_block
```python
async def execute_parallel_block(
    self,
    parallel_block: Dict[str, Any],
    step_executor: Callable,
    security_context: Optional[SecurityContext] = None
) -> ParallelExecutionResult:
    """
    Execute parallel block with specified strategy.
    
    Args:
        parallel_block: Parallel execution configuration
        step_executor: Function to execute individual steps
        security_context: Optional security context
        
    Returns:
        ParallelExecutionResult with execution details
        
    Example:
        ```python
        parallel_block = {
            'name': 'data_processing',
            'strategy': 'all',
            'steps': ['step1', 'step2', 'step3'],
            'max_concurrency': 5,
            'timeout_seconds': 30.0
        }
        
        async def step_executor(step, context=None):
            # Your step implementation
            return f"result_for_{step}"
        
        result = await executor.execute_parallel_block(
            parallel_block, 
            step_executor
        )
        ```
    """
```

##### execute_parallel_steps
```python
async def execute_parallel_steps(
    self,
    steps: List[Any],
    step_executor: Callable,
    strategy: ParallelStrategy = ParallelStrategy.ALL,
    max_concurrency: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
    security_context: Optional[SecurityContext] = None
) -> List[ParallelTaskResult]:
    """
    Execute steps in parallel with specified strategy.
    
    Args:
        steps: List of steps to execute
        step_executor: Function to execute each step
        strategy: Execution strategy (ALL, ANY_SUCCESS, RACE, etc.)
        max_concurrency: Maximum concurrent tasks
        timeout_seconds: Timeout for execution
        security_context: Optional security context
        
    Returns:
        List of ParallelTaskResult objects
    """
```

#### Parallel Strategies

```python
from enum import Enum

class ParallelStrategy(Enum):
    """Parallel execution strategies."""
    ALL = "all"                    # Wait for all tasks to complete
    ANY_SUCCESS = "any_success"    # Return on first success
    RACE = "race"                  # Return on first completion
    THROTTLED = "throttled"        # Throttled execution
    CUSTOM = "custom"              # Custom strategy
```

#### Result Classes

##### ParallelExecutionResult
```python
@dataclass
class ParallelExecutionResult:
    """Result of parallel block execution."""
    block_name: str
    strategy: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    cancelled_tasks: int
    overall_status: str
    execution_time: float
    results: List[ParallelTaskResult]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

##### ParallelTaskResult
```python
@dataclass
class ParallelTaskResult:
    """Result of individual parallel task."""
    task_id: str
    status: str  # "completed", "failed", "cancelled"
    result: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## Distributed Execution API

### DistributedTaskQueue

The `DistributedTaskQueue` manages distributed task execution across multiple workers.

#### Class Definition
```python
class DistributedTaskQueue:
    def __init__(
        self,
        broker: Optional[MessageBroker] = None,
        enable_security: bool = False,
        security_manager: Optional[SecurityManager] = None,
        worker_timeout: float = 300.0,
        max_retries: int = 3,
        enable_auto_scaling: bool = True
    ):
        """
        Initialize distributed task queue.
        
        Args:
            broker: Message broker for task distribution
            enable_security: Enable security validation
            security_manager: Security manager instance
            worker_timeout: Timeout for worker tasks
            max_retries: Maximum retry attempts
            enable_auto_scaling: Enable automatic worker scaling
        """
```

#### Core Methods

##### submit_task
```python
async def submit_task(
    self,
    task_type: str,
    payload: Dict[str, Any],
    priority: int = 1,
    timeout: Optional[float] = None,
    security_context: Optional[SecurityContext] = None
) -> str:
    """
    Submit task for distributed execution.
    
    Args:
        task_type: Type of task to execute
        payload: Task payload/parameters
        priority: Task priority (1=high, 5=low)
        timeout: Task timeout in seconds
        security_context: Security context for task
        
    Returns:
        Task ID for tracking
        
    Example:
        ```python
        task_id = await queue.submit_task(
            task_type="data_analysis",
            payload={"dataset": "large_dataset.csv"},
            priority=2,
            timeout=600.0
        )
        ```
    """
```

##### get_task_status
```python
async def get_task_status(self, task_id: str) -> Dict[str, Any]:
    """
    Get current status of submitted task.
    
    Args:
        task_id: ID of task to check
        
    Returns:
        Task status information
        
    Example:
        ```python
        status = await queue.get_task_status(task_id)
        print(f"Status: {status['state']}")
        print(f"Progress: {status['progress']}%")
        ```
    """
```

##### get_task_result
```python
async def get_task_result(
    self, 
    task_id: str, 
    timeout: Optional[float] = None
) -> Any:
    """
    Get result of completed task.
    
    Args:
        task_id: ID of task
        timeout: Wait timeout in seconds
        
    Returns:
        Task result
        
    Raises:
        TaskNotFoundError: Task doesn't exist
        TaskTimeoutError: Task timed out
        TaskFailedError: Task execution failed
    """
```

##### register_worker
```python
async def register_worker(
    self,
    worker_id: str,
    capabilities: List[str],
    max_concurrent_tasks: int = 5,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Register worker with the distributed queue.
    
    Args:
        worker_id: Unique worker identifier
        capabilities: List of task types worker can handle
        max_concurrent_tasks: Maximum concurrent tasks
        metadata: Additional worker metadata
        
    Returns:
        True if registration successful
    """
```

### Message Brokers

#### MemoryMessageBroker
```python
class MemoryMessageBroker(MessageBroker):
    """In-memory message broker for development/testing."""
    
    def __init__(self):
        """Initialize in-memory broker."""
        pass
```

#### RedisMessageBroker
```python
class RedisMessageBroker(MessageBroker):
    """Redis-based message broker for production."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        connection_pool_size: int = 20,
        socket_timeout: float = 5.0
    ):
        """
        Initialize Redis message broker.
        
        Args:
            redis_url: Redis connection URL
            connection_pool_size: Connection pool size
            socket_timeout: Socket timeout in seconds
        """
```

---

## Event-Driven Execution API

### EventDrivenExecutor

The `EventDrivenExecutor` provides reactive, event-driven execution patterns.

#### Class Definition
```python
class EventDrivenExecutor:
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        enable_security: bool = False,
        security_manager: Optional[SecurityManager] = None,
        max_concurrent_handlers: int = 100
    ):
        """
        Initialize event-driven executor.
        
        Args:
            event_bus: Event bus for event routing
            enable_security: Enable security validation
            security_manager: Security manager instance
            max_concurrent_handlers: Maximum concurrent event handlers
        """
```

#### Core Methods

##### register_event_handler
```python
async def register_event_handler(
    self,
    event_type: str,
    handler: Callable,
    filter_conditions: Optional[Dict[str, Any]] = None,
    priority: int = 1
) -> str:
    """
    Register event handler for specific event type.
    
    Args:
        event_type: Type of events to handle
        handler: Async function to handle events
        filter_conditions: Optional event filtering
        priority: Handler priority (1=high, 5=low)
        
    Returns:
        Handler ID for management
        
    Example:
        ```python
        async def task_complete_handler(event):
            print(f"Task {event.data['task_id']} completed")
            
        handler_id = await executor.register_event_handler(
            "task_completed",
            task_complete_handler,
            filter_conditions={"priority": "high"}
        )
        ```
    """
```

##### trigger_event
```python
async def trigger_event(
    self,
    event_type: str,
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Trigger event in the system.
    
    Args:
        event_type: Type of event
        data: Event data
        metadata: Optional metadata
        
    Returns:
        Event ID
        
    Example:
        ```python
        event_id = await executor.trigger_event(
            "user_action",
            {"action": "login", "user_id": "123"},
            {"source": "web_app", "timestamp": time.time()}
        )
        ```
    """
```

##### start_websocket_server
```python
async def start_websocket_server(
    self,
    host: str = "localhost",
    port: int = 8080,
    max_connections: int = 1000
) -> None:
    """
    Start WebSocket server for real-time event streaming.
    
    Args:
        host: Server host
        port: Server port
        max_connections: Maximum WebSocket connections
        
    Example:
        ```python
        await executor.start_websocket_server(
            host="0.0.0.0",
            port=8080,
            max_connections=5000
        )
        ```
    """
```

### Event System Classes

#### Event
```python
@dataclass
class Event:
    """Event data structure."""
    event_id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
```

#### EventBus Interface
```python
class EventBus(ABC):
    """Abstract event bus interface."""
    
    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish event to bus."""
        pass
    
    @abstractmethod
    async def subscribe(
        self, 
        event_type: str, 
        handler: Callable
    ) -> str:
        """Subscribe to event type."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        pass
```

---

## Security API

### SecurityManager

The `SecurityManager` provides comprehensive security controls for all execution engines.

#### Class Definition
```python
class SecurityManager:
    def __init__(
        self,
        audit_enabled: bool = True,
        audit_log_path: Optional[str] = None,
        session_timeout: float = 3600.0,
        max_concurrent_sessions: int = 100
    ):
        """
        Initialize security manager.
        
        Args:
            audit_enabled: Enable audit logging
            audit_log_path: Path to audit log file
            session_timeout: Session timeout in seconds
            max_concurrent_sessions: Maximum concurrent sessions
        """
```

#### Core Methods

##### create_security_context
```python
async def create_security_context(
    self,
    user_id: str,
    permission_level: PermissionLevel,
    capabilities: List[str],
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> SecurityContext:
    """
    Create security context for user session.
    
    Args:
        user_id: Unique user identifier
        permission_level: Permission level (READ_ONLY, READ_WRITE, ADMIN)
        capabilities: List of capability names
        session_id: Optional session identifier
        metadata: Additional context metadata
        
    Returns:
        SecurityContext object
        
    Example:
        ```python
        context = await security_manager.create_security_context(
            user_id="user123",
            permission_level=PermissionLevel.READ_WRITE,
            capabilities=["read_data", "execute_basic", "write_results"]
        )
        ```
    """
```

##### validate_action
```python
async def validate_action(
    self,
    security_context: SecurityContext,
    action: SecurityAction,
    resource_type: ResourceType,
    resource_id: Optional[str] = None
) -> bool:
    """
    Validate if action is permitted for context.
    
    Args:
        security_context: Security context to validate
        action: Action being attempted
        resource_type: Type of resource being accessed
        resource_id: Specific resource identifier
        
    Returns:
        True if action is permitted
        
    Raises:
        PermissionDeniedError: If action not permitted
    """
```

##### get_audit_trail
```python
async def get_audit_trail(
    self,
    user_id: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    limit: int = 1000
) -> List[AuditEvent]:
    """
    Retrieve audit trail for user or time period.
    
    Args:
        user_id: Filter by user ID
        start_time: Start timestamp
        end_time: End timestamp
        limit: Maximum number of events
        
    Returns:
        List of AuditEvent objects
    """
```

### Security Classes

#### SecurityContext
```python
@dataclass
class SecurityContext:
    """Security context for authenticated sessions."""
    user_id: str
    session_id: str
    permission_level: PermissionLevel
    capabilities: Set[Capability]
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if context has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def has_capability(self, capability_name: str) -> bool:
        """Check if context has specific capability."""
        return any(cap.name == capability_name for cap in self.capabilities)
```

#### Capability
```python
@dataclass(frozen=True)
class Capability:
    """Security capability definition."""
    name: str
    description: str
    actions: FrozenSet[SecurityAction]
    resource_types: FrozenSet[ResourceType]
    
    def allows_action(
        self, 
        action: SecurityAction, 
        resource_type: ResourceType
    ) -> bool:
        """Check if capability allows specific action on resource type."""
        return action in self.actions and resource_type in self.resource_types
```

#### Permission Levels and Enums
```python
class PermissionLevel(Enum):
    """Permission levels for users."""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"

class SecurityAction(Enum):
    """Types of security actions."""
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    EXECUTE_TASK = "execute_task"
    MANAGE_WORKERS = "manage_workers"
    ADMIN_OPERATION = "admin_operation"

class ResourceType(Enum):
    """Types of system resources."""
    DATA = "data"
    COMPUTE = "compute"
    NETWORK = "network"
    STORAGE = "storage"
    SYSTEM = "system"
```

---

## Observability API

### ObservabilityManager

The `ObservabilityManager` provides comprehensive monitoring, metrics, and tracing.

#### Class Definition
```python
class ObservabilityManager:
    def __init__(
        self,
        service_name: str = "namel3ss",
        enable_metrics: bool = True,
        enable_tracing: bool = True,
        enable_health_monitoring: bool = True,
        metrics_port: int = 9090,
        jaeger_endpoint: Optional[str] = None
    ):
        """
        Initialize observability manager.
        
        Args:
            service_name: Service name for telemetry
            enable_metrics: Enable Prometheus metrics
            enable_tracing: Enable distributed tracing
            enable_health_monitoring: Enable health checks
            metrics_port: Port for metrics endpoint
            jaeger_endpoint: Jaeger collector endpoint
        """
```

#### Core Methods

##### record_metric
```python
async def record_metric(
    self,
    metric_name: str,
    value: float,
    labels: Optional[Dict[str, str]] = None,
    metric_type: MetricType = MetricType.COUNTER
) -> None:
    """
    Record metric value.
    
    Args:
        metric_name: Name of metric
        value: Metric value
        labels: Optional metric labels
        metric_type: Type of metric (COUNTER, GAUGE, HISTOGRAM)
        
    Example:
        ```python
        await observability.record_metric(
            "tasks_completed",
            1.0,
            labels={"strategy": "all", "status": "success"},
            metric_type=MetricType.COUNTER
        )
        ```
    """
```

##### start_trace
```python
@asynccontextmanager
async def start_trace(
    self,
    operation_name: str,
    tags: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Start distributed trace for operation.
    
    Args:
        operation_name: Name of operation being traced
        tags: Optional trace tags
        
    Example:
        ```python
        async with observability.start_trace("parallel_execution") as span:
            span.set_tag("strategy", "all")
            span.set_tag("task_count", len(tasks))
            # Perform traced operation
            result = await execute_parallel_tasks(tasks)
            span.set_tag("result_count", len(result))
        ```
    """
```

##### get_metrics_summary
```python
async def get_metrics_summary(self) -> Dict[str, Any]:
    """
    Get summary of current metrics.
    
    Returns:
        Dictionary containing metric summaries
        
    Example:
        ```python
        metrics = await observability.get_metrics_summary()
        print(f"Total requests: {metrics['requests_total']}")
        print(f"Error rate: {metrics['error_rate']}%")
        print(f"Average response time: {metrics['avg_response_time']}ms")
        ```
    """
```

### Health Monitoring

#### HealthMonitor
```python
class HealthMonitor:
    """Health monitoring and checks."""
    
    async def add_health_check(
        self,
        name: str,
        check_function: Callable[[], bool],
        timeout_seconds: float = 5.0,
        critical: bool = True
    ) -> None:
        """
        Add health check to monitoring.
        
        Args:
            name: Health check name
            check_function: Function that returns health status
            timeout_seconds: Check timeout
            critical: Whether failure is critical
        """
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        pass
```

---

## Configuration API

### Configuration Management

#### ConfigurationManager
```python
class ConfigurationManager:
    """Centralized configuration management."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        pass
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        pass
    
    async def reload_config(self) -> None:
        """Reload configuration from file."""
        pass
```

#### Environment-specific Configurations

```python
# Development configuration
development_config = {
    "parallel": {
        "default_max_concurrency": 5,
        "enable_security": False,
        "enable_observability": True
    },
    "distributed": {
        "enable": False,
        "message_broker": {
            "type": "memory"
        }
    }
}

# Production configuration  
production_config = {
    "parallel": {
        "default_max_concurrency": 50,
        "enable_security": True,
        "enable_observability": True
    },
    "distributed": {
        "enable": True,
        "message_broker": {
            "type": "redis",
            "url": "redis://redis-cluster:6379/0"
        }
    },
    "security": {
        "enable_audit": True,
        "session_timeout": 3600,
        "tls": {
            "enable": True,
            "cert_path": "/etc/ssl/certs/namel3ss.crt"
        }
    }
}
```

---

## Error Handling

### Exception Hierarchy

```python
class Namel3ssError(Exception):
    """Base exception for Namel3ss runtime."""
    pass

class ExecutionError(Namel3ssError):
    """Base execution error."""
    pass

class ParallelExecutionError(ExecutionError):
    """Parallel execution error."""
    pass

class DistributedExecutionError(ExecutionError):
    """Distributed execution error."""
    pass

class SecurityError(Namel3ssError):
    """Security-related error."""
    pass

class PermissionDeniedError(SecurityError):
    """Permission denied error."""
    pass

class ObservabilityError(Namel3ssError):
    """Observability system error."""
    pass

# Task-specific errors
class TaskTimeoutError(ExecutionError):
    """Task execution timeout."""
    pass

class TaskNotFoundError(ExecutionError):
    """Task not found."""
    pass

class TaskFailedError(ExecutionError):
    """Task execution failed."""
    pass

class WorkerNotAvailableError(DistributedExecutionError):
    """No workers available for task."""
    pass
```

### Error Response Format

```python
@dataclass
class ErrorResponse:
    """Standardized error response."""
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error": {
                "code": self.error_code,
                "message": self.error_message,
                "details": self.details,
                "timestamp": self.timestamp,
                "request_id": self.request_id
            }
        }
```

---

## Examples & Usage Patterns

### Basic Parallel Execution

```python
import asyncio
from namel3ss.runtime.parallel import ParallelExecutor, ParallelStrategy

async def basic_parallel_example():
    # Initialize executor
    executor = ParallelExecutor(default_max_concurrency=10)
    
    # Define step executor
    async def process_item(item, context=None):
        # Simulate processing
        await asyncio.sleep(0.1)
        return f"processed_{item}"
    
    # Execute parallel block
    parallel_block = {
        'name': 'data_processing',
        'strategy': 'all',
        'steps': ['item1', 'item2', 'item3', 'item4', 'item5'],
        'max_concurrency': 3
    }
    
    result = await executor.execute_parallel_block(
        parallel_block, 
        process_item
    )
    
    print(f"Completed {result.completed_tasks} tasks")
    print(f"Results: {[r.result for r in result.results]}")

# Run example
asyncio.run(basic_parallel_example())
```

### Distributed Task Processing

```python
import asyncio
from namel3ss.runtime.distributed import DistributedTaskQueue, RedisMessageBroker

async def distributed_example():
    # Setup message broker
    broker = RedisMessageBroker("redis://localhost:6379/0")
    
    # Initialize distributed queue
    queue = DistributedTaskQueue(broker=broker)
    await queue.start()
    
    try:
        # Submit tasks
        task_ids = []
        for i in range(10):
            task_id = await queue.submit_task(
                task_type="data_analysis",
                payload={"batch_id": i, "data_size": 1000},
                priority=1
            )
            task_ids.append(task_id)
            print(f"Submitted task: {task_id}")
        
        # Wait for results
        results = []
        for task_id in task_ids:
            result = await queue.get_task_result(task_id, timeout=300.0)
            results.append(result)
            print(f"Task {task_id} completed: {result}")
            
    finally:
        await queue.stop()

asyncio.run(distributed_example())
```

### Event-Driven Processing

```python
import asyncio
from namel3ss.runtime.events import EventDrivenExecutor, MemoryEventBus

async def event_driven_example():
    # Initialize event system
    event_bus = MemoryEventBus()
    executor = EventDrivenExecutor(event_bus=event_bus)
    
    # Register event handlers
    async def task_completed_handler(event):
        print(f"Task completed: {event.data}")
        
        # Trigger follow-up action
        await executor.trigger_event(
            "task_processed",
            {"task_id": event.data["task_id"], "status": "processed"}
        )
    
    async def task_processed_handler(event):
        print(f"Task processed: {event.data}")
    
    # Register handlers
    await executor.register_event_handler(
        "task_completed", 
        task_completed_handler
    )
    await executor.register_event_handler(
        "task_processed", 
        task_processed_handler
    )
    
    # Trigger events
    await executor.trigger_event(
        "task_completed",
        {"task_id": "task_001", "result": "success"}
    )
    
    # Allow events to process
    await asyncio.sleep(1)

asyncio.run(event_driven_example())
```

### Security-Enabled Execution

```python
import asyncio
from namel3ss.runtime.parallel import ParallelExecutor
from namel3ss.runtime.security import SecurityManager, PermissionLevel

async def secure_execution_example():
    # Initialize security manager
    security_manager = SecurityManager(audit_enabled=True)
    
    # Create security context
    security_context = await security_manager.create_security_context(
        user_id="user123",
        permission_level=PermissionLevel.READ_WRITE,
        capabilities=["read_data", "execute_basic", "write_results"]
    )
    
    # Initialize executor with security
    executor = ParallelExecutor(
        enable_security=True,
        security_manager=security_manager
    )
    
    # Define secure step executor
    async def secure_process(item, context=None):
        # Security context is automatically validated
        security_ctx = context.get('security_context')
        print(f"Processing {item} for user {security_ctx.user_id}")
        return f"secure_processed_{item}"
    
    # Execute with security context
    parallel_block = {
        'name': 'secure_processing',
        'strategy': 'all',
        'steps': ['data1', 'data2', 'data3']
    }
    
    result = await executor.execute_parallel_block(
        parallel_block,
        secure_process,
        security_context=security_context
    )
    
    # Check audit trail
    audit_trail = security_manager.get_audit_trail(user_id="user123")
    print(f"Audit events: {len(audit_trail)}")

asyncio.run(secure_execution_example())
```

### Complete Integration Example

```python
import asyncio
from namel3ss.runtime.coordinator import DistributedParallelExecutor
from namel3ss.runtime.parallel import ParallelExecutor
from namel3ss.runtime.distributed import DistributedTaskQueue, MemoryMessageBroker
from namel3ss.runtime.security import SecurityManager, PermissionLevel
from namel3ss.runtime.observability import ObservabilityManager

async def complete_integration_example():
    # Initialize all components
    security_manager = SecurityManager(audit_enabled=True)
    observability = ObservabilityManager(
        service_name="integration_example",
        enable_metrics=True,
        enable_tracing=True
    )
    
    broker = MemoryMessageBroker()
    distributed_queue = DistributedTaskQueue(
        broker=broker,
        enable_security=True,
        security_manager=security_manager
    )
    
    parallel_executor = ParallelExecutor(
        enable_security=True,
        security_manager=security_manager,
        enable_observability=True,
        observability_manager=observability
    )
    
    # Initialize coordinator
    coordinator = DistributedParallelExecutor(
        parallel_executor=parallel_executor,
        distributed_queue=distributed_queue,
        enable_security=True,
        security_manager=security_manager
    )
    
    # Create security context
    security_context = await security_manager.create_security_context(
        user_id="admin_user",
        permission_level=PermissionLevel.ADMIN,
        capabilities=["read_data", "execute_basic", "execute_advanced", "admin_operations"]
    )
    
    await distributed_queue.start()
    
    try:
        # Define processing function
        async def advanced_processor(item, context=None):
            # Record metrics
            await observability.record_metric(
                "items_processed", 1.0, {"type": "advanced"}
            )
            
            # Simulate complex processing
            async with observability.start_trace("item_processing") as span:
                span.set_tag("item", item)
                await asyncio.sleep(0.1)  # Simulate work
                result = f"advanced_processed_{item}"
                span.set_tag("result", result)
                return result
        
        # Execute distributed parallel processing
        parallel_block = {
            'name': 'advanced_integration',
            'strategy': 'all',
            'steps': [f'item_{i}' for i in range(20)],
            'max_concurrency': 5,
            'distribution_policy': 'local_first'
        }
        
        result = await coordinator.execute_distributed_parallel(
            parallel_block,
            advanced_processor,
            security_context=security_context
        )
        
        print(f"Integration example completed: {result}")
        
        # Get metrics summary
        metrics = await observability.get_metrics_summary()
        print(f"Metrics summary: {metrics}")
        
        # Check audit trail
        audit_events = security_manager.get_audit_trail(user_id="admin_user")
        print(f"Security events logged: {len(audit_events)}")
        
    finally:
        await distributed_queue.stop()

asyncio.run(complete_integration_example())
```

---

## SDK & Client Libraries

### Python SDK

The primary SDK is the Python library itself. Install with:

```bash
pip install namel3ss[production]
```

### REST API Client

```python
import aiohttp
from typing import Any, Dict, Optional

class Namel3ssClient:
    """REST API client for Namel3ss."""
    
    def __init__(
        self, 
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
    
    async def submit_parallel_task(
        self, 
        parallel_block: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Submit parallel execution task."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(
                f"{self.base_url}/api/v1/parallel/execute",
                json=parallel_block,
                headers=headers
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task execution status."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(
                f"{self.base_url}/api/v1/tasks/{task_id}/status",
                headers=headers
            ) as response:
                response.raise_for_status()
                return await response.json()
```

### JavaScript/TypeScript SDK

```typescript
interface ParallelBlock {
    name: string;
    strategy: 'all' | 'any_success' | 'race' | 'throttled';
    steps: any[];
    max_concurrency?: number;
    timeout_seconds?: number;
}

interface TaskResult {
    task_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    result?: any;
    error?: string;
    execution_time?: number;
}

class Namel3ssClient {
    private baseUrl: string;
    private apiKey?: string;
    
    constructor(baseUrl: string, apiKey?: string) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.apiKey = apiKey;
    }
    
    async submitParallelTask(parallelBlock: ParallelBlock): Promise<TaskResult> {
        const headers: Record<string, string> = {
            'Content-Type': 'application/json'
        };
        
        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }
        
        const response = await fetch(
            `${this.baseUrl}/api/v1/parallel/execute`,
            {
                method: 'POST',
                headers,
                body: JSON.stringify(parallelBlock)
            }
        );
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    async getTaskStatus(taskId: string): Promise<TaskResult> {
        const headers: Record<string, string> = {};
        
        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }
        
        const response = await fetch(
            `${this.baseUrl}/api/v1/tasks/${taskId}/status`,
            { headers }
        );
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
}
```

---

## Summary

This comprehensive API documentation provides:

✅ **Complete API Reference** - All classes, methods, and interfaces  
✅ **Detailed Examples** - Practical usage patterns and integration examples  
✅ **Error Handling** - Exception hierarchy and error response formats  
✅ **Security Integration** - Authentication, authorization, and audit capabilities  
✅ **Observability Features** - Metrics, tracing, and health monitoring APIs  
✅ **Client Libraries** - Python, JavaScript/TypeScript SDK examples  
✅ **Configuration Management** - Environment-specific configuration patterns  

The API is designed for:
- **Enterprise Integration** - Production-ready with security and monitoring
- **Developer Experience** - Clear, consistent, and well-documented interfaces  
- **Scalability** - From single-machine to distributed cluster deployments
- **Flexibility** - Modular design allows selective feature usage

For implementation guides and deployment instructions, see the [Production Deployment Guide](PRODUCTION_DEPLOYMENT_GUIDE.md) and [Best Practices Guide](BEST_PRACTICES.md).