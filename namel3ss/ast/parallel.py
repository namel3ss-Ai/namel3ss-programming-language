"""
Parallel and distributed execution AST nodes for Namel3ss.

This module defines AST nodes for expressing parallelism and distribution
in the Namel3ss language:
- ParallelBlock: Groups of steps that execute concurrently
- DistributedExecution: Remote execution on worker pools
- EventTrigger: Event-driven workflow triggers
- WorkerPool: Worker pool definitions
- TaskQueue: Queue configurations

These constructs integrate with existing workflow infrastructure while
maintaining strict backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from .base import Expression, SourceLocation
from .security import PermissionLevel, CapabilityType


# =============================================================================
# Execution Strategies and Policies
# =============================================================================

class ParallelStrategy(Enum):
    """Strategy for aggregating results from parallel execution."""
    
    ALL = "all"               # Wait for all tasks, fail if any fails
    ANY_SUCCESS = "any_success"  # Return on first success, cancel others
    RACE = "race"            # Return first completion (success or failure)
    MAP_REDUCE = "map_reduce"  # Apply reduce function to all results
    COLLECT = "collect"       # Collect all results, continue on failures


class DistributionPolicy(Enum):
    """Policy for distributing tasks across workers."""
    
    ROUND_ROBIN = "round_robin"      # Distribute evenly across workers
    LOAD_BASED = "load_based"        # Send to least loaded worker
    LOCALITY_AWARE = "locality_aware" # Prefer local/nearby workers
    STICKY = "sticky"                # Keep related tasks on same worker
    RANDOM = "random"                # Random distribution


class QueueType(Enum):
    """Type of task queue implementation."""
    
    REDIS_STREAMS = "redis_streams"  # Redis streams (recommended)
    REDIS_LISTS = "redis_lists"      # Redis lists (simple)
    MEMORY = "memory"                # In-memory (single process)
    EXTERNAL = "external"            # External queue system (RabbitMQ, etc.)


# =============================================================================
# Parallel Execution Constructs
# =============================================================================

@dataclass
class ParallelBlock:
    """
    A group of workflow steps that execute concurrently.
    
    Represents parallel execution of independent steps with configurable
    aggregation strategies, concurrency limits, and error handling.
    
    Example DSL:
        parallel analysis_batch {
            strategy: all
            max_concurrency: 5
            timeout_seconds: 300
            
            steps: [
                { kind: tool, target: sentiment_analyzer },
                { kind: tool, target: entity_extractor },
                { kind: prompt, target: summarizer }
            ]
        }
    """
    name: str
    steps: List[Any] = field(default_factory=list)  # List of ChainStep or WorkflowNode
    strategy: ParallelStrategy = ParallelStrategy.ALL
    max_concurrency: Optional[int] = None  # Limit concurrent tasks
    timeout_seconds: Optional[float] = None
    
    # Error handling
    fail_fast: bool = True  # Cancel remaining on first failure (for ALL strategy)
    retry_failed: bool = False
    max_retries: int = 3
    
    # Result aggregation
    reduce_function: Optional[str] = None  # Function name for MAP_REDUCE
    output_key: Optional[str] = None
    
    # Dependencies (steps that must complete before this parallel block)
    depends_on: List[str] = field(default_factory=list)
    
    # Metadata
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[SourceLocation] = None


@dataclass  
class FanOut:
    """
    Fan-out operation that splits input data across parallel workers.
    
    Takes input data and distributes it across multiple parallel executions
    of the same step or workflow.
    
    Example DSL:
        fan_out batch_processing {
            target: document_processor
            split_by: documents  # Split input.documents array
            batch_size: 10       # Process 10 documents per worker
            max_workers: 20
        }
    """
    name: str
    target: str  # Step or workflow to execute in parallel
    split_by: str  # Field name to split (e.g., "documents", "items")
    batch_size: Optional[int] = None  # Items per batch
    max_workers: Optional[int] = None
    
    # Strategy for combining results
    combine_strategy: ParallelStrategy = ParallelStrategy.COLLECT
    reduce_function: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[SourceLocation] = None


@dataclass
class FanIn:
    """
    Fan-in operation that aggregates results from parallel executions.
    
    Collects and combines results from multiple parallel branches
    into a single output value.
    """
    name: str
    inputs: List[str] = field(default_factory=list)  # Step names to wait for
    combine_function: Optional[str] = None  # Aggregation function
    output_key: str = "combined_result"
    
    # Timeout for waiting on inputs
    timeout_seconds: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[SourceLocation] = None


# =============================================================================
# Distributed Execution Constructs  
# =============================================================================

@dataclass
class WorkerPool:
    """
    Definition of a worker pool for distributed execution.
    
    Defines a group of workers that can process tasks from specific queues,
    with security policies and resource limits.
    
    Example DSL:
        worker_pool analysis_workers {
            queue_name: "analysis_tasks"
            min_workers: 2
            max_workers: 10
            
            capabilities: [NETWORK, HTTP_READ]
            permission_level: READ_WRITE
            allowed_tools: [web_search, document_analyzer]
            
            scaling: {
                metric: queue_depth
                target_value: 5
                scale_up_threshold: 10
                scale_down_threshold: 2
            }
        }
    """
    name: str
    queue_name: str
    description: Optional[str] = None
    
    # Worker scaling
    min_workers: int = 1
    max_workers: int = 10
    auto_scaling: bool = True
    
    # Security configuration
    capabilities: List[CapabilityType] = field(default_factory=list)
    permission_level: PermissionLevel = PermissionLevel.READ_ONLY
    allowed_tools: List[str] = field(default_factory=list)
    allowed_agents: List[str] = field(default_factory=list)
    
    # Resource limits
    max_memory_mb: Optional[int] = None
    max_cpu_cores: Optional[float] = None
    timeout_seconds: Optional[float] = None
    
    # Scaling policy
    scaling_metric: str = "queue_depth"  # queue_depth, cpu_usage, memory_usage
    target_value: float = 5.0
    scale_up_threshold: float = 10.0
    scale_down_threshold: float = 2.0
    
    # Worker configuration
    worker_config: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[SourceLocation] = None


@dataclass
class TaskQueue:
    """
    Configuration for task queues used in distributed execution.
    
    Defines queue properties, reliability guarantees, and broker settings
    for distributed task execution.
    
    Example DSL:
        task_queue critical_tasks {
            type: redis_streams
            reliability: at_least_once
            max_retries: 5
            retry_backoff: exponential
            
            dead_letter_queue: critical_dlq
            visibility_timeout: 300
            
            redis_config: {
                stream_name: "namel3ss:tasks:critical"
                consumer_group: "workers"
            }
        }
    """
    name: str
    queue_type: QueueType = QueueType.REDIS_STREAMS
    description: Optional[str] = None
    
    # Reliability and retry configuration  
    reliability: str = "at_least_once"  # at_most_once, at_least_once, exactly_once
    max_retries: int = 3
    retry_backoff: str = "exponential"  # linear, exponential, fixed
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    
    # Queue behavior
    visibility_timeout: float = 300.0  # Seconds before message becomes visible again
    message_retention: float = 86400.0  # 24 hours default
    max_message_size: int = 1048576  # 1MB default
    
    # Dead letter queue
    dead_letter_queue: Optional[str] = None
    max_dead_letter_retries: int = 3
    
    # Broker-specific configuration
    redis_config: Dict[str, Any] = field(default_factory=dict)
    rabbitmq_config: Dict[str, Any] = field(default_factory=dict)
    external_config: Dict[str, Any] = field(default_factory=dict)
    
    # Monitoring
    enable_metrics: bool = True
    metric_tags: Dict[str, str] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[SourceLocation] = None


@dataclass
class DistributedExecution:
    """
    Configuration for executing steps on remote worker pools.
    
    Specifies how to distribute workflow steps across worker pools,
    with load balancing, locality preferences, and failure handling.
    
    Example DSL:
        distributed heavy_computation {
            worker_pool: gpu_workers
            queue: ml_tasks
            distribution_policy: load_based
            locality_preference: us_east_1
            
            failure_mode: retry_local
            max_wait_seconds: 600
        }
    """
    worker_pool: str
    queue_name: Optional[str] = None
    distribution_policy: DistributionPolicy = DistributionPolicy.ROUND_ROBIN
    
    # Locality and affinity
    locality_preference: Optional[str] = None  # Region, zone, etc.
    worker_affinity: Optional[str] = None  # Specific worker requirements
    avoid_workers: List[str] = field(default_factory=list)
    
    # Failure handling
    failure_mode: str = "retry_remote"  # retry_remote, retry_local, fail_fast
    max_wait_seconds: Optional[float] = None
    fallback_to_local: bool = True
    
    # Resource requirements
    required_memory_mb: Optional[int] = None
    required_cpu_cores: Optional[float] = None
    required_capabilities: List[CapabilityType] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[SourceLocation] = None


# =============================================================================
# Event-Driven Execution Constructs
# =============================================================================

@dataclass
class EventTrigger:
    """
    Event-driven trigger for workflow execution.
    
    Defines conditions and events that can trigger workflow execution,
    enabling reactive and event-driven architectures.
    
    Example DSL:
        event_trigger new_customer {
            event_type: dataset_change
            dataset: customers
            filter: { action: insert }
            
            trigger_workflow: customer_onboarding
            trigger_delay: 5  # seconds
            deduplicate_window: 60
        }
    """
    name: str
    event_type: str  # dataset_change, webhook, schedule, manual, message
    description: Optional[str] = None
    
    # Event source configuration
    source_config: Dict[str, Any] = field(default_factory=dict)
    
    # Event filtering
    filter_expression: Optional[Union[str, Expression]] = None
    filter_fields: Dict[str, Any] = field(default_factory=dict)
    
    # Trigger behavior
    trigger_target: str  # Chain, workflow, or agent to trigger
    trigger_delay: Optional[float] = None  # Delay in seconds
    deduplicate_window: Optional[float] = None  # Dedup window in seconds
    
    # Execution context
    context_mapping: Dict[str, str] = field(default_factory=dict)  # event_field -> context_field
    execution_mode: str = "async"  # async, sync, background
    
    # Rate limiting
    max_triggers_per_minute: Optional[int] = None
    max_triggers_per_hour: Optional[int] = None
    
    # Security
    required_capabilities: List[CapabilityType] = field(default_factory=list)
    permission_level: PermissionLevel = PermissionLevel.READ_ONLY
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[SourceLocation] = None


@dataclass
class EventHandler:
    """
    Handler for processing specific types of events.
    
    Defines how to process events, including transformation,
    routing, and integration with workflows.
    
    Example DSL:
        event_handler task_completion {
            event_types: [task_completed, task_failed]
            
            transform_function: normalize_task_event
            route_to: task_status_updater
            
            retry_policy: {
                max_retries: 3
                backoff: exponential
            }
        }
    """
    name: str
    event_types: List[str] = field(default_factory=list)
    description: Optional[str] = None
    
    # Event processing
    transform_function: Optional[str] = None
    enrichment_sources: List[str] = field(default_factory=list)
    
    # Routing
    route_to: Optional[str] = None  # Target workflow/chain/agent
    routing_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error handling
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    dead_letter_queue: Optional[str] = None
    
    # Security
    required_capabilities: List[CapabilityType] = field(default_factory=list)
    permission_level: PermissionLevel = PermissionLevel.READ_ONLY
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[SourceLocation] = None


# =============================================================================
# Extended Workflow Nodes
# =============================================================================

@dataclass
class ParallelStep:
    """
    A workflow step that executes a parallel block.
    
    Integrates parallel execution into the existing workflow system
    as a first-class step type.
    
    Example DSL:
        step analyze_data {
            kind: parallel
            target: data_analysis_parallel
            parallel_config: {
                max_concurrency: 10
                strategy: all
            }
        }
    """
    kind: str = "parallel"  # Step kind identifier
    target: str  # Name of ParallelBlock to execute
    parallel_config: Dict[str, Any] = field(default_factory=dict)
    
    # Standard step properties
    name: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    output_key: Optional[str] = None
    stop_on_error: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[SourceLocation] = None


@dataclass
class DistributedStep:
    """
    A workflow step that executes on a distributed worker pool.
    
    Represents remote execution of a step with queue-based distribution
    and worker pool management.
    
    Example DSL:
        step heavy_processing {
            kind: distributed
            target: ml_model_inference
            worker_pool: gpu_cluster
            queue: ml_tasks
            
            distributed_config: {
                timeout_seconds: 300
                retry_on_failure: true
            }
        }
    """
    kind: str = "distributed"  # Step kind identifier
    target: str  # Step/workflow to execute remotely
    worker_pool: str
    queue_name: Optional[str] = None
    
    distributed_config: DistributedExecution = field(default_factory=DistributedExecution)
    
    # Standard step properties
    name: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    output_key: Optional[str] = None
    stop_on_error: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[SourceLocation] = None


@dataclass
class EventStep:
    """
    A workflow step that processes events.
    
    Represents event-driven execution within workflows,
    allowing chains to respond to events and emit events.
    
    Example DSL:
        step handle_webhook {
            kind: event
            event_type: webhook_received
            handler: webhook_processor
            
            emit_events: [processing_started]
        }
    """
    kind: str = "event"  # Step kind identifier
    event_type: str
    handler: str  # Event handler name
    
    # Event emission
    emit_events: List[str] = field(default_factory=list)
    event_data_mapping: Dict[str, Any] = field(default_factory=dict)
    
    # Standard step properties
    name: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    output_key: Optional[str] = None
    stop_on_error: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[SourceLocation] = None


# =============================================================================
# Union Types and Exports
# =============================================================================

# Extended workflow node types that include parallel and distributed execution
ParallelWorkflowNode = Union[
    Any,  # Existing WorkflowNode types (ChainStep, WorkflowIfBlock, etc.)
    ParallelBlock,
    FanOut,
    FanIn,
    ParallelStep,
    DistributedStep, 
    EventStep,
]

# All parallel/distributed execution constructs
ParallelConstruct = Union[
    ParallelBlock,
    FanOut,
    FanIn,
    WorkerPool,
    TaskQueue,
    DistributedExecution,
    EventTrigger,
    EventHandler,
]


__all__ = [
    # Enums
    "ParallelStrategy",
    "DistributionPolicy", 
    "QueueType",
    
    # Parallel execution
    "ParallelBlock",
    "FanOut",
    "FanIn",
    "ParallelStep",
    
    # Distributed execution
    "WorkerPool",
    "TaskQueue", 
    "DistributedExecution",
    "DistributedStep",
    
    # Event-driven execution
    "EventTrigger",
    "EventHandler",
    "EventStep",
    
    # Union types
    "ParallelWorkflowNode",
    "ParallelConstruct",
]