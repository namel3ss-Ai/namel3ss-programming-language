"""
Runtime-agnostic IR specifications for Namel3ss.

This module defines the intermediate representation (IR) that sits between
the language AST and concrete runtime implementations. All types here are
deliberately free of runtime-specific concepts (no FastAPI, HTTP status codes,
React components, etc.).

Design Principles:
------------------
1. **Runtime Agnostic**: No imports of web frameworks, cloud SDKs, etc.
2. **Serializable**: All types are dataclasses that can be JSON-serialized
3. **Complete**: Captures all language semantics needed by runtimes
4. **Versioned**: IR can evolve independently of language syntax
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Import security and parallel execution types for IR metadata
try:
    from namel3ss.ast.security import PermissionLevel, SecurityPolicy
    from namel3ss.ast.parallel import ParallelStrategy, DistributionPolicy, QueueType
    HAS_SECURITY = True
    HAS_PARALLEL = True
except ImportError:
    HAS_SECURITY = False
    HAS_PARALLEL = False
    PermissionLevel = None
    SecurityPolicy = None
    ParallelStrategy = None
    DistributionPolicy = None
    QueueType = None


# =============================================================================
# Enumerations
# =============================================================================

class HTTPMethod(str, Enum):
    """HTTP methods for endpoints (runtime-agnostic)"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class MemoryScope(str, Enum):
    """Memory scope types"""
    GLOBAL = "global"
    SESSION = "session"
    CONVERSATION = "conversation"
    REQUEST = "request"


class CacheStrategy(str, Enum):
    """Caching strategies"""
    NONE = "none"
    TTL = "ttl"
    LRU = "lru"
    FIFO = "fifo"


# =============================================================================
# Type System
# =============================================================================

@dataclass
class SchemaField:
    """A field in a structured schema"""
    name: str
    type_spec: TypeSpec
    description: Optional[str] = None
    required: bool = True
    default: Optional[Any] = None


@dataclass
class TypeSpec:
    """
    Runtime-agnostic type specification.
    
    Represents types from the language without commitment to a specific
    validation library (Pydantic, JSON Schema, TypeScript types, etc.).
    """
    kind: str  # "string", "integer", "float", "boolean", "object", "array", "union", etc.
    fields: Optional[List[SchemaField]] = None  # For object types
    item_type: Optional[TypeSpec] = None  # For array types
    enum_values: Optional[List[str]] = None  # For enum types
    union_types: Optional[List[TypeSpec]] = None  # For union types
    constraints: Dict[str, Any] = field(default_factory=dict)  # min, max, pattern, etc.


# =============================================================================
# Backend IR Specifications
# =============================================================================

@dataclass
class EndpointIR:
    """
    Runtime-agnostic API endpoint specification.
    
    Represents a single API endpoint without committing to HTTP, gRPC,
    GraphQL, or any specific protocol.
    """
    path: str
    method: HTTPMethod
    input_schema: TypeSpec
    output_schema: TypeSpec
    handler_type: str  # "prompt", "agent", "tool", "chain", "dataset"
    handler_ref: str  # Reference to the handler (e.g., "prompts.ClassifyTicket")
    description: Optional[str] = None
    auth_required: bool = False
    rate_limit: Optional[int] = None  # requests per minute
    timeout: Optional[float] = None  # seconds
    cache_strategy: CacheStrategy = CacheStrategy.NONE
    cache_ttl: Optional[int] = None  # seconds
    
    # Security metadata
    required_permission_level: Optional[str] = None  # Minimum permission required
    allowed_capabilities: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptSpec:
    """Structured prompt specification"""
    name: str
    input_schema: TypeSpec
    output_schema: TypeSpec
    template: str
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_message: Optional[str] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    memory_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSpec:
    """Multi-agent specification"""
    name: str
    nodes: List[Dict[str, Any]]  # Agent graph nodes
    edges: List[Dict[str, Any]]  # Agent graph edges
    entry_point: str
    handoff_logic: Dict[str, Any]
    state_schema: TypeSpec
    
    # Security metadata
    allowed_tools: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    permission_level: Optional[str] = None  # Serialized PermissionLevel
    security_policy: Optional[Dict[str, Any]] = None  # Serialized SecurityPolicy
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    """Tool specification"""
    name: str
    description: str
    input_schema: TypeSpec
    output_schema: TypeSpec
    implementation_type: str  # "python", "http", "database", etc.
    implementation_ref: str  # Reference to implementation
    
    # Security metadata
    required_capabilities: List[str] = field(default_factory=list)
    permission_level: Optional[str] = None  # Serialized PermissionLevel
    timeout_seconds: Optional[float] = None
    rate_limit_per_minute: Optional[int] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSpec:
    """Dataset specification"""
    name: str
    source_type: str  # "sql", "csv", "api", "vector", etc.
    source_config: Dict[str, Any]
    schema: List[SchemaField]
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    cache_policy: Optional[Dict[str, Any]] = None
    refresh_policy: Optional[Dict[str, Any]] = None
    
    # Data binding & access control
    access_policy: Optional[Dict[str, Any]] = None  # Read-only, allow CRUD
    primary_key: Optional[str] = None  # Required for updates/deletes
    realtime_enabled: bool = False  # Push updates via WebSocket/Redis
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameSpec:
    """Frame (table) specification"""
    name: str
    columns: List[SchemaField]
    source_dataset: Optional[str] = None
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    indexes: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorySpec:
    """Memory system specification"""
    name: str
    scope: MemoryScope
    kind: str  # "conversation", "key_value", "vector", etc.
    max_items: Optional[int] = None
    ttl: Optional[int] = None  # seconds
    embedding_model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainSpec:
    """Chain/workflow specification"""
    name: str
    steps: List[Dict[str, Any]]
    input_schema: TypeSpec
    output_schema: TypeSpec
    error_handling: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerSpec:
    """Planning and reasoning specification"""
    name: str
    planner_type: str  # "react", "chain_of_thought", "graph_based"
    goal: str
    input_schema: TypeSpec
    output_schema: TypeSpec
    
    # ReAct-specific configuration
    max_cycles: Optional[int] = None
    reasoning_prompt: Optional[str] = None
    action_tools: List[str] = field(default_factory=list)
    success_condition: Optional[Dict[str, Any]] = None
    fallback_action: Optional[str] = None
    
    # Chain-of-Thought configuration
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)
    step_prompts: Dict[str, str] = field(default_factory=dict)
    step_tools: Dict[str, List[str]] = field(default_factory=dict)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    
    # Graph-based configuration  
    initial_state: Optional[Dict[str, Any]] = None
    goal_state: Optional[Dict[str, Any]] = None
    search_policy: Optional[Dict[str, Any]] = None
    state_transitions: List[Dict[str, Any]] = field(default_factory=list)
    heuristic_function: Optional[str] = None
    max_search_time: Optional[float] = None
    
    # Security metadata
    allowed_tools: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list) 
    permission_level: Optional[str] = None
    security_policy: Optional[Dict[str, Any]] = None
    
    # Performance & monitoring
    timeout_seconds: Optional[float] = None
    max_memory_usage: Optional[int] = None  # bytes
    trace_enabled: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchPolicySpec:
    """Search policy specification for graph-based planners"""
    policy_type: str  # "beam_search", "greedy_search", "mcts"
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Beam search parameters
    beam_width: Optional[int] = None
    max_depth: Optional[int] = None
    scoring_function: Optional[str] = None
    
    # Greedy search parameters  
    max_steps: Optional[int] = None
    confidence_threshold: Optional[float] = None
    
    # MCTS parameters
    num_simulations: Optional[int] = None
    exploration_constant: Optional[float] = None
    max_simulation_depth: Optional[int] = None


@dataclass
class PlanningWorkflowSpec:
    """High-level planning workflow specification"""
    name: str
    input_schema: TypeSpec
    output_schema: TypeSpec
    stages: List[Dict[str, Any]] = field(default_factory=list)
    stage_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    global_context: Dict[str, Any] = field(default_factory=dict)
    error_handling: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsightSpec:
    """Insight specification"""
    name: str
    query: str
    dataset_ref: str
    aggregations: List[Dict[str, Any]] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalModelSpec:
    """Local model deployment specification"""
    name: str
    engine_type: str  # "vllm", "ollama", "local_ai", "lm_studio"
    model_name: str
    model_path: Optional[str] = None  # Path to model files for local models
    
    # Deployment configuration
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    
    # Engine-specific configurations
    vllm_config: Optional[Dict[str, Any]] = None
    ollama_config: Optional[Dict[str, Any]] = None
    local_ai_config: Optional[Dict[str, Any]] = None
    
    # Model configuration
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Hardware requirements
    gpu_required: bool = False
    min_vram_gb: Optional[float] = None
    min_ram_gb: Optional[float] = None
    cpu_cores: Optional[int] = None
    
    # Serving configuration
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_requests: int = 100
    max_model_len: Optional[int] = None
    
    # Fine-tuning support
    supports_fine_tuning: bool = False
    fine_tuning_config: Optional[Dict[str, Any]] = None
    
    # Health and monitoring
    health_check_endpoint: str = "/health"
    metrics_enabled: bool = True
    
    # Security
    api_key_required: bool = False
    allowed_origins: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendIR:
    """
    Complete backend intermediate representation.
    
    This is the top-level IR that runtime adapters consume to generate
    concrete backend implementations (FastAPI, Lambda, gRPC, etc.).
    """
    app_name: str
    app_version: str = "1.0.0"
    
    # API specifications
    endpoints: List[EndpointIR] = field(default_factory=list)
    
    # AI/ML components
    prompts: List[PromptSpec] = field(default_factory=list)
    agents: List[AgentSpec] = field(default_factory=list)
    tools: List[ToolSpec] = field(default_factory=list)
    chains: List[ChainSpec] = field(default_factory=list)
    planners: List[PlannerSpec] = field(default_factory=list)
    planning_workflows: List[PlanningWorkflowSpec] = field(default_factory=list)
    
    # Local model deployments
    local_models: List[LocalModelSpec] = field(default_factory=list)
    
    # Data components
    datasets: List[DatasetSpec] = field(default_factory=list)
    frames: List[FrameSpec] = field(default_factory=list)
    insights: List[InsightSpec] = field(default_factory=list)
    
    # Data binding & realtime
    update_channels: List[UpdateChannelSpec] = field(default_factory=list)
    
    # Parallel and distributed execution
    parallel_blocks: List[ParallelBlockSpec] = field(default_factory=list)
    worker_pools: List[WorkerPoolSpec] = field(default_factory=list)
    task_queues: List[TaskQueueSpec] = field(default_factory=list)
    distributed_configs: List[DistributedExecutionSpec] = field(default_factory=list)
    event_triggers: List[EventTriggerSpec] = field(default_factory=list)
    event_handlers: List[EventHandlerSpec] = field(default_factory=list)
    
    # State management
    memory: List[MemorySpec] = field(default_factory=list)
    
    # Configuration
    database_config: Optional[Dict[str, Any]] = None
    auth_config: Optional[Dict[str, Any]] = None
    cors_config: Optional[Dict[str, Any]] = None
    realtime_config: Optional[Dict[str, Any]] = None  # WebSocket/Redis configuration
    distributed_config: Optional[Dict[str, Any]] = None  # Distributed execution configuration
    
    # Security configuration
    security_config: Optional[Dict[str, Any]] = None  # Global security settings
    agent_tool_mappings: Dict[str, List[str]] = field(default_factory=dict)  # agent_name -> [tool_names]
    capability_requirements: Dict[str, List[str]] = field(default_factory=dict)  # tool_name -> [capabilities]
    permission_levels: Dict[str, str] = field(default_factory=dict)  # agent/tool_name -> permission_level
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    ir_version: str = "0.1.0"  # IR format version


# =============================================================================
# Data Binding Specifications
# =============================================================================

@dataclass
class DataBindingSpec:
    """
    Specification for dynamic data binding between UI components and datasets.
    
    Controls how data flows between backend datasets and frontend components,
    including pagination, filtering, sorting, and realtime updates.
    """
    # Source configuration
    dataset_name: str
    endpoint_path: str  # Generated API endpoint path
    
    # Read behavior
    page_size: int = 50
    enable_sorting: bool = True
    sortable_fields: List[str] = field(default_factory=list)  # Whitelist of sortable columns
    enable_filtering: bool = True
    filterable_fields: List[str] = field(default_factory=list)  # Whitelist of filterable columns
    enable_search: bool = False
    searchable_fields: List[str] = field(default_factory=list)
    
    # Write behavior
    editable: bool = False
    enable_create: bool = False
    enable_update: bool = False
    enable_delete: bool = False
    create_endpoint: Optional[str] = None  # POST /api/datasets/{name}
    update_endpoint: Optional[str] = None  # PATCH /api/datasets/{name}/{id}
    delete_endpoint: Optional[str] = None  # DELETE /api/datasets/{name}/{id}
    
    # Realtime updates
    subscribe_to_changes: bool = False
    websocket_topic: Optional[str] = None  # e.g., "dataset:{name}:changes"
    polling_interval: Optional[int] = None  # Fallback polling interval in seconds
    
    # Optimization
    cache_ttl: Optional[int] = None  # Client-side cache TTL
    optimistic_updates: bool = True
    
    # Field mapping
    field_mapping: Dict[str, str] = field(default_factory=dict)  # component_field -> dataset_column
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UpdateChannelSpec:
    """
    Specification for realtime update channels.
    
    Defines how dataset changes are broadcasted to subscribed clients.
    """
    name: str  # Channel/topic name
    dataset_name: str
    event_types: List[str] = field(default_factory=lambda: ["create", "update", "delete"])
    transport: str = "websocket"  # "websocket" | "sse" | "polling"
    requires_auth: bool = True
    required_capabilities: List[str] = field(default_factory=list)
    
    # Redis pub/sub configuration (when realtime extra is available)
    redis_channel: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Parallel and Distributed Execution IR Specifications
# =============================================================================

@dataclass
class ParallelBlockSpec:
    """
    IR specification for parallel execution blocks.
    
    Runtime-agnostic representation of parallel step execution
    with configurable strategies and concurrency control.
    """
    name: str
    steps: List[Dict[str, Any]]  # Step specifications to execute in parallel
    strategy: str = "all"  # ParallelStrategy enum value as string
    max_concurrency: Optional[int] = None
    timeout_seconds: Optional[float] = None
    
    # Error handling
    fail_fast: bool = True
    retry_failed: bool = False
    max_retries: int = 3
    
    # Result aggregation
    reduce_function: Optional[str] = None
    output_key: Optional[str] = None
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Security metadata
    required_permission_level: Optional[str] = None
    allowed_capabilities: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerPoolSpec:
    """
    IR specification for distributed worker pools.
    
    Defines worker pool configuration, scaling policies,
    and security constraints for distributed execution.
    """
    name: str
    queue_name: str
    description: Optional[str] = None
    
    # Worker scaling
    min_workers: int = 1
    max_workers: int = 10
    auto_scaling: bool = True
    scaling_metric: str = "queue_depth"
    target_value: float = 5.0
    scale_up_threshold: float = 10.0
    scale_down_threshold: float = 2.0
    
    # Security configuration (as strings for IR)
    capabilities: List[str] = field(default_factory=list)
    permission_level: str = "read_only"
    allowed_tools: List[str] = field(default_factory=list)
    allowed_agents: List[str] = field(default_factory=list)
    
    # Resource limits
    max_memory_mb: Optional[int] = None
    max_cpu_cores: Optional[float] = None
    timeout_seconds: Optional[float] = None
    
    # Worker configuration
    worker_config: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskQueueSpec:
    """
    IR specification for task queues.
    
    Runtime-agnostic queue configuration for distributed
    task execution with reliability and monitoring.
    """
    name: str
    queue_type: str = "redis_streams"  # QueueType enum value as string
    description: Optional[str] = None
    
    # Reliability configuration
    reliability: str = "at_least_once"
    max_retries: int = 3
    retry_backoff: str = "exponential"
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    
    # Queue behavior
    visibility_timeout: float = 300.0
    message_retention: float = 86400.0
    max_message_size: int = 1048576
    
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


@dataclass
class DistributedExecutionSpec:
    """
    IR specification for distributed step execution.
    
    Defines how to execute workflow steps on remote workers
    with load balancing and failure handling.
    """
    worker_pool: str
    queue_name: Optional[str] = None
    distribution_policy: str = "round_robin"  # DistributionPolicy enum as string
    
    # Locality and affinity
    locality_preference: Optional[str] = None
    worker_affinity: Optional[str] = None
    avoid_workers: List[str] = field(default_factory=list)
    
    # Failure handling
    failure_mode: str = "retry_remote"
    max_wait_seconds: Optional[float] = None
    fallback_to_local: bool = True
    
    # Resource requirements
    required_memory_mb: Optional[int] = None
    required_cpu_cores: Optional[float] = None
    required_capabilities: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventTriggerSpec:
    """
    IR specification for event-driven triggers.
    
    Runtime-agnostic representation of event triggers
    that can start workflow execution.
    """
    name: str
    event_type: str
    trigger_target: str  # Chain/workflow/agent to trigger
    description: Optional[str] = None
    
    # Event source configuration
    source_config: Dict[str, Any] = field(default_factory=dict)
    
    # Event filtering
    filter_expression: Optional[str] = None
    filter_fields: Dict[str, Any] = field(default_factory=dict)
    
    # Trigger behavior
    trigger_delay: Optional[float] = None
    deduplicate_window: Optional[float] = None
    
    # Execution context
    context_mapping: Dict[str, str] = field(default_factory=dict)
    execution_mode: str = "async"
    
    # Rate limiting
    max_triggers_per_minute: Optional[int] = None
    max_triggers_per_hour: Optional[int] = None
    
    # Security
    required_capabilities: List[str] = field(default_factory=list)
    permission_level: str = "read_only"
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventHandlerSpec:
    """
    IR specification for event handlers.
    
    Defines event processing logic and routing
    in a runtime-agnostic manner.
    """
    name: str
    event_types: List[str] = field(default_factory=list)
    description: Optional[str] = None
    
    # Event processing
    transform_function: Optional[str] = None
    enrichment_sources: List[str] = field(default_factory=list)
    
    # Routing
    route_to: Optional[str] = None
    routing_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error handling
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    dead_letter_queue: Optional[str] = None
    
    # Security
    required_capabilities: List[str] = field(default_factory=list)
    permission_level: str = "read_only"
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Frontend IR Specifications
# =============================================================================

@dataclass
class ComponentSpec:
    """Frontend component specification"""
    name: str
    type: str  # "text", "table", "chart", "form", etc.
    props: Dict[str, Any] = field(default_factory=dict)
    data_source: Optional[str] = None  # Reference to endpoint/dataset
    
    # Data binding configuration
    binding: Optional[DataBindingSpec] = None
    
    children: List[ComponentSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Data Display Component IR Specifications
# =============================================================================

@dataclass
class IRColumnConfig:
    """IR specification for table column configuration."""
    id: str
    label: str
    field: Optional[str] = None
    width: Optional[Union[str, int]] = None
    align: str = "left"  # "left" | "center" | "right"
    sortable: bool = True
    format: Optional[str] = None  # "currency" | "date" | "number" | "percentage"
    transform: Optional[Union[str, Dict[str, Any]]] = None
    render_template: Optional[str] = None


@dataclass
class IRToolbarConfig:
    """IR specification for data table toolbar."""
    search: Optional[Dict[str, Any]] = None  # {field: str, placeholder: str}
    filters: List[Dict[str, Any]] = field(default_factory=list)  # [{field, type, options}, ...]
    bulk_actions: List[Dict[str, Any]] = field(default_factory=list)  # Action specs
    actions: List[Dict[str, Any]] = field(default_factory=list)  # Additional toolbar actions


@dataclass
class IRDataTable:
    """
    IR specification for professional data table component.
    
    Runtime-agnostic representation of table with columns, sorting,
    filtering, pagination, row actions, and toolbar.
    """
    title: str
    source_type: str  # "dataset" | "table" | "frame"
    source: str  # Dataset/table/frame name
    
    # Column configuration
    columns: List[IRColumnConfig] = field(default_factory=list)
    
    # Row actions (per-row operations)
    row_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Toolbar configuration
    toolbar: Optional[IRToolbarConfig] = None
    
    # Filtering and sorting
    filter_by: Optional[str] = None  # SQL WHERE clause or filter expression
    sort_by: Optional[str] = None  # Column name
    default_sort: Optional[Dict[str, str]] = None  # {column: str, direction: "asc"|"desc"}
    
    # Pagination
    page_size: int = 50
    enable_pagination: bool = True
    
    # Empty state
    empty_state: Optional[Dict[str, Any]] = None
    
    # Data binding
    binding: Optional[DataBindingSpec] = None
    
    # Styling
    layout: Optional[Dict[str, Any]] = None
    style: Optional[Dict[str, Any]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRListItemConfig:
    """IR specification for list item configuration."""
    avatar: Optional[Dict[str, Any]] = None  # Avatar config
    title: Union[str, Dict[str, Any]] = ""  # Static or dynamic
    subtitle: Optional[Union[str, Dict[str, Any]]] = None
    metadata: Dict[str, Union[str, Dict[str, Any]]] = field(default_factory=dict)  # Key-value pairs
    actions: List[Dict[str, Any]] = field(default_factory=list)
    badge: Optional[Dict[str, Any]] = None
    icon: Optional[str] = None
    state_class: Optional[Dict[str, str]] = None  # Conditional CSS classes


@dataclass
class IRDataList:
    """
    IR specification for data list component.
    
    Runtime-agnostic representation of vertical list for activity feeds,
    notifications, search results with avatar, title, subtitle, metadata.
    """
    title: str
    source_type: str  # "dataset" | "table" | "frame"
    source: str
    
    # Item configuration
    item: Optional[IRListItemConfig] = None
    
    # Variants
    variant: str = "default"  # "default" | "compact" | "detailed"
    dividers: bool = True
    
    # Filtering and search
    filter_by: Optional[str] = None
    enable_search: bool = False
    search_placeholder: Optional[str] = None
    
    # Pagination
    page_size: int = 50
    enable_pagination: bool = True
    
    # Empty state
    empty_state: Optional[Dict[str, Any]] = None
    
    # Data binding
    binding: Optional[DataBindingSpec] = None
    
    # Styling
    layout: Optional[Dict[str, Any]] = None
    style: Optional[Dict[str, Any]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRSparklineConfig:
    """IR specification for sparkline mini-chart."""
    data_source: str
    x_field: str
    y_field: str
    color: Optional[str] = None
    variant: str = "line"  # "line" | "bar" | "area"


@dataclass
class IRStatSummary:
    """
    IR specification for stat summary/KPI card component.
    
    Runtime-agnostic representation of metric cards with value,
    delta, trend indicator, and optional sparkline.
    """
    label: str
    source_type: str  # "dataset" | "table" | "frame"
    source: str
    
    # Value configuration
    value: str  # Field name or expression
    format: Optional[str] = None  # "currency" | "number" | "percentage"
    prefix: Optional[str] = None  # "$", etc.
    suffix: Optional[str] = None  # "%", " users", etc.
    
    # Delta/comparison
    delta: Optional[Dict[str, Any]] = None  # {value, format, label}
    trend: Optional[Union[str, Dict[str, Any]]] = None  # "up" | "down" | "neutral" or dynamic
    comparison_period: Optional[str] = None  # "vs last week", etc.
    
    # Sparkline
    sparkline: Optional[IRSparklineConfig] = None
    
    # Styling
    color: Optional[str] = None  # Accent color
    icon: Optional[str] = None
    
    # Data binding
    binding: Optional[DataBindingSpec] = None
    
    # Layout
    layout: Optional[Dict[str, Any]] = None
    style: Optional[Dict[str, Any]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRTimelineItem:
    """IR specification for timeline item configuration."""
    timestamp: Union[str, Dict[str, Any]] = ""  # Field or expression
    title: Union[str, Dict[str, Any]] = ""
    description: Optional[Union[str, Dict[str, Any]]] = None
    icon: Optional[Union[str, Dict[str, Any]]] = None
    status: Optional[Union[str, Dict[str, Any]]] = None  # "success" | "warning" | "error" | "info"
    color: Optional[str] = None
    actions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class IRTimeline:
    """
    IR specification for timeline component.
    
    Runtime-agnostic representation of chronological event timeline
    with timestamps, icons, status indicators, and actions.
    """
    title: str
    source_type: str  # "dataset" | "table" | "frame"
    source: str
    
    # Item configuration
    item: Optional[IRTimelineItem] = None
    
    # Display options
    variant: str = "default"  # "default" | "compact" | "detailed"
    show_timestamps: bool = True
    group_by_date: bool = False
    
    # Filtering and sorting
    filter_by: Optional[str] = None
    sort_by: Optional[str] = None
    
    # Pagination
    page_size: int = 50
    enable_pagination: bool = True
    
    # Empty state
    empty_state: Optional[Dict[str, Any]] = None
    
    # Data binding
    binding: Optional[DataBindingSpec] = None
    
    # Styling
    layout: Optional[Dict[str, Any]] = None
    style: Optional[Dict[str, Any]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRAvatarItem:
    """IR specification for avatar item configuration."""
    name: Optional[Union[str, Dict[str, Any]]] = None
    image_url: Optional[Union[str, Dict[str, Any]]] = None
    initials: Optional[Union[str, Dict[str, Any]]] = None
    color: Optional[Union[str, Dict[str, Any]]] = None
    status: Optional[Union[str, Dict[str, Any]]] = None  # "online" | "offline" | "busy" | "away"
    tooltip: Optional[Union[str, Dict[str, Any]]] = None


@dataclass
class IRAvatarGroup:
    """
    IR specification for avatar group component.
    
    Runtime-agnostic representation of compact user/agent display
    with avatars, status indicators, and "+N more" overflow.
    """
    title: Optional[str] = None
    source_type: str = "dataset"  # "dataset" | "table" | "frame"
    source: str = ""
    
    # Item configuration
    item: Optional[IRAvatarItem] = None
    
    # Display options
    max_visible: int = 5
    size: str = "md"  # "xs" | "sm" | "md" | "lg" | "xl"
    variant: str = "stacked"  # "stacked" | "grid"
    
    # Filtering
    filter_by: Optional[str] = None
    
    # Data binding
    binding: Optional[DataBindingSpec] = None
    
    # Styling
    layout: Optional[Dict[str, Any]] = None
    style: Optional[Dict[str, Any]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRChartConfig:
    """IR specification for enhanced chart configuration."""
    variant: str = "line"  # "line" | "bar" | "pie" | "area" | "scatter"
    x_field: str = ""
    y_fields: List[str] = field(default_factory=list)  # Multi-series support
    
    # Grouping and stacking
    group_by: Optional[str] = None
    stacked: bool = False
    
    # Line/area specific
    smooth: bool = True
    fill: bool = True
    
    # Chart elements
    legend: Optional[Dict[str, Any]] = None  # {position: "top"|"bottom"|"left"|"right", show: bool}
    tooltip: Optional[Dict[str, Any]] = None  # {show: bool, format: str}
    x_axis: Optional[Dict[str, Any]] = None  # {label: str, format: str, rotate: int}
    y_axis: Optional[Dict[str, Any]] = None  # {label: str, format: str}
    
    # Colors
    colors: List[str] = field(default_factory=list)  # Custom colors for series
    color_scheme: Optional[str] = None  # Predefined color scheme


@dataclass
class IRDataChart:
    """
    IR specification for advanced data chart component.
    
    Runtime-agnostic representation of multi-series charts with
    comprehensive configuration for legend, tooltip, axes, colors.
    """
    title: str
    source_type: str  # "dataset" | "table" | "frame"
    source: str
    
    # Chart configuration
    config: Optional[IRChartConfig] = None
    
    # Filtering and sorting
    filter_by: Optional[str] = None
    sort_by: Optional[str] = None
    
    # Empty state
    empty_state: Optional[Dict[str, Any]] = None
    
    # Data binding
    binding: Optional[DataBindingSpec] = None
    
    # Styling
    layout: Optional[Dict[str, Any]] = None
    style: Optional[Dict[str, Any]] = None
    height: Optional[Union[str, int]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Layout Primitive IR Specifications
# =============================================================================

@dataclass
class IRStackLayout:
    """
    IR specification for stack layout.
    
    Runtime-agnostic representation of flexbox-like linear layouts.
    """
    direction: str = "vertical"  # "vertical" | "horizontal"
    gap: Union[str, int] = "medium"  # "small" | "medium" | "large" | px value
    align: str = "stretch"  # "start" | "center" | "end" | "stretch"
    justify: str = "start"  # "start" | "center" | "end" | "space_between" | "space_around" | "space_evenly"
    wrap: bool = False
    
    children: List[Union[ComponentSpec, 'IRStackLayout', 'IRGridLayout', 'IRSplitLayout', 'IRTabsLayout', 'IRAccordionLayout']] = field(default_factory=list)
    
    style: Dict[str, Any] = field(default_factory=dict)
    layout_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRGridLayout:
    """
    IR specification for grid layout.
    
    Runtime-agnostic representation of CSS Grid layouts.
    """
    columns: Union[int, str] = "auto"  # Number or "auto"
    min_column_width: Optional[str] = None  # "200px" | "12rem" | token
    gap: Union[str, int] = "medium"
    responsive: bool = True
    
    children: List[Union[ComponentSpec, 'IRStackLayout', 'IRGridLayout', 'IRSplitLayout', 'IRTabsLayout', 'IRAccordionLayout']] = field(default_factory=list)
    
    style: Dict[str, Any] = field(default_factory=dict)
    layout_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRSplitLayout:
    """
    IR specification for split pane layout.
    
    Runtime-agnostic representation of resizable split layouts.
    """
    left: List[Union[ComponentSpec, 'IRStackLayout', 'IRGridLayout', 'IRSplitLayout', 'IRTabsLayout', 'IRAccordionLayout']] = field(default_factory=list)
    right: List[Union[ComponentSpec, 'IRStackLayout', 'IRGridLayout', 'IRSplitLayout', 'IRTabsLayout', 'IRAccordionLayout']] = field(default_factory=list)
    ratio: float = 0.5  # 0.0 to 1.0
    resizable: bool = False
    orientation: str = "horizontal"  # "horizontal" | "vertical"
    
    style: Dict[str, Any] = field(default_factory=dict)
    layout_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRTabItem:
    """IR specification for a single tab."""
    id: str
    label: str
    icon: Optional[str] = None
    badge: Optional[Union[str, Dict[str, Any]]] = None
    content: List[Union[ComponentSpec, 'IRStackLayout', 'IRGridLayout', 'IRSplitLayout', 'IRTabsLayout', 'IRAccordionLayout']] = field(default_factory=list)


@dataclass
class IRTabsLayout:
    """
    IR specification for tabbed interface.
    
    Runtime-agnostic representation of tab-based navigation.
    """
    tabs: List[IRTabItem] = field(default_factory=list)
    default_tab: Optional[str] = None
    persist_state: bool = True  # Persist in URL or state
    
    style: Dict[str, Any] = field(default_factory=dict)
    layout_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRAccordionItem:
    """IR specification for a single accordion item."""
    id: str
    title: str
    description: Optional[str] = None
    icon: Optional[str] = None
    default_open: bool = False
    content: List[Union[ComponentSpec, 'IRStackLayout', 'IRGridLayout', 'IRSplitLayout', 'IRTabsLayout', 'IRAccordionLayout']] = field(default_factory=list)


@dataclass
class IRAccordionLayout:
    """
    IR specification for accordion/collapsible sections.
    
    Runtime-agnostic representation of collapsible content panels.
    """
    items: List[IRAccordionItem] = field(default_factory=list)
    multiple: bool = False  # Allow multiple items expanded
    
    style: Dict[str, Any] = field(default_factory=dict)
    layout_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteSpec:
    """Frontend route specification"""
    path: str
    page_ref: str
    auth_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PageSpec:
    """Frontend page specification"""
    name: str
    slug: str
    title: str
    components: List[ComponentSpec] = field(default_factory=list)
    layout: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrontendIR:
    """
    Complete frontend intermediate representation.
    
    Runtime adapters consume this to generate static sites, React apps,
    Vue apps, or any other frontend implementation.
    """
    app_name: str
    app_version: str = "1.0.0"
    
    pages: List[PageSpec] = field(default_factory=list)
    routes: List[RouteSpec] = field(default_factory=list)
    
    # Configuration
    theme: Dict[str, Any] = field(default_factory=dict)
    api_base_url: str = "/api"
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    ir_version: str = "0.1.0"
