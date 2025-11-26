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

# Import design tokens for IR-level type safety
try:
    from namel3ss.ast.design_tokens import (
        VariantType,
        ToneType,
        DensityType,
        SizeType,
        ThemeType,
        ColorSchemeType,
    )
    HAS_DESIGN_TOKENS = True
except ImportError:
    HAS_DESIGN_TOKENS = False
    VariantType = None
    ToneType = None
    DensityType = None
    SizeType = None
    ThemeType = None
    ColorSchemeType = None

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
# DESIGN TOKENS (IR Level)
# =============================================================================

@dataclass
class ComponentDesignTokensIR:
    """
    Design tokens for UI components at the IR level.
    
    These are runtime-agnostic and will be mapped to concrete design system
    implementations (Tailwind, Material-UI, etc.) during codegen.
    """
    variant: Optional[str] = None  # "elevated" | "outlined" | "ghost" | "subtle"
    tone: Optional[str] = None  # "neutral" | "primary" | "success" | "warning" | "danger"
    density: Optional[str] = None  # "comfortable" | "compact"
    size: Optional[str] = None  # "xs" | "sm" | "md" | "lg" | "xl"
    theme: Optional[str] = None  # "light" | "dark" | "system" (component-level override)
    color_scheme: Optional[str] = None  # "blue" | "green" | "violet" | etc. (component-level override)


@dataclass
class AppLevelDesignTokensIR:
    """
    App-level design tokens at the IR level.
    
    These affect the entire application and cascade to child components.
    """
    theme: Optional[str] = None  # "light" | "dark" | "system"
    color_scheme: Optional[str] = None  # "blue" | "green" | "violet" | "rose" | etc.


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
    
    # Frontend IR (pages, routes, UI components)
    frontend: Optional['FrontendIR'] = None
    
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
# Form IR Specifications
# =============================================================================

@dataclass
class IRFormField:
    """
    Runtime-agnostic form field specification.
    
    Represents a single form field with validation, bindings, and
    conditional rendering logic.
    """
    name: str
    component: str  # Field type: text_input, select, textarea, checkbox, etc.
    label: Optional[str] = None
    placeholder: Optional[str] = None
    help_text: Optional[str] = None
    required: bool = False
    
    # Initial/default values (as expressions that runtime can evaluate)
    default_value: Optional[str] = None  # Expression string
    initial_value: Optional[str] = None  # Expression string
    
    # Validation rules
    validation: Dict[str, Any] = field(default_factory=dict)  # Contains min_length, max_length, pattern, min_value, max_value, step
    
    # Options for select/multiselect/radio_group
    options_binding: Optional[str] = None  # Bind to dataset or provider
    static_options: List[Dict[str, Any]] = field(default_factory=list)  # Fallback static options
    
    # Conditional rendering (as expression strings)
    disabled_expr: Optional[str] = None
    visible_expr: Optional[str] = None
    
    # Component-specific configuration
    component_config: Dict[str, Any] = field(default_factory=dict)  # multiple, accept, max_file_size, etc.
    
    # Field-level design tokens
    design_tokens: Optional[ComponentDesignTokensIR] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRForm:
    """
    Runtime-agnostic form specification.
    
    Captures complete form definition with fields, actions, bindings,
    and validation rules for any runtime to implement.
    """
    name: str  # Derived from title or context
    title: str
    fields: List[IRFormField]
    
    # Layout
    layout_mode: str = "vertical"  # "vertical" | "horizontal" | "inline"
    
    # Action integration
    submit_action: Optional[str] = None  # Action name/reference
    submit_action_type: str = "custom"  # "custom" | "create" | "update" | "delete"
    submit_endpoint: Optional[str] = None  # HTTP endpoint if applicable
    
    # Data bindings
    initial_values_binding: Optional[str] = None  # Dataset/query reference
    initial_values_expr: Optional[str] = None  # Expression for loading data
    
    # Form-level configuration
    validation_mode: str = "on_blur"  # "on_blur" | "on_change" | "on_submit"
    submit_button_text: str = "Submit"
    reset_button: bool = False
    
    # User feedback
    loading_text: Optional[str] = None
    success_message: Optional[str] = None
    error_message: Optional[str] = None
    
    # Validation schema (generated from fields)
    validation_schema: Dict[str, Any] = field(default_factory=dict)
    
    # Security
    requires_auth: bool = False
    required_permission_level: Optional[str] = None
    
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
    
    # Design tokens
    design_tokens: Optional[ComponentDesignTokensIR] = None
    
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
    
    # Design tokens
    design_tokens: Optional[ComponentDesignTokensIR] = None
    
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


# =============================================================================
# NAVIGATION & APP CHROME IR SPECIFICATIONS
# =============================================================================


@dataclass
class IRNavItem:
    """IR specification for navigation item."""
    id: str
    label: str
    route: Optional[str] = None
    icon: Optional[str] = None
    badge: Optional[Dict[str, Any]] = None
    action: Optional[str] = None  # Action ID to trigger
    condition: Optional[str] = None  # Conditional visibility expression
    children: List['IRNavItem'] = field(default_factory=list)  # Nested navigation
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRNavSection:
    """IR specification for navigation section grouping."""
    id: str
    label: str
    items: List[str] = field(default_factory=list)  # Nav item IDs
    collapsible: bool = False
    collapsed_by_default: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRSidebar:
    """
    IR specification for sidebar navigation.
    
    Runtime-agnostic representation of sidebar with:
    - Hierarchical navigation items
    - Section grouping
    - Collapsible behavior
    - Icon and badge support
    """
    items: List[IRNavItem] = field(default_factory=list)
    sections: List[IRNavSection] = field(default_factory=list)
    collapsible: bool = False
    collapsed_by_default: bool = False
    width: Optional[str] = None  # "narrow" | "normal" | "wide" | px/rem value
    position: str = "left"  # "left" | "right"
    
    # Validate routes exist
    validated_routes: List[str] = field(default_factory=list)  # Routes that were validated
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRNavbarAction:
    """IR specification for navbar action button."""
    id: str
    label: Optional[str] = None
    icon: Optional[str] = None
    type: str = "button"  # "button" | "menu" | "toggle"
    action: Optional[str] = None  # Action ID to trigger
    menu_items: List[IRNavItem] = field(default_factory=list)  # For type="menu"
    condition: Optional[str] = None  # Conditional visibility
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRNavbar:
    """
    IR specification for application navbar/topbar.
    
    Runtime-agnostic representation of top-level navigation with:
    - Branding (logo, title)
    - Global actions (user menu, theme toggle, etc.)
    - Responsive behavior
    """
    logo: Optional[str] = None  # Asset reference
    title: Optional[str] = None  # App title or expression
    actions: List[IRNavbarAction] = field(default_factory=list)
    position: str = "top"  # "top" | "bottom"
    sticky: bool = True
    
    # Validate actions exist
    validated_actions: List[str] = field(default_factory=list)  # Actions that were validated
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRBreadcrumbItem:
    """IR specification for single breadcrumb item."""
    label: str  # Can be expression
    route: Optional[str] = None  # If None, renders as text
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRBreadcrumbs:
    """
    IR specification for breadcrumb navigation.
    
    Runtime-agnostic representation of breadcrumb trail with:
    - Explicit items
    - Auto-derivation from routing
    """
    items: List[IRBreadcrumbItem] = field(default_factory=list)
    auto_derive: bool = False  # Auto-generate from route hierarchy
    separator: str = "/"
    
    # For auto-derive, this is populated at IR build time
    derived_from_route: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRCommandSource:
    """IR specification for command palette source."""
    type: str = "routes"  # "routes" | "actions" | "custom" | "api"
    filter: Optional[str] = None  # Filter expression
    custom_items: List[Dict[str, Any]] = field(default_factory=list)
    # API-backed source fields
    id: Optional[str] = None  # Unique identifier
    endpoint: Optional[str] = None  # API endpoint URL
    label: Optional[str] = None  # Display label
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRCommandPalette:
    """
    IR specification for command palette.
    
    Runtime-agnostic representation of power-user interface with:
    - Keyboard shortcut activation
    - Command search and filtering
    - Integration with routes and actions registry
    """
    shortcut: str = "ctrl+k"  # Keyboard shortcut
    sources: List[IRCommandSource] = field(default_factory=list)
    placeholder: str = "Search commands..."
    max_results: int = 10
    
    # Populated at IR build time from routes and actions
    available_routes: List[Dict[str, str]] = field(default_factory=list)  # [{label, path}]
    available_actions: List[Dict[str, str]] = field(default_factory=list)  # [{label, id}]
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# FEEDBACK COMPONENTS IR (Modal, Toast)
# ============================================================


@dataclass
class IRModalAction:
    """IR specification for modal action button."""
    label: str
    action: Optional[str] = None  # Action name to trigger
    variant: str = "default"  # "default" | "primary" | "destructive" | "ghost" | "link"
    close: bool = True  # Whether clicking closes modal
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRModal:
    """
    IR specification for modal dialog.
    
    Runtime-agnostic representation of modal overlay with:
    - Configurable size and dismissibility
    - Nested content (any statements)
    - Action buttons with variants
    - Trigger integration with actions
    """
    id: str  # Unique identifier
    title: str
    description: Optional[str] = None
    content: List[Any] = field(default_factory=list)  # Nested IR components
    actions: List[IRModalAction] = field(default_factory=list)
    size: str = "md"  # "sm" | "md" | "lg" | "xl" | "full"
    dismissible: bool = True
    trigger: Optional[str] = None  # Action name that opens modal
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRToast:
    """
    IR specification for toast notification.
    
    Runtime-agnostic representation of temporary notification with:
    - Variant styling (success, error, warning, info)
    - Auto-dismiss timing
    - Optional action button
    - Flexible positioning
    """
    id: str  # Unique identifier
    title: str
    description: Optional[str] = None
    variant: str = "default"  # "default" | "success" | "error" | "warning" | "info"
    duration: int = 3000  # ms (0 = manual dismiss only)
    action_label: Optional[str] = None
    action: Optional[str] = None
    position: str = "top-right"  # "top" | "top-right" | "top-left" | "bottom" | "bottom-right" | "bottom-left"
    trigger: Optional[str] = None  # Action name that shows toast
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# AI Semantic Components IR
# =============================================================================

@dataclass
class IRChatThread:
    """
    IR specification for multi-message AI conversation display.
    
    Runtime-agnostic representation capturing:
    - Message list binding to real conversation data
    - Grouping and display preferences
    - Streaming configuration
    - Interaction capabilities
    """
    id: str
    messages_binding: str  # e.g., "conversation.messages", "agent.chat_history"
    group_by: str = "role"  # "role" | "speaker" | "timestamp" | "none"
    show_timestamps: bool = True
    show_avatar: bool = True
    reverse_order: bool = False
    auto_scroll: bool = True
    max_height: Optional[str] = None
    # Streaming
    streaming_enabled: bool = False
    streaming_source: Optional[str] = None
    # Display
    show_role_labels: bool = True
    show_token_count: bool = False
    enable_copy: bool = True
    enable_regenerate: bool = False
    variant: str = "default"  # "default" | "compact" | "detailed"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRAgentPanel:
    """
    IR specification for agent state and metrics display.
    
    Runtime-agnostic representation of agent information including:
    - Agent identification and status
    - Performance metrics (tokens, cost, latency)
    - Environment and configuration
    - Tool availability
    """
    id: str
    agent_binding: str  # e.g., "current_agent", "agent.researcher"
    metrics_binding: Optional[str] = None
    # Display flags
    show_status: bool = True
    show_metrics: bool = True
    show_profile: bool = False
    show_limits: bool = False
    show_last_error: bool = False
    show_tools: bool = False
    # Metrics
    show_tokens: bool = True
    show_cost: bool = True
    show_latency: bool = True
    show_model: bool = True
    # Layout
    variant: str = "card"  # "card" | "inline" | "sidebar"
    compact: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRToolCallView:
    """
    IR specification for tool invocation display.
    
    Runtime-agnostic representation of tool calls with:
    - Input/output visibility
    - Timing and status information
    - Filtering capabilities
    - Interaction features (retry, copy)
    """
    id: str
    calls_binding: str  # e.g., "run.tool_calls", "agent.tools_used"
    # Display
    show_inputs: bool = True
    show_outputs: bool = True
    show_timing: bool = True
    show_status: bool = True
    show_raw_payload: bool = False
    # Filtering
    filter_tool_name: Optional[List[str]] = None
    filter_status: Optional[List[str]] = None
    # Layout
    variant: str = "list"  # "list" | "table" | "timeline"
    expandable: bool = True
    max_height: Optional[str] = None
    # Interaction
    enable_retry: bool = False
    enable_copy: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRLogView:
    """
    IR specification for log/trace display.
    
    Runtime-agnostic representation of log viewing with:
    - Log entry binding to real logging system
    - Level and search filtering
    - Auto-refresh and tailing
    - Virtualization for performance
    """
    id: str
    logs_binding: str  # e.g., "run.logs", "agent.traces", "app.logs"
    # Filtering
    level_filter: Optional[List[str]] = None
    search_enabled: bool = True
    search_placeholder: str = "Search logs..."
    # Display
    show_timestamp: bool = True
    show_level: bool = True
    show_metadata: bool = False
    show_source: bool = False
    # Behavior
    auto_scroll: bool = True
    auto_refresh: bool = False
    refresh_interval: int = 5000
    max_entries: int = 1000
    # Layout
    variant: str = "default"  # "default" | "compact" | "detailed"
    max_height: Optional[str] = None
    virtualized: bool = True
    # Interaction
    enable_copy: bool = True
    enable_download: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IREvaluationResult:
    """
    IR specification for evaluation results display.
    
    Runtime-agnostic representation of eval run analysis:
    - Aggregate metrics and distributions
    - Error analysis and examples
    - Comparison capabilities
    - Filtering and drill-down
    """
    id: str
    eval_run_binding: str  # e.g., "eval.run_123", "latest_eval"
    # Display
    show_summary: bool = True
    show_histograms: bool = True
    show_error_table: bool = True
    show_metadata: bool = False
    # Metrics
    metrics_to_show: Optional[List[str]] = None
    primary_metric: Optional[str] = None
    # Filtering
    filter_metric: Optional[str] = None
    filter_min_score: Optional[float] = None
    filter_max_score: Optional[float] = None
    filter_status: Optional[List[str]] = None
    # Error analysis
    show_error_distribution: bool = True
    show_error_examples: bool = True
    max_error_examples: int = 10
    # Layout
    variant: str = "dashboard"  # "dashboard" | "detailed" | "compact"
    # Comparison
    comparison_run_binding: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRDiffView:
    """
    IR specification for text/code diff display.
    
    Runtime-agnostic representation of side-by-side or unified diffs:
    - Left/right content bindings
    - Diff algorithm configuration
    - Syntax highlighting for code
    - Display and interaction options
    """
    id: str
    left_binding: str  # e.g., "version.v1.output", "prompt.original"
    right_binding: str  # e.g., "version.v2.output", "prompt.modified"
    # Display
    mode: str = "split"  # "unified" | "split"
    content_type: str = "text"  # "text" | "code" | "markdown"
    language: Optional[str] = None
    # Diff options
    ignore_whitespace: bool = False
    ignore_case: bool = False
    context_lines: int = 3
    # Display options
    show_line_numbers: bool = True
    highlight_inline_changes: bool = True
    show_legend: bool = True
    # Layout
    max_height: Optional[str] = None
    # Interaction
    enable_copy: bool = True
    enable_download: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    
    # App-level design tokens (cascade to children)
    design_tokens: Optional[AppLevelDesignTokensIR] = None


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
    
    # App-level design tokens
    design_tokens: Optional[AppLevelDesignTokensIR] = None
    api_base_url: str = "/api"
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    ir_version: str = "0.1.0"
