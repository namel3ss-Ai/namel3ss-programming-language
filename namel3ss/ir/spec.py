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

# Import security types for IR metadata
try:
    from namel3ss.ast.security import PermissionLevel, SecurityPolicy
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False
    PermissionLevel = None
    SecurityPolicy = None


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
class InsightSpec:
    """Insight specification"""
    name: str
    query: str
    dataset_ref: str
    aggregations: List[Dict[str, Any]] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)
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
    
    # Data components
    datasets: List[DatasetSpec] = field(default_factory=list)
    frames: List[FrameSpec] = field(default_factory=list)
    insights: List[InsightSpec] = field(default_factory=list)
    
    # Data binding & realtime
    update_channels: List[UpdateChannelSpec] = field(default_factory=list)
    
    # State management
    memory: List[MemorySpec] = field(default_factory=list)
    
    # Configuration
    database_config: Optional[Dict[str, Any]] = None
    auth_config: Optional[Dict[str, Any]] = None
    cors_config: Optional[Dict[str, Any]] = None
    realtime_config: Optional[Dict[str, Any]] = None  # WebSocket/Redis configuration
    
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
