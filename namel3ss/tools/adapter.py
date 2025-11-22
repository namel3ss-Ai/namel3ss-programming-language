"""
Production-grade Tool Adapter Framework for Namel3ss.

This module defines the core ToolAdapter protocol that all tool implementations
must conform to. It provides a runtime- and codegen-friendly abstraction layer
between N3 DSL/AST and tool execution.

The framework is designed to be:
- Language-agnostic at the conceptual level (extensible to other languages)
- Idiomatic in Python (using protocols, type hints, async support)
- Friendly for code generation (machine-readable metadata and schemas)
- Production-ready (observability, error handling, validation)

Architecture:
    ToolAdapter defines the contract for all tools
    ↓
    Concrete implementations (HTTP, Python, LangChain, OpenAPI, LLM)
    ↓
    Runtime execution with tracing, metrics, error handling

Key Components:
    - ToolAdapter: Core protocol/interface
    - ToolMetadata: Structured metadata (name, version, tags, etc.)
    - ToolConfig: Per-tool configuration (timeouts, retries, auth)
    - ToolContext: Execution context (logger, tracer, correlation IDs)
    - SyncToolAdapter/AsyncToolAdapter: Sync/async execution modes
    - StreamingToolAdapter: Streaming response support

Example:
    from namel3ss.tools.adapter import ToolAdapter, ToolMetadata, ToolConfig
    from namel3ss.tools.schemas import ToolInputModel, ToolOutputModel
    from pydantic import BaseModel
    
    class WeatherInput(ToolInputModel):
        location: str
        units: str = "metric"
    
    class WeatherOutput(ToolOutputModel):
        temperature: float
        condition: str
    
    class WeatherTool(ToolAdapter[WeatherInput, WeatherOutput]):
        def get_metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="weather",
                description="Get current weather",
                version="1.0.0",
                tags=["weather", "api"]
            )
        
        def get_input_schema(self) -> type[WeatherInput]:
            return WeatherInput
        
        def get_output_schema(self) -> type[WeatherOutput]:
            return WeatherOutput
        
        async def invoke(self, input: WeatherInput, context: ToolContext) -> WeatherOutput:
            # Implementation
            return WeatherOutput(temperature=72.0, condition="sunny")

Thread Safety:
    ToolAdapter instances should be thread-safe for invocation.
    Configuration should be immutable after initialization.

Performance:
    - Adapters should be reusable (avoid per-call initialization)
    - Use connection pooling for HTTP/DB tools
    - Cache metadata and schemas after first access
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel

from namel3ss.tools.errors import ToolError

# Optional: OpenTelemetry tracing support
try:
    from namel3ss.features import has_opentelemetry
    if has_opentelemetry():
        from opentelemetry import trace
        from opentelemetry.trace import Tracer, Span
        _HAS_OTEL = True
    else:
        _HAS_OTEL = False
        Tracer = Any  # type: ignore
        Span = Any  # type: ignore
except ImportError:
    _HAS_OTEL = False
    Tracer = Any  # type: ignore
    Span = Any  # type: ignore


# Type variables for input/output schemas
TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)
TChunk = TypeVar("TChunk", bound=BaseModel)


class ToolCategory(str, Enum):
    """Standard tool categories for classification."""
    
    HTTP = "http"
    PYTHON = "python"
    DATABASE = "database"
    VECTOR_SEARCH = "vector_search"
    LLM = "llm"
    LANGCHAIN = "langchain"
    OPENAPI = "openapi"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ToolMetadata:
    """
    Immutable metadata for a tool adapter.
    
    This metadata is used by:
    - Runtime: Tool discovery, validation, execution
    - Codegen: SDK generation, client stubs, OpenAPI specs
    - UI: Tool catalog, documentation, search
    - Observability: Tracing, metrics, logging
    
    Attributes:
        name: Unique tool identifier (e.g., "weather_api", "calculate_sum")
        description: Human-readable description (used in prompts, docs)
        version: Semantic version (e.g., "1.0.0", "2.1.3")
        category: Tool category for classification
        tags: Additional labels for search/filtering
        author: Tool author/maintainer
        source: Source system (e.g., "openapi", "langchain", "custom")
        namespace: Logical namespace for grouping related tools
    
    Design Notes:
        - Frozen dataclass ensures immutability
        - All fields are JSON-serializable for codegen
        - Name should be unique within a namespace
        - Version follows semver for compatibility tracking
    
    Example:
        >>> metadata = ToolMetadata(
        ...     name="weather",
        ...     description="Get current weather conditions",
        ...     version="1.0.0",
        ...     category=ToolCategory.HTTP,
        ...     tags=["weather", "api", "external"],
        ...     author="Platform Team",
        ...     source="openapi",
        ...     namespace="integrations.weather"
        ... )
    """
    
    name: str
    description: str
    version: str
    category: ToolCategory = ToolCategory.CUSTOM
    tags: tuple[str, ...] = field(default_factory=tuple)
    author: Optional[str] = None
    source: Optional[str] = None
    namespace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category.value,
            "tags": list(self.tags),
            "author": self.author,
            "source": self.source,
            "namespace": self.namespace,
        }
    
    def get_full_name(self) -> str:
        """Get fully qualified name (namespace.name)."""
        if self.namespace:
            return f"{self.namespace}.{self.name}"
        return self.name


@dataclass
class ToolConfig:
    """
    Per-tool configuration for execution behavior.
    
    Configuration covers:
    - Timeouts and retries
    - Authentication and authorization
    - Rate limiting
    - Caching
    - Custom parameters
    
    Attributes:
        timeout: Maximum execution time in seconds
        max_retries: Number of retry attempts on failure
        retry_delay: Delay between retries in seconds
        auth_token: Bearer token or API key
        auth_header: Custom auth header name
        rate_limit_per_minute: Max calls per minute (None = unlimited)
        cache_ttl: Cache TTL in seconds (None = no caching)
        endpoints: Custom endpoint URLs (for HTTP tools)
        headers: Custom HTTP headers
        custom: Tool-specific configuration
    
    Design Notes:
        - Mutable for runtime configuration updates
        - All fields optional with sensible defaults
        - Custom dict allows tool-specific parameters
        - Thread-safe reads; external synchronization needed for writes
    
    Example:
        >>> config = ToolConfig(
        ...     timeout=30.0,
        ...     max_retries=3,
        ...     auth_token="sk-...",
        ...     rate_limit_per_minute=60,
        ...     cache_ttl=300,
        ...     custom={"region": "us-west-2", "debug": True}
        ... )
    """
    
    timeout: float = 30.0
    max_retries: int = 0
    retry_delay: float = 1.0
    auth_token: Optional[str] = None
    auth_header: Optional[str] = None
    rate_limit_per_minute: Optional[int] = None
    cache_ttl: Optional[int] = None
    endpoints: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get custom configuration value."""
        return self.custom.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set custom configuration value."""
        self.custom[key] = value


@dataclass
class ToolContext:
    """
    Execution context passed to tool adapters.
    
    Provides tools with access to:
    - Logging infrastructure
    - Distributed tracing (OpenTelemetry)
    - Correlation IDs for request tracking
    - User/session context
    - Environment variables
    
    Attributes:
        logger: Python logger instance for tool output
        tracer: OpenTelemetry tracer for distributed tracing
        span: Current trace span (if active)
        correlation_id: Request correlation ID
        user_id: User identifier (if authenticated)
        session_id: Session identifier (if applicable)
        environment: Environment variables/config
        timestamp: Context creation timestamp
    
    Design Notes:
        - Passed to every tool invocation
        - Enables observability without coupling to specific systems
        - Optional fields allow progressive adoption
        - Immutable after creation
    
    Example:
        >>> from opentelemetry import trace
        >>> tracer = trace.get_tracer(__name__)
        >>> context = ToolContext(
        ...     logger=logging.getLogger("tools"),
        ...     tracer=tracer,
        ...     correlation_id="req-123",
        ...     user_id="user-456",
        ...     environment={"region": "us-west-2"}
        ... )
    """
    
    logger: logging.Logger
    tracer: Optional[Tracer] = None
    span: Optional[Span] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    
    def child_span(self, name: str) -> Span:
        """Create child span for nested operations."""
        if not _HAS_OTEL:
            # Return no-op if OpenTelemetry not available
            return None  # type: ignore
        
        if self.tracer:
            if self.span:
                with trace.use_span(self.span):
                    return self.tracer.start_span(name)
            else:
                return self.tracer.start_span(name)
        # Return no-op span if tracer not available
        return trace.get_tracer(__name__).start_span(name)
    
    def log(self, level: str, message: str, **kwargs: Any) -> None:
        """Log message with context."""
        extra = {
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            **kwargs,
        }
        getattr(self.logger, level)(message, extra=extra)


@runtime_checkable
class ToolAdapter(Protocol, Generic[TInput, TOutput]):
    """
    Core protocol for all tool adapters.
    
    This protocol defines the contract that all tool implementations must satisfy.
    It is runtime-checkable, allowing isinstance() checks.
    
    Key Methods:
        get_metadata(): Return tool metadata
        get_input_schema(): Return input model class
        get_output_schema(): Return output model class
        get_config(): Return tool configuration
        invoke(): Execute tool with input and context
    
    Design Principles:
        - Protocol-based (structural subtyping, no inheritance required)
        - Generic over input/output types for type safety
        - Async by default (sync adapters use wrappers)
        - Observable (context provides logger, tracer)
        - Validated (Pydantic models for I/O)
    
    Type Safety:
        The generic parameters TInput and TOutput ensure type safety:
        - Input must be a Pydantic BaseModel
        - Output must be a Pydantic BaseModel
        - Type checkers can validate input/output usage
    
    Example:
        >>> class MyTool(ToolAdapter[MyInput, MyOutput]):
        ...     def get_metadata(self) -> ToolMetadata:
        ...         return ToolMetadata(name="my_tool", description="...", version="1.0.0")
        ...     
        ...     def get_input_schema(self) -> type[MyInput]:
        ...         return MyInput
        ...     
        ...     def get_output_schema(self) -> type[MyOutput]:
        ...         return MyOutput
        ...     
        ...     def get_config(self) -> ToolConfig:
        ...         return ToolConfig(timeout=30.0)
        ...     
        ...     async def invoke(self, input: MyInput, context: ToolContext) -> MyOutput:
        ...         # Implementation
        ...         return MyOutput(...)
    
    Runtime Validation:
        >>> tool: ToolAdapter = MyTool()
        >>> assert isinstance(tool, ToolAdapter)  # Protocol check
    """
    
    def get_metadata(self) -> ToolMetadata:
        """
        Get tool metadata.
        
        Metadata is used for:
        - Tool discovery and cataloging
        - Code generation (SDK clients, OpenAPI specs)
        - UI rendering (tool palette, documentation)
        - Observability (tracing, metrics)
        
        Returns:
            ToolMetadata with name, description, version, etc.
        
        Implementation Notes:
            - Should return same metadata on every call (can cache)
            - Metadata should be immutable
            - Name must be unique within namespace
        """
        ...
    
    def get_input_schema(self) -> type[TInput]:
        """
        Get input schema as Pydantic model class.
        
        The input schema defines:
        - Required and optional fields
        - Field types and validation rules
        - Field descriptions (for prompts/docs)
        - Default values
        
        Returns:
            Pydantic BaseModel class representing input structure
        
        Implementation Notes:
            - Return the class itself, not an instance
            - Schema is used for validation before invoke()
            - Can be exported to JSON Schema for codegen
        
        Example:
            >>> class MyInput(ToolInputModel):
            ...     query: str
            ...     limit: int = 10
            >>> 
            >>> def get_input_schema(self) -> type[MyInput]:
            ...     return MyInput
        """
        ...
    
    def get_output_schema(self) -> type[TOutput]:
        """
        Get output schema as Pydantic model class.
        
        The output schema defines:
        - Return value structure
        - Field types
        - Field descriptions
        
        Returns:
            Pydantic BaseModel class representing output structure
        
        Implementation Notes:
            - Return the class itself, not an instance
            - Schema is used for validation after invoke()
            - Can be exported to JSON Schema for codegen
        """
        ...
    
    def get_config(self) -> ToolConfig:
        """
        Get tool configuration.
        
        Configuration includes:
        - Timeouts and retries
        - Authentication credentials
        - Rate limits
        - Custom parameters
        
        Returns:
            ToolConfig instance
        
        Implementation Notes:
            - Can return new instance each time or cached instance
            - Config may be mutable for runtime updates
            - Sensitive values (auth tokens) should be handled securely
        """
        ...
    
    async def invoke(
        self,
        input: TInput,
        context: ToolContext,
    ) -> TOutput:
        """
        Execute the tool with validated input.
        
        This is the main execution method. It:
        1. Receives validated input (Pydantic model)
        2. Performs the tool's operation
        3. Returns validated output (Pydantic model)
        
        Args:
            input: Validated input model
            context: Execution context (logger, tracer, etc.)
        
        Returns:
            Output model with results
        
        Raises:
            ToolError: For tool-specific errors
            ToolExecutionError: For runtime execution errors
            ToolValidationError: For input/output validation errors
            TimeoutError: If execution exceeds timeout
        
        Implementation Guidelines:
            - Use context.logger for logging
            - Use context.tracer for distributed tracing
            - Respect config.timeout
            - Implement retries based on config.max_retries
            - Validate output before returning
            - Wrap external exceptions in ToolError
        
        Example:
            >>> async def invoke(self, input: MyInput, context: ToolContext) -> MyOutput:
            ...     context.log("info", f"Processing {input.query}")
            ...     
            ...     with context.child_span("external_api_call"):
            ...         result = await call_external_api(input.query)
            ...     
            ...     return MyOutput(result=result)
        """
        ...


class BaseToolAdapter(ABC, Generic[TInput, TOutput]):
    """
    Abstract base class for tool adapters.
    
    Provides common functionality for tool implementations:
    - Metadata caching
    - Configuration management
    - Error handling helpers
    - Validation wrappers
    
    Subclasses should:
    1. Define input/output Pydantic models
    2. Implement get_metadata()
    3. Implement invoke()
    
    Example:
        >>> class CalculatorTool(BaseToolAdapter[CalculatorInput, CalculatorOutput]):
        ...     def get_metadata(self) -> ToolMetadata:
        ...         return ToolMetadata(
        ...             name="calculator",
        ...             description="Perform calculations",
        ...             version="1.0.0",
        ...             category=ToolCategory.PYTHON
        ...         )
        ...     
        ...     def get_input_schema(self) -> type[CalculatorInput]:
        ...         return CalculatorInput
        ...     
        ...     def get_output_schema(self) -> type[CalculatorOutput]:
        ...         return CalculatorOutput
        ...     
        ...     async def invoke(
        ...         self,
        ...         input: CalculatorInput,
        ...         context: ToolContext
        ...     ) -> CalculatorOutput:
        ...         result = eval(input.expression)
        ...         return CalculatorOutput(result=result)
    """
    
    def __init__(self, config: Optional[ToolConfig] = None):
        """
        Initialize adapter with optional configuration.
        
        Args:
            config: Tool configuration (uses defaults if None)
        """
        self._config = config or ToolConfig()
        self._metadata_cache: Optional[ToolMetadata] = None
    
    @abstractmethod
    def get_metadata(self) -> ToolMetadata:
        """Get tool metadata (implemented by subclasses)."""
        ...
    
    @abstractmethod
    def get_input_schema(self) -> type[TInput]:
        """Get input schema (implemented by subclasses)."""
        ...
    
    @abstractmethod
    def get_output_schema(self) -> type[TOutput]:
        """Get output schema (implemented by subclasses)."""
        ...
    
    def get_config(self) -> ToolConfig:
        """Get tool configuration."""
        return self._config
    
    @abstractmethod
    async def invoke(
        self,
        input: TInput,
        context: ToolContext,
    ) -> TOutput:
        """Execute tool (implemented by subclasses)."""
        ...
    
    def validate_input(self, data: Dict[str, Any]) -> TInput:
        """
        Validate and parse input data.
        
        Args:
            data: Raw input dictionary
        
        Returns:
            Validated input model
        
        Raises:
            ToolValidationError: If validation fails
        """
        from namel3ss.tools.errors import ToolValidationError
        
        try:
            input_class = self.get_input_schema()
            return input_class(**data)
        except Exception as e:
            metadata = self.get_metadata()
            raise ToolValidationError(
                f"Input validation failed: {e}",
                code="TOOL003",
                tool_name=metadata.name,
                original_error=e,
            ) from e
    
    def validate_output(self, data: Any) -> TOutput:
        """
        Validate and parse output data.
        
        Args:
            data: Raw output data
        
        Returns:
            Validated output model
        
        Raises:
            ToolValidationError: If validation fails
        """
        from namel3ss.tools.errors import ToolValidationError
        
        try:
            output_class = self.get_output_schema()
            if isinstance(data, output_class):
                return data
            elif isinstance(data, dict):
                return output_class(**data)
            else:
                return output_class(result=data)
        except Exception as e:
            metadata = self.get_metadata()
            raise ToolValidationError(
                f"Output validation failed: {e}",
                code="TOOL004",
                tool_name=metadata.name,
                original_error=e,
            ) from e


@runtime_checkable
class StreamingToolAdapter(Protocol, Generic[TInput, TChunk]):
    """
    Protocol for streaming tool adapters.
    
    Streaming tools return results incrementally as they become available,
    rather than waiting for complete execution. This is useful for:
    - LLM completions (token-by-token streaming)
    - Large file processing (chunk-by-chunk)
    - Real-time data feeds
    - Long-running operations with progress updates
    
    Methods:
        invoke_stream(): Returns AsyncIterator of chunks
        
    Example:
        >>> class StreamingLLMTool(StreamingToolAdapter[LLMInput, LLMChunk]):
        ...     async def invoke_stream(
        ...         self,
        ...         input: LLMInput,
        ...         context: ToolContext
        ...     ) -> AsyncIterator[LLMChunk]:
        ...         async for token in llm.stream(input.prompt):
        ...             yield LLMChunk(token=token)
    
    Usage:
        >>> tool = StreamingLLMTool()
        >>> async for chunk in tool.invoke_stream(input, context):
        ...     print(chunk.token, end="")
    """
    
    async def invoke_stream(
        self,
        input: TInput,
        context: ToolContext,
    ) -> AsyncIterator[TChunk]:
        """
        Execute tool with streaming output.
        
        Args:
            input: Validated input model
            context: Execution context
        
        Yields:
            Output chunks as they become available
        
        Raises:
            ToolError: For tool-specific errors
        """
        ...
        yield  # Make this a generator for Protocol
