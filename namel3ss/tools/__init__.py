"""
Tool subsystem for Namel3ss - first-class tool blocks with extensible providers.

This package provides:
- BaseTool: Abstract interface for all tool implementations (legacy)
- ToolAdapter: Production-grade adapter protocol (new framework)
- ToolRegistry: Central registry for tool instances
- Tool factory: create_tool() for instantiating tools by type
- Built-in tool types: http, python, database, vector_search
- Validation: Centralized validators for tool configuration
- Errors: Domain-specific exceptions with error codes
- Schemas: Pydantic models for input/output validation
- Streaming: Support for streaming tool responses

Tool Adapter Framework (New):
    The Tool Adapter Framework provides a production-grade abstraction for
    creating and managing tools with:
    - Strongly-typed I/O with Pydantic schemas
    - Comprehensive error handling
    - Observability (logging, tracing, metrics)
    - Streaming support
    - Configuration management

Legacy vs. New:
    - Legacy: BaseTool (existing code, backward compatible)
    - New: ToolAdapter (production-grade, type-safe, observable)
    
    Both are supported by the registry. New code should use ToolAdapter.

Example (Legacy):
    from namel3ss.tools import create_tool, get_registry
    
    # Create a tool
    weather_tool = create_tool(
        name="weather",
        tool_type="http",
        endpoint="https://api.weather.com/v1/current",
        method="GET",
        timeout=10.0
    )
    
    # Execute tool
    result = weather_tool.execute(location="San Francisco", units="metric")
    print(result.output)

Example (New Framework):
    from namel3ss.tools.adapter import BaseToolAdapter, ToolMetadata, ToolCategory
    from namel3ss.tools.schemas import ToolInputModel, ToolOutputModel
    from pydantic import Field
    
    class WeatherInput(ToolInputModel):
        location: str = Field(..., description="City name")
        units: str = Field("metric", description="Temperature units")
    
    class WeatherOutput(ToolOutputModel):
        temperature: float
        condition: str
    
    class WeatherTool(BaseToolAdapter[WeatherInput, WeatherOutput]):
        def get_metadata(self):
            return ToolMetadata(
                name="weather",
                description="Get weather conditions",
                version="1.0.0",
                category=ToolCategory.HTTP
            )
        
        def get_input_schema(self):
            return WeatherInput
        
        def get_output_schema(self):
            return WeatherOutput
        
        async def invoke(self, input, context):
            # Implementation
            return WeatherOutput(temperature=72.0, condition="sunny")

Validation:
    from namel3ss.tools.validation import validate_tool_config
    from namel3ss.tools.errors import ToolValidationError
    
    try:
        validate_tool_config(name="test", tool_type="http", endpoint="https://api.com")
    except ToolValidationError as e:
        print(f"Validation failed: {e.format()}")
"""

# Legacy components
from .base import BaseTool, ToolResult, ToolError
from .registry import ToolRegistry, get_registry, reset_registry
from .factory import create_tool, register_provider, get_provider_class
from .errors import (
    ToolValidationError,
    ToolRegistrationError,
    ToolExecutionError,
    ToolTimeoutError,
    ToolConfigurationError,
    ToolAuthenticationError,
    serialize_tool_error,
    deserialize_tool_error,
)
from .validation import (
    validate_tool_name,
    validate_tool_type,
    validate_timeout,
    validate_schema,
    validate_http_method,
    validate_http_endpoint,
    validate_http_headers,
    validate_python_code,
    validate_tool_instance,
    validate_execution_inputs,
    validate_tool_config,
)

# New Tool Adapter Framework
from .adapter import (
    ToolAdapter,
    BaseToolAdapter,
    StreamingToolAdapter,
    ToolMetadata,
    ToolConfig,
    ToolContext,
    ToolCategory,
)
from .schemas import (
    ToolInputModel,
    ToolOutputModel,
    ToolChunkModel,
    ToolErrorModel,
    SimpleTextInput,
    SimpleTextOutput,
    KeyValueInput,
    KeyValueOutput,
    ListInput,
    ListOutput,
    JSONInput,
    JSONOutput,
    merge_schemas,
    extend_schema,
    schema_to_json_schema,
    validate_against_schema,
)
from .streaming import (
    StreamBuffer,
    StreamAggregator,
    StreamingContext,
    rate_limit_stream,
    batch_stream,
    filter_stream,
    map_stream,
    collect_stream,
    take_stream,
    merge_streams,
)

__all__ = [
    # Legacy core types
    "BaseTool",
    "ToolResult",
    "ToolError",
    # Registry
    "ToolRegistry",
    "get_registry",
    "reset_registry",
    # Factory
    "create_tool",
    "register_provider",
    "get_provider_class",
    # Errors
    "ToolValidationError",
    "ToolRegistrationError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolConfigurationError",
    "ToolAuthenticationError",
    "serialize_tool_error",
    "deserialize_tool_error",
    # Validation
    "validate_tool_name",
    "validate_tool_type",
    "validate_timeout",
    "validate_schema",
    "validate_http_method",
    "validate_http_endpoint",
    "validate_http_headers",
    "validate_python_code",
    "validate_tool_instance",
    "validate_execution_inputs",
    "validate_tool_config",
    # Tool Adapter Framework - Core
    "ToolAdapter",
    "BaseToolAdapter",
    "StreamingToolAdapter",
    "ToolMetadata",
    "ToolConfig",
    "ToolContext",
    "ToolCategory",
    # Tool Adapter Framework - Schemas
    "ToolInputModel",
    "ToolOutputModel",
    "ToolChunkModel",
    "ToolErrorModel",
    "SimpleTextInput",
    "SimpleTextOutput",
    "KeyValueInput",
    "KeyValueOutput",
    "ListInput",
    "ListOutput",
    "JSONInput",
    "JSONOutput",
    "merge_schemas",
    "extend_schema",
    "schema_to_json_schema",
    "validate_against_schema",
    # Tool Adapter Framework - Streaming
    "StreamBuffer",
    "StreamAggregator",
    "StreamingContext",
    "rate_limit_stream",
    "batch_stream",
    "filter_stream",
    "map_stream",
    "collect_stream",
    "take_stream",
    "merge_streams",
]
