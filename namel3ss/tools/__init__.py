"""
Tool subsystem for Namel3ss - first-class tool blocks with extensible providers.

This package provides:
- BaseTool: Abstract interface for all tool implementations
- ToolRegistry: Central registry for tool instances
- Tool factory: create_tool() for instantiating tools by type
- Built-in tool types: http, python, database, vector_search
- Validation: Centralized validators for tool configuration
- Errors: Domain-specific exceptions with error codes

Example usage:
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
    
    # Validation
    from namel3ss.tools.validation import validate_tool_config
    from namel3ss.tools.errors import ToolValidationError
    
    try:
        validate_tool_config(name="test", tool_type="http", endpoint="https://api.com")
    except ToolValidationError as e:
        print(f"Validation failed: {e.format()}")
"""

from .base import BaseTool, ToolResult, ToolError
from .registry import ToolRegistry, get_registry, reset_registry
from .factory import create_tool, register_provider, get_provider_class
from .errors import ToolValidationError, ToolRegistrationError, ToolExecutionError
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

__all__ = [
    # Core types
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
]
