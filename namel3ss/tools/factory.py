"""Tool factory for creating tool instances by type."""

import os
from typing import Any, Dict, Optional, Type

from .base import BaseTool, ToolError
from .registry import get_registry
from .errors import ToolValidationError
from .validation import validate_tool_config


# Registry of tool provider classes
_TOOL_PROVIDERS: Dict[str, Type[BaseTool]] = {}


def register_provider(tool_type: str, provider_class: Type[BaseTool]) -> None:
    """
    Register a tool provider class for a given type.
    
    Args:
        tool_type: Tool type identifier (e.g., "http", "python")
        provider_class: Class implementing BaseTool
    """
    _TOOL_PROVIDERS[tool_type] = provider_class


def get_provider_class(tool_type: str) -> Type[BaseTool]:
    """
    Get the provider class for a tool type.
    
    Args:
        tool_type: Tool type identifier
        
    Returns:
        Provider class
        
    Raises:
        ToolError: If provider not found
    """
    # Try lazy loading for built-in providers
    if tool_type not in _TOOL_PROVIDERS:
        if tool_type == "http":
            from .http_tool import HttpTool
            register_provider("http", HttpTool)
        elif tool_type == "python":
            from .python_tool import PythonTool
            register_provider("python", PythonTool)
        # Add more built-in types as needed
    
    provider_class = _TOOL_PROVIDERS.get(tool_type)
    if provider_class is None:
        raise ToolError(f"Unknown tool type: {tool_type}")
    
    return provider_class


def create_tool(
    name: str,
    tool_type: str,
    *,
    endpoint: Optional[str] = None,
    method: str = "POST",
    input_schema: Optional[Dict[str, Any]] = None,
    output_schema: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    register: bool = False,
    **config: Any,
) -> BaseTool:
    """
    Create a tool instance using the factory pattern.
    
    Validates configuration and instantiates the appropriate tool class
    based on tool_type. Optionally registers in global registry.
    
    Args:
        name: Tool identifier (must be unique if registering)
        tool_type: Type of tool (http, python, database, etc.)
        endpoint: Tool endpoint (for http/api tools)
        method: HTTP method (for http tools)
        input_schema: JSON schema for inputs
        output_schema: JSON schema for outputs
        headers: HTTP headers (for http tools)
        timeout: Execution timeout in seconds (default: 30.0)
        register: If True, register in global registry
        **config: Tool-specific configuration
        
    Returns:
        Instantiated tool
        
    Raises:
        ToolValidationError: If configuration is invalid
        ToolError: If tool type unknown or creation fails
        
    Example:
        Create HTTP tool:
        >>> tool = create_tool(
        ...     name="weather",
        ...     tool_type="http",
        ...     endpoint="https://api.weather.com/v1/current",
        ...     method="GET",
        ...     timeout=10.0,
        ...     register=True
        ... )
        >>> result = tool.execute(location="NYC")
        
        Create Python tool:
        >>> def add(a, b):
        ...     return a + b
        >>> tool = create_tool(
        ...     name="calculator",
        ...     tool_type="python",
        ...     function=add,
        ...     register=True
        ... )
        >>> result = tool.execute(a=5, b=3)
    
    Best Practices:
        - Validate configuration before creating tools
        - Use descriptive, unique names for registry
        - Set appropriate timeouts for operations
        - Define input/output schemas for type safety
        - Register tools during initialization, not in hot paths
    """
    # Validate configuration before creation
    validate_tool_config(
        name=name,
        tool_type=tool_type,
        timeout=timeout,
        input_schema=input_schema,
        output_schema=output_schema,
        endpoint=endpoint,
        method=method,
        headers=headers,
        code=config.get("code"),
        function=config.get("function"),
    )
    
    provider_class = get_provider_class(tool_type)
    
    # Build config dict with all parameters
    full_config = {
        "name": name,
        "tool_type": tool_type,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "timeout": timeout,
        **config,
    }
    
    # Add type-specific parameters
    if endpoint is not None:
        full_config["endpoint"] = endpoint
    if method:
        full_config["method"] = method
    if headers:
        full_config["headers"] = headers
    
    try:
        tool = provider_class(**full_config)
    except Exception as e:
        raise ToolError(
            f"Failed to create tool '{name}' of type '{tool_type}': {e}",
            tool_name=name,
            original_error=e,
        )
    
    if register:
        registry = get_registry()
        registry.update(name, tool)
    
    return tool


def register_tool(name: str, tool: BaseTool) -> None:
    """
    Register a tool instance in the global registry.
    
    Args:
        name: Tool identifier
        tool: Tool instance
        
    Raises:
        ValueError: If tool already registered
    """
    registry = get_registry()
    registry.register(name, tool)
