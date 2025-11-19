"""
Tool subsystem for Namel3ss - first-class tool blocks with extensible providers.

This package provides:
- BaseTool: Abstract interface for all tool implementations
- ToolRegistry: Central registry for tool instances
- Tool factory: create_tool() for instantiating tools by type
- Built-in tool types: http, python, database, vector_search

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
"""

from .base import BaseTool, ToolResult, ToolError
from .registry import ToolRegistry, get_registry, reset_registry
from .factory import create_tool, register_provider, get_provider_class

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolError",
    "ToolRegistry",
    "get_registry",
    "reset_registry",
    "create_tool",
    "register_provider",
    "get_provider_class",
]
