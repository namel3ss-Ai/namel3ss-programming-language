"""
Tool registry for managing tool instances at runtime.

Provides a centralized registry for storing, retrieving, and managing tool instances.
Supports registration, update, lookup, and listing operations with validation.

Architecture:
    The registry maintains a dictionary of tool instances keyed by name.
    A global singleton instance is provided via get_registry().
    Tools can be registered, updated, retrieved, and listed.

Thread Safety:
    The current implementation is not thread-safe. In concurrent environments,
    wrap registry operations in locks or use thread-local registries.

Example:
    from namel3ss.tools import create_tool, get_registry
    
    # Create and register a tool
    tool = create_tool(
        name="weather",
        tool_type="http",
        endpoint="https://api.weather.com",
        register=True  # Automatically registers
    )
    
    # Later, retrieve from registry
    registry = get_registry()
    weather_tool = registry.get("weather")
    result = weather_tool.execute(location="NYC")

Best Practices:
    - Use descriptive, unique tool names
    - Register tools during initialization, not execution
    - Use get_required() when tool must exist (raises KeyError)
    - Use get() when tool might not exist (returns None)
    - Call list_tools() to see all registered tools
    - Use reset_registry() only in tests
"""

from typing import Dict, Optional

from .base import BaseTool
from .errors import ToolRegistrationError, ToolValidationError
from .validation import validate_tool_instance, validate_tool_name


class ToolRegistry:
    """
    Registry for storing and retrieving tool instances.
    
    Provides a centralized location for managing tool instances at runtime.
    Supports registration, update, retrieval, and enumeration operations.
    
    Attributes:
        _tools: Internal dictionary mapping tool names to BaseTool instances
    
    Methods:
        register(): Register new tool (raises ValueError if exists)
        update(): Register or update tool (overwrites existing)
        get(): Retrieve tool by name (returns None if not found)
        get_required(): Retrieve tool by name (raises KeyError if not found)
        has(): Check if tool exists
        list_tools(): Get all tool names and types
        clear(): Remove all tools
    
    Example:
        >>> from namel3ss.tools import ToolRegistry, create_tool
        >>> 
        >>> registry = ToolRegistry()
        >>> tool = create_tool(name="api", tool_type="http", endpoint="https://api.com")
        >>> registry.register("api", tool)
        >>> 
        >>> # Retrieve tool
        >>> api_tool = registry.get("api")
        >>> print(api_tool.name)
        api
        >>> 
        >>> # List all tools
        >>> print(registry.list_tools())
        {'api': 'http'}
    
    Design Notes:
        - Uses dict for O(1) lookup by name
        - register() enforces uniqueness, update() allows replacement
        - get_required() preferred when tool must exist
        - Thread-safety must be added externally if needed
    """
    
    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, name: str, tool: BaseTool) -> None:
        """
        Register a tool instance.
        
        Args:
            name: Tool identifier
            tool: Tool instance
            
        Raises:
            ToolValidationError: If name or tool invalid
            ToolRegistrationError: If tool already registered
        """
        # Validate inputs
        validate_tool_name(name)
        validate_tool_instance(tool, tool_name=name)
        
        if name in self._tools:
            existing_type = self._tools[name].get_tool_type()
            raise ToolRegistrationError(
                f"Tool '{name}' is already registered",
                code="TOOL026",
                tool_name=name,
                conflict=f"Existing tool of type '{existing_type}'",
            )
        self._tools[name] = tool
    
    def update(self, name: str, tool: BaseTool) -> None:
        """
        Update or register a tool instance.
        
        Args:
            name: Tool identifier
            tool: Tool instance
            
        Raises:
            ToolValidationError: If name or tool invalid
        """
        # Validate inputs
        validate_tool_name(name)
        validate_tool_instance(tool, tool_name=name)
        
        self._tools[name] = tool
    
    def get(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            name: Tool identifier
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def get_required(self, name: str) -> BaseTool:
        """
        Get a tool by name, raising error if not found.
        
        Args:
            name: Tool identifier
            
        Returns:
            Tool instance
            
        Raises:
            KeyError: If tool not found
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found in registry")
        return tool
    
    def has(self, name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            name: Tool identifier
            
        Returns:
            True if tool exists
        """
        return name in self._tools
    
    def list_tools(self) -> Dict[str, str]:
        """
        List all registered tools with their types.
        
        Returns:
            Dict mapping tool names to tool types
        """
        return {name: tool.get_tool_type() for name, tool in self._tools.items()}
    
    def clear(self) -> None:
        """Remove all tools from registry."""
        self._tools.clear()
    
    def __contains__(self, name: str) -> bool:
        """Support 'name in registry' syntax."""
        return name in self._tools
    
    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())})"


# Global registry instance
_GLOBAL_REGISTRY: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.
    
    Returns:
        Singleton ToolRegistry
    """
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ToolRegistry()
    return _GLOBAL_REGISTRY


def reset_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = None
