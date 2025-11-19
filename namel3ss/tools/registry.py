"""Tool registry for managing tool instances at runtime."""

from typing import Dict, Optional

from .base import BaseTool


class ToolRegistry:
    """Registry for storing and retrieving tool instances."""
    
    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, name: str, tool: BaseTool) -> None:
        """
        Register a tool instance.
        
        Args:
            name: Tool identifier
            tool: Tool instance
            
        Raises:
            ValueError: If tool already registered
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
        self._tools[name] = tool
    
    def update(self, name: str, tool: BaseTool) -> None:
        """
        Update or register a tool instance.
        
        Args:
            name: Tool identifier
            tool: Tool instance
        """
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
