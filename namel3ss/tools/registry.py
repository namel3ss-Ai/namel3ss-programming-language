"""
Tool registry for managing tool instances at runtime.

Provides a centralized registry for storing, retrieving, and managing tool instances.
Supports registration, update, lookup, and listing operations with validation.

Now enhanced to support the Tool Adapter Framework:
- Register both legacy BaseTool and new ToolAdapter instances
- Metadata-based discovery and search
- Tag-based filtering
- Namespace support

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

from typing import Dict, List, Optional, Union

from .base import BaseTool
from .errors import ToolRegistrationError, ToolValidationError
from .validation import validate_tool_instance, validate_tool_name


# Import ToolAdapter if available (optional dependency for backward compat)
try:
    from .adapter import ToolAdapter, ToolMetadata
    HAS_ADAPTER = True
except ImportError:
    HAS_ADAPTER = False
    ToolAdapter = None  # type: ignore
    ToolMetadata = None  # type: ignore


class ToolRegistry:
    """
    Registry for storing and retrieving tool instances.
    
    Provides a centralized location for managing tool instances at runtime.
    Supports both legacy BaseTool and new ToolAdapter instances.
    
    Enhanced Features:
        - Metadata-based discovery
        - Tag filtering
        - Namespace support
        - Adapter protocol support
    
    Attributes:
        _tools: Internal dictionary mapping tool names to tool instances
        _metadata: Cached metadata for adapters
    
    Methods:
        register(): Register new tool (raises ValueError if exists)
        register_adapter(): Register ToolAdapter instance
        update(): Register or update tool (overwrites existing)
        get(): Retrieve tool by name (returns None if not found)
        get_required(): Retrieve tool by name (raises KeyError if not found)
        get_adapter(): Retrieve tool as ToolAdapter (if applicable)
        has(): Check if tool exists
        list_tools(): Get all tool names and types
        find_by_tag(): Find tools by tag
        find_by_namespace(): Find tools by namespace
        get_metadata(): Get metadata for tool
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
        self._tools: Dict[str, Union[BaseTool, Any]] = {}
        self._metadata: Dict[str, Any] = {}  # Cached metadata for adapters
    
    def register(self, name: str, tool: Union[BaseTool, Any]) -> None:
        """
        Register a tool instance (legacy or adapter).
        
        Args:
            name: Tool identifier
            tool: Tool instance (BaseTool or ToolAdapter)
            
        Raises:
            ToolValidationError: If name or tool invalid
            ToolRegistrationError: If tool already registered
        """
        # Validate inputs
        validate_tool_name(name)
        
        # Validate tool (legacy or adapter)
        if isinstance(tool, BaseTool):
            validate_tool_instance(tool, tool_name=name)
        elif HAS_ADAPTER and isinstance(tool, ToolAdapter):
            # Adapter validation
            try:
                metadata = tool.get_metadata()
                self._metadata[name] = metadata
            except Exception as e:
                raise ToolValidationError(
                    f"Failed to get metadata from adapter: {e}",
                    code="TOOL002",
                    tool_name=name,
                    original_error=e,
                ) from e
        else:
            # Check if it implements the protocol
            if HAS_ADAPTER:
                from .adapter import ToolAdapter
                if not isinstance(tool, ToolAdapter):
                    raise ToolValidationError(
                        f"Tool must be BaseTool or ToolAdapter instance, got {type(tool)}",
                        code="TOOL002",
                        tool_name=name,
                        value=type(tool).__name__,
                        expected="BaseTool or ToolAdapter",
                    )
        
        if name in self._tools:
            existing_type = self._get_tool_type(self._tools[name])
            raise ToolRegistrationError(
                f"Tool '{name}' is already registered",
                code="TOOL026",
                tool_name=name,
                conflict=f"Existing tool of type '{existing_type}'",
            )
        self._tools[name] = tool
    
    def register_adapter(self, adapter: Any) -> str:
        """
        Register a ToolAdapter using its metadata name.
        
        Args:
            adapter: ToolAdapter instance
        
        Returns:
            Registered tool name
        
        Raises:
            ToolValidationError: If adapter invalid
            ToolRegistrationError: If tool already registered
        
        Example:
            >>> adapter = MyToolAdapter()
            >>> name = registry.register_adapter(adapter)
            >>> print(f"Registered as: {name}")
        """
        if not HAS_ADAPTER:
            raise ToolValidationError(
                "ToolAdapter not available (adapter module not imported)",
                code="TOOL002",
            )
        
        try:
            metadata = adapter.get_metadata()
            name = metadata.name
        except Exception as e:
            raise ToolValidationError(
                f"Failed to get adapter metadata: {e}",
                code="TOOL002",
                original_error=e,
            ) from e
        
        self.register(name, adapter)
        return name
    
    def update(self, name: str, tool: Union[BaseTool, Any]) -> None:
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
        
        if isinstance(tool, BaseTool):
            validate_tool_instance(tool, tool_name=name)
        elif HAS_ADAPTER and isinstance(tool, ToolAdapter):
            try:
                metadata = tool.get_metadata()
                self._metadata[name] = metadata
            except Exception as e:
                raise ToolValidationError(
                    f"Failed to get metadata: {e}",
                    code="TOOL002",
                    tool_name=name,
                    original_error=e,
                ) from e
        
        self._tools[name] = tool
    
    def get(self, name: str) -> Optional[Union[BaseTool, Any]]:
        """
        Get a tool by name.
        
        Args:
            name: Tool identifier
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def get_required(self, name: str) -> Union[BaseTool, Any]:
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
    
    def get_adapter(self, name: str) -> Optional[Any]:
        """
        Get tool as ToolAdapter (if applicable).
        
        Args:
            name: Tool identifier
        
        Returns:
            ToolAdapter instance or None
        """
        tool = self.get(name)
        if tool and HAS_ADAPTER and isinstance(tool, ToolAdapter):
            return tool
        return None
    
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
        return {name: self._get_tool_type(tool) for name, tool in self._tools.items()}
    
    def find_by_tag(self, tag: str) -> List[str]:
        """
        Find tools by tag.
        
        Args:
            tag: Tag to search for
        
        Returns:
            List of tool names with matching tag
        
        Example:
            >>> registry.find_by_tag("api")
            ['weather_api', 'stock_api']
        """
        results = []
        for name, tool in self._tools.items():
            metadata = self._get_metadata(name, tool)
            if metadata and tag in metadata.get("tags", []):
                results.append(name)
        return results
    
    def find_by_namespace(self, namespace: str) -> List[str]:
        """
        Find tools by namespace.
        
        Args:
            namespace: Namespace to search for
        
        Returns:
            List of tool names in namespace
        
        Example:
            >>> registry.find_by_namespace("integrations")
            ['integrations.weather', 'integrations.stock']
        """
        results = []
        for name, tool in self._tools.items():
            metadata = self._get_metadata(name, tool)
            if metadata and metadata.get("namespace") == namespace:
                results.append(name)
        return results
    
    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a tool.
        
        Args:
            name: Tool identifier
        
        Returns:
            Metadata dict or None
        """
        tool = self.get(name)
        if not tool:
            return None
        return self._get_metadata(name, tool)
    
    def _get_tool_type(self, tool: Union[BaseTool, Any]) -> str:
        """Get tool type string."""
        if isinstance(tool, BaseTool):
            return tool.get_tool_type()
        elif HAS_ADAPTER and isinstance(tool, ToolAdapter):
            metadata = tool.get_metadata()
            return metadata.category.value if hasattr(metadata, "category") else "adapter"
        else:
            return "unknown"
    
    def _get_metadata(self, name: str, tool: Union[BaseTool, Any]) -> Optional[Dict[str, Any]]:
        """Get metadata dict for tool."""
        # Check cache first
        if name in self._metadata:
            metadata = self._metadata[name]
            if hasattr(metadata, "to_dict"):
                return metadata.to_dict()
            return metadata
        
        # Try to get from adapter
        if HAS_ADAPTER and isinstance(tool, ToolAdapter):
            try:
                metadata = tool.get_metadata()
                self._metadata[name] = metadata
                return metadata.to_dict()
            except:
                pass
        
        # Fallback for BaseTool
        if isinstance(tool, BaseTool):
            return {
                "name": name,
                "tool_type": tool.get_tool_type(),
                "tags": [],
            }
        
        return None
    
    def clear(self) -> None:
        """Remove all tools from registry."""
        self._tools.clear()
        self._metadata.clear()
    
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
