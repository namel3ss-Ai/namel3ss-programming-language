"""
LangChain adapter for importing tools from LangChain tool definitions.

Supports importing from LangChain BaseTool instances and converting them
to functions compatible with the ToolRegistry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Type
import inspect


@dataclass
class LangChainToolConfig:
    """Configuration for LangChain tool import."""
    name_prefix: str = ""
    include_async: bool = True
    strip_langchain_prefix: bool = True


class LangChainAdapter:
    """
    Adapter for importing tools from LangChain.
    
    Features:
    - Converts LangChain BaseTool instances to registry-compatible functions
    - Handles both sync and async tools
    - Preserves tool schemas and descriptions
    - Supports custom tool classes
    
    Example:
        >>> from langchain.tools import DuckDuckGoSearchRun
        >>> adapter = LangChainAdapter()
        >>> search_tool = DuckDuckGoSearchRun()
        >>> tool_func = adapter.import_tool(search_tool)
        >>> registry.register_tool(tool_func)
    """
    
    def __init__(self):
        self._imported_tools: Dict[str, Callable] = {}
    
    def import_tool(
        self,
        langchain_tool: Any,
        name_prefix: str = "",
        strip_prefix: bool = True,
    ) -> Callable:
        """
        Import a single LangChain tool.
        
        Args:
            langchain_tool: LangChain BaseTool instance
            name_prefix: Optional prefix for tool name
            strip_prefix: Remove "langchain_" prefix if present
        
        Returns:
            Callable tool function for registry
        """
        # Extract tool metadata
        tool_name = getattr(langchain_tool, "name", langchain_tool.__class__.__name__)
        description = getattr(langchain_tool, "description", "")
        
        # Strip LangChain prefix if requested
        if strip_prefix and tool_name.startswith("langchain_"):
            tool_name = tool_name[len("langchain_"):]
        
        # Add prefix
        if name_prefix:
            tool_name = f"{name_prefix}{tool_name}"
        
        # Create wrapper function
        async def tool_func(**kwargs) -> Any:
            """Wrapper for LangChain tool."""
            # LangChain tools typically accept a single input string or dict
            # Try to detect the input format
            if hasattr(langchain_tool, "args_schema"):
                # Structured input
                result = await self._run_tool_async(langchain_tool, kwargs)
            else:
                # Single string input
                input_str = kwargs.get("input", kwargs.get("query", ""))
                result = await self._run_tool_async(langchain_tool, input_str)
            
            return result
        
        # Set function metadata
        tool_func.__name__ = tool_name
        tool_func.__doc__ = description
        
        # Extract input schema
        input_schema = self._extract_input_schema(langchain_tool)
        
        # Attach metadata
        tool_func._tool_metadata = {
            "name": tool_name,
            "description": description,
            "input_schema": input_schema,
            "output_schema": {"type": "object"},
            "tags": ["langchain"],
            "source": "langchain",
        }
        
        self._imported_tools[tool_name] = tool_func
        return tool_func
    
    def import_tools(
        self,
        langchain_tools: List[Any],
        name_prefix: str = "",
        strip_prefix: bool = True,
    ) -> List[Callable]:
        """
        Import multiple LangChain tools.
        
        Args:
            langchain_tools: List of LangChain BaseTool instances
            name_prefix: Optional prefix for tool names
            strip_prefix: Remove "langchain_" prefix if present
        
        Returns:
            List of callable tool functions
        """
        return [
            self.import_tool(tool, name_prefix, strip_prefix)
            for tool in langchain_tools
        ]
    
    async def _run_tool_async(self, tool: Any, tool_input: Any) -> Any:
        """Run a LangChain tool (handle both sync and async)."""
        # Check if tool has async run method
        if hasattr(tool, "arun"):
            return await tool.arun(tool_input)
        elif hasattr(tool, "_arun"):
            return await tool._arun(tool_input)
        
        # Fallback to sync run
        if hasattr(tool, "run"):
            # Run in executor to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, tool.run, tool_input)
        elif hasattr(tool, "_run"):
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, tool._run, tool_input)
        
        raise ValueError(f"Tool {tool} has no run method")
    
    def _extract_input_schema(self, tool: Any) -> Dict[str, Any]:
        """Extract input schema from LangChain tool."""
        # Check for explicit args_schema
        if hasattr(tool, "args_schema") and tool.args_schema:
            return self._pydantic_to_json_schema(tool.args_schema)
        
        # Check for args property
        if hasattr(tool, "args"):
            args = tool.args
            if isinstance(args, dict):
                return {
                    "type": "object",
                    "properties": {
                        k: {"type": "string", "description": v}
                        for k, v in args.items()
                    },
                }
        
        # Fallback to simple input schema
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Tool input",
                }
            },
            "required": ["input"],
        }
    
    def _pydantic_to_json_schema(self, pydantic_model: Type) -> Dict[str, Any]:
        """Convert Pydantic model to JSON schema."""
        try:
            # Pydantic v2
            if hasattr(pydantic_model, "model_json_schema"):
                return pydantic_model.model_json_schema()
            # Pydantic v1
            elif hasattr(pydantic_model, "schema"):
                return pydantic_model.schema()
        except Exception:
            pass
        
        # Fallback
        return {"type": "object"}
    
    def get_imported_tools(self) -> Dict[str, Callable]:
        """Get all imported tools."""
        return self._imported_tools.copy()
