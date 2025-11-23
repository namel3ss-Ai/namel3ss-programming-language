"""
Mock tool framework for deterministic testing of namel3ss applications.

This module provides mock implementations for various tools (HTTP, database, 
vector search, etc.) that can replace real tool implementations during testing
to ensure deterministic, offline test execution.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Pattern, Union
from unittest.mock import MagicMock

from namel3ss.testing import MockToolSpec, MockToolResponse
from namel3ss.tools.base import BaseTool, ToolResult
from namel3ss.tools.http_tool import HttpTool


class MockToolRegistry:
    """
    Registry for managing mock tool implementations during testing.
    
    Allows registering mock responses for specific tools and input patterns,
    then intercepts tool calls during test execution to return deterministic
    results instead of calling real external services.
    """
    
    def __init__(self):
        """Initialize empty mock tool registry."""
        self.tool_mocks: Dict[str, List[ToolMock]] = {}
        self.call_history: List[Dict[str, Any]] = []
        self._original_tools: Dict[str, BaseTool] = {}
    
    def register_mock(
        self,
        tool_name: str,
        input_pattern: Optional[Dict[str, Any]] = None,
        response: Optional[MockToolResponse] = None,
        response_function: Optional[Callable] = None,
        priority: int = 0
    ) -> None:
        """
        Register a mock response for a specific tool and input pattern.
        
        Args:
            tool_name: Name of the tool to mock
            input_pattern: Pattern to match against tool inputs (None matches any)
            response: Fixed mock response to return
            response_function: Function to generate dynamic mock response
            priority: Priority for matching (higher = checked first)
            
        Example:
            >>> registry = MockToolRegistry()
            >>> registry.register_mock(
            ...     tool_name="http_api",
            ...     input_pattern={"url": "*/users/*"},
            ...     response=MockToolResponse(
            ...         output={"user_id": "123", "name": "Test User"},
            ...         success=True
            ...     )
            ... )
        """
        if response is None and response_function is None:
            raise ValueError("Either response or response_function must be provided")
        
        mock = ToolMock(
            input_pattern=input_pattern,
            response=response,
            response_function=response_function,
            priority=priority
        )
        
        if tool_name not in self.tool_mocks:
            self.tool_mocks[tool_name] = []
        
        # Insert in priority order (highest first)
        inserted = False
        for i, existing_mock in enumerate(self.tool_mocks[tool_name]):
            if mock.priority > existing_mock.priority:
                self.tool_mocks[tool_name].insert(i, mock)
                inserted = True
                break
        if not inserted:
            self.tool_mocks[tool_name].append(mock)
    
    def mock_tool_call(self, tool_name: str, **inputs) -> ToolResult:
        """
        Execute mock tool call instead of real tool.
        
        Args:
            tool_name: Name of the tool being called
            **inputs: Tool input parameters
            
        Returns:
            ToolResult with mock response
            
        Raises:
            KeyError: If no mock is registered for the tool
        """
        # Record call for inspection
        call_record = {
            "timestamp": time.time(),
            "tool_name": tool_name,
            "inputs": inputs
        }
        self.call_history.append(call_record)
        
        # Find matching mock
        if tool_name not in self.tool_mocks:
            raise KeyError(f"No mocks registered for tool '{tool_name}'")
        
        for mock in self.tool_mocks[tool_name]:
            if mock.matches_input(inputs):
                mock_response = mock.get_response(inputs)
                
                return ToolResult(
                    output=mock_response.output,
                    success=mock_response.success,
                    error=mock_response.error,
                    metadata={"mock": True, **mock_response.metadata}
                )
        
        # No mock matched
        return ToolResult(
            output=None,
            success=False,
            error=f"No mock response found for tool '{tool_name}' with inputs {inputs}",
            metadata={"mock": True, "error": "no_mock_match"}
        )
    
    def clear_history(self) -> None:
        """Clear call history for fresh test state."""
        self.call_history.clear()
    
    def get_call_count(self, tool_name: Optional[str] = None) -> int:
        """
        Get call count for specific tool or all tools.
        
        Args:
            tool_name: Specific tool name, or None for all tools
            
        Returns:
            Number of calls made
        """
        if tool_name is None:
            return len(self.call_history)
        
        return len([call for call in self.call_history if call["tool_name"] == tool_name])
    
    def get_calls_for_tool(self, tool_name: str) -> List[Dict[str, Any]]:
        """Get all calls made to a specific tool."""
        return [call for call in self.call_history if call["tool_name"] == tool_name]


class ToolMock:
    """
    Individual mock configuration for a specific tool input pattern.
    
    Handles matching input patterns and generating appropriate responses.
    """
    
    def __init__(
        self,
        input_pattern: Optional[Dict[str, Any]] = None,
        response: Optional[MockToolResponse] = None,
        response_function: Optional[Callable] = None,
        priority: int = 0
    ):
        """
        Initialize tool mock configuration.
        
        Args:
            input_pattern: Pattern to match against inputs
            response: Fixed mock response
            response_function: Function to generate dynamic response
            priority: Matching priority
        """
        self.input_pattern = input_pattern
        self.response = response
        self.response_function = response_function
        self.priority = priority
    
    def matches_input(self, inputs: Dict[str, Any]) -> bool:
        """
        Check if the given inputs match this mock's pattern.
        
        Args:
            inputs: Tool input parameters
            
        Returns:
            True if inputs match the pattern
        """
        if self.input_pattern is None:
            return True  # Match all inputs
        
        return self._pattern_matches(self.input_pattern, inputs)
    
    def get_response(self, inputs: Dict[str, Any]) -> MockToolResponse:
        """
        Generate mock response for the given inputs.
        
        Args:
            inputs: Tool input parameters
            
        Returns:
            MockToolResponse to return
        """
        if self.response_function:
            return self.response_function(inputs)
        
        if self.response:
            return self.response
        
        # Default response
        return MockToolResponse(
            output="Default mock response",
            success=True
        )
    
    def _pattern_matches(self, pattern: Any, value: Any) -> bool:
        """
        Check if a value matches a pattern.
        
        Supports:
        - Exact matches for primitive types
        - Wildcard patterns with '*'
        - Regex patterns for strings
        - Nested matching for dictionaries
        
        Args:
            pattern: Pattern to match against
            value: Value to check
            
        Returns:
            True if value matches pattern
        """
        if pattern == value:
            return True
        
        if isinstance(pattern, str) and isinstance(value, str):
            # Handle wildcard and regex patterns
            if '*' in pattern:
                # Convert wildcard to regex
                regex_pattern = pattern.replace('*', '.*')
                return bool(re.match(regex_pattern, value))
            
            # Check if pattern is a regex
            if pattern.startswith('r/') and pattern.endswith('/'):
                regex_pattern = pattern[2:-1]
                return bool(re.search(regex_pattern, value))
        
        if isinstance(pattern, dict) and isinstance(value, dict):
            # Check if all pattern keys match
            for key, pattern_value in pattern.items():
                if key not in value:
                    return False
                if not self._pattern_matches(pattern_value, value[key]):
                    return False
            return True
        
        if isinstance(pattern, list) and isinstance(value, list):
            # Check if lists match element-wise
            if len(pattern) != len(value):
                return False
            return all(self._pattern_matches(p, v) for p, v in zip(pattern, value))
        
        return False


class MockHttpTool(HttpTool):
    """
    Mock HTTP tool that returns configured responses instead of making real HTTP calls.
    
    Inherits from HttpTool to maintain the same interface while providing
    deterministic responses for testing.
    """
    
    def __init__(self, registry: MockToolRegistry, **kwargs):
        """
        Initialize mock HTTP tool.
        
        Args:
            registry: Mock tool registry for response configuration
            **kwargs: Standard HttpTool parameters
        """
        super().__init__(**kwargs)
        self.registry = registry
    
    def execute(self, **inputs) -> ToolResult:
        """
        Execute mock HTTP request.
        
        Args:
            **inputs: Request parameters
            
        Returns:
            ToolResult with mock response
        """
        # Use registry to get mock response instead of making real HTTP call
        return self.registry.mock_tool_call(self.name, **inputs)


def create_mock_registry_from_specs(specs: List[MockToolSpec]) -> MockToolRegistry:
    """
    Create a MockToolRegistry from a list of MockToolSpec configurations.
    
    Args:
        specs: List of tool mock specifications
        
    Returns:
        Configured MockToolRegistry
        
    Example:
        >>> specs = [
        ...     MockToolSpec(
        ...         tool_name="weather_api",
        ...         input_pattern={"city": "*"},
        ...         response=MockToolResponse(
        ...             output={"temperature": 72, "condition": "sunny"},
        ...             success=True
        ...         )
        ...     )
        ... ]
        >>> registry = create_mock_registry_from_specs(specs)
        >>> result = registry.mock_tool_call("weather_api", city="San Francisco")
        >>> result.output["temperature"]
        72
    """
    registry = MockToolRegistry()
    
    for spec in specs:
        registry.register_mock(
            tool_name=spec.tool_name,
            input_pattern=spec.input_pattern,
            response=spec.response
        )
    
    return registry


class MockDatabaseTool:
    """
    Mock database tool for testing applications that use database operations.
    
    Provides deterministic responses for SQL queries and database operations
    without requiring a real database connection.
    """
    
    def __init__(self, registry: MockToolRegistry, name: str = "mock_db"):
        """
        Initialize mock database tool.
        
        Args:
            registry: Mock tool registry
            name: Tool name for identification
        """
        self.registry = registry
        self.name = name
        self.schema: Dict[str, Any] = {}
        self.data: Dict[str, List[Dict[str, Any]]] = {}
    
    def setup_table(self, table_name: str, schema: Dict[str, str], data: List[Dict[str, Any]] = None):
        """
        Set up a mock table with schema and initial data.
        
        Args:
            table_name: Name of the table
            schema: Column definitions {column_name: type}
            data: Initial data rows
        """
        self.schema[table_name] = schema
        self.data[table_name] = data or []
    
    def execute(self, **inputs) -> ToolResult:
        """
        Execute mock database operation.
        
        Args:
            **inputs: Query parameters (sql, params, etc.)
            
        Returns:
            ToolResult with mock query result
        """
        return self.registry.mock_tool_call(self.name, **inputs)


class MockVectorSearchTool:
    """
    Mock vector search tool for testing RAG and semantic search applications.
    
    Provides deterministic search results without requiring a real vector database.
    """
    
    def __init__(self, registry: MockToolRegistry, name: str = "mock_vector_search"):
        """
        Initialize mock vector search tool.
        
        Args:
            registry: Mock tool registry
            name: Tool name for identification
        """
        self.registry = registry
        self.name = name
        self.embeddings: Dict[str, List[float]] = {}
        self.documents: Dict[str, Dict[str, Any]] = {}
    
    def add_mock_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """
        Add a mock document to the search index.
        
        Args:
            doc_id: Document identifier
            content: Document content
            metadata: Document metadata
        """
        self.documents[doc_id] = {
            "content": content,
            "metadata": metadata or {}
        }
        # Mock embedding (just use hash of content for deterministic results)
        self.embeddings[doc_id] = [float(hash(content) % 1000) / 1000.0] * 512
    
    def execute(self, **inputs) -> ToolResult:
        """
        Execute mock vector search.
        
        Args:
            **inputs: Search parameters (query, top_k, etc.)
            
        Returns:
            ToolResult with mock search results
        """
        return self.registry.mock_tool_call(self.name, **inputs)


__all__ = [
    "MockToolRegistry",
    "ToolMock",
    "MockHttpTool",
    "MockDatabaseTool", 
    "MockVectorSearchTool",
    "create_mock_registry_from_specs"
]