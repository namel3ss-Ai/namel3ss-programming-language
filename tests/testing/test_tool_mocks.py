"""
Tests for namel3ss tool mocking system.

This module tests the mock tool framework that allows testing namel3ss 
applications without making live HTTP requests, database connections, 
or vector search calls.
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from namel3ss.testing.tools import (
    MockToolRegistry, ToolMock, MockHttpTool, MockDatabaseTool,
    MockVectorSearchTool, create_mock_registry_from_specs,
    _match_input_pattern
)
from namel3ss.testing import MockToolSpec, MockToolResponse
from namel3ss.tools.base import BaseTool


class TestToolMock:
    """Test the base ToolMock class."""
    
    def test_tool_mock_creation(self):
        """Test creating a basic tool mock."""
        mock = ToolMock("test_tool")
        
        assert mock.name == "test_tool"
        assert mock._responses == []
    
    def test_add_response_mapping(self):
        """Test adding response mappings to a tool mock."""
        mock = ToolMock("test_tool")
        
        mock.add_response(
            input_pattern={"action": "get_data"},
            output={"status": "success", "data": [1, 2, 3]},
            success=True
        )
        
        assert len(mock._responses) == 1
        response = mock._responses[0]
        assert response.input_pattern == {"action": "get_data"}
        assert response.output["status"] == "success"
    
    def test_execute_with_matching_pattern(self):
        """Test executing a tool mock with matching input pattern."""
        mock = ToolMock("test_tool")
        mock.add_response(
            input_pattern={"endpoint": "/api/users"},
            output={"users": ["alice", "bob"]},
            success=True
        )
        
        result = mock.execute({"endpoint": "/api/users"})
        
        assert result.success is True
        assert result.output["users"] == ["alice", "bob"]
        assert result.metadata["mock"] is True
    
    def test_execute_with_no_matching_pattern(self):
        """Test executing with no matching pattern raises error."""
        mock = ToolMock("test_tool")
        mock.add_response(
            input_pattern={"action": "specific"},
            output="response"
        )
        
        with pytest.raises(ValueError, match="No mock response found"):
            mock.execute({"action": "different"})
    
    def test_execute_with_fallback(self):
        """Test executing with fallback response."""
        mock = ToolMock("test_tool")
        mock.add_response(
            input_pattern={"specific": "value"},
            output="specific response"
        )
        mock.set_fallback(output="fallback response", success=False)
        
        result = mock.execute({"different": "input"})
        
        assert result.output == "fallback response"
        assert result.success is False
        assert result.metadata["used_fallback"] is True
    
    def test_execute_with_response_delay(self):
        """Test that response delays are recorded in metadata."""
        mock = ToolMock("test_tool")
        mock.add_response(
            input_pattern={"test": "input"},
            output="delayed response",
            delay_ms=500
        )
        
        result = mock.execute({"test": "input"})
        
        assert result.metadata["delay_ms"] == 500


class TestMockHttpTool:
    """Test the MockHttpTool implementation."""
    
    def test_mock_http_tool_creation(self):
        """Test creating a mock HTTP tool."""
        tool = MockHttpTool("http_client")
        
        assert tool.name == "http_client"
        assert tool.tool_type == "http"
    
    def test_mock_http_get_request(self):
        """Test mocking HTTP GET requests."""
        tool = MockHttpTool("http_client")
        tool.add_response(
            input_pattern={"method": "GET", "url": "https://api.example.com/users"},
            output={"status_code": 200, "data": {"users": ["alice", "bob"]}},
            success=True
        )
        
        result = tool.execute({
            "method": "GET",
            "url": "https://api.example.com/users"
        })
        
        assert result.success is True
        assert result.output["status_code"] == 200
        assert "users" in result.output["data"]
    
    def test_mock_http_post_request(self):
        """Test mocking HTTP POST requests with payload matching."""
        tool = MockHttpTool("http_client")
        tool.add_response(
            input_pattern={
                "method": "POST",
                "url": "https://api.example.com/users",
                "payload": {"name": "charlie"}
            },
            output={"status_code": 201, "id": 123},
            success=True
        )
        
        result = tool.execute({
            "method": "POST", 
            "url": "https://api.example.com/users",
            "payload": {"name": "charlie"}
        })
        
        assert result.success is True
        assert result.output["id"] == 123
    
    def test_mock_http_error_response(self):
        """Test mocking HTTP error responses."""
        tool = MockHttpTool("http_client")
        tool.add_response(
            input_pattern={"method": "GET", "url": "https://api.example.com/error"},
            output={"status_code": 404, "error": "Not found"},
            success=False,
            error="HTTP 404 Error"
        )
        
        result = tool.execute({
            "method": "GET",
            "url": "https://api.example.com/error"
        })
        
        assert result.success is False
        assert result.error == "HTTP 404 Error"
        assert result.output["status_code"] == 404


class TestMockDatabaseTool:
    """Test the MockDatabaseTool implementation."""
    
    def test_mock_database_tool_creation(self):
        """Test creating a mock database tool."""
        tool = MockDatabaseTool("db_client")
        
        assert tool.name == "db_client"
        assert tool.tool_type == "database"
    
    def test_mock_database_query(self):
        """Test mocking database queries."""
        tool = MockDatabaseTool("db_client")
        tool.add_response(
            input_pattern={"query": "SELECT * FROM users WHERE active = true"},
            output={
                "rows": [
                    {"id": 1, "name": "alice", "active": True},
                    {"id": 2, "name": "bob", "active": True}
                ],
                "count": 2
            },
            success=True
        )
        
        result = tool.execute({
            "query": "SELECT * FROM users WHERE active = true"
        })
        
        assert result.success is True
        assert result.output["count"] == 2
        assert len(result.output["rows"]) == 2
    
    def test_mock_database_insert(self):
        """Test mocking database inserts."""
        tool = MockDatabaseTool("db_client")
        tool.add_response(
            input_pattern={
                "operation": "INSERT",
                "table": "users",
                "data": {"name": "charlie", "email": "charlie@example.com"}
            },
            output={"inserted_id": 3, "rows_affected": 1},
            success=True
        )
        
        result = tool.execute({
            "operation": "INSERT",
            "table": "users", 
            "data": {"name": "charlie", "email": "charlie@example.com"}
        })
        
        assert result.success is True
        assert result.output["inserted_id"] == 3
    
    def test_mock_database_connection_error(self):
        """Test mocking database connection errors."""
        tool = MockDatabaseTool("db_client")
        tool.set_fallback(
            output=None,
            success=False,
            error="Database connection timeout"
        )
        
        result = tool.execute({"query": "SELECT 1"})
        
        assert result.success is False
        assert "timeout" in result.error


class TestMockVectorSearchTool:
    """Test the MockVectorSearchTool implementation."""
    
    def test_mock_vector_search_tool_creation(self):
        """Test creating a mock vector search tool."""
        tool = MockVectorSearchTool("vector_client")
        
        assert tool.name == "vector_client"
        assert tool.tool_type == "vector_search"
    
    def test_mock_vector_similarity_search(self):
        """Test mocking vector similarity searches."""
        tool = MockVectorSearchTool("vector_client")
        tool.add_response(
            input_pattern={"query": "machine learning", "top_k": 5},
            output={
                "results": [
                    {"id": "doc1", "score": 0.95, "content": "ML is a subset of AI"},
                    {"id": "doc2", "score": 0.87, "content": "Neural networks are ML models"},
                    {"id": "doc3", "score": 0.82, "content": "Deep learning uses neural networks"}
                ],
                "total": 3
            },
            success=True
        )
        
        result = tool.execute({"query": "machine learning", "top_k": 5})
        
        assert result.success is True
        assert len(result.output["results"]) == 3
        assert result.output["results"][0]["score"] == 0.95
    
    def test_mock_vector_upsert(self):
        """Test mocking vector upserts."""
        tool = MockVectorSearchTool("vector_client")
        tool.add_response(
            input_pattern={
                "operation": "upsert",
                "vectors": [{"id": "new_doc", "values": [0.1, 0.2, 0.3]}]
            },
            output={"upserted_count": 1, "ids": ["new_doc"]},
            success=True
        )
        
        result = tool.execute({
            "operation": "upsert",
            "vectors": [{"id": "new_doc", "values": [0.1, 0.2, 0.3]}]
        })
        
        assert result.success is True
        assert result.output["upserted_count"] == 1


class TestMockToolRegistry:
    """Test the MockToolRegistry implementation."""
    
    def test_mock_registry_creation(self):
        """Test creating a mock tool registry."""
        registry = MockToolRegistry()
        
        assert registry._tools == {}
    
    def test_register_mock_tool(self):
        """Test registering a mock tool."""
        registry = MockToolRegistry()
        tool = MockHttpTool("http_client")
        
        registry.register_tool(tool)
        
        assert "http_client" in registry._tools
        assert registry._tools["http_client"] is tool
    
    def test_get_registered_tool(self):
        """Test getting a registered tool."""
        registry = MockToolRegistry()
        tool = MockDatabaseTool("db_client")
        registry.register_tool(tool)
        
        retrieved = registry.get_tool("db_client")
        
        assert retrieved is tool
    
    def test_get_unregistered_tool_raises_error(self):
        """Test that getting an unregistered tool raises KeyError."""
        registry = MockToolRegistry()
        
        with pytest.raises(KeyError, match="Tool 'unknown_tool' not found"):
            registry.get_tool("unknown_tool")
    
    def test_list_tools(self):
        """Test listing all registered tools."""
        registry = MockToolRegistry()
        registry.register_tool(MockHttpTool("http1"))
        registry.register_tool(MockDatabaseTool("db1"))
        
        tools = registry.list_tools()
        
        assert len(tools) == 2
        assert "http1" in tools
        assert "db1" in tools
    
    def test_registry_as_context_manager(self):
        """Test using registry as a context manager."""
        original_registry = Mock()
        
        with patch('namel3ss.tools.get_tool_registry', return_value=original_registry):
            registry = MockToolRegistry()
            tool = MockHttpTool("test_tool")
            registry.register_tool(tool)
            
            with registry:
                # Inside context, should use mock registry
                pass
            
            # Outside context, should restore original


class TestInputPatternMatching:
    """Test the input pattern matching utility."""
    
    def test_exact_dict_match(self):
        """Test exact dictionary matching."""
        pattern = {"action": "get", "resource": "users"}
        input_data = {"action": "get", "resource": "users"}
        
        assert _match_input_pattern(pattern, input_data) is True
    
    def test_partial_dict_match(self):
        """Test partial dictionary matching."""
        pattern = {"action": "get"}
        input_data = {"action": "get", "resource": "users", "limit": 10}
        
        assert _match_input_pattern(pattern, input_data) is True
    
    def test_dict_no_match(self):
        """Test dictionary non-matching."""
        pattern = {"action": "get", "resource": "posts"}
        input_data = {"action": "get", "resource": "users"}
        
        assert _match_input_pattern(pattern, input_data) is False
    
    def test_missing_required_field(self):
        """Test missing required field in input."""
        pattern = {"action": "get", "resource": "users"}
        input_data = {"action": "get"}
        
        assert _match_input_pattern(pattern, input_data) is False
    
    def test_string_exact_match(self):
        """Test exact string matching."""
        pattern = "SELECT * FROM users"
        input_data = "SELECT * FROM users"
        
        assert _match_input_pattern(pattern, input_data) is True
    
    def test_string_no_match(self):
        """Test string non-matching."""
        pattern = "SELECT * FROM users"
        input_data = "SELECT * FROM posts"
        
        assert _match_input_pattern(pattern, input_data) is False
    
    def test_nested_dict_match(self):
        """Test nested dictionary matching."""
        pattern = {
            "request": {
                "method": "POST",
                "headers": {"Content-Type": "application/json"}
            }
        }
        input_data = {
            "request": {
                "method": "POST", 
                "url": "/api/users",
                "headers": {"Content-Type": "application/json", "Authorization": "Bearer token"}
            },
            "timeout": 5000
        }
        
        assert _match_input_pattern(pattern, input_data) is True
    
    def test_list_exact_match(self):
        """Test exact list matching."""
        pattern = ["item1", "item2"]
        input_data = ["item1", "item2"]
        
        assert _match_input_pattern(pattern, input_data) is True
    
    def test_list_no_match(self):
        """Test list non-matching."""
        pattern = ["item1", "item2"]
        input_data = ["item1", "item3"]
        
        assert _match_input_pattern(pattern, input_data) is False


class TestCreateMockRegistryFromSpecs:
    """Test creating mock registry from specifications."""
    
    def test_create_registry_from_single_spec(self):
        """Test creating registry from a single MockToolSpec."""
        spec = MockToolSpec(
            tool_name="http_client",
            tool_type="http",
            input_pattern={"method": "GET", "url": "https://api.example.com"},
            response=MockToolResponse(
                output={"status": 200, "data": "success"},
                success=True
            )
        )
        
        registry = create_mock_registry_from_specs([spec])
        
        tool = registry.get_tool("http_client")
        assert isinstance(tool, MockHttpTool)
        
        result = tool.execute({"method": "GET", "url": "https://api.example.com"})
        assert result.output["status"] == 200
    
    def test_create_registry_from_multiple_specs(self):
        """Test creating registry from multiple specs."""
        specs = [
            MockToolSpec(
                tool_name="http_client",
                tool_type="http",
                input_pattern={"method": "GET"},
                response=MockToolResponse(output={"http": "response"})
            ),
            MockToolSpec(
                tool_name="db_client", 
                tool_type="database",
                input_pattern={"query": "SELECT 1"},
                response=MockToolResponse(output={"db": "response"})
            )
        ]
        
        registry = create_mock_registry_from_specs(specs)
        
        assert len(registry.list_tools()) == 2
        assert "http_client" in registry.list_tools()
        assert "db_client" in registry.list_tools()
    
    def test_create_registry_with_unknown_tool_type(self):
        """Test that unknown tool types raise ValueError."""
        spec = MockToolSpec(
            tool_name="unknown_tool",
            tool_type="unknown_type",
            input_pattern={},
            response=MockToolResponse(output="test")
        )
        
        with pytest.raises(ValueError, match="Unknown tool type"):
            create_mock_registry_from_specs([spec])
    
    def test_create_registry_empty_specs(self):
        """Test creating registry with empty specs list."""
        registry = create_mock_registry_from_specs([])
        
        assert len(registry.list_tools()) == 0