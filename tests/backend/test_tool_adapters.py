"""
Tests for tool adapters (OpenAPI, LangChain, LLM).
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from n3_server.adapters import (
    OpenAPIAdapter,
    LangChainAdapter,
    LLMToolWrapper,
    create_llm_tool,
)


@pytest.mark.asyncio
async def test_openapi_adapter_import_from_dict():
    """Test importing tools from OpenAPI spec dictionary."""
    spec = {
        "openapi": "3.0.0",
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/users": {
                "get": {
                    "operationId": "list_users",
                    "summary": "List all users",
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "schema": {"type": "integer"},
                            "required": False,
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Success",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "array"}
                                }
                            },
                        }
                    },
                }
            }
        },
    }
    
    adapter = OpenAPIAdapter()
    tools = await adapter.import_from_dict(spec)
    
    assert len(tools) == 1
    tool_func = tools[0]
    assert tool_func.__name__ == "list_users"
    assert hasattr(tool_func, "_tool_metadata")
    assert tool_func._tool_metadata["source"] == "openapi"
    assert "openapi" in tool_func._tool_metadata["tags"]


@pytest.mark.asyncio
async def test_openapi_adapter_with_path_parameters():
    """Test OpenAPI tool with path parameters."""
    spec = {
        "openapi": "3.0.0",
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/users/{id}": {
                "get": {
                    "operationId": "get_user",
                    "summary": "Get user by ID",
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "schema": {"type": "string"},
                            "required": True,
                        }
                    ],
                }
            }
        },
    }
    
    adapter = OpenAPIAdapter()
    tools = await adapter.import_from_dict(spec)
    
    assert len(tools) == 1
    tool_func = tools[0]
    
    # Check input schema includes path parameter
    input_schema = tool_func._tool_metadata["input_schema"]
    assert "id" in input_schema["properties"]
    assert "id" in input_schema["required"]


@pytest.mark.asyncio
async def test_openapi_adapter_with_request_body():
    """Test OpenAPI tool with request body."""
    spec = {
        "openapi": "3.0.0",
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/users": {
                "post": {
                    "operationId": "create_user",
                    "summary": "Create a new user",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"type": "object"}
                            }
                        },
                    },
                }
            }
        },
    }
    
    adapter = OpenAPIAdapter()
    tools = await adapter.import_from_dict(spec)
    
    assert len(tools) == 1
    tool_func = tools[0]
    
    # Check input schema includes body
    input_schema = tool_func._tool_metadata["input_schema"]
    assert "body" in input_schema["properties"]
    assert "body" in input_schema["required"]


@pytest.mark.asyncio
async def test_openapi_adapter_operation_filter():
    """Test filtering operations during import."""
    spec = {
        "openapi": "3.0.0",
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/users": {
                "get": {"operationId": "list_users", "summary": "List users"},
                "post": {"operationId": "create_user", "summary": "Create user"},
            }
        },
    }
    
    adapter = OpenAPIAdapter()
    
    # Filter to only include list_users
    tools = await adapter.import_from_dict(
        spec,
        operation_filter=lambda op: op.operation_id == "list_users",
    )
    
    assert len(tools) == 1
    assert tools[0].__name__ == "list_users"


@pytest.mark.asyncio
async def test_langchain_adapter_import_tool():
    """Test importing a LangChain tool."""
    # Mock LangChain tool
    mock_tool = Mock()
    mock_tool.name = "search"
    mock_tool.description = "Search the web"
    mock_tool.run = Mock(return_value="Search results")
    mock_tool.args_schema = None
    
    adapter = LangChainAdapter()
    tool_func = adapter.import_tool(mock_tool)
    
    assert tool_func.__name__ == "search"
    assert tool_func.__doc__ == "Search the web"
    assert hasattr(tool_func, "_tool_metadata")
    assert tool_func._tool_metadata["source"] == "langchain"


@pytest.mark.asyncio
async def test_langchain_adapter_strip_prefix():
    """Test stripping langchain_ prefix from tool names."""
    mock_tool = Mock()
    mock_tool.name = "langchain_search"
    mock_tool.description = "Search tool"
    mock_tool.run = Mock(return_value="Results")
    mock_tool.args_schema = None
    
    adapter = LangChainAdapter()
    tool_func = adapter.import_tool(mock_tool, strip_prefix=True)
    
    assert tool_func.__name__ == "search"


@pytest.mark.asyncio
async def test_langchain_adapter_with_name_prefix():
    """Test adding name prefix to LangChain tools."""
    mock_tool = Mock()
    mock_tool.name = "search"
    mock_tool.description = "Search tool"
    mock_tool.run = Mock(return_value="Results")
    mock_tool.args_schema = None
    
    adapter = LangChainAdapter()
    tool_func = adapter.import_tool(mock_tool, name_prefix="web_")
    
    assert tool_func.__name__ == "web_search"


@pytest.mark.asyncio
async def test_langchain_adapter_import_multiple():
    """Test importing multiple LangChain tools."""
    tools = []
    for i in range(3):
        tool = Mock()
        tool.name = f"tool_{i}"
        tool.description = f"Tool {i}"
        tool.run = Mock(return_value=f"Result {i}")
        tool.args_schema = None
        tools.append(tool)
    
    adapter = LangChainAdapter()
    imported = adapter.import_tools(tools)
    
    assert len(imported) == 3
    assert [t.__name__ for t in imported] == ["tool_0", "tool_1", "tool_2"]


@pytest.mark.asyncio
async def test_llm_tool_wrapper_create_tool():
    """Test creating an LLM-powered tool."""
    from namel3ss.llm import LLMRegistry, BaseLLM, LLMResponse, ChatMessage
    
    # Mock LLM
    mock_llm = Mock(spec=BaseLLM)
    mock_llm.name = "test_llm"
    mock_llm.provider = "openai"
    mock_llm.generate_chat_async = AsyncMock(
        return_value=LLMResponse(
            text="Summarized text",
            raw={},
            model="gpt-4",
        )
    )
    
    # Register mock LLM
    registry = LLMRegistry()
    registry.register(mock_llm)
    
    # Create tool
    wrapper = LLMToolWrapper()
    wrapper.llm_registry = registry
    
    tool_func = wrapper.create_tool(
        name="summarize",
        description="Summarize text",
        llm_name="test_llm",
        system_prompt="You are a summarizer.",
    )
    
    assert tool_func.__name__ == "summarize"
    assert tool_func.__doc__ == "Summarize text"
    assert hasattr(tool_func, "_tool_metadata")
    assert tool_func._tool_metadata["source"] == "llm"


@pytest.mark.asyncio
async def test_llm_tool_wrapper_execute_tool():
    """Test executing an LLM-powered tool."""
    from namel3ss.llm import LLMRegistry, BaseLLM, LLMResponse, ChatMessage
    
    # Mock LLM
    mock_llm = Mock(spec=BaseLLM)
    mock_llm.name = "test_llm"
    mock_llm.provider = "openai"
    mock_llm.generate_chat_async = AsyncMock(
        return_value=LLMResponse(
            text="Summary of the text",
            raw={},
            model="gpt-4",
        )
    )
    
    # Register mock LLM
    registry = LLMRegistry()
    registry.register(mock_llm)
    
    # Create and execute tool
    wrapper = LLMToolWrapper()
    wrapper.llm_registry = registry
    
    tool_func = wrapper.create_tool(
        name="summarize",
        description="Summarize text",
        llm_name="test_llm",
    )
    
    result = await tool_func(input="Long text to summarize")
    
    assert result == "Summary of the text"
    mock_llm.generate_chat_async.assert_called_once()


@pytest.mark.asyncio
async def test_llm_tool_wrapper_json_response():
    """Test LLM tool with JSON response format."""
    from namel3ss.llm import LLMRegistry, BaseLLM, LLMResponse
    from namel3ss.llm.openai_llm import OpenAILLM
    
    # Mock OpenAI LLM (supports response_format)
    mock_llm = Mock(spec=OpenAILLM)
    mock_llm.name = "test_openai"
    mock_llm.provider = "openai"
    mock_llm.generate_chat_async = AsyncMock(
        return_value=LLMResponse(
            text='{"summary": "Test summary", "keywords": ["test", "summary"]}',
            raw={},
            model="gpt-4",
        )
    )
    
    # Register mock LLM
    registry = LLMRegistry()
    registry.register(mock_llm)
    
    # Create tool with JSON response
    wrapper = LLMToolWrapper()
    wrapper.llm_registry = registry
    
    output_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "keywords": {"type": "array"},
        },
    }
    
    tool_func = wrapper.create_tool(
        name="extract_info",
        description="Extract information",
        llm_name="test_openai",
        response_format="json",
        output_schema=output_schema,
    )
    
    result = await tool_func(input="Some text")
    
    assert isinstance(result, dict)
    assert "summary" in result
    assert "keywords" in result


@pytest.mark.asyncio
async def test_create_llm_tool_convenience():
    """Test create_llm_tool convenience function."""
    from namel3ss.llm import LLMRegistry, BaseLLM, LLMResponse
    
    # Mock LLM
    mock_llm = Mock(spec=BaseLLM)
    mock_llm.name = "test_llm"
    mock_llm.provider = "test"
    mock_llm.generate_chat_async = AsyncMock(
        return_value=LLMResponse(text="Result", raw={}, model="test")
    )
    
    # Register mock LLM
    registry = LLMRegistry()
    registry.register(mock_llm)
    
    # Create tool using convenience function
    with patch("n3_server.adapters.llm_tool_wrapper.get_llm_registry", return_value=registry):
        tool_func = create_llm_tool(
            name="test_tool",
            description="Test tool",
            llm_name="test_llm",
        )
    
    assert callable(tool_func)
    assert tool_func.__name__ == "test_tool"


@pytest.mark.asyncio
async def test_openapi_adapter_close():
    """Test closing OpenAPI adapter HTTP client."""
    adapter = OpenAPIAdapter()
    
    # Create client
    await adapter._get_client()
    assert adapter._http_client is not None
    
    # Close client
    await adapter.close()
    assert adapter._http_client is None
