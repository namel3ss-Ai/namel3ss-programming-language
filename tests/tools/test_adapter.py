"""Unit tests for Tool Adapter Framework core components."""

import asyncio
import logging
from datetime import datetime
from typing import AsyncIterator

import pytest
from pydantic import Field

from namel3ss.tools.adapter import (
    BaseToolAdapter,
    StreamingToolAdapter,
    ToolAdapter,
    ToolCategory,
    ToolConfig,
    ToolContext,
    ToolMetadata,
)
from namel3ss.tools.errors import (
    ToolExecutionError,
    ToolTimeoutError,
    ToolValidationError,
)
from namel3ss.tools.schemas import (
    ToolChunkModel,
    ToolInputModel,
    ToolOutputModel,
)


# Test schemas

class TestInput(ToolInputModel):
    """Test input schema."""
    query: str = Field(..., description="Query string")
    limit: int = Field(10, ge=1, le=100, description="Result limit")


class TestOutput(ToolOutputModel):
    """Test output schema."""
    result: str = Field(..., description="Result string")
    count: int = Field(..., description="Result count")


class TestChunk(ToolChunkModel):
    """Test chunk schema for streaming."""
    token: str = Field(..., description="Token")


# Test adapters

class SimpleToolAdapter(BaseToolAdapter[TestInput, TestOutput]):
    """Simple test tool adapter."""
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="simple_tool",
            description="A simple test tool",
            version="1.0.0",
            category=ToolCategory.CUSTOM,
            tags=("test", "simple"),
        )
    
    def get_input_schema(self) -> type[TestInput]:
        return TestInput
    
    def get_output_schema(self) -> type[TestOutput]:
        return TestOutput
    
    async def invoke(
        self,
        input: TestInput,
        context: ToolContext,
    ) -> TestOutput:
        context.log("info", f"Processing query: {input.query}")
        return TestOutput(
            result=f"Processed: {input.query}",
            count=input.limit,
        )


class ErrorToolAdapter(BaseToolAdapter[TestInput, TestOutput]):
    """Tool that raises errors for testing."""
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="error_tool",
            description="Tool that raises errors",
            version="1.0.0",
        )
    
    def get_input_schema(self) -> type[TestInput]:
        return TestInput
    
    def get_output_schema(self) -> type[TestOutput]:
        return TestOutput
    
    async def invoke(
        self,
        input: TestInput,
        context: ToolContext,
    ) -> TestOutput:
        raise ToolExecutionError(
            "Tool execution failed",
            code="TOOL031",
            tool_name="error_tool",
            retryable=True,
        )


class TimeoutToolAdapter(BaseToolAdapter[TestInput, TestOutput]):
    """Tool that simulates timeout."""
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="timeout_tool",
            description="Tool that times out",
            version="1.0.0",
        )
    
    def get_input_schema(self) -> type[TestInput]:
        return TestInput
    
    def get_output_schema(self) -> type[TestOutput]:
        return TestOutput
    
    async def invoke(
        self,
        input: TestInput,
        context: ToolContext,
    ) -> TestOutput:
        await asyncio.sleep(2.0)  # Simulate long operation
        return TestOutput(result="done", count=1)


class StreamingTestAdapter:
    """Streaming tool adapter for testing."""
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="streaming_tool",
            description="A streaming test tool",
            version="1.0.0",
        )
    
    def get_input_schema(self) -> type[TestInput]:
        return TestInput
    
    def get_output_schema(self) -> type[TestChunk]:
        return TestChunk
    
    async def invoke_stream(
        self,
        input: TestInput,
        context: ToolContext,
    ) -> AsyncIterator[TestChunk]:
        """Stream tokens from query."""
        tokens = input.query.split()
        for i, token in enumerate(tokens):
            chunk = TestChunk(
                token=token,
                sequence=i,
                is_final=(i == len(tokens) - 1),
            )
            yield chunk
            await asyncio.sleep(0.01)  # Simulate streaming delay


# Tests

@pytest.mark.asyncio
async def test_tool_adapter_protocol():
    """Test ToolAdapter protocol compliance."""
    adapter = SimpleToolAdapter()
    
    # Check protocol compliance
    assert isinstance(adapter, ToolAdapter)
    
    # Check methods exist
    assert hasattr(adapter, "get_metadata")
    assert hasattr(adapter, "get_input_schema")
    assert hasattr(adapter, "get_output_schema")
    assert hasattr(adapter, "get_config")
    assert hasattr(adapter, "invoke")


@pytest.mark.asyncio
async def test_tool_metadata():
    """Test ToolMetadata functionality."""
    metadata = ToolMetadata(
        name="test_tool",
        description="Test tool description",
        version="1.2.3",
        category=ToolCategory.HTTP,
        tags=("api", "external"),
        author="Test Author",
        source="openapi",
        namespace="integrations",
    )
    
    assert metadata.name == "test_tool"
    assert metadata.description == "Test tool description"
    assert metadata.version == "1.2.3"
    assert metadata.category == ToolCategory.HTTP
    assert metadata.tags == ("api", "external")
    assert metadata.author == "Test Author"
    assert metadata.source == "openapi"
    assert metadata.namespace == "integrations"
    
    # Test to_dict
    metadata_dict = metadata.to_dict()
    assert metadata_dict["name"] == "test_tool"
    assert metadata_dict["category"] == "http"
    assert metadata_dict["tags"] == ["api", "external"]
    
    # Test get_full_name
    assert metadata.get_full_name() == "integrations.test_tool"


@pytest.mark.asyncio
async def test_tool_config():
    """Test ToolConfig functionality."""
    config = ToolConfig(
        timeout=60.0,
        max_retries=3,
        retry_delay=2.0,
        auth_token="test-token",
        rate_limit_per_minute=100,
        custom={"key": "value"},
    )
    
    assert config.timeout == 60.0
    assert config.max_retries == 3
    assert config.retry_delay == 2.0
    assert config.auth_token == "test-token"
    assert config.rate_limit_per_minute == 100
    
    # Test custom config
    assert config.get("key") == "value"
    assert config.get("missing", "default") == "default"
    
    config.set("new_key", "new_value")
    assert config.get("new_key") == "new_value"


@pytest.mark.asyncio
async def test_tool_context():
    """Test ToolContext functionality."""
    logger = logging.getLogger("test")
    
    context = ToolContext(
        logger=logger,
        correlation_id="test-123",
        user_id="user-456",
        environment={"env": "test"},
    )
    
    assert context.logger == logger
    assert context.correlation_id == "test-123"
    assert context.user_id == "user-456"
    assert context.environment["env"] == "test"
    assert isinstance(context.timestamp, datetime)


@pytest.mark.asyncio
async def test_simple_tool_adapter():
    """Test simple tool adapter execution."""
    adapter = SimpleToolAdapter()
    
    # Get metadata
    metadata = adapter.get_metadata()
    assert metadata.name == "simple_tool"
    assert metadata.version == "1.0.0"
    
    # Get schemas
    input_schema = adapter.get_input_schema()
    output_schema = adapter.get_output_schema()
    assert input_schema == TestInput
    assert output_schema == TestOutput
    
    # Invoke tool
    logger = logging.getLogger("test")
    context = ToolContext(logger=logger)
    
    input_data = TestInput(query="hello world", limit=5)
    output = await adapter.invoke(input_data, context)
    
    assert isinstance(output, TestOutput)
    assert output.result == "Processed: hello world"
    assert output.count == 5


@pytest.mark.asyncio
async def test_tool_adapter_validation():
    """Test input/output validation."""
    adapter = SimpleToolAdapter()
    
    # Valid input
    valid_input = {"query": "test", "limit": 10}
    validated = adapter.validate_input(valid_input)
    assert isinstance(validated, TestInput)
    assert validated.query == "test"
    assert validated.limit == 10
    
    # Invalid input (missing required field)
    with pytest.raises(ToolValidationError) as exc_info:
        adapter.validate_input({"limit": 10})
    assert "TOOL003" in str(exc_info.value)
    
    # Invalid input (constraint violation)
    with pytest.raises(ToolValidationError) as exc_info:
        adapter.validate_input({"query": "test", "limit": 200})
    assert "TOOL003" in str(exc_info.value)


@pytest.mark.asyncio
async def test_error_tool_adapter():
    """Test tool that raises errors."""
    adapter = ErrorToolAdapter()
    logger = logging.getLogger("test")
    context = ToolContext(logger=logger)
    
    input_data = TestInput(query="test", limit=10)
    
    with pytest.raises(ToolExecutionError) as exc_info:
        await adapter.invoke(input_data, context)
    
    error = exc_info.value
    assert error.code == "TOOL031"
    assert error.tool_name == "error_tool"
    assert error.retryable is True


@pytest.mark.asyncio
async def test_timeout_tool_adapter():
    """Test tool timeout handling."""
    adapter = TimeoutToolAdapter(config=ToolConfig(timeout=0.5))
    logger = logging.getLogger("test")
    context = ToolContext(logger=logger)
    
    input_data = TestInput(query="test", limit=10)
    
    # Should timeout after 0.5 seconds
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            adapter.invoke(input_data, context),
            timeout=adapter.get_config().timeout,
        )


@pytest.mark.asyncio
async def test_streaming_tool_adapter():
    """Test streaming tool adapter."""
    adapter = StreamingTestAdapter()
    logger = logging.getLogger("test")
    context = ToolContext(logger=logger)
    
    input_data = TestInput(query="hello world test", limit=10)
    
    chunks = []
    async for chunk in adapter.invoke_stream(input_data, context):
        assert isinstance(chunk, TestChunk)
        chunks.append(chunk)
    
    assert len(chunks) == 3
    assert chunks[0].token == "hello"
    assert chunks[0].sequence == 0
    assert chunks[0].is_final is False
    
    assert chunks[1].token == "world"
    assert chunks[1].sequence == 1
    
    assert chunks[2].token == "test"
    assert chunks[2].sequence == 2
    assert chunks[2].is_final is True


@pytest.mark.asyncio
async def test_streaming_adapter_protocol():
    """Test StreamingToolAdapter protocol compliance."""
    adapter = StreamingTestAdapter()
    
    # Check protocol compliance
    assert isinstance(adapter, StreamingToolAdapter)
    assert hasattr(adapter, "invoke_stream")


@pytest.mark.asyncio
async def test_tool_config_defaults():
    """Test ToolConfig default values."""
    config = ToolConfig()
    
    assert config.timeout == 30.0
    assert config.max_retries == 0
    assert config.retry_delay == 1.0
    assert config.auth_token is None
    assert config.rate_limit_per_minute is None
    assert config.cache_ttl is None
    assert config.endpoints == {}
    assert config.headers == {}
    assert config.custom == {}


@pytest.mark.asyncio
async def test_base_tool_adapter_config():
    """Test BaseToolAdapter configuration."""
    custom_config = ToolConfig(timeout=120.0, max_retries=5)
    adapter = SimpleToolAdapter(config=custom_config)
    
    config = adapter.get_config()
    assert config.timeout == 120.0
    assert config.max_retries == 5


@pytest.mark.asyncio
async def test_tool_context_logging():
    """Test ToolContext logging functionality."""
    import io
    import logging
    
    # Create logger with string handler
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(io.StringIO())
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    context = ToolContext(
        logger=logger,
        correlation_id="test-123",
        user_id="user-456",
    )
    
    # Log message
    context.log("info", "Test message", extra_field="value")
    
    # Should not raise
    context.log("error", "Error message")
    context.log("warning", "Warning message")
    context.log("debug", "Debug message")


@pytest.mark.asyncio
async def test_tool_metadata_immutability():
    """Test ToolMetadata is immutable."""
    metadata = ToolMetadata(
        name="test",
        description="Test",
        version="1.0.0",
    )
    
    # Should not be able to modify
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        metadata.name = "modified"  # type: ignore


@pytest.mark.asyncio
async def test_multiple_tool_adapters():
    """Test multiple tool adapters can coexist."""
    adapter1 = SimpleToolAdapter()
    adapter2 = ErrorToolAdapter()
    adapter3 = TimeoutToolAdapter()
    
    metadata1 = adapter1.get_metadata()
    metadata2 = adapter2.get_metadata()
    metadata3 = adapter3.get_metadata()
    
    assert metadata1.name == "simple_tool"
    assert metadata2.name == "error_tool"
    assert metadata3.name == "timeout_tool"
    
    assert metadata1.name != metadata2.name
    assert metadata2.name != metadata3.name
