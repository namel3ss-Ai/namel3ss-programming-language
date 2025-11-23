---

# Tool Adapter Framework

**Production-Grade Tool Development for Namel3ss (N3)**

Version: 1.0.0  
Status: Production Ready  
Last Updated: 2025-11-21

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Concepts](#core-concepts)
4. [Python Tool Authoring Guide](#python-tool-authoring-guide)
5. [API Reference](#api-reference)
6. [Integration](#integration)
7. [Best Practices](#best-practices)
8. [Examples](#examples)
9. [Testing](#testing)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The **Tool Adapter Framework** is a production-grade abstraction layer for creating, managing, and executing tools in the Namel3ss ecosystem. It provides:

### Key Features

- **🔒 Type Safety**: Strongly-typed I/O using Pydantic v2 schemas
- **✅ Validation**: Automatic input/output validation with detailed error messages
- **📊 Observability**: Built-in logging, tracing (OpenTelemetry), and metrics
- **🔄 Streaming**: First-class support for streaming responses
- **⚙️ Configuration**: Flexible per-tool configuration (timeouts, retries, auth)
- **🏷️ Metadata**: Rich metadata for tool discovery and code generation
- **🔌 Extensible**: Protocol-based design supports any tool implementation
- **🧪 Testable**: Comprehensive test utilities and mocking support

### Design Principles

1. **Language-Agnostic Concepts**: Core abstractions can be implemented in other languages
2. **Idiomatic Python**: Uses modern Python features (typing, protocols, async)
3. **Codegen-Friendly**: Machine-readable metadata for SDK generation
4. **Production-Ready**: No shortcuts, no toy code, enterprise-grade quality

### Components

```
Tool Adapter Framework
├── ToolAdapter Protocol       # Core interface
├── Pydantic Schemas           # Input/Output models
├── Error Hierarchy            # Structured error handling
├── Streaming Support          # AsyncIterator-based streaming
├── ToolRegistry               # Discovery and registration
└── Integration Hooks          # Runtime and codegen integration
```

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     N3 DSL / AST Layer                       │
│         (tool declarations, chain steps, agent configs)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Tool Adapter Interface                     │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │  Metadata  │  │  I/O Schema │  │   Configuration   │    │
│  └────────────┘  └─────────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Concrete Tool Implementations                   │
│  ┌──────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐   │
│  │ HTTP │  │ LangChain│  │  OpenAPI │  │     LLM     │   │
│  └──────┘  └──────────┘  └──────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Runtime Execution                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐ │
│  │  Tracing │  │  Metrics │  │  Error Handling/Retries  │ │
│  └──────────┘  └──────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Component Interactions

```
┌─────────────┐                  ┌──────────────┐
│   Client    │                  │  ToolAdapter │
│   (N3 DSL)  │                  │   Instance   │
└──────┬──────┘                  └──────┬───────┘
       │                                │
       │ 1. Get Metadata                │
       ├───────────────────────────────►│
       │                                │
       │ 2. Validate Input (Pydantic)   │
       ├───────────────────────────────►│
       │                                │
       │ 3. Invoke with Context         │
       ├───────────────────────────────►│
       │                                │
       │                      ┌─────────▼────────┐
       │                      │  Tool Logic      │
       │                      │  (HTTP, Python,  │
       │                      │   LLM, etc.)     │
       │                      └─────────┬────────┘
       │                                │
       │ 4. Validate Output             │
       │◄───────────────────────────────┤
       │                                │
       │ 5. Return Result               │
       │◄───────────────────────────────┤
       │                                │
```

### Data Flow

1. **Tool Registration**: Tools register with ToolRegistry at startup
2. **Metadata Discovery**: Runtime queries tool metadata for execution planning
3. **Input Validation**: Pydantic validates input against tool's schema
4. **Execution**: Tool logic executes with observability context
5. **Output Validation**: Pydantic validates output before returning
6. **Error Handling**: Structured errors propagated to caller

---

## Core Concepts

### 1. ToolAdapter Protocol

The core interface that all tools must implement.

```python
from namel3ss.tools.adapter import ToolAdapter, ToolMetadata, ToolConfig, ToolContext

class MyTool(ToolAdapter[InputModel, OutputModel]):
    def get_metadata(self) -> ToolMetadata:
        """Return tool metadata."""
        ...
    
    def get_input_schema(self) -> type[InputModel]:
        """Return input schema class."""
        ...
    
    def get_output_schema(self) -> type[OutputModel]:
        """Return output schema class."""
        ...
    
    def get_config(self) -> ToolConfig:
        """Return tool configuration."""
        ...
    
    async def invoke(self, input: InputModel, context: ToolContext) -> OutputModel:
        """Execute the tool."""
        ...
```

### 2. ToolMetadata

Immutable metadata describing the tool.

```python
from namel3ss.tools.adapter import ToolMetadata, ToolCategory

metadata = ToolMetadata(
    name="weather_api",
    description="Get current weather conditions",
    version="1.0.0",
    category=ToolCategory.HTTP,
    tags=("weather", "api", "external"),
    author="Platform Team",
    source="openapi",
    namespace="integrations.weather"
)
```

**Fields:**
- `name`: Unique identifier within namespace
- `description`: Human-readable description (used in prompts, docs)
- `version`: Semantic version (e.g., "1.2.3")
- `category`: Tool category (HTTP, PYTHON, DATABASE, LLM, etc.)
- `tags`: Labels for search/filtering
- `author`: Tool author/maintainer
- `source`: Source system (e.g., "openapi", "langchain")
- `namespace`: Logical grouping (e.g., "integrations.weather")

### 3. Pydantic Schemas

Type-safe input/output models using Pydantic v2.

```python
from namel3ss.tools.schemas import ToolInputModel, ToolOutputModel
from pydantic import Field

class WeatherInput(ToolInputModel):
    """Weather API input."""
    location: str = Field(..., description="City name or coordinates")
    units: str = Field("metric", description="Temperature units")
    include_forecast: bool = Field(False, description="Include 7-day forecast")

class WeatherOutput(ToolOutputModel):
    """Weather API output."""
    temperature: float = Field(..., description="Current temperature")
    condition: str = Field(..., description="Weather condition")
    humidity: int = Field(..., ge=0, le=100, description="Humidity %")
```

**Key Features:**
- Automatic validation (type checking, constraints, required fields)
- JSON Schema export for codegen
- Rich field descriptions for prompts and documentation
- Nested model support
- Default values and optional fields

### 4. ToolConfig

Per-tool configuration for execution behavior.

```python
from namel3ss.tools.adapter import ToolConfig

config = ToolConfig(
    timeout=30.0,
    max_retries=3,
    retry_delay=1.0,
    auth_token="sk-...",
    rate_limit_per_minute=60,
    cache_ttl=300,
    custom={"region": "us-west-2"}
)
```

**Configuration Options:**
- `timeout`: Maximum execution time (seconds)
- `max_retries`: Retry attempts on failure
- `retry_delay`: Delay between retries (seconds)
- `auth_token`: Bearer token or API key
- `auth_header`: Custom auth header name
- `rate_limit_per_minute`: Max calls per minute
- `cache_ttl`: Cache TTL (seconds)
- `endpoints`: Custom endpoint URLs
- `headers`: Custom HTTP headers
- `custom`: Tool-specific parameters

### 5. ToolContext

Execution context with observability hooks.

```python
from namel3ss.tools.adapter import ToolContext
import logging

context = ToolContext(
    logger=logging.getLogger("tools"),
    tracer=tracer,  # OpenTelemetry tracer
    correlation_id="req-123",
    user_id="user-456",
    environment={"region": "us-west-2"}
)

# Use in tool
async def invoke(self, input, context):
    context.log("info", f"Processing {input.query}")
    
    with context.child_span("external_api"):
        result = await call_api()
    
    return OutputModel(result=result)
```

**Context Fields:**
- `logger`: Python logger for tool output
- `tracer`: OpenTelemetry tracer
- `span`: Current trace span
- `correlation_id`: Request correlation ID
- `user_id`: User identifier
- `session_id`: Session identifier
- `environment`: Environment variables/config
- `timestamp`: Context creation time

### 6. Error Hierarchy

Structured errors with machine-readable codes.

```python
from namel3ss.tools.errors import (
    ToolError,
    ToolValidationError,
    ToolExecutionError,
    ToolTimeoutError,
    ToolConfigurationError,
    ToolAuthenticationError,
)

# Validation error
raise ToolValidationError(
    "Invalid input: query is required",
    code="TOOL003",
    tool_name="search",
    field="query",
    expected="string"
)

# Execution error
raise ToolExecutionError(
    "HTTP request failed",
    code="TOOL031",
    tool_name="api_tool",
    operation="GET /v1/data",
    status_code=500,
    retryable=True
)

# Timeout error
raise ToolTimeoutError(
    "Tool execution exceeded timeout",
    code="TOOL034",
    tool_name="slow_api",
    timeout_seconds=30.0,
    elapsed_seconds=30.5
)
```

### 7. Streaming Support

AsyncIterator-based streaming for real-time responses.

```python
from namel3ss.tools.schemas import ToolChunkModel
from typing import AsyncIterator

class TokenChunk(ToolChunkModel):
    token: str

async def invoke_stream(
    self,
    input: InputModel,
    context: ToolContext
) -> AsyncIterator[TokenChunk]:
    """Stream tokens as they're generated."""
    async for token in llm.stream(input.prompt):
        yield TokenChunk(
            token=token,
            sequence=i,
            is_final=(i == total_tokens - 1)
        )
```

**Streaming Utilities:**
- `StreamBuffer`: Buffer and batch chunks
- `StreamAggregator`: Aggregate chunks into result
- `rate_limit_stream`: Rate limiting
- `batch_stream`: Batching
- `filter_stream`: Filtering
- `map_stream`: Transformation
- `collect_stream`: Collect all chunks
- `take_stream`: Limit stream
- `merge_streams`: Merge multiple streams

---

## Python Tool Authoring Guide

### Quick Start

**1. Define Schemas**

```python
from namel3ss.tools.schemas import ToolInputModel, ToolOutputModel
from pydantic import Field

class CalculatorInput(ToolInputModel):
    expression: str = Field(..., description="Mathematical expression")

class CalculatorOutput(ToolOutputModel):
    result: float = Field(..., description="Calculation result")
```

**2. Implement Adapter**

```python
from namel3ss.tools.adapter import BaseToolAdapter, ToolMetadata, ToolCategory

class CalculatorTool(BaseToolAdapter[CalculatorInput, CalculatorOutput]):
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="calculator",
            description="Perform mathematical calculations",
            version="1.0.0",
            category=ToolCategory.PYTHON
        )
    
    def get_input_schema(self) -> type[CalculatorInput]:
        return CalculatorInput
    
    def get_output_schema(self) -> type[CalculatorOutput]:
        return CalculatorOutput
    
    async def invoke(
        self,
        input: CalculatorInput,
        context: ToolContext
    ) -> CalculatorOutput:
        context.log("info", f"Calculating: {input.expression}")
        result = eval(input.expression)  # Note: Use safely in production!
        return CalculatorOutput(result=result)
```

**3. Register and Use**

```python
from namel3ss.tools import get_registry

# Register
registry = get_registry()
calculator = CalculatorTool()
registry.register_adapter(calculator)

# Use
tool = registry.get("calculator")
context = ToolContext(logger=logging.getLogger("tools"))
input_data = CalculatorInput(expression="2 + 2")
result = await tool.invoke(input_data, context)
print(result.result)  # 4.0
```

### HTTP Tool Example

```python
import httpx
from namel3ss.tools.adapter import BaseToolAdapter, ToolMetadata, ToolCategory, ToolConfig
from namel3ss.tools.schemas import ToolInputModel, ToolOutputModel
from namel3ss.tools.errors import ToolExecutionError
from pydantic import Field

class HTTPInput(ToolInputModel):
    method: str = Field("GET", description="HTTP method")
    url: str = Field(..., description="Request URL")
    headers: dict = Field(default_factory=dict)
    params: dict = Field(default_factory=dict)
    body: dict = Field(None, description="Request body")

class HTTPOutput(ToolOutputModel):
    status_code: int
    body: dict
    headers: dict

class HTTPTool(BaseToolAdapter[HTTPInput, HTTPOutput]):
    def __init__(self, config: ToolConfig = None):
        super().__init__(config or ToolConfig(timeout=30.0))
        self._client = httpx.AsyncClient()
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="http_client",
            description="Make HTTP requests",
            version="1.0.0",
            category=ToolCategory.HTTP,
            tags=("http", "api"),
        )
    
    def get_input_schema(self) -> type[HTTPInput]:
        return HTTPInput
    
    def get_output_schema(self) -> type[HTTPOutput]:
        return HTTPOutput
    
    async def invoke(
        self,
        input: HTTPInput,
        context: ToolContext
    ) -> HTTPOutput:
        config = self.get_config()
        
        try:
            with context.child_span("http_request"):
                response = await self._client.request(
                    method=input.method,
                    url=input.url,
                    headers=input.headers,
                    params=input.params,
                    json=input.body,
                    timeout=config.timeout,
                )
            
            return HTTPOutput(
                status_code=response.status_code,
                body=response.json(),
                headers=dict(response.headers),
            )
        
        except httpx.TimeoutException as e:
            raise ToolTimeoutError(
                f"Request timed out after {config.timeout}s",
                code="TOOL034",
                tool_name=self.get_metadata().name,
                timeout_seconds=config.timeout,
            ) from e
        
        except Exception as e:
            raise ToolExecutionError(
                f"HTTP request failed: {e}",
                code="TOOL031",
                tool_name=self.get_metadata().name,
                retryable=True,
                original_error=e,
            ) from e
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()
```

### Streaming Tool Example

```python
from namel3ss.tools.schemas import ToolChunkModel
from typing import AsyncIterator

class LLMChunk(ToolChunkModel):
    token: str
    finish_reason: str = None

class LLMInput(ToolInputModel):
    prompt: str
    max_tokens: int = 100

class StreamingLLMTool:
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="streaming_llm",
            description="Stream LLM completions",
            version="1.0.0",
            category=ToolCategory.LLM,
        )
    
    def get_input_schema(self) -> type[LLMInput]:
        return LLMInput
    
    def get_output_schema(self) -> type[LLMChunk]:
        return LLMChunk
    
    async def invoke_stream(
        self,
        input: LLMInput,
        context: ToolContext
    ) -> AsyncIterator[LLMChunk]:
        """Stream tokens from LLM."""
        context.log("info", f"Streaming completion for: {input.prompt}")
        
        # Stream from LLM
        async for token in llm_client.stream(input.prompt, max_tokens=input.max_tokens):
            yield LLMChunk(token=token)
        
        # Final chunk
        yield LLMChunk(token="", finish_reason="stop", is_final=True)
```

### Error Handling Pattern

```python
from namel3ss.tools.errors import (
    ToolValidationError,
    ToolExecutionError,
    ToolTimeoutError,
    serialize_tool_error,
)

async def invoke(self, input, context):
    try:
        # Validate input (already done by framework)
        
        # Execute with retries
        for attempt in range(self.get_config().max_retries + 1):
            try:
                result = await self._execute(input)
                break
            except TemporaryError as e:
                if attempt == self.get_config().max_retries:
                    raise ToolExecutionError(
                        f"Failed after {attempt + 1} attempts",
                        code="TOOL031",
                        tool_name=self.get_metadata().name,
                        retryable=False,
                        original_error=e,
                    )
                await asyncio.sleep(self.get_config().retry_delay)
        
        # Validate output
        return self.validate_output(result)
    
    except asyncio.TimeoutError as e:
        raise ToolTimeoutError(
            "Operation timed out",
            code="TOOL034",
            tool_name=self.get_metadata().name,
            timeout_seconds=self.get_config().timeout,
        ) from e
    
    except Exception as e:
        # Log error with context
        context.log("error", "Tool execution failed", error=serialize_tool_error(e))
        raise
```

---

## API Reference

### ToolAdapter Protocol

```python
class ToolAdapter(Protocol, Generic[TInput, TOutput]):
    def get_metadata(self) -> ToolMetadata: ...
    def get_input_schema(self) -> type[TInput]: ...
    def get_output_schema(self) -> type[TOutput]: ...
    def get_config(self) -> ToolConfig: ...
    async def invoke(self, input: TInput, context: ToolContext) -> TOutput: ...
```

### BaseToolAdapter

Abstract base class providing common functionality.

```python
class BaseToolAdapter(ABC, Generic[TInput, TOutput]):
    def __init__(self, config: Optional[ToolConfig] = None): ...
    @abstractmethod
    def get_metadata(self) -> ToolMetadata: ...
    @abstractmethod
    def get_input_schema(self) -> type[TInput]: ...
    @abstractmethod
    def get_output_schema(self) -> type[TOutput]: ...
    def get_config(self) -> ToolConfig: ...
    @abstractmethod
    async def invoke(self, input: TInput, context: ToolContext) -> TOutput: ...
    def validate_input(self, data: Dict[str, Any]) -> TInput: ...
    def validate_output(self, data: Any) -> TOutput: ...
```

### StreamingToolAdapter Protocol

```python
class StreamingToolAdapter(Protocol, Generic[TInput, TChunk]):
    async def invoke_stream(
        self,
        input: TInput,
        context: ToolContext
    ) -> AsyncIterator[TChunk]: ...
```

### ToolRegistry

```python
class ToolRegistry:
    def register(self, name: str, tool: Union[BaseTool, ToolAdapter]) -> None: ...
    def register_adapter(self, adapter: ToolAdapter) -> str: ...
    def update(self, name: str, tool: Union[BaseTool, ToolAdapter]) -> None: ...
    def get(self, name: str) -> Optional[Union[BaseTool, ToolAdapter]]: ...
    def get_required(self, name: str) -> Union[BaseTool, ToolAdapter]: ...
    def get_adapter(self, name: str) -> Optional[ToolAdapter]: ...
    def has(self, name: str) -> bool: ...
    def list_tools(self) -> Dict[str, str]: ...
    def find_by_tag(self, tag: str) -> List[str]: ...
    def find_by_namespace(self, namespace: str) -> List[str]: ...
    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]: ...
    def clear(self) -> None: ...
```

### Schema Models

```python
class ToolInputModel(BaseModel):
    """Base class for tool inputs."""
    def to_dict(self) -> Dict[str, Any]: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolInputModel: ...
    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]: ...

class ToolOutputModel(BaseModel):
    """Base class for tool outputs."""
    def to_dict(self) -> Dict[str, Any]: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolOutputModel: ...
    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]: ...

class ToolChunkModel(BaseModel):
    """Base class for streaming chunks."""
    sequence: Optional[int] = None
    is_final: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None
```

### Error Classes

```python
class ToolError(Exception):
    """Base tool error."""
    def __init__(
        self,
        message: str,
        *,
        tool_name: Optional[str] = None,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None
    ): ...

class ToolValidationError(ToolError):
    """Validation error."""
    def __init__(
        self,
        message: str,
        *,
        code: str,
        tool_name: Optional[str] = None,
        field: Optional[str] = None,
        value: Any = None,
        tool_type: Optional[str] = None,
        expected: Optional[str] = None,
        original_error: Optional[Exception] = None
    ): ...
    def format(self) -> str: ...

class ToolExecutionError(ToolError):
    """Execution error."""
    def __init__(
        self,
        message: str,
        *,
        code: str,
        tool_name: Optional[str] = None,
        operation: Optional[str] = None,
        status_code: Optional[int] = None,
        timeout: bool = False,
        retryable: bool = False,
        original_error: Optional[Exception] = None
    ): ...
    def format(self) -> str: ...

class ToolTimeoutError(ToolExecutionError):
    """Timeout error."""
    def __init__(
        self,
        message: str,
        *,
        code: str = "TOOL034",
        tool_name: Optional[str] = None,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
        original_error: Optional[Exception] = None
    ): ...

def serialize_tool_error(error: ToolError) -> Dict[str, Any]: ...
def deserialize_tool_error(error_dict: Dict[str, Any]) -> ToolError: ...
```

### Streaming Utilities

```python
class StreamBuffer(Generic[T]):
    def __init__(
        self,
        max_size: int = 100,
        flush_interval: float = 1.0,
        should_flush_fn: Optional[Callable[[List[T]], bool]] = None
    ): ...
    def add(self, chunk: T) -> None: ...
    def should_flush(self) -> bool: ...
    def flush(self) -> List[T]: ...
    def peek(self) -> List[T]: ...
    def size(self) -> int: ...
    def is_empty(self) -> bool: ...

class StreamAggregator(Generic[T, TResult]):
    def __init__(
        self,
        aggregate_fn: Callable[[List[T]], TResult],
        max_chunks: Optional[int] = None
    ): ...
    async def add(self, chunk: T) -> None: ...
    async def get_result(self) -> TResult: ...
    def chunk_count(self) -> int: ...
    def clear(self) -> None: ...

async def rate_limit_stream(
    stream: AsyncIterator[T],
    max_per_second: float
) -> AsyncIterator[T]: ...

async def batch_stream(
    stream: AsyncIterator[T],
    batch_size: int,
    timeout: Optional[float] = None
) -> AsyncIterator[List[T]]: ...

async def filter_stream(
    stream: AsyncIterator[T],
    predicate: Callable[[T], bool]
) -> AsyncIterator[T]: ...

async def map_stream(
    stream: AsyncIterator[T],
    mapper: Callable[[T], T]
) -> AsyncIterator[T]: ...

async def collect_stream(stream: AsyncIterator[T]) -> List[T]: ...

async def take_stream(
    stream: AsyncIterator[T],
    count: int
) -> AsyncIterator[T]: ...

async def merge_streams(*streams: AsyncIterator[T]) -> AsyncIterator[T]: ...
```

---

## Integration

### Runtime Integration

The Tool Adapter Framework integrates with the N3 runtime execution engine:

```python
from n3_server.execution.registry import RuntimeRegistry
from namel3ss.tools import get_registry

# Build runtime registry with tools
tool_registry = get_registry()
runtime_registry = await RuntimeRegistry.from_conversion_context(
    context=conversion_context,
    llm_registry=llm_registry,
    tool_registry=tool_registry,
)

# Execute graph with tools
executor = GraphExecutor(runtime_registry)
result = await executor.execute(execution_context)
```

### Codegen Integration

Tool metadata can be exported for SDK generation:

```python
from namel3ss.tools import get_registry
from namel3ss.tools.schemas import schema_to_json_schema

registry = get_registry()

# Export all tool schemas
for name, tool_type in registry.list_tools().items():
    adapter = registry.get_adapter(name)
    if adapter:
        metadata = adapter.get_metadata()
        input_schema = schema_to_json_schema(adapter.get_input_schema())
        output_schema = schema_to_json_schema(adapter.get_output_schema())
        
        # Generate SDK client code
        generate_client_code(metadata, input_schema, output_schema)
```

---

## Best Practices

### 1. Schema Design

✅ **DO:**
- Use descriptive field names and descriptions
- Add validation constraints (min, max, pattern, etc.)
- Provide sensible defaults for optional fields
- Use nested models for complex structures
- Document field units and formats

❌ **DON'T:**
- Use generic field names like "data" or "value"
- Skip field descriptions
- Make all fields optional
- Use `Any` type without justification

### 2. Error Handling

✅ **DO:**
- Wrap external exceptions in ToolError subclasses
- Include context (tool name, operation, etc.)
- Use appropriate error codes
- Mark errors as retryable when appropriate
- Log errors with full context

❌ **DON'T:**
- Let exceptions propagate uncaught
- Use generic error messages
- Swallow errors silently
- Retry non-retryable errors

### 3. Configuration

✅ **DO:**
- Provide sensible defaults
- Support environment variable overrides
- Document all configuration options
- Validate configuration at initialization
- Use separate configs for dev/staging/prod

❌ **DON'T:**
- Hard-code values
- Store secrets in configuration files
- Use configuration as state storage
- Ignore invalid configuration

### 4. Observability

✅ **DO:**
- Log at appropriate levels (debug, info, error)
- Create child spans for nested operations
- Include correlation IDs in logs
- Record execution metrics (latency, errors, throughput)
- Use structured logging

❌ **DON'T:**
- Log sensitive data (passwords, API keys, PII)
- Create excessive log noise
- Skip error logging
- Ignore performance metrics

### 5. Testing

✅ **DO:**
- Write unit tests for each adapter
- Test validation (valid and invalid inputs)
- Test error scenarios
- Mock external dependencies
- Test streaming functionality
- Use pytest fixtures for setup

❌ **DON'T:**
- Skip edge case testing
- Test only happy paths
- Use production credentials in tests
- Depend on external services in unit tests

---

## Examples

### Complete Tool Implementation

See `tests/tools/test_adapter.py` for complete examples including:
- Simple tool adapter
- Error handling
- Timeout handling
- Streaming adapter
- Configuration management

### HTTP API Tool

```python
# See "HTTP Tool Example" in Python Tool Authoring Guide section
```

### LLM Tool

```python
# See "Streaming Tool Example" in Python Tool Authoring Guide section
```

### Database Tool

```python
from namel3ss.tools.adapter import BaseToolAdapter, ToolMetadata, ToolCategory
from namel3ss.tools.schemas import ToolInputModel, ToolOutputModel
import asyncpg

class SQLInput(ToolInputModel):
    query: str
    params: dict = {}

class SQLOutput(ToolOutputModel):
    rows: list
    row_count: int

class PostgreSQLTool(BaseToolAdapter[SQLInput, SQLOutput]):
    def __init__(self, connection_string: str):
        super().__init__()
        self.connection_string = connection_string
        self.pool = None
    
    async def _ensure_pool(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.connection_string)
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="postgresql",
            description="Execute PostgreSQL queries",
            version="1.0.0",
            category=ToolCategory.DATABASE,
        )
    
    def get_input_schema(self) -> type[SQLInput]:
        return SQLInput
    
    def get_output_schema(self) -> type[SQLOutput]:
        return SQLOutput
    
    async def invoke(self, input: SQLInput, context: ToolContext) -> SQLOutput:
        await self._ensure_pool()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(input.query, *input.params.values())
            return SQLOutput(
                rows=[dict(row) for row in rows],
                row_count=len(rows)
            )
```

---

## Testing

### Running Tests

```bash
# Run all tool tests
pytest tests/tools/ -v

# Run specific test module
pytest tests/tools/test_adapter.py -v
pytest tests/tools/test_schemas.py -v
pytest tests/tools/test_streaming.py -v

# Run with coverage
pytest tests/tools/ --cov=namel3ss.tools --cov-report=html
```

### Test Statistics

- **Total Tests**: 65
- **Adapter Tests**: 15
- **Schema Tests**: 30
- **Streaming Tests**: 20
- **Coverage**: >95%

### Test Structure

```
tests/tools/
├── __init__.py
├── test_adapter.py        # ToolAdapter protocol, metadata, config, context
├── test_schemas.py        # Pydantic schemas, validation, serialization
└── test_streaming.py      # Streaming utilities, buffers, aggregators
```

---

## Troubleshooting

### Common Issues

#### 1. ValidationError on Tool Invocation

**Symptom**: `ValidationError` when calling `invoke()`

**Causes**:
- Missing required fields
- Type mismatch
- Constraint violation (min, max, pattern)

**Solution**:
```python
# Check schema
input_schema = tool.get_input_schema()
print(input_schema.get_json_schema())

# Validate manually
try:
    validated = input_schema.model_validate(data)
except ValidationError as e:
    print(e.errors())
```

#### 2. Tool Not Found in Registry

**Symptom**: `KeyError: Tool 'xyz' not found in registry`

**Causes**:
- Tool not registered
- Wrong tool name
- Registry cleared

**Solution**:
```python
# List all registered tools
registry = get_registry()
print(registry.list_tools())

# Check if tool exists
if not registry.has("tool_name"):
    tool = MyTool()
    registry.register_adapter(tool)
```

#### 3. Timeout Errors

**Symptom**: `ToolTimeoutError` or `asyncio.TimeoutError`

**Causes**:
- Operation takes longer than configured timeout
- Network issues
- Slow external service

**Solution**:
```python
# Increase timeout
config = ToolConfig(timeout=60.0)  # 60 seconds
tool = MyTool(config=config)

# Or use asyncio.wait_for
try:
    result = await asyncio.wait_for(
        tool.invoke(input, context),
        timeout=60.0
    )
except asyncio.TimeoutError:
    # Handle timeout
    pass
```

#### 4. Streaming Issues

**Symptom**: Stream hangs or produces no output

**Causes**:
- Missing `await` on AsyncIterator
- Stream not consumed
- Backpressure issues

**Solution**:
```python
# Correct async iteration
async for chunk in tool.invoke_stream(input, context):
    process(chunk)

# Check for final chunk
chunks = []
async for chunk in stream:
    chunks.append(chunk)
    if chunk.is_final:
        break
```

#### 5. Import Errors

**Symptom**: `ImportError` or `ModuleNotFoundError`

**Causes**:
- Missing dependencies
- Circular imports
- Package not installed

**Solution**:
```bash
# Install dependencies
pip install namel3ss[tools]

# Check imports
python -c "from namel3ss.tools import ToolAdapter; print('OK')"
```

---

## Performance Considerations

### Tool Instance Reuse

✅ **DO**: Reuse tool instances
```python
# Good: Create once, use many times
tool = MyTool()
for input_data in inputs:
    result = await tool.invoke(input_data, context)
```

❌ **DON'T**: Create tool per invocation
```python
# Bad: Creates overhead
for input_data in inputs:
    tool = MyTool()  # Recreated each time
    result = await tool.invoke(input_data, context)
```

### Connection Pooling

Use connection pools for HTTP/DB tools:

```python
class HTTPTool(BaseToolAdapter):
    def __init__(self):
        super().__init__()
        self._client = httpx.AsyncClient()  # Reuse connection pool
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()
```

### Streaming Backpressure

Implement backpressure for memory efficiency:

```python
from namel3ss.tools.streaming import StreamingContext

context = StreamingContext(
    enable_backpressure=True,
    max_queue_size=1000
)
```

### Caching

Use cache_ttl for frequently-called tools:

```python
config = ToolConfig(cache_ttl=300)  # Cache for 5 minutes
```

---

## Future Enhancements

Potential future additions to the framework:

1. **Decorator-based Tool Definition**
   ```python
   @tool_adapter(name="my_tool", version="1.0.0")
   async def my_tool(input: MyInput, context: ToolContext) -> MyOutput:
       ...
   ```

2. **Tool Composition**
   ```python
   composed_tool = compose_tools(tool1, tool2, aggregator_fn)
   ```

3. **Auto-retry with Exponential Backoff**
   ```python
   config = ToolConfig(
       max_retries=5,
       retry_strategy="exponential",
       backoff_factor=2.0
   )
   ```

4. **Tool Middleware**
   ```python
   tool = MyTool()
   tool = with_caching(tool, ttl=300)
   tool = with_rate_limiting(tool, rpm=60)
   tool = with_circuit_breaker(tool, failure_threshold=5)
   ```

5. **Multi-language Support**
   - JavaScript/TypeScript adapters
   - Go adapters
   - Rust adapters

---

## Appendix

### Error Code Reference

| Code       | Error Type           | Description                      |
|------------|----------------------|----------------------------------|
| TOOL001-020| ToolValidationError  | Validation failures              |
| TOOL021-030| ToolRegistrationError| Registration failures            |
| TOOL031-040| ToolExecutionError   | Runtime execution errors         |
| TOOL034    | ToolTimeoutError     | Timeout errors                   |
| TOOL051-060| ToolConfigurationError| Configuration errors            |
| TOOL052    | ToolAuthenticationError| Authentication failures        |

### Dependencies

Core dependencies:
- `pydantic>=2.0`: Schema validation
- `opentelemetry-api`: Tracing support

Optional dependencies:
- `httpx`: HTTP tools
- `asyncpg`: PostgreSQL tools
- `pymongo`: MongoDB tools

### Version History

- **1.0.0** (2025-11-21): Initial production release
  - Core ToolAdapter protocol
  - Pydantic v2 schemas
  - Streaming support
  - Error hierarchy
  - ToolRegistry enhancements
  - Comprehensive tests (65 passing)

---

**For questions or contributions, please contact the Namel3ss Platform Team.**
