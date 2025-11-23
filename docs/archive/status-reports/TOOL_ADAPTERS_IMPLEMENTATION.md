# Tool Adapters Implementation

## Overview

The Tool Adapters system enables importing tools from external sources into the N3 ToolRegistry. Supports **OpenAPI specifications**, **LangChain tools**, and **LLM-powered tools** with all N3 LLM providers (OpenAI, Anthropic, Vertex AI, Azure OpenAI, Ollama).

## Components

### 1. OpenAPI Adapter (`n3_server/adapters/openapi_adapter.py`)

Parses OpenAPI 3.0/3.1 specifications and generates executable tool wrappers.

**Features:**
- ✅ Parses JSON and YAML specs
- ✅ Handles path, query, header parameters
- ✅ Supports request bodies
- ✅ Authentication (Bearer tokens, API keys)
- ✅ Operation filtering
- ✅ Automatic schema conversion

**Example:**
```python
from n3_server.adapters import OpenAPIAdapter

adapter = OpenAPIAdapter()

# Import from URL
tools = await adapter.import_from_url(
    spec_url="https://api.example.com/openapi.json",
    base_url="https://api.example.com",
    auth_token="sk-...",
    name_prefix="example_",
)

# Register tools
for tool in tools:
    registry.register_tool(tool)
```

### 2. LangChain Adapter (`n3_server/adapters/langchain_adapter.py`)

Converts LangChain BaseTool instances to registry-compatible functions.

**Features:**
- ✅ Sync and async tool support
- ✅ Schema extraction from Pydantic models
- ✅ Name prefix/suffix support
- ✅ Batch import

**Example:**
```python
from langchain.tools import DuckDuckGoSearchRun
from n3_server.adapters import LangChainAdapter

adapter = LangChainAdapter()

# Import single tool
search_tool = DuckDuckGoSearchRun()
tool_func = adapter.import_tool(search_tool, name_prefix="web_")
registry.register_tool(tool_func)

# Import multiple tools
tools = [Tool1(), Tool2(), Tool3()]
imported = adapter.import_tools(tools)
for tool in imported:
    registry.register_tool(tool)
```

### 3. LLM Tool Wrapper (`n3_server/adapters/llm_tool_wrapper.py`)

Creates LLM-powered tools using any registered N3 LLM provider.

**Supported LLMs:**
- **OpenAI**: GPT-3.5, GPT-4, GPT-4 Turbo
- **Anthropic**: Claude 2, Claude 3 (Opus, Sonnet, Haiku)
- **Vertex AI**: PaLM 2, Gemini Pro
- **Azure OpenAI**: All Azure-deployed models
- **Ollama**: Local models (Llama 2, Mistral, etc.)

**Features:**
- ✅ Text and JSON response formats
- ✅ Structured output with schemas
- ✅ Custom system prompts
- ✅ Temperature and token control

**Example:**
```python
from n3_server.adapters import LLMToolWrapper, create_llm_tool

# Method 1: Using wrapper
wrapper = LLMToolWrapper()
summarizer = wrapper.create_tool(
    name="summarize",
    description="Summarize long text",
    llm_name="gpt4",
    system_prompt="You are an expert summarizer.",
    temperature=0.5,
    max_tokens=500,
)

# Method 2: Convenience function
analyzer = create_llm_tool(
    name="sentiment_analyzer",
    description="Analyze sentiment of text",
    llm_name="claude3",
    system_prompt="Analyze sentiment: positive, negative, or neutral.",
    response_format="json",
    output_schema={
        "type": "object",
        "properties": {
            "sentiment": {"type": "string"},
            "confidence": {"type": "number"},
        },
    },
)
```

## API Integration

### Import OpenAPI Tools

```http
POST /api/tools/import/openapi
Content-Type: application/json

{
  "specUrl": "https://api.example.com/openapi.json",
  "baseUrl": "https://api.example.com",
  "authToken": "Bearer sk-...",
  "namePrefix": "example_",
  "operationIds": ["list_users", "get_user"]
}
```

**Response:**
```json
{
  "success": true,
  "toolsImported": 2,
  "toolNames": ["example_list_users", "example_get_user"]
}
```

### Create LLM-Powered Tool

```http
POST /api/tools/create/llm
Content-Type: application/json

{
  "name": "summarize",
  "description": "Summarize long text",
  "llmName": "gpt4",
  "systemPrompt": "You are an expert summarizer.",
  "temperature": 0.5,
  "maxTokens": 500,
  "responseFormat": "text"
}
```

**Response:**
```json
{
  "success": true,
  "toolName": "summarize"
}
```

## LLM Provider Examples

### OpenAI Tools

```python
from n3_server.adapters.llm_tool_wrapper import create_openai_tool

# Create OpenAI-powered tool
translator = create_openai_tool(
    name="translate",
    description="Translate text to another language",
    model="gpt-4",
    system_prompt="You are a professional translator.",
    temperature=0.3,
)

result = await translator(
    input="Bonjour le monde",
    target_language="English"
)
# Output: "Hello world"
```

### Anthropic Tools

```python
from n3_server.adapters.llm_tool_wrapper import create_anthropic_tool

# Create Claude-powered tool
reviewer = create_anthropic_tool(
    name="code_reviewer",
    description="Review code for issues",
    model="claude-3-sonnet-20240229",
    system_prompt="You are an expert code reviewer.",
)

result = await reviewer(input="def foo():\n  return x + y")
# Output: Analysis of code issues
```

### Vertex AI Tools

```python
from n3_server.adapters.llm_tool_wrapper import create_vertex_tool

# Create Gemini-powered tool
analyzer = create_vertex_tool(
    name="image_analyzer",
    description="Analyze images",
    model="gemini-pro-vision",
    system_prompt="Describe what you see in the image.",
)
```

### Azure OpenAI Tools

```python
from n3_server.adapters.llm_tool_wrapper import create_azure_tool

# Create Azure OpenAI tool
assistant = create_azure_tool(
    name="customer_support",
    description="Answer customer questions",
    deployment_name="gpt-4-deployment",
    system_prompt="You are a helpful customer support agent.",
)
```

### Ollama Tools (Local)

```python
from n3_server.adapters.llm_tool_wrapper import create_ollama_tool

# Create Ollama tool (local model)
local_summarizer = create_ollama_tool(
    name="local_summarize",
    description="Summarize text using local model",
    model="llama2",
    system_prompt="Summarize concisely.",
)
```

## OpenAPI Spec Examples

### Simple GET Endpoint

```yaml
openapi: 3.0.0
paths:
  /weather:
    get:
      operationId: get_weather
      summary: Get weather data
      parameters:
        - name: city
          in: query
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Success
          content:
            application/json:
              schema:
                type: object
                properties:
                  temperature: {type: number}
                  conditions: {type: string}
```

**Generated Tool:**
```python
async def get_weather(city: str) -> dict:
    """Get weather data"""
    # Automatically generated HTTP call
    response = await client.get(
        "https://api.example.com/weather",
        params={"city": city}
    )
    return response.json()
```

### POST with Request Body

```yaml
paths:
  /users:
    post:
      operationId: create_user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                name: {type: string}
                email: {type: string}
      responses:
        '201':
          description: User created
```

**Generated Tool:**
```python
async def create_user(body: dict) -> dict:
    """Create a new user"""
    response = await client.post(
        "https://api.example.com/users",
        json=body
    )
    return response.json()
```

### Path Parameters

```yaml
paths:
  /users/{id}:
    get:
      operationId: get_user
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
```

**Generated Tool:**
```python
async def get_user(id: str) -> dict:
    """Get user by ID"""
    response = await client.get(
        f"https://api.example.com/users/{id}"
    )
    return response.json()
```

## LangChain Integration Examples

### Search Tool

```python
from langchain.tools import DuckDuckGoSearchRun
from n3_server.adapters import LangChainAdapter

adapter = LangChainAdapter()

# Import search tool
search = DuckDuckGoSearchRun()
search_func = adapter.import_tool(search)

# Use in agent
result = await search_func(input="What is N3 programming language?")
```

### Custom LangChain Tool

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    expression: str = Field(description="Math expression to evaluate")

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate math expressions"
    args_schema = CalculatorInput
    
    def _run(self, expression: str) -> str:
        return str(eval(expression))

# Import custom tool
calc = CalculatorTool()
calc_func = adapter.import_tool(calc)
result = await calc_func(expression="2 + 2")  # "4"
```

## Structured Output with LLMs

### JSON Response Format

```python
wrapper = LLMToolWrapper()

extractor = wrapper.create_tool(
    name="extract_entities",
    description="Extract named entities from text",
    llm_name="gpt4",
    response_format="json",
    output_schema={
        "type": "object",
        "properties": {
            "people": {
                "type": "array",
                "items": {"type": "string"}
            },
            "organizations": {
                "type": "array",
                "items": {"type": "string"}
            },
            "locations": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    },
    system_prompt="Extract people, organizations, and locations from text."
)

result = await extractor(
    input="Elon Musk founded SpaceX in California."
)
# {
#   "people": ["Elon Musk"],
#   "organizations": ["SpaceX"],
#   "locations": ["California"]
# }
```

## Testing

### Run Tests

```bash
# Run all adapter tests
pytest tests/backend/test_tool_adapters.py -v

# Run specific test
pytest tests/backend/test_tool_adapters.py::test_openapi_adapter_import_from_dict -v

# With coverage
pytest tests/backend/test_tool_adapters.py --cov=n3_server.adapters --cov-report=html
```

### Test Coverage

The test suite covers:
- ✅ OpenAPI spec parsing (JSON/YAML)
- ✅ Path, query, header parameters
- ✅ Request bodies
- ✅ Operation filtering
- ✅ LangChain tool import (sync/async)
- ✅ Name prefix/stripping
- ✅ LLM tool creation (all providers)
- ✅ JSON response formatting
- ✅ Schema extraction

## Advanced Usage

### Filter OpenAPI Operations

```python
# Only import GET operations
tools = await adapter.import_from_url(
    spec_url=url,
    operation_filter=lambda op: op.method == "GET"
)

# Only import operations with specific tags
tools = await adapter.import_from_url(
    spec_url=url,
    operation_filter=lambda op: "user" in op.tags
)

# Combine filters
def my_filter(op):
    return op.method in ["GET", "POST"] and "admin" not in op.path

tools = await adapter.import_from_url(spec_url=url, operation_filter=my_filter)
```

### Batch LLM Tool Creation

```python
wrapper = LLMToolWrapper()

tools_config = [
    {"name": "summarize", "llm_name": "gpt4", "description": "Summarize text"},
    {"name": "translate", "llm_name": "claude3", "description": "Translate text"},
    {"name": "analyze", "llm_name": "gemini", "description": "Analyze content"},
]

tools = []
for config in tools_config:
    tool = wrapper.create_tool(**config)
    tools.append(tool)
    registry.register_tool(tool)
```

### Custom Authentication

```python
# Bearer token
tools = await adapter.import_from_url(
    spec_url=url,
    auth_token="Bearer sk-..."
)

# API Key (add to headers manually)
adapter = OpenAPIAdapter()
client = await adapter._get_client()
client.headers["X-API-Key"] = "your-api-key"
tools = await adapter.import_from_url(spec_url=url)
```

## Performance Considerations

### HTTP Client Reuse

OpenAPIAdapter reuses HTTP client across requests:
```python
adapter = OpenAPIAdapter()

# Import multiple specs (same client)
tools1 = await adapter.import_from_url(url1)
tools2 = await adapter.import_from_url(url2)

# Close when done
await adapter.close()
```

### LLM Caching

LLM responses can be cached for repeated inputs:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_summarize(text: str) -> str:
    # Cache summary results
    return summarize_tool(input=text)
```

## Error Handling

### OpenAPI Errors

```python
try:
    tools = await adapter.import_from_url(spec_url)
except httpx.HTTPError as e:
    print(f"Failed to fetch spec: {e}")
except ValueError as e:
    print(f"Invalid spec: {e}")
```

### LLM Errors

```python
from namel3ss.llm import LLMError

try:
    result = await llm_tool(input="...")
except LLMError as e:
    print(f"LLM call failed: {e}")
    print(f"Provider: {e.provider}, Model: {e.model}")
```

## Related Documentation

- [Agent Graph Editor Guide](./AGENT_GRAPH_EDITOR_GUIDE.md)
- [Execution Engine Implementation](./EXECUTION_ENGINE_IMPLEMENTATION.md)
- [N3 LLM Provider System](./namel3ss/llm/README.md)
