# Tool Adapters - Quick Reference

## Import OpenAPI Tools

```bash
curl -X POST http://localhost:8000/api/tools/import/openapi \
  -H "Content-Type: application/json" \
  -d '{
    "specUrl": "https://api.example.com/openapi.json",
    "baseUrl": "https://api.example.com",
    "authToken": "Bearer sk-...",
    "namePrefix": "example_"
  }'
```

## Create LLM Tool

```bash
curl -X POST http://localhost:8000/api/tools/create/llm \
  -H "Content-Type: application/json" \
  -d '{
    "name": "summarize",
    "description": "Summarize text",
    "llmName": "gpt4",
    "systemPrompt": "You are an expert summarizer.",
    "temperature": 0.7,
    "maxTokens": 500
  }'
```

## Programmatic Usage

### OpenAPI

```python
from n3_server.adapters import OpenAPIAdapter

adapter = OpenAPIAdapter()
tools = await adapter.import_from_url(
    "https://api.example.com/openapi.json",
    base_url="https://api.example.com",
    auth_token="Bearer sk-..."
)

for tool in tools:
    registry.register_tool(tool)
```

### LangChain

```python
from n3_server.adapters import LangChainAdapter
from langchain.tools import DuckDuckGoSearchRun

adapter = LangChainAdapter()
search = DuckDuckGoSearchRun()
tool = adapter.import_tool(search, name_prefix="web_")
registry.register_tool(tool)
```

### LLM Tools

```python
from n3_server.adapters import LLMToolWrapper

wrapper = LLMToolWrapper()
summarizer = wrapper.create_tool(
    name="summarize",
    description="Summarize text",
    llm_name="gpt4",
    system_prompt="You are an expert summarizer."
)

result = await summarizer(input="Long text...")
```

## Supported LLMs

| Provider | Models | Import |
|----------|--------|--------|
| OpenAI | GPT-3.5, GPT-4, GPT-4 Turbo | `from namel3ss.llm.openai_llm import OpenAILLM` |
| Anthropic | Claude 2, Claude 3 | `from namel3ss.llm.anthropic_llm import AnthropicLLM` |
| Vertex AI | PaLM 2, Gemini | `from namel3ss.llm.vertex_llm import VertexLLM` |
| Azure OpenAI | All Azure models | `from namel3ss.llm.azure_openai_llm import AzureOpenAILLM` |
| Ollama | Local models | `from namel3ss.llm.ollama_llm import OllamaLLM` |

## LLM Tool Creators

```python
from n3_server.adapters.llm_tool_wrapper import (
    create_openai_tool,
    create_anthropic_tool,
    create_vertex_tool,
    create_azure_tool,
    create_ollama_tool,
)

# OpenAI
gpt_tool = create_openai_tool("summarize", "Summarize text", model="gpt-4")

# Anthropic
claude_tool = create_anthropic_tool("analyze", "Analyze text", model="claude-3-sonnet-20240229")

# Vertex AI
gemini_tool = create_vertex_tool("describe", "Describe content", model="gemini-pro")

# Azure OpenAI
azure_tool = create_azure_tool("translate", "Translate text", deployment_name="gpt-4")

# Ollama (local)
local_tool = create_ollama_tool("chat", "Chat assistant", model="llama2")
```

## JSON Response Format

```python
extractor = wrapper.create_tool(
    name="extract_entities",
    description="Extract entities",
    llm_name="gpt4",
    response_format="json",
    output_schema={
        "type": "object",
        "properties": {
            "people": {"type": "array"},
            "locations": {"type": "array"}
        }
    }
)

result = await extractor(input="Elon Musk founded SpaceX in California.")
# {"people": ["Elon Musk"], "locations": ["California"]}
```

## OpenAPI Filtering

```python
# Filter by operation ID
tools = await adapter.import_from_url(
    url,
    operation_filter=lambda op: op.operation_id in ["list_users", "get_user"]
)

# Filter by method
tools = await adapter.import_from_url(
    url,
    operation_filter=lambda op: op.method == "GET"
)

# Filter by tags
tools = await adapter.import_from_url(
    url,
    operation_filter=lambda op: "user" in op.tags
)
```

## Testing

```bash
# Run adapter tests
pytest tests/backend/test_tool_adapters.py -v

# Test specific adapter
pytest tests/backend/test_tool_adapters.py -k openapi -v
pytest tests/backend/test_tool_adapters.py -k langchain -v
pytest tests/backend/test_tool_adapters.py -k llm -v

# With coverage
pytest tests/backend/test_tool_adapters.py --cov=n3_server.adapters
```

## Common Patterns

### Batch Import

```python
# Import multiple OpenAPI specs
specs = [url1, url2, url3]
all_tools = []
for spec_url in specs:
    tools = await adapter.import_from_url(spec_url)
    all_tools.extend(tools)

# Register all
for tool in all_tools:
    registry.register_tool(tool)
```

### LLM Tool Factory

```python
def create_analysis_tools(llm_name: str):
    wrapper = LLMToolWrapper()
    return {
        "summarize": wrapper.create_tool(
            "summarize", "Summarize text", llm_name,
            system_prompt="Summarize concisely."
        ),
        "sentiment": wrapper.create_tool(
            "sentiment", "Analyze sentiment", llm_name,
            system_prompt="Analyze sentiment: positive/negative/neutral."
        ),
        "extract": wrapper.create_tool(
            "extract", "Extract key points", llm_name,
            system_prompt="Extract key points as bullet list."
        ),
    }

# Create tools for multiple LLMs
gpt_tools = create_analysis_tools("gpt4")
claude_tools = create_analysis_tools("claude3")
```

### Error Handling

```python
try:
    tools = await adapter.import_from_url(url, auth_token=token)
except httpx.HTTPError as e:
    print(f"HTTP error: {e}")
except ValueError as e:
    print(f"Invalid spec: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Tool Metadata

```python
# Access tool metadata
tool = tools[0]
metadata = tool._tool_metadata

print(metadata["name"])          # Tool name
print(metadata["description"])   # Description
print(metadata["input_schema"])  # Input schema
print(metadata["output_schema"]) # Output schema
print(metadata["tags"])          # Tags
print(metadata["source"])        # "openapi", "langchain", "llm"
```

## Related Docs

- [Full Implementation Guide](./TOOL_ADAPTERS_IMPLEMENTATION.md)
- [Agent Graph Editor Guide](./AGENT_GRAPH_EDITOR_GUIDE.md)
- [Execution Engine Reference](./EXECUTION_ENGINE_QUICK_REF.md)
