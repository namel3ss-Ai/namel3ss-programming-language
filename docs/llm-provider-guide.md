# LLM Provider Subsystem Guide

## Overview

The Namel3ss LLM provider subsystem provides a clean, extensible interface for integrating multiple LLM providers into your applications. It supports OpenAI, Anthropic, Google Vertex AI, Azure OpenAI, and Ollama, with a provider-agnostic API for seamless provider switching.

## Architecture

The LLM subsystem is organized into several key components:

### Core Components

1. **Base Abstractions** (`namel3ss/llm/base.py`)
   - `BaseLLM`: Abstract base class for all LLM providers
   - `LLMResponse`: Standardized response format
   - `ChatMessage`: Chat message representation
   - `LLMError`: LLM-specific exceptions

2. **Registry** (`namel3ss/llm/registry.py`)
   - `LLMRegistry`: Manages LLM instances by name
   - `get_registry()`: Access global registry singleton

3. **Factory** (`namel3ss/llm/factory.py`)
   - `create_llm()`: Create LLM instances with configuration
   - `register_provider()`: Register new provider classes
   - Automatic API key resolution from environment variables

4. **Provider Implementations**
   - `OpenAILLM` - OpenAI GPT models
   - `AnthropicLLM` - Anthropic Claude models
   - `VertexLLM` - Google Vertex AI / PaLM / Gemini
   - `AzureOpenAILLM` - Azure OpenAI Service
   - `OllamaLLM` - Local Ollama deployment

## Quick Start

### Basic Usage

```python
from namel3ss.llm import create_llm

# Create an OpenAI LLM (requires OPENAI_API_KEY env var)
llm = create_llm('my_gpt4', 'openai', 'gpt-4', {
    'temperature': 0.7,
    'max_tokens': 1024
})

# Generate text
response = llm.generate('What is the capital of France?')
print(response.text)  # "Paris"

# Access token usage
print(f"Used {response.total_tokens} tokens")
```

### Chat Completions

```python
from namel3ss.llm import create_llm, ChatMessage

llm = create_llm('my_claude', 'anthropic', 'claude-3-opus-20240229')

messages = [
    ChatMessage(role='user', content='Hello!'),
    ChatMessage(role='assistant', content='Hi! How can I help you?'),
    ChatMessage(role='user', content='What is 2+2?'),
]

response = llm.generate_chat(messages)
print(response.text)  # "4"
```

### Streaming

```python
llm = create_llm('my_gpt4', 'openai', 'gpt-4')

# Check if streaming is supported
if llm.supports_streaming():
    for chunk in llm.stream('Tell me a story'):
        print(chunk, end='', flush=True)
```

### Using the Registry

```python
from namel3ss.llm import create_llm, get_registry

# Create and automatically register
llm = create_llm('my_llm', 'openai', 'gpt-4', register=True)

# Retrieve from registry later
registry = get_registry()
llm = registry.get_required('my_llm')
response = llm.generate('Hello')
```

## Provider Configuration

### OpenAI

```python
llm = create_llm('my_openai', 'openai', 'gpt-4', {
    'api_key': 'sk-...',  # or set OPENAI_API_KEY env var
    'api_base': 'https://api.openai.com/v1',  # optional
    'organization': 'org-...',  # optional
    'temperature': 0.7,
    'max_tokens': 1024,
})
```

**Environment Variables:**
- `OPENAI_API_KEY`: OpenAI API key (required)
- `OPENAI_API_BASE`: Custom API base URL (optional)

**Supported Models:** `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`, etc.

### Anthropic

```python
llm = create_llm('my_anthropic', 'anthropic', 'claude-3-opus-20240229', {
    'api_key': 'sk-ant-...',  # or set ANTHROPIC_API_KEY env var
    'api_base': 'https://api.anthropic.com',  # optional
    'temperature': 0.7,
    'max_tokens': 1024,
})
```

**Environment Variables:**
- `ANTHROPIC_API_KEY`: Anthropic API key (required)

**Supported Models:** `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`, `claude-2.1`, etc.

### Google Vertex AI

```python
llm = create_llm('my_vertex', 'vertex', 'gemini-pro', {
    'project_id': 'my-project',  # or set GOOGLE_CLOUD_PROJECT env var
    'location': 'us-central1',  # optional, default: us-central1
    'credentials_path': '/path/to/credentials.json',  # optional
    'temperature': 0.7,
    'max_tokens': 1024,
})
```

**Environment Variables:**
- `GOOGLE_CLOUD_PROJECT`: GCP project ID (required)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account JSON (optional)

**Supported Models:** `gemini-pro`, `gemini-pro-vision`, `text-bison`, `chat-bison`, etc.

**Dependencies:** `pip install google-cloud-aiplatform`

### Azure OpenAI

```python
llm = create_llm('my_azure', 'azure_openai', 'gpt-4', {
    'api_key': 'xxx',  # or set AZURE_OPENAI_API_KEY env var
    'endpoint': 'https://xxx.openai.azure.com',  # or set AZURE_OPENAI_ENDPOINT env var
    'deployment_name': 'my-gpt4-deployment',  # required
    'api_version': '2023-05-15',  # optional
    'temperature': 0.7,
    'max_tokens': 1024,
})
```

**Environment Variables:**
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key (required)
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL (required)

**Note:** `deployment_name` must match your Azure deployment name.

### Ollama (Local)

```python
llm = create_llm('my_ollama', 'ollama', 'llama2', {
    'base_url': 'http://localhost:11434',  # or set OLLAMA_BASE_URL env var
    'temperature': 0.7,
    'max_tokens': 1024,
})
```

**Environment Variables:**
- `OLLAMA_BASE_URL`: Ollama API base URL (optional, default: `http://localhost:11434`)

**Supported Models:** Any model installed in your local Ollama instance (e.g., `llama2`, `mistral`, `codellama`)

**Setup:** Install Ollama from https://ollama.ai and run `ollama serve`

## Advanced Features

### Batch Generation

```python
prompts = [
    'What is 2+2?',
    'What is 3+3?',
    'What is 4+4?',
]

responses = llm.generate_batch(prompts)
for response in responses:
    print(response.text)
```

### Custom Configuration

```python
llm = create_llm('my_llm', 'openai', 'gpt-4', {
    'temperature': 0.9,
    'max_tokens': 2048,
    'top_p': 0.95,
    'frequency_penalty': 0.5,
    'presence_penalty': 0.5,
    'stop': ['\n\n', '###'],
})
```

### Error Handling

```python
from namel3ss.llm import LLMError

try:
    response = llm.generate('Hello')
except LLMError as e:
    print(f"Provider: {e.provider}")
    print(f"Model: {e.model}")
    print(f"Status: {e.status_code}")
    print(f"Error: {e}")
    if e.original_error:
        print(f"Original: {e.original_error}")
```

### Response Inspection

```python
response = llm.generate('Hello')

# Access response data
print(response.text)              # Generated text
print(response.model)             # Model used
print(response.finish_reason)     # Why generation stopped
print(response.prompt_tokens)     # Tokens in prompt
print(response.completion_tokens) # Tokens in completion
print(response.total_tokens)      # Total tokens used
print(response.metadata)          # Provider-specific metadata
print(response.raw)               # Raw API response
```

## Extending the System

### Creating a Custom Provider

```python
from namel3ss.llm import BaseLLM, LLMResponse, ChatMessage, register_provider

class MyCustomLLM(BaseLLM):
    def generate(self, prompt, **kwargs):
        # Implement your generation logic
        text = self._call_my_api(prompt)
        return LLMResponse(
            text=text,
            raw={},
            model=self.model
        )
    
    def generate_chat(self, messages, **kwargs):
        # Implement chat logic
        prompt = '\n'.join(f"{m.role}: {m.content}" for m in messages)
        return self.generate(prompt, **kwargs)
    
    def supports_streaming(self):
        return False  # or True if you support streaming
    
    def get_provider_name(self):
        return 'my_custom'

# Register your provider
register_provider('my_custom', MyCustomLLM)

# Use it
llm = create_llm('test', 'my_custom', 'my-model')
```

## DSL Integration

The LLM provider subsystem integrates with Namel3ss DSL through LLM blocks:

```namel3ss
llm my_gpt4 {
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 1024
}

llm my_claude {
  provider: "anthropic"
  model: "claude-3-opus-20240229"
  temperature: 0.5
}
```

These blocks are parsed during compilation and registered in the runtime, making them available for chain execution.

## Best Practices

1. **Use Environment Variables**: Store API keys in environment variables rather than hardcoding them
2. **Set Appropriate Timeouts**: Different providers have different latencies; adjust timeouts accordingly
3. **Handle Errors Gracefully**: Wrap LLM calls in try-except blocks to handle API failures
4. **Monitor Token Usage**: Track `response.total_tokens` to manage costs
5. **Use Streaming for Long Responses**: Enable streaming for better user experience with long generations
6. **Cache Responses**: Consider caching LLM responses to reduce API calls and costs
7. **Test with Local Models**: Use Ollama for development to avoid API costs

## Troubleshooting

### API Key Not Found

```
LLMError: OpenAI API key not found for LLM 'my_llm'.
Set OPENAI_API_KEY environment variable or provide 'api_key' in config.
```

**Solution:** Set the appropriate environment variable or provide `api_key` in config

### Provider Not Found

```
LLMError: Unknown LLM provider 'my_provider'.
Available providers: openai, anthropic, vertex, azure_openai, ollama
```

**Solution:** Use one of the supported provider names or register a custom provider

### Import Errors

```
LLMError: httpx is required for OpenAI provider.
Install it with: pip install httpx
```

**Solution:** Install the required dependency: `pip install httpx`

For Vertex AI: `pip install google-cloud-aiplatform`

### Connection Errors

```
LLMError: Ollama API request failed: Connection refused
```

**Solution:** Ensure Ollama is running: `ollama serve`

## Dependencies

Core dependencies:
- `httpx` - HTTP client (required for OpenAI, Anthropic, Azure, Ollama)

Optional dependencies:
- `google-cloud-aiplatform` - Required for Vertex AI provider

Install all:
```bash
pip install httpx google-cloud-aiplatform
```

## Performance Considerations

- **Timeouts**: Default timeout is 60 seconds; increase for slower models
- **Streaming**: Reduces time-to-first-token but doesn't reduce total generation time
- **Batch Requests**: Default implementation is sequential; override for parallel processing
- **Connection Pooling**: HTTP clients are cached per LLM instance for connection reuse

## Testing

Run tests:
```bash
pytest tests/llm/ -v
```

Current test coverage: 47 tests covering:
- Base abstractions (13 tests)
- Registry operations (16 tests)
- Factory and provider management (18 tests)

## Roadmap

Future enhancements:
- [ ] Async support (`agenerate`, `agenerate_chat`)
- [ ] Function calling / tool use support
- [ ] Vision model support (image inputs)
- [ ] Embedding model support
- [ ] Retry logic with exponential backoff
- [ ] Request/response logging and observability
- [ ] Cost tracking and rate limiting
- [ ] Model fallback and load balancing
- [ ] Additional providers (Cohere, AI21, Hugging Face)

## Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com)
- [Google Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Ollama Documentation](https://ollama.ai/docs)
