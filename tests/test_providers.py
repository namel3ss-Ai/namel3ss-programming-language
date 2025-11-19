"""Tests for N3Provider system."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from namel3ss.providers import (
    N3Provider,
    ProviderMessage,
    ProviderResponse,
    ProviderError,
    create_provider_from_spec,
    ProviderRegistry,
    load_config_for_provider,
    ProviderConfigError,
)


class MockProvider(N3Provider):
    """Mock provider for testing."""
    
    def __init__(self, name: str, model: str, config=None):
        super().__init__(name, model, config)
        self.generate_calls = []
        self.stream_calls = []
    
    async def generate(self, messages, **kwargs):
        self.generate_calls.append((messages, kwargs))
        return ProviderResponse(
            model=self.model,
            output_text="Mock response",
            raw={"mock": True},
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            finish_reason="stop",
        )
    
    def supports_streaming(self):
        return True
    
    async def stream(self, messages, **kwargs):
        self.stream_calls.append((messages, kwargs))
        for chunk in ["Hello", " ", "World", "!"]:
            yield chunk


@pytest.mark.asyncio
async def test_provider_message_creation():
    """Test creating ProviderMessage."""
    msg = ProviderMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.to_dict() == {"role": "user", "content": "Hello"}


@pytest.mark.asyncio
async def test_provider_response_from_llm():
    """Test creating ProviderResponse."""
    response = ProviderResponse(
        model="test-model",
        output_text="Test output",
        raw={"test": True},
        usage={"prompt_tokens": 5, "completion_tokens": 10},
    )
    
    assert response.model == "test-model"
    assert response.output_text == "Test output"
    assert response.prompt_tokens == 5
    assert response.completion_tokens == 10
    assert response.total_tokens == 15


@pytest.mark.asyncio
async def test_mock_provider_generate():
    """Test mock provider generation."""
    provider = MockProvider("test", "test-model")
    messages = [ProviderMessage(role="user", content="Hello")]
    
    response = await provider.generate(messages, temperature=0.5)
    
    assert response.output_text == "Mock response"
    assert response.model == "test-model"
    assert len(provider.generate_calls) == 1
    assert provider.generate_calls[0][1]["temperature"] == 0.5


@pytest.mark.asyncio
async def test_mock_provider_stream():
    """Test mock provider streaming."""
    provider = MockProvider("test", "test-model")
    messages = [ProviderMessage(role="user", content="Hello")]
    
    chunks = []
    async for chunk in provider.stream(messages):
        chunks.append(chunk)
    
    assert chunks == ["Hello", " ", "World", "!"]
    assert len(provider.stream_calls) == 1


@pytest.mark.asyncio
async def test_provider_batch_default():
    """Test default batch generation."""
    provider = MockProvider("test", "test-model")
    
    batch = [
        [ProviderMessage(role="user", content="Hello")],
        [ProviderMessage(role="user", content="Hi")],
    ]
    
    responses = await provider.generate_batch(batch)
    
    assert len(responses) == 2
    assert all(r.output_text == "Mock response" for r in responses)


def test_provider_registry_create_and_register():
    """Test provider registry operations."""
    from namel3ss.providers.factory import register_provider_class
    
    # Register mock provider BEFORE creating registry
    register_provider_class("mock", MockProvider)
    
    # Also register in config loaders
    from namel3ss.providers.config import PROVIDER_CONFIG_LOADERS
    PROVIDER_CONFIG_LOADERS["mock"] = lambda cfg: cfg or {}
    
    registry = ProviderRegistry()
    
    # Create and register provider
    provider = registry.create_and_register(
        name="test_provider",
        provider_type="mock",
        model="test-model",
        config={"temperature": 0.7},
    )
    
    assert provider.name == "test_provider"
    assert provider.model == "test-model"
    
    # Get provider
    retrieved = registry.get("test_provider")
    assert retrieved is provider
    
    # Get or create (should return existing)
    same_provider = registry.get_or_create(
        name="test_provider",
        provider_type="mock",
        model="test-model",
    )
    assert same_provider is provider


def test_provider_registry_context_manager():
    """Test provider registry as context manager."""
    from namel3ss.providers.factory import register_provider_class
    
    register_provider_class("mock", MockProvider)
    
    with ProviderRegistry() as registry:
        provider = registry.create_and_register(
            name="test",
            provider_type="mock",
            model="test-model",
        )
        assert registry.get("test") is provider
    
    # After context exit, registry should be cleared
    # (but provider object still exists)
    assert provider.name == "test"


def test_config_loading_unknown_provider():
    """Test loading config for unknown provider type."""
    with pytest.raises(ProviderConfigError, match="Unknown provider type"):
        load_config_for_provider("nonexistent_provider")


def test_config_merge():
    """Test configuration merging."""
    from namel3ss.providers.config import merge_configs
    
    base = {"temperature": 0.7, "max_tokens": 100}
    override = {"temperature": 0.5, "top_p": 0.9}
    
    merged = merge_configs(base, override)
    
    assert merged["temperature"] == 0.5  # Overridden
    assert merged["max_tokens"] == 100  # From base
    assert merged["top_p"] == 0.9  # From override


@pytest.mark.asyncio
async def test_provider_error_handling():
    """Test provider error handling."""
    
    class FailingProvider(N3Provider):
        async def generate(self, messages, **kwargs):
            raise ProviderError("Test error", provider="test", model="test-model")
    
    provider = FailingProvider("test", "test-model")
    messages = [ProviderMessage(role="user", content="Hello")]
    
    with pytest.raises(ProviderError, match="Test error"):
        await provider.generate(messages)


@pytest.mark.asyncio
async def test_streaming_not_supported():
    """Test provider that doesn't support streaming."""
    
    class NonStreamingProvider(N3Provider):
        async def generate(self, messages, **kwargs):
            return ProviderResponse(
                model=self.model,
                output_text="Response",
                raw={},
            )
        
        def supports_streaming(self):
            return False
    
    provider = NonStreamingProvider("test", "test-model")
    messages = [ProviderMessage(role="user", content="Hello")]
    
    # Should raise NotImplementedError when trying to stream
    with pytest.raises(NotImplementedError, match="Streaming not supported"):
        # Need to await the generator creation and iterate
        async for _ in provider.stream(messages):
            pass


def test_openai_config_loading():
    """Test OpenAI configuration loading."""
    from namel3ss.providers.config import load_openai_config
    
    config = load_openai_config({"api_key": "test-key"})
    
    assert config["api_key"] == "test-key"
    assert config["base_url"] == "https://api.openai.com/v1"
    assert config["timeout"] == 60


def test_anthropic_config_loading():
    """Test Anthropic configuration loading."""
    from namel3ss.providers.config import load_anthropic_config
    
    config = load_anthropic_config({"api_key": "test-key"})
    
    assert config["api_key"] == "test-key"
    assert config["base_url"] == "https://api.anthropic.com"
    assert config["version"] == "2023-06-01"


@pytest.mark.asyncio
async def test_openai_provider_initialization():
    """Test OpenAI provider initialization."""
    from namel3ss.providers.openai_provider import OpenAIProvider
    
    # Should fail without API key
    with pytest.raises(ProviderError, match="API key not found"):
        OpenAIProvider("test", "gpt-4", {})
    
    # Should succeed with API key
    provider = OpenAIProvider("test", "gpt-4", {"api_key": "test-key"})
    assert provider.name == "test"
    assert provider.model == "gpt-4"
    assert provider.api_key == "test-key"


@pytest.mark.asyncio
async def test_anthropic_provider_initialization():
    """Test Anthropic provider initialization."""
    from namel3ss.providers.anthropic_provider import AnthropicProvider
    
    # Should fail without API key
    with pytest.raises(ProviderError, match="API key not found"):
        AnthropicProvider("test", "claude-3-opus", {})
    
    # Should succeed with API key
    provider = AnthropicProvider("test", "claude-3-opus", {"api_key": "test-key"})
    assert provider.name == "test"
    assert provider.model == "claude-3-opus"
    assert provider.api_key == "test-key"


@pytest.mark.asyncio
async def test_azure_openai_provider_initialization():
    """Test Azure OpenAI provider initialization."""
    from namel3ss.providers.azure_openai_provider import AzureOpenAIProvider
    
    # Should fail without API key
    with pytest.raises(ProviderError, match="API key not found"):
        AzureOpenAIProvider("test", "gpt-4", {})
    
    # Should fail without endpoint
    with pytest.raises(ProviderError, match="endpoint not found"):
        AzureOpenAIProvider("test", "gpt-4", {"api_key": "test-key"})
    
    # Should succeed with both
    provider = AzureOpenAIProvider(
        "test",
        "gpt-4",
        {"api_key": "test-key", "endpoint": "https://test.openai.azure.com"}
    )
    assert provider.name == "test"
    assert provider.api_key == "test-key"
    assert provider.endpoint == "https://test.openai.azure.com"


@pytest.mark.asyncio
async def test_local_provider_initialization():
    """Test Local provider initialization."""
    from namel3ss.providers.local_provider import LocalProvider
    
    # Should fail without engine URL
    with pytest.raises(ProviderError, match="engine URL not found"):
        LocalProvider("test", "llama2", {})
    
    # Should succeed with engine URL
    provider = LocalProvider(
        "test",
        "llama2",
        {"engine_url": "http://localhost:11434"}
    )
    assert provider.name == "test"
    assert provider.engine_url == "http://localhost:11434"
    assert provider.engine_type == "ollama"  # Auto-detected


@pytest.mark.asyncio
async def test_http_provider_initialization():
    """Test HTTP provider initialization."""
    from namel3ss.providers.http_provider import HttpProvider
    
    # Should fail without base URL
    with pytest.raises(ProviderError, match="base URL not found"):
        HttpProvider("test", "custom-model", {})
    
    # Should succeed with base URL
    provider = HttpProvider(
        "test",
        "custom-model",
        {"base_url": "https://custom-llm.example.com"}
    )
    assert provider.name == "test"
    assert provider.base_url == "https://custom-llm.example.com"


@pytest.mark.asyncio
@patch('httpx.AsyncClient')
async def test_openai_provider_generate_mock(mock_client_class):
    """Test OpenAI provider generate with mocked HTTP."""
    from namel3ss.providers.openai_provider import OpenAIProvider
    
    # Mock HTTP response
    mock_response = AsyncMock()
    mock_response.json.return_value = {
        "choices": [{
            "message": {"content": "Hello from OpenAI!"},
            "finish_reason": "stop"
        }],
        "model": "gpt-4",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    mock_response.raise_for_status = Mock()
    
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client_class.return_value = mock_client
    
    # Create provider and generate
    provider = OpenAIProvider("test", "gpt-4", {"api_key": "test-key"})
    messages = [ProviderMessage(role="user", content="Hello")]
    
    response = await provider.generate(messages)
    
    assert response.output_text == "Hello from OpenAI!"
    assert response.model == "gpt-4"
    assert response.total_tokens == 15
    
    # Verify HTTP call was made
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert "https://api.openai.com/v1/chat/completions" in str(call_args)


@pytest.mark.asyncio
async def test_provider_context_manager():
    """Test provider used as context manager."""
    # Note: Base N3Provider doesn't implement async context manager
    # This tests that providers can be used without context manager
    provider = MockProvider("ctx", "test-model")
    
    messages = [ProviderMessage(role="user", content="Test")]
    response = await provider.generate(messages)
    assert response.output_text == "Mock response"
    
    # Context manager support is optional and provider-specific
    # For now, just verify basic usage works


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
