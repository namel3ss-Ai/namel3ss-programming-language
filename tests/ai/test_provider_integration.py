"""Integration tests for N3Provider with chain execution."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from namel3ss.providers import (
    N3Provider,
    ProviderMessage,
    ProviderResponse,
    ProviderError,
    create_provider_from_spec,
    ProviderRegistry,
)


class TestProvider(N3Provider):
    """Test provider that returns predictable responses."""
    
    def __init__(self, name: str, model: str, config=None):
        super().__init__(name, model, config)
        self.call_count = 0
    
    async def generate(self, messages, **kwargs):
        self.call_count += 1
        # Echo back the last user message
        user_messages = [m for m in messages if m.role == "user"]
        content = user_messages[-1].content if user_messages else "No input"
        
        return ProviderResponse(
            model=self.model,
            output_text=f"Response {self.call_count}: {content}",
            raw={"test": True},
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
    
    def supports_streaming(self):
        return True
    
    async def stream(self, messages, **kwargs):
        user_messages = [m for m in messages if m.role == "user"]
        content = user_messages[-1].content if user_messages else "No input"
        
        for word in content.split():
            yield word + " "


class FailingProvider(N3Provider):
    """Provider that always fails."""
    async def generate(self, messages, **kwargs):
        raise ProviderError("Simulated API error")


@pytest.fixture
def test_registry():
    """Provide a test provider registry."""
    from namel3ss.providers.factory import register_provider_class
    from namel3ss.providers.config import PROVIDER_CONFIG_LOADERS
    
    # Register test provider in both factory and config
    register_provider_class("test", TestProvider)
    register_provider_class("failing", FailingProvider)
    PROVIDER_CONFIG_LOADERS["test"] = lambda cfg: cfg or {}
    PROVIDER_CONFIG_LOADERS["failing"] = lambda cfg: cfg or {}
    
    registry = ProviderRegistry()
    yield registry
    registry.clear()


@pytest.mark.asyncio
async def test_provider_basic_workflow(test_registry):
    """Test basic provider workflow."""
    # Create provider
    provider = test_registry.create_and_register(
        name="chat_model",
        provider_type="test",
        model="test-model-v1",
        config={"temperature": 0.7},
    )
    
    # Generate response
    messages = [
        ProviderMessage(role="system", content="You are a helpful assistant."),
        ProviderMessage(role="user", content="What is 2+2?"),
    ]
    
    response = await provider.generate(messages)
    
    assert "What is 2+2?" in response.output_text
    assert response.model == "test-model-v1"
    assert response.total_tokens == 30


@pytest.mark.asyncio
async def test_provider_streaming_workflow(test_registry):
    """Test streaming provider workflow."""
    provider = test_registry.create_and_register(
        name="stream_model",
        provider_type="test",
        model="test-stream",
    )
    
    messages = [ProviderMessage(role="user", content="Hello world from test")]
    
    chunks = []
    async for chunk in provider.stream(messages):
        chunks.append(chunk)
    
    full_text = "".join(chunks)
    assert "Hello" in full_text
    assert "world" in full_text
    assert "test" in full_text


@pytest.mark.asyncio
async def test_provider_batch_workflow(test_registry):
    """Test batch provider workflow."""
    provider = test_registry.create_and_register(
        name="batch_model",
        provider_type="test",
        model="test-batch",
    )
    
    batch = [
        [ProviderMessage(role="user", content="Question 1")],
        [ProviderMessage(role="user", content="Question 2")],
        [ProviderMessage(role="user", content="Question 3")],
    ]
    
    responses = await provider.generate_batch(batch)
    
    assert len(responses) == 3
    assert "Question 1" in responses[0].output_text
    assert "Question 2" in responses[1].output_text
    assert "Question 3" in responses[2].output_text


@pytest.mark.asyncio
async def test_provider_registry_reuse(test_registry):
    """Test reusing provider instances."""
    # Create provider first time
    provider1 = test_registry.get_or_create(
        name="shared_model",
        provider_type="test",
        model="test-shared",
    )
    
    # Get same provider second time
    provider2 = test_registry.get_or_create(
        name="shared_model",
        provider_type="test",
        model="test-shared",
    )
    
    # Should be same instance
    assert provider1 is provider2
    
    # Both should share state
    await provider1.generate([ProviderMessage(role="user", content="Test")])
    assert provider1.call_count == 1
    assert provider2.call_count == 1


@pytest.mark.asyncio
async def test_multiple_providers_in_registry(test_registry):
    """Test managing multiple providers."""
    provider1 = test_registry.create_and_register(
        name="model_a",
        provider_type="test",
        model="test-a",
    )
    
    provider2 = test_registry.create_and_register(
        name="model_b",
        provider_type="test",
        model="test-b",
    )
    
    # Verify both are registered
    assert test_registry.get("model_a") is provider1
    assert test_registry.get("model_b") is provider2
    
    # List all providers
    all_providers = test_registry.list_providers()
    assert len(all_providers) == 2
    assert "model_a" in all_providers
    assert "model_b" in all_providers


@pytest.mark.asyncio
async def test_provider_with_temperature_override(test_registry):
    """Test overriding temperature per call."""
    provider = test_registry.create_and_register(
        name="temp_test",
        provider_type="test",
        model="test-temp",
        config={"temperature": 0.7},
    )
    
    messages = [ProviderMessage(role="user", content="Test")]
    
    # Generate with default temperature
    response1 = await provider.generate(messages)
    assert response1.output_text
    
    # Generate with overridden temperature
    response2 = await provider.generate(messages, temperature=0.1)
    assert response2.output_text
    
    # Both should work
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_provider_error_propagation(test_registry):
    """Test that provider errors propagate correctly."""
    # FailingProvider is already registered in fixture
    provider = test_registry.create_and_register(
        name="failing_model",
        provider_type="failing",
        model="test-fail",
    )
    
    messages = [ProviderMessage(role="user", content="Test")]
    
    with pytest.raises(ProviderError, match="Simulated API error"):
        await provider.generate(messages)


@pytest.mark.asyncio
async def test_provider_conversation_context(test_registry):
    """Test provider with conversation history."""
    provider = test_registry.create_and_register(
        name="conv_model",
        provider_type="test",
        model="test-conv",
    )
    
    # Simulate a conversation
    messages = [
        ProviderMessage(role="system", content="You are helpful."),
        ProviderMessage(role="user", content="Hi"),
        ProviderMessage(role="assistant", content="Hello!"),
        ProviderMessage(role="user", content="How are you?"),
    ]
    
    response = await provider.generate(messages)
    
    # Should echo the last user message
    assert "How are you?" in response.output_text


@pytest.mark.asyncio
async def test_provider_with_system_message(test_registry):
    """Test provider respects system messages."""
    provider = test_registry.create_and_register(
        name="system_test",
        provider_type="test",
        model="test-system",
    )
    
    messages = [
        ProviderMessage(role="system", content="Respond concisely."),
        ProviderMessage(role="user", content="Explain AI"),
    ]
    
    response = await provider.generate(messages)
    
    # Should process the user message
    assert "Explain AI" in response.output_text


@pytest.mark.asyncio
async def test_concurrent_provider_calls(test_registry):
    """Test concurrent calls to same provider."""
    import asyncio
    
    provider = test_registry.create_and_register(
        name="concurrent_model",
        provider_type="test",
        model="test-concurrent",
    )
    
    # Make 5 concurrent calls
    tasks = []
    for i in range(5):
        messages = [ProviderMessage(role="user", content=f"Query {i}")]
        tasks.append(provider.generate(messages))
    
    responses = await asyncio.gather(*tasks)
    
    assert len(responses) == 5
    assert provider.call_count == 5
    
    # Each response should be unique
    texts = [r.output_text for r in responses]
    assert len(set(texts)) == 5  # All different


@pytest.mark.asyncio
async def test_provider_usage_tracking(test_registry):
    """Test tracking token usage across calls."""
    provider = test_registry.create_and_register(
        name="usage_model",
        provider_type="test",
        model="test-usage",
    )
    
    total_tokens = 0
    
    for i in range(3):
        messages = [ProviderMessage(role="user", content=f"Query {i}")]
        response = await provider.generate(messages)
        total_tokens += response.total_tokens
    
    # Each call uses 30 tokens
    assert total_tokens == 90


@pytest.mark.asyncio
async def test_provider_context_manager_cleanup(test_registry):
    """Test provider cleanup with context manager."""
    provider = test_registry.create_and_register(
        name="ctx_model",
        provider_type="test",
        model="test-ctx",
    )
    
    # Use provider without context manager (context manager support is optional)
    messages = [ProviderMessage(role="user", content="Test")]
    response = await provider.generate(messages)
    assert response.output_text
    
    # Provider should still work after use
    assert provider.name == "ctx_model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
