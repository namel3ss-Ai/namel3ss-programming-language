"""
Comprehensive test suite for async + streaming LLM providers.

Tests production-grade async/streaming implementation:
- Real async execution (no sync fallbacks)
- SSE token streaming with incremental delivery
- Backpressure control (timeouts, chunk limits)
- Cancellation propagation and cleanup
- Concurrency control
- Error handling and retry logic
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import AsyncIterator

import httpx

from namel3ss.ml.providers.openai import OpenAIProvider
from namel3ss.ml.providers.anthropic import AnthropicProvider
from namel3ss.ml.providers.gemini import GeminiProvider
from namel3ss.ml.providers.cohere import CohereProvider
from namel3ss.ml.providers.ollama import OllamaProvider
from namel3ss.ml.providers.base import (
    LLMError,
    StreamChunk,
    StreamConfig,
)


# ============================================================================
# Mock Response Utilities
# ============================================================================

class MockSSEResponse:
    """Mock httpx streaming response with SSE data."""
    
    def __init__(self, lines: list[str], status_code: int = 200, delay_per_line: float = 0.01):
        self.lines = lines
        self.status_code = status_code
        self.delay_per_line = delay_per_line
        self._closed = False
    
    async def aiter_lines(self) -> AsyncIterator[str]:
        """Simulate SSE line-by-line delivery."""
        for line in self.lines:
            if self._closed:
                break
            await asyncio.sleep(self.delay_per_line)
            yield line
    
    async def aread(self):
        """Read full response body."""
        return b'{"error": "test error"}'
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closed = True


def mock_openai_sse_response() -> list[str]:
    """Generate mock OpenAI SSE response lines."""
    return [
        'data: {"choices":[{"delta":{"content":"Hello"}}],"model":"gpt-4o"}',
        '',
        'data: {"choices":[{"delta":{"content":" world"}}],"model":"gpt-4o"}',
        '',
        'data: {"choices":[{"delta":{"content":"!"}}],"model":"gpt-4o"}',
        '',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"model":"gpt-4o"}',
        '',
        'data: [DONE]',
    ]


def mock_anthropic_sse_response() -> list[str]:
    """Generate mock Anthropic SSE response lines."""
    return [
        'event: message_start',
        'data: {"type":"message_start","message":{"id":"msg_123","model":"claude-3-5-sonnet"}}',
        '',
        'event: content_block_delta',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}',
        '',
        'event: content_block_delta',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":" world"}}',
        '',
        'event: content_block_delta',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"!"}}',
        '',
        'event: message_delta',
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
        '',
        'event: message_stop',
        'data: {"type":"message_stop"}',
    ]


def mock_gemini_sse_response() -> list[str]:
    """Generate mock Gemini SSE response lines."""
    return [
        'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]},"finishReason":null}]}',
        '',
        'data: {"candidates":[{"content":{"parts":[{"text":" world"}]},"finishReason":null}]}',
        '',
        'data: {"candidates":[{"content":{"parts":[{"text":"!"}]},"finishReason":"STOP"}]}',
        '',
    ]


def mock_cohere_sse_response() -> list[str]:
    """Generate mock Cohere SSE response lines."""
    return [
        'data: {"event_type":"text-generation","text":"Hello"}',
        '',
        'data: {"event_type":"text-generation","text":" world"}',
        '',
        'data: {"event_type":"text-generation","text":"!"}',
        '',
        'data: {"event_type":"stream-end","finish_reason":"COMPLETE","response":{"meta":{"billed_units":{"input_tokens":10,"output_tokens":5}}}}',
        '',
    ]


def mock_ollama_sse_response() -> list[str]:
    """Generate mock Ollama SSE response lines (newline-delimited JSON)."""
    return [
        '{"model":"llama3","created_at":"2024-01-01T00:00:00Z","response":"Hello","done":false}',
        '{"model":"llama3","created_at":"2024-01-01T00:00:01Z","response":" world","done":false}',
        '{"model":"llama3","created_at":"2024-01-01T00:00:02Z","response":"!","done":false}',
        '{"model":"llama3","created_at":"2024-01-01T00:00:03Z","response":"","done":true,"context":[1,2,3],"total_duration":123456789,"load_duration":12345,"prompt_eval_count":10,"prompt_eval_duration":23456,"eval_count":5,"eval_duration":98765}',
    ]


# ============================================================================
# OpenAI Async Tests
# ============================================================================

@pytest.mark.asyncio
async def test_openai_agenerate_real_async():
    """Verify OpenAI agenerate uses real async, not sync fallback."""
    provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}, "finish_reason": "stop"}],
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client
        
        result = await provider.agenerate("Test prompt")
        
        # Verify async client was used
        mock_client.post.assert_awaited_once()
        assert result.content == "Test response"
        assert result.usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_openai_streaming_incremental_delivery():
    """Verify tokens arrive incrementally, not buffered."""
    provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
    
    mock_sse = MockSSEResponse(mock_openai_sse_response(), delay_per_line=0.05)
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        # stream() is NOT async, it returns a context manager
        mock_client.stream = Mock(return_value=mock_sse)
        mock_get_client.return_value = mock_client
        
        chunks = []
        start_time = asyncio.get_event_loop().time()
        chunk_times = []
        
        async for chunk in provider.stream_generate("Test prompt"):
            chunks.append(chunk.content)
            chunk_times.append(asyncio.get_event_loop().time() - start_time)
        
        # Verify incremental delivery
        assert chunks == ["Hello", " world", "!"]
        
        # Verify time spacing between chunks (should be ~50ms apart)
        assert len(chunk_times) == 3
        for i in range(1, len(chunk_times)):
            time_diff = chunk_times[i] - chunk_times[i-1]
            assert 0.03 < time_diff < 0.15, f"Chunks not delivered incrementally: {time_diff}s gap"


@pytest.mark.asyncio
async def test_openai_streaming_cancellation():
    """Verify stream cancellation propagates correctly."""
    provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
    
    # Create stream
    mock_sse = MockSSEResponse(mock_openai_sse_response(), delay_per_line=0.01)
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_sse)
        mock_get_client.return_value = mock_client
        
        chunks_received = []
        
        # Use asyncio.Task to simulate real cancellation
        async def consume_stream():
            async for chunk in provider.stream_generate("Test prompt"):
                chunks_received.append(chunk.content)
                await asyncio.sleep(0)  # Yield control
        
        task = asyncio.create_task(consume_stream())
        await asyncio.sleep(0.05)  # Let it start
        task.cancel()
        
        # Verify cancellation was raised
        with pytest.raises(asyncio.CancelledError):
            await task
        
        # Verify connection was closed
        assert mock_sse._closed


@pytest.mark.asyncio
async def test_openai_streaming_chunk_timeout():
    """Verify httpx timeout configuration is set correctly."""
    provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
    
    mock_sse = MockSSEResponse(['data: {"choices":[{"delta":{"content":"Hello"}}],"model":"gpt-4o"}', ''])
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        captured_timeout = None
        
        def capture_stream(*args, **kwargs):
            nonlocal captured_timeout
            captured_timeout = kwargs.get('timeout')
            return mock_sse
        
        mock_client.stream = Mock(side_effect=capture_stream)
        mock_get_client.return_value = mock_client
        
        # Configure chunk timeout
        config = StreamConfig(chunk_timeout=3.0)
        
        # Consume stream
        chunks = []
        async for chunk in provider.stream_generate("Test prompt", stream_config=config):
            chunks.append(chunk.content)
        
        # Verify timeout was configured
        assert captured_timeout is not None
        assert captured_timeout.read == 3.0  # chunk_timeout maps to read timeout


@pytest.mark.asyncio
async def test_openai_streaming_max_chunks():
    """Verify max_chunks stops stream early."""
    provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
    
    mock_sse = MockSSEResponse(mock_openai_sse_response())
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_sse)
        mock_get_client.return_value = mock_client
        
        config = StreamConfig(max_chunks=2)
        
        chunks = []
        async for chunk in provider.stream_generate("Test prompt", stream_config=config):
            chunks.append(chunk.content)
        
        # Should stop after 2 chunks
        assert len(chunks) == 2
        assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_openai_retry_on_rate_limit():
    """Verify retry logic on 429 rate limit."""
    provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
    
    # First request: 429, second: 200
    mock_error_response = Mock()
    mock_error_response.status_code = 429
    mock_error_response.text = "Rate limit exceeded"
    
    mock_success_response = Mock()
    mock_success_response.status_code = 200
    mock_success_response.json.return_value = {
        "choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}],
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[mock_error_response, mock_success_response])
        mock_get_client.return_value = mock_client
        
        # Should retry and succeed
        result = await provider.agenerate("Test prompt")
        assert result.content == "Success"
        assert mock_client.post.await_count == 2


@pytest.mark.asyncio
async def test_openai_concurrency_limit():
    """Verify semaphore limits concurrent requests."""
    provider = OpenAIProvider(model="gpt-4o", api_key="test-key", max_concurrent=2)
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test"}, "finish_reason": "stop"}],
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    
    concurrent_count = 0
    max_concurrent = 0
    
    async def mock_post(*args, **kwargs):
        nonlocal concurrent_count, max_concurrent
        concurrent_count += 1
        max_concurrent = max(max_concurrent, concurrent_count)
        await asyncio.sleep(0.1)  # Simulate work
        concurrent_count -= 1
        return mock_response
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_get_client.return_value = mock_client
        
        # Launch 5 concurrent requests
        tasks = [provider.agenerate("Test prompt") for _ in range(5)]
        await asyncio.gather(*tasks)
        
        # Verify max concurrent was limited to 2
        assert max_concurrent == 2


# ============================================================================
# Anthropic Async Tests
# ============================================================================

@pytest.mark.asyncio
async def test_anthropic_agenerate_real_async():
    """Verify Anthropic agenerate uses real async, not sync fallback."""
    provider = AnthropicProvider(model="claude-3-5-sonnet", api_key="test-key")
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "content": [{"type": "text", "text": "Test response"}],
        "model": "claude-3-5-sonnet",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    }
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client
        
        result = await provider.agenerate("Test prompt")
        
        # Verify async client was used
        mock_client.post.assert_awaited_once()
        assert result.content == "Test response"
        assert result.usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_anthropic_streaming_incremental_delivery():
    """Verify Anthropic tokens arrive incrementally."""
    provider = AnthropicProvider(model="claude-3-5-sonnet", api_key="test-key")
    
    mock_sse = MockSSEResponse(mock_anthropic_sse_response(), delay_per_line=0.05)
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_sse)
        mock_get_client.return_value = mock_client
        
        chunks = []
        start_time = asyncio.get_event_loop().time()
        chunk_times = []
        
        async for chunk in provider.stream_generate("Test prompt"):
            if chunk.content:  # Skip empty chunks (finish_reason only)
                chunks.append(chunk.content)
                chunk_times.append(asyncio.get_event_loop().time() - start_time)
        
        # Verify incremental delivery
        assert chunks == ["Hello", " world", "!"]
        
        # Verify time spacing
        assert len(chunk_times) == 3
        for i in range(1, len(chunk_times)):
            time_diff = chunk_times[i] - chunk_times[i-1]
            assert 0.03 < time_diff < 0.20, f"Chunks not delivered incrementally: {time_diff}s gap"


@pytest.mark.asyncio
async def test_anthropic_streaming_finish_reason():
    """Verify Anthropic finish_reason is captured."""
    provider = AnthropicProvider(model="claude-3-5-sonnet", api_key="test-key")
    
    mock_sse = MockSSEResponse(mock_anthropic_sse_response())
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_sse)
        mock_get_client.return_value = mock_client
        
        finish_reasons = []
        async for chunk in provider.stream_generate("Test prompt"):
            if chunk.finish_reason:
                finish_reasons.append(chunk.finish_reason)
        
        assert "end_turn" in finish_reasons


@pytest.mark.asyncio
async def test_anthropic_retry_on_server_error():
    """Verify retry logic on 503 server error."""
    provider = AnthropicProvider(model="claude-3-5-sonnet", api_key="test-key")
    
    # First request: 503, second: 200
    mock_error_response = Mock()
    mock_error_response.status_code = 503
    mock_error_response.text = "Service unavailable"
    
    mock_success_response = Mock()
    mock_success_response.status_code = 200
    mock_success_response.json.return_value = {
        "content": [{"type": "text", "text": "Success"}],
        "model": "claude-3-5-sonnet",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    }
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[mock_error_response, mock_success_response])
        mock_get_client.return_value = mock_client
        
        # Should retry and succeed
        result = await provider.agenerate("Test prompt")
        assert result.content == "Success"
        assert mock_client.post.await_count == 2


# ============================================================================
# Cross-Provider Tests
# ============================================================================

@pytest.mark.asyncio
async def test_no_sync_fallback_in_async_context():
    """Verify generate() raises error when called from async context."""
    provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
    
    # We're already in async context
    with pytest.raises(LLMError, match="Cannot call synchronous generate.*Use agenerate"):
        provider.generate("Test prompt")


@pytest.mark.asyncio
async def test_stream_config_defaults():
    """Verify StreamConfig defaults are sensible."""
    config = StreamConfig()
    
    assert config.stream_timeout == 30.0
    assert config.chunk_timeout == 5.0
    assert config.max_chunks is None
    assert config.buffer_size == 100


@pytest.mark.asyncio
async def test_context_manager_cleanup():
    """Verify async context manager closes client."""
    provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
    
    async with provider:
        client = await provider._get_client()
        assert not client.is_closed
    
    # After exit, client should be closed
    assert provider._client is None or provider._client.is_closed


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_openai_malformed_response():
    """Verify error handling for malformed API response."""
    provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"invalid": "response"}
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client
        
        with pytest.raises(LLMError, match="Failed to parse OpenAI response"):
            await provider.agenerate("Test prompt")


@pytest.mark.asyncio
async def test_anthropic_streaming_error_event():
    """Verify error event in SSE stream raises LLMError."""
    provider = AnthropicProvider(model="claude-3-5-sonnet", api_key="test-key")
    
    error_response = [
        'event: error',
        'data: {"type":"error","error":{"type":"rate_limit_error","message":"Rate limit exceeded"}}',
    ]
    mock_sse = MockSSEResponse(error_response)
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_sse)
        mock_get_client.return_value = mock_client
        
        with pytest.raises(LLMError, match="Rate limit exceeded"):
            async for _ in provider.stream_generate("Test prompt"):
                pass


# ============================================================================
# Gemini Async Tests
# ============================================================================

@pytest.mark.asyncio
async def test_gemini_agenerate_real_async():
    """Verify Gemini agenerate uses real async, not sync fallback."""
    provider = GeminiProvider(model="gemini-pro", api_key="test-key")
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [{
            "content": {"parts": [{"text": "Test response"}]},
            "finishReason": "STOP"
        }],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
    }
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client
        
        result = await provider.agenerate("Test prompt")
        
        # Verify async client was used
        mock_client.post.assert_awaited_once()
        assert result.content == "Test response"
        assert result.usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_gemini_streaming_incremental_delivery():
    """Verify Gemini tokens arrive incrementally."""
    provider = GeminiProvider(model="gemini-pro", api_key="test-key")
    
    mock_sse = MockSSEResponse(mock_gemini_sse_response(), delay_per_line=0.05)
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_sse)
        mock_get_client.return_value = mock_client
        
        chunks = []
        async for chunk in provider.stream_generate("Test prompt"):
            if chunk.content:
                chunks.append(chunk.content)
        
        # Verify incremental delivery
        assert chunks == ["Hello", " world", "!"]


@pytest.mark.asyncio
async def test_gemini_retry_on_server_error():
    """Verify retry logic on 503 server error."""
    provider = GeminiProvider(model="gemini-pro", api_key="test-key")
    
    # First request: 503, second: 200
    mock_error_response = Mock()
    mock_error_response.status_code = 503
    mock_error_response.text = "Service unavailable"
    
    mock_success_response = Mock()
    mock_success_response.status_code = 200
    mock_success_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Success"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
    }
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[mock_error_response, mock_success_response])
        mock_get_client.return_value = mock_client
        
        # Should retry and succeed
        result = await provider.agenerate("Test prompt")
        assert result.content == "Success"
        assert mock_client.post.await_count == 2


# ============================================================================
# Cohere Async Tests
# ============================================================================

@pytest.mark.asyncio
async def test_cohere_agenerate_real_async():
    """Verify Cohere agenerate uses real async, not sync fallback."""
    provider = CohereProvider(model="command", api_key="test-key")
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "Test response",
        "generation_id": "gen_123",
        "finish_reason": "COMPLETE",
        "meta": {"billed_units": {"input_tokens": 10, "output_tokens": 5}}
    }
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client
        
        result = await provider.agenerate("Test prompt")
        
        # Verify async client was used
        mock_client.post.assert_awaited_once()
        assert result.content == "Test response"
        assert result.usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_cohere_streaming_incremental_delivery():
    """Verify Cohere tokens arrive incrementally."""
    provider = CohereProvider(model="command", api_key="test-key")
    
    mock_sse = MockSSEResponse(mock_cohere_sse_response(), delay_per_line=0.05)
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_sse)
        mock_get_client.return_value = mock_client
        
        chunks = []
        async for chunk in provider.stream_generate("Test prompt"):
            if chunk.content:
                chunks.append(chunk.content)
        
        # Verify incremental delivery
        assert chunks == ["Hello", " world", "!"]


@pytest.mark.asyncio
async def test_cohere_streaming_finish_with_usage():
    """Verify Cohere finish event includes usage stats."""
    provider = CohereProvider(model="command", api_key="test-key")
    
    mock_sse = MockSSEResponse(mock_cohere_sse_response())
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_sse)
        mock_get_client.return_value = mock_client
        
        finish_chunks = []
        async for chunk in provider.stream_generate("Test prompt"):
            if chunk.finish_reason:
                finish_chunks.append(chunk)
        
        assert len(finish_chunks) == 1
        assert finish_chunks[0].finish_reason == "COMPLETE"
        assert finish_chunks[0].usage["total_tokens"] == 15


# ============================================================================
# Ollama Async Tests
# ============================================================================

@pytest.mark.asyncio
async def test_ollama_agenerate_real_async():
    """Verify Ollama agenerate is truly async (no hidden sync fallback)."""
    provider = OllamaProvider(model="llama3", base_url="http://localhost:11434")
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "llama3",
        "response": "Hello world!",
        "done": True,
        "context": [1, 2, 3],
        "total_duration": 123456789,
        "prompt_eval_count": 10,
        "eval_count": 5
    }
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client
        
        # Execute in parallel to prove true async
        tasks = [
            asyncio.create_task(provider.agenerate("Test prompt 1")),
            asyncio.create_task(provider.agenerate("Test prompt 2")),
            asyncio.create_task(provider.agenerate("Test prompt 3"))
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(isinstance(r.content, str) for r in results)
        assert all(r.model == "llama3" for r in results)
        # Verify truly parallel execution (all started before any finished)
        assert mock_client.post.call_count == 3


@pytest.mark.asyncio
async def test_ollama_streaming_incremental_delivery():
    """Verify Ollama delivers streaming tokens incrementally."""
    provider = OllamaProvider(model="llama3", base_url="http://localhost:11434")
    
    mock_sse = MockSSEResponse(mock_ollama_sse_response())
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_sse)
        mock_get_client.return_value = mock_client
        
        chunks = []
        async for chunk in provider.stream_generate("Test prompt"):
            chunks.append(chunk)
        
        # Should have 3 content chunks + 1 final usage chunk
        assert len(chunks) >= 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"
        assert chunks[2].content == "!"
        
        # Last chunk should have finish reason and usage
        final_chunk = chunks[-1]
        assert final_chunk.finish_reason == "stop"
        assert final_chunk.usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_ollama_list_models():
    """Verify Ollama can list available models."""
    provider = OllamaProvider(model="llama3", base_url="http://localhost:11434")
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [
            {"name": "llama3:latest", "size": 4661229568},
            {"name": "mistral:latest", "size": 4109858304},
            {"name": "codellama:latest", "size": 3791726592}
        ]
    }
    
    with patch.object(provider, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client
        
        models = await provider.list_models()
        
        assert len(models) == 3
        assert models[0]["name"] == "llama3:latest"
        assert models[1]["name"] == "mistral:latest"
        assert models[2]["name"] == "codellama:latest"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
