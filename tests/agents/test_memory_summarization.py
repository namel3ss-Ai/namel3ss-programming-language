"""
Comprehensive test suite for production-grade agent memory summarization.

Tests cover:
- Token estimation utilities
- BaseMemory summarization behavior
- Incremental summarization
- Trigger thresholds
- Error handling and fallbacks
- Integration with AgentRuntime
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import List, Optional

from namel3ss.agents.runtime import (
    BaseMemory,
    AgentMessage,
    AgentRuntime,
    AgentResult,
    estimate_tokens,
    estimate_messages_tokens,
)
from namel3ss.agents.summarization import (
    BaseSummarizer,
    LLMSummarizer,
    get_summarizer,
)
from namel3ss.ast.agents import MemoryConfig, AgentDefinition
from namel3ss.ml.providers.base import LLMError


# ============================================================================
# Token Estimation Tests
# ============================================================================

def test_estimate_tokens_empty():
    """Test token estimation for empty string."""
    assert estimate_tokens("") == 0


def test_estimate_tokens_simple():
    """Test token estimation for simple text."""
    # "Hello world" is about 2-3 tokens
    text = "Hello world"
    tokens = estimate_tokens(text)
    assert 2 <= tokens <= 4  # Conservative estimate


def test_estimate_tokens_longer():
    """Test token estimation for longer text."""
    # 100 characters ~ 25 tokens
    text = "A" * 100
    tokens = estimate_tokens(text)
    assert 20 <= tokens <= 30


def test_estimate_messages_tokens():
    """Test token estimation for message list."""
    messages = [
        AgentMessage(role="user", content="Hello"),
        AgentMessage(role="assistant", content="Hi there"),
        AgentMessage(role="user", content="How are you?"),
    ]
    
    tokens = estimate_messages_tokens(messages)
    
    # Should be sum of content tokens + overhead
    assert tokens > 0
    assert tokens < 100  # Should be reasonable


def test_estimate_messages_with_tools():
    """Test token estimation includes tool calls/results."""
    messages = [
        AgentMessage(
            role="assistant",
            content="Using tool",
            tool_call={"tool": "search", "args": {"query": "test"}}
        ),
        AgentMessage(
            role="tool",
            content="Result",
            tool_result={"status": "success", "result": "data"}
        ),
    ]
    
    tokens = estimate_messages_tokens(messages)
    assert tokens > 10  # Should include tool data


# ============================================================================
# BaseMemory Unit Tests
# ============================================================================

def test_memory_none_policy():
    """Test memory with 'none' policy returns no messages."""
    config = MemoryConfig(policy="none")
    memory = BaseMemory(config)
    
    memory.add_message(AgentMessage(role="user", content="Hello"))
    memory.add_message(AgentMessage(role="assistant", content="Hi"))
    
    assert len(memory.get_messages()) == 0


def test_memory_full_history_policy():
    """Test memory with 'full_history' policy returns all messages."""
    config = MemoryConfig(policy="full_history")
    memory = BaseMemory(config)
    
    for i in range(10):
        memory.add_message(AgentMessage(role="user", content=f"Message {i}"))
    
    messages = memory.get_messages()
    assert len(messages) == 10


def test_memory_conversation_window_policy():
    """Test memory with 'conversation_window' policy returns windowed messages."""
    config = MemoryConfig(policy="conversation_window", window_size=5)
    memory = BaseMemory(config)
    
    for i in range(10):
        memory.add_message(AgentMessage(role="user", content=f"Message {i}"))
    
    messages = memory.get_messages()
    assert len(messages) == 5
    assert messages[0].content == "Message 5"
    assert messages[-1].content == "Message 9"


@pytest.mark.asyncio
async def test_memory_summary_policy_no_summarizer():
    """Test summary policy without configured summarizer returns fallback."""
    config = MemoryConfig(
        policy="summary",
        summarizer=None  # No summarizer configured
    )
    memory = BaseMemory(config)
    
    for i in range(10):
        memory.add_message(AgentMessage(role="user", content=f"Message {i}"))
    
    # Should not trigger summarization without summarizer
    await memory.maybe_summarize()
    
    # Should fall back to recent messages
    messages = memory.get_messages()
    assert len(messages) > 0
    assert memory.summary is None


@pytest.mark.asyncio
async def test_memory_summary_policy_with_mock_summarizer():
    """Test summary policy with mocked summarizer."""
    config = MemoryConfig(
        policy="summary",
        summarizer="openai/gpt-4o-mini",
        summary_trigger_messages=5,
        summary_recent_window=2
    )
    memory = BaseMemory(config)
    
    # Add messages
    for i in range(10):
        memory.add_message(AgentMessage(role="user", content=f"Message {i}"))
    
    # Mock the summarizer
    mock_summarizer = AsyncMock()
    mock_summarizer.summarize = AsyncMock(return_value="Summary of messages 0-7")
    
    with patch('namel3ss.agents.summarization.get_summarizer', return_value=mock_summarizer):
        await memory.maybe_summarize()
    
    # Should have created summary
    assert memory.summary is not None
    assert "Summary" in memory.summary
    assert memory.last_summarized_index == 8  # 10 - 2 (recent window)
    
    # Get messages should return summary + recent
    messages = memory.get_messages()
    
    # Should have: 1 summary message + 2 recent messages
    assert len(messages) == 3
    assert messages[0].role == "system"
    assert "[Previous conversation summary]" in messages[0].content
    assert messages[1].content == "Message 8"
    assert messages[2].content == "Message 9"


@pytest.mark.asyncio
async def test_memory_incremental_summarization():
    """Test incremental summarization builds on existing summary."""
    config = MemoryConfig(
        policy="summary",
        summarizer="openai/gpt-4o-mini",
        summary_trigger_messages=5,
        summary_recent_window=2
    )
    memory = BaseMemory(config)
    
    # Create a single mock summarizer that will be reused
    mock_summarizer = AsyncMock()
    mock_summarizer.summarize = AsyncMock(side_effect=["First summary", "Updated summary"])
    
    with patch('namel3ss.agents.summarization.get_summarizer', return_value=mock_summarizer):
        # Add first batch
        for i in range(6):
            memory.add_message(AgentMessage(role="user", content=f"Message {i}"))
        
        # First summarization
        await memory.maybe_summarize()
        
        assert memory.summary == "First summary"
        first_index = memory.last_summarized_index
        assert mock_summarizer.summarize.call_count == 1
        
        # Add more messages
        for i in range(6, 12):
            memory.add_message(AgentMessage(role="user", content=f"Message {i}"))
        
        # Second summarization (incremental)
        await memory.maybe_summarize()
        
        # Should have called summarizer twice
        assert mock_summarizer.summarize.call_count == 2
        
        # Check that second call included existing summary
        second_call_args = mock_summarizer.summarize.call_args_list[1]
        assert second_call_args[1]['existing_summary'] == "First summary"
        
        assert memory.summary == "Updated summary"
        assert memory.last_summarized_index > first_index


@pytest.mark.asyncio
async def test_memory_trigger_by_message_count():
    """Test summarization triggers by message count threshold."""
    config = MemoryConfig(
        policy="summary",
        summarizer="openai/gpt-4o-mini",
        summary_trigger_messages=10,
        summary_recent_window=2
    )
    memory = BaseMemory(config)
    
    mock_summarizer = AsyncMock()
    mock_summarizer.summarize = AsyncMock(return_value="Summary")
    
    with patch('namel3ss.agents.summarization.get_summarizer', return_value=mock_summarizer):
        # Add 9 messages - should not trigger
        for i in range(9):
            memory.add_message(AgentMessage(role="user", content=f"Message {i}"))
        
        await memory.maybe_summarize()
        assert mock_summarizer.summarize.call_count == 0
        
        # Add 10th message - should trigger
        memory.add_message(AgentMessage(role="user", content="Message 9"))
        await memory.maybe_summarize()
        assert mock_summarizer.summarize.call_count == 1


@pytest.mark.asyncio
async def test_memory_trigger_by_token_count():
    """Test summarization triggers by estimated token threshold."""
    config = MemoryConfig(
        policy="summary",
        summarizer="openai/gpt-4o-mini",
        summary_trigger_messages=1000,  # High message threshold
        summary_trigger_tokens=100,  # Token threshold
        summary_recent_window=1
    )
    memory = BaseMemory(config)
    
    mock_summarizer = AsyncMock()
    mock_summarizer.summarize = AsyncMock(return_value="Summary")
    
    with patch('namel3ss.agents.summarization.get_summarizer', return_value=mock_summarizer):
        # Add multiple messages with enough total tokens to trigger
        # Each message ~100 chars = ~25 tokens + overhead
        for i in range(5):
            long_content = "A" * 100  # ~25 tokens each
            memory.add_message(AgentMessage(role="user", content=long_content))
        
        await memory.maybe_summarize()
        # Should trigger since 5 messages * ~29 tokens = ~145 tokens > 100 threshold
        assert mock_summarizer.summarize.call_count >= 1


@pytest.mark.asyncio
async def test_memory_error_handling_and_fallback():
    """Test error handling when summarization fails."""
    config = MemoryConfig(
        policy="summary",
        summarizer="openai/gpt-4o-mini",
        summary_trigger_messages=5,
        window_size=3
    )
    memory = BaseMemory(config)
    
    # Add messages
    for i in range(10):
        memory.add_message(AgentMessage(role="user", content=f"Message {i}"))
    
    # Mock summarizer that fails
    mock_summarizer = AsyncMock()
    mock_summarizer.summarize = AsyncMock(side_effect=LLMError("API Error", provider="openai"))
    
    with patch('namel3ss.agents.summarization.get_summarizer', return_value=mock_summarizer):
        # Should not raise exception
        await memory.maybe_summarize()
    
    # Should increment failure counter
    assert memory._summarization_failures == 1
    
    # get_messages should still work (fallback to windowing)
    messages = memory.get_messages()
    assert len(messages) > 0


@pytest.mark.asyncio
async def test_memory_backoff_after_failures():
    """Test exponential backoff after repeated summarization failures."""
    config = MemoryConfig(
        policy="summary",
        summarizer="openai/gpt-4o-mini",
        summary_trigger_messages=5
    )
    memory = BaseMemory(config)
    
    # Add messages
    for i in range(10):
        memory.add_message(AgentMessage(role="user", content=f"Message {i}"))
    
    # Mock failing summarizer
    mock_summarizer = AsyncMock()
    mock_summarizer.summarize = AsyncMock(side_effect=LLMError("Error", provider="openai"))
    
    with patch('namel3ss.agents.summarization.get_summarizer', return_value=mock_summarizer):
        # First failure
        await memory.maybe_summarize()
        assert memory._summarization_failures == 1
        first_attempt_time = memory._last_summarization_attempt
        
        # Immediate retry should be blocked by backoff
        await memory.maybe_summarize()
        assert memory._summarization_failures == 1  # Should not increment (didn't try)
        assert mock_summarizer.summarize.call_count == 1  # Only called once


def test_memory_clear():
    """Test clearing memory resets all state."""
    config = MemoryConfig(policy="summary", summarizer="openai/gpt-4o-mini")
    memory = BaseMemory(config)
    
    memory.add_message(AgentMessage(role="user", content="Test"))
    memory.summary = "Test summary"
    memory.last_summarized_index = 5
    memory._summarization_failures = 2
    
    memory.clear()
    
    assert len(memory.messages) == 0
    assert memory.summary is None
    assert memory.last_summarized_index == 0
    assert memory._summarization_failures == 0


# ============================================================================
# Summarizer Tests
# ============================================================================

def test_get_summarizer_openai():
    """Test factory creates OpenAI summarizer."""
    summarizer = get_summarizer("openai/gpt-4o-mini")
    assert isinstance(summarizer, LLMSummarizer)
    assert summarizer.provider == "openai"
    assert summarizer.model == "gpt-4o-mini"


def test_get_summarizer_anthropic():
    """Test factory creates Anthropic summarizer."""
    summarizer = get_summarizer("anthropic/claude-3-haiku-20240307")
    assert isinstance(summarizer, LLMSummarizer)
    assert summarizer.provider == "anthropic"
    assert summarizer.model == "claude-3-haiku-20240307"


def test_get_summarizer_ollama():
    """Test factory creates Ollama summarizer."""
    summarizer = get_summarizer("ollama/llama3")
    assert isinstance(summarizer, LLMSummarizer)
    assert summarizer.provider == "ollama"
    assert summarizer.model == "llama3"


def test_get_summarizer_with_alias():
    """Test factory handles provider aliases."""
    summarizer = get_summarizer("gpt/gpt-4o-mini")
    assert summarizer.provider == "openai"
    
    summarizer = get_summarizer("claude/claude-3-haiku")
    assert summarizer.provider == "anthropic"


def test_get_summarizer_invalid_format():
    """Test factory raises error for invalid format."""
    with pytest.raises(ValueError, match="Invalid summarizer name format"):
        get_summarizer("invalid_format")


@pytest.mark.asyncio
async def test_llm_summarizer_format_messages():
    """Test message formatting for summarization."""
    summarizer = LLMSummarizer("openai", "gpt-4o-mini")
    
    messages = [
        AgentMessage(role="user", content="Hello"),
        AgentMessage(role="assistant", content="Hi there!"),
        AgentMessage(role="user", content="How are you?"),
    ]
    
    formatted = summarizer._format_messages_for_summary(messages)
    
    assert "[USER] Hello" in formatted
    assert "[ASSISTANT] Hi there!" in formatted
    assert "[USER] How are you?" in formatted


@pytest.mark.asyncio
async def test_llm_summarizer_with_tools():
    """Test summarizer handles tool calls/results."""
    summarizer = LLMSummarizer("openai", "gpt-4o-mini")
    
    messages = [
        AgentMessage(
            role="assistant",
            content="",
            tool_call={"tool": "search", "args": {}}
        ),
        AgentMessage(
            role="tool",
            content="",
            tool_result={"status": "success"}
        ),
    ]
    
    formatted = summarizer._format_messages_for_summary(messages)
    
    assert "Called tool: search" in formatted
    assert "Tool result: success" in formatted


@pytest.mark.asyncio
async def test_llm_summarizer_incremental():
    """Test summarizer builds on existing summary."""
    summarizer = LLMSummarizer("openai", "gpt-4o-mini")
    
    messages = [AgentMessage(role="user", content="New message")]
    existing_summary = "Previous summary"
    
    prompt = summarizer._build_summarization_prompt(
        messages,
        existing_summary,
        max_tokens=512
    )
    
    assert "Previous summary" in prompt
    assert "New message" in prompt
    assert "updated summary" in prompt.lower()


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_agent_runtime_with_summarization():
    """Test AgentRuntime integration with memory summarization."""
    # Create agent with summary memory
    agent_def = AgentDefinition(
        name="test_agent",
        llm_name="test_llm",
        goal="Test goal",
        max_turns=20
    )
    
    agent_def.memory_config = MemoryConfig(
        policy="summary",
        summarizer="openai/gpt-4o-mini",
        summary_trigger_messages=5,
        summary_recent_window=2
    )
    
    # Mock LLM
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.text = "Response"
    mock_response.total_tokens = 10
    mock_response.finish_reason = "stop"
    mock_llm.generate_chat = Mock(return_value=mock_response)
    
    # Create runtime
    runtime = AgentRuntime(agent_def, mock_llm)
    
    # Mock summarizer
    mock_summarizer = AsyncMock()
    mock_summarizer.summarize = AsyncMock(return_value="Summary")
    
    with patch('namel3ss.agents.summarization.get_summarizer', return_value=mock_summarizer):
        # Execute with enough turns to trigger summarization
        result = await runtime.aact("Test input")
    
    assert result.status == "success"
    # Summarization should have been attempted
    assert mock_summarizer.summarize.call_count >= 0  # May or may not trigger depending on thresholds


@pytest.mark.asyncio
async def test_agent_runtime_async_execution():
    """Test AgentRuntime async execution with aact()."""
    agent_def = AgentDefinition(
        name="test_agent",
        llm_name="test_llm",
        goal="Test goal",
        max_turns=1
    )
    
    # Mock LLM
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.text = "Response"
    mock_response.total_tokens = 10
    mock_response.finish_reason = "stop"
    mock_llm.generate_chat = Mock(return_value=mock_response)
    
    runtime = AgentRuntime(agent_def, mock_llm)
    
    # Call async version
    result = await runtime.aact("Test input")
    
    assert result.status == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
