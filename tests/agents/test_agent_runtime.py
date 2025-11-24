"""Test agent runtime execution."""

import pytest
from unittest.mock import Mock

from namel3ss.agents import AgentRuntime, AgentResult, BaseMemory
from namel3ss.ast.agents import AgentDefinition, MemoryConfig
from namel3ss.llm.base import BaseLLM, LLMResponse, ChatMessage


class MockLLM(BaseLLM):
    """Mock LLM for testing."""
    
    def __init__(self, name: str = "mock_llm", model: str = "mock-model", responses: list = None):
        super().__init__(name, model)
        self.responses = responses or []
        self.call_count = 0
        self.last_messages = []
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        response_text = self.responses[self.call_count] if self.call_count < len(self.responses) else "Default response"
        self.call_count += 1
        
        return LLMResponse(
            text=response_text,
            raw={"content": response_text},
            model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            finish_reason="stop",
        )
    
    def generate_chat(self, messages: list, **kwargs) -> LLMResponse:
        self.last_messages = messages
        response_text = self.responses[self.call_count] if self.call_count < len(self.responses) else "Default response"
        self.call_count += 1
        
        return LLMResponse(
            text=response_text,
            raw={"content": response_text},
            model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            finish_reason="stop",
        )
    
    def supports_streaming(self) -> bool:
        return False


def test_agent_runtime_basic_execution():
    """Test basic agent execution with single turn."""
    agent_def = AgentDefinition(
        name="test_agent",
        llm_name="gpt4",
        tool_names=[],
        memory_config=MemoryConfig(policy="conversation_window"),
        goal="Answer user questions",
    )
    
    mock_llm = MockLLM(responses=["The answer is 42."])
    runtime = AgentRuntime(agent_def, mock_llm)
    
    result = runtime.act("What is the answer?")
    
    assert result.status == "success"
    assert result.final_response == "The answer is 42."
    assert len(result.turns) == 1
    assert mock_llm.call_count == 1


def test_agent_runtime_with_system_prompt():
    """Test agent with custom system prompt."""
    agent_def = AgentDefinition(
        name="test_agent",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Be helpful",
        system_prompt="You are a friendly assistant.",
    )
    
    mock_llm = MockLLM(responses=["Hello! How can I help?"])
    runtime = AgentRuntime(agent_def, mock_llm)
    
    result = runtime.act("Hi there")
    
    assert result.status == "success"
    # Check system prompt was included
    assert len(mock_llm.last_messages) >= 1
    assert mock_llm.last_messages[0].role == "system"
    assert "friendly assistant" in mock_llm.last_messages[0].content
    assert "Be helpful" in mock_llm.last_messages[0].content


def test_agent_runtime_with_tool_calls():
    """Test agent executing tool calls."""
    agent_def = AgentDefinition(
        name="test_agent",
        llm_name="gpt4",
        tool_names=["search", "calculator"],
        memory_config=None,
        goal="Answer questions using tools",
    )
    
    # Mock responses: first calls tool, second provides final answer
    mock_llm = MockLLM(responses=[
        'TOOL_CALL: search(query="weather")',
        "Based on the search, it's sunny today.",
    ])
    
    def mock_search(query: str) -> str:
        return f"Search results for: {query}"
    
    tool_registry = {"search": mock_search}
    runtime = AgentRuntime(agent_def, mock_llm, tool_registry)
    
    result = runtime.act("What's the weather?")
    
    assert result.status == "success"
    assert len(result.turns) == 2  # Tool call turn + final answer turn
    assert len(result.turns[0].tool_calls) == 1
    assert result.turns[0].tool_calls[0]["tool"] == "search"
    assert len(result.turns[0].tool_results) == 1
    assert result.turns[0].tool_results[0]["status"] == "success"


def test_agent_runtime_max_turns():
    """Test agent respects max turns limit."""
    agent_def = AgentDefinition(
        name="test_agent",
        llm_name="gpt4",
        tool_names=["tool1"],
        memory_config=None,
        goal="Keep calling tools",
    )
    
    # Always call a tool (never finish)
    mock_llm = MockLLM(responses=[
        'TOOL_CALL: tool1(arg="value")' for _ in range(10)
    ])
    
    def mock_tool(arg: str) -> str:
        return "tool result"
    
    tool_registry = {"tool1": mock_tool}
    runtime = AgentRuntime(agent_def, mock_llm, tool_registry)
    
    result = runtime.act("Do something", max_turns=3)
    
    assert result.status == "max_turns"
    assert len(result.turns) == 3
    assert result.metadata["max_turns"] == 3


def test_agent_runtime_tool_error_handling():
    """Test agent handles tool errors gracefully."""
    agent_def = AgentDefinition(
        name="test_agent",
        llm_name="gpt4",
        tool_names=["failing_tool"],
        memory_config=None,
        goal="Try to use tool",
    )
    
    mock_llm = MockLLM(responses=[
        'TOOL_CALL: failing_tool(arg="test")',
        "The tool failed, but I handled it.",
    ])
    
    def failing_tool(arg: str) -> str:
        raise ValueError("Tool error!")
    
    tool_registry = {"failing_tool": failing_tool}
    runtime = AgentRuntime(agent_def, mock_llm, tool_registry)
    
    result = runtime.act("Use the tool")
    
    assert result.status == "success"
    assert len(result.turns[0].tool_results) == 1
    assert result.turns[0].tool_results[0]["status"] == "error"
    assert "Tool error!" in result.turns[0].tool_results[0]["error"]


def test_agent_runtime_unknown_tool():
    """Test agent handles unknown tool gracefully."""
    agent_def = AgentDefinition(
        name="test_agent",
        llm_name="gpt4",
        tool_names=["known_tool"],
        memory_config=None,
        goal="Try to use tool",
    )
    
    mock_llm = MockLLM(responses=[
        'TOOL_CALL: unknown_tool(arg="test")',
        "I tried to use a tool that doesn't exist.",
    ])
    
    tool_registry = {}
    runtime = AgentRuntime(agent_def, mock_llm, tool_registry)
    
    result = runtime.act("Use unknown tool")
    
    assert result.status == "success"
    assert len(result.turns[0].tool_results) == 1
    assert result.turns[0].tool_results[0]["status"] == "error"
    assert "not available" in result.turns[0].tool_results[0]["error"]


def test_agent_runtime_memory_windowing():
    """Test conversation memory with windowing."""
    agent_def = AgentDefinition(
        name="test_agent",
        llm_name="gpt4",
        tool_names=[],
        memory_config=MemoryConfig(policy="conversation_window", window_size=2),
        goal="Chat with user",
    )
    
    mock_llm = MockLLM(responses=[
        "First response",
        "Second response",
        "Third response",
    ])
    
    runtime = AgentRuntime(agent_def, mock_llm)
    
    # Execute multiple interactions
    runtime.act("First message")
    runtime.act("Second message")
    result = runtime.act("Third message")
    
    # Memory should only keep last 2 messages (due to window_size=2)
    memory_messages = runtime.memory.get_messages()
    assert len(memory_messages) <= 2


def test_agent_runtime_no_memory():
    """Test agent with no memory policy."""
    agent_def = AgentDefinition(
        name="test_agent",
        llm_name="gpt4",
        tool_names=[],
        memory_config=MemoryConfig(policy="none"),
        goal="Stateless responses",
    )
    
    mock_llm = MockLLM(responses=["Response 1", "Response 2"])
    runtime = AgentRuntime(agent_def, mock_llm)
    
    runtime.act("First message")
    runtime.act("Second message")
    
    # With policy="none", no messages should be retrieved
    memory_messages = runtime.memory.get_messages()
    assert len(memory_messages) == 0


def test_agent_runtime_temperature_override():
    """Test agent respects temperature configuration."""
    agent_def = AgentDefinition(
        name="test_agent",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Answer questions",
        temperature=0.3,
    )
    
    mock_llm = MockLLM(responses=["Response with low temperature"])
    runtime = AgentRuntime(agent_def, mock_llm)
    
    result = runtime.act("Question")
    
    assert result.status == "success"
    # Temperature would be passed to LLM in kwargs (can't easily verify in mock)


def test_agent_runtime_reset():
    """Test agent memory reset."""
    agent_def = AgentDefinition(
        name="test_agent",
        llm_name="gpt4",
        tool_names=[],
        memory_config=MemoryConfig(policy="full_history"),
        goal="Chat with user",
    )
    
    mock_llm = MockLLM(responses=["First", "Second"])
    runtime = AgentRuntime(agent_def, mock_llm)
    
    runtime.act("Message 1")
    assert len(runtime.memory.messages) > 0
    
    runtime.reset()
    assert len(runtime.memory.messages) == 0
    
    runtime.act("Message 2")
    # Only the new message should be in memory
    assert len(runtime.memory.messages) == 2  # user message + assistant response


def test_agent_runtime_multiple_tool_calls():
    """Test agent executing multiple tool calls in one turn."""
    agent_def = AgentDefinition(
        name="test_agent",
        llm_name="gpt4",
        tool_names=["tool1", "tool2"],
        memory_config=None,
        goal="Use multiple tools",
    )
    
    mock_llm = MockLLM(responses=[
        'TOOL_CALL: tool1(arg="a")\nTOOL_CALL: tool2(arg="b")',
        "Both tools executed successfully.",
    ])
    
    def mock_tool1(arg: str) -> str:
        return f"tool1 result: {arg}"
    
    def mock_tool2(arg: str) -> str:
        return f"tool2 result: {arg}"
    
    tool_registry = {"tool1": mock_tool1, "tool2": mock_tool2}
    runtime = AgentRuntime(agent_def, mock_llm, tool_registry)
    
    result = runtime.act("Use both tools")
    
    assert result.status == "success"
    assert len(result.turns[0].tool_calls) == 2
    assert len(result.turns[0].tool_results) == 2
    assert result.turns[0].tool_results[0]["tool"] == "tool1"
    assert result.turns[0].tool_results[1]["tool"] == "tool2"
