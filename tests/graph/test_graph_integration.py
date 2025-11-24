"""Test integration of graphs with chain execution and backend runtime."""

import pytest
from namel3ss.agents.factory import run_graph_from_state, build_graph_executor
from namel3ss.agents import GraphExecutor
from namel3ss.ast.agents import (
    AgentDefinition,
    GraphDefinition,
    GraphEdge,
    MemoryConfig,
)
from namel3ss.llm.base import BaseLLM, LLMResponse


class MockLLM(BaseLLM):
    """Mock LLM for testing."""
    
    def __init__(self, name: str = "mock_llm", model: str = "mock-model", responses: list = None):
        super().__init__(name, model)
        self.responses = responses or []
        self.call_count = 0
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        response_text = self.responses[self.call_count] if self.call_count < len(self.responses) else "Default"
        self.call_count += 1
        return LLMResponse(
            text=response_text,
            raw={"content": response_text},
            model=self.model,
            finish_reason="stop",
        )
    
    def generate_chat(self, messages: list, **kwargs) -> LLMResponse:
        response_text = self.responses[self.call_count] if self.call_count < len(self.responses) else "Default"
        self.call_count += 1
        return LLMResponse(
            text=response_text,
            raw={"content": response_text},
            model=self.model,
            finish_reason="stop",
        )
    
    def supports_streaming(self) -> bool:
        return False


def test_run_graph_from_state_basic():
    """Test executing graph from encoded state (simulating backend runtime)."""
    # Encode graph as dict (simulating backend state encoding)
    graphs = {
        "research_flow": {
            "name": "research_flow",
            "start_agent": "researcher",
            "edges": [
                {
                    "from_agent": "researcher",
                    "to_agent": "writer",
                    "condition": None,
                }
            ],
            "termination_agents": ["writer"],
            "termination_condition": None,
            "max_hops": 10,
            "timeout_ms": None,
        }
    }
    
    # Encode agents as dict
    agents = {
        "researcher": {
            "name": "researcher",
            "llm_name": "gpt4",
            "tool_names": [],
            "memory_config": None,
            "goal": "Research topics",
            "system_prompt": None,
            "max_turns": None,
            "temperature": None,
            "config": {},
        },
        "writer": {
            "name": "writer",
            "llm_name": "gpt4",
            "tool_names": [],
            "memory_config": None,
            "goal": "Write content",
            "system_prompt": None,
            "max_turns": None,
            "temperature": None,
            "config": {},
        },
    }
    
    # LLM registry
    llm = MockLLM(responses=[
        "Research findings on topic",
        "Written article based on research",
    ])
    llms = {"gpt4": llm}
    
    # Execute graph from state
    result = run_graph_from_state(
        "research_flow",
        graphs,
        agents,
        llms,
        {},
        "Research and write about AI",
    )
    
    assert result["status"] == "success"
    assert result["final_response"] == "Written article based on research"
    assert len(result["hops"]) == 2
    assert result["hops"][0]["agent_name"] == "researcher"
    assert result["hops"][1]["agent_name"] == "writer"


def test_run_graph_from_state_with_memory():
    """Test graph execution with agent memory configuration."""
    graphs = {
        "chat_flow": {
            "name": "chat_flow",
            "start_agent": "chatbot",
            "edges": [],
            "termination_agents": ["chatbot"],
            "termination_condition": None,
            "max_hops": 10,
            "timeout_ms": None,
        }
    }
    
    agents = {
        "chatbot": {
            "name": "chatbot",
            "llm_name": "gpt4",
            "tool_names": [],
            "memory_config": {
                "policy": "conversation_window",
                "window_size": 5,
                "max_items": None,
                "config": {},
            },
            "goal": "Chat with user",
            "system_prompt": "You are a friendly assistant",
            "max_turns": None,
            "temperature": 0.7,
            "config": {},
        },
    }
    
    llm = MockLLM(responses=["Hello! How can I help you?"])
    llms = {"gpt4": llm}
    
    result = run_graph_from_state(
        "chat_flow",
        graphs,
        agents,
        llms,
        {},
        "Hi there!",
    )
    
    assert result["status"] == "success"
    assert "Hello" in result["final_response"]


def test_run_graph_from_state_not_found():
    """Test error handling when graph doesn't exist."""
    result = run_graph_from_state(
        "nonexistent_graph",
        {},  # Empty graphs
        {},
        {},
        {},
        "Input",
    )
    
    assert result["status"] == "error"
    assert "not found" in result["error"]


def test_run_graph_from_state_with_tools():
    """Test graph execution with tool registry."""
    graphs = {
        "calc_flow": {
            "name": "calc_flow",
            "start_agent": "calculator_agent",
            "edges": [],
            "termination_agents": ["calculator_agent"],
            "termination_condition": None,
            "max_hops": 10,
            "timeout_ms": None,
        }
    }
    
    agents = {
        "calculator_agent": {
            "name": "calculator_agent",
            "llm_name": "gpt4",
            "tool_names": ["add"],
            "memory_config": None,
            "goal": "Calculate sums",
            "system_prompt": None,
            "max_turns": None,
            "temperature": None,
            "config": {},
        },
    }
    
    llm = MockLLM(responses=[
        'TOOL_CALL: add(a="5", b="3")',
        "The sum is 8",
    ])
    llms = {"gpt4": llm}
    
    def add_tool(a: str, b: str) -> str:
        return str(int(a) + int(b))
    
    tools = {"add": add_tool}
    
    result = run_graph_from_state(
        "calc_flow",
        graphs,
        agents,
        llms,
        tools,
        "What is 5 + 3?",
    )
    
    assert result["status"] == "success"
    assert len(result["hops"]) == 1
    assert result["hops"][0]["turns"] > 0  # Agent made multiple turns (tool call + response)


def test_run_graph_from_state_with_context():
    """Test passing context variables through graph execution."""
    graphs = {
        "contextual_flow": {
            "name": "contextual_flow",
            "start_agent": "agent",
            "edges": [],
            "termination_agents": ["agent"],
            "termination_condition": None,
            "max_hops": 10,
            "timeout_ms": None,
        }
    }
    
    agents = {
        "agent": {
            "name": "agent",
            "llm_name": "gpt4",
            "tool_names": [],
            "memory_config": None,
            "goal": "Process with context",
            "system_prompt": None,
            "max_turns": None,
            "temperature": None,
            "config": {},
        },
    }
    
    llm = MockLLM(responses=["Processed with context"])
    llms = {"gpt4": llm}
    
    context = {
        "user_id": "123",
        "session_id": "abc",
    }
    
    result = run_graph_from_state(
        "contextual_flow",
        graphs,
        agents,
        llms,
        {},
        "Input",
        context=context,
    )
    
    assert result["status"] == "success"


def test_run_graph_from_state_conditional_routing():
    """Test conditional routing is preserved in state encoding."""
    graphs = {
        "router_flow": {
            "name": "router_flow",
            "start_agent": "router",
            "edges": [
                {
                    "from_agent": "router",
                    "to_agent": "handler_a",
                    "condition": "contains('route_a')",
                },
                {
                    "from_agent": "router",
                    "to_agent": "handler_b",
                    "condition": "contains('route_b')",
                },
            ],
            "termination_agents": ["handler_a", "handler_b"],
            "termination_condition": None,
            "max_hops": 10,
            "timeout_ms": None,
        }
    }
    
    agents = {
        "router": {
            "name": "router",
            "llm_name": "gpt4",
            "tool_names": [],
            "memory_config": None,
            "goal": "Route requests",
            "system_prompt": None,
            "max_turns": None,
            "temperature": None,
            "config": {},
        },
        "handler_a": {
            "name": "handler_a",
            "llm_name": "gpt4",
            "tool_names": [],
            "memory_config": None,
            "goal": "Handle A",
            "system_prompt": None,
            "max_turns": None,
            "temperature": None,
            "config": {},
        },
        "handler_b": {
            "name": "handler_b",
            "llm_name": "gpt4",
            "tool_names": [],
            "memory_config": None,
            "goal": "Handle B",
            "system_prompt": None,
            "max_turns": None,
            "temperature": None,
            "config": {},
        },
    }
    
    llm = MockLLM(responses=[
        "Routing to route_a",
        "Handler A processed",
    ])
    llms = {"gpt4": llm}
    
    result = run_graph_from_state(
        "router_flow",
        graphs,
        agents,
        llms,
        {},
        "Process via A",
    )
    
    assert result["status"] == "success"
    assert result["hops"][1]["agent_name"] == "handler_a"


def test_build_graph_executor_factory():
    """Test the build_graph_executor factory function."""
    agent1 = AgentDefinition(
        name="agent1",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Task 1",
    )
    
    agent2 = AgentDefinition(
        name="agent2",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Task 2",
    )
    
    graph_def = GraphDefinition(
        name="test_graph",
        start_agent="agent1",
        edges=[
            GraphEdge(from_agent="agent1", to_agent="agent2", condition=None),
        ],
        termination_agents=["agent2"],
        termination_condition=None,
        max_hops=10,
    )
    
    agent_registry = {"agent1": agent1, "agent2": agent2}
    llm = MockLLM(responses=["Response 1", "Response 2"])
    llm_registry = {"gpt4": llm}
    
    executor = build_graph_executor(
        graph_def,
        agent_registry,
        llm_registry,
    )
    
    assert isinstance(executor, GraphExecutor)
    assert executor.graph_def.name == "test_graph"
    assert len(executor.agent_runtimes) == 2


def test_graph_result_serialization():
    """Test that graph result converts to dict properly."""
    graphs = {
        "simple_flow": {
            "name": "simple_flow",
            "start_agent": "agent",
            "edges": [],
            "termination_agents": ["agent"],
            "termination_condition": None,
            "max_hops": 10,
            "timeout_ms": None,
        }
    }
    
    agents = {
        "agent": {
            "name": "agent",
            "llm_name": "gpt4",
            "tool_names": [],
            "memory_config": None,
            "goal": "Simple task",
            "system_prompt": None,
            "max_turns": None,
            "temperature": None,
            "config": {},
        },
    }
    
    llm = MockLLM(responses=["Result"])
    llms = {"gpt4": llm}
    
    result = run_graph_from_state(
        "simple_flow",
        graphs,
        agents,
        llms,
        {},
        "Input",
    )
    
    # Verify result structure
    assert "status" in result
    assert "final_response" in result
    assert "hops" in result
    assert "start_agent" in result
    assert "end_agent" in result
    assert "metadata" in result
    
    # Verify hops structure
    assert len(result["hops"]) > 0
    hop = result["hops"][0]
    assert "agent_name" in hop
    assert "response" in hop
    assert "next_agent" in hop
    assert "routing_decision" in hop
    assert "turns" in hop
    assert "metadata" in hop
