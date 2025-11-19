"""Test graph executor for multi-agent orchestration."""

import pytest
from unittest.mock import Mock

from namel3ss.agents import GraphExecutor, AgentRuntime
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


def test_graph_executor_linear_flow():
    """Test simple linear graph: A -> B -> C."""
    # Define agents
    agent_a = AgentDefinition(
        name="agent_a",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Start the flow",
    )
    
    agent_b = AgentDefinition(
        name="agent_b",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Process middle step",
    )
    
    agent_c = AgentDefinition(
        name="agent_c",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Finish the flow",
    )
    
    # Define graph: A -> B -> C
    graph = GraphDefinition(
        name="linear_flow",
        start_agent="agent_a",
        edges=[
            GraphEdge(from_agent="agent_a", to_agent="agent_b", condition=None),
            GraphEdge(from_agent="agent_b", to_agent="agent_c", condition=None),
        ],
        termination_agents=["agent_c"],
        termination_condition=None,
        max_hops=10,
    )
    
    # Create registries
    agent_registry = {
        "agent_a": agent_a,
        "agent_b": agent_b,
        "agent_c": agent_c,
    }
    
    llm = MockLLM(responses=[
        "Output from A",
        "Output from B",
        "Output from C",
    ])
    llm_registry = {"gpt4": llm}
    
    # Execute graph
    executor = GraphExecutor(graph, agent_registry, llm_registry)
    result = executor.execute("Start input")
    
    assert result.status == "success"
    assert result.final_response == "Output from C"
    assert len(result.hops) == 3
    assert result.hops[0].agent_name == "agent_a"
    assert result.hops[1].agent_name == "agent_b"
    assert result.hops[2].agent_name == "agent_c"
    assert result.start_agent == "agent_a"
    assert result.end_agent == "agent_c"


def test_graph_executor_conditional_routing():
    """Test graph with conditional routing."""
    agent_router = AgentDefinition(
        name="router",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Route requests",
    )
    
    agent_path_a = AgentDefinition(
        name="path_a",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Handle path A",
    )
    
    agent_path_b = AgentDefinition(
        name="path_b",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Handle path B",
    )
    
    # Graph with conditional edges
    graph = GraphDefinition(
        name="conditional_flow",
        start_agent="router",
        edges=[
            GraphEdge(from_agent="router", to_agent="path_a", condition="contains('option_a')"),
            GraphEdge(from_agent="router", to_agent="path_b", condition="contains('option_b')"),
        ],
        termination_agents=["path_a", "path_b"],
        termination_condition=None,
        max_hops=10,
    )
    
    agent_registry = {
        "router": agent_router,
        "path_a": agent_path_a,
        "path_b": agent_path_b,
    }
    
    # Test path A
    llm_a = MockLLM(responses=[
        "Routing to option_a",
        "Path A result",
    ])
    executor_a = GraphExecutor(graph, agent_registry, {"gpt4": llm_a})
    result_a = executor_a.execute("Choose option A")
    
    assert result_a.status == "success"
    assert result_a.hops[1].agent_name == "path_a"
    
    # Test path B
    llm_b = MockLLM(responses=[
        "Routing to option_b",
        "Path B result",
    ])
    executor_b = GraphExecutor(graph, agent_registry, {"gpt4": llm_b})
    result_b = executor_b.execute("Choose option B")
    
    assert result_b.status == "success"
    assert result_b.hops[1].agent_name == "path_b"


def test_graph_executor_max_hops():
    """Test graph respects max_hops limit."""
    agent_a = AgentDefinition(
        name="agent_a",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Loop forever",
    )
    
    # Graph with self-loop
    graph = GraphDefinition(
        name="loop_flow",
        start_agent="agent_a",
        edges=[
            GraphEdge(from_agent="agent_a", to_agent="agent_a", condition=None),
        ],
        termination_agents=[],
        termination_condition=None,
        max_hops=5,
    )
    
    agent_registry = {"agent_a": agent_a}
    llm = MockLLM(responses=["Loop " + str(i) for i in range(10)])
    llm_registry = {"gpt4": llm}
    
    executor = GraphExecutor(graph, agent_registry, llm_registry)
    result = executor.execute("Start loop")
    
    assert result.status == "max_hops"
    assert len(result.hops) == 5
    assert result.metadata["max_hops"] == 5


def test_graph_executor_termination_condition():
    """Test graph with global termination condition."""
    agent_a = AgentDefinition(
        name="agent_a",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Process until done",
    )
    
    agent_b = AgentDefinition(
        name="agent_b",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Continue processing",
    )
    
    # Graph with termination condition
    graph = GraphDefinition(
        name="conditional_termination",
        start_agent="agent_a",
        edges=[
            GraphEdge(from_agent="agent_a", to_agent="agent_b", condition=None),
            GraphEdge(from_agent="agent_b", to_agent="agent_a", condition=None),
        ],
        termination_agents=[],
        termination_condition="contains('DONE')",
        max_hops=10,
    )
    
    agent_registry = {"agent_a": agent_a, "agent_b": agent_b}
    llm = MockLLM(responses=[
        "Processing step 1",
        "Processing step 2",
        "DONE - all complete",
    ])
    llm_registry = {"gpt4": llm}
    
    executor = GraphExecutor(graph, agent_registry, llm_registry)
    result = executor.execute("Start processing")
    
    assert result.status == "success"
    assert len(result.hops) == 3
    assert "DONE" in result.final_response
    assert result.metadata["termination_reason"] == "condition"


def test_graph_executor_no_outgoing_edges():
    """Test graph handles agent with no outgoing edges."""
    agent_a = AgentDefinition(
        name="agent_a",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Single agent",
    )
    
    # Graph with no edges (single agent)
    graph = GraphDefinition(
        name="single_agent",
        start_agent="agent_a",
        edges=[],
        termination_agents=[],
        termination_condition=None,
        max_hops=10,
    )
    
    agent_registry = {"agent_a": agent_a}
    llm = MockLLM(responses=["Single agent response"])
    llm_registry = {"gpt4": llm}
    
    executor = GraphExecutor(graph, agent_registry, llm_registry)
    result = executor.execute("Input")
    
    assert result.status == "success"
    assert len(result.hops) == 1
    assert result.hops[0].next_agent is None
    assert result.metadata["termination_reason"] == "no_path"


def test_graph_executor_timeout():
    """Test graph respects timeout."""
    import time
    
    agent_a = AgentDefinition(
        name="agent_a",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Process slowly",
    )
    
    # Custom LLM that delays
    class SlowMockLLM(MockLLM):
        def generate_chat(self, messages: list, **kwargs) -> LLMResponse:
            time.sleep(0.2)  # 200ms delay per call
            return super().generate_chat(messages, **kwargs)
    
    graph = GraphDefinition(
        name="timeout_test",
        start_agent="agent_a",
        edges=[
            GraphEdge(from_agent="agent_a", to_agent="agent_a", condition=None),
        ],
        termination_agents=[],
        termination_condition=None,
        max_hops=100,
        timeout_ms=300,  # 300ms timeout
    )
    
    agent_registry = {"agent_a": agent_a}
    llm = SlowMockLLM(responses=["Response " + str(i) for i in range(100)])
    llm_registry = {"gpt4": llm}
    
    executor = GraphExecutor(graph, agent_registry, llm_registry)
    result = executor.execute("Start")
    
    assert result.status == "timeout"
    assert len(result.hops) < 100  # Should timeout before max_hops
    assert result.metadata["timeout_ms"] == 300


def test_graph_executor_with_tools():
    """Test graph with agents using tools."""
    agent_a = AgentDefinition(
        name="agent_a",
        llm_name="gpt4",
        tool_names=["calculator"],
        memory_config=None,
        goal="Calculate something",
    )
    
    agent_b = AgentDefinition(
        name="agent_b",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Summarize result",
    )
    
    graph = GraphDefinition(
        name="tool_flow",
        start_agent="agent_a",
        edges=[
            GraphEdge(from_agent="agent_a", to_agent="agent_b", condition=None),
        ],
        termination_agents=["agent_b"],
        termination_condition=None,
        max_hops=10,
    )
    
    agent_registry = {"agent_a": agent_a, "agent_b": agent_b}
    llm = MockLLM(responses=[
        'TOOL_CALL: calculator(expr="2+2")',
        "The calculation is complete",
        "Summary: result is 4",
    ])
    llm_registry = {"gpt4": llm}
    
    def calculator(expr: str) -> str:
        return str(eval(expr))
    
    tool_registry = {"calculator": calculator}
    
    executor = GraphExecutor(graph, agent_registry, llm_registry, tool_registry)
    result = executor.execute("Calculate 2+2")
    
    assert result.status == "success"
    assert len(result.hops) == 2
    # First hop should have tool calls
    assert len(result.hops[0].agent_result.turns) > 0


def test_graph_executor_state_passing():
    """Test that state (output) passes between agents."""
    agent_a = AgentDefinition(
        name="agent_a",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Generate data",
    )
    
    agent_b = AgentDefinition(
        name="agent_b",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Transform data",
    )
    
    graph = GraphDefinition(
        name="state_passing",
        start_agent="agent_a",
        edges=[
            GraphEdge(from_agent="agent_a", to_agent="agent_b", condition=None),
        ],
        termination_agents=["agent_b"],
        termination_condition=None,
        max_hops=10,
    )
    
    agent_registry = {"agent_a": agent_a, "agent_b": agent_b}
    
    # Track what input agent_b receives
    received_inputs = []
    
    class TrackingMockLLM(MockLLM):
        def generate_chat(self, messages: list, **kwargs) -> LLMResponse:
            # Capture the last user message
            for msg in reversed(messages):
                if msg.role == "user":
                    received_inputs.append(msg.content)
                    break
            return super().generate_chat(messages, **kwargs)
    
    llm = TrackingMockLLM(responses=[
        "Output from agent A",
        "Output from agent B",
    ])
    llm_registry = {"gpt4": llm}
    
    executor = GraphExecutor(graph, agent_registry, llm_registry)
    result = executor.execute("Initial input")
    
    assert result.status == "success"
    # Agent B should receive agent A's output as input
    assert "Output from agent A" in received_inputs


def test_graph_executor_fallback_routing():
    """Test fallback routing when no conditions match."""
    agent_router = AgentDefinition(
        name="router",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Route requests",
    )
    
    agent_default = AgentDefinition(
        name="default",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Handle default case",
    )
    
    # Graph with conditions that won't match
    graph = GraphDefinition(
        name="fallback_test",
        start_agent="router",
        edges=[
            GraphEdge(from_agent="router", to_agent="default", condition="contains('never_match')"),
        ],
        termination_agents=["default"],
        termination_condition=None,
        max_hops=10,
    )
    
    agent_registry = {"router": agent_router, "default": agent_default}
    llm = MockLLM(responses=[
        "Something else",
        "Default handler response",
    ])
    llm_registry = {"gpt4": llm}
    
    executor = GraphExecutor(graph, agent_registry, llm_registry)
    result = executor.execute("Input")
    
    assert result.status == "success"
    assert result.hops[1].agent_name == "default"
    assert result.hops[0].routing_decision == "fallback"


def test_graph_executor_reset():
    """Test graph executor reset clears all agent memories."""
    agent_a = AgentDefinition(
        name="agent_a",
        llm_name="gpt4",
        tool_names=[],
        memory_config=MemoryConfig(policy="full_history"),
        goal="Remember everything",
    )
    
    graph = GraphDefinition(
        name="reset_test",
        start_agent="agent_a",
        edges=[],
        termination_agents=["agent_a"],
        termination_condition=None,
        max_hops=10,
    )
    
    agent_registry = {"agent_a": agent_a}
    llm = MockLLM(responses=["Response 1", "Response 2"])
    llm_registry = {"gpt4": llm}
    
    executor = GraphExecutor(graph, agent_registry, llm_registry)
    
    # Execute once
    executor.execute("Input 1")
    assert len(executor.agent_runtimes["agent_a"].memory.messages) > 0
    
    # Reset
    executor.reset()
    assert len(executor.agent_runtimes["agent_a"].memory.messages) == 0
    
    # Execute again
    executor.execute("Input 2")
    # Should only have messages from second execution
    assert len(executor.agent_runtimes["agent_a"].memory.messages) == 2  # user + assistant
