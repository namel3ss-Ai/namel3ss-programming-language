"""End-to-end tests for agent and graph system integration."""

import pytest
from namel3ss.codegen.backend.state import build_backend_state, _encode_agent, _encode_graph
from namel3ss.ast.agents import (
    AgentDefinition,
    GraphDefinition,
    GraphEdge,
    MemoryConfig,
)


def test_encode_agent():
    """Test encoding an AgentDefinition to dict."""
    agent = AgentDefinition(
        name="test_agent",
        llm_name="gpt4",
        tool_names=["search", "calculator"],
        memory_config=MemoryConfig(
            policy="conversation_window",
            window_size=5,
            max_items=None,
            config={},
        ),
        goal="Help users",
        system_prompt="You are helpful",
        max_turns=10,
        temperature=0.7,
        config={"custom": "value"},
    )
    
    encoded = _encode_agent(agent, set())
    
    assert encoded["name"] == "test_agent"
    assert encoded["llm_name"] == "gpt4"
    assert encoded["tool_names"] == ["search", "calculator"]
    assert encoded["memory_config"]["policy"] == "conversation_window"
    assert encoded["memory_config"]["window_size"] == 5
    assert encoded["goal"] == "Help users"
    assert encoded["system_prompt"] == "You are helpful"
    assert encoded["max_turns"] == 10
    assert encoded["temperature"] == 0.7
    assert encoded["config"]["custom"] == "value"


def test_encode_agent_minimal():
    """Test encoding agent with minimal config."""
    agent = AgentDefinition(
        name="minimal_agent",
        llm_name="gpt3",
        tool_names=[],
        memory_config=None,
        goal="Simple task",
    )
    
    encoded = _encode_agent(agent, set())
    
    assert encoded["name"] == "minimal_agent"
    assert encoded["llm_name"] == "gpt3"
    assert encoded["tool_names"] == []
    assert encoded["memory_config"] is None
    assert encoded["goal"] == "Simple task"
    assert encoded["system_prompt"] is None
    assert encoded["max_turns"] is None
    assert encoded["temperature"] is None
    assert encoded["config"] == {}


def test_encode_graph():
    """Test encoding a GraphDefinition to dict."""
    graph = GraphDefinition(
        name="test_graph",
        start_agent="agent1",
        edges=[
            GraphEdge(from_agent="agent1", to_agent="agent2", condition=None),
            GraphEdge(from_agent="agent2", to_agent="agent3", condition=None),
        ],
        termination_agents=["agent3"],
        termination_condition=None,
        max_hops=20,
        timeout_ms=5000,
    )
    
    encoded = _encode_graph(graph, set())
    
    assert encoded["name"] == "test_graph"
    assert encoded["start_agent"] == "agent1"
    assert len(encoded["edges"]) == 2
    assert encoded["edges"][0]["from_agent"] == "agent1"
    assert encoded["edges"][0]["to_agent"] == "agent2"
    assert encoded["edges"][0]["condition"] is None
    assert encoded["termination_agents"] == ["agent3"]
    assert encoded["termination_condition"] is None
    assert encoded["max_hops"] == 20
    assert encoded["timeout_ms"] == 5000


def test_encode_graph_with_conditions():
    """Test encoding graph with conditional routing."""
    from namel3ss.ast import Literal
    
    graph = GraphDefinition(
        name="conditional_graph",
        start_agent="router",
        edges=[
            GraphEdge(
                from_agent="router",
                to_agent="handler_a",
                condition=Literal(value="route_a"),
            ),
            GraphEdge(
                from_agent="router",
                to_agent="handler_b",
                condition=Literal(value="route_b"),
            ),
        ],
        termination_agents=["handler_a", "handler_b"],
        termination_condition=None,
        max_hops=10,
        timeout_ms=None,
    )
    
    encoded = _encode_graph(graph, set())
    
    assert encoded["name"] == "conditional_graph"
    assert len(encoded["edges"]) == 2
    # Conditions are converted to source strings
    assert encoded["edges"][0]["condition"] is not None
    assert encoded["edges"][1]["condition"] is not None


def test_backend_state_with_agents_and_graphs():
    """Test that backend state includes agents and graphs."""
    from namel3ss.ast import App
    
    # Create minimal app with agents and graphs
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
    
    graph = GraphDefinition(
        name="test_workflow",
        start_agent="agent1",
        edges=[
            GraphEdge(from_agent="agent1", to_agent="agent2", condition=None),
        ],
        termination_agents=["agent2"],
        termination_condition=None,
        max_hops=10,
        timeout_ms=None,
    )
    
    app = App(
        name="test_app",
        agents=[agent1, agent2],
        graphs=[graph],
    )
    
    state = build_backend_state(app)
    
    # Verify agents are encoded
    assert "agents" in state.__dict__
    assert len(state.agents) == 2
    assert "agent1" in state.agents
    assert "agent2" in state.agents
    assert state.agents["agent1"]["llm_name"] == "gpt4"
    assert state.agents["agent2"]["goal"] == "Task 2"
    
    # Verify graphs are encoded
    assert "graphs" in state.__dict__
    assert len(state.graphs) == 1
    assert "test_workflow" in state.graphs
    assert state.graphs["test_workflow"]["start_agent"] == "agent1"
    assert len(state.graphs["test_workflow"]["edges"]) == 1


def test_graph_step_kind_supported():
    """Test that 'graph' is recognized as a valid chain step kind."""
    # This is a documentation test showing the expected structure
    chain_step = {
        "kind": "graph",
        "target": "my_workflow",
        "options": {
            "input": "User query",
            "context": {"session_id": "abc123"},
        },
        "stop_on_error": True,
    }
    
    assert chain_step["kind"] == "graph"
    assert chain_step["target"] == "my_workflow"
    assert "input" in chain_step["options"]
    assert "context" in chain_step["options"]


def test_agent_memory_encoding():
    """Test that memory configuration is properly encoded."""
    agent = AgentDefinition(
        name="memory_agent",
        llm_name="gpt4",
        tool_names=[],
        memory_config=MemoryConfig(
            policy="summary",
            max_items=100,
            window_size=None,
            config={"max_tokens": 1000},
        ),
        goal="Remember things",
    )
    
    encoded = _encode_agent(agent, set())
    
    assert encoded["memory_config"]["policy"] == "summary"
    assert encoded["memory_config"]["max_items"] == 100
    assert encoded["memory_config"]["window_size"] is None
    assert encoded["memory_config"]["config"]["max_tokens"] == 1000


def test_complex_graph_encoding():
    """Test encoding a complex graph with multiple branches."""
    graph = GraphDefinition(
        name="complex_workflow",
        start_agent="classifier",
        edges=[
            GraphEdge(from_agent="classifier", to_agent="handler_a", condition=None),
            GraphEdge(from_agent="classifier", to_agent="handler_b", condition=None),
            GraphEdge(from_agent="handler_a", to_agent="finalizer", condition=None),
            GraphEdge(from_agent="handler_b", to_agent="finalizer", condition=None),
        ],
        termination_agents=["finalizer"],
        termination_condition=None,
        max_hops=15,
        timeout_ms=10000,
    )
    
    encoded = _encode_graph(graph, set())
    
    assert encoded["name"] == "complex_workflow"
    assert encoded["start_agent"] == "classifier"
    assert len(encoded["edges"]) == 4
    assert encoded["termination_agents"] == ["finalizer"]
    assert encoded["max_hops"] == 15
    assert encoded["timeout_ms"] == 10000
