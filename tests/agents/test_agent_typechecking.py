"""Test type checking/validation for agent and graph definitions."""

import pytest
from namel3ss.ast import (
    AgentDefinition,
    App,
    GraphDefinition,
    GraphEdge,
    MemoryConfig,
    Module,
)
from namel3ss.ast.ai import LLMDefinition, ToolDefinition
from namel3ss.errors import N3TypeError
from namel3ss.types.checker import AppTypeChecker


def test_agent_with_valid_llm_and_tools():
    """Test that agent with valid LLM and tool references passes validation."""
    llm = LLMDefinition(
        name="gpt4",
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
    )
    
    tool = ToolDefinition(
        name="search",
        type="http",
        endpoint="https://api.example.com/search",
        method="POST",
        timeout=30,
    )
    
    agent = AgentDefinition(
        name="researcher",
        llm_name="gpt4",
        tool_names=["search"],
        memory_config=MemoryConfig(policy="conversation_window", window_size=10),
        goal="Research topics and provide summaries",
    )
    
    app = App(
        name="test_app",
        llms=[llm],
        tools=[tool],
        agents=[agent],
        graphs=[],
    )
    
    module = Module(name="test", body=[app])
    checker = AppTypeChecker()
    
    # Should not raise
    checker.check_module(module)


def test_agent_with_unknown_llm():
    """Test that agent referencing unknown LLM raises error."""
    tool = ToolDefinition(
        name="search",
        type="http",
        endpoint="https://api.example.com/search",
        method="POST",
        timeout=30,
    )
    
    agent = AgentDefinition(
        name="researcher",
        llm_name="unknown_llm",  # Does not exist
        tool_names=["search"],
        memory_config=MemoryConfig(policy="conversation_window"),
        goal="Research topics",
    )
    
    app = App(
        name="test_app",
        llms=[],  # Empty - LLM doesn't exist
        tools=[tool],
        agents=[agent],
        graphs=[],
    )
    
    module = Module(name="test", body=[app])
    checker = AppTypeChecker()
    
    with pytest.raises(N3TypeError) as exc_info:
        checker.check_module(module)
    
    assert "unknown LLM 'unknown_llm'" in str(exc_info.value)
    assert "researcher" in str(exc_info.value)


def test_agent_with_unknown_tool():
    """Test that agent referencing unknown tool raises error."""
    llm = LLMDefinition(
        name="gpt4",
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
    )
    
    agent = AgentDefinition(
        name="researcher",
        llm_name="gpt4",
        tool_names=["unknown_tool"],  # Does not exist
        memory_config=MemoryConfig(policy="conversation_window"),
        goal="Research topics",
    )
    
    app = App(
        name="test_app",
        llms=[llm],
        tools=[],  # Empty - tool doesn't exist
        agents=[agent],
        graphs=[],
    )
    
    module = Module(name="test", body=[app])
    checker = AppTypeChecker()
    
    with pytest.raises(N3TypeError) as exc_info:
        checker.check_module(module)
    
    assert "unknown tool 'unknown_tool'" in str(exc_info.value)
    assert "researcher" in str(exc_info.value)


def test_agent_with_invalid_memory_policy():
    """Test that agent with invalid memory policy raises error."""
    llm = LLMDefinition(
        name="gpt4",
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
    )
    
    agent = AgentDefinition(
        name="researcher",
        llm_name="gpt4",
        tool_names=[],
        memory_config=MemoryConfig(policy="invalid_policy"),  # Invalid
        goal="Research topics",
    )
    
    app = App(
        name="test_app",
        llms=[llm],
        tools=[],
        agents=[agent],
        graphs=[],
    )
    
    module = Module(name="test", body=[app])
    checker = AppTypeChecker()
    
    with pytest.raises(N3TypeError) as exc_info:
        checker.check_module(module)
    
    assert "invalid memory policy 'invalid_policy'" in str(exc_info.value)
    assert "researcher" in str(exc_info.value)


def test_agent_with_invalid_temperature():
    """Test that agent with out-of-range temperature raises error."""
    llm = LLMDefinition(
        name="gpt4",
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
    )
    
    agent = AgentDefinition(
        name="researcher",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Research topics",
        temperature=3.0,  # Invalid - must be 0-2
    )
    
    app = App(
        name="test_app",
        llms=[llm],
        tools=[],
        agents=[agent],
        graphs=[],
    )
    
    module = Module(name="test", body=[app])
    checker = AppTypeChecker()
    
    with pytest.raises(N3TypeError) as exc_info:
        checker.check_module(module)
    
    assert "temperature must be between 0 and 2" in str(exc_info.value)
    assert "researcher" in str(exc_info.value)


def test_graph_with_valid_agents():
    """Test that graph with valid agent references passes validation."""
    llm = LLMDefinition(
        name="gpt4",
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
    )
    
    agent1 = AgentDefinition(
        name="researcher",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Research",
    )
    
    agent2 = AgentDefinition(
        name="writer",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Write",
    )
    
    graph = GraphDefinition(
        name="research_flow",
        start_agent="researcher",
        edges=[
            GraphEdge(from_agent="researcher", to_agent="writer", condition=None),
        ],
        termination_agents=["writer"],
        termination_condition=None,
        max_hops=10,
    )
    
    app = App(
        name="test_app",
        llms=[llm],
        tools=[],
        agents=[agent1, agent2],
        graphs=[graph],
    )
    
    module = Module(name="test", body=[app])
    checker = AppTypeChecker()
    
    # Should not raise
    checker.check_module(module)


def test_graph_with_unknown_start_agent():
    """Test that graph with unknown start agent raises error."""
    llm = LLMDefinition(
        name="gpt4",
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
    )
    
    agent = AgentDefinition(
        name="researcher",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Research",
    )
    
    graph = GraphDefinition(
        name="research_flow",
        start_agent="unknown_agent",  # Does not exist
        edges=[],
        termination_agents=[],
        termination_condition=None,
        max_hops=10,
    )
    
    app = App(
        name="test_app",
        llms=[llm],
        tools=[],
        agents=[agent],
        graphs=[graph],
    )
    
    module = Module(name="test", body=[app])
    checker = AppTypeChecker()
    
    with pytest.raises(N3TypeError) as exc_info:
        checker.check_module(module)
    
    assert "unknown start agent 'unknown_agent'" in str(exc_info.value)
    assert "research_flow" in str(exc_info.value)


def test_graph_with_unknown_edge_agent():
    """Test that graph with unknown agent in edge raises error."""
    llm = LLMDefinition(
        name="gpt4",
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
    )
    
    agent = AgentDefinition(
        name="researcher",
        llm_name="gpt4",
        tool_names=[],
        memory_config=None,
        goal="Research",
    )
    
    graph = GraphDefinition(
        name="research_flow",
        start_agent="researcher",
        edges=[
            GraphEdge(from_agent="researcher", to_agent="unknown_agent", condition=None),
        ],
        termination_agents=[],
        termination_condition=None,
        max_hops=10,
    )
    
    app = App(
        name="test_app",
        llms=[llm],
        tools=[],
        agents=[agent],
        graphs=[graph],
    )
    
    module = Module(name="test", body=[app])
    checker = AppTypeChecker()
    
    with pytest.raises(N3TypeError) as exc_info:
        checker.check_module(module)
    
    assert "unknown to_agent 'unknown_agent'" in str(exc_info.value)
    assert "research_flow" in str(exc_info.value)


def test_graph_with_unreachable_agents():
    """Test that graph with unreachable agents raises error."""
    llm = LLMDefinition(
        name="gpt4",
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
    )
    
    agent1 = AgentDefinition(name="a1", llm_name="gpt4", tool_names=[], memory_config=None, goal="Task 1")
    agent2 = AgentDefinition(name="a2", llm_name="gpt4", tool_names=[], memory_config=None, goal="Task 2")
    agent3 = AgentDefinition(name="a3", llm_name="gpt4", tool_names=[], memory_config=None, goal="Task 3")
    
    # a1 -> a2, but a3 is unreachable
    graph = GraphDefinition(
        name="flow",
        start_agent="a1",
        edges=[
            GraphEdge(from_agent="a1", to_agent="a2", condition=None),
            GraphEdge(from_agent="a3", to_agent="a1", condition=None),  # a3 is not reachable from a1
        ],
        termination_agents=["a2"],
        termination_condition=None,
        max_hops=10,
    )
    
    app = App(
        name="test_app",
        llms=[llm],
        tools=[],
        agents=[agent1, agent2, agent3],
        graphs=[graph],
    )
    
    module = Module(name="test", body=[app])
    checker = AppTypeChecker()
    
    with pytest.raises(N3TypeError) as exc_info:
        checker.check_module(module)
    
    assert "unreachable agents" in str(exc_info.value)
    assert "a3" in str(exc_info.value)
    assert "flow" in str(exc_info.value)


def test_graph_with_complex_valid_structure():
    """Test that graph with complex but valid structure passes validation."""
    llm = LLMDefinition(
        name="gpt4",
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
    )
    
    agents = [
        AgentDefinition(name=f"agent{i}", llm_name="gpt4", tool_names=[], memory_config=None, goal=f"Task {i}")
        for i in range(1, 6)
    ]
    
    # Complex graph: a1 -> a2 -> a3 -> a4 -> a5 (linear)
    #                 a1 -> a3 (skip)
    #                 a2 -> a5 (skip)
    graph = GraphDefinition(
        name="complex_flow",
        start_agent="agent1",
        edges=[
            GraphEdge(from_agent="agent1", to_agent="agent2", condition=None),
            GraphEdge(from_agent="agent1", to_agent="agent3", condition=None),
            GraphEdge(from_agent="agent2", to_agent="agent3", condition=None),
            GraphEdge(from_agent="agent2", to_agent="agent5", condition=None),
            GraphEdge(from_agent="agent3", to_agent="agent4", condition=None),
            GraphEdge(from_agent="agent4", to_agent="agent5", condition=None),
        ],
        termination_agents=["agent5"],
        termination_condition=None,
        max_hops=20,
    )
    
    app = App(
        name="test_app",
        llms=[llm],
        tools=[],
        agents=agents,
        graphs=[graph],
    )
    
    module = Module(name="test", body=[app])
    checker = AppTypeChecker()
    
    # Should not raise - all agents are reachable
    checker.check_module(module)
