"""Tests for agent and graph AST parsing."""

import pytest
from namel3ss.parser import Parser
from namel3ss.ast import AgentDefinition, GraphDefinition, App


def test_agent_parsing_basic():
    """Test parsing a basic agent definition."""
    source = """
app "Agent Test"

llm gpt4:
    provider: openai
    model: gpt-4

tool search:
    type: http
    endpoint: "https://api.example.com/search"

agent researcher {
    llm: gpt4
    tools: [search]
    memory: "conversation"
    goal: "Gather accurate information"
}
"""
    
    parser = Parser(source, path="test.n3")
    module = parser.parse()
    
    assert len(module.body) == 1
    app = module.body[0]
    assert isinstance(app, App)
    
    # Check agent was parsed
    assert len(app.agents) == 1
    agent = app.agents[0]
    assert isinstance(agent, AgentDefinition)
    assert agent.name == "researcher"
    assert agent.llm_name == "gpt4"
    assert agent.tool_names == ["search"]
    assert agent.memory_config == "conversation"
    assert agent.goal == "Gather accurate information"


def test_agent_parsing_with_optional_fields():
    """Test parsing an agent with optional fields."""
    source = """
app "Agent Test"

llm gpt4:
    provider: openai
    model: gpt-4

agent researcher {
    llm: gpt4
    tools: []
    goal: "Research topics"
    system_prompt: "You are a researcher"
    max_turns: 10
    temperature: 0.8
}
"""
    
    parser = Parser(source, path="test.n3")
    module = parser.parse()
    
    app = module.body[0]
    agent = app.agents[0]
    
    assert agent.name == "researcher"
    assert agent.system_prompt == "You are a researcher"
    assert agent.max_turns == 10
    assert agent.temperature == 0.8


def test_graph_parsing_basic():
    """Test parsing a basic graph definition."""
    source = """
app "Graph Test"

llm gpt4:
    provider: openai
    model: gpt-4

agent researcher {
    llm: gpt4
    tools: []
    goal: "Research"
}

agent decider {
    llm: gpt4
    tools: []
    goal: "Decide"
}

graph support_flow {
    start: researcher
    edges: [
        { from: researcher, to: decider, when: "done_research" }
    ]
    termination: decider
    max_hops: 32
}
"""
    
    parser = Parser(source, path="test.n3")
    module = parser.parse()
    
    app = module.body[0]
    
    # Check agents were parsed
    assert len(app.agents) == 2
    assert app.agents[0].name == "researcher"
    assert app.agents[1].name == "decider"
    
    # Check graph was parsed
    assert len(app.graphs) == 1
    graph = app.graphs[0]
    assert isinstance(graph, GraphDefinition)
    assert graph.name == "support_flow"
    assert graph.start_agent == "researcher"
    assert len(graph.edges) == 1
    assert graph.edges[0].from_agent == "researcher"
    assert graph.edges[0].to_agent == "decider"
    assert graph.edges[0].condition == "done_research"
    assert graph.termination_agents == ["decider"]
    assert graph.max_hops == 32


def test_graph_with_multiple_edges():
    """Test parsing a graph with multiple edges."""
    source = """
app "Multi-Edge Graph"

llm gpt4:
    provider: openai
    model: gpt-4

agent a1 {
    llm: gpt4
    tools: []
    goal: "A1"
}

agent a2 {
    llm: gpt4
    tools: []
    goal: "A2"
}

agent a3 {
    llm: gpt4
    tools: []
    goal: "A3"
}

graph flow {
    start: a1
    edges: [
        { from: a1, to: a2, when: "condition1" },
        { from: a2, to: a3, when: "condition2" },
        { from: a2, to: a1, when: "retry" }
    ]
    termination: a3
}
"""
    
    parser = Parser(source, path="test.n3")
    module = parser.parse()
    
    app = module.body[0]
    graph = app.graphs[0]
    
    assert len(graph.edges) == 3
    assert graph.edges[0].from_agent == "a1"
    assert graph.edges[0].to_agent == "a2"
    assert graph.edges[1].from_agent == "a2"
    assert graph.edges[1].to_agent == "a3"
    assert graph.edges[2].from_agent == "a2"
    assert graph.edges[2].to_agent == "a1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
