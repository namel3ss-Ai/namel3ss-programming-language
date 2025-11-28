"""
Tests for N3 AST to Graph JSON converter.
"""

import pytest
from n3_server.converter import N3ASTConverter
from namel3ss.ast.agents import AgentDefinition, GraphDefinition, GraphEdge as N3GraphEdge
from namel3ss.ast.ai_workflows import Chain, ChainStep
from namel3ss.ast.ai_prompts import Prompt, PromptArgument


def test_agent_to_node():
    """Test converting AgentDefinition to graph node."""
    converter = N3ASTConverter()
    
    agent = AgentDefinition(
        name="researcher",
        llm_name="gpt-4",
        tool_names=["search", "analyze"],
        goal="Research and analyze information",
        system_prompt="You are a research assistant",
        max_turns=10,
        temperature=0.7,
    )
    
    node = converter.agent_to_node(agent)
    
    assert node.type == "agent"
    assert node.label == "researcher"
    assert node.data["llm"] == "gpt-4"
    assert node.data["tools"] == ["search", "analyze"]
    assert node.data["goal"] == "Research and analyze information"
    assert node.data["temperature"] == 0.7
    assert node.position is not None


def test_prompt_to_node():
    """Test converting Prompt to graph node."""
    converter = N3ASTConverter()
    
    prompt = Prompt(
        name="summarize",
        template="Summarize the following text: {text}",
        model="gpt-4",
        parameters={"temperature": 0.5},
        args=[PromptArgument(name="text", arg_type="string")],
    )
    
    node = converter.prompt_to_node(prompt)
    
    assert node.type == "prompt"
    assert node.label == "summarize"
    assert node.data["name"] == "summarize"
    assert node.data["model"] == "gpt-4"
    assert node.data["parameters"]["temperature"] == 0.5


def test_chain_to_graph():
    """Test converting Chain to graph nodes and edges."""
    converter = N3ASTConverter()
    
    chain = Chain(
        name="support_flow",
        steps=[
            ChainStep(kind="prompt", target="classify", options={"text": "$input"}),
            ChainStep(kind="tool", target="search_kb", options={"query": "$classification"}),
            ChainStep(kind="prompt", target="generate_response", options={"context": "$kb_results"}),
        ],
        input_key="input",
    )
    
    nodes, edges = converter.chain_to_graph(chain)
    
    # Should have start + 3 steps + end = 5 nodes
    assert len(nodes) == 5
    
    # Should have 4 edges connecting them
    assert len(edges) == 4
    
    # Check start and end nodes
    start_nodes = [n for n in nodes if n.type == "start"]
    end_nodes = [n for n in nodes if n.type == "end"]
    assert len(start_nodes) == 1
    assert len(end_nodes) == 1
    
    # Check step nodes
    prompt_nodes = [n for n in nodes if n.type == "prompt"]
    tool_nodes = [n for n in nodes if n.type == "pythonHook"]
    assert len(prompt_nodes) == 2
    assert len(tool_nodes) == 1


def test_agent_graph_to_graph():
    """Test converting multi-agent GraphDefinition to graph."""
    converter = N3ASTConverter()
    
    graph = GraphDefinition(
        name="support_graph",
        start_agent="researcher",
        edges=[
            N3GraphEdge(from_agent="researcher", to_agent="analyzer", condition="done_research"),
            N3GraphEdge(from_agent="analyzer", to_agent="responder", condition="analysis_complete"),
        ],
        termination_agents=["responder"],
    )
    
    nodes, edges = converter.agent_graph_to_graph(graph)
    
    # Should have start + 3 agents + end = 5 nodes
    assert len(nodes) == 5
    
    # Should have start->researcher + 2 agent edges + responder->end = 4 edges
    assert len(edges) == 4
    
    # Check agent references
    agent_nodes = [n for n in nodes if n.type == "agent"]
    assert len(agent_nodes) == 3
    assert set(n.data["name"] for n in agent_nodes) == {"researcher", "analyzer", "responder"}


def test_ast_to_graph_json():
    """Test full AST to graph JSON conversion."""
    converter = N3ASTConverter()
    
    agent = AgentDefinition(
        name="assistant",
        llm_name="gpt-4",
        tool_names=["calculator"],
        goal="Help with calculations",
    )
    
    chain = Chain(
        name="calc_flow",
        steps=[
            ChainStep(kind="tool", target="calculator", options={"operation": "add"}),
        ],
        input_key="input",
    )
    
    graph_json = converter.ast_to_graph_json(
        project_id="test-123",
        name="Test Project",
        chains=[chain],
        agents=[agent],
    )
    
    assert graph_json.projectId == "test-123"
    assert graph_json.name == "Test Project"
    assert len(graph_json.chains) == 1
    assert len(graph_json.agents) == 1
    assert len(graph_json.nodes) > 0
    assert graph_json.activeRootId != ""


def test_graph_json_to_chain():
    """Test converting graph JSON back to Chain AST."""
    converter = N3ASTConverter()
    
    nodes = [
        {"id": "start-1", "type": "start", "label": "START", "data": {}},
        {"id": "step-1", "type": "prompt", "label": "classify", "data": {"target": "classify", "options": {"text": "$input"}}},
        {"id": "step-2", "type": "pythonHook", "label": "search", "data": {"target": "search_kb", "options": {"query": "$query"}}},
        {"id": "end-1", "type": "end", "label": "END", "data": {}},
    ]
    
    edges = [
        {"id": "e1", "source": "start-1", "target": "step-1"},
        {"id": "e2", "source": "step-1", "target": "step-2"},
        {"id": "e3", "source": "step-2", "target": "end-1"},
    ]
    
    chain = converter.graph_json_to_chain(nodes, edges, "test_chain")
    
    assert chain.name == "test_chain"
    assert len(chain.steps) == 2
    assert chain.steps[0].kind == "prompt"
    assert chain.steps[0].target == "classify"
    assert chain.steps[1].kind == "tool"
    assert chain.steps[1].target == "search_kb"


def test_graph_json_to_agent():
    """Test converting graph node back to AgentDefinition."""
    converter = N3ASTConverter()
    
    node = {
        "id": "agent-1",
        "type": "agent",
        "label": "researcher",
        "data": {
            "name": "researcher",
            "llm": "gpt-4",
            "tools": ["search", "analyze"],
            "memory": "conversation",
            "goal": "Research information",
            "systemPrompt": "You are helpful",
            "maxTurns": 10,
            "temperature": 0.7,
        }
    }
    
    agent = converter.graph_json_to_agent(node)
    
    assert agent.name == "researcher"
    assert agent.llm_name == "gpt-4"
    assert agent.tool_names == ["search", "analyze"]
    assert agent.goal == "Research information"
    assert agent.max_turns == 10
    assert agent.temperature == 0.7


def test_roundtrip_conversion():
    """Test that AST -> Graph -> AST preserves structure."""
    converter = N3ASTConverter()
    
    # Original chain
    original_chain = Chain(
        name="roundtrip_test",
        steps=[
            ChainStep(kind="prompt", target="step1", options={"a": 1}),
            ChainStep(kind="tool", target="step2", options={"b": 2}),
        ],
        input_key="input",
    )
    
    # Convert to graph
    nodes, edges = converter.chain_to_graph(original_chain)
    
    # Convert back to chain
    reconstructed_chain = converter.graph_json_to_chain(
        [converter._node_to_dict(n) for n in nodes],
        [converter._edge_to_dict(e) for e in edges],
        "roundtrip_test"
    )
    
    # Verify structure preserved
    assert reconstructed_chain.name == original_chain.name
    assert len(reconstructed_chain.steps) == len(original_chain.steps)
    
    for i, (orig, recon) in enumerate(zip(original_chain.steps, reconstructed_chain.steps)):
        assert recon.kind == orig.kind, f"Step {i} kind mismatch"
        assert recon.target == orig.target, f"Step {i} target mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
