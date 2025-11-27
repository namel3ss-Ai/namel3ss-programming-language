"""
Test Phase 1: Parser Unification - Tool/Agent Syntax Compatibility
Tests that both quoted and unquoted names work for tools, agents, LLMs, memory, and prompts.
"""
import pytest
from namel3ss.parser import Parser


@pytest.fixture
def phase1_source():
    return """
app "Phase 1 Test"

# Unquoted tool
tool generate_quiz {
    description: "Generate questions"
    parameters: {
        topic: { type: "string", required: true }
    }
}

# Quoted tool
tool "grade_answer" {
    description: "Grade answers"
}

# Unquoted agent
agent quiz_maker {
    llm: "gpt-4"
    tools: ["generate_quiz"]
    temperature: 0.7
}

# Quoted agent
agent "grading_agent" {
    llm: "gpt-4"
    tools: ["grade_answer"]
}

# Unquoted LLM
llm my_llm {
    provider: "openai"
    model: "gpt-4"
}

# Quoted LLM
llm "another_llm" {
    provider: "anthropic"
}

# Unquoted memory
memory conversation_history {
    type: "conversation"
}

# Quoted memory
memory "quiz_context" {
    type: "semantic"
}

# Unquoted prompt
prompt quiz_generation {
    model: "gpt-4"
    template: "Generate quiz"
}

# Quoted prompt
prompt "grading_rubric" {
    model: "gpt-4"
    template: "Grade answer"
}
"""


def test_phase1_unquoted_tools(phase1_source):
    """Test that unquoted tool names work"""
    parser = Parser(phase1_source)
    module = parser.parse()
    app = module.body[0]
    
    tool_names = [t.name for t in app.tools]
    assert "generate_quiz" in tool_names
    assert len(app.tools) == 2


def test_phase1_quoted_tools(phase1_source):
    """Test that quoted tool names work"""
    parser = Parser(phase1_source)
    module = parser.parse()
    app = module.body[0]
    
    tool_names = [t.name for t in app.tools]
    assert "grade_answer" in tool_names


def test_phase1_unquoted_agents(phase1_source):
    """Test that unquoted agent names work"""
    parser = Parser(phase1_source)
    module = parser.parse()
    app = module.body[0]
    
    agent_names = [a.name for a in app.agents]
    assert "quiz_maker" in agent_names
    assert len(app.agents) == 2


def test_phase1_quoted_agents(phase1_source):
    """Test that quoted agent names work"""
    parser = Parser(phase1_source)
    module = parser.parse()
    app = module.body[0]
    
    agent_names = [a.name for a in app.agents]
    assert "grading_agent" in agent_names


def test_phase1_unquoted_llms(phase1_source):
    """Test that unquoted LLM names work"""
    parser = Parser(phase1_source)
    module = parser.parse()
    app = module.body[0]
    
    llm_names = [l.name for l in app.llms]
    assert "my_llm" in llm_names
    assert len(app.llms) == 2


def test_phase1_quoted_llms(phase1_source):
    """Test that quoted LLM names work"""
    parser = Parser(phase1_source)
    module = parser.parse()
    app = module.body[0]
    
    llm_names = [l.name for l in app.llms]
    assert "another_llm" in llm_names


def test_phase1_unquoted_memory(phase1_source):
    """Test that unquoted memory names work"""
    parser = Parser(phase1_source)
    module = parser.parse()
    app = module.body[0]
    
    memory_names = [m.name for m in app.memories]
    assert "conversation_history" in memory_names
    assert len(app.memories) == 2


def test_phase1_quoted_memory(phase1_source):
    """Test that quoted memory names work"""
    parser = Parser(phase1_source)
    module = parser.parse()
    app = module.body[0]
    
    memory_names = [m.name for m in app.memories]
    assert "quiz_context" in memory_names


def test_phase1_unquoted_prompts(phase1_source):
    """Test that unquoted prompt names work"""
    parser = Parser(phase1_source)
    module = parser.parse()
    app = module.body[0]
    
    prompt_names = [p.name for p in app.prompts]
    assert "quiz_generation" in prompt_names
    assert len(app.prompts) == 2


def test_phase1_quoted_prompts(phase1_source):
    """Test that quoted prompt names work"""
    parser = Parser(phase1_source)
    module = parser.parse()
    app = module.body[0]
    
    prompt_names = [p.name for p in app.prompts]
    assert "grading_rubric" in prompt_names


def test_phase1_all_counts(phase1_source):
    """Test that all declaration types have correct counts"""
    parser = Parser(phase1_source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.tools) == 2, "Should have 2 tools (1 quoted, 1 unquoted)"
    assert len(app.agents) == 2, "Should have 2 agents (1 quoted, 1 unquoted)"
    assert len(app.llms) == 2, "Should have 2 LLMs (1 quoted, 1 unquoted)"
    assert len(app.memories) == 2, "Should have 2 memories (1 quoted, 1 unquoted)"
    assert len(app.prompts) == 2, "Should have 2 prompts (1 quoted, 1 unquoted)"
