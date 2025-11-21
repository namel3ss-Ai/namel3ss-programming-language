"""Tests for config filtering and aliasing system.

This test suite verifies the production-grade configuration filtering
system that safely transforms DSL config dicts into AST constructor arguments.

Test Categories:
1. Dataclass introspection
2. DSL → AST field aliasing  
3. Unknown field routing to config/metadata sinks
4. Preservation of dataclass defaults
5. Error handling for invalid configurations
"""

import pytest
from namel3ss.lang.parser import N3Parser
from namel3ss.lang.parser.errors import N3SyntaxError
from namel3ss.lang.parser.config_filter import (
    filter_config_for_dataclass,
    build_dataclass_with_config,
    AGENT_ALIASES,
    LLM_ALIASES,
    _get_dataclass_fields,
    _has_config_sink,
)
from namel3ss.ast import AgentDefinition, LLMDefinition, GraphDefinition


class TestDataclassIntrospection:
    """Test dataclass field extraction and metadata checking."""
    
    def test_get_dataclass_fields(self):
        """Test extraction of fields from a dataclass."""
        fields = _get_dataclass_fields(AgentDefinition)
        
        # Should include all defined fields
        assert "name" in fields
        assert "llm_name" in fields
        assert "tool_names" in fields
        assert "memory_config" in fields
        assert "config" in fields
        assert "metadata" in fields
        
        # Should not include non-existent fields
        assert "nonexistent_field" not in fields
    
    def test_has_config_sink(self):
        """Test detection of config/metadata sink fields."""
        # AgentDefinition has both config and metadata (config takes precedence)
        has_sink, field_name = _has_config_sink(AgentDefinition)
        assert has_sink is True
        assert field_name == "config"
        
        # LLMDefinition has metadata
        has_sink, field_name = _has_config_sink(LLMDefinition)
        assert has_sink is True
        assert field_name == "metadata"


class TestAliasing:
    """Test DSL → AST field name aliasing."""
    
    def test_agent_llm_alias(self):
        """Test that 'llm' in DSL maps to 'llm_name' in AST."""
        source = """
agent "test_agent" {
    llm: "gpt-4"
    goal: "Test goal"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        agent = module.body[0]
        assert isinstance(agent, AgentDefinition)
        assert agent.llm_name == "gpt-4"
        assert agent.goal == "Test goal"
    
    def test_agent_tools_alias(self):
        """Test that 'tools' maps to 'tool_names'."""
        source = """
agent "test_agent" {
    llm: "gpt-4"
    tools: ["search", "calculator"]
    goal: "Test goal"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        agent = module.body[0]
        assert agent.tool_names == ["search", "calculator"]
    
    def test_agent_memory_alias(self):
        """Test that 'memory' maps to 'memory_config'."""
        source = """
agent "test_agent" {
    llm: "gpt-4"
    memory: "conversation"
    goal: "Test goal"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        agent = module.body[0]
        assert agent.memory_config == "conversation"
    
    def test_agent_system_alias(self):
        """Test that 'system' maps to 'system_prompt'."""
        source = """
agent "test_agent" {
    llm: "gpt-4"
    system: "You are a helpful assistant."
    goal: "Test goal"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        agent = module.body[0]
        assert agent.system_prompt == "You are a helpful assistant."
    
    def test_llm_system_alias(self):
        """Test that 'system' maps to 'system_prompt' for LLMs."""
        source = """
llm "test_llm" {
    model: "gpt-4"
    system: "You are a test assistant."
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        llm = module.body[0]
        assert isinstance(llm, LLMDefinition)
        assert llm.system_prompt == "You are a test assistant."
    
    def test_graph_start_alias(self):
        """Test that 'start' maps to 'start_agent' for graphs."""
        source = """
graph "test_graph" {
    start: "agent1"
    edges: []
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        graph = module.body[0]
        assert isinstance(graph, GraphDefinition)
        assert graph.start_agent == "agent1"


class TestUnknownFieldRouting:
    """Test that unknown fields are properly routed to config/metadata sinks."""
    
    def test_agent_unknown_fields_to_config(self):
        """Test that unknown fields in agent declarations go to config."""
        source = """
agent "test_agent" {
    llm: "gpt-4"
    goal: "Test goal"
    custom_field: "custom_value"
    another_field: 42
    nested_config: {
        key1: "value1",
        key2: "value2"
    }
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        agent = module.body[0]
        assert isinstance(agent, AgentDefinition)
        
        # Known fields should be set correctly
        assert agent.llm_name == "gpt-4"
        assert agent.goal == "Test goal"
        
        # Unknown fields should be in config
        assert agent.config["custom_field"] == "custom_value"
        assert agent.config["another_field"] == 42
        assert agent.config["nested_config"]["key1"] == "value1"
    
    def test_llm_unknown_fields_to_metadata(self):
        """Test that unknown fields in LLM declarations go to metadata."""
        source = """
llm "test_llm" {
    model: "gpt-4"
    temperature: 0.7
    cost_tier: "premium"
    use_case: "documentation"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        llm = module.body[0]
        assert isinstance(llm, LLMDefinition)
        
        # Known fields
        assert llm.model == "gpt-4"
        assert llm.temperature == 0.7
        
        # Unknown fields in metadata
        assert llm.metadata["cost_tier"] == "premium"
        assert llm.metadata["use_case"] == "documentation"
    
    def test_agent_with_explicit_config_and_unknown_fields(self):
        """Test that explicit config field merges with unknown fields."""
        source = """
agent "test_agent" {
    llm: "gpt-4"
    goal: "Test goal"
    config: {
        explicit_key: "explicit_value"
    }
    unknown_key: "unknown_value"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        agent = module.body[0]
        
        # Both explicit and unknown should be in config
        assert agent.config["explicit_key"] == "explicit_value"
        assert agent.config["unknown_key"] == "unknown_value"


class TestDefaultPreservation:
    """Test that dataclass defaults are preserved and not overridden."""
    
    def test_agent_defaults_not_overridden(self):
        """Test that omitted fields use dataclass defaults."""
        source = """
agent "minimal_agent" {
    llm: "gpt-4"
    goal: "Minimal agent"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        agent = module.body[0]
        
        # Explicitly set fields
        assert agent.llm_name == "gpt-4"
        assert agent.goal == "Minimal agent"
        
        # Default fields should have their default values
        assert agent.tool_names == []  # default_factory=list
        assert agent.memory_config is None  # default None
        assert agent.system_prompt is None  # default None
        assert agent.max_turns is None  # default None
        assert agent.temperature is None  # default None
    
    def test_llm_defaults_preserved(self):
        """Test LLM default values."""
        source = """
llm "minimal_llm" {
    model: "gpt-4"
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        llm = module.body[0]
        
        assert llm.model == "gpt-4"
        assert llm.temperature is None  # default
        assert llm.max_tokens is None  # default
        assert llm.stream is True  # default True
        assert llm.tools == []  # default_factory=list


class TestComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_agent_with_all_features(self):
        """Test agent with aliases, known fields, and unknown fields."""
        source = """
agent "complex_agent" {
    llm: "gpt-4"
    tools: ["search_web", "calculator", "database_query"]
    memory: "conversation"
    system: "You are a research assistant with access to multiple tools."
    goal: "Answer user queries accurately using available tools"
    max_turns: 10
    temperature: 0.7
    top_p: 0.9
    
    # Unknown fields for config
    retry_policy: "exponential_backoff"
    max_retries: 3
    timeout_seconds: 30
    priority: "high"
    tags: ["research", "production"]
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        agent = module.body[0]
        
        # Aliased fields
        assert agent.llm_name == "gpt-4"
        assert agent.tool_names == ["search_web", "calculator", "database_query"]
        assert agent.memory_config == "conversation"
        assert agent.system_prompt == "You are a research assistant with access to multiple tools."
        
        # Direct known fields
        assert agent.goal == "Answer user queries accurately using available tools"
        assert agent.max_turns == 10
        assert agent.temperature == 0.7
        assert agent.top_p == 0.9
        
        # Unknown fields in config
        assert agent.config["retry_policy"] == "exponential_backoff"
        assert agent.config["max_retries"] == 3
        assert agent.config["timeout_seconds"] == 30
        assert agent.config["priority"] == "high"
        assert agent.config["tags"] == ["research", "production"]
    
    def test_llm_with_nested_config(self):
        """Test LLM with nested configuration objects."""
        source = """
llm "advanced_llm" {
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 2000
    
    safety: {
        content_filter: "strict",
        pii_detection: true,
        toxicity_threshold: 0.1
    }
    
    # Unknown fields
    billing: {
        tier: "premium",
        cost_tracking: true
    }
    monitoring: {
        log_level: "info",
        trace_enabled: true
    }
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        llm = module.body[0]
        
        # Known fields
        assert llm.model == "gpt-4"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 2000
        assert llm.safety["content_filter"] == "strict"
        
        # Unknown nested objects in metadata
        assert llm.metadata["billing"]["tier"] == "premium"
        assert llm.metadata["monitoring"]["log_level"] == "info"


class TestUnitLevelFiltering:
    """Test the filter_config_for_dataclass function directly."""
    
    def test_filter_with_aliases(self):
        """Test filtering with alias mapping."""
        config = {
            "llm": "gpt-4",
            "tools": ["search"],
            "goal": "test",
            "unknown_field": 42
        }
        
        kwargs, leftover = filter_config_for_dataclass(
            config, AgentDefinition, AGENT_ALIASES
        )
        
        # Aliased field
        assert "llm_name" in kwargs
        assert kwargs["llm_name"] == "gpt-4"
        
        # Aliased field
        assert "tool_names" in kwargs
        assert kwargs["tool_names"] == ["search"]
        
        # Known field
        assert "goal" in kwargs
        assert kwargs["goal"] == "test"
        
        # Unknown field
        assert "unknown_field" in leftover
        assert leftover["unknown_field"] == 42
    
    def test_build_with_config_sink(self):
        """Test build_dataclass_with_config merges leftover into config."""
        config = {
            "llm": "gpt-4",
            "goal": "test",
            "custom1": "value1",
            "custom2": 42
        }
        
        agent = build_dataclass_with_config(
            AgentDefinition,
            config=config,
            aliases=AGENT_ALIASES,
            name="test_agent"
        )
        
        assert agent.name == "test_agent"
        assert agent.llm_name == "gpt-4"
        assert agent.goal == "test"
        assert agent.config["custom1"] == "value1"
        assert agent.config["custom2"] == 42


class TestBackwardsCompatibility:
    """Test that existing DSL code still works."""
    
    def test_existing_agent_syntax(self):
        """Test that existing agent declarations parse correctly."""
        source = """
agent "support_agent" {
    llm: "gpt-4"
    tools: ["search_docs", "create_ticket"]
    goal: "Help users with their support queries"
    temperature: 0.7
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        agent = module.body[0]
        assert agent.name == "support_agent"
        assert agent.llm_name == "gpt-4"
        assert len(agent.tool_names) == 2
    
    def test_existing_llm_syntax(self):
        """Test that existing LLM declarations parse correctly."""
        source = """
llm "creative_writer" {
    model: "claude-3-opus"
    temperature: 0.9
    max_tokens: 4000
}
"""
        parser = N3Parser(source, path="test.n3")
        module = parser.parse()
        
        llm = module.body[0]
        assert llm.name == "creative_writer"
        assert llm.model == "claude-3-opus"
        assert llm.temperature == 0.9
