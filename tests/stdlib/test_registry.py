"""Tests for standard library registry system."""

import pytest
from namel3ss.stdlib.registry import (
    StandardLibraryRegistry,
    ComponentType,
    get_memory_policy,
    get_llm_config_field,
    get_tool_spec,
    list_memory_policies,
    list_llm_config_fields,
    list_tool_categories,
    search_stdlib,
    get_stdlib_component,
    validate_stdlib_config,
    suggest_stdlib_config,
    get_stdlib_summary,
)
from namel3ss.stdlib.memory.policies import MemoryPolicy
from namel3ss.stdlib.llm.config import LLMConfigField
from namel3ss.stdlib.tools import ToolCategory


class TestComponentTypes:
    """Test standard library component type definitions."""
    
    def test_component_type_enum(self):
        """Test component type enum values."""
        expected_types = {"memory", "llm", "tool"}
        actual_types = {comp.value for comp in ComponentType}
        assert expected_types.issubset(actual_types)
    
    def test_component_type_mapping(self):
        """Test that each component type has proper mapping."""
        registry = StandardLibraryRegistry()
        
        # Each component type should have entries
        assert len(registry.memory_policies) > 0
        assert len(registry.llm_config_fields) > 0
        assert len(registry.tool_specs) > 0


class TestMemoryPolicyRegistry:
    """Test memory policy registry functionality."""
    
    def test_get_memory_policy_by_enum(self):
        """Test getting memory policy by enum value."""
        policy = get_memory_policy(MemoryPolicy.CONVERSATION_WINDOW)
        assert policy.policy == MemoryPolicy.CONVERSATION_WINDOW
        assert policy.description is not None
    
    def test_get_memory_policy_by_string(self):
        """Test getting memory policy by string value."""
        policy = get_memory_policy("conversation_window")
        assert policy.policy == MemoryPolicy.CONVERSATION_WINDOW
    
    def test_get_memory_policy_invalid(self):
        """Test error handling for invalid memory policy."""
        with pytest.raises(ValueError, match="Unknown memory policy"):
            get_memory_policy("invalid_policy")
    
    def test_list_memory_policies(self):
        """Test listing all memory policy names."""
        policies = list_memory_policies()
        expected = ["none", "conversation_window", "full_history", "summary"]
        
        for policy in expected:
            assert policy in policies
    
    def test_memory_policy_search(self):
        """Test searching memory policies by description."""
        results = search_stdlib("window", component_type="memory")
        assert len(results) > 0
        assert any("conversation_window" in str(result) for result in results)


class TestLLMConfigRegistry:
    """Test LLM config field registry functionality."""
    
    def test_get_llm_config_field_by_enum(self):
        """Test getting LLM config field by enum value."""
        field = get_llm_config_field(LLMConfigField.TEMPERATURE)
        assert field.field == LLMConfigField.TEMPERATURE
        assert field.description is not None
        assert field.default_value is not None
    
    def test_get_llm_config_field_by_string(self):
        """Test getting LLM config field by string value."""
        field = get_llm_config_field("temperature")
        assert field.field == LLMConfigField.TEMPERATURE
    
    def test_get_llm_config_field_invalid(self):
        """Test error handling for invalid LLM config field."""
        with pytest.raises(ValueError, match="Unknown LLM config field"):
            get_llm_config_field("invalid_field")
    
    def test_list_llm_config_fields(self):
        """Test listing all LLM config field names."""
        fields = list_llm_config_fields()
        expected = ["temperature", "max_tokens", "top_p", "frequency_penalty"]
        
        for field in expected:
            assert field in fields
    
    def test_llm_config_search(self):
        """Test searching LLM config fields by description."""
        results = search_stdlib("randomness", component_type="llm")
        assert len(results) > 0
        # Should find temperature field
        assert any("temperature" in str(result) for result in results)


class TestToolSpecRegistry:
    """Test tool specification registry functionality."""
    
    def test_get_tool_spec_by_enum(self):
        """Test getting tool spec by enum value."""
        spec = get_tool_spec(ToolCategory.HTTP)
        assert spec.category == ToolCategory.HTTP
        assert len(spec.required_fields) > 0
    
    def test_get_tool_spec_by_string(self):
        """Test getting tool spec by string value."""
        spec = get_tool_spec("http")
        assert spec.category == ToolCategory.HTTP
    
    def test_get_tool_spec_invalid(self):
        """Test error handling for invalid tool category."""
        with pytest.raises(ValueError, match="Unknown tool category"):
            get_tool_spec("invalid_tool")
    
    def test_list_tool_categories(self):
        """Test listing all tool category names."""
        categories = list_tool_categories()
        expected = ["http", "database", "vector_search"]
        
        for category in expected:
            assert category in categories
    
    def test_tool_spec_search(self):
        """Test searching tool specs by description."""
        results = search_stdlib("HTTP", component_type="tool")
        assert len(results) > 0
        # Should find HTTP tool spec
        assert any("http" in str(result) for result in results)


class TestGenericRegistry:
    """Test generic registry functionality."""
    
    def test_get_stdlib_component_memory(self):
        """Test getting memory component through generic interface."""
        component = get_stdlib_component("memory", "conversation_window")
        assert component.policy == MemoryPolicy.CONVERSATION_WINDOW
    
    def test_get_stdlib_component_llm(self):
        """Test getting LLM component through generic interface."""
        component = get_stdlib_component("llm", "temperature")
        assert component.field == LLMConfigField.TEMPERATURE
    
    def test_get_stdlib_component_tool(self):
        """Test getting tool component through generic interface."""
        component = get_stdlib_component("tool", "http")
        assert component.category == ToolCategory.HTTP
    
    def test_get_stdlib_component_invalid_type(self):
        """Test error handling for invalid component type."""
        with pytest.raises(ValueError, match="Unknown component type"):
            get_stdlib_component("invalid", "anything")
    
    def test_get_stdlib_component_invalid_name(self):
        """Test error handling for invalid component name."""
        with pytest.raises(ValueError):
            get_stdlib_component("memory", "invalid_policy")


class TestSearchFunctionality:
    """Test standard library search capabilities."""
    
    def test_search_all_components(self):
        """Test searching across all component types."""
        results = search_stdlib("config")
        
        # Should find results from multiple component types
        component_types = set()
        for result in results:
            if "memory" in str(result).lower():
                component_types.add("memory")
            elif "llm" in str(result).lower() or "temperature" in str(result).lower():
                component_types.add("llm")
            elif "tool" in str(result).lower():
                component_types.add("tool")
        
        assert len(component_types) > 0
    
    def test_search_filtered_by_type(self):
        """Test searching with component type filter."""
        memory_results = search_stdlib("policy", component_type="memory")
        llm_results = search_stdlib("token", component_type="llm") 
        tool_results = search_stdlib("request", component_type="tool")
        
        assert len(memory_results) > 0
        assert len(llm_results) > 0
        assert len(tool_results) > 0
    
    def test_search_case_insensitive(self):
        """Test case insensitive search."""
        results_lower = search_stdlib("temperature")
        results_upper = search_stdlib("TEMPERATURE")
        results_mixed = search_stdlib("Temperature")
        
        assert len(results_lower) > 0
        assert len(results_upper) == len(results_lower)
        assert len(results_mixed) == len(results_lower)
    
    def test_search_no_results(self):
        """Test search with no matching results."""
        results = search_stdlib("nonexistent_search_term_12345")
        assert len(results) == 0
    
    def test_search_partial_match(self):
        """Test partial matching in search."""
        results = search_stdlib("temp")  # Should match "temperature"
        assert len(results) > 0
        assert any("temperature" in str(result).lower() for result in results)


class TestConfigValidation:
    """Test standard library configuration validation."""
    
    def test_validate_memory_config(self):
        """Test validating memory policy configuration."""
        valid_config = {
            "policy": "conversation_window",
            "window_size": 10
        }
        
        errors = validate_stdlib_config("memory", valid_config)
        assert len(errors) == 0
    
    def test_validate_llm_config(self):
        """Test validating LLM configuration."""
        valid_config = {
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.9
        }
        
        errors = validate_stdlib_config("llm", valid_config)
        assert len(errors) == 0
    
    def test_validate_tool_config(self):
        """Test validating tool configuration."""
        valid_config = {
            "category": "http",
            "method": "GET",
            "url": "https://api.example.com",
            "description": "Test API"
        }
        
        errors = validate_stdlib_config("tool", valid_config)
        assert len(errors) == 0
    
    def test_validate_invalid_config(self):
        """Test validation of invalid configurations."""
        invalid_memory = {"policy": "invalid_policy"}
        invalid_llm = {"temperature": -1.0}  # Out of range
        invalid_tool = {"category": "http"}  # Missing required fields
        
        memory_errors = validate_stdlib_config("memory", invalid_memory)
        llm_errors = validate_stdlib_config("llm", invalid_llm)
        tool_errors = validate_stdlib_config("tool", invalid_tool)
        
        assert len(memory_errors) > 0
        assert len(llm_errors) > 0
        assert len(tool_errors) > 0


class TestConfigSuggestions:
    """Test standard library configuration suggestions."""
    
    def test_suggest_memory_config(self):
        """Test memory configuration suggestions."""
        config = suggest_stdlib_config("memory")
        assert "policy" in config
        assert config["policy"] in ["none", "conversation_window", "full_history", "summary"]
    
    def test_suggest_llm_config(self):
        """Test LLM configuration suggestions."""
        config = suggest_stdlib_config("llm")
        assert "temperature" in config
        assert "max_tokens" in config
        assert 0.0 <= config["temperature"] <= 2.0
        assert config["max_tokens"] > 0
    
    def test_suggest_tool_config(self):
        """Test tool configuration suggestions."""
        config = suggest_stdlib_config("tool", category="http")
        assert "category" in config
        assert "method" in config
        assert "timeout" in config
        assert config["category"] == "http"
    
    def test_suggest_with_overrides(self):
        """Test suggestions with custom overrides."""
        config = suggest_stdlib_config("llm", temperature=0.5, custom_field="value")
        assert config["temperature"] == 0.5
        assert config["custom_field"] == "value"


class TestRegistrySummary:
    """Test standard library summary functionality."""
    
    def test_get_stdlib_summary(self):
        """Test getting complete standard library summary."""
        summary = get_stdlib_summary()
        
        # Should have all component types
        assert "memory" in summary
        assert "llm" in summary
        assert "tool" in summary
        
        # Each section should have items
        assert len(summary["memory"]) > 0
        assert len(summary["llm"]) > 0
        assert len(summary["tool"]) > 0
    
    def test_stdlib_summary_structure(self):
        """Test standard library summary structure."""
        summary = get_stdlib_summary()
        
        # Memory policies
        for policy in summary["memory"]:
            assert "name" in policy
            assert "description" in policy
        
        # LLM config fields
        for field in summary["llm"]:
            assert "name" in field
            assert "description" in field
            assert "type" in field
            assert "default" in field
        
        # Tool categories
        for tool in summary["tool"]:
            assert "name" in tool
            assert "description" in field
            assert "required_fields" in tool
            assert "optional_fields" in tool


class TestRegistryIntegration:
    """Test integration scenarios with the registry system."""
    
    def test_component_cross_references(self):
        """Test that components can reference each other."""
        # Get an LLM config that might reference memory
        llm_fields = list_llm_config_fields()
        memory_policies = list_memory_policies()
        
        # Both should be available for cross-referencing
        assert len(llm_fields) > 0
        assert len(memory_policies) > 0
    
    def test_registry_consistency(self):
        """Test registry internal consistency."""
        registry = StandardLibraryRegistry()
        
        # Verify all listed policies are accessible
        for policy_name in list_memory_policies():
            policy = get_memory_policy(policy_name)
            assert policy is not None
        
        # Verify all listed fields are accessible
        for field_name in list_llm_config_fields():
            field = get_llm_config_field(field_name)
            assert field is not None
        
        # Verify all listed tools are accessible
        for tool_name in list_tool_categories():
            if tool_name in ["http", "database", "vector_search"]:  # Only standard specs
                tool = get_tool_spec(tool_name)
                assert tool is not None
    
    def test_registry_completeness(self):
        """Test that registry covers all expected components."""
        # Essential memory policies
        essential_policies = ["none", "conversation_window", "full_history"]
        for policy in essential_policies:
            assert policy in list_memory_policies()
        
        # Essential LLM config fields  
        essential_fields = ["temperature", "max_tokens", "top_p"]
        for field in essential_fields:
            assert field in list_llm_config_fields()
        
        # Essential tool categories
        essential_tools = ["http", "database", "vector_search"]
        for tool in essential_tools:
            assert tool in list_tool_categories()
    
    def test_registry_performance(self):
        """Test registry performance for repeated access."""
        import time
        
        start_time = time.time()
        
        # Perform many registry operations
        for _ in range(100):
            list_memory_policies()
            list_llm_config_fields()
            list_tool_categories()
            search_stdlib("config")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly (under 1 second)
        assert duration < 1.0