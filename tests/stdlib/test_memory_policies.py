"""Tests for memory policy standard library components."""

import pytest
from namel3ss.stdlib.memory import (
    MemoryPolicy,
    MemoryPolicySpec,
    STANDARD_MEMORY_POLICIES,
    MemoryValidationError,
    get_memory_policy_spec,
    list_memory_policies,
    validate_memory_config,
    validate_memory_config_strict,
    suggest_memory_config,
)


class TestMemoryPolicies:
    """Test memory policy definitions and specifications."""
    
    def test_policy_enum_values(self):
        """Test that policy enum has expected values."""
        expected_policies = {"none", "conversation_window", "full_history", "summary"}
        actual_policies = {policy.value for policy in MemoryPolicy}
        assert actual_policies == expected_policies
    
    def test_standard_policies_coverage(self):
        """Test that all enum values have specifications."""
        for policy in MemoryPolicy:
            assert policy in STANDARD_MEMORY_POLICIES
            spec = STANDARD_MEMORY_POLICIES[policy]
            assert isinstance(spec, MemoryPolicySpec)
            assert spec.policy == policy
    
    def test_policy_descriptions(self):
        """Test that all policies have descriptions."""
        for policy, spec in STANDARD_MEMORY_POLICIES.items():
            assert isinstance(spec.description, str)
            assert len(spec.description) > 0
    
    def test_policy_capabilities(self):
        """Test policy capability flags."""
        # None policy should support nothing
        none_spec = STANDARD_MEMORY_POLICIES[MemoryPolicy.NONE]
        assert not none_spec.supports_max_items
        assert not none_spec.supports_window_size
        assert not none_spec.supports_summarization
        
        # Conversation window should support window and max items
        window_spec = STANDARD_MEMORY_POLICIES[MemoryPolicy.CONVERSATION_WINDOW]
        assert window_spec.supports_max_items
        assert window_spec.supports_window_size
        assert not window_spec.supports_summarization
        
        # Full history should support max items
        full_spec = STANDARD_MEMORY_POLICIES[MemoryPolicy.FULL_HISTORY]
        assert full_spec.supports_max_items
        assert not full_spec.supports_window_size
        assert not full_spec.supports_summarization
        
        # Summary should support everything
        summary_spec = STANDARD_MEMORY_POLICIES[MemoryPolicy.SUMMARY]
        assert summary_spec.supports_max_items
        assert not summary_spec.supports_window_size
        assert summary_spec.supports_summarization
    
    def test_default_configs(self):
        """Test that default configurations are valid."""
        for policy, spec in STANDARD_MEMORY_POLICIES.items():
            assert isinstance(spec.default_config, dict)
            
            # Validate the default config against the policy
            errors = validate_memory_config(policy, spec.default_config)
            assert not errors, f"Default config for {policy.value} has errors: {errors}"


class TestMemoryPolicyLookup:
    """Test memory policy lookup functions."""
    
    def test_get_policy_spec_by_enum(self):
        """Test getting spec by enum value."""
        spec = get_memory_policy_spec(MemoryPolicy.CONVERSATION_WINDOW)
        assert spec.policy == MemoryPolicy.CONVERSATION_WINDOW
        assert spec.supports_window_size
    
    def test_get_policy_spec_by_string(self):
        """Test getting spec by string value."""
        spec = get_memory_policy_spec("conversation_window")
        assert spec.policy == MemoryPolicy.CONVERSATION_WINDOW
    
    def test_get_policy_spec_invalid(self):
        """Test error on invalid policy."""
        with pytest.raises(ValueError, match="Unknown memory policy 'invalid'"):
            get_memory_policy_spec("invalid")
    
    def test_list_memory_policies(self):
        """Test listing all policy names."""
        policies = list_memory_policies()
        expected = ["none", "conversation_window", "full_history", "summary"]
        assert set(policies) == set(expected)


class TestMemoryValidation:
    """Test memory configuration validation."""
    
    def test_validate_none_policy(self):
        """Test validation of none policy."""
        # Should accept empty config
        errors = validate_memory_config("none", {})
        assert not errors
        
        # Should reject unsupported parameters
        errors = validate_memory_config("none", {"max_items": 10})
        assert "max_items" in errors
        
        errors = validate_memory_config("none", {"window_size": 5})
        assert "window_size" in errors
        
        errors = validate_memory_config("none", {"summarizer": "gpt-4"})
        assert "summarizer" in errors
    
    def test_validate_conversation_window(self):
        """Test validation of conversation window policy."""
        # Should accept supported parameters
        errors = validate_memory_config("conversation_window", {
            "window_size": 10,
            "max_items": 20
        })
        assert not errors
        
        # Should validate parameter types
        errors = validate_memory_config("conversation_window", {"window_size": "invalid"})
        assert "window_size" in errors
        
        errors = validate_memory_config("conversation_window", {"window_size": -1})
        assert "window_size" in errors
        
        # Should validate logical constraints
        errors = validate_memory_config("conversation_window", {
            "window_size": 100,
            "max_items": 10
        })
        assert "window_size" in errors
    
    def test_validate_summary_policy(self):
        """Test validation of summary policy."""
        # Should accept valid summary config
        errors = validate_memory_config("summary", {
            "summarizer": "openai/gpt-4o-mini",
            "max_summary_tokens": 512,
            "summary_trigger_messages": 20,
            "summary_trigger_tokens": 4000,
            "summary_recent_window": 5
        })
        assert not errors
        
        # Should validate summarizer format
        errors = validate_memory_config("summary", {"summarizer": "invalid"})
        assert "summarizer" in errors
        
        # Should validate token limits
        errors = validate_memory_config("summary", {"max_summary_tokens": -1})
        assert "max_summary_tokens" in errors
        
        errors = validate_memory_config("summary", {"max_summary_tokens": 10000})
        assert "max_summary_tokens" in errors
    
    def test_validate_strict_mode(self):
        """Test strict validation mode."""
        # Should pass with valid config
        validate_memory_config_strict("conversation_window", {"window_size": 10})
        
        # Should raise exception with invalid config
        with pytest.raises(MemoryValidationError):
            validate_memory_config_strict("conversation_window", {"window_size": -1})
    
    def test_suggest_config(self):
        """Test configuration suggestion."""
        # Should return default config
        config = suggest_memory_config("conversation_window")
        assert "window_size" in config
        assert config["window_size"] == 10
        
        # Should apply overrides
        config = suggest_memory_config("conversation_window", window_size=15)
        assert config["window_size"] == 15
    
    def test_invalid_policy_validation(self):
        """Test validation with invalid policy name."""
        with pytest.raises(MemoryValidationError, match="Unknown memory policy"):
            validate_memory_config_strict("invalid_policy", {})


class TestMemoryPolicyIntegration:
    """Test integration scenarios with memory policies."""
    
    def test_realistic_configs(self):
        """Test realistic memory configurations."""
        configs = [
            # Stateless chatbot
            ("none", {}),
            
            # Support chat with recent context
            ("conversation_window", {"window_size": 20, "max_items": 50}),
            
            # Research assistant with full history
            ("full_history", {"max_items": 1000}),
            
            # Production agent with summarization
            ("summary", {
                "summarizer": "anthropic/claude-3-haiku",
                "max_summary_tokens": 1024,
                "summary_trigger_messages": 30,
                "summary_recent_window": 10
            })
        ]
        
        for policy, config in configs:
            errors = validate_memory_config(policy, config)
            assert not errors, f"Realistic config for {policy} failed: {errors}"
    
    def test_migration_scenarios(self):
        """Test migration between policy types."""
        # Should be able to upgrade from none to window
        base_config = suggest_memory_config("none")
        window_config = suggest_memory_config("conversation_window")
        assert validate_memory_config("conversation_window", window_config) == {}
        
        # Should be able to upgrade from window to summary
        summary_config = suggest_memory_config("summary")
        assert validate_memory_config("summary", summary_config) == {}