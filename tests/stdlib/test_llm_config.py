"""Tests for LLM configuration standard library components."""

import pytest
from namel3ss.stdlib.llm import (
    LLMConfigField,
    LLMConfigSpec,
    STANDARD_LLM_FIELDS,
    LLMValidationError,
    get_llm_config_spec,
    list_llm_config_fields,
    validate_llm_config,
    validate_llm_config_strict,
    suggest_llm_config,
    get_standard_llm_config,
)


class TestLLMConfigFields:
    """Test LLM configuration field definitions."""
    
    def test_field_enum_values(self):
        """Test that field enum has expected core values."""
        expected_fields = {
            "temperature", "max_tokens", "top_p", "top_k",
            "frequency_penalty", "presence_penalty", "stop_sequences",
            "seed", "stream", "system_prompt", "provider", "model"
        }
        actual_fields = {field.value for field in LLMConfigField}
        assert expected_fields.issubset(actual_fields)
    
    def test_standard_fields_coverage(self):
        """Test that all enum values have specifications."""
        for field in LLMConfigField:
            assert field in STANDARD_LLM_FIELDS
            spec = STANDARD_LLM_FIELDS[field]
            assert isinstance(spec, LLMConfigSpec)
            assert spec.field == field
    
    def test_field_descriptions(self):
        """Test that all fields have descriptions."""
        for field, spec in STANDARD_LLM_FIELDS.items():
            assert isinstance(spec.description, str)
            assert len(spec.description) > 0
    
    def test_required_fields(self):
        """Test identification of required fields."""
        required_fields = [
            field for field, spec in STANDARD_LLM_FIELDS.items() 
            if spec.required
        ]
        
        # Provider and model should be required
        required_names = {field.value for field in required_fields}
        assert "provider" in required_names
        assert "model" in required_names
    
    def test_field_types(self):
        """Test field type specifications."""
        type_mapping = {
            LLMConfigField.TEMPERATURE: 'float',
            LLMConfigField.MAX_TOKENS: 'int',
            LLMConfigField.STREAM: 'bool',
            LLMConfigField.STOP_SEQUENCES: 'list',
            LLMConfigField.PROVIDER: 'str',
        }
        
        for field, expected_type in type_mapping.items():
            spec = STANDARD_LLM_FIELDS[field]
            assert spec.field_type == expected_type
    
    def test_value_ranges(self):
        """Test field value range specifications."""
        # Temperature should be 0-2
        temp_spec = STANDARD_LLM_FIELDS[LLMConfigField.TEMPERATURE]
        assert temp_spec.min_value == 0.0
        assert temp_spec.max_value == 2.0
        
        # Top P should be 0-1
        top_p_spec = STANDARD_LLM_FIELDS[LLMConfigField.TOP_P]
        assert top_p_spec.min_value == 0.0
        assert top_p_spec.max_value == 1.0
        
        # Penalties should be -2 to 2
        freq_spec = STANDARD_LLM_FIELDS[LLMConfigField.FREQUENCY_PENALTY]
        assert freq_spec.min_value == -2.0
        assert freq_spec.max_value == 2.0


class TestLLMConfigValidation:
    """Test LLM configuration validation logic."""
    
    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.9
        }
        errors = validate_llm_config(config)
        assert not errors
    
    def test_missing_required_fields(self):
        """Test validation fails for missing required fields."""
        config = {"temperature": 0.7}  # Missing provider and model
        errors = validate_llm_config(config)
        
        assert "provider" in errors
        assert "model" in errors
    
    def test_invalid_types(self):
        """Test validation catches type errors."""
        config = {
            "provider": "openai",
            "model": "gpt-4", 
            "temperature": "hot",  # Should be float
            "max_tokens": "lots",  # Should be int
            "stream": "yes"        # Should be bool
        }
        errors = validate_llm_config(config)
        
        assert "temperature" in errors
        assert "max_tokens" in errors
        assert "stream" in errors
    
    def test_out_of_range_values(self):
        """Test validation catches out-of-range values."""
        config = {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 5.0,     # Max is 2.0
            "top_p": -0.1,         # Min is 0.0
            "max_tokens": -100     # Min is 1
        }
        errors = validate_llm_config(config)
        
        assert "temperature" in errors
        assert "top_p" in errors  
        assert "max_tokens" in errors
    
    def test_invalid_enum_values(self):
        """Test validation of enum constrained fields."""
        config = {
            "provider": "unknown_provider",
            "model": "gpt-4"
        }
        errors = validate_llm_config(config)
        
        assert "provider" in errors
    
    def test_cross_field_validation(self):
        """Test cross-field validation rules."""
        config = {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.05,   # Very low
            "top_p": 0.05         # Very low
        }
        errors = validate_llm_config(config)
        
        # Should warn about restrictive sampling
        assert "temperature" in errors
    
    def test_stop_sequences_validation(self):
        """Test validation of stop sequences."""
        # Valid stop sequences
        config = {
            "provider": "openai",
            "model": "gpt-4",
            "stop_sequences": ["\\n\\n", "END"]
        }
        errors = validate_llm_config(config)
        assert "stop_sequences" not in errors
        
        # Invalid stop sequences
        config["stop_sequences"] = ["", "valid"]  # Empty sequence
        errors = validate_llm_config(config)
        assert "stop_sequences" in errors
        
        config["stop_sequences"] = [123, "valid"]  # Non-string
        errors = validate_llm_config(config)
        assert "stop_sequences" in errors
    
    def test_strict_validation(self):
        """Test strict validation mode."""
        valid_config = {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7
        }
        
        # Should pass with valid config
        validate_llm_config_strict(valid_config)
        
        # Should raise exception with invalid config
        invalid_config = {"provider": "invalid"}
        with pytest.raises(LLMValidationError):
            validate_llm_config_strict(invalid_config)


class TestLLMConfigSuggestions:
    """Test LLM configuration suggestion functionality."""
    
    def test_suggest_general_config(self):
        """Test general use case suggestions."""
        config = suggest_llm_config("openai", "gpt-4")
        
        assert config["provider"] == "openai"
        assert config["model"] == "gpt-4"
        assert "temperature" in config
        assert "max_tokens" in config
        
        # Should be valid
        errors = validate_llm_config(config)
        assert not errors
    
    def test_suggest_creative_config(self):
        """Test creative use case suggestions."""
        config = suggest_llm_config("anthropic", "claude-3-opus", "creative")
        
        assert config["temperature"] == 0.8  # Higher for creativity
        assert config["frequency_penalty"] == 0.1
        
        errors = validate_llm_config(config)
        assert not errors
    
    def test_suggest_precise_config(self):
        """Test precise use case suggestions."""
        config = suggest_llm_config("openai", "gpt-4", "precise")
        
        assert config["temperature"] == 0.2  # Lower for precision
        
        errors = validate_llm_config(config)
        assert not errors
    
    def test_suggest_with_overrides(self):
        """Test suggestions with custom overrides."""
        config = suggest_llm_config(
            "openai", "gpt-4", "general",
            temperature=0.3,
            custom_field="custom_value"
        )
        
        assert config["temperature"] == 0.3
        assert config["custom_field"] == "custom_value"
    
    def test_get_standard_config(self):
        """Test getting standard default configuration."""
        config = get_standard_llm_config()
        
        # Should have default values for fields with defaults
        assert "temperature" in config
        assert "max_tokens" in config
        assert "stream" in config
        
        # Should not have required fields without defaults
        assert "provider" not in config
        assert "model" not in config


class TestLLMConfigFieldLookup:
    """Test LLM configuration field lookup functions."""
    
    def test_get_config_spec_by_enum(self):
        """Test getting spec by enum value."""
        spec = get_llm_config_spec(LLMConfigField.TEMPERATURE)
        assert spec.field == LLMConfigField.TEMPERATURE
        assert spec.field_type == 'float'
    
    def test_get_config_spec_by_string(self):
        """Test getting spec by string value."""
        spec = get_llm_config_spec("temperature")
        assert spec.field == LLMConfigField.TEMPERATURE
    
    def test_get_config_spec_invalid(self):
        """Test error on invalid field."""
        with pytest.raises(ValueError, match="Unknown LLM config field"):
            get_llm_config_spec("invalid_field")
    
    def test_list_config_fields(self):
        """Test listing all field names."""
        fields = list_llm_config_fields()
        assert "temperature" in fields
        assert "max_tokens" in fields
        assert "provider" in fields
        assert len(fields) == len(LLMConfigField)


class TestLLMConfigIntegration:
    """Test integration scenarios with LLM configurations."""
    
    def test_provider_specific_configs(self):
        """Test configurations for different providers.""" 
        providers = [
            ("openai", "gpt-4"),
            ("anthropic", "claude-3-opus"),
            ("azure", "gpt-35-turbo"),
            ("vertex", "gemini-pro"),
            ("ollama", "llama2"),
            ("local", "mistral-7b")
        ]
        
        for provider, model in providers:
            config = suggest_llm_config(provider, model)
            errors = validate_llm_config(config)
            assert not errors, f"Config for {provider}/{model} failed: {errors}"
    
    def test_realistic_production_configs(self):
        """Test realistic production configurations."""
        configs = [
            # High-throughput API
            {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 150,
                "stream": True
            },
            
            # Creative writing assistant  
            {
                "provider": "anthropic",
                "model": "claude-3-opus",
                "temperature": 0.9,
                "top_p": 0.95,
                "frequency_penalty": 0.3,
                "system_prompt": "You are a creative writing assistant."
            },
            
            # Code generation
            {
                "provider": "openai", 
                "model": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 2048,
                "stop_sequences": ["```", "\\n\\n\\n"]
            }
        ]
        
        for config in configs:
            errors = validate_llm_config(config)
            assert not errors, f"Production config failed: {errors}"