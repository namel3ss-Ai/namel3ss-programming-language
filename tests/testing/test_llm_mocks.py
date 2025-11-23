"""
Tests for namel3ss LLM mocking system.

This module tests the deterministic LLM mocking functionality that allows
testing namel3ss applications without making live API calls.
"""

import pytest
import re
from typing import Dict, Any, List

from namel3ss.testing.mocks import (
    MockLLMProvider, MockN3Provider, ResponseMapping, 
    create_mock_llm_from_spec, _compile_patterns
)
from namel3ss.testing import MockLLMSpec, MockLLMResponse


class TestResponseMapping:
    """Test the ResponseMapping utility class."""
    
    def test_exact_pattern_matching(self):
        """Test exact string pattern matching."""
        responses = {
            "Hello world": "Hi there!",
            "What is AI?": "AI is artificial intelligence."
        }
        
        mapping = ResponseMapping(responses)
        
        assert mapping.get_response("Hello world") == "Hi there!"
        assert mapping.get_response("What is AI?") == "AI is artificial intelligence."
        assert mapping.get_response("Unknown prompt") is None
    
    def test_regex_pattern_matching(self):
        """Test regex pattern matching."""
        responses = {
            r"Hello (\w+)": "Hi {1}!",
            r"Count to (\d+)": "1, 2, 3... {1}",
            r"Tell me about (.*)": "Here's information about {1}."
        }
        
        mapping = ResponseMapping(responses)
        
        assert mapping.get_response("Hello Alice") == "Hi Alice!"
        assert mapping.get_response("Count to 5") == "1, 2, 3... 5"
        assert mapping.get_response("Tell me about Python") == "Here's information about Python."
    
    def test_fallback_response(self):
        """Test fallback response for unmatched patterns."""
        responses = {
            "specific": "Specific response"
        }
        fallback = "Default response"
        
        mapping = ResponseMapping(responses, fallback)
        
        assert mapping.get_response("specific") == "Specific response"
        assert mapping.get_response("unknown") == "Default response"
    
    def test_no_fallback_returns_none(self):
        """Test that unmatched patterns return None without fallback."""
        mapping = ResponseMapping({"specific": "response"})
        assert mapping.get_response("unknown") is None
    
    def test_pattern_compilation(self):
        """Test that patterns are properly compiled."""
        responses = {
            r"Hello (\w+)": "Hi {1}!",
            "exact_match": "exact response"
        }
        
        compiled = _compile_patterns(responses)
        
        # Should have compiled regex for pattern, kept string for exact
        assert any(hasattr(pattern, 'pattern') for pattern in compiled.keys())
        assert any(isinstance(pattern, str) for pattern in compiled.keys())


class TestMockLLMProvider:
    """Test the MockLLMProvider implementation."""
    
    def test_basic_mock_creation(self):
        """Test creating a basic mock LLM provider."""
        responses = {
            "Hello": "Hi there!",
            r"My name is (\w+)": "Nice to meet you, {1}!"
        }
        
        provider = MockLLMProvider("mock_model", responses)
        
        assert provider.model_name == "mock_model"
        assert provider._responses is not None
    
    def test_mock_generate_with_exact_match(self):
        """Test generation with exact string matching."""
        responses = {"Hello world": "Hi there!"}
        provider = MockLLMProvider("mock_model", responses)
        
        result = provider.generate("Hello world")
        
        assert result.output_text == "Hi there!"
        assert result.metadata["mock"] is True
        assert result.metadata["pattern_matched"] == "Hello world"
    
    def test_mock_generate_with_regex_match(self):
        """Test generation with regex pattern matching."""
        responses = {r"Hello (\w+)": "Hi {1}!"}
        provider = MockLLMProvider("mock_model", responses)
        
        result = provider.generate("Hello Alice")
        
        assert result.output_text == "Hi Alice!"
        assert result.metadata["pattern_matched"] is not None
    
    def test_mock_generate_with_fallback(self):
        """Test generation with fallback response."""
        responses = {"specific": "specific response"}
        fallback = "Default mock response"
        
        provider = MockLLMProvider("mock_model", responses, fallback)
        
        result = provider.generate("unknown prompt")
        
        assert result.output_text == "Default mock response"
        assert result.metadata["used_fallback"] is True
    
    def test_mock_generate_no_match_no_fallback(self):
        """Test generation with no match and no fallback raises error."""
        provider = MockLLMProvider("mock_model", {"specific": "response"})
        
        with pytest.raises(ValueError, match="No mock response found"):
            provider.generate("unknown prompt")
    
    def test_mock_generate_with_metadata(self):
        """Test that generated responses include proper metadata."""
        provider = MockLLMProvider("mock_model", {"test": "response"})
        
        result = provider.generate("test")
        
        assert "mock" in result.metadata
        assert "model_name" in result.metadata
        assert "pattern_matched" in result.metadata
        assert result.metadata["mock"] is True
        assert result.metadata["model_name"] == "mock_model"
    
    def test_mock_with_custom_response_objects(self):
        """Test using MockLLMResponse objects directly."""
        custom_response = MockLLMResponse(
            output_text="Custom response",
            metadata={"custom": True},
            delay_ms=100
        )
        
        provider = MockLLMProvider("mock_model", {"test": custom_response})
        
        result = provider.generate("test")
        
        assert result.output_text == "Custom response"
        assert result.metadata["custom"] is True
        assert result.metadata["mock"] is True  # Should merge metadata


class TestMockN3Provider:
    """Test the MockN3Provider implementation."""
    
    def test_mock_n3_provider_creation(self):
        """Test creating a mock N3 provider."""
        config = {"api_key": "mock_key", "base_url": "http://mock.api"}
        responses = {"test prompt": "test response"}
        
        provider = MockN3Provider(config, responses)
        
        assert provider.config == config
        assert provider._llm_mock is not None
    
    def test_mock_n3_provider_generate(self):
        """Test N3 provider generation delegation."""
        config = {"api_key": "mock"}
        responses = {"Hello": "Hi!"}
        
        provider = MockN3Provider(config, responses)
        result = provider.generate("Hello")
        
        assert result.output_text == "Hi!"
        assert result.metadata["mock"] is True
    
    def test_mock_n3_provider_with_model_override(self):
        """Test N3 provider with model name override."""
        config = {"model": "custom-model"}
        responses = {"test": "response"}
        
        provider = MockN3Provider(config, responses)
        result = provider.generate("test")
        
        assert result.metadata["model_name"] == "custom-model"


class TestMockLLMFromSpec:
    """Test creating mock LLMs from specifications."""
    
    def test_create_mock_from_single_spec(self):
        """Test creating a mock from a single MockLLMSpec."""
        response = MockLLMResponse(output_text="Mock response", delay_ms=50)
        spec = MockLLMSpec(
            model_name="test_model",
            prompt_pattern="Hello",
            response=response
        )
        
        provider = create_mock_llm_from_spec([spec])
        
        assert provider.model_name == "test_model"
        result = provider.generate("Hello")
        assert result.output_text == "Mock response"
    
    def test_create_mock_from_multiple_specs(self):
        """Test creating a mock from multiple MockLLMSpec objects."""
        specs = [
            MockLLMSpec(
                model_name="test_model",
                prompt_pattern="Hello",
                response=MockLLMResponse(output_text="Hi!")
            ),
            MockLLMSpec(
                model_name="test_model", 
                prompt_pattern=r"Count (\d+)",
                response=MockLLMResponse(output_text="Counted {1}")
            )
        ]
        
        provider = create_mock_llm_from_spec(specs)
        
        assert provider.generate("Hello").output_text == "Hi!"
        assert provider.generate("Count 5").output_text == "Counted 5"
    
    def test_create_mock_with_different_models_uses_first(self):
        """Test that when specs have different models, first one is used."""
        specs = [
            MockLLMSpec("model1", "test1", MockLLMResponse("response1")),
            MockLLMSpec("model2", "test2", MockLLMResponse("response2"))
        ]
        
        provider = create_mock_llm_from_spec(specs)
        
        assert provider.model_name == "model1"
    
    def test_create_mock_empty_specs_raises_error(self):
        """Test that empty specs list raises ValueError."""
        with pytest.raises(ValueError, match="at least one MockLLMSpec"):
            create_mock_llm_from_spec([])
    
    def test_create_mock_with_fallback_spec(self):
        """Test creating mock with fallback response."""
        specs = [
            MockLLMSpec("test_model", "specific", MockLLMResponse("specific response"))
        ]
        fallback = MockLLMResponse("fallback response")
        
        provider = create_mock_llm_from_spec(specs, fallback)
        
        assert provider.generate("specific").output_text == "specific response"
        assert provider.generate("unknown").output_text == "fallback response"


class TestPatternCompilation:
    """Test the pattern compilation utility."""
    
    def test_compile_string_patterns(self):
        """Test that string patterns remain as strings."""
        patterns = {
            "exact_string": "response1",
            "another_string": "response2"
        }
        
        compiled = _compile_patterns(patterns)
        
        assert "exact_string" in compiled
        assert isinstance(list(compiled.keys())[0], str)
    
    def test_compile_regex_patterns(self):
        """Test that regex patterns are compiled."""
        patterns = {
            r"Hello (\w+)": "Hi {1}!",
            r"Count to (\d+)": "Counted {1}"
        }
        
        compiled = _compile_patterns(patterns)
        
        # All keys should be compiled regex objects
        for pattern in compiled.keys():
            assert hasattr(pattern, 'pattern')
            assert hasattr(pattern, 'match')
    
    def test_compile_mixed_patterns(self):
        """Test compiling a mix of string and regex patterns."""
        patterns = {
            "exact_match": "exact_response",
            r"regex_(\w+)": "regex_{1}"
        }
        
        compiled = _compile_patterns(patterns)
        
        assert len(compiled) == 2
        
        # Should have one string and one compiled regex
        types = [type(k) for k in compiled.keys()]
        assert str in types
        assert any(hasattr(k, 'pattern') for k in compiled.keys())
    
    def test_compile_invalid_regex_raises_error(self):
        """Test that invalid regex patterns raise errors."""
        patterns = {
            r"invalid_regex[": "response"  # Unclosed bracket
        }
        
        with pytest.raises(re.error):
            _compile_patterns(patterns)