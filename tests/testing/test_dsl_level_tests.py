"""
Tests for namel3ss test specification DSL and test case models.

This module tests the test specification parsing, validation, and data structures
that form the foundation of the namel3ss testing framework.
"""

import pytest
import yaml
from pathlib import Path
from typing import Dict, Any

from namel3ss.testing import (
    load_test_suite, TestSuite, TestCase, TestAssertion, 
    AssertionType, TargetType, MockLLMSpec, MockLLMResponse,
    MockToolSpec, MockToolResponse
)


class TestTestSpecificationDSL:
    """Test the test specification DSL parsing and validation."""
    
    def test_load_valid_test_suite(self):
        """Test loading a valid test suite from YAML."""
        test_file = Path(__file__).parent / "fixtures" / "tests" / "simple_test_app.test.yaml"
        
        suite = load_test_suite(test_file)
        
        assert isinstance(suite, TestSuite)
        assert suite.name == "Simple Test App Test Suite"
        assert suite.app_module == "tests/testing/fixtures/apps/simple_test_app.ai"
        assert len(suite.cases) == 4
        assert len(suite.global_mocks['llms']) == 2
        
    def test_load_complex_test_suite(self):
        """Test loading a more complex test suite with tools and agents."""
        test_file = Path(__file__).parent / "fixtures" / "tests" / "complex_test_app.test.yaml"
        
        suite = load_test_suite(test_file)
        
        assert isinstance(suite, TestSuite)
        assert suite.name == "Complex Test App Test Suite"
        assert len(suite.cases) == 5
        assert 'tools' in suite.global_mocks
        assert len(suite.global_mocks['tools']) == 2
    
    def test_missing_required_fields_raises_error(self, tmp_path):
        """Test that missing required fields raise validation errors."""
        invalid_test = tmp_path / "invalid.test.yaml"
        invalid_test.write_text("""
        # Missing app_module field
        name: "Invalid Test"
        cases: []
        """)
        
        with pytest.raises(ValueError, match="must specify 'app_module'"):
            load_test_suite(invalid_test)
    
    def test_invalid_assertion_type_raises_error(self, tmp_path):
        """Test that invalid assertion types raise errors."""
        invalid_test = tmp_path / "invalid_assertion.test.yaml"
        invalid_test.write_text("""
        app_module: "app.ai"
        name: "Invalid Assertion Test"
        cases:
          - name: "test_case"
            target: {type: "prompt", name: "test"}
            assertions:
              - type: "invalid_type"
                value: "test"
        """)
        
        with pytest.raises(ValueError, match="Invalid assertion type"):
            load_test_suite(invalid_test)
    
    def test_file_not_found_raises_error(self):
        """Test that missing files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_test_suite("nonexistent.test.yaml")
    
    def test_invalid_yaml_raises_error(self, tmp_path):
        """Test that invalid YAML syntax raises YAMLError.""" 
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("""
        app_module: "app.ai"
        name: "Test"
        cases:
          - name: "test"
            target: {type: prompt, name: test  # Missing closing brace
        """)
        
        with pytest.raises(yaml.YAMLError):
            load_test_suite(invalid_yaml)


class TestTestCaseModel:
    """Test the TestCase data model and validation."""
    
    def test_test_case_creation(self):
        """Test creating a TestCase object."""
        assertions = [
            TestAssertion(type=AssertionType.CONTAINS, value="hello"),
            TestAssertion(type=AssertionType.HAS_KEYS, value=["field1", "field2"])
        ]
        
        case = TestCase(
            name="test_example",
            target={"type": "prompt", "name": "greeting"},
            inputs={"name": "Alice"},
            assertions=assertions,
            mocks={"llms": []},
            timeout_ms=5000,
            description="Example test case"
        )
        
        assert case.name == "test_example"
        assert case.target["type"] == "prompt"
        assert case.inputs["name"] == "Alice"
        assert len(case.assertions) == 2
        assert case.timeout_ms == 5000
    
    def test_test_assertion_types(self):
        """Test all supported assertion types can be created."""
        assertion_configs = [
            (AssertionType.EQUALS, "expected_value"),
            (AssertionType.NOT_EQUALS, "unexpected_value"),
            (AssertionType.CONTAINS, "substring"),
            (AssertionType.NOT_CONTAINS, "missing_substring"),
            (AssertionType.MATCHES, r"\\d+"),
            (AssertionType.NOT_MATCHES, r"[a-z]+"),
            (AssertionType.HAS_KEYS, ["key1", "key2"]),
            (AssertionType.MISSING_KEYS, ["bad_key"]),
            (AssertionType.HAS_LENGTH, 5),
            (AssertionType.TYPE_IS, "dict"),
            (AssertionType.JSON_PATH, "expected_value"),
            (AssertionType.FIELD_EXISTS, "field_name"),
            (AssertionType.FIELD_MISSING, "missing_field")
        ]
        
        for assertion_type, value in assertion_configs:
            assertion = TestAssertion(
                type=assertion_type,
                value=value,
                path="$.path" if assertion_type == AssertionType.JSON_PATH else None,
                description=f"Test {assertion_type.value} assertion"
            )
            
            assert assertion.type == assertion_type
            assert assertion.value == value


class TestMockSpecifications:
    """Test mock specification data models."""
    
    def test_mock_llm_spec_creation(self):
        """Test creating MockLLMSpec objects."""
        response = MockLLMResponse(
            output_text="Mock response",
            metadata={"mock": True},
            delay_ms=100
        )
        
        spec = MockLLMSpec(
            model_name="test_model",
            prompt_pattern=r"Hello.*",
            response=response
        )
        
        assert spec.model_name == "test_model"
        assert spec.prompt_pattern == r"Hello.*"
        assert spec.response.output_text == "Mock response"
        assert spec.response.delay_ms == 100
    
    def test_mock_tool_spec_creation(self):
        """Test creating MockToolSpec objects."""
        response = MockToolResponse(
            output={"status": "success", "data": [1, 2, 3]},
            success=True,
            metadata={"request_id": "123"}
        )
        
        spec = MockToolSpec(
            tool_name="api_tool",
            input_pattern={"endpoint": "/api/data"},
            response=response
        )
        
        assert spec.tool_name == "api_tool"
        assert spec.input_pattern["endpoint"] == "/api/data"
        assert spec.response.output["status"] == "success"
        assert spec.response.success is True
    
    def test_mock_llm_response_defaults(self):
        """Test MockLLMResponse default values."""
        response = MockLLMResponse(output_text="Test response")
        
        assert response.output_text == "Test response"
        assert response.metadata == {}
        assert response.delay_ms == 0
    
    def test_mock_tool_response_defaults(self):
        """Test MockToolResponse default values."""
        response = MockToolResponse(output="Test output")
        
        assert response.output == "Test output"
        assert response.success is True
        assert response.error is None
        assert response.metadata == {}


class TestTargetTypes:
    """Test target type enumeration and validation."""
    
    def test_target_type_values(self):
        """Test that all expected target types are available."""
        assert TargetType.PROMPT.value == "prompt"
        assert TargetType.AGENT.value == "agent"
        assert TargetType.CHAIN.value == "chain"
        assert TargetType.APP.value == "app"
    
    def test_target_type_from_string(self):
        """Test creating TargetType from string values."""
        assert TargetType("prompt") == TargetType.PROMPT
        assert TargetType("agent") == TargetType.AGENT
        assert TargetType("chain") == TargetType.CHAIN
        assert TargetType("app") == TargetType.APP
    
    def test_invalid_target_type_raises_error(self):
        """Test that invalid target types raise ValueError."""
        with pytest.raises(ValueError):
            TargetType("invalid_target")


class TestAssertionTypes:
    """Test assertion type enumeration and validation."""
    
    def test_assertion_type_values(self):
        """Test that all expected assertion types have correct values."""
        expected_types = {
            "equals", "not_equals", "contains", "not_contains",
            "matches", "not_matches", "has_keys", "missing_keys",
            "has_length", "type_is", "json_path", "field_exists", "field_missing"
        }
        
        actual_types = {t.value for t in AssertionType}
        assert actual_types == expected_types
    
    def test_assertion_type_from_string(self):
        """Test creating AssertionType from string values."""
        assert AssertionType("equals") == AssertionType.EQUALS
        assert AssertionType("contains") == AssertionType.CONTAINS
        assert AssertionType("json_path") == AssertionType.JSON_PATH
    
    def test_invalid_assertion_type_raises_error(self):
        """Test that invalid assertion types raise ValueError."""
        with pytest.raises(ValueError):
            AssertionType("invalid_assertion")