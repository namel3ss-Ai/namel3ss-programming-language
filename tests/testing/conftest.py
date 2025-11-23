"""
Pytest configuration and fixtures for namel3ss testing framework tests.

This module provides shared fixtures and configuration for testing the
namel3ss testing framework itself.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from namel3ss.testing import (
    TestSuite, TestCase, TestAssertion, AssertionType, TargetType,
    MockLLMSpec, MockLLMResponse, MockToolSpec, MockToolResponse
)
from namel3ss.testing.mocks import MockLLMProvider
from namel3ss.testing.tools import MockToolRegistry, MockHttpTool


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_test_case():
    """Create a sample test case for testing."""
    return TestCase(
        name="sample_test",
        target={"type": "prompt", "name": "greeting"},
        inputs={"name": "Alice"},
        assertions=[
            TestAssertion(type=AssertionType.CONTAINS, value="Hello"),
            TestAssertion(type=AssertionType.HAS_LENGTH, value=5)
        ],
        timeout_ms=5000,
        description="Sample test case for testing framework"
    )


@pytest.fixture
def sample_test_suite(sample_test_case):
    """Create a sample test suite for testing."""
    return TestSuite(
        name="Sample Test Suite",
        app_module="sample_app.ai",
        cases=[sample_test_case],
        global_mocks={
            'llms': [
                MockLLMSpec(
                    model_name="test_model",
                    prompt_pattern="Hello.*",
                    response=MockLLMResponse(output_text="Hello Alice!")
                )
            ],
            'tools': []
        },
        description="Sample test suite for testing framework"
    )


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider for testing."""
    responses = {
        "Hello": "Hi there!",
        r"My name is (\w+)": "Nice to meet you, {1}!",
        "What is AI?": "AI stands for Artificial Intelligence."
    }
    return MockLLMProvider("test_model", responses, fallback="Default response")


@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry for testing."""
    registry = MockToolRegistry()
    
    # Add HTTP tool mock
    http_tool = MockHttpTool("http_client")
    http_tool.add_response(
        input_pattern={"method": "GET", "url": "https://api.example.com/data"},
        output={"status": 200, "data": {"message": "success"}},
        success=True
    )
    registry.register_tool(http_tool)
    
    return registry


@pytest.fixture
def mock_application():
    """Create a mock namel3ss application for testing."""
    app = Mock()
    
    # Mock prompts
    greeting_prompt = Mock()
    greeting_prompt.execute.return_value = "Hello Alice! How can I help you today?"
    
    analysis_prompt = Mock()
    analysis_prompt.execute.return_value = {
        "sentiment": "positive",
        "confidence": 0.95,
        "keywords": ["hello", "help", "today"]
    }
    
    app.get_prompt.side_effect = lambda name: {
        "greeting": greeting_prompt,
        "analysis": analysis_prompt
    }.get(name)
    
    # Mock agents
    assistant_agent = Mock()
    assistant_agent.run.return_value = {
        "response": "I'd be happy to help you!",
        "confidence": 0.9
    }
    
    app.get_agent.return_value = assistant_agent
    
    # Mock chains
    workflow_chain = Mock()
    workflow_chain.run.return_value = {
        "steps": ["step1", "step2", "step3"],
        "result": "Workflow completed"
    }
    
    app.get_chain.return_value = workflow_chain
    
    # Mock full app run
    app.run.return_value = {
        "status": "completed",
        "results": {"processed": 10, "success_rate": 1.0}
    }
    
    return app


@pytest.fixture
def mock_ast():
    """Create a mock AST for testing."""
    ast = Mock()
    ast.type = "application"
    ast.name = "test_app"
    ast.prompts = []
    ast.agents = []
    ast.chains = []
    return ast


@pytest.fixture
def sample_assertion_test_cases():
    """Create test cases for different assertion types."""
    return [
        # Equals assertion
        {
            "assertion": TestAssertion(type=AssertionType.EQUALS, value="expected"),
            "pass_data": "expected",
            "fail_data": "actual"
        },
        # Contains assertion
        {
            "assertion": TestAssertion(type=AssertionType.CONTAINS, value="hello"),
            "pass_data": "hello world",
            "fail_data": "goodbye world"
        },
        # Has keys assertion
        {
            "assertion": TestAssertion(type=AssertionType.HAS_KEYS, value=["name", "age"]),
            "pass_data": {"name": "Alice", "age": 30, "city": "NYC"},
            "fail_data": {"name": "Alice", "city": "NYC"}
        },
        # Has length assertion
        {
            "assertion": TestAssertion(type=AssertionType.HAS_LENGTH, value=3),
            "pass_data": [1, 2, 3],
            "fail_data": [1, 2]
        },
        # Type assertion
        {
            "assertion": TestAssertion(type=AssertionType.TYPE_IS, value="dict"),
            "pass_data": {"key": "value"},
            "fail_data": "string"
        },
        # Regex match assertion
        {
            "assertion": TestAssertion(type=AssertionType.MATCHES, value=r"\d+"),
            "pass_data": "123",
            "fail_data": "abc"
        },
        # Field exists assertion
        {
            "assertion": TestAssertion(type=AssertionType.FIELD_EXISTS, value="name"),
            "pass_data": {"name": "Alice", "age": 30},
            "fail_data": {"age": 30, "city": "NYC"}
        },
        # JSON path assertion
        {
            "assertion": TestAssertion(
                type=AssertionType.JSON_PATH,
                value="Alice",
                path="$.user.name"
            ),
            "pass_data": {"user": {"name": "Alice", "age": 30}},
            "fail_data": {"user": {"name": "Bob", "age": 30}}
        }
    ]


@pytest.fixture
def sample_yaml_test_content():
    """Create sample YAML test file content."""
    return """
app_module: "sample_app.ai"
name: "Sample Test Suite"
description: "Test suite for demonstration"

global_mocks:
  llms:
    - model_name: "gpt-4"
      prompt_pattern: "Hello.*"
      response:
        output_text: "Hi there!"
        delay_ms: 100
    - model_name: "claude"
      prompt_pattern: r"Analyze: (.*)"
      response:
        output_text: "Analysis of {1}: positive sentiment"
        metadata:
          confidence: 0.95

  tools:
    - tool_name: "api_client"
      tool_type: "http"
      input_pattern:
        method: "GET"
        url: "https://api.example.com/data"
      response:
        output:
          status_code: 200
          data: {"result": "success"}
        success: true

cases:
  - name: "test_greeting_prompt"
    description: "Test the greeting prompt functionality"
    target:
      type: "prompt"
      name: "greeting"
    inputs:
      name: "Alice"
    assertions:
      - type: "contains"
        value: "Hello"
        description: "Response should contain greeting"
      - type: "has_length"
        value: 5
        description: "Response should have reasonable length"
    timeout_ms: 5000

  - name: "test_analysis_chain"
    description: "Test the analysis chain"
    target:
      type: "chain"
      name: "analysis_workflow"
    inputs:
      text: "Sample text for analysis"
    assertions:
      - type: "has_keys"
        value: ["sentiment", "confidence"]
        description: "Result should have analysis fields"
      - type: "json_path"
        path: "$.sentiment"
        value: "positive"
        description: "Should detect positive sentiment"
    """


@pytest.fixture
def complex_test_data():
    """Create complex nested test data for assertion testing."""
    return {
        "user": {
            "id": 123,
            "name": "Alice Smith",
            "profile": {
                "email": "alice@example.com",
                "preferences": {
                    "theme": "dark",
                    "notifications": True
                }
            },
            "posts": [
                {"id": 1, "title": "Hello World", "likes": 42},
                {"id": 2, "title": "Testing 123", "likes": 18}
            ]
        },
        "metadata": {
            "timestamp": "2024-01-15T10:30:00Z",
            "version": "1.0.0",
            "status": "active"
        }
    }


@pytest.fixture(autouse=True)
def reset_mocks():
    """Automatically reset all mocks after each test."""
    yield
    # Any cleanup code would go here


@pytest.fixture
def patch_namel3ss_loader():
    """Patch namel3ss loader components for testing."""
    with patch('namel3ss.loader.ApplicationLoader') as mock_loader_class, \
         patch('namel3ss.resolver.Resolver') as mock_resolver_class, \
         patch('namel3ss.types.checker.TypeChecker') as mock_checker_class:
        
        yield {
            'loader': mock_loader_class,
            'resolver': mock_resolver_class,
            'checker': mock_checker_class
        }


# Test markers for categorizing tests
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their location."""
    for item in items:
        # Mark integration tests
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests (integration tests are typically slow)
        if "integration" in item.nodeid or "e2e" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Everything else is a unit test
        else:
            item.add_marker(pytest.mark.unit)