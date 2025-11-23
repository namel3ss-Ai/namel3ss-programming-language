"""
Test specification schemas and data structures for namel3ss application testing.

This module defines the test specification DSL that allows developers to write
deterministic tests for their .ai applications, targeting prompts, agents, 
chains, and other namel3ss constructs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


class TargetType(Enum):
    """Types of testable targets in namel3ss applications."""
    PROMPT = "prompt"
    AGENT = "agent" 
    CHAIN = "chain"
    APP = "app"  # Full application test


class AssertionType(Enum):
    """Types of assertions supported in test specifications."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES = "matches"
    NOT_MATCHES = "not_matches"
    HAS_KEYS = "has_keys"
    MISSING_KEYS = "missing_keys"
    HAS_LENGTH = "has_length"
    TYPE_IS = "type_is"
    JSON_PATH = "json_path"
    FIELD_EXISTS = "field_exists"
    FIELD_MISSING = "field_missing"


@dataclass
class TestAssertion:
    """
    A single assertion to evaluate against test output.
    
    Examples:
        # Simple content assertion
        TestAssertion(type=AssertionType.CONTAINS, value="analysis complete")
        
        # JSON structure assertion  
        TestAssertion(type=AssertionType.HAS_KEYS, value=["sentiment", "topics"])
        
        # JSONPath assertion
        TestAssertion(type=AssertionType.JSON_PATH, path="$.sentiment", value="positive")
    """
    type: AssertionType
    value: Any
    path: Optional[str] = None  # For JSON path assertions
    description: Optional[str] = None
    

@dataclass 
class MockLLMResponse:
    """Configuration for a mock LLM response."""
    output_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    delay_ms: int = 0  # Simulated response delay


@dataclass
class MockLLMSpec:
    """
    Specification for mocking LLM provider responses.
    
    Maps LLM requests (model + prompt patterns) to deterministic responses.
    """
    model_name: str
    prompt_pattern: Optional[str] = None  # Regex pattern to match prompts
    response: Optional[MockLLMResponse] = None
    responses: Optional[List[MockLLMResponse]] = None  # For multi-turn agents
    

@dataclass
class MockToolResponse:
    """Configuration for a mock tool response."""
    output: Any
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class MockToolSpec:
    """
    Specification for mocking tool invocations.
    
    Maps tool calls (name + input patterns) to deterministic responses.
    """
    tool_name: str
    input_pattern: Optional[Dict[str, Any]] = None  # Pattern to match inputs
    response: MockToolResponse


@dataclass
class TestCase:
    """
    A single test case defining inputs, target execution, and expected outputs.
    
    Example YAML:
        name: "analyze positive content"
        target:
          type: prompt
          name: analyze_content
        inputs:
          content: "This is excellent work!"
        assertions:
          - type: contains
            value: "positive"
          - type: has_keys  
            value: ["sentiment", "topics"]
        mocks:
          llms:
            - model_name: content_analyzer_llm
              response:
                output_text: |
                  {
                    "topics": ["work", "feedback"],
                    "sentiment": "positive", 
                    "risk_level": "low",
                    "notes": "Positive feedback on work quality"
                  }
    """
    name: str
    target: Dict[str, Any]  # {type: "prompt", name: "analyze_content"}
    inputs: Dict[str, Any] = field(default_factory=dict)
    assertions: List[TestAssertion] = field(default_factory=list)
    mocks: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 30000
    description: Optional[str] = None


@dataclass
class TestSuite:
    """
    A collection of test cases targeting a specific .ai application or module.
    
    Example YAML:
        app_module: "examples/content-analyzer/app.ai"
        name: "Content Analyzer Test Suite"
        setup:
          fixtures_dir: "./fixtures"
        cases:
          - name: "analyze positive content"
            target: {type: prompt, name: analyze_content}
            inputs: {content: "Great work!"}
            assertions:
              - {type: contains, value: "positive"}
    """
    app_module: str  # Path to .ai file
    name: str
    cases: List[TestCase] = field(default_factory=list)
    setup: Dict[str, Any] = field(default_factory=dict)
    teardown: Dict[str, Any] = field(default_factory=dict)
    global_mocks: Dict[str, Any] = field(default_factory=dict)
    

def load_test_suite(test_file_path: Union[str, Path]) -> TestSuite:
    """
    Load a test suite from a YAML file.
    
    Args:
        test_file_path: Path to the test suite YAML file
        
    Returns:
        Parsed TestSuite object
        
    Raises:
        FileNotFoundError: If test file doesn't exist
        yaml.YAMLError: If test file has invalid YAML syntax
        ValueError: If test specification is invalid
        
    Example:
        >>> suite = load_test_suite("tests/content_analyzer.test.yaml")
        >>> len(suite.cases)
        3
    """
    test_path = Path(test_file_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_file_path}")
        
    with open(test_path, 'r', encoding='utf-8') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in test file {test_file_path}: {e}")
    
    return _parse_test_suite(data, test_path)


def _parse_test_suite(data: Dict[str, Any], source_path: Path) -> TestSuite:
    """Parse YAML data into TestSuite object with validation."""
    if not isinstance(data, dict):
        raise ValueError(f"Test suite must be a YAML object, got {type(data)}")
        
    # Required fields
    if 'app_module' not in data:
        raise ValueError("Test suite must specify 'app_module' field")
    if 'name' not in data:
        raise ValueError("Test suite must specify 'name' field")
        
    # Parse test cases
    cases = []
    for case_data in data.get('cases', []):
        cases.append(_parse_test_case(case_data))
    
    return TestSuite(
        app_module=data['app_module'],
        name=data['name'],
        cases=cases,
        setup=data.get('setup', {}),
        teardown=data.get('teardown', {}),
        global_mocks=data.get('global_mocks', {})
    )


def _parse_test_case(data: Dict[str, Any]) -> TestCase:
    """Parse YAML data into TestCase object with validation."""
    if not isinstance(data, dict):
        raise ValueError(f"Test case must be a YAML object, got {type(data)}")
        
    if 'name' not in data:
        raise ValueError("Test case must specify 'name' field")
    if 'target' not in data:
        raise ValueError("Test case must specify 'target' field")
        
    # Parse assertions
    assertions = []
    for assertion_data in data.get('assertions', []):
        assertions.append(_parse_assertion(assertion_data))
        
    return TestCase(
        name=data['name'],
        target=data['target'],
        inputs=data.get('inputs', {}),
        assertions=assertions,
        mocks=data.get('mocks', {}),
        timeout_ms=data.get('timeout_ms', 30000),
        description=data.get('description')
    )


def _parse_assertion(data: Dict[str, Any]) -> TestAssertion:
    """Parse YAML data into TestAssertion object with validation."""
    if not isinstance(data, dict):
        raise ValueError(f"Assertion must be a YAML object, got {type(data)}")
        
    if 'type' not in data:
        raise ValueError("Assertion must specify 'type' field")
        
    try:
        assertion_type = AssertionType(data['type'])
    except ValueError:
        valid_types = [t.value for t in AssertionType]
        raise ValueError(f"Invalid assertion type '{data['type']}'. Valid types: {valid_types}")
        
    return TestAssertion(
        type=assertion_type,
        value=data.get('value'),
        path=data.get('path'),
        description=data.get('description')
    )


__all__ = [
    "TargetType",
    "AssertionType", 
    "TestAssertion",
    "MockLLMResponse",
    "MockLLMSpec",
    "MockToolResponse", 
    "MockToolSpec",
    "TestCase",
    "TestSuite",
    "load_test_suite"
]