"""
pytest integration for namel3ss application testing.

This module provides helpers and fixtures that enable teams to run namel3ss
application tests from within their existing pytest test suites, allowing
seamless integration with Python-based CI/CD pipelines.
"""

from __future__ import annotations

import asyncio
import pytest
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from namel3ss.testing import load_test_suite, TestSuite, TestCase
from namel3ss.testing.runner import TestRunner, TestResult, TestSuiteResult


class NamelessTestPlugin:
    """
    pytest plugin for namel3ss application testing.
    
    Automatically discovers and runs namel3ss test suites within pytest,
    converting test results to proper pytest outcomes.
    """
    
    def __init__(self):
        """Initialize the pytest plugin."""
        self.collected_suites: List[TestSuite] = []
        self.runner = TestRunner()
    
    def pytest_collect_file(self, parent, path):
        """
        Collect namel3ss test files for pytest execution.
        
        Args:
            parent: Parent collector
            path: Path to potential test file
            
        Returns:
            NamelessTestFile collector if this is a namel3ss test file
        """
        # Check if this is a namel3ss test file
        if path.ext in ['.yaml', '.yml'] and '.test.' in path.basename:
            return NamelessTestFile.from_parent(parent, fspath=path)
        return None


class NamelessTestFile(pytest.File):
    """
    pytest File collector for namel3ss test suite files.
    
    Represents a single .test.yaml file containing namel3ss test cases.
    """
    
    def collect(self):
        """
        Collect individual test cases from the namel3ss test suite.
        
        Yields:
            NamelessTestItem for each test case in the suite
        """
        try:
            suite = load_test_suite(self.fspath)
            for case in suite.cases:
                yield NamelessTestItem.from_parent(
                    self, 
                    name=case.name,
                    suite=suite,
                    case=case
                )
        except Exception as e:
            # Create a single failing test item for the load error
            yield NamelessTestItem.from_parent(
                self,
                name="load_error",
                suite=None,
                case=None,
                load_error=str(e)
            )


class NamelessTestItem(pytest.Item):
    """
    pytest Item for individual namel3ss test cases.
    
    Represents a single test case within a namel3ss test suite.
    """
    
    def __init__(self, name, parent, suite=None, case=None, load_error=None):
        """
        Initialize test item.
        
        Args:
            name: Test case name
            parent: Parent collector
            suite: TestSuite containing this case
            case: TestCase to execute
            load_error: Error message if suite failed to load
        """
        super().__init__(name, parent)
        self.suite = suite
        self.case = case
        self.load_error = load_error
    
    def runtest(self):
        """
        Execute the namel3ss test case.
        
        Raises:
            NamelessTestFailure: If test case fails
            Exception: If test execution encounters errors
        """
        if self.load_error:
            raise Exception(f"Failed to load test suite: {self.load_error}")
        
        if not self.suite or not self.case:
            raise Exception("Test case not properly initialized")
        
        # Run the test case
        runner = TestRunner()
        result = asyncio.run(self._run_single_case(runner))
        
        if not result.success:
            if result.error:
                raise NamelessTestFailure(f"Test execution failed: {result.error}")
            else:
                # Format assertion failures
                failed_assertions = [ar for ar in result.assertion_results if not ar.passed]
                if failed_assertions:
                    failure_messages = []
                    for assertion_result in failed_assertions:
                        if assertion_result.error:
                            failure_messages.append(f"Assertion error: {assertion_result.error}")
                        else:
                            failure_messages.append(
                                f"Assertion failed: {assertion_result.assertion.type.value} "
                                f"(expected: {assertion_result.assertion.value}, "
                                f"actual: {assertion_result.actual_value})"
                            )
                    raise NamelessTestFailure("\\n".join(failure_messages))
                else:
                    raise NamelessTestFailure("Test failed for unknown reason")
    
    async def _run_single_case(self, runner: TestRunner) -> TestResult:
        """
        Run a single test case using the namel3ss test runner.
        
        Args:
            runner: TestRunner instance
            
        Returns:
            TestResult from case execution
        """
        # Load the application
        app = await runner._load_application(self.suite.app_module)
        
        # Setup global mocks
        runner._setup_global_mocks(self.suite.global_mocks)
        
        # Run the specific case
        return await runner._run_test_case(self.case, app)
    
    def repr_failure(self, excinfo):
        """
        Format test failure information for pytest output.
        
        Args:
            excinfo: Exception information
            
        Returns:
            Formatted failure representation
        """
        if isinstance(excinfo.value, NamelessTestFailure):
            return str(excinfo.value)
        else:
            return super().repr_failure(excinfo)


class NamelessTestFailure(Exception):
    """Exception raised when a namel3ss test case fails."""
    pass


# pytest fixtures for namel3ss testing


@pytest.fixture
def namel3ss_test_runner():
    """
    Provide a configured TestRunner instance for pytest tests.
    
    Returns:
        TestRunner instance ready for test execution
        
    Example:
        def test_my_application(namel3ss_test_runner):
            suite = load_test_suite("tests/my_app.test.yaml")
            result = asyncio.run(namel3ss_test_runner.run_test_suite(suite))
            assert result.failed_cases == 0
    """
    return TestRunner(verbose=False)


@pytest.fixture
def namel3ss_mock_setup(namel3ss_test_runner):
    """
    Provide helper functions for setting up namel3ss mocks in pytest.
    
    Args:
        namel3ss_test_runner: TestRunner fixture
        
    Returns:
        MockSetup helper object
        
    Example:
        def test_prompt_with_mocks(namel3ss_mock_setup):
            namel3ss_mock_setup.add_llm_mock(
                model="gpt-4",
                response="This is a test response"
            )
            # Run test that uses the mocked LLM
    """
    return MockSetup(namel3ss_test_runner)


class MockSetup:
    """
    Helper class for setting up mocks in pytest tests.
    
    Provides convenient methods for configuring LLM and tool mocks
    within individual pytest test functions.
    """
    
    def __init__(self, runner: TestRunner):
        """
        Initialize mock setup helper.
        
        Args:
            runner: TestRunner instance to configure
        """
        self.runner = runner
        
    def add_llm_mock(
        self, 
        model: str, 
        response: str, 
        prompt_pattern: Optional[str] = None,
        priority: int = 0
    ) -> None:
        """
        Add a mock LLM response for testing.
        
        Args:
            model: LLM model name to mock
            response: Response text to return
            prompt_pattern: Optional regex pattern to match prompts
            priority: Priority for response matching
            
        Example:
            mock_setup.add_llm_mock(
                model="content_analyzer_llm",
                response='{"sentiment": "positive", "topics": ["work"]}'
            )
        """
        from namel3ss.testing.mocks import MockLLMProvider
        
        if model not in self.runner.mock_llm_registry:
            self.runner.mock_llm_registry[model] = MockLLMProvider()
        
        mock_llm = self.runner.mock_llm_registry[model]
        mock_llm.add_response_mapping(
            model=model,
            prompt_pattern=prompt_pattern,
            response=response,
            priority=priority
        )
    
    def add_tool_mock(
        self, 
        tool_name: str, 
        response: Any, 
        input_pattern: Optional[Dict[str, Any]] = None,
        success: bool = True
    ) -> None:
        """
        Add a mock tool response for testing.
        
        Args:
            tool_name: Tool name to mock
            response: Response data to return
            input_pattern: Optional pattern to match tool inputs
            success: Whether the mock should indicate success
            
        Example:
            mock_setup.add_tool_mock(
                tool_name="weather_api",
                response={"temperature": 72, "condition": "sunny"},
                input_pattern={"city": "San Francisco"}
            )
        """
        from namel3ss.testing.tools import MockToolRegistry
        from namel3ss.testing import MockToolResponse
        
        if self.runner.mock_tool_registry is None:
            self.runner.mock_tool_registry = MockToolRegistry()
        
        mock_response = MockToolResponse(
            output=response,
            success=success
        )
        
        self.runner.mock_tool_registry.register_mock(
            tool_name=tool_name,
            input_pattern=input_pattern,
            response=mock_response,
            priority=100  # High priority for test-specific mocks
        )
    
    def clear_mocks(self) -> None:
        """Clear all configured mocks for clean test state."""
        self.runner.mock_llm_registry.clear()
        if self.runner.mock_tool_registry:
            self.runner.mock_tool_registry.clear_history()


# Utility functions for direct pytest integration


def run_namel3ss_test_suite(test_file: Union[str, Path], **kwargs) -> TestSuiteResult:
    """
    Run a namel3ss test suite directly from pytest.
    
    Args:
        test_file: Path to test suite file
        **kwargs: Additional arguments for TestRunner
        
    Returns:
        TestSuiteResult with execution details
        
    Example:
        def test_content_analyzer():
            result = run_namel3ss_test_suite("tests/content_analyzer.test.yaml")
            assert result.failed_cases == 0
            assert result.passed_cases == 3
    """
    suite = load_test_suite(test_file)
    runner = TestRunner(**kwargs)
    return asyncio.run(runner.run_test_suite(suite))


def run_namel3ss_test_case(
    app_module: Union[str, Path],
    case_config: Dict[str, Any],
    **kwargs
) -> TestResult:
    """
    Run a single namel3ss test case directly from pytest.
    
    Args:
        app_module: Path to .ai application file
        case_config: Test case configuration dict
        **kwargs: Additional arguments for TestRunner
        
    Returns:
        TestResult from case execution
        
    Example:
        def test_single_prompt():
            result = run_namel3ss_test_case(
                app_module="examples/content-analyzer/app.ai",
                case_config={
                    "name": "test_positive_analysis",
                    "target": {"type": "prompt", "name": "analyze_content"},
                    "inputs": {"content": "Great work!"},
                    "assertions": [{"type": "contains", "value": "positive"}],
                    "mocks": {
                        "llms": [{
                            "model_name": "content_analyzer_llm",
                            "response": {"output_text": "positive sentiment detected"}
                        }]
                    }
                }
            )
            assert result.success
    """
    from namel3ss.testing import TestCase, TestAssertion, AssertionType
    
    # Convert dict config to TestCase object
    assertions = []
    for assertion_config in case_config.get('assertions', []):
        assertions.append(TestAssertion(
            type=AssertionType(assertion_config['type']),
            value=assertion_config['value'],
            path=assertion_config.get('path'),
            description=assertion_config.get('description')
        ))
    
    case = TestCase(
        name=case_config['name'],
        target=case_config['target'],
        inputs=case_config.get('inputs', {}),
        assertions=assertions,
        mocks=case_config.get('mocks', {}),
        timeout_ms=case_config.get('timeout_ms', 30000),
        description=case_config.get('description')
    )
    
    # Run the case
    runner = TestRunner(**kwargs)
    app = asyncio.run(runner._load_application(str(app_module)))
    return asyncio.run(runner._run_test_case(case, app))


def parametrize_namel3ss_tests(test_suite_file: Union[str, Path]):
    """
    pytest parametrization decorator for namel3ss test suites.
    
    Automatically parametrizes a pytest function with all test cases
    from a namel3ss test suite file.
    
    Args:
        test_suite_file: Path to test suite file
        
    Returns:
        pytest.mark.parametrize decorator
        
    Example:
        @parametrize_namel3ss_tests("tests/content_analyzer.test.yaml")
        def test_content_analyzer_cases(case_name, case_config, app_module):
            result = run_namel3ss_test_case(app_module, case_config)
            assert result.success, f"Test case '{case_name}' failed"
    """
    suite = load_test_suite(test_suite_file)
    
    # Create parameter list
    case_params = []
    for case in suite.cases:
        case_config = {
            'name': case.name,
            'target': case.target,
            'inputs': case.inputs,
            'assertions': [
                {
                    'type': assertion.type.value,
                    'value': assertion.value,
                    'path': assertion.path,
                    'description': assertion.description
                }
                for assertion in case.assertions
            ],
            'mocks': case.mocks,
            'timeout_ms': case.timeout_ms,
            'description': case.description
        }
        case_params.append((case.name, case_config, suite.app_module))
    
    return pytest.mark.parametrize(
        "case_name,case_config,app_module",
        case_params,
        ids=[case.name for case in suite.cases]
    )


__all__ = [
    "NamelessTestPlugin",
    "NamelessTestFile", 
    "NamelessTestItem",
    "NamelessTestFailure",
    "MockSetup",
    "run_namel3ss_test_suite",
    "run_namel3ss_test_case", 
    "parametrize_namel3ss_tests"
]