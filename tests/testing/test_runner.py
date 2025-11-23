"""
Tests for namel3ss test runner core functionality.

This module tests the core test execution engine that integrates with the
namel3ss parser, resolver, typechecker, and runtime to execute tests.
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

from namel3ss.testing.runner import TestRunner, TestResult, TestExecutionError
from namel3ss.testing import TestSuite, TestCase, TestAssertion, AssertionType, TargetType
from namel3ss.testing.mocks import MockLLMProvider
from namel3ss.testing.tools import MockToolRegistry, MockHttpTool


class TestTestResult:
    """Test the TestResult data structure."""
    
    def test_test_result_creation(self):
        """Test creating a TestResult object."""
        result = TestResult(
            test_name="test_example",
            passed=True,
            execution_time_ms=150,
            output="Success output",
            assertions_passed=3,
            assertions_total=3
        )
        
        assert result.test_name == "test_example"
        assert result.passed is True
        assert result.execution_time_ms == 150
        assert result.output == "Success output"
        assert result.assertions_passed == 3
        assert result.assertions_total == 3
        assert result.error is None
    
    def test_test_result_with_error(self):
        """Test TestResult with error information."""
        result = TestResult(
            test_name="test_failed",
            passed=False,
            execution_time_ms=75,
            output=None,
            assertions_passed=1,
            assertions_total=3,
            error="Assertion failed: expected 'hello' but got 'world'"
        )
        
        assert result.passed is False
        assert result.error is not None
        assert "Assertion failed" in result.error


class TestTestRunner:
    """Test the TestRunner implementation."""
    
    def test_test_runner_creation(self):
        """Test creating a TestRunner instance."""
        runner = TestRunner(verbose=True, timeout_ms=10000)
        
        assert runner.verbose is True
        assert runner.timeout_ms == 10000
    
    def test_test_runner_default_values(self):
        """Test TestRunner default values."""
        runner = TestRunner()
        
        assert runner.verbose is False
        assert runner.timeout_ms == 30000
    
    @patch('namel3ss.loader.ApplicationLoader')
    @patch('namel3ss.resolver.Resolver')  
    @patch('namel3ss.types.checker.TypeChecker')
    def test_load_application_success(self, mock_checker, mock_resolver, mock_loader):
        """Test successfully loading an application."""
        # Mock the application loading pipeline
        mock_app = Mock()
        mock_ast = Mock()
        mock_resolved_ast = Mock()
        mock_checked_ast = Mock()
        
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = (mock_app, mock_ast)
        mock_loader.return_value = mock_loader_instance
        
        mock_resolver_instance = Mock()
        mock_resolver_instance.resolve.return_value = mock_resolved_ast
        mock_resolver.return_value = mock_resolver_instance
        
        mock_checker_instance = Mock()
        mock_checker_instance.check.return_value = mock_checked_ast
        mock_checker.return_value = mock_checker_instance
        
        runner = TestRunner()
        app, ast = runner._load_application("test_app.ai")
        
        assert app is mock_app
        assert ast is mock_checked_ast
        mock_loader_instance.load.assert_called_once_with("test_app.ai")
        mock_resolver_instance.resolve.assert_called_once_with(mock_ast)
        mock_checker_instance.check.assert_called_once_with(mock_resolved_ast)
    
    @patch('namel3ss.loader.ApplicationLoader')
    def test_load_application_file_not_found(self, mock_loader):
        """Test loading application with missing file."""
        mock_loader_instance = Mock()
        mock_loader_instance.load.side_effect = FileNotFoundError("File not found")
        mock_loader.return_value = mock_loader_instance
        
        runner = TestRunner()
        
        with pytest.raises(TestExecutionError, match="Failed to load application"):
            runner._load_application("nonexistent.ai")
    
    def test_evaluate_equals_assertion(self):
        """Test evaluating equals assertion."""
        runner = TestRunner()
        assertion = TestAssertion(type=AssertionType.EQUALS, value="expected")
        
        # Should pass
        result = runner._evaluate_assertion(assertion, "expected")
        assert result.passed is True
        assert result.message == "Value equals 'expected'"
        
        # Should fail
        result = runner._evaluate_assertion(assertion, "actual")
        assert result.passed is False
        assert "Expected 'expected' but got 'actual'" in result.message
    
    def test_evaluate_contains_assertion(self):
        """Test evaluating contains assertion."""
        runner = TestRunner()
        assertion = TestAssertion(type=AssertionType.CONTAINS, value="hello")
        
        # Should pass
        result = runner._evaluate_assertion(assertion, "hello world")
        assert result.passed is True
        
        # Should fail
        result = runner._evaluate_assertion(assertion, "goodbye world")
        assert result.passed is False
        assert "Expected output to contain 'hello'" in result.message
    
    def test_evaluate_has_keys_assertion(self):
        """Test evaluating has_keys assertion."""
        runner = TestRunner()
        assertion = TestAssertion(type=AssertionType.HAS_KEYS, value=["name", "age"])
        
        # Should pass
        result = runner._evaluate_assertion(assertion, {"name": "Alice", "age": 30, "city": "NYC"})
        assert result.passed is True
        
        # Should fail
        result = runner._evaluate_assertion(assertion, {"name": "Alice"})
        assert result.passed is False
        assert "Missing keys: ['age']" in result.message
    
    def test_evaluate_has_length_assertion(self):
        """Test evaluating has_length assertion."""
        runner = TestRunner()
        assertion = TestAssertion(type=AssertionType.HAS_LENGTH, value=3)
        
        # Should pass
        result = runner._evaluate_assertion(assertion, [1, 2, 3])
        assert result.passed is True
        
        # Should fail
        result = runner._evaluate_assertion(assertion, [1, 2])
        assert result.passed is False
        assert "Expected length 3 but got 2" in result.message
    
    def test_evaluate_type_is_assertion(self):
        """Test evaluating type_is assertion."""
        runner = TestRunner()
        assertion = TestAssertion(type=AssertionType.TYPE_IS, value="dict")
        
        # Should pass
        result = runner._evaluate_assertion(assertion, {"key": "value"})
        assert result.passed is True
        
        # Should fail 
        result = runner._evaluate_assertion(assertion, "string")
        assert result.passed is False
        assert "Expected type 'dict' but got 'str'" in result.message
    
    def test_evaluate_json_path_assertion(self):
        """Test evaluating json_path assertion."""
        runner = TestRunner()
        assertion = TestAssertion(
            type=AssertionType.JSON_PATH,
            value="Alice",
            path="$.user.name"
        )
        
        data = {"user": {"name": "Alice", "age": 30}}
        
        # Should pass
        result = runner._evaluate_assertion(assertion, data)
        assert result.passed is True
        
        # Should fail with wrong value
        assertion_wrong = TestAssertion(
            type=AssertionType.JSON_PATH,
            value="Bob", 
            path="$.user.name"
        )
        result = runner._evaluate_assertion(assertion_wrong, data)
        assert result.passed is False
    
    def test_evaluate_field_exists_assertion(self):
        """Test evaluating field_exists assertion."""
        runner = TestRunner()
        assertion = TestAssertion(type=AssertionType.FIELD_EXISTS, value="name")
        
        # Should pass
        result = runner._evaluate_assertion(assertion, {"name": "Alice"})
        assert result.passed is True
        
        # Should fail
        result = runner._evaluate_assertion(assertion, {"age": 30})
        assert result.passed is False
        assert "Field 'name' does not exist" in result.message
    
    def test_evaluate_matches_assertion(self):
        """Test evaluating regex matches assertion."""
        runner = TestRunner()
        assertion = TestAssertion(type=AssertionType.MATCHES, value=r"\d+")
        
        # Should pass
        result = runner._evaluate_assertion(assertion, "123")
        assert result.passed is True
        
        # Should fail
        result = runner._evaluate_assertion(assertion, "abc")
        assert result.passed is False
        assert "Output does not match pattern" in result.message
    
    def test_evaluate_assertion_missing_path_for_json_path(self):
        """Test that json_path assertion without path raises error."""
        runner = TestRunner()
        assertion = TestAssertion(type=AssertionType.JSON_PATH, value="test")
        
        with pytest.raises(ValueError, match="JSON_PATH assertion requires 'path'"):
            runner._evaluate_assertion(assertion, {"data": "test"})
    
    @patch.object(TestRunner, '_load_application')
    @patch.object(TestRunner, '_execute_target')
    def test_run_single_test_case_success(self, mock_execute, mock_load):
        """Test running a single test case successfully."""
        # Setup mocks
        mock_app = Mock()
        mock_ast = Mock()
        mock_load.return_value = (mock_app, mock_ast)
        mock_execute.return_value = "Hello World"
        
        # Create test case
        test_case = TestCase(
            name="test_hello",
            target={"type": "prompt", "name": "greeting"},
            inputs={"name": "Alice"},
            assertions=[
                TestAssertion(type=AssertionType.CONTAINS, value="Hello")
            ]
        )
        
        runner = TestRunner()
        result = runner._run_test_case(test_case, "test_app.ai")
        
        assert result.test_name == "test_hello"
        assert result.passed is True
        assert result.assertions_passed == 1
        assert result.assertions_total == 1
        assert result.output == "Hello World"
    
    @patch.object(TestRunner, '_load_application')
    @patch.object(TestRunner, '_execute_target')
    def test_run_single_test_case_assertion_failure(self, mock_execute, mock_load):
        """Test running a test case with assertion failure."""
        # Setup mocks
        mock_load.return_value = (Mock(), Mock())
        mock_execute.return_value = "Goodbye World"
        
        # Create test case with failing assertion
        test_case = TestCase(
            name="test_hello",
            target={"type": "prompt", "name": "greeting"},
            assertions=[
                TestAssertion(type=AssertionType.CONTAINS, value="Hello")
            ]
        )
        
        runner = TestRunner()
        result = runner._run_test_case(test_case, "test_app.ai")
        
        assert result.passed is False
        assert result.assertions_passed == 0
        assert result.assertions_total == 1
        assert "Expected output to contain 'Hello'" in result.error
    
    @patch.object(TestRunner, '_load_application')
    @patch.object(TestRunner, '_execute_target')
    def test_run_test_case_execution_error(self, mock_execute, mock_load):
        """Test handling execution errors during test case run."""
        mock_load.return_value = (Mock(), Mock())
        mock_execute.side_effect = Exception("Execution failed")
        
        test_case = TestCase(
            name="test_error",
            target={"type": "prompt", "name": "test"},
            assertions=[]
        )
        
        runner = TestRunner()
        result = runner._run_test_case(test_case, "test_app.ai")
        
        assert result.passed is False
        assert "Execution failed" in result.error
    
    @patch.object(TestRunner, '_run_test_case')
    def test_run_test_suite_all_pass(self, mock_run_case):
        """Test running a test suite where all cases pass."""
        # Mock successful test results
        mock_run_case.side_effect = [
            TestResult("test1", True, 100, "output1", 1, 1),
            TestResult("test2", True, 150, "output2", 2, 2)
        ]
        
        # Create test suite
        test_suite = TestSuite(
            name="Test Suite",
            app_module="app.ai",
            cases=[
                TestCase("test1", {"type": "prompt", "name": "p1"}, assertions=[]),
                TestCase("test2", {"type": "prompt", "name": "p2"}, assertions=[])
            ]
        )
        
        runner = TestRunner()
        results = runner.run_test_suite(test_suite)
        
        assert len(results) == 2
        assert all(r.passed for r in results)
        assert mock_run_case.call_count == 2
    
    @patch.object(TestRunner, '_run_test_case')
    def test_run_test_suite_with_failures(self, mock_run_case):
        """Test running a test suite with some failures."""
        # Mock mixed results
        mock_run_case.side_effect = [
            TestResult("test1", True, 100, "output1", 1, 1),
            TestResult("test2", False, 150, None, 0, 1, error="Failed")
        ]
        
        test_suite = TestSuite(
            name="Test Suite", 
            app_module="app.ai",
            cases=[
                TestCase("test1", {"type": "prompt", "name": "p1"}, assertions=[]),
                TestCase("test2", {"type": "prompt", "name": "p2"}, assertions=[])
            ]
        )
        
        runner = TestRunner()
        results = runner.run_test_suite(test_suite)
        
        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False


class TestTestExecutionError:
    """Test the TestExecutionError exception class."""
    
    def test_test_execution_error_creation(self):
        """Test creating a TestExecutionError."""
        error = TestExecutionError("Test execution failed", "test_name")
        
        assert str(error) == "Test execution failed"
        assert error.test_name == "test_name"
    
    def test_test_execution_error_no_test_name(self):
        """Test TestExecutionError without test name."""
        error = TestExecutionError("General error")
        
        assert str(error) == "General error"
        assert error.test_name is None


class TestTargetExecution:
    """Test executing different target types."""
    
    @patch.object(TestRunner, '_load_application')
    def test_execute_prompt_target(self, mock_load):
        """Test executing a prompt target."""
        # Setup mock application with prompt
        mock_prompt = Mock()
        mock_prompt.execute.return_value = "Hello World"
        
        mock_app = Mock()
        mock_app.get_prompt.return_value = mock_prompt
        
        mock_load.return_value = (mock_app, Mock())
        
        runner = TestRunner()
        result = runner._execute_target(
            {"type": "prompt", "name": "greeting"},
            {"name": "Alice"},
            "app.ai"
        )
        
        assert result == "Hello World"
        mock_app.get_prompt.assert_called_once_with("greeting")
        mock_prompt.execute.assert_called_once_with({"name": "Alice"})
    
    @patch.object(TestRunner, '_load_application')
    def test_execute_agent_target(self, mock_load):
        """Test executing an agent target.""" 
        # Setup mock application with agent
        mock_agent = Mock()
        mock_agent.run.return_value = {"response": "Agent response"}
        
        mock_app = Mock()
        mock_app.get_agent.return_value = mock_agent
        
        mock_load.return_value = (mock_app, Mock())
        
        runner = TestRunner()
        result = runner._execute_target(
            {"type": "agent", "name": "assistant"},
            {"query": "Help me"},
            "app.ai"
        )
        
        assert result == {"response": "Agent response"}
        mock_app.get_agent.assert_called_once_with("assistant")
    
    @patch.object(TestRunner, '_load_application')
    def test_execute_chain_target(self, mock_load):
        """Test executing a chain target."""
        # Setup mock application with chain
        mock_chain = Mock()
        mock_chain.run.return_value = ["step1", "step2", "final"]
        
        mock_app = Mock()
        mock_app.get_chain.return_value = mock_chain
        
        mock_load.return_value = (mock_app, Mock())
        
        runner = TestRunner()
        result = runner._execute_target(
            {"type": "chain", "name": "workflow"},
            {"input": "data"},
            "app.ai"
        )
        
        assert result == ["step1", "step2", "final"]
        mock_app.get_chain.assert_called_once_with("workflow")
    
    @patch.object(TestRunner, '_load_application')
    def test_execute_app_target(self, mock_load):
        """Test executing an app target."""
        # Setup mock application 
        mock_app = Mock()
        mock_app.run.return_value = "App result"
        
        mock_load.return_value = (mock_app, Mock())
        
        runner = TestRunner()
        result = runner._execute_target(
            {"type": "app"},
            {"config": "value"},
            "app.ai"
        )
        
        assert result == "App result"
        mock_app.run.assert_called_once_with({"config": "value"})
    
    @patch.object(TestRunner, '_load_application')
    def test_execute_unknown_target_type_raises_error(self, mock_load):
        """Test that unknown target type raises error."""
        mock_load.return_value = (Mock(), Mock())
        
        runner = TestRunner()
        
        with pytest.raises(ValueError, match="Unknown target type"):
            runner._execute_target(
                {"type": "unknown_type"},
                {},
                "app.ai"
            )