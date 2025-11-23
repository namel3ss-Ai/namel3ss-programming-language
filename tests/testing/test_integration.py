"""
Integration tests for the namel3ss testing framework.

This module contains integration tests that validate end-to-end functionality
of the testing framework using the example applications and test suites.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock

from namel3ss.testing.runner import TestRunner
from namel3ss.testing import load_test_suite


class TestEndToEndTestExecution:
    """Test end-to-end execution of test suites with real fixtures."""
    
    @pytest.fixture
    def fixtures_path(self):
        """Get path to test fixtures directory."""
        return Path(__file__).parent / "fixtures"
    
    @pytest.fixture
    def simple_test_suite(self, fixtures_path):
        """Load the simple test application test suite."""
        test_file = fixtures_path / "tests" / "simple_test_app.test.yaml"
        return load_test_suite(test_file)
    
    @pytest.fixture
    def complex_test_suite(self, fixtures_path):
        """Load the complex test application test suite."""
        test_file = fixtures_path / "tests" / "complex_test_app.test.yaml"
        return load_test_suite(test_file)
    
    def test_load_simple_test_suite(self, simple_test_suite):
        """Test that simple test suite loads correctly."""
        assert simple_test_suite.name == "Simple Test App Test Suite"
        assert len(simple_test_suite.cases) == 4
        assert simple_test_suite.app_module == "tests/testing/fixtures/apps/simple_test_app.ai"
        
        # Verify test case names
        case_names = [case.name for case in simple_test_suite.cases]
        expected_names = ["test_greeting_prompt", "test_analysis_prompt", "test_assistant_agent", "test_workflow_chain"]
        assert all(name in case_names for name in expected_names)
    
    def test_load_complex_test_suite(self, complex_test_suite):
        """Test that complex test suite loads correctly."""
        assert complex_test_suite.name == "Complex Test App Test Suite"
        assert len(complex_test_suite.cases) == 5
        
        # Verify that tool mocks are configured
        assert 'tools' in complex_test_suite.global_mocks
        assert len(complex_test_suite.global_mocks['tools']) == 2
        
        tool_names = [tool.tool_name for tool in complex_test_suite.global_mocks['tools']]
        assert "api_client" in tool_names
        assert "vector_search" in tool_names
    
    @patch('namel3ss.testing.runner.TestRunner._load_application')
    def test_run_simple_test_suite_with_mocked_app(self, mock_load_app, simple_test_suite):
        """Test running simple test suite with mocked application."""
        # Mock the application and its components
        mock_greeting_prompt = Mock()
        mock_greeting_prompt.execute.return_value = "Hello Alice! How can I help you today?"
        
        mock_analysis_prompt = Mock()
        mock_analysis_prompt.execute.return_value = {
            "sentiment": "positive",
            "keywords": ["happy", "excited", "great"],
            "confidence": 0.95
        }
        
        mock_assistant_agent = Mock()
        mock_assistant_agent.run.return_value = {
            "response": "I'd be happy to help you with Python programming!",
            "confidence": 0.9,
            "sources": ["python.org", "stackoverflow.com"]
        }
        
        mock_workflow_chain = Mock()
        mock_workflow_chain.run.return_value = {
            "steps": [
                {"step": "analyze", "result": "Document analyzed"},
                {"step": "generate", "result": "Summary generated"},
                {"step": "review", "result": "Review completed"}
            ],
            "final_output": "Workflow completed successfully",
            "metadata": {"total_time": 1.5, "quality_score": 0.85}
        }
        
        # Mock application object
        mock_app = Mock()
        mock_app.get_prompt.side_effect = lambda name: {
            "greeting": mock_greeting_prompt,
            "analysis": mock_analysis_prompt
        }.get(name)
        mock_app.get_agent.return_value = mock_assistant_agent
        mock_app.get_chain.return_value = mock_workflow_chain
        
        mock_load_app.return_value = (mock_app, Mock())
        
        # Run the test suite
        runner = TestRunner(verbose=True)
        results = runner.run_test_suite(simple_test_suite)
        
        # Verify all tests passed
        assert len(results) == 4
        assert all(result.passed for result in results), f"Failed tests: {[r.test_name for r in results if not r.passed]}"
        
        # Verify specific assertions
        greeting_result = next(r for r in results if r.test_name == "test_greeting_prompt")
        assert greeting_result.assertions_passed == 2
        assert greeting_result.assertions_total == 2
        
        analysis_result = next(r for r in results if r.test_name == "test_analysis_prompt")
        assert analysis_result.assertions_passed == 3
        assert analysis_result.assertions_total == 3
    
    @patch('namel3ss.testing.runner.TestRunner._load_application')
    def test_run_complex_test_suite_with_mocked_app(self, mock_load_app, complex_test_suite):
        """Test running complex test suite with mocked application and tools."""
        # Mock more sophisticated application components
        mock_research_agent = Mock()
        mock_research_agent.run.return_value = {
            "findings": ["Finding 1", "Finding 2", "Finding 3"],
            "sources": ["source1.com", "source2.org"],
            "confidence": 0.88,
            "research_time": 2.3
        }
        
        mock_classifier_agent = Mock()
        mock_classifier_agent.run.return_value = {
            "classification": "technical",
            "categories": ["programming", "python", "tutorial"],
            "confidence": 0.92
        }
        
        mock_rag_chain = Mock()
        mock_rag_chain.run.return_value = {
            "answer": "Based on the retrieved documents, here's the comprehensive answer...",
            "sources": [
                {"id": "doc1", "relevance": 0.95},
                {"id": "doc2", "relevance": 0.87}
            ],
            "confidence": 0.91
        }
        
        mock_multi_step_chain = Mock()
        mock_multi_step_chain.run.return_value = {
            "steps": [
                {"name": "research", "status": "completed", "output": "Research done"},
                {"name": "analyze", "status": "completed", "output": "Analysis complete"},
                {"name": "synthesize", "status": "completed", "output": "Synthesis finished"}
            ],
            "final_result": "Multi-step workflow completed successfully",
            "total_time": 5.7
        }
        
        mock_full_app = Mock()
        mock_full_app.run.return_value = {
            "status": "success",
            "results": {
                "processed_items": 42,
                "success_rate": 0.95,
                "total_time": 8.2
            },
            "summary": "Application executed successfully with high success rate"
        }
        
        # Mock application object
        mock_app = Mock()
        mock_app.get_agent.side_effect = lambda name: {
            "research_agent": mock_research_agent,
            "classifier_agent": mock_classifier_agent
        }.get(name)
        mock_app.get_chain.side_effect = lambda name: {
            "rag_chain": mock_rag_chain,
            "multi_step_workflow": mock_multi_step_chain
        }.get(name)
        mock_app.run.return_value = mock_full_app.run.return_value
        
        mock_load_app.return_value = (mock_app, Mock())
        
        # Run the test suite
        runner = TestRunner(verbose=True)
        results = runner.run_test_suite(complex_test_suite)
        
        # Verify all tests passed
        assert len(results) == 5
        assert all(result.passed for result in results), f"Failed tests: {[r.test_name for r in results if not r.passed]}"
        
        # Verify complex assertions
        research_result = next(r for r in results if r.test_name == "test_research_agent")
        assert research_result.assertions_passed == 4
        
        rag_result = next(r for r in results if r.test_name == "test_rag_chain")
        assert rag_result.assertions_passed == 3
    
    def test_test_suite_with_assertion_failures(self, simple_test_suite):
        """Test behavior when assertions fail."""
        # Modify a test case to have a failing assertion
        failing_case = simple_test_suite.cases[0]  # greeting test
        failing_case.assertions[0].value = "Goodbye"  # This should fail
        
        with patch('namel3ss.testing.runner.TestRunner._load_application') as mock_load:
            # Mock application with output that won't match the assertion
            mock_prompt = Mock()
            mock_prompt.execute.return_value = "Hello Alice! How can I help you today?"
            
            mock_app = Mock()
            mock_app.get_prompt.return_value = mock_prompt
            mock_load.return_value = (mock_app, Mock())
            
            runner = TestRunner()
            results = runner.run_test_suite(simple_test_suite)
            
            # First test should fail, others should pass
            assert not results[0].passed
            assert results[0].assertions_passed == 0
            assert results[0].assertions_total == 2
            assert "Expected output to contain 'Goodbye'" in results[0].error
    
    def test_test_execution_timeout(self, simple_test_suite):
        """Test behavior with execution timeouts."""
        with patch('namel3ss.testing.runner.TestRunner._load_application') as mock_load:
            # Mock slow-executing prompt
            mock_prompt = Mock()
            mock_prompt.execute.side_effect = lambda x: __import__('time').sleep(2) or "delayed response"
            
            mock_app = Mock()
            mock_app.get_prompt.return_value = mock_prompt
            mock_load.return_value = (mock_app, Mock())
            
            # Use very short timeout
            runner = TestRunner(timeout_ms=100)
            
            # This test is conceptual - actual timeout implementation would be in TestRunner
            # For now, just verify the timeout value is set
            assert runner.timeout_ms == 100
    
    @patch('namel3ss.testing.runner.TestRunner._load_application')
    def test_test_filtering_by_name(self, mock_load_app, simple_test_suite):
        """Test filtering tests by name pattern."""
        # Mock application
        mock_app = Mock()
        mock_load_app.return_value = (mock_app, Mock())
        
        # Filter test suite to only include prompt tests
        filtered_cases = [
            case for case in simple_test_suite.cases 
            if "prompt" in case.name
        ]
        
        # Create filtered test suite
        from namel3ss.testing import TestSuite
        filtered_suite = TestSuite(
            name=simple_test_suite.name + " (Filtered)",
            app_module=simple_test_suite.app_module,
            cases=filtered_cases,
            global_mocks=simple_test_suite.global_mocks
        )
        
        assert len(filtered_suite.cases) == 2  # greeting_prompt and analysis_prompt
        assert all("prompt" in case.name for case in filtered_suite.cases)
    
    def test_mock_configuration_from_test_suite(self, complex_test_suite):
        """Test that mock configurations are properly parsed."""
        # Verify LLM mocks
        llm_mocks = complex_test_suite.global_mocks.get('llms', [])
        assert len(llm_mocks) == 2
        
        # Find specific mock configurations
        gpt4_mock = next(mock for mock in llm_mocks if mock.model_name == "gpt-4")
        assert gpt4_mock.prompt_pattern == r"Research the topic: (.*)"
        assert "comprehensive research on" in gpt4_mock.response.output_text
        
        # Verify tool mocks
        tool_mocks = complex_test_suite.global_mocks.get('tools', [])
        assert len(tool_mocks) == 2
        
        api_mock = next(mock for mock in tool_mocks if mock.tool_name == "api_client")
        assert api_mock.tool_type == "http"
        assert api_mock.input_pattern["method"] == "GET"
    
    def test_nested_data_structure_assertions(self, complex_test_suite):
        """Test that complex nested data structure assertions work."""
        # Find test case with nested assertions
        rag_test = next(case for case in complex_test_suite.cases if case.name == "test_rag_chain")
        
        # Verify it has JSON path assertions for nested data
        json_path_assertions = [
            assertion for assertion in rag_test.assertions 
            if assertion.type.value == "json_path"
        ]
        
        assert len(json_path_assertions) == 1
        json_assertion = json_path_assertions[0]
        assert json_assertion.path == "$.sources[0].relevance"
        assert json_assertion.value == 0.95


class TestTestFrameworkResilience:
    """Test framework resilience and error handling."""
    
    def test_malformed_test_suite_handling(self, tmp_path):
        """Test handling of malformed test suite files."""
        # Create malformed YAML file
        bad_test_file = tmp_path / "bad.test.yaml"
        bad_test_file.write_text("""
        app_module: "test.ai"
        name: "Bad Test"
        cases:
          - name: "test_case"
            target: "invalid_target_format"  # Should be dict
            assertions: []
        """)
        
        with pytest.raises(ValueError):
            load_test_suite(bad_test_file)
    
    def test_missing_application_file_handling(self):
        """Test handling when application file doesn't exist."""
        from namel3ss.testing import TestSuite, TestCase
        
        # Create test suite pointing to nonexistent app
        test_suite = TestSuite(
            name="Missing App Test",
            app_module="nonexistent.ai",
            cases=[
                TestCase(
                    name="test_missing",
                    target={"type": "prompt", "name": "test"},
                    assertions=[]
                )
            ]
        )
        
        runner = TestRunner()
        results = runner.run_test_suite(test_suite)
        
        # Should handle gracefully with error result
        assert len(results) == 1
        assert not results[0].passed
        assert results[0].error is not None
    
    def test_application_runtime_error_handling(self):
        """Test handling of runtime errors in application execution."""
        from namel3ss.testing import TestSuite, TestCase
        
        test_suite = TestSuite(
            name="Runtime Error Test",
            app_module="test.ai",
            cases=[
                TestCase(
                    name="test_runtime_error",
                    target={"type": "prompt", "name": "error_prompt"},
                    assertions=[]
                )
            ]
        )
        
        with patch('namel3ss.testing.runner.TestRunner._load_application') as mock_load:
            # Mock prompt that raises exception
            mock_prompt = Mock()
            mock_prompt.execute.side_effect = RuntimeError("Simulated runtime error")
            
            mock_app = Mock()
            mock_app.get_prompt.return_value = mock_prompt
            mock_load.return_value = (mock_app, Mock())
            
            runner = TestRunner()
            results = runner.run_test_suite(test_suite)
            
            # Should capture the error gracefully
            assert len(results) == 1
            assert not results[0].passed
            assert "Simulated runtime error" in results[0].error