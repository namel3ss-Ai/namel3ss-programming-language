"""
Tests for namel3ss CLI test command integration.

This module tests the enhanced CLI test command functionality that provides
native test discovery, execution, and reporting for namel3ss applications.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from argparse import Namespace
import tempfile

from namel3ss.cli.commands.tools import cmd_test, _discover_test_files
from namel3ss.testing.runner import TestRunner, TestResult
from namel3ss.testing import TestSuite, TestCase, load_test_suite


class TestTestFileDiscovery:
    """Test the test file discovery functionality."""
    
    def test_discover_test_files_single_file(self, tmp_path):
        """Test discovering a single test file."""
        # Create a test file
        test_file = tmp_path / "app.test.yaml"
        test_file.write_text("""
        app_module: "app.ai"
        name: "Test Suite"
        cases: []
        """)
        
        files = _discover_test_files(str(test_file))
        
        assert len(files) == 1
        assert files[0] == str(test_file)
    
    def test_discover_test_files_directory(self, tmp_path):
        """Test discovering test files in a directory."""
        # Create multiple test files
        (tmp_path / "app1.test.yaml").write_text("app_module: app1.ai\nname: Test1\ncases: []")
        (tmp_path / "app2.test.yaml").write_text("app_module: app2.ai\nname: Test2\ncases: []")
        (tmp_path / "regular.yaml").write_text("not: a test file")
        (tmp_path / "app3.test.yml").write_text("app_module: app3.ai\nname: Test3\ncases: []")
        
        files = _discover_test_files(str(tmp_path))
        
        assert len(files) == 3
        file_names = [Path(f).name for f in files]
        assert "app1.test.yaml" in file_names
        assert "app2.test.yaml" in file_names
        assert "app3.test.yml" in file_names
        assert "regular.yaml" not in file_names
    
    def test_discover_test_files_recursive(self, tmp_path):
        """Test recursive discovery of test files."""
        # Create nested directory structure
        subdir = tmp_path / "tests" / "unit"
        subdir.mkdir(parents=True)
        
        (tmp_path / "root.test.yaml").write_text("app_module: root.ai\nname: Root\ncases: []")
        (subdir / "nested.test.yaml").write_text("app_module: nested.ai\nname: Nested\ncases: []")
        
        files = _discover_test_files(str(tmp_path))
        
        assert len(files) == 2
        file_names = [Path(f).name for f in files]
        assert "root.test.yaml" in file_names
        assert "nested.test.yaml" in file_names
    
    def test_discover_test_files_nonexistent_path(self):
        """Test discovery with nonexistent path raises error."""
        with pytest.raises(FileNotFoundError):
            _discover_test_files("/nonexistent/path")
    
    def test_discover_test_files_no_tests_found(self, tmp_path):
        """Test discovery with no test files found."""
        # Create regular files but no test files
        (tmp_path / "config.yaml").write_text("config: value")
        (tmp_path / "data.json").write_text("{}")
        
        files = _discover_test_files(str(tmp_path))
        
        assert len(files) == 0


class TestCLITestCommand:
    """Test the main CLI test command functionality."""
    
    @patch('namel3ss.cli.commands.tools.TestRunner')
    @patch('namel3ss.cli.commands.tools.load_test_suite')
    def test_cmd_test_single_file_success(self, mock_load_suite, mock_runner_class):
        """Test running tests on a single file with all tests passing."""
        # Mock test suite
        mock_suite = Mock(spec=TestSuite)
        mock_suite.name = "Test Suite"
        mock_load_suite.return_value = mock_suite
        
        # Mock successful test results
        mock_runner = Mock(spec=TestRunner)
        mock_runner.run_test_suite.return_value = [
            TestResult("test1", True, 100, "output1", 1, 1),
            TestResult("test2", True, 150, "output2", 1, 1)
        ]
        mock_runner_class.return_value = mock_runner
        
        # Create args
        args = Namespace(
            path="test.yaml",
            verbose=False,
            timeout=30000,
            filter=None,
            fail_fast=False,
            output_format="text"
        )
        
        # Capture stdout
        with patch('sys.stdout', new_callable=Mock) as mock_stdout:
            result = cmd_test(args)
        
        assert result == 0  # Success exit code
        mock_load_suite.assert_called_once_with("test.yaml")
        mock_runner.run_test_suite.assert_called_once_with(mock_suite)
    
    @patch('namel3ss.cli.commands.tools.TestRunner')
    @patch('namel3ss.cli.commands.tools.load_test_suite')
    def test_cmd_test_with_failures(self, mock_load_suite, mock_runner_class):
        """Test running tests with some failures."""
        # Mock test suite
        mock_suite = Mock(spec=TestSuite)
        mock_load_suite.return_value = mock_suite
        
        # Mock mixed test results
        mock_runner = Mock(spec=TestRunner)
        mock_runner.run_test_suite.return_value = [
            TestResult("test1", True, 100, "output1", 1, 1),
            TestResult("test2", False, 150, None, 0, 1, error="Assertion failed")
        ]
        mock_runner_class.return_value = mock_runner
        
        args = Namespace(
            path="test.yaml",
            verbose=False, 
            timeout=30000,
            filter=None,
            fail_fast=False,
            output_format="text"
        )
        
        result = cmd_test(args)
        
        assert result == 1  # Failure exit code
    
    @patch('namel3ss.cli.commands.tools._discover_test_files')
    @patch('namel3ss.cli.commands.tools.TestRunner')
    @patch('namel3ss.cli.commands.tools.load_test_suite')
    def test_cmd_test_directory_discovery(self, mock_load_suite, mock_runner_class, mock_discover):
        """Test running tests with directory discovery."""
        # Mock file discovery
        mock_discover.return_value = ["test1.yaml", "test2.yaml"]
        
        # Mock test suites
        mock_suite1 = Mock(spec=TestSuite)
        mock_suite1.name = "Suite 1"
        mock_suite2 = Mock(spec=TestSuite) 
        mock_suite2.name = "Suite 2"
        mock_load_suite.side_effect = [mock_suite1, mock_suite2]
        
        # Mock test results
        mock_runner = Mock(spec=TestRunner)
        mock_runner.run_test_suite.side_effect = [
            [TestResult("test1", True, 100, "output1", 1, 1)],
            [TestResult("test2", True, 150, "output2", 1, 1)]
        ]
        mock_runner_class.return_value = mock_runner
        
        args = Namespace(
            path="test_dir/",
            verbose=False,
            timeout=30000,
            filter=None,
            fail_fast=False,
            output_format="text"
        )
        
        result = cmd_test(args)
        
        assert result == 0
        assert mock_load_suite.call_count == 2
        assert mock_runner.run_test_suite.call_count == 2
    
    @patch('namel3ss.cli.commands.tools.TestRunner')
    @patch('namel3ss.cli.commands.tools.load_test_suite')
    def test_cmd_test_with_filter(self, mock_load_suite, mock_runner_class):
        """Test running tests with test name filter."""
        # Mock test suite with multiple cases
        mock_case1 = Mock(spec=TestCase)
        mock_case1.name = "test_login"
        mock_case2 = Mock(spec=TestCase)
        mock_case2.name = "test_logout"
        mock_case3 = Mock(spec=TestCase)
        mock_case3.name = "setup_database"
        
        mock_suite = Mock(spec=TestSuite)
        mock_suite.name = "Filtered Suite"
        mock_suite.cases = [mock_case1, mock_case2, mock_case3]
        mock_load_suite.return_value = mock_suite
        
        # Mock test runner
        mock_runner = Mock(spec=TestRunner)
        mock_runner_class.return_value = mock_runner
        
        args = Namespace(
            path="test.yaml",
            verbose=False,
            timeout=30000,
            filter="test_",  # Should match test_login and test_logout
            fail_fast=False,
            output_format="text"
        )
        
        cmd_test(args)
        
        # Verify that the test suite was filtered
        filtered_suite = mock_runner.run_test_suite.call_args[0][0]
        assert len(filtered_suite.cases) == 2
        assert all("test_" in case.name for case in filtered_suite.cases)
    
    @patch('namel3ss.cli.commands.tools.TestRunner')
    @patch('namel3ss.cli.commands.tools.load_test_suite')
    def test_cmd_test_verbose_output(self, mock_load_suite, mock_runner_class):
        """Test verbose output formatting."""
        mock_suite = Mock(spec=TestSuite)
        mock_suite.name = "Verbose Test Suite"
        mock_load_suite.return_value = mock_suite
        
        mock_runner = Mock(spec=TestRunner)
        mock_runner.run_test_suite.return_value = [
            TestResult("test_verbose", True, 250, "Detailed output", 2, 2)
        ]
        mock_runner_class.return_value = mock_runner
        
        args = Namespace(
            path="test.yaml",
            verbose=True,  # Enable verbose output
            timeout=30000,
            filter=None,
            fail_fast=False,
            output_format="text"
        )
        
        with patch('sys.stdout', new_callable=Mock) as mock_stdout:
            cmd_test(args)
        
        # Verify that TestRunner was created with verbose=True
        mock_runner_class.assert_called_once()
        call_kwargs = mock_runner_class.call_args[1]
        assert call_kwargs['verbose'] is True
    
    @patch('namel3ss.cli.commands.tools.TestRunner')
    @patch('namel3ss.cli.commands.tools.load_test_suite')
    def test_cmd_test_fail_fast(self, mock_load_suite, mock_runner_class):
        """Test fail-fast behavior stops on first failure."""
        mock_suite = Mock(spec=TestSuite)
        mock_load_suite.return_value = mock_suite
        
        # Mock test results with a failure
        mock_runner = Mock(spec=TestRunner)
        mock_runner.run_test_suite.return_value = [
            TestResult("test1", True, 100, "output1", 1, 1),
            TestResult("test2", False, 150, None, 0, 1, error="Failed"),
            TestResult("test3", True, 200, "output3", 1, 1)  # Should not be reached
        ]
        mock_runner_class.return_value = mock_runner
        
        args = Namespace(
            path="test.yaml",
            verbose=False,
            timeout=30000,
            filter=None,
            fail_fast=True,
            output_format="text"
        )
        
        result = cmd_test(args)
        
        assert result == 1  # Should exit with failure code
    
    @patch('namel3ss.cli.commands.tools.TestRunner')
    @patch('namel3ss.cli.commands.tools.load_test_suite')
    def test_cmd_test_custom_timeout(self, mock_load_suite, mock_runner_class):
        """Test custom timeout configuration."""
        mock_suite = Mock(spec=TestSuite)
        mock_load_suite.return_value = mock_suite
        
        mock_runner = Mock(spec=TestRunner)
        mock_runner.run_test_suite.return_value = []
        mock_runner_class.return_value = mock_runner
        
        args = Namespace(
            path="test.yaml",
            verbose=False,
            timeout=60000,  # Custom timeout
            filter=None,
            fail_fast=False,
            output_format="text"
        )
        
        cmd_test(args)
        
        # Verify TestRunner was created with custom timeout
        mock_runner_class.assert_called_once()
        call_kwargs = mock_runner_class.call_args[1]
        assert call_kwargs['timeout_ms'] == 60000
    
    @patch('namel3ss.cli.commands.tools.load_test_suite')
    def test_cmd_test_invalid_test_file(self, mock_load_suite):
        """Test handling invalid test file."""
        mock_load_suite.side_effect = ValueError("Invalid test file format")
        
        args = Namespace(
            path="invalid.yaml",
            verbose=False,
            timeout=30000,
            filter=None,
            fail_fast=False,
            output_format="text"
        )
        
        with patch('sys.stderr', new_callable=Mock):
            result = cmd_test(args)
        
        assert result == 1  # Should exit with error code
    
    @patch('namel3ss.cli.commands.tools._discover_test_files')
    def test_cmd_test_no_test_files_found(self, mock_discover):
        """Test behavior when no test files are found."""
        mock_discover.return_value = []
        
        args = Namespace(
            path="empty_dir/",
            verbose=False,
            timeout=30000,
            filter=None,
            fail_fast=False,
            output_format="text"
        )
        
        with patch('sys.stdout', new_callable=Mock) as mock_stdout:
            result = cmd_test(args)
        
        assert result == 0  # Should succeed but with no tests run
    
    @patch('namel3ss.cli.commands.tools.TestRunner')
    @patch('namel3ss.cli.commands.tools.load_test_suite')
    def test_cmd_test_json_output_format(self, mock_load_suite, mock_runner_class):
        """Test JSON output format."""
        mock_suite = Mock(spec=TestSuite)
        mock_suite.name = "JSON Test Suite"
        mock_load_suite.return_value = mock_suite
        
        mock_runner = Mock(spec=TestRunner)
        mock_runner.run_test_suite.return_value = [
            TestResult("test_json", True, 100, "output", 1, 1)
        ]
        mock_runner_class.return_value = mock_runner
        
        args = Namespace(
            path="test.yaml",
            verbose=False,
            timeout=30000,
            filter=None,
            fail_fast=False,
            output_format="json"
        )
        
        with patch('sys.stdout', new_callable=Mock) as mock_stdout:
            result = cmd_test(args)
        
        # Should output JSON format (implementation would format as JSON)
        assert result == 0
    
    def test_cmd_test_file_not_found(self):
        """Test handling when test file doesn't exist."""
        args = Namespace(
            path="nonexistent.yaml",
            verbose=False,
            timeout=30000,
            filter=None,
            fail_fast=False,
            output_format="text"
        )
        
        with patch('sys.stderr', new_callable=Mock):
            result = cmd_test(args)
        
        assert result == 1  # Should exit with error code


class TestTestOutputFormatting:
    """Test test result output formatting."""
    
    def test_format_test_results_summary(self):
        """Test formatting test results summary."""
        results = [
            TestResult("test1", True, 100, "output1", 1, 1),
            TestResult("test2", False, 150, None, 0, 1, error="Failed"),
            TestResult("test3", True, 200, "output3", 2, 2)
        ]
        
        # This would be tested with the actual formatting function
        # when it's implemented in the CLI module
        assert len(results) == 3
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        
        assert passed_count == 2
        assert failed_count == 1
    
    def test_format_verbose_output(self):
        """Test verbose output includes detailed information."""
        result = TestResult(
            "test_detailed", 
            True, 
            250, 
            "Detailed test output with multiple lines",
            3, 
            3
        )
        
        # Verbose output should include:
        # - Test name
        # - Execution time 
        # - Full output
        # - Assertion counts
        assert result.test_name == "test_detailed"
        assert result.execution_time_ms == 250
        assert result.output is not None
        assert result.assertions_passed == 3
        assert result.assertions_total == 3