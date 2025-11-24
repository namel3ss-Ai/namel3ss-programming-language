"""
Meta-tests for the conformance test runner.

These tests validate the conformance test infrastructure itself,
ensuring the runner correctly executes tests, compares results,
and handles edge cases.
"""

import pytest
from pathlib import Path
import tempfile
import yaml

from namel3ss.conformance.models import (
    ConformanceTestDescriptor,
    TestPhase,
    TestStatus,
    ParseExpectation,
    discover_conformance_tests,
)
from namel3ss.conformance.runner import (
    ConformanceRunner,
    TestResult,
    PhaseResult,
)


class TestTestDescriptorLoading:
    """Test loading and validation of test descriptors."""
    
    def test_load_valid_descriptor(self, tmp_path):
        """Test loading a valid test descriptor."""
        descriptor_path = tmp_path / "test.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "test-001"
category: "parse"
name: "Test case"
description: "Test description"
phases:
  - parse
sources:
  - path: "test.ai"
expect:
  parse:
    status: "success"
""")
        
        descriptor = ConformanceTestDescriptor.from_file(descriptor_path)
        
        assert descriptor.spec_version == "1.0.0"
        assert descriptor.language_version == "1.0.0"
        assert descriptor.test_id == "test-001"
        assert descriptor.category == "parse"
        assert TestPhase.PARSE in descriptor.phases
        assert descriptor.expect.parse.status == TestStatus.SUCCESS
    
    def test_missing_required_fields(self, tmp_path):
        """Test that missing required fields raise errors."""
        descriptor_path = tmp_path / "invalid.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
# Missing required fields
""")
        
        with pytest.raises(KeyError):
            ConformanceTestDescriptor.from_file(descriptor_path)
    
    def test_invalid_phase(self, tmp_path):
        """Test that invalid phase names are rejected."""
        descriptor_path = tmp_path / "invalid.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "test-001"
category: "parse"
name: "Test"
phases:
  - invalid_phase
sources:
  - path: "test.ai"
expect:
  parse:
    status: "success"
""")
        
        with pytest.raises((ValueError, KeyError)):
            ConformanceTestDescriptor.from_file(descriptor_path)
    
    def test_inline_source_vs_file(self, tmp_path):
        """Test that inline source is properly handled."""
        descriptor_path = tmp_path / "inline.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "test-002"
category: "parse"
name: "Inline test"
phases:
  - parse
sources:
  - content: 'app "TestApp"'
expect:
  parse:
    status: "success"
""")
        
        descriptor = ConformanceTestDescriptor.from_file(descriptor_path)
        assert descriptor.sources[0].content == 'app "TestApp"'
        assert descriptor.sources[0].path is None


class TestTestDiscovery:
    """Test the test discovery mechanism."""
    
    def test_discover_tests_in_directory(self, tmp_path):
        """Test discovering all .test.yaml files."""
        # Create test directory structure
        parse_dir = tmp_path / "parse" / "valid"
        parse_dir.mkdir(parents=True)
        
        test1 = parse_dir / "test1.test.yaml"
        test1.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "test-001"
category: "parse"
name: "Test 1"
phases: [parse]
sources:
  - content: 'app "Test1"'
expect:
  parse:
    status: "success"
""")
        
        test2 = parse_dir / "test2.test.yaml"
        test2.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "test-002"
category: "parse"
name: "Test 2"
phases: [parse]
sources:
  - content: 'app "Test2"'
expect:
  parse:
    status: "success"
""")
        
        tests = discover_conformance_tests(tmp_path)
        assert len(tests) == 2
        assert {t.test_id for t in tests} == {"test-001", "test-002"}
    
    def test_filter_by_category(self, tmp_path):
        """Test filtering tests by category."""
        parse_dir = tmp_path / "parse"
        parse_dir.mkdir(parents=True)
        types_dir = tmp_path / "types"
        types_dir.mkdir(parents=True)
        
        (parse_dir / "test.test.yaml").write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "parse-001"
category: "parse"
name: "Parse test"
phases: [parse]
sources:
  - content: 'app "Test"'
expect:
  parse:
    status: "success"
""")
        
        (types_dir / "test.test.yaml").write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "types-001"
category: "types"
name: "Type test"
phases: [typecheck]
sources:
  - content: 'app "Test"'
expect:
  typecheck:
    status: "success"
""")
        
        all_tests = discover_conformance_tests(tmp_path)
        assert len(all_tests) == 2
        
        parse_tests = discover_conformance_tests(tmp_path, category="parse")
        assert len(parse_tests) == 1
        assert parse_tests[0].category == "parse"
    
    def test_filter_by_test_id(self, tmp_path):
        """Test filtering by specific test ID."""
        test_dir = tmp_path / "parse"
        test_dir.mkdir(parents=True)
        
        for i in range(1, 4):
            (test_dir / f"test{i}.test.yaml").write_text(f"""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "test-{i:03d}"
category: "parse"
name: "Test {i}"
phases: [parse]
sources:
  - content: 'app "Test{i}"'
expect:
  parse:
    status: "success"
""")
        
        filtered = discover_conformance_tests(tmp_path, test_id="test-002")
        assert len(filtered) == 1
        assert filtered[0].test_id == "test-002"


class TestParsePhaseExecution:
    """Test execution of parse phase tests."""
    
    def test_successful_parse(self, tmp_path):
        """Test that valid source parses successfully."""
        runner = ConformanceRunner()
        
        descriptor_path = tmp_path / "test.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "parse-success"
category: "parse"
name: "Successful parse"
phases: [parse]
sources:
  - content: 'app "ValidApp"'
expect:
  parse:
    status: "success"
""")
        
        descriptor = ConformanceTestDescriptor.from_file(descriptor_path)
        result = runner.run_test(descriptor)
        
        assert result.result == TestResult.PASS
        assert len(result.phase_results) == 1
        assert result.phase_results[0].phase == TestPhase.PARSE
        assert result.phase_results[0].result == TestResult.PASS
    
    def test_expected_parse_error(self, tmp_path):
        """Test that invalid source produces expected error."""
        runner = ConformanceRunner()
        
        descriptor_path = tmp_path / "test.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "parse-error"
category: "parse"
name: "Expected parse error"
phases: [parse]
sources:
  - content: 'app "Unclosed'
expect:
  parse:
    status: "error"
""")
        
        descriptor = ConformanceTestDescriptor.from_file(descriptor_path)
        result = runner.run_test(descriptor)
        
        assert result.result == TestResult.PASS  # Pass because error was expected
        assert result.phase_results[0].result == TestResult.PASS
    
    def test_unexpected_parse_error(self, tmp_path):
        """Test that unexpected parse errors are caught."""
        runner = ConformanceRunner()
        
        descriptor_path = tmp_path / "test.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "unexpected-error"
category: "parse"
name: "Unexpected error"
phases: [parse]
sources:
  - content: 'app "Unclosed'
expect:
  parse:
    status: "success"
""")
        
        descriptor = ConformanceTestDescriptor.from_file(descriptor_path)
        result = runner.run_test(descriptor)
        
        assert result.result == TestResult.FAIL
        assert result.phase_results[0].result == TestResult.FAIL
        assert "Unexpected parse error" in result.phase_results[0].message
    
    def test_unexpected_success(self, tmp_path):
        """Test detection when parsing succeeds but error was expected."""
        runner = ConformanceRunner()
        
        descriptor_path = tmp_path / "test.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "unexpected-success"
category: "parse"
name: "Unexpected success"
phases: [parse]
sources:
  - content: 'app "ValidApp"'
expect:
  parse:
    status: "error"
""")
        
        descriptor = ConformanceTestDescriptor.from_file(descriptor_path)
        result = runner.run_test(descriptor)
        
        assert result.result == TestResult.FAIL
        assert "Expected parse error but parsing succeeded" in result.phase_results[0].message


class TestBatchExecution:
    """Test batch execution of multiple tests."""
    
    def test_run_all_tests(self, tmp_path):
        """Test running all discovered tests."""
        test_dir = tmp_path / "parse"
        test_dir.mkdir(parents=True)
        
        # Create multiple test files
        for i, status in enumerate(["success", "error"], 1):
            (test_dir / f"test{i}.test.yaml").write_text(f"""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "test-{i:03d}"
category: "parse"
name: "Test {i}"
phases: [parse]
sources:
  - content: 'app "Test{i}"'
expect:
  parse:
    status: "{status}"
""")
        
        runner = ConformanceRunner()
        results = runner.run_all_tests(tmp_path)
        
        assert len(results) == 2
        # First should pass (valid syntax, expects success)
        assert results[0].result == TestResult.PASS
        # Second should fail (valid syntax, expects error)
        assert results[1].result == TestResult.FAIL
    
    def test_summary_statistics(self, tmp_path):
        """Test that summary statistics are calculated correctly."""
        test_dir = tmp_path / "parse"
        test_dir.mkdir(parents=True)
        
        # Create 3 passing tests and 1 failing test
        for i in range(1, 5):
            status = "success" if i <= 3 else "error"
            (test_dir / f"test{i}.test.yaml").write_text(f"""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "test-{i:03d}"
category: "parse"
name: "Test {i}"
phases: [parse]
sources:
  - content: 'app "Test{i}"'
expect:
  parse:
    status: "{status}"
""")
        
        runner = ConformanceRunner()
        results = runner.run_all_tests(tmp_path)
        
        passed = sum(1 for r in results if r.result == TestResult.PASS)
        failed = sum(1 for r in results if r.result == TestResult.FAIL)
        
        assert passed == 3
        assert failed == 1


class TestErrorHandling:
    """Test error handling in the conformance runner."""
    
    def test_missing_source_file(self, tmp_path):
        """Test handling of missing source file."""
        descriptor_path = tmp_path / "test.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "missing-file"
category: "parse"
name: "Missing file test"
phases: [parse]
sources:
  - path: "nonexistent.ai"
expect:
  parse:
    status: "success"
""")
        
        descriptor = ConformanceTestDescriptor.from_file(descriptor_path)
        runner = ConformanceRunner()
        result = runner.run_test(descriptor)
        
        # Should be an ERROR (infrastructure issue, not a test failure)
        assert result.result == TestResult.ERROR
    
    def test_malformed_yaml(self, tmp_path):
        """Test handling of malformed YAML."""
        descriptor_path = tmp_path / "bad.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
invalid: yaml: structure: [
""")
        
        with pytest.raises(yaml.YAMLError):
            ConformanceTestDescriptor.from_file(descriptor_path)


class TestResultFormatting:
    """Test result output formatting."""
    
    def test_human_readable_output(self, tmp_path):
        """Test human-readable output format."""
        runner = ConformanceRunner()
        
        descriptor_path = tmp_path / "test.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "format-test"
category: "parse"
name: "Format test"
phases: [parse]
sources:
  - content: 'app "Test"'
expect:
  parse:
    status: "success"
""")
        
        descriptor = ConformanceTestDescriptor.from_file(descriptor_path)
        result = runner.run_test(descriptor)
        
        # print_summary uses internal state, so we need to run tests first
        results = runner.run_all_tests(tmp_path)
        
        # Should be able to print summary without error
        try:
            runner.print_summary()
        except Exception as e:
            pytest.fail(f"print_summary raised exception: {e}")
    
    def test_json_output_format(self, tmp_path):
        """Test JSON output format via to_dict."""
        import json
        
        runner = ConformanceRunner()
        
        descriptor_path = tmp_path / "test.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "json-test"
category: "parse"
name: "JSON test"
phases: [parse]
sources:
  - content: 'app "Test"'
expect:
  parse:
    status: "success"
""")
        
        descriptor = ConformanceTestDescriptor.from_file(descriptor_path)
        result = runner.run_test(descriptor)
        
        # Convert to dict for JSON serialization
        result_dict = result.to_dict()
        
        # Should be valid JSON-serializable
        try:
            json_str = json.dumps(result_dict)
            data = json.loads(json_str)
            assert "test_id" in data
            assert data["test_id"] == "json-test"
            assert "result" in data
        except (TypeError, json.JSONDecodeError) as e:
            pytest.fail(f"Invalid JSON output: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
