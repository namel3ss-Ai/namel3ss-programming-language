"""
Tests for conformance test discovery and filtering.
"""

import pytest
from pathlib import Path

from namel3ss.conformance.models import (
    discover_conformance_tests,
    ConformanceTestDescriptor,
)


class TestTestDiscovery:
    """Test discovery of conformance tests."""
    
    def test_discover_in_nested_structure(self, tmp_path):
        """Test discovering tests in nested directory structure."""
        # Create nested structure
        (tmp_path / "parse" / "valid").mkdir(parents=True)
        (tmp_path / "parse" / "invalid").mkdir(parents=True)
        (tmp_path / "types" / "valid").mkdir(parents=True)
        
        # Create test files
        (tmp_path / "parse" / "valid" / "test1.test.yaml").write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "parse-valid-001"
category: "parse"
name: "Test 1"
phases: [parse]
sources:
  - content: 'app "Test1"'
expect:
  parse:
    status: "success"
""")
        
        (tmp_path / "parse" / "invalid" / "test2.test.yaml").write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "parse-invalid-001"
category: "parse"
name: "Test 2"
phases: [parse]
sources:
  - content: 'invalid syntax'
expect:
  parse:
    status: "error"
""")
        
        (tmp_path / "types" / "valid" / "test3.test.yaml").write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "types-valid-001"
category: "types"
name: "Test 3"
phases: [typecheck]
sources:
  - content: 'app "Test3"'
expect:
  typecheck:
    status: "success"
""")
        
        tests = discover_conformance_tests(tmp_path)
        
        assert len(tests) == 3
        test_ids = {t.test_id for t in tests}
        assert test_ids == {"parse-valid-001", "parse-invalid-001", "types-valid-001"}
    
    def test_discover_ignores_non_test_files(self, tmp_path):
        """Test that non-.test.yaml files are ignored."""
        test_dir = tmp_path / "parse"
        test_dir.mkdir(parents=True)
        
        # Create test file
        (test_dir / "valid.test.yaml").write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "test-001"
category: "parse"
name: "Valid test"
phases: [parse]
sources:
  - content: 'app "Test"'
expect:
  parse:
    status: "success"
""")
        
        # Create non-test files that should be ignored
        (test_dir / "README.md").write_text("Documentation")
        (test_dir / "fixture.ai").write_text('app "Fixture"')
        (test_dir / "data.yaml").write_text("not: a test")
        
        tests = discover_conformance_tests(tmp_path)
        
        assert len(tests) == 1
        assert tests[0].test_id == "test-001"
    
    def test_discover_empty_directory(self, tmp_path):
        """Test discovering in empty directory."""
        tests = discover_conformance_tests(tmp_path)
        
        assert len(tests) == 0
    
    def test_discover_with_category_filter(self, tmp_path):
        """Test category filtering during discovery."""
        (tmp_path / "parse").mkdir(parents=True)
        (tmp_path / "types").mkdir(parents=True)
        
        # Create parse test
        (tmp_path / "parse" / "test1.test.yaml").write_text("""
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
        
        # Create types test
        (tmp_path / "types" / "test2.test.yaml").write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "types-001"
category: "types"
name: "Types test"
phases: [typecheck]
sources:
  - content: 'app "Test"'
expect:
  typecheck:
    status: "success"
""")
        
        # Filter by parse category
        parse_tests = discover_conformance_tests(tmp_path, category="parse")
        assert len(parse_tests) == 1
        assert parse_tests[0].category == "parse"
        
        # Filter by types category
        types_tests = discover_conformance_tests(tmp_path, category="types")
        assert len(types_tests) == 1
        assert types_tests[0].category == "types"
    
    def test_discover_with_test_id_filter(self, tmp_path):
        """Test test_id filtering during discovery."""
        test_dir = tmp_path / "parse"
        test_dir.mkdir(parents=True)
        
        # Create multiple tests
        for i in range(1, 4):
            (test_dir / f"test{i}.test.yaml").write_text(f"""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "parse-{i:03d}"
category: "parse"
name: "Test {i}"
phases: [parse]
sources:
  - content: 'app "Test{i}"'
expect:
  parse:
    status: "success"
""")
        
        # Filter by specific test ID
        filtered = discover_conformance_tests(tmp_path, test_id="parse-002")
        assert len(filtered) == 1
        assert filtered[0].test_id == "parse-002"
        
        # No match
        no_match = discover_conformance_tests(tmp_path, test_id="nonexistent")
        assert len(no_match) == 0
    
    def test_discover_handles_malformed_yaml(self, tmp_path):
        """Test that malformed YAML files are handled gracefully."""
        test_dir = tmp_path / "parse"
        test_dir.mkdir(parents=True)
        
        # Create valid test
        (test_dir / "valid.test.yaml").write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "valid-001"
category: "parse"
name: "Valid test"
phases: [parse]
sources:
  - content: 'app "Test"'
expect:
  parse:
    status: "success"
""")
        
        # Create malformed YAML
        (test_dir / "malformed.test.yaml").write_text("""
invalid: yaml: [[[
""")
        
        # Discovery should continue despite malformed file
        tests = discover_conformance_tests(tmp_path)
        
        # Should find at least the valid test
        assert len(tests) >= 1
        assert any(t.test_id == "valid-001" for t in tests)


class TestTestFiltering:
    """Test filtering of discovered tests."""
    
    def test_filter_by_tags(self, tmp_path):
        """Test filtering tests by tags."""
        test_dir = tmp_path / "parse"
        test_dir.mkdir(parents=True)
        
        # Create tests with different tags
        (test_dir / "test1.test.yaml").write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "test-001"
category: "parse"
name: "Test 1"
tags: ["basic", "syntax"]
phases: [parse]
sources:
  - content: 'app "Test1"'
expect:
  parse:
    status: "success"
""")
        
        (test_dir / "test2.test.yaml").write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "test-002"
category: "parse"
name: "Test 2"
tags: ["advanced", "features"]
phases: [parse]
sources:
  - content: 'app "Test2"'
expect:
  parse:
    status: "success"
""")
        
        all_tests = discover_conformance_tests(tmp_path)
        assert len(all_tests) == 2
        
        # Filter by tag
        basic_tests = [t for t in all_tests if "basic" in t.tags]
        assert len(basic_tests) == 1
        assert basic_tests[0].test_id == "test-001"
    
    def test_filter_by_language_version(self, tmp_path):
        """Test filtering by language version."""
        test_dir = tmp_path / "parse"
        test_dir.mkdir(parents=True)
        
        # Create tests for different language versions
        (test_dir / "test1.test.yaml").write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "test-v1"
category: "parse"
name: "V1 Test"
phases: [parse]
sources:
  - content: 'app "Test"'
expect:
  parse:
    status: "success"
""")
        
        (test_dir / "test2.test.yaml").write_text("""
spec_version: "1.0.0"
language_version: "1.1.0"
test_id: "test-v1.1"
category: "parse"
name: "V1.1 Test"
phases: [parse]
sources:
  - content: 'app "Test"'
expect:
  parse:
    status: "success"
""")
        
        all_tests = discover_conformance_tests(tmp_path)
        
        # Filter by language version
        v1_tests = [t for t in all_tests if t.language_version == "1.0.0"]
        assert len(v1_tests) == 1
        assert v1_tests[0].test_id == "test-v1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
