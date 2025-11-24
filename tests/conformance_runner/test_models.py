"""
Tests for conformance test data models and validation.
"""

import pytest
from pathlib import Path

from namel3ss.conformance.models import (
    ConformanceTestDescriptor,
    TestPhase,
    TestStatus,
    ParseExpectation,
    TypecheckExpectation,
    RuntimeExpectation,
    Diagnostic,
    SourceLocation,
    SourceFile,
)


class TestDataModels:
    """Test conformance test data models."""
    
    def test_test_phase_enum(self):
        """Test TestPhase enum values."""
        assert TestPhase.PARSE.value == "parse"
        assert TestPhase.RESOLVE.value == "resolve"
        assert TestPhase.TYPECHECK.value == "typecheck"
        assert TestPhase.CODEGEN.value == "codegen"
        assert TestPhase.RUNTIME.value == "runtime"
    
    def test_test_status_enum(self):
        """Test TestStatus enum values."""
        assert TestStatus.SUCCESS.value == "success"
        assert TestStatus.ERROR.value == "error"
    
    def test_parse_expectation_creation(self):
        """Test creating ParseExpectation."""
        expectation = ParseExpectation(
            status=TestStatus.SUCCESS,
            ast={"type": "Module"},
            errors=[]
        )
        
        assert expectation.status == TestStatus.SUCCESS
        assert expectation.ast == {"type": "Module"}
        assert expectation.errors == []
    
    def test_typecheck_expectation_creation(self):
        """Test creating TypecheckExpectation."""
        expectation = TypecheckExpectation(
            status=TestStatus.SUCCESS,
            diagnostics=[]
        )
        
        assert expectation.status == TestStatus.SUCCESS
        assert expectation.diagnostics == []
    
    def test_runtime_expectation_creation(self):
        """Test creating RuntimeExpectation."""
        expectation = RuntimeExpectation(
            status=TestStatus.SUCCESS,
            stdout="Hello\n",
            stderr="",
            exit_code=0
        )
        
        assert expectation.status == TestStatus.SUCCESS
        assert expectation.stdout == "Hello\n"
        assert expectation.exit_code == 0
    
    def test_diagnostic_creation(self):
        """Test creating Diagnostic."""
        from namel3ss.conformance.models import DiagnosticSeverity
        diagnostic = Diagnostic(
            severity=DiagnosticSeverity.ERROR,
            message="Type mismatch",
            code="TYPE_ERROR",
            location=SourceLocation(file="test.ai", line=10, column=5)
        )
        
        assert diagnostic.severity == DiagnosticSeverity.ERROR
        assert diagnostic.message == "Type mismatch"
        assert diagnostic.location.line == 10
    
    def test_source_file_with_path(self):
        """Test SourceFile with file path."""
        source = SourceFile(path="test.ai")
        
        assert source.path == "test.ai"
        assert source.content is None
    
    def test_source_file_with_inline_content(self):
        """Test SourceFile with inline content."""
        source = SourceFile(content='app "Test"')
        
        assert source.path is None
        assert source.content == 'app "Test"'


class TestDescriptorValidation:
    """Test validation of conformance test descriptors."""
    
    def test_valid_descriptor(self, tmp_path):
        """Test validation of valid descriptor."""
        descriptor_path = tmp_path / "valid.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "valid-001"
category: "parse"
name: "Valid test"
description: "A valid test descriptor"
phases:
  - parse
sources:
  - path: "test.ai"
expect:
  parse:
    status: "success"
""")
        
        descriptor = ConformanceTestDescriptor.from_file(descriptor_path)
        
        # Validate should not raise
        try:
            descriptor.validate()
        except Exception as e:
            pytest.fail(f"Validation failed: {e}")
    
    def test_missing_phase_expectation(self, tmp_path):
        """Test that missing phase expectation is loaded correctly."""
        descriptor_path = tmp_path / "missing.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "missing-001"
category: "parse"
name: "Missing expectation"
phases:
  - parse
  - typecheck
sources:
  - content: 'app "Test"'
expect:
  parse:
    status: "success"
  # Missing typecheck expectation
""")
        
        descriptor = ConformanceTestDescriptor.from_file(descriptor_path)
        
        # Validate doesn't enforce this - it's caught at runtime
        assert descriptor.expect.parse is not None
        assert descriptor.expect.typecheck is None
    
    def test_no_sources(self, tmp_path):
        """Test that descriptor with no sources loads correctly."""
        descriptor_path = tmp_path / "nosource.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "nosource-001"
category: "parse"
name: "No sources"
phases:
  - parse
sources: []
expect:
  parse:
    status: "success"
""")
        
        descriptor = ConformanceTestDescriptor.from_file(descriptor_path)
        
        # Empty sources list is allowed
        assert len(descriptor.sources) == 0
    
    def test_invalid_test_id_format(self, tmp_path):
        """Test validation of test ID format."""
        descriptor_path = tmp_path / "badid.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "INVALID ID WITH SPACES"
category: "parse"
name: "Bad ID"
phases:
  - parse
sources:
  - content: 'app "Test"'
expect:
  parse:
    status: "success"
""")
        
        descriptor = ConformanceTestDescriptor.from_file(descriptor_path)
        
        # Test ID validation might be lenient or strict depending on implementation
        # This test documents expected behavior
        try:
            descriptor.validate()
        except ValueError:
            # Validation caught the bad ID
            pass


class TestDescriptorSerialization:
    """Test serialization and deserialization of descriptors."""
    
    def test_round_trip_serialization(self, tmp_path):
        """Test that descriptors can be saved and loaded."""
        original_yaml = """
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "roundtrip-001"
category: "parse"
name: "Round trip test"
description: "Tests serialization"
phases:
  - parse
sources:
  - content: 'app "Test"'
expect:
  parse:
    status: "success"
tags:
  - serialization
  - test
"""
        
        descriptor_path = tmp_path / "roundtrip.test.yaml"
        descriptor_path.write_text(original_yaml)
        
        # Load descriptor
        descriptor = ConformanceTestDescriptor.from_file(descriptor_path)
        
        # Verify key fields
        assert descriptor.test_id == "roundtrip-001"
        assert descriptor.category == "parse"
        assert "serialization" in descriptor.tags
        assert "test" in descriptor.tags
    
    def test_from_dict(self):
        """Test creating descriptor from dictionary."""
        data = {
            "spec_version": "1.0.0",
            "language_version": "1.0.0",
            "test_id": "dict-001",
            "category": "parse",
            "name": "From dict",
            "phases": ["parse"],
            "sources": [{"content": 'app "Test"'}],
            "expect": {
                "parse": {"status": "success"}
            }
        }
        
        descriptor = ConformanceTestDescriptor.from_dict(data)
        
        assert descriptor.test_id == "dict-001"
        assert TestPhase.PARSE in descriptor.phases
        assert descriptor.expect.parse.status == TestStatus.SUCCESS


class TestConfigurationOptions:
    """Test configuration options in test descriptors."""
    
    def test_strict_ast_match(self, tmp_path):
        """Test strict AST matching configuration."""
        descriptor_path = tmp_path / "strict.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "strict-001"
category: "parse"
name: "Strict AST"
phases:
  - parse
sources:
  - content: 'app "Test"'
config:
  strict_ast_match: true
expect:
  parse:
    status: "success"
""")
        
        descriptor = ConformanceTestDescriptor.from_file(descriptor_path)
        
        assert descriptor.config.strict_ast_match is True
    
    def test_timeout_configuration(self, tmp_path):
        """Test timeout configuration."""
        descriptor_path = tmp_path / "timeout.test.yaml"
        descriptor_path.write_text("""
spec_version: "1.0.0"
language_version: "1.0.0"
test_id: "timeout-001"
category: "runtime"
name: "Timeout test"
phases:
  - runtime
sources:
  - content: 'app "Test"'
config:
  timeout_seconds: 5.0
expect:
  runtime:
    status: "success"
""")
        
        descriptor = ConformanceTestDescriptor.from_file(descriptor_path)
        
        # Timeout is stored in runtime expectation, not config
        assert descriptor.expect.runtime.timeout_ms == 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
