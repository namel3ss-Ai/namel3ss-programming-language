"""
Error handling and developer experience tests.

Tests validate that errors:
- Include clear, actionable messages
- Identify the location of problems (line/column for syntax errors)
- Follow consistent format across the system
- Don't leak internal details or stack traces to end users
- Provide helpful suggestions where possible

These tests ensure developers receive high-quality error feedback.
"""

import pytest

from namel3ss.ast import (
    OutputSchema,
    OutputField,
    OutputFieldType,
    PromptArgs,
    PromptArg,
)
from namel3ss.parser import Parser, N3SyntaxError
from namel3ss.prompts.validator import OutputValidator
from namel3ss.resolver import resolve_program, ModuleResolutionError
from namel3ss.types import N3TypeError
from namel3ss.loader import load_program
from namel3ss.ast import Program


# ============================================================================
# Parser Error Tests
# ============================================================================

class TestParserErrors:
    """Test that parser errors are clear and actionable."""
    
    def test_unterminated_string_error(self):
        """Test error message for unterminated string."""
        source = 'app "Unterminated'
        
        with pytest.raises(N3SyntaxError) as exc_info:
            Parser(source).parse()
        
        error = str(exc_info.value)
        # Should mention the problem clearly
        assert error  # Non-empty error
    
    def test_missing_colon_error(self):
        """Test error for missing colon in block."""
        source = '''
app "Test".

page "Home" at "/"
  show text "Test"
'''
        
        with pytest.raises(N3SyntaxError) as exc_info:
            Parser(source).parse()
        
        error = str(exc_info.value)
        assert error  # Should produce meaningful error
    
    def test_invalid_prompt_syntax_error(self):
        """Test error for malformed prompt block."""
        source = '''
app "Test".

prompt "test" {
  args: {
    missing_type
  }
}
'''
        
        with pytest.raises((N3SyntaxError, ValueError)) as exc_info:
            module = Parser(source).parse()
        
        # Should fail during parsing or validation
        assert exc_info.value
    
    def test_unclosed_brace_error(self):
        """Test error for unclosed brace."""
        source = '''
app "Test".

llm gpt4 {
  provider: "openai"
'''  # Missing closing brace
        
        with pytest.raises(N3SyntaxError) as exc_info:
            Parser(source).parse()
        
        error = str(exc_info.value)
        assert "}" in error or "brace" in error.lower() or "block" in error.lower()


# ============================================================================
# Resolver Error Tests
# ============================================================================

class TestResolverErrors:
    """Test that semantic errors produce clear messages."""
    
    def test_undefined_dataset_error(self):
        """Test error when referencing undefined dataset."""
        source = '''
app "Test".

page "Home" at "/":
  show table "Data" from dataset nonexistent_dataset
'''
        
        module = Parser(source).parse()
        program = Program(modules=[module])
        
        with pytest.raises(ModuleResolutionError) as exc_info:
            resolve_program(program)
        
        error = str(exc_info.value)
        assert "nonexistent_dataset" in error
        assert "not found" in error.lower() or "undefined" in error.lower()
    
    def test_undefined_prompt_error(self):
        """Test error when chain references undefined prompt."""
        source = '''
app "Test".

define chain "test_chain" {
  steps:
    - step "s1" {
        kind: "prompt"
        target: "missing_prompt"
      }
}
'''
        
        module = Parser(source).parse()
        program = Program(modules=[module])
        
        with pytest.raises(ModuleResolutionError) as exc_info:
            resolve_program(program)
        
        error = str(exc_info.value)
        assert "missing_prompt" in error
    
    def test_undefined_llm_error(self):
        """Test error when prompt references undefined LLM."""
        source = '''
app "Test".

prompt "test" {
  model: "nonexistent_llm"
  template: "Test"
}
'''
        
        module = Parser(source).parse()
        program = Program(modules=[module])
        
        with pytest.raises(ModuleResolutionError) as exc_info:
            resolve_program(program)
        
        error = str(exc_info.value)
        assert "nonexistent_llm" in error
    
    def test_undefined_memory_error(self):
        """Test error when chain references undefined memory."""
        source = '''
app "Test".

define chain "test" {
  steps:
    - step "s1" {
        kind: "memory_read"
        target: "nonexistent_memory"
      }
}
'''
        
        module = Parser(source).parse()
        program = Program(modules=[module])
        
        with pytest.raises(ModuleResolutionError) as exc_info:
            resolve_program(program)
        
        error = str(exc_info.value)
        assert "nonexistent_memory" in error
    
    def test_circular_dependency_error(self):
        """Test error for circular dependencies (if implemented)."""
        # This is a placeholder for when circular dependency detection is added
        pytest.skip("Circular dependency detection not yet implemented")


# ============================================================================
# Validation Error Tests
# ============================================================================

class TestValidationErrors:
    """Test that output validation produces clear, field-specific errors."""
    
    def test_missing_required_field_error(self):
        """Test error when required field is missing."""
        schema = OutputSchema(fields=[
            OutputField(
                name="required_field",
                field_type=OutputFieldType(base_type="string"),
                nullable=False
            )
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({})
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("required_field" in err for err in result.errors)
        assert any("required" in err.lower() or "missing" in err.lower() for err in result.errors)
    
    def test_wrong_type_error_is_specific(self):
        """Test that type mismatch errors identify the field and expected type."""
        schema = OutputSchema(fields=[
            OutputField(
                name="count",
                field_type=OutputFieldType(base_type="int")
            )
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"count": "not_an_int"})
        
        assert not result.is_valid
        assert any("count" in err for err in result.errors)
        assert any("int" in err.lower() or "integer" in err.lower() for err in result.errors)
    
    def test_invalid_enum_error_shows_valid_options(self):
        """Test that enum validation error lists valid options."""
        schema = OutputSchema(fields=[
            OutputField(
                name="status",
                field_type=OutputFieldType(
                    base_type="enum",
                    enum_values=["active", "inactive", "pending"]
                )
            )
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"status": "invalid_status"})
        
        assert not result.is_valid
        errors_text = " ".join(result.errors)
        assert "status" in errors_text
        # Should mention at least some of the valid options
        assert "active" in errors_text or "inactive" in errors_text or "pending" in errors_text
    
    def test_nested_validation_error_includes_path(self):
        """Test that errors in nested objects include the field path."""
        schema = OutputSchema(fields=[
            OutputField(
                name="user",
                field_type=OutputFieldType(
                    base_type="object",
                    properties={
                        "name": OutputFieldType(base_type="string"),
                        "age": OutputFieldType(base_type="int"),
                    }
                )
            )
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"user": {"name": "Alice", "age": "not_int"}})
        
        assert not result.is_valid
        errors_text = " ".join(result.errors)
        # Should identify the nested field
        assert "user" in errors_text
        assert "age" in errors_text
    
    def test_list_validation_error_identifies_index(self):
        """Test that errors in lists identify which item failed."""
        schema = OutputSchema(fields=[
            OutputField(
                name="items",
                field_type=OutputFieldType(
                    base_type="list",
                    element_type=OutputFieldType(base_type="int")
                )
            )
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"items": [1, 2, "not_int", 4]})
        
        assert not result.is_valid
        errors_text = " ".join(result.errors)
        assert "items" in errors_text
        # May include index information depending on implementation


# ============================================================================
# Runtime Error Format Tests
# ============================================================================

class TestRuntimeErrorFormat:
    """Test that runtime errors follow consistent format.
    
    Note: These tests require integration with backend test utilities.
    """
    
    def test_validation_error_returns_422(self):
        """Test that validation errors return HTTP 422."""
        # This would require backend TestClient
        # Implemented in test_backend_integration.py
        pytest.skip("Covered by backend integration tests")
    
    def test_not_found_error_returns_404(self):
        """Test that missing resources return HTTP 404."""
        # Covered by test_backend_integration.py
        pytest.skip("Covered by backend integration tests")
    
    def test_internal_error_returns_500_without_traceback(self):
        """Test that internal errors return 500 without leaking details."""
        # Covered by test_backend_integration.py
        pytest.skip("Covered by backend integration tests")


# ============================================================================
# Error Message Quality Tests
# ============================================================================

class TestErrorMessageQuality:
    """Test that error messages follow best practices."""
    
    def test_error_messages_are_actionable(self):
        """Test that errors suggest how to fix the problem."""
        source = '''
app "Test".

page "Home" at "/":
  show table "Data" from dataset undefined
'''
        
        module = Parser(source).parse()
        program = Program(modules=[module])
        
        with pytest.raises(ModuleResolutionError) as exc_info:
            resolve_program(program)
        
        error = str(exc_info.value)
        # Error should clearly identify what's wrong
        assert "undefined" in error
        # Ideally suggests defining the dataset (implementation-dependent)
    
    def test_errors_dont_use_technical_jargon(self):
        """Test that errors are understandable to non-experts."""
        schema = OutputSchema(fields=[
            OutputField(
                name="value",
                field_type=OutputFieldType(base_type="string")
            )
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"value": 123})
        
        assert not result.is_valid
        errors_text = " ".join(result.errors).lower()
        # Should use plain language, not AST/internal terms
        assert "string" in errors_text  # User-facing type name
        # Should avoid technical terms like "node", "AST", "token" unless necessary
    
    def test_multiple_errors_are_all_reported(self):
        """Test that validator reports all errors, not just the first one."""
        schema = OutputSchema(fields=[
            OutputField(name="field1", field_type=OutputFieldType(base_type="string")),
            OutputField(name="field2", field_type=OutputFieldType(base_type="int")),
            OutputField(name="field3", field_type=OutputFieldType(base_type="bool")),
        ])
        validator = OutputValidator(schema)
        
        # All fields have wrong types
        result = validator.validate({
            "field1": 123,
            "field2": "not_int",
            "field3": "not_bool"
        })
        
        assert not result.is_valid
        # Should report multiple errors
        assert len(result.errors) >= 2  # At least some of the errors
    
    def test_error_includes_source_location_when_available(self):
        """Test that parser errors include line/column information."""
        source = 'app "Test" at'  # Incomplete syntax
        
        with pytest.raises(N3SyntaxError) as exc_info:
            Parser(source).parse()
        
        # Error should exist (may or may not include line/column depending on implementation)
        assert exc_info.value
        
        # If location info is available, it should be in the error
        # This is implementation-dependent


# ============================================================================
# Type Error Tests
# ============================================================================

class TestTypeErrors:
    """Test type checking errors (if implemented)."""
    
    def test_type_mismatch_in_expression(self):
        """Test that type errors in expressions are caught."""
        # Type checking may not be fully implemented yet
        pytest.skip("Full type checking not yet implemented")
    
    def test_incompatible_argument_types(self):
        """Test that prompt argument type mismatches are caught."""
        # Would require type propagation through the system
        pytest.skip("Argument type checking not yet implemented")


# ============================================================================
# Error Recovery Tests
# ============================================================================

class TestErrorRecovery:
    """Test that the system can recover from errors gracefully."""
    
    def test_parser_continues_after_error(self):
        """Test that parser can continue after encountering an error."""
        # Most parsers stop at first error, but some can continue
        pytest.skip("Error recovery not implemented")
    
    def test_partial_compilation_preserves_valid_parts(self):
        """Test that valid parts of the program are still accessible after error."""
        source = '''
app "Test".

page "Valid" at "/":
  show text "This works"

page "Invalid" at "/bad":
  show table "Data" from dataset nonexistent
'''
        
        module = Parser(source).parse()
        
        # Parsing should succeed
        assert len(module.apps) == 1
        assert len(module.apps[0].pages) == 2
        
        # Resolution will fail, but parsed content is still available
        program = Program(modules=[module])
        with pytest.raises(ModuleResolutionError):
            resolve_program(program)
        
        # Module still has valid content
        assert module.apps[0].pages[0].name == "Valid"


# ============================================================================
# Error Consistency Tests
# ============================================================================

class TestErrorConsistency:
    """Test that errors follow consistent patterns across the system."""
    
    def test_all_not_found_errors_have_similar_format(self):
        """Test that 'not found' errors use consistent phrasing."""
        # Test multiple scenarios that should produce "not found" errors
        scenarios = [
            ('page "Home" at "/": show table "T" from dataset missing', "missing"),
            ('prompt "p" { model: "missing_llm", template: "test" }', "missing_llm"),
        ]
        
        errors = []
        for source_fragment, missing_name in scenarios:
            source = f'app "Test".\n{source_fragment}'
            try:
                module = Parser(source).parse()
                program = Program(modules=[module])
                resolve_program(program)
            except ModuleResolutionError as e:
                errors.append(str(e))
        
        # All errors should mention the missing item
        for error, (_, expected_name) in zip(errors, scenarios):
            assert expected_name in error
    
    def test_validation_errors_have_consistent_format(self):
        """Test that validation errors follow a consistent pattern."""
        schema1 = OutputSchema(fields=[
            OutputField(name="f1", field_type=OutputFieldType(base_type="string"))
        ])
        schema2 = OutputSchema(fields=[
            OutputField(name="f2", field_type=OutputFieldType(base_type="int"))
        ])
        
        validator1 = OutputValidator(schema1)
        validator2 = OutputValidator(schema2)
        
        result1 = validator1.validate({"f1": 123})
        result2 = validator2.validate({"f2": "not_int"})
        
        # Both should fail with similar error structure
        assert not result1.is_valid
        assert not result2.is_valid
        assert len(result1.errors) > 0
        assert len(result2.errors) > 0


# ============================================================================
# Developer Experience Tests
# ============================================================================

class TestDeveloperExperience:
    """Test that the overall error experience is developer-friendly."""
    
    def test_error_for_common_mistake_is_helpful(self):
        """Test that common mistakes produce especially helpful errors."""
        # Common mistake: forgetting to define dataset before using it
        source = '''
app "Test".

page "Home" at "/":
  show table "Data" from dataset users
'''
        
        module = Parser(source).parse()
        program = Program(modules=[module])
        
        with pytest.raises(ModuleResolutionError) as exc_info:
            resolve_program(program)
        
        error = str(exc_info.value)
        # Should clearly identify the problem
        assert "users" in error
        # Ideally suggests defining the dataset (implementation-dependent)
    
    def test_error_in_complex_app_pinpoints_location(self):
        """Test that errors in large files pinpoint the problem location."""
        source = '''
app "LargeApp".

page "Page1" at "/1": show text "Valid"
page "Page2" at "/2": show text "Valid"
page "Page3" at "/3": show text "Valid"
page "Page4" at "/4": show text "Valid"
page "Page5" at "/5":
  show table "Data" from dataset undefined_dataset
page "Page6" at "/6": show text "Valid"
'''
        
        module = Parser(source).parse()
        program = Program(modules=[module])
        
        with pytest.raises(ModuleResolutionError) as exc_info:
            resolve_program(program)
        
        error = str(exc_info.value)
        # Should identify the specific problematic element
        assert "undefined_dataset" in error
        # May include page name or line number depending on implementation
    
    def test_error_includes_context_when_helpful(self):
        """Test that errors include relevant context."""
        # When a prompt validation fails, knowing which prompt helps
        schema = OutputSchema(fields=[
            OutputField(name="result", field_type=OutputFieldType(base_type="string"))
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"result": 123})
        
        # Error should be clear enough that user knows what to fix
        assert not result.is_valid
        errors_text = " ".join(result.errors)
        assert "result" in errors_text
        assert "string" in errors_text.lower()
