"""Tests for structured prompt validation."""

import pytest
from namel3ss.ast import (
    Prompt, PromptArgument, OutputSchema, OutputField, OutputFieldType, EnumType
)
from namel3ss.resolver import _validate_structured_prompt
from namel3ss.errors import N3ValidationError


class TestArgumentValidation:
    """Test validation of prompt arguments."""
    
    def test_valid_args(self):
        """Test that valid arguments pass validation."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test {text} {count}",
            args=[
                PromptArgument(name="text", arg_type="string", required=True),
                PromptArgument(name="count", arg_type="int", required=False, default=10),
            ]
        )
        # Should not raise
        _validate_structured_prompt(prompt)
    
    def test_duplicate_arg_names(self):
        """Test that duplicate argument names are rejected."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            args=[
                PromptArgument(name="text", arg_type="string"),
                PromptArgument(name="text", arg_type="int"),
            ]
        )
        with pytest.raises(N3ValidationError, match="Duplicate argument.*text"):
            _validate_structured_prompt(prompt)
    
    def test_invalid_arg_type(self):
        """Test that invalid argument types are rejected."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            args=[
                PromptArgument(name="x", arg_type="invalid_type"),
            ]
        )
        with pytest.raises(N3ValidationError, match="Invalid argument type"):
            _validate_structured_prompt(prompt)
    
    def test_required_after_optional(self):
        """Test that required args cannot follow optional args."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            args=[
                PromptArgument(name="optional", arg_type="string", required=False, default="x"),
                PromptArgument(name="required", arg_type="string", required=True),
            ]
        )
        with pytest.raises(N3ValidationError, match="Required argument.*after optional"):
            _validate_structured_prompt(prompt)


class TestOutputSchemaValidation:
    """Test validation of output schemas."""
    
    def test_valid_schema(self):
        """Test that valid schemas pass validation."""
        schema = OutputSchema(fields=[
            OutputField(
                name="category",
                field_type=OutputFieldType(base_type="enum", enum_values=["a", "b"]),
                required=True
            ),
            OutputField(
                name="score",
                field_type=OutputFieldType(base_type="float"),
                required=True
            ),
        ])
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            output_schema=schema
        )
        # Should not raise
        _validate_structured_prompt(prompt)
    
    def test_duplicate_field_names(self):
        """Test that duplicate field names are rejected."""
        schema = OutputSchema(fields=[
            OutputField(name="field", field_type=OutputFieldType(base_type="string")),
            OutputField(name="field", field_type=OutputFieldType(base_type="int")),
        ])
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            output_schema=schema
        )
        with pytest.raises(N3ValidationError, match="Duplicate field.*field"):
            _validate_structured_prompt(prompt)
    
    def test_invalid_field_type(self):
        """Test that invalid field types are rejected."""
        schema = OutputSchema(fields=[
            OutputField(
                name="field",
                field_type=OutputFieldType(base_type="invalid_type")
            ),
        ])
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            output_schema=schema
        )
        with pytest.raises(N3ValidationError, match="Invalid type"):
            _validate_structured_prompt(prompt)
    
    def test_enum_without_values(self):
        """Test that enum types must have values."""
        schema = OutputSchema(fields=[
            OutputField(
                name="category",
                field_type=OutputFieldType(base_type="enum", enum_values=None)
            ),
        ])
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            output_schema=schema
        )
        with pytest.raises(N3ValidationError, match="Enum.*must have values"):
            _validate_structured_prompt(prompt)
    
    def test_empty_enum_values(self):
        """Test that enum must have at least one value."""
        schema = OutputSchema(fields=[
            OutputField(
                name="category",
                field_type=OutputFieldType(base_type="enum", enum_values=[])
            ),
        ])
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            output_schema=schema
        )
        with pytest.raises(N3ValidationError, match="at least one value"):
            _validate_structured_prompt(prompt)
    
    def test_list_without_element_type(self):
        """Test that list types must specify element type."""
        schema = OutputSchema(fields=[
            OutputField(
                name="items",
                field_type=OutputFieldType(base_type="list", element_type=None)
            ),
        ])
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            output_schema=schema
        )
        with pytest.raises(N3ValidationError, match="List.*must specify element type"):
            _validate_structured_prompt(prompt)


class TestTemplatePlaceholderValidation:
    """Test validation of template placeholders."""
    
    def test_valid_placeholders(self):
        """Test that all placeholders have corresponding args."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Hello {name}, you are {age} years old",
            args=[
                PromptArgument(name="name", arg_type="string"),
                PromptArgument(name="age", arg_type="int"),
            ]
        )
        # Should not raise
        _validate_structured_prompt(prompt)
    
    def test_undefined_placeholder(self):
        """Test that undefined placeholders are rejected."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Hello {name}, you are {age} years old",
            args=[
                PromptArgument(name="name", arg_type="string"),
                # missing 'age' argument
            ]
        )
        with pytest.raises(N3ValidationError, match="Undefined placeholder.*age"):
            _validate_structured_prompt(prompt)
    
    def test_unused_args_allowed(self):
        """Test that unused args are allowed (may be used programmatically)."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Hello {name}",
            args=[
                PromptArgument(name="name", arg_type="string"),
                PromptArgument(name="unused", arg_type="string"),
            ]
        )
        # Should not raise - unused args are allowed
        _validate_structured_prompt(prompt)


class TestNestedValidation:
    """Test validation of nested structures."""
    
    def test_valid_nested_object(self):
        """Test that valid nested objects pass validation."""
        nested_fields = [
            OutputField(name="author", field_type=OutputFieldType(base_type="string")),
            OutputField(name="date", field_type=OutputFieldType(base_type="int")),
        ]
        schema = OutputSchema(fields=[
            OutputField(
                name="metadata",
                field_type=OutputFieldType(base_type="object", nested_fields=nested_fields)
            ),
        ])
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            output_schema=schema
        )
        # Should not raise
        _validate_structured_prompt(prompt)
    
    def test_nested_enum(self):
        """Test validation of enums in nested objects."""
        nested_fields = [
            OutputField(
                name="status",
                field_type=OutputFieldType(base_type="enum", enum_values=["active", "inactive"])
            ),
        ]
        schema = OutputSchema(fields=[
            OutputField(
                name="user",
                field_type=OutputFieldType(base_type="object", nested_fields=nested_fields)
            ),
        ])
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            output_schema=schema
        )
        # Should not raise
        _validate_structured_prompt(prompt)
    
    def test_invalid_nested_field(self):
        """Test that invalid nested fields are caught."""
        nested_fields = [
            OutputField(name="field", field_type=OutputFieldType(base_type="invalid")),
        ]
        schema = OutputSchema(fields=[
            OutputField(
                name="nested",
                field_type=OutputFieldType(base_type="object", nested_fields=nested_fields)
            ),
        ])
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            output_schema=schema
        )
        with pytest.raises(N3ValidationError, match="Invalid type"):
            _validate_structured_prompt(prompt)
