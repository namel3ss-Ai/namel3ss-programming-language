"""Tests for OutputValidator."""

import pytest
from namel3ss.prompts.validator import OutputValidator, ValidationResult
from namel3ss.ast import OutputSchema, OutputField, OutputFieldType


class TestPrimitiveValidation:
    """Test validation of primitive types."""
    
    def test_valid_string(self):
        """Test validating string values."""
        schema = OutputSchema(fields=[
            OutputField(name="text", field_type=OutputFieldType(base_type="string"))
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"text": "hello"})
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_invalid_string_type(self):
        """Test that non-string values are rejected for string fields."""
        schema = OutputSchema(fields=[
            OutputField(name="text", field_type=OutputFieldType(base_type="string"))
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"text": 123})
        assert not result.is_valid
        assert any("text" in err and "string" in err for err in result.errors)
    
    def test_valid_int(self):
        """Test validating integer values."""
        schema = OutputSchema(fields=[
            OutputField(name="count", field_type=OutputFieldType(base_type="int"))
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"count": 42})
        assert result.is_valid
    
    def test_valid_float(self):
        """Test validating float values."""
        schema = OutputSchema(fields=[
            OutputField(name="score", field_type=OutputFieldType(base_type="float"))
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"score": 3.14})
        assert result.is_valid
        
        # int should be accepted for float
        result = validator.validate({"score": 42})
        assert result.is_valid
    
    def test_valid_bool(self):
        """Test validating boolean values."""
        schema = OutputSchema(fields=[
            OutputField(name="active", field_type=OutputFieldType(base_type="bool"))
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"active": True})
        assert result.is_valid
        
        result = validator.validate({"active": False})
        assert result.is_valid


class TestEnumValidation:
    """Test validation of enum types."""
    
    def test_valid_enum_value(self):
        """Test that valid enum values pass."""
        schema = OutputSchema(fields=[
            OutputField(
                name="category",
                field_type=OutputFieldType(base_type="enum", enum_values=["a", "b", "c"])
            )
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"category": "a"})
        assert result.is_valid
    
    def test_invalid_enum_value(self):
        """Test that invalid enum values are rejected."""
        schema = OutputSchema(fields=[
            OutputField(
                name="category",
                field_type=OutputFieldType(base_type="enum", enum_values=["a", "b", "c"])
            )
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"category": "d"})
        assert not result.is_valid
        assert any("category" in err and "a, b, c" in err for err in result.errors)


class TestListValidation:
    """Test validation of list types."""
    
    def test_valid_list(self):
        """Test validating list values."""
        schema = OutputSchema(fields=[
            OutputField(
                name="tags",
                field_type=OutputFieldType(
                    base_type="list",
                    element_type=OutputFieldType(base_type="string")
                )
            )
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"tags": ["a", "b", "c"]})
        assert result.is_valid
    
    def test_empty_list(self):
        """Test that empty lists are valid."""
        schema = OutputSchema(fields=[
            OutputField(
                name="tags",
                field_type=OutputFieldType(
                    base_type="list",
                    element_type=OutputFieldType(base_type="string")
                )
            )
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"tags": []})
        assert result.is_valid
    
    def test_invalid_list_element(self):
        """Test that invalid list elements are caught."""
        schema = OutputSchema(fields=[
            OutputField(
                name="scores",
                field_type=OutputFieldType(
                    base_type="list",
                    element_type=OutputFieldType(base_type="int")
                )
            )
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"scores": [1, 2, "three"]})
        assert not result.is_valid
        assert any("scores[2]" in err for err in result.errors)
    
    def test_list_of_objects(self):
        """Test validating lists of objects."""
        schema = OutputSchema(fields=[
            OutputField(
                name="items",
                field_type=OutputFieldType(
                    base_type="list",
                    element_type=OutputFieldType(
                        base_type="object",
                        nested_fields=[
                            OutputField(name="id", field_type=OutputFieldType(base_type="int")),
                            OutputField(name="name", field_type=OutputFieldType(base_type="string")),
                        ]
                    )
                )
            )
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({
            "items": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ]
        })
        assert result.is_valid


class TestObjectValidation:
    """Test validation of nested object types."""
    
    def test_valid_nested_object(self):
        """Test validating nested objects."""
        schema = OutputSchema(fields=[
            OutputField(
                name="metadata",
                field_type=OutputFieldType(
                    base_type="object",
                    nested_fields=[
                        OutputField(name="author", field_type=OutputFieldType(base_type="string")),
                        OutputField(name="timestamp", field_type=OutputFieldType(base_type="int")),
                    ]
                )
            )
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({
            "metadata": {
                "author": "Alice",
                "timestamp": 1234567890
            }
        })
        assert result.is_valid
    
    def test_missing_nested_field(self):
        """Test that missing nested fields are caught."""
        schema = OutputSchema(fields=[
            OutputField(
                name="metadata",
                field_type=OutputFieldType(
                    base_type="object",
                    nested_fields=[
                        OutputField(name="author", field_type=OutputFieldType(base_type="string"), required=True),
                    ]
                )
            )
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"metadata": {}})
        assert not result.is_valid
        assert any("author" in err for err in result.errors)


class TestRequiredFields:
    """Test validation of required fields."""
    
    def test_missing_required_field(self):
        """Test that missing required fields are caught."""
        schema = OutputSchema(fields=[
            OutputField(name="required", field_type=OutputFieldType(base_type="string"), required=True),
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({})
        assert not result.is_valid
        assert any("required" in err and "Missing" in err for err in result.errors)
    
    def test_missing_optional_field(self):
        """Test that missing optional fields are allowed."""
        schema = OutputSchema(fields=[
            OutputField(name="optional", field_type=OutputFieldType(base_type="string"), required=False),
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({})
        assert result.is_valid


class TestUnexpectedFields:
    """Test handling of unexpected fields."""
    
    def test_unexpected_field_rejected(self):
        """Test that unexpected fields are rejected by default."""
        schema = OutputSchema(fields=[
            OutputField(name="expected", field_type=OutputFieldType(base_type="string")),
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"expected": "value", "unexpected": "value"})
        assert not result.is_valid
        assert any("unexpected" in err for err in result.errors)


class TestNullableFields:
    """Test validation of nullable types."""
    
    def test_nullable_field_with_null(self):
        """Test that null is accepted for nullable fields."""
        schema = OutputSchema(fields=[
            OutputField(
                name="optional",
                field_type=OutputFieldType(base_type="string", nullable=True),
                required=True
            ),
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"optional": None})
        assert result.is_valid
    
    def test_non_nullable_field_with_null(self):
        """Test that null is rejected for non-nullable fields."""
        schema = OutputSchema(fields=[
            OutputField(
                name="required",
                field_type=OutputFieldType(base_type="string", nullable=False),
                required=True
            ),
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({"required": None})
        assert not result.is_valid
        assert any("null" in err.lower() for err in result.errors)


class TestComplexSchemas:
    """Test validation of complex nested schemas."""
    
    def test_deeply_nested_structure(self):
        """Test validating deeply nested structures."""
        schema = OutputSchema(fields=[
            OutputField(
                name="data",
                field_type=OutputFieldType(
                    base_type="object",
                    nested_fields=[
                        OutputField(
                            name="items",
                            field_type=OutputFieldType(
                                base_type="list",
                                element_type=OutputFieldType(
                                    base_type="object",
                                    nested_fields=[
                                        OutputField(name="id", field_type=OutputFieldType(base_type="int")),
                                        OutputField(
                                            name="tags",
                                            field_type=OutputFieldType(
                                                base_type="list",
                                                element_type=OutputFieldType(base_type="string")
                                            )
                                        ),
                                    ]
                                )
                            )
                        )
                    ]
                )
            )
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({
            "data": {
                "items": [
                    {"id": 1, "tags": ["a", "b"]},
                    {"id": 2, "tags": ["c"]}
                ]
            }
        })
        assert result.is_valid
    
    def test_error_in_deeply_nested_field(self):
        """Test that errors in deeply nested fields are reported with paths."""
        schema = OutputSchema(fields=[
            OutputField(
                name="data",
                field_type=OutputFieldType(
                    base_type="object",
                    nested_fields=[
                        OutputField(
                            name="items",
                            field_type=OutputFieldType(
                                base_type="list",
                                element_type=OutputFieldType(
                                    base_type="object",
                                    nested_fields=[
                                        OutputField(name="id", field_type=OutputFieldType(base_type="int")),
                                    ]
                                )
                            )
                        )
                    ]
                )
            )
        ])
        validator = OutputValidator(schema)
        
        result = validator.validate({
            "data": {
                "items": [
                    {"id": "not_an_int"}  # Invalid type
                ]
            }
        })
        assert not result.is_valid
        # Error should indicate the path to the invalid field
        assert any("data" in err and "items" in err and "id" in err for err in result.errors)


class TestValidationRaising:
    """Test validate_and_raise method."""
    
    def test_validate_and_raise_on_error(self):
        """Test that validate_and_raise raises on validation errors."""
        schema = OutputSchema(fields=[
            OutputField(name="field", field_type=OutputFieldType(base_type="string")),
        ])
        validator = OutputValidator(schema)
        
        from namel3ss.prompts.validator import ValidationError
        with pytest.raises(ValidationError):
            validator.validate_and_raise({"field": 123})
    
    def test_validate_and_raise_on_success(self):
        """Test that validate_and_raise doesn't raise on success."""
        schema = OutputSchema(fields=[
            OutputField(name="field", field_type=OutputFieldType(base_type="string")),
        ])
        validator = OutputValidator(schema)
        
        # Should not raise
        validator.validate_and_raise({"field": "valid"})
