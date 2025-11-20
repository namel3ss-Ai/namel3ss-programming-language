"""
Tests for nested object support in output_schema for structured prompts.

Covers:
- Parser: nested object syntax parsing
- AST: nested field type representation and JSON schema generation
- Validation: nested object validation with OutputValidator
- Backend: runtime reconstruction and encoding
"""

import pytest
import json
from namel3ss.parser import Parser
from namel3ss.ast import (
    OutputSchema,
    OutputField,
    OutputFieldType,
)
from namel3ss.prompts.validator import OutputValidator, ValidationError
from namel3ss.errors import N3Error


class TestNestedObjectParsing:
    """Test parsing of nested object types in output_schema."""
    
    def test_parse_simple_nested_object(self):
        """Test parsing a simple nested object."""
        code = """prompt "classify_user" {
    model: gpt-4o
    template: "Classify the user"
    output_schema: {
        user: {
            name: string,
            age: int
        },
        status: string
    }
}
"""
        parser = Parser(code)
        module = parser.parse()
        
        # Extract prompts from module (may be in App or directly in body)
        prompts = []
        for node in module.body:
            if hasattr(node, 'prompts'):
                prompts.extend(node.prompts)
            elif hasattr(node, 'output_schema'):
                prompts.append(node)
        
        assert len(prompts) == 1
        prompt = prompts[0]
        assert prompt.output_schema is not None
        
        # Check top-level fields
        assert len(prompt.output_schema.fields) == 2
        user_field = prompt.output_schema.fields[0]
        status_field = prompt.output_schema.fields[1]
        
        assert user_field.name == "user"
        assert user_field.field_type.base_type == "object"
        assert user_field.field_type.nested_fields is not None
        
        # Check nested fields
        nested_fields = user_field.field_type.nested_fields
        assert len(nested_fields) == 2
        assert nested_fields[0].name == "name"
        assert nested_fields[0].field_type.base_type == "string"
        assert nested_fields[1].name == "age"
        assert nested_fields[1].field_type.base_type == "int"
        
        # Check other field
        assert status_field.name == "status"
        assert status_field.field_type.base_type == "string"
    
    def test_parse_deeply_nested_objects(self):
        """Test parsing objects nested multiple levels deep."""
        code = """prompt "get_profile" {
    model: gpt-4o
    template: "Get user profile"
    output_schema: {
        user: {
            profile: {
                name: string,
                contact: {
                    email: string,
                    phone: string
                }
            },
            id: int
        }
    }
}
"""
        parser = Parser(code)
        module = parser.parse()
        
        # Extract prompts from module (may be in App or directly in body)
        prompts = []
        for node in module.body:
            if hasattr(node, 'prompts'):
                prompts.extend(node.prompts)
            elif hasattr(node, 'output_schema'):
                prompts.append(node)
        
        assert len(prompts) == 1
        prompt = prompts[0]
        user_field = prompt.output_schema.fields[0]
        
        # Level 1: user
        assert user_field.field_type.base_type == "object"
        assert len(user_field.field_type.nested_fields) == 2
        
        # Level 2: profile
        profile_field = user_field.field_type.nested_fields[0]
        assert profile_field.name == "profile"
        assert profile_field.field_type.base_type == "object"
        assert len(profile_field.field_type.nested_fields) == 2
        
        # Level 3: contact
        contact_field = profile_field.field_type.nested_fields[1]
        assert contact_field.name == "contact"
        assert contact_field.field_type.base_type == "object"
        assert len(contact_field.field_type.nested_fields) == 2
        
        # Level 4: primitives
        email_field = contact_field.field_type.nested_fields[0]
        assert email_field.name == "email"
        assert email_field.field_type.base_type == "string"
    
    def test_parse_list_of_objects(self):
        """Test parsing list[object] type."""
        code = """prompt "get_users" {
    model: gpt-4o
    template: "Get all users"
    output_schema: {
        users: list[{
            name: string,
            email: string,
            age: int
        }],
        count: int
    }
}
"""
        parser = Parser(code)
        module = parser.parse()
        
        # Extract prompts from module (may be in App or directly in body)
        prompts = []
        for node in module.body:
            if hasattr(node, 'prompts'):
                prompts.extend(node.prompts)
            elif hasattr(node, 'output_schema'):
                prompts.append(node)
        
        assert len(prompts) == 1
        prompt = prompts[0]
        users_field = prompt.output_schema.fields[0]
        
        # Check list type
        assert users_field.name == "users"
        assert users_field.field_type.base_type == "list"
        
        # Check element type is object
        element_type = users_field.field_type.element_type
        assert element_type is not None
        assert element_type.base_type == "object"
        assert len(element_type.nested_fields) == 3
        
        # Check nested fields in list element
        assert element_type.nested_fields[0].name == "name"
        assert element_type.nested_fields[0].field_type.base_type == "string"
        assert element_type.nested_fields[1].name == "email"
        assert element_type.nested_fields[2].name == "age"
        assert element_type.nested_fields[2].field_type.base_type == "int"
    
    def test_parse_mixed_nested_structure(self):
        """Test complex schema with nested objects, lists, and enums."""
        code = """prompt "analyze_order" {
    model: gpt-4o
    template: "Analyze order"
    output_schema: {
        order: {
            id: string,
            customer: {
                name: string,
                email: string,
                tier: enum["bronze", "silver", "gold"]
            },
            items: list[{
                name: string,
                quantity: int,
                price: float
            }],
            total: float
        },
        status: enum["pending", "approved", "rejected"],
        tags: list[string]
    }
}
"""
        parser = Parser(code)
        module = parser.parse()
        
        # Extract prompts from module (may be in App or directly in body)
        prompts = []
        for node in module.body:
            if hasattr(node, 'prompts'):
                prompts.extend(node.prompts)
            elif hasattr(node, 'output_schema'):
                prompts.append(node)
        
        assert len(prompts) == 1
        prompt = prompts[0]
        schema = prompt.output_schema
        assert len(schema.fields) == 3
        
        # Check order object
        order_field = schema.fields[0]
        assert order_field.field_type.base_type == "object"
        order_nested = order_field.field_type.nested_fields
        assert len(order_nested) == 4
        
        # Check nested customer object
        customer_field = order_nested[1]
        assert customer_field.name == "customer"
        assert customer_field.field_type.base_type == "object"
        customer_nested = customer_field.field_type.nested_fields
        assert len(customer_nested) == 3
        
        # Check enum in nested object
        tier_field = customer_nested[2]
        assert tier_field.name == "tier"
        assert tier_field.field_type.base_type == "enum"
        assert tier_field.field_type.enum_values == ["bronze", "silver", "gold"]
        
        # Check list of objects
        items_field = order_nested[2]
        assert items_field.name == "items"
        assert items_field.field_type.base_type == "list"
        item_type = items_field.field_type.element_type
        assert item_type.base_type == "object"
        assert len(item_type.nested_fields) == 3
    
    def test_parse_error_empty_nested_object(self):
        """Test that empty nested objects raise an error."""
        code = """prompt "test" {
    model: gpt-4o
    template: "Test"
    output_schema: {
        user: {
        }
    }
}
"""
        parser = Parser(code)
        with pytest.raises(N3Error, match="must have at least one field"):
            parser.parse()
    
    def test_parse_error_malformed_nested_syntax(self):
        """Test error handling for malformed nested object syntax."""
        code = """prompt "test" {
    model: gpt-4o
    template: "Test"
    output_schema: {
        user: {
            name
        }
    }
}
"""
        parser = Parser(code)
        with pytest.raises(N3Error, match="Expected 'field_name: type'"):
            parser.parse()


class TestNestedJSONSchemaGeneration:
    """Test JSON Schema generation for nested objects."""
    
    def test_simple_nested_to_json_schema(self):
        """Test JSON Schema generation for simple nested object."""
        schema = OutputSchema(fields=[
            OutputField(
                name="user",
                field_type=OutputFieldType(
                    base_type="object",
                    nested_fields=[
                        OutputField(
                            name="name",
                            field_type=OutputFieldType(base_type="string"),
                            required=True
                        ),
                        OutputField(
                            name="age",
                            field_type=OutputFieldType(base_type="int"),
                            required=True
                        )
                    ]
                ),
                required=True
            ),
            OutputField(
                name="status",
                field_type=OutputFieldType(base_type="string"),
                required=True
            )
        ])
        
        json_schema = schema.to_json_schema()
        
        # Check structure
        assert json_schema["type"] == "object"
        assert "user" in json_schema["properties"]
        assert "status" in json_schema["properties"]
        assert json_schema["required"] == ["user", "status"]
        
        # Check nested object
        user_schema = json_schema["properties"]["user"]
        assert user_schema["type"] == "object"
        assert "name" in user_schema["properties"]
        assert "age" in user_schema["properties"]
        assert user_schema["properties"]["name"]["type"] == "string"
        assert user_schema["properties"]["age"]["type"] == "integer"
        assert user_schema["required"] == ["name", "age"]
    
    def test_deeply_nested_to_json_schema(self):
        """Test JSON Schema generation for deeply nested structure."""
        schema = OutputSchema(fields=[
            OutputField(
                name="data",
                field_type=OutputFieldType(
                    base_type="object",
                    nested_fields=[
                        OutputField(
                            name="level1",
                            field_type=OutputFieldType(
                                base_type="object",
                                nested_fields=[
                                    OutputField(
                                        name="level2",
                                        field_type=OutputFieldType(
                                            base_type="object",
                                            nested_fields=[
                                                OutputField(
                                                    name="value",
                                                    field_type=OutputFieldType(base_type="string"),
                                                    required=True
                                                )
                                            ]
                                        ),
                                        required=True
                                    )
                                ]
                            ),
                            required=True
                        )
                    ]
                ),
                required=True
            )
        ])
        
        json_schema = schema.to_json_schema()
        
        # Navigate nested structure
        data_props = json_schema["properties"]["data"]
        assert data_props["type"] == "object"
        
        level1_props = data_props["properties"]["level1"]
        assert level1_props["type"] == "object"
        
        level2_props = level1_props["properties"]["level2"]
        assert level2_props["type"] == "object"
        
        value_props = level2_props["properties"]["value"]
        assert value_props["type"] == "string"
    
    def test_list_of_objects_to_json_schema(self):
        """Test JSON Schema generation for list of objects."""
        schema = OutputSchema(fields=[
            OutputField(
                name="users",
                field_type=OutputFieldType(
                    base_type="list",
                    element_type=OutputFieldType(
                        base_type="object",
                        nested_fields=[
                            OutputField(
                                name="name",
                                field_type=OutputFieldType(base_type="string"),
                                required=True
                            ),
                            OutputField(
                                name="email",
                                field_type=OutputFieldType(base_type="string"),
                                required=True
                            )
                        ]
                    )
                ),
                required=True
            )
        ])
        
        json_schema = schema.to_json_schema()
        
        users_schema = json_schema["properties"]["users"]
        assert users_schema["type"] == "array"
        
        items_schema = users_schema["items"]
        assert items_schema["type"] == "object"
        assert "name" in items_schema["properties"]
        assert "email" in items_schema["properties"]
        assert items_schema["required"] == ["name", "email"]


class TestNestedObjectValidation:
    """Test validation of nested objects with OutputValidator."""
    
    def test_validate_simple_nested_object_valid(self):
        """Test validation passes for valid nested object."""
        schema = OutputSchema(fields=[
            OutputField(
                name="user",
                field_type=OutputFieldType(
                    base_type="object",
                    nested_fields=[
                        OutputField(
                            name="name",
                            field_type=OutputFieldType(base_type="string"),
                            required=True
                        ),
                        OutputField(
                            name="age",
                            field_type=OutputFieldType(base_type="int"),
                            required=True
                        )
                    ]
                ),
                required=True
            )
        ])
        
        validator = OutputValidator(schema)
        
        # Valid output
        output = {
            "user": {
                "name": "Alice",
                "age": 30
            }
        }
        
        result = validator.validate(output)
        assert result.valid
        assert len(result.errors) == 0
        assert result.validated_output == output
    
    def test_validate_nested_missing_required_field(self):
        """Test validation fails when nested required field is missing."""
        schema = OutputSchema(fields=[
            OutputField(
                name="user",
                field_type=OutputFieldType(
                    base_type="object",
                    nested_fields=[
                        OutputField(
                            name="name",
                            field_type=OutputFieldType(base_type="string"),
                            required=True
                        ),
                        OutputField(
                            name="email",
                            field_type=OutputFieldType(base_type="string"),
                            required=True
                        )
                    ]
                ),
                required=True
            )
        ])
        
        validator = OutputValidator(schema)
        
        # Missing 'email' in nested object
        output = {
            "user": {
                "name": "Bob"
            }
        }
        
        result = validator.validate(output)
        assert not result.valid
        assert len(result.errors) == 1
        assert "user.email" in result.errors[0].field_path
        assert "Missing required" in str(result.errors[0])
    
    def test_validate_nested_wrong_type(self):
        """Test validation fails when nested field has wrong type."""
        schema = OutputSchema(fields=[
            OutputField(
                name="user",
                field_type=OutputFieldType(
                    base_type="object",
                    nested_fields=[
                        OutputField(
                            name="age",
                            field_type=OutputFieldType(base_type="int"),
                            required=True
                        )
                    ]
                ),
                required=True
            )
        ])
        
        validator = OutputValidator(schema)
        
        # Wrong type for age
        output = {
            "user": {
                "age": "thirty"
            }
        }
        
        result = validator.validate(output)
        assert not result.valid
        assert len(result.errors) == 1
        assert "user.age" in result.errors[0].field_path
        assert "must be an integer" in str(result.errors[0])
    
    def test_validate_deeply_nested_objects(self):
        """Test validation of deeply nested object structures."""
        schema = OutputSchema(fields=[
            OutputField(
                name="data",
                field_type=OutputFieldType(
                    base_type="object",
                    nested_fields=[
                        OutputField(
                            name="user",
                            field_type=OutputFieldType(
                                base_type="object",
                                nested_fields=[
                                    OutputField(
                                        name="profile",
                                        field_type=OutputFieldType(
                                            base_type="object",
                                            nested_fields=[
                                                OutputField(
                                                    name="name",
                                                    field_type=OutputFieldType(base_type="string"),
                                                    required=True
                                                )
                                            ]
                                        ),
                                        required=True
                                    )
                                ]
                            ),
                            required=True
                        )
                    ]
                ),
                required=True
            )
        ])
        
        validator = OutputValidator(schema)
        
        # Valid deeply nested
        output = {
            "data": {
                "user": {
                    "profile": {
                        "name": "Charlie"
                    }
                }
            }
        }
        
        result = validator.validate(output)
        assert result.valid
        
        # Invalid: missing deep field
        output_invalid = {
            "data": {
                "user": {
                    "profile": {}
                }
            }
        }
        
        result = validator.validate(output_invalid)
        assert not result.valid
        assert "data.user.profile.name" in result.errors[0].field_path
    
    def test_validate_list_of_objects(self):
        """Test validation of list containing objects."""
        schema = OutputSchema(fields=[
            OutputField(
                name="users",
                field_type=OutputFieldType(
                    base_type="list",
                    element_type=OutputFieldType(
                        base_type="object",
                        nested_fields=[
                            OutputField(
                                name="name",
                                field_type=OutputFieldType(base_type="string"),
                                required=True
                            ),
                            OutputField(
                                name="age",
                                field_type=OutputFieldType(base_type="int"),
                                required=True
                            )
                        ]
                    )
                ),
                required=True
            )
        ])
        
        validator = OutputValidator(schema)
        
        # Valid list of objects
        output = {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ]
        }
        
        result = validator.validate(output)
        assert result.valid
        
        # Invalid: missing field in list element
        output_invalid = {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob"}  # Missing age
            ]
        }
        
        result = validator.validate(output_invalid)
        assert not result.valid
        assert "users[1].age" in result.errors[0].field_path
    
    def test_validate_nested_with_enums(self):
        """Test validation of nested objects containing enums."""
        schema = OutputSchema(fields=[
            OutputField(
                name="order",
                field_type=OutputFieldType(
                    base_type="object",
                    nested_fields=[
                        OutputField(
                            name="status",
                            field_type=OutputFieldType(
                                base_type="enum",
                                enum_values=["pending", "approved", "rejected"]
                            ),
                            required=True
                        ),
                        OutputField(
                            name="priority",
                            field_type=OutputFieldType(
                                base_type="enum",
                                enum_values=["low", "medium", "high"]
                            ),
                            required=True
                        )
                    ]
                ),
                required=True
            )
        ])
        
        validator = OutputValidator(schema)
        
        # Valid
        output = {
            "order": {
                "status": "approved",
                "priority": "high"
            }
        }
        
        result = validator.validate(output)
        assert result.valid
        
        # Invalid enum value in nested object
        output_invalid = {
            "order": {
                "status": "completed",  # Invalid
                "priority": "high"
            }
        }
        
        result = validator.validate(output_invalid)
        assert not result.valid
        assert "order.status" in result.errors[0].field_path
        assert "invalid enum value" in str(result.errors[0]).lower()
    
    def test_validate_complex_mixed_structure(self):
        """Test validation of complex structure with all supported types."""
        schema = OutputSchema(fields=[
            OutputField(
                name="analysis",
                field_type=OutputFieldType(
                    base_type="object",
                    nested_fields=[
                        OutputField(
                            name="summary",
                            field_type=OutputFieldType(base_type="string"),
                            required=True
                        ),
                        OutputField(
                            name="metrics",
                            field_type=OutputFieldType(
                                base_type="object",
                                nested_fields=[
                                    OutputField(
                                        name="score",
                                        field_type=OutputFieldType(base_type="float"),
                                        required=True
                                    ),
                                    OutputField(
                                        name="confidence",
                                        field_type=OutputFieldType(base_type="float"),
                                        required=True
                                    )
                                ]
                            ),
                            required=True
                        ),
                        OutputField(
                            name="tags",
                            field_type=OutputFieldType(
                                base_type="list",
                                element_type=OutputFieldType(base_type="string")
                            ),
                            required=True
                        ),
                        OutputField(
                            name="recommendations",
                            field_type=OutputFieldType(
                                base_type="list",
                                element_type=OutputFieldType(
                                    base_type="object",
                                    nested_fields=[
                                        OutputField(
                                            name="action",
                                            field_type=OutputFieldType(base_type="string"),
                                            required=True
                                        ),
                                        OutputField(
                                            name="priority",
                                            field_type=OutputFieldType(
                                                base_type="enum",
                                                enum_values=["low", "medium", "high"]
                                            ),
                                            required=True
                                        )
                                    ]
                                )
                            ),
                            required=True
                        )
                    ]
                ),
                required=True
            )
        ])
        
        validator = OutputValidator(schema)
        
        # Valid complex output
        output = {
            "analysis": {
                "summary": "Overall positive sentiment",
                "metrics": {
                    "score": 0.85,
                    "confidence": 0.92
                },
                "tags": ["positive", "detailed", "actionable"],
                "recommendations": [
                    {"action": "Follow up with customer", "priority": "high"},
                    {"action": "Update documentation", "priority": "medium"}
                ]
            }
        }
        
        result = validator.validate(output)
        assert result.valid
        assert result.validated_output == output


class TestBackendRuntimeEncoding:
    """Test backend runtime encoding of nested output schemas."""
    
    def test_encode_and_reconstruct_nested_schema(self):
        """Test that nested schemas can be encoded and reconstructed."""
        from namel3ss.codegen.backend.state import _encode_output_schema
        
        # Create a nested schema
        schema = OutputSchema(fields=[
            OutputField(
                name="user",
                field_type=OutputFieldType(
                    base_type="object",
                    nested_fields=[
                        OutputField(
                            name="name",
                            field_type=OutputFieldType(base_type="string"),
                            required=True
                        ),
                        OutputField(
                            name="roles",
                            field_type=OutputFieldType(
                                base_type="list",
                                element_type=OutputFieldType(base_type="string")
                            ),
                            required=True
                        )
                    ]
                ),
                required=True
            )
        ])
        
        # Encode to dict
        encoded = _encode_output_schema(schema, set())
        
        # Check encoding
        assert "fields" in encoded
        assert len(encoded["fields"]) == 1
        
        user_field = encoded["fields"][0]
        assert user_field["name"] == "user"
        assert user_field["field_type"]["base_type"] == "object"
        assert "nested_fields" in user_field["field_type"]
        
        nested = user_field["field_type"]["nested_fields"]
        assert len(nested) == 2
        assert nested[0]["name"] == "name"
        assert nested[0]["field_type"]["base_type"] == "string"
        assert nested[1]["name"] == "roles"
        assert nested[1]["field_type"]["base_type"] == "list"
        assert nested[1]["field_type"]["element_type"]["base_type"] == "string"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
