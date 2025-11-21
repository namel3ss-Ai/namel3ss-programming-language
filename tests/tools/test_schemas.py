"""Unit tests for Tool Adapter Framework schemas."""

import pytest
from pydantic import Field, ValidationError

from namel3ss.tools.schemas import (
    ToolInputModel,
    ToolOutputModel,
    ToolChunkModel,
    ToolErrorModel,
    SimpleTextInput,
    SimpleTextOutput,
    KeyValueInput,
    KeyValueOutput,
    ListInput,
    ListOutput,
    JSONInput,
    JSONOutput,
    merge_schemas,
    extend_schema,
    schema_to_json_schema,
    validate_against_schema,
)


# Custom test schemas

class CustomInput(ToolInputModel):
    """Custom input schema for testing."""
    field1: str = Field(..., description="First field")
    field2: int = Field(default=10, ge=0, le=100, description="Second field")
    optional_field: str = Field(None, description="Optional field")


class CustomOutput(ToolOutputModel):
    """Custom output schema for testing."""
    result: str = Field(..., description="Result string")
    count: int = Field(..., description="Result count")
    metadata: dict = Field(default_factory=dict, description="Metadata")


# Tests

def test_tool_input_model_validation():
    """Test ToolInputModel validation."""
    # Valid input
    input_data = CustomInput(field1="test", field2=50)
    assert input_data.field1 == "test"
    assert input_data.field2 == 50
    assert input_data.optional_field is None
    
    # With optional field
    input_data2 = CustomInput(field1="test", field2=30, optional_field="optional")
    assert input_data2.optional_field == "optional"
    
    # Invalid input (missing required field)
    with pytest.raises(ValidationError):
        CustomInput(field2=50)
    
    # Invalid input (constraint violation)
    with pytest.raises(ValidationError):
        CustomInput(field1="test", field2=200)
    
    # Invalid input (wrong type)
    with pytest.raises(ValidationError):
        CustomInput(field1="test", field2="not an int")


def test_tool_input_model_extra_forbid():
    """Test ToolInputModel forbids extra fields."""
    # Should reject extra fields
    with pytest.raises(ValidationError) as exc_info:
        CustomInput(field1="test", field2=50, extra_field="value")
    
    assert "extra_field" in str(exc_info.value).lower()


def test_tool_output_model_validation():
    """Test ToolOutputModel validation."""
    # Valid output
    output = CustomOutput(result="success", count=10)
    assert output.result == "success"
    assert output.count == 10
    assert output.metadata == {}
    
    # With metadata
    output2 = CustomOutput(
        result="success",
        count=10,
        metadata={"key": "value"}
    )
    assert output2.metadata == {"key": "value"}


def test_tool_output_model_extra_allow():
    """Test ToolOutputModel allows extra fields."""
    # Should allow extra fields
    output = CustomOutput(result="success", count=10, extra_field="value")
    assert output.result == "success"
    assert hasattr(output, "extra_field")


def test_tool_chunk_model():
    """Test ToolChunkModel."""
    chunk = ToolChunkModel(
        sequence=0,
        is_final=False,
        metadata={"chunk_type": "text"}
    )
    
    assert chunk.sequence == 0
    assert chunk.is_final is False
    assert chunk.metadata == {"chunk_type": "text"}
    assert chunk.timestamp is not None


def test_tool_error_model():
    """Test ToolErrorModel."""
    error = ToolErrorModel(
        code="TOOL003",
        message="Validation failed",
        tool_name="test_tool",
        field="field1",
        details={"expected": "string", "got": "int"}
    )
    
    assert error.code == "TOOL003"
    assert error.message == "Validation failed"
    assert error.tool_name == "test_tool"
    assert error.field == "field1"
    assert error.details == {"expected": "string", "got": "int"}
    assert error.timestamp is not None


def test_tool_input_to_dict():
    """Test ToolInputModel to_dict()."""
    input_data = CustomInput(field1="test", field2=50)
    data_dict = input_data.to_dict()
    
    assert isinstance(data_dict, dict)
    assert data_dict["field1"] == "test"
    assert data_dict["field2"] == 50


def test_tool_input_to_json():
    """Test ToolInputModel to_json()."""
    input_data = CustomInput(field1="test", field2=50)
    json_str = input_data.to_json()
    
    assert isinstance(json_str, str)
    assert "test" in json_str
    assert "50" in json_str


def test_tool_input_from_dict():
    """Test ToolInputModel from_dict()."""
    data_dict = {"field1": "test", "field2": 50}
    input_data = CustomInput.from_dict(data_dict)
    
    assert isinstance(input_data, CustomInput)
    assert input_data.field1 == "test"
    assert input_data.field2 == 50


def test_tool_input_get_json_schema():
    """Test ToolInputModel get_json_schema()."""
    schema = CustomInput.get_json_schema()
    
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "field1" in schema["properties"]
    assert "field2" in schema["properties"]
    assert schema["properties"]["field1"]["type"] == "string"
    assert schema["properties"]["field2"]["type"] == "integer"


def test_simple_text_input():
    """Test SimpleTextInput."""
    input_data = SimpleTextInput(text="hello world")
    assert input_data.text == "hello world"


def test_simple_text_output():
    """Test SimpleTextOutput."""
    output = SimpleTextOutput(result="hello world")
    assert output.result == "hello world"


def test_key_value_input():
    """Test KeyValueInput."""
    input_data = KeyValueInput(key="name", value="John")
    assert input_data.key == "name"
    assert input_data.value == "John"


def test_key_value_output():
    """Test KeyValueOutput."""
    output = KeyValueOutput(key="result", value={"status": "success"})
    assert output.key == "result"
    assert output.value == {"status": "success"}


def test_list_input():
    """Test ListInput."""
    input_data = ListInput(items=["a", "b", "c"])
    assert input_data.items == ["a", "b", "c"]


def test_list_output():
    """Test ListOutput."""
    output = ListOutput(items=[1, 2, 3])
    assert output.items == [1, 2, 3]


def test_json_input():
    """Test JSONInput."""
    input_data = JSONInput(data={"key": "value", "count": 10})
    assert input_data.data == {"key": "value", "count": 10}


def test_json_output():
    """Test JSONOutput."""
    output = JSONOutput(data={"result": "success", "items": [1, 2, 3]})
    assert output.data == {"result": "success", "items": [1, 2, 3]}


def test_merge_schemas():
    """Test merge_schemas()."""
    class Schema1(ToolInputModel):
        field1: str
    
    class Schema2(ToolInputModel):
        field2: int
    
    MergedSchema = merge_schemas(Schema1, Schema2, name="TestMerged")
    
    # Should have both fields
    merged = MergedSchema(field1="test", field2=10)
    assert merged.field1 == "test"
    assert merged.field2 == 10


def test_merge_schemas_conflict():
    """Test merge_schemas() with field conflicts (last wins)."""
    class Schema1(ToolInputModel):
        field: str
    
    class Schema2(ToolInputModel):
        field: int
    
    MergedSchema = merge_schemas(Schema1, Schema2, name="TestConflict")
    
    # Second schema wins
    merged = MergedSchema(field=10)
    assert merged.field == 10


def test_extend_schema():
    """Test extend_schema()."""
    class BaseSchema(ToolInputModel):
        base_field: str
    
    ExtendedSchema = extend_schema(
        BaseSchema,
        additional_field=(int, Field(10, ge=0)),
        another_field=(str, Field("default")),
    )
    
    extended = ExtendedSchema(
        base_field="test",
        additional_field=20,
        another_field="value"
    )
    
    assert extended.base_field == "test"
    assert extended.additional_field == 20
    assert extended.another_field == "value"


def test_schema_to_json_schema():
    """Test schema_to_json_schema()."""
    schema = schema_to_json_schema(CustomInput)
    
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "field1" in schema["properties"]
    assert "field2" in schema["properties"]


def test_validate_against_schema():
    """Test validate_against_schema()."""
    data = {"field1": "test", "field2": 50}
    validated = validate_against_schema(data, CustomInput)
    
    assert isinstance(validated, CustomInput)
    assert validated.field1 == "test"
    assert validated.field2 == 50


def test_validate_against_schema_invalid():
    """Test validate_against_schema() with invalid data."""
    data = {"field2": 50}  # Missing required field1
    
    with pytest.raises(ValidationError):
        validate_against_schema(data, CustomInput)


def test_tool_input_whitespace_stripping():
    """Test ToolInputModel strips whitespace."""
    class WhitespaceInput(ToolInputModel):
        text: str
    
    input_data = WhitespaceInput(text="  hello world  ")
    assert input_data.text == "hello world"  # Whitespace stripped


def test_tool_output_timestamp():
    """Test ToolChunkModel includes timestamp."""
    chunk = ToolChunkModel()
    assert chunk.timestamp is not None
    
    from datetime import datetime
    assert isinstance(chunk.timestamp, datetime)


def test_nested_schemas():
    """Test nested schema models."""
    class NestedInput(ToolInputModel):
        field: str
    
    class ParentInput(ToolInputModel):
        nested: NestedInput
        count: int
    
    input_data = ParentInput(
        nested=NestedInput(field="test"),
        count=10
    )
    
    assert input_data.nested.field == "test"
    assert input_data.count == 10


def test_schema_defaults():
    """Test schema field defaults."""
    class DefaultsInput(ToolInputModel):
        required: str
        optional: str = "default_value"
        optional_int: int = 42
    
    input_data = DefaultsInput(required="test")
    assert input_data.required == "test"
    assert input_data.optional == "default_value"
    assert input_data.optional_int == 42


def test_schema_field_validation():
    """Test schema field validators."""
    class ValidatedInput(ToolInputModel):
        email: str = Field(..., pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        age: int = Field(..., ge=0, le=120)
    
    # Valid input
    input_data = ValidatedInput(email="test@example.com", age=30)
    assert input_data.email == "test@example.com"
    assert input_data.age == 30
    
    # Invalid email
    with pytest.raises(ValidationError):
        ValidatedInput(email="invalid-email", age=30)
    
    # Invalid age
    with pytest.raises(ValidationError):
        ValidatedInput(email="test@example.com", age=150)


def test_tool_error_model_serialization():
    """Test ToolErrorModel serialization."""
    error = ToolErrorModel(
        code="TOOL003",
        message="Test error",
        tool_name="test_tool",
    )
    
    # To dict
    error_dict = error.to_dict()
    assert error_dict["code"] == "TOOL003"
    assert error_dict["message"] == "Test error"
    
    # To JSON
    json_str = error.to_json()
    assert "TOOL003" in json_str
    assert "Test error" in json_str
