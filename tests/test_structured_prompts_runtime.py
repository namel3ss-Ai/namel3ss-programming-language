"""Tests for PromptProgram runtime execution."""

import pytest
from namel3ss.prompts.runtime import PromptProgram
from namel3ss.ast import (
    Prompt, PromptArgument, OutputSchema, OutputField, OutputFieldType
)


class TestPromptProgramArguments:
    """Test PromptProgram argument handling."""
    
    def test_render_with_valid_args(self):
        """Test rendering with valid arguments."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Hello {name}, you are {age} years old",
            args=[
                PromptArgument(name="name", arg_type="string"),
                PromptArgument(name="age", arg_type="int"),
            ]
        )
        program = PromptProgram(prompt)
        
        rendered = program.render_prompt({"name": "Alice", "age": 25})
        assert rendered == "Hello Alice, you are 25 years old"
    
    def test_apply_defaults(self):
        """Test that default values are applied."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Max words: {max_words}",
            args=[
                PromptArgument(name="max_words", arg_type="int", required=False, default=100),
            ]
        )
        program = PromptProgram(prompt)
        
        rendered = program.render_prompt({})
        assert rendered == "Max words: 100"
    
    def test_missing_required_arg(self):
        """Test that missing required args raise error."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Hello {name}",
            args=[
                PromptArgument(name="name", arg_type="string", required=True),
            ]
        )
        program = PromptProgram(prompt)
        
        with pytest.raises(ValueError, match="Missing required argument.*name"):
            program.render_prompt({})
    
    def test_unexpected_arg(self):
        """Test that unexpected args are rejected."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Hello",
            args=[]
        )
        program = PromptProgram(prompt)
        
        with pytest.raises(ValueError, match="Unexpected argument.*unknown"):
            program.render_prompt({"unknown": "value"})


class TestTypeCoercion:
    """Test type coercion for arguments."""
    
    def test_coerce_string(self):
        """Test string coercion."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="{value}",
            args=[PromptArgument(name="value", arg_type="string")]
        )
        program = PromptProgram(prompt)
        
        # int to string
        rendered = program.render_prompt({"value": 123})
        assert rendered == "123"
    
    def test_coerce_int(self):
        """Test int coercion."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="{value}",
            args=[PromptArgument(name="value", arg_type="int")]
        )
        program = PromptProgram(prompt)
        
        # string to int
        rendered = program.render_prompt({"value": "42"})
        assert rendered == "42"
        
        # float to int (truncate)
        rendered = program.render_prompt({"value": 42.7})
        assert rendered == "42"
    
    def test_coerce_float(self):
        """Test float coercion."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="{value}",
            args=[PromptArgument(name="value", arg_type="float")]
        )
        program = PromptProgram(prompt)
        
        # string to float
        rendered = program.render_prompt({"value": "3.14"})
        assert rendered == "3.14"
        
        # int to float
        rendered = program.render_prompt({"value": 42})
        assert rendered == "42.0"
    
    def test_coerce_bool(self):
        """Test bool coercion."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="{value}",
            args=[PromptArgument(name="value", arg_type="bool")]
        )
        program = PromptProgram(prompt)
        
        # string to bool
        rendered = program.render_prompt({"value": "true"})
        assert rendered == "True"
        
        rendered = program.render_prompt({"value": "false"})
        assert rendered == "False"
    
    def test_invalid_coercion(self):
        """Test that invalid coercion raises error."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="{value}",
            args=[PromptArgument(name="value", arg_type="int")]
        )
        program = PromptProgram(prompt)
        
        with pytest.raises(ValueError, match="Cannot coerce.*int"):
            program.render_prompt({"value": "not_a_number"})


class TestOutputSchemaGeneration:
    """Test JSON Schema generation from OutputSchema."""
    
    def test_simple_schema(self):
        """Test generating schema for primitives."""
        schema = OutputSchema(fields=[
            OutputField(name="text", field_type=OutputFieldType(base_type="string"), required=True),
            OutputField(name="count", field_type=OutputFieldType(base_type="int"), required=True),
        ])
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            output_schema=schema
        )
        program = PromptProgram(prompt)
        
        json_schema = program.get_output_schema()
        assert json_schema["type"] == "object"
        assert "text" in json_schema["properties"]
        assert "count" in json_schema["properties"]
        assert json_schema["properties"]["text"]["type"] == "string"
        assert json_schema["properties"]["count"]["type"] == "integer"
        assert set(json_schema["required"]) == {"text", "count"}
    
    def test_enum_schema(self):
        """Test generating schema for enums."""
        schema = OutputSchema(fields=[
            OutputField(
                name="category",
                field_type=OutputFieldType(base_type="enum", enum_values=["a", "b", "c"]),
                required=True
            ),
        ])
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            output_schema=schema
        )
        program = PromptProgram(prompt)
        
        json_schema = program.get_output_schema()
        assert json_schema["properties"]["category"]["enum"] == ["a", "b", "c"]
    
    def test_list_schema(self):
        """Test generating schema for lists."""
        schema = OutputSchema(fields=[
            OutputField(
                name="tags",
                field_type=OutputFieldType(
                    base_type="list",
                    element_type=OutputFieldType(base_type="string")
                ),
                required=True
            ),
        ])
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            output_schema=schema
        )
        program = PromptProgram(prompt)
        
        json_schema = program.get_output_schema()
        assert json_schema["properties"]["tags"]["type"] == "array"
        assert json_schema["properties"]["tags"]["items"]["type"] == "string"
    
    def test_optional_fields(self):
        """Test that optional fields are not in required list."""
        schema = OutputSchema(fields=[
            OutputField(name="required", field_type=OutputFieldType(base_type="string"), required=True),
            OutputField(name="optional", field_type=OutputFieldType(base_type="string"), required=False),
        ])
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Test",
            output_schema=schema
        )
        program = PromptProgram(prompt)
        
        json_schema = program.get_output_schema()
        assert json_schema["required"] == ["required"]
        assert "optional" not in json_schema["required"]


class TestComplexTemplates:
    """Test rendering of complex templates."""
    
    def test_multiline_template(self):
        """Test rendering multiline templates."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="""
Line 1: {arg1}
Line 2: {arg2}
Line 3: {arg3}
""".strip(),
            args=[
                PromptArgument(name="arg1", arg_type="string"),
                PromptArgument(name="arg2", arg_type="string"),
                PromptArgument(name="arg3", arg_type="string"),
            ]
        )
        program = PromptProgram(prompt)
        
        rendered = program.render_prompt({"arg1": "A", "arg2": "B", "arg3": "C"})
        assert "Line 1: A" in rendered
        assert "Line 2: B" in rendered
        assert "Line 3: C" in rendered
    
    def test_list_in_template(self):
        """Test rendering list arguments in template."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Items: {items}",
            args=[PromptArgument(name="items", arg_type="list")]
        )
        program = PromptProgram(prompt)
        
        rendered = program.render_prompt({"items": ["a", "b", "c"]})
        assert "['a', 'b', 'c']" in rendered or '["a", "b", "c"]' in rendered
    
    def test_object_in_template(self):
        """Test rendering object arguments in template."""
        prompt = Prompt(
            name="test",
            model="gpt4",
            template="Data: {data}",
            args=[PromptArgument(name="data", arg_type="object")]
        )
        program = PromptProgram(prompt)
        
        rendered = program.render_prompt({"data": {"key": "value"}})
        assert "key" in rendered
        assert "value" in rendered
