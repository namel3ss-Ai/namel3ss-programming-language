"""Tests for structured prompt parsing."""

import pytest
from namel3ss.parser.ai import AIParserMixin
from namel3ss.ast import Prompt, PromptArgument, OutputSchema, OutputField, OutputFieldType


class TestPromptArgumentParsing:
    """Test parsing of prompt arguments."""
    
    def test_parse_simple_args(self):
        """Test parsing simple typed arguments."""
        text = """
prompt test {
    model: gpt4
    args: {
        text: string,
        count: int
    }
    template: "Test {text} {count}"
}
"""
        parser = AIParser(text)
        prompt = parser._parse_prompt(0)
        
        assert prompt.name == "test"
        assert len(prompt.args) == 2
        assert prompt.args[0].name == "text"
        assert prompt.args[0].arg_type == "string"
        assert prompt.args[0].required is True
        assert prompt.args[1].name == "count"
        assert prompt.args[1].arg_type == "int"
    
    def test_parse_args_with_defaults(self):
        """Test parsing arguments with default values."""
        text = """
prompt test {
    model: gpt4
    args: {
        text: string,
        max_words: int = 100,
        style: string = "concise"
    }
    template: "Test"
}
"""
        parser = AIParser(text)
        prompt = parser._parse_prompt(0)
        
        assert len(prompt.args) == 3
        assert prompt.args[0].default is None
        assert prompt.args[0].required is True
        assert prompt.args[1].default == 100
        assert prompt.args[1].required is False
        assert prompt.args[2].default == "concise"
        assert prompt.args[2].required is False
    
    def test_parse_all_arg_types(self):
        """Test parsing all supported argument types."""
        text = """
prompt test {
    model: gpt4
    args: {
        str_arg: string,
        int_arg: int,
        float_arg: float,
        bool_arg: bool,
        list_arg: list,
        obj_arg: object
    }
    template: "Test"
}
"""
        parser = AIParser(text)
        prompt = parser._parse_prompt(0)
        
        types = [arg.arg_type for arg in prompt.args]
        assert types == ["string", "int", "float", "bool", "list", "object"]


class TestOutputSchemaParsing:
    """Test parsing of output schemas."""
    
    def test_parse_simple_schema(self):
        """Test parsing simple output schema with primitives."""
        text = """
prompt test {
    model: gpt4
    template: "Test"
    output_schema: {
        text: string,
        count: int,
        score: float,
        active: bool
    }
}
"""
        parser = AIParser(text)
        prompt = parser._parse_prompt(0)
        
        assert prompt.output_schema is not None
        assert len(prompt.output_schema.fields) == 4
        
        fields = {f.name: f for f in prompt.output_schema.fields}
        assert fields["text"].field_type.base_type == "string"
        assert fields["count"].field_type.base_type == "int"
        assert fields["score"].field_type.base_type == "float"
        assert fields["active"].field_type.base_type == "bool"
    
    def test_parse_enum_type(self):
        """Test parsing enum types in output schema."""
        text = """
prompt test {
    model: gpt4
    template: "Test"
    output_schema: {
        category: enum["spam", "important", "social"]
    }
}
"""
        parser = AIParser(text)
        prompt = parser._parse_prompt(0)
        
        field = prompt.output_schema.fields[0]
        assert field.name == "category"
        assert field.field_type.base_type == "enum"
        assert field.field_type.enum_values == ["spam", "important", "social"]
    
    def test_parse_list_type(self):
        """Test parsing list types in output schema."""
        text = """
prompt test {
    model: gpt4
    template: "Test"
    output_schema: {
        tags: list[string],
        scores: list[float]
    }
}
"""
        parser = AIParser(text)
        prompt = parser._parse_prompt(0)
        
        fields = {f.name: f for f in prompt.output_schema.fields}
        assert fields["tags"].field_type.base_type == "list"
        assert fields["tags"].field_type.element_type.base_type == "string"
        assert fields["scores"].field_type.base_type == "list"
        assert fields["scores"].field_type.element_type.base_type == "float"
    
    def test_parse_nested_object(self):
        """Test parsing nested object types."""
        text = """
prompt test {
    model: gpt4
    template: "Test"
    output_schema: {
        metadata: {
            author: string,
            timestamp: int
        }
    }
}
"""
        parser = AIParser(text)
        prompt = parser._parse_prompt(0)
        
        field = prompt.output_schema.fields[0]
        assert field.name == "metadata"
        assert field.field_type.base_type == "object"
        assert len(field.field_type.nested_fields) == 2
        
        nested = {f.name: f for f in field.field_type.nested_fields}
        assert nested["author"].field_type.base_type == "string"
        assert nested["timestamp"].field_type.base_type == "int"
    
    def test_parse_complex_nested_schema(self):
        """Test parsing complex nested structures."""
        text = """
prompt test {
    model: gpt4
    template: "Test"
    output_schema: {
        entities: list[{
            name: string,
            type: enum["person", "place", "thing"],
            confidence: float
        }]
    }
}
"""
        parser = AIParser(text)
        prompt = parser._parse_prompt(0)
        
        field = prompt.output_schema.fields[0]
        assert field.field_type.base_type == "list"
        assert field.field_type.element_type.base_type == "object"
        
        nested = {f.name: f for f in field.field_type.element_type.nested_fields}
        assert nested["name"].field_type.base_type == "string"
        assert nested["type"].field_type.base_type == "enum"
        assert nested["type"].field_type.enum_values == ["person", "place", "thing"]
        assert nested["confidence"].field_type.base_type == "float"


class TestParsingErrors:
    """Test error handling in parsing."""
    
    def test_invalid_arg_type(self):
        """Test that invalid argument types raise errors."""
        text = """
prompt test {
    model: gpt4
    args: { x: invalid_type }
    template: "Test"
}
"""
        parser = AIParser(text)
        with pytest.raises(Exception):
            parser._parse_prompt(0)
    
    def test_invalid_enum_syntax(self):
        """Test that malformed enum syntax raises errors."""
        text = """
prompt test {
    model: gpt4
    template: "Test"
    output_schema: {
        category: enum[spam, important]
    }
}
"""
        parser = AIParser(text)
        with pytest.raises(Exception):
            parser._parse_prompt(0)
    
    def test_missing_default_type(self):
        """Test that defaults must match type."""
        text = """
prompt test {
    model: gpt4
    args: { count: int = "not_a_number" }
    template: "Test"
}
"""
        parser = AIParser(text)
        # Parser should handle this, but validator should catch it
        prompt = parser._parse_prompt(0)
        assert prompt.args[0].default == "not_a_number"


class TestBackwardCompatibility:
    """Test that legacy prompts still work."""
    
    def test_legacy_prompt_without_args(self):
        """Test parsing legacy prompts without args."""
        text = """
prompt test {
    model: gpt4
    template: "Hello world"
}
"""
        parser = AIParser(text)
        prompt = parser._parse_prompt(0)
        
        assert prompt.name == "test"
        assert len(prompt.args) == 0
        assert prompt.output_schema is None
    
    def test_legacy_prompt_with_input_fields(self):
        """Test parsing legacy prompts with input_fields."""
        text = """
prompt test {
    model: gpt4
    input: [
        { name: "text", type: "string" }
    ]
    template: "Process {text}"
}
"""
        parser = AIParser(text)
        prompt = parser._parse_prompt(0)
        
        assert len(prompt.input_fields) == 1
        assert len(prompt.args) == 0
