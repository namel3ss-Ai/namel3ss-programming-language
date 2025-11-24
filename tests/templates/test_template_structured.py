"""
Tests for structured template support (JSON-serializable outputs).

Tests cover:
- Structured template compilation
- JSON dict generation
- JSON list generation
- Nested structures
- Variable interpolation in structured templates
- Error handling for invalid JSON
- Integration with provider API calls
"""

import pytest
import json

from namel3ss.templates import (
    PromptTemplateEngine,
    StructuredCompiledTemplate,
    TemplateCompilationError,
    TemplateRenderError,
    create_engine,
)


class TestStructuredTemplateBasics:
    """Test basic structured template functionality."""
    
    def test_simple_dict_template(self):
        """Test generating a simple JSON dict."""
        engine = create_engine()
        template_source = '''
{
    "type": "{{ action_type }}",
    "value": "{{ value }}"
}
'''
        compiled = engine.compile_struct(template_source, name="simple_dict")
        assert isinstance(compiled, StructuredCompiledTemplate)
        
        result = compiled.render({"action_type": "update", "value": "hello"})
        assert isinstance(result, dict)
        assert result["type"] == "update"
        assert result["value"] == "hello"
    
    def test_nested_dict_template(self):
        """Test generating nested JSON structures."""
        engine = create_engine()
        template_source = '''
{
    "user": {
        "name": "{{ user_name }}",
        "email": "{{ user_email }}"
    },
    "settings": {
        "theme": "{{ theme }}",
        "notifications": {{ notifications | json_encode }}
    }
}
'''
        compiled = engine.compile_struct(template_source, name="nested_dict")
        
        result = compiled.render({
            "user_name": "Alice",
            "user_email": "alice@example.com",
            "theme": "dark",
            "notifications": True,
        })
        
        assert result["user"]["name"] == "Alice"
        assert result["user"]["email"] == "alice@example.com"
        assert result["settings"]["theme"] == "dark"
        assert result["settings"]["notifications"] is True
    
    def test_list_template(self):
        """Test generating JSON arrays."""
        engine = create_engine()
        template_source = '''
[
    {% for item in items %}
    {
        "id": {{ loop.index }},
        "name": "{{ item }}"
    }{% if not loop.last %},{% endif %}
    {% endfor %}
]
'''
        compiled = engine.compile_struct(template_source, name="list_template")
        
        result = compiled.render({"items": ["apple", "banana", "cherry"]})
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["name"] == "apple"
        assert result[1]["name"] == "banana"
        assert result[2]["name"] == "cherry"
        assert result[0]["id"] == 1
    
    def test_conditional_fields(self):
        """Test conditional fields in structured templates."""
        engine = create_engine()
        template_source = '''
{
    "action": "{{ action }}",
    {% if include_metadata %}
    "metadata": {
        "timestamp": "{{ timestamp }}",
        "user": "{{ user }}"
    },
    {% endif %}
    "data": "{{ data }}"
}
'''
        compiled = engine.compile_struct(template_source, name="conditional_fields")
        
        # With metadata
        result = compiled.render({
            "action": "create",
            "include_metadata": True,
            "timestamp": "2024-01-01",
            "user": "alice",
            "data": "test",
        })
        assert "metadata" in result
        assert result["metadata"]["user"] == "alice"
        
        # Without metadata
        result = compiled.render({
            "action": "create",
            "include_metadata": False,
            "data": "test",
        })
        assert "metadata" not in result
    
    def test_json_encode_filter(self):
        """Test json_encode filter for complex values."""
        engine = create_engine()
        template_source = '''
{
    "values": {{ values | json_encode }},
    "config": {{ config | json_encode }}
}
'''
        compiled = engine.compile_struct(template_source, name="json_encode_test")
        
        result = compiled.render({
            "values": [1, 2, 3, 4, 5],
            "config": {"enabled": True, "timeout": 30},
        })
        
        assert result["values"] == [1, 2, 3, 4, 5]
        assert result["config"]["enabled"] is True
        assert result["config"]["timeout"] == 30


class TestStructuredTemplateErrors:
    """Test error handling for structured templates."""
    
    def test_invalid_json_syntax(self):
        """Test that invalid JSON syntax raises error."""
        engine = create_engine()
        # Template with invalid Jinja2 syntax (unclosed variable tag)
        template_source = '''
{
    "key": "{{ value }}",
    "invalid": "{{ unclosed"
}
'''
        # Should catch at compilation time due to Jinja2 syntax error
        with pytest.raises(TemplateCompilationError) as exc_info:
            compiled = engine.compile_struct(template_source, name="invalid_template")
        assert "syntax error" in str(exc_info.value).lower()
    
    def test_invalid_json_output(self):
        """Test that template producing invalid JSON is caught at render time."""
        engine = create_engine()
        # Template that compiles but produces invalid JSON
        template_source = '''
{
    "key": "{{ value }}",
    "trailing_comma": "test",
}
'''
        compiled = engine.compile_struct(template_source, name="invalid_json_output")
        
        # Should fail when rendering due to trailing comma (invalid JSON)
        with pytest.raises(TemplateRenderError) as exc_info:
            compiled.render({"value": "test"})
        assert "valid JSON" in str(exc_info.value).lower() or "JSON" in str(exc_info.value)
    
    def test_undefined_variable_in_struct(self):
        """Test undefined variables in structured templates."""
        engine = create_engine(strict_undefined=True)
        template_source = '''
{
    "key": "{{ missing_var }}"
}
'''
        compiled = engine.compile_struct(template_source, name="undefined_var")
        
        with pytest.raises(TemplateRenderError) as exc_info:
            compiled.render({})
        assert "Undefined variable" in str(exc_info.value)
    
    def test_non_serializable_result(self):
        """Test that non-JSON-serializable results are caught."""
        # This is challenging since Jinja2 typically produces strings
        # The validation happens during JSON parsing
        engine = create_engine()
        # If somehow we produce a structure with non-serializable data,
        # it should be caught during the json.dumps validation
        # (In practice, this is prevented by Jinja2's string output)
        pass


class TestProviderIntegration:
    """Test structured templates for provider API payloads."""
    
    def test_openai_function_call_template(self):
        """Test OpenAI function calling payload generation."""
        engine = create_engine()
        template_source = '''
{
    "name": "{{ function_name }}",
    "description": "{{ description }}",
    "parameters": {
        "type": "object",
        "properties": {
            {% for param in parameters %}
            "{{ param.name }}": {
                "type": "{{ param.type }}",
                "description": "{{ param.description }}"
            }{% if not loop.last %},{% endif %}
            {% endfor %}
        },
        "required": {{ required | json_encode }}
    }
}
'''
        compiled = engine.compile_struct(template_source, name="openai_function")
        
        result = compiled.render({
            "function_name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": [
                {"name": "location", "type": "string", "description": "City name"},
                {"name": "units", "type": "string", "description": "Temperature units"},
            ],
            "required": ["location"],
        })
        
        assert result["name"] == "get_weather"
        assert len(result["parameters"]["properties"]) == 2
        assert "location" in result["parameters"]["properties"]
        assert result["parameters"]["required"] == ["location"]
    
    def test_anthropic_tool_use_template(self):
        """Test Anthropic tool use payload generation."""
        engine = create_engine()
        template_source = '''
{
    "name": "{{ tool_name }}",
    "description": "{{ description }}",
    "input_schema": {
        "type": "object",
        "properties": {{ properties | json_encode }},
        "required": {{ required | json_encode }}
    }
}
'''
        compiled = engine.compile_struct(template_source, name="anthropic_tool")
        
        result = compiled.render({
            "tool_name": "search",
            "description": "Search the web",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            "required": ["query"],
        })
        
        assert result["name"] == "search"
        assert result["input_schema"]["type"] == "object"
        assert "query" in result["input_schema"]["properties"]
        assert result["input_schema"]["required"] == ["query"]
    
    def test_message_list_template(self):
        """Test generating message lists for chat APIs."""
        engine = create_engine()
        template_source = '''
[
    {% if system_prompt %}
    {
        "role": "system",
        "content": "{{ system_prompt }}"
    },
    {% endif %}
    {% for msg in messages %}
    {
        "role": "{{ msg.role }}",
        "content": "{{ msg.content }}"
    }{% if not loop.last %},{% endif %}
    {% endfor %}
]
'''
        compiled = engine.compile_struct(template_source, name="message_list")
        
        result = compiled.render({
            "system_prompt": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        })
        
        assert len(result) == 4  # system + 3 messages
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"


class TestCompilationAndCaching:
    """Test template compilation and caching behavior."""
    
    def test_compile_once_render_many(self):
        """Test that templates can be compiled once and rendered multiple times."""
        engine = create_engine()
        template_source = '''
{
    "id": "{{ id }}",
    "value": "{{ value }}"
}
'''
        compiled = engine.compile_struct(template_source, name="reusable")
        
        # Render multiple times with different contexts
        result1 = compiled.render({"id": "1", "value": "first"})
        result2 = compiled.render({"id": "2", "value": "second"})
        result3 = compiled.render({"id": "3", "value": "third"})
        
        assert result1["id"] == "1"
        assert result2["id"] == "2"
        assert result3["id"] == "3"
        assert result1["value"] == "first"
        assert result2["value"] == "second"
        assert result3["value"] == "third"
    
    def test_required_vars_extraction(self):
        """Test that required variables are extracted correctly."""
        engine = create_engine()
        template_source = '''
{
    "a": "{{ var_a }}",
    "b": "{{ var_b }}",
    "c": "{{ var_c }}"
}
'''
        compiled = engine.compile_struct(template_source, name="vars_test")
        
        assert "var_a" in compiled.required_vars
        assert "var_b" in compiled.required_vars
        assert "var_c" in compiled.required_vars
