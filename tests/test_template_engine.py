"""
Comprehensive tests for PromptTemplateEngine.

Tests cover:
- Basic variable substitution
- Conditionals (if/else)
- Loops (for)
- Nested object access
- Custom filters
- Security (code execution attempts)
- Error handling
- Template compilation
"""

import pytest
from datetime import datetime

from namel3ss.templates import (
    PromptTemplateEngine,
    CompiledTemplate,
    TemplateCompilationError,
    TemplateRenderError,
    TemplateSecurityError,
    create_engine,
    get_default_engine,
)


class TestBasicVariableSubstitution:
    """Test basic variable substitution functionality."""
    
    def test_simple_variable(self):
        """Test simple variable substitution."""
        engine = create_engine()
        result = engine.render("Hello {{ name }}!", {"name": "World"})
        assert result == "Hello World!"
    
    def test_multiple_variables(self):
        """Test multiple variable substitution."""
        engine = create_engine()
        result = engine.render(
            "{{ greeting }} {{ name }}!",
            {"greeting": "Hello", "name": "World"},
        )
        assert result == "Hello World!"
    
    def test_nested_object_access(self):
        """Test accessing nested object properties."""
        engine = create_engine()
        result = engine.render(
            "User: {{ user.profile.name }}, Age: {{ user.profile.age }}",
            {
                "user": {
                    "profile": {
                        "name": "Alice",
                        "age": 30,
                    }
                }
            },
        )
        assert result == "User: Alice, Age: 30"
    
    def test_list_access(self):
        """Test accessing list items by index."""
        engine = create_engine()
        result = engine.render(
            "First: {{ items[0] }}, Second: {{ items[1] }}",
            {"items": ["apple", "banana", "cherry"]},
        )
        assert result == "First: apple, Second: banana"
    
    def test_undefined_variable_strict(self):
        """Test that undefined variables raise error in strict mode."""
        engine = create_engine(strict_undefined=True)
        with pytest.raises(TemplateRenderError) as exc_info:
            engine.render("Hello {{ name }}!", {})
        assert "Undefined variable" in str(exc_info.value)
    
    def test_whitespace_trimming(self):
        """Test that trim_blocks and lstrip_blocks work correctly."""
        engine = create_engine()
        template = """
        {% if True %}
        Content
        {% endif %}
        """
        result = engine.render(template, {})
        # Should trim leading/trailing whitespace from blocks
        assert "Content" in result
        assert result.strip() == "Content"


class TestConditionals:
    """Test conditional rendering."""
    
    def test_if_true(self):
        """Test if block when condition is true."""
        engine = create_engine()
        result = engine.render(
            "{% if show %}Visible{% endif %}",
            {"show": True},
        )
        assert result == "Visible"
    
    def test_if_false(self):
        """Test if block when condition is false."""
        engine = create_engine()
        result = engine.render(
            "{% if show %}Visible{% endif %}",
            {"show": False},
        )
        assert result == ""
    
    def test_if_else(self):
        """Test if-else blocks."""
        engine = create_engine()
        
        result1 = engine.render(
            "{% if premium %}Premium{% else %}Free{% endif %}",
            {"premium": True},
        )
        assert result1 == "Premium"
        
        result2 = engine.render(
            "{% if premium %}Premium{% else %}Free{% endif %}",
            {"premium": False},
        )
        assert result2 == "Free"
    
    def test_if_elif_else(self):
        """Test if-elif-else chain."""
        engine = create_engine()
        template = "{% if score >= 90 %}A{% elif score >= 80 %}B{% else %}C{% endif %}"
        
        assert engine.render(template, {"score": 95}) == "A"
        assert engine.render(template, {"score": 85}) == "B"
        assert engine.render(template, {"score": 70}) == "C"
    
    def test_nested_conditionals(self):
        """Test nested if statements."""
        engine = create_engine()
        template = """
        {% if outer %}
        Outer
        {% if inner %}Inner{% endif %}
        {% endif %}
        """
        
        result = engine.render(template, {"outer": True, "inner": True})
        assert "Outer" in result
        assert "Inner" in result
        
        result = engine.render(template, {"outer": True, "inner": False})
        assert "Outer" in result
        assert "Inner" not in result


class TestLoops:
    """Test loop rendering."""
    
    def test_for_loop_list(self):
        """Test for loop over list."""
        engine = create_engine()
        result = engine.render(
            "{% for item in items %}{{ item }}, {% endfor %}",
            {"items": ["a", "b", "c"]},
        )
        assert result == "a, b, c, "
    
    def test_for_loop_dict(self):
        """Test for loop over dictionary."""
        engine = create_engine()
        result = engine.render(
            "{% for key, value in data.items() %}{{ key }}={{ value }}, {% endfor %}",
            {"data": {"x": 1, "y": 2}},
        )
        assert "x=1" in result
        assert "y=2" in result
    
    def test_for_loop_with_index(self):
        """Test for loop with loop.index."""
        engine = create_engine()
        result = engine.render(
            "{% for item in items %}{{ loop.index }}:{{ item }}, {% endfor %}",
            {"items": ["a", "b", "c"]},
        )
        assert result == "1:a, 2:b, 3:c, "
    
    def test_for_loop_first_last(self):
        """Test loop.first and loop.last."""
        engine = create_engine()
        template = """
        {% for item in items %}
        {% if loop.first %}First: {% endif %}{{ item }}{% if loop.last %} Last{% endif %}
        {% endfor %}
        """
        result = engine.render(template, {"items": ["a", "b", "c"]})
        assert "First: a" in result
        assert "c Last" in result
    
    def test_nested_loops(self):
        """Test nested for loops."""
        engine = create_engine()
        template = """
        {% for row in matrix %}
        {% for col in row %}{{ col }} {% endfor %}
        {% endfor %}
        """
        result = engine.render(
            template,
            {"matrix": [[1, 2], [3, 4]]},
        )
        assert "1 2" in result
        assert "3 4" in result
    
    def test_for_loop_empty(self):
        """Test for loop with empty list."""
        engine = create_engine()
        result = engine.render(
            "{% for item in items %}{{ item }}{% endfor %}",
            {"items": []},
        )
        assert result == ""


class TestFilters:
    """Test custom filters."""
    
    def test_truncate_filter(self):
        """Test truncate filter."""
        engine = create_engine()
        result = engine.render(
            "{{ text|truncate(10, '...') }}",
            {"text": "This is a very long text"},
        )
        assert result == "This is..."
        assert len(result) == 10
    
    def test_truncate_short_text(self):
        """Test truncate with text shorter than limit."""
        engine = create_engine()
        result = engine.render(
            "{{ text|truncate(50) }}",
            {"text": "Short"},
        )
        assert result == "Short"
    
    def test_title_filter(self):
        """Test title case filter."""
        engine = create_engine()
        result = engine.render(
            "{{ text|title }}",
            {"text": "hello world"},
        )
        assert result == "Hello World"
    
    def test_uppercase_filter(self):
        """Test uppercase filter."""
        engine = create_engine()
        result = engine.render(
            "{{ text|uppercase }}",
            {"text": "hello"},
        )
        assert result == "HELLO"
    
    def test_lowercase_filter(self):
        """Test lowercase filter."""
        engine = create_engine()
        result = engine.render(
            "{{ text|lowercase }}",
            {"text": "HELLO"},
        )
        assert result == "hello"
    
    def test_strip_filter(self):
        """Test strip whitespace filter."""
        engine = create_engine()
        result = engine.render(
            "{{ text|strip }}",
            {"text": "  hello  "},
        )
        assert result == "hello"
    
    def test_json_encode_filter(self):
        """Test JSON encoding filter."""
        engine = create_engine()
        result = engine.render(
            "{{ data|json_encode }}",
            {"data": {"key": "value", "num": 42}},
        )
        assert '"key": "value"' in result or '"key":"value"' in result
        assert "42" in result
    
    def test_json_encode_indent(self):
        """Test JSON encoding with indentation."""
        engine = create_engine()
        result = engine.render(
            "{{ data|json_encode(2) }}",
            {"data": {"key": "value"}},
        )
        assert "\n" in result  # Indented JSON has newlines
    
    def test_list_join_filter(self):
        """Test list join filter."""
        engine = create_engine()
        result = engine.render(
            "{{ items|list_join(', ') }}",
            {"items": ["apple", "banana", "cherry"]},
        )
        assert result == "apple, banana, cherry"
    
    def test_default_filter(self):
        """Test default value filter."""
        engine = create_engine()
        result = engine.render(
            "{{ value|default('N/A') }}",
            {"value": None},
        )
        assert result == "N/A"
        
        result = engine.render(
            "{{ value|default('N/A') }}",
            {"value": ""},
        )
        assert result == "N/A"
        
        result = engine.render(
            "{{ value|default('N/A') }}",
            {"value": "Present"},
        )
        assert result == "Present"
    
    def test_length_filter(self):
        """Test length filter."""
        engine = create_engine()
        result = engine.render(
            "{{ items|length }}",
            {"items": [1, 2, 3, 4, 5]},
        )
        assert result == "5"
    
    def test_format_date_filter(self):
        """Test date formatting filter."""
        engine = create_engine()
        dt = datetime(2024, 1, 15, 14, 30)
        result = engine.render(
            "{{ date|format_date('%Y-%m-%d') }}",
            {"date": dt},
        )
        assert result == "2024-01-15"
    
    def test_format_date_iso_string(self):
        """Test date formatting with ISO string input."""
        engine = create_engine()
        result = engine.render(
            "{{ date|format_date('%Y-%m-%d') }}",
            {"date": "2024-01-15T14:30:00"},
        )
        assert result == "2024-01-15"
    
    def test_chained_filters(self):
        """Test chaining multiple filters."""
        engine = create_engine()
        result = engine.render(
            "{{ text|strip|uppercase|truncate(5) }}",
            {"text": "  hello world  "},
        )
        assert result == "HE..."


class TestSecurity:
    """Test security features and sandboxing."""
    
    def test_no_import(self):
        """Test that __import__ is not accessible."""
        engine = create_engine()
        template = "{{ __import__('os') }}"
        with pytest.raises((TemplateSecurityError, TemplateCompilationError)) as exc_info:
            engine.compile(template)
        # Should raise security-related error
        assert "dangerous" in str(exc_info.value).lower() or "import" in str(exc_info.value).lower()
    
    def test_no_builtins(self):
        """Test that __builtins__ is not accessible."""
        engine = create_engine()
        template = "{{ __builtins__ }}"
        with pytest.raises((TemplateSecurityError, TemplateCompilationError)) as exc_info:
            engine.compile(template)
        assert "dangerous" in str(exc_info.value).lower() or "builtins" in str(exc_info.value).lower()
    
    def test_no_eval(self):
        """Test that eval is not accessible."""
        engine = create_engine()
        template = "{{ eval('1+1') }}"
        with pytest.raises((TemplateSecurityError, TemplateCompilationError)) as exc_info:
            engine.compile(template)
        assert "dangerous" in str(exc_info.value).lower() or "eval" in str(exc_info.value).lower()
    
    def test_no_exec(self):
        """Test that exec is not accessible."""
        engine = create_engine()
        template = "{{ exec('print(1)') }}"
        with pytest.raises((TemplateSecurityError, TemplateCompilationError)) as exc_info:
            engine.compile(template)
        assert "dangerous" in str(exc_info.value).lower() or "exec" in str(exc_info.value).lower()
    
    def test_no_open(self):
        """Test that open is not accessible."""
        engine = create_engine()
        template = "{{ open('/etc/passwd') }}"
        with pytest.raises((TemplateSecurityError, TemplateCompilationError)) as exc_info:
            engine.compile(template)
        assert "dangerous" in str(exc_info.value).lower() or "open" in str(exc_info.value).lower()
    
    def test_no_class_access(self):
        """Test that __class__ is not accessible."""
        engine = create_engine()
        template = "{{ ''.__class__ }}"
        with pytest.raises((TemplateSecurityError, TemplateCompilationError)) as exc_info:
            engine.compile(template)
        assert "dangerous" in str(exc_info.value).lower() or "class" in str(exc_info.value).lower()
    
    def test_safe_globals_only(self):
        """Test that only safe globals are available."""
        engine = create_engine()
        # These should work (safe functions)
        result = engine.render("{{ range(3)|list|length }}", {})
        assert result == "3"
        
        result = engine.render("{{ min([1, 2, 3]) }}", {})
        assert result == "1"
        
        result = engine.render("{{ max([1, 2, 3]) }}", {})
        assert result == "3"
    
    def test_context_isolation(self):
        """Test that templates don't share context."""
        engine = create_engine()
        result1 = engine.render("{{ x }}", {"x": 1})
        result2 = engine.render("{{ x }}", {"x": 2})
        assert result1 == "1"
        assert result2 == "2"


class TestCompilation:
    """Test template compilation and validation."""
    
    def test_compile_valid_template(self):
        """Test compiling a valid template."""
        engine = create_engine()
        compiled = engine.compile("Hello {{ name }}!", name="test")
        assert isinstance(compiled, CompiledTemplate)
        assert compiled.name == "test"
        assert "name" in compiled.required_vars
    
    def test_compile_syntax_error(self):
        """Test that syntax errors are caught during compilation."""
        engine = create_engine()
        with pytest.raises(TemplateCompilationError):
            engine.compile("{% if %}", name="invalid")
    
    def test_compile_unclosed_block(self):
        """Test that unclosed blocks are caught."""
        engine = create_engine()
        with pytest.raises(TemplateCompilationError):
            engine.compile("{% if True %}", name="unclosed")
    
    def test_extract_required_variables(self):
        """Test extraction of required variables."""
        engine = create_engine()
        compiled = engine.compile("{{ a }} and {{ b }} and {{ c }}", name="test")
        assert compiled.required_vars == {"a", "b", "c"}
    
    def test_extract_variables_in_loops(self):
        """Test variable extraction from loops."""
        engine = create_engine()
        compiled = engine.compile(
            "{% for item in items %}{{ item }}{% endfor %}",
            name="test",
        )
        assert "items" in compiled.required_vars
        assert "item" not in compiled.required_vars  # loop variable
    
    def test_reuse_compiled_template(self):
        """Test that compiled templates can be reused."""
        engine = create_engine()
        compiled = engine.compile("Hello {{ name }}!", name="test")
        
        result1 = compiled.render({"name": "Alice"})
        result2 = compiled.render({"name": "Bob"})
        
        assert result1 == "Hello Alice!"
        assert result2 == "Hello Bob!"


class TestValidation:
    """Test variable validation."""
    
    def test_validate_all_variables_provided(self):
        """Test validation when all variables are provided."""
        engine = create_engine()
        compiled = engine.compile("{{ a }} {{ b }}", name="test")
        missing = engine.validate_variables(compiled, {"a": 1, "b": 2})
        assert missing == []
    
    def test_validate_missing_variables(self):
        """Test validation detects missing variables."""
        engine = create_engine()
        compiled = engine.compile("{{ a }} {{ b }} {{ c }}", name="test")
        missing = engine.validate_variables(compiled, {"a": 1})
        assert set(missing) == {"b", "c"}
    
    def test_validate_extra_variables_ok(self):
        """Test that extra variables don't cause issues."""
        engine = create_engine()
        compiled = engine.compile("{{ a }}", name="test")
        missing = engine.validate_variables(compiled, {"a": 1, "b": 2, "c": 3})
        assert missing == []


class TestErrorHandling:
    """Test error handling and messages."""
    
    def test_render_error_includes_template_name(self):
        """Test that render errors include template name."""
        engine = create_engine()
        compiled = engine.compile("{{ undefined_var }}", name="my_template")
        with pytest.raises(TemplateRenderError) as exc_info:
            compiled.render({})
        assert exc_info.value.template_name == "my_template"
    
    def test_compilation_error_includes_line_number(self):
        """Test that compilation errors include line numbers."""
        engine = create_engine()
        template = """
        Line 1
        {% if True
        Line 3
        """
        with pytest.raises(TemplateCompilationError) as exc_info:
            engine.compile(template, name="test")
        # Should have line number info
        assert exc_info.value.line_number is not None


class TestFactory:
    """Test factory functions."""
    
    def test_create_engine_default(self):
        """Test creating engine with defaults."""
        engine = create_engine()
        assert isinstance(engine, PromptTemplateEngine)
        assert engine.env.autoescape is False
    
    def test_create_engine_custom(self):
        """Test creating engine with custom settings."""
        engine = create_engine(autoescape=True, strict_undefined=False)
        assert engine.env.autoescape is True
    
    def test_get_default_engine(self):
        """Test getting default global engine."""
        engine1 = get_default_engine()
        engine2 = get_default_engine()
        assert engine1 is engine2  # Same instance
    
    def test_custom_filters(self):
        """Test registering custom filters."""
        def custom_filter(value):
            return f"[{value}]"
        
        engine = create_engine(custom_filters={"brackets": custom_filter})
        result = engine.render("{{ text|brackets }}", {"text": "hello"})
        assert result == "[hello]"


class TestComplexTemplates:
    """Test complex real-world template scenarios."""
    
    def test_ai_prompt_with_context(self):
        """Test realistic AI prompt template."""
        engine = create_engine()
        template = """
You are {{ role }}.

Context:
{% for doc in documents %}
- {{ doc.title }}: {{ doc.content|truncate(100) }}
{% endfor %}

Task: {{ task }}

{% if examples %}
Examples:
{% for ex in examples %}
Input: {{ ex.input }}
Output: {{ ex.output }}
{% endfor %}
{% endif %}

Please respond with your analysis.
"""
        result = engine.render(
            template,
            {
                "role": "a helpful assistant",
                "documents": [
                    {"title": "Doc1", "content": "Content 1"},
                    {"title": "Doc2", "content": "Content 2"},
                ],
                "task": "Summarize the documents",
                "examples": [
                    {"input": "test", "output": "result"},
                ],
            },
        )
        assert "helpful assistant" in result
        assert "Doc1" in result
        assert "Summarize" in result
        assert "Examples:" in result
    
    def test_conditional_formatting(self):
        """Test conditional formatting in prompts."""
        engine = create_engine()
        template = """
Generate a response {% if formal %}in formal style{% else %}in casual style{% endif %}.
{% if max_words %}Limit to {{ max_words }} words.{% endif %}
"""
        result = engine.render(
            template,
            {"formal": True, "max_words": 100},
        )
        assert "formal style" in result
        assert "100 words" in result
    
    def test_nested_data_structures(self):
        """Test accessing deeply nested data."""
        engine = create_engine()
        template = """
User: {{ user.profile.personal.name }}
Email: {{ user.profile.contact.email }}
Role: {{ user.permissions.role }}
"""
        result = engine.render(
            template,
            {
                "user": {
                    "profile": {
                        "personal": {"name": "Alice"},
                        "contact": {"email": "alice@example.com"},
                    },
                    "permissions": {"role": "admin"},
                }
            },
        )
        assert "Alice" in result
        assert "alice@example.com" in result
        assert "admin" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
