"""
Integration tests for TemplatePrompt with Jinja2 template engine.

Tests the integration between TemplatePrompt and PromptTemplateEngine,
ensuring backward compatibility and new advanced features work correctly.
"""

import pytest

from namel3ss.prompts.template_prompt import TemplatePrompt
from namel3ss.prompts.base import PromptError


class TestBasicTemplatePrompt:
    """Test basic TemplatePrompt functionality."""
    
    def test_simple_variable_substitution(self):
        """Test simple variable substitution."""
        prompt = TemplatePrompt(
            name="greeting",
            template="Hello {{ name }}!",
        )
        result = prompt.render(name="World")
        assert result.rendered == "Hello World!"
        assert result.variables == {"name": "World"}
    
    def test_multiple_variables(self):
        """Test multiple variable substitution."""
        prompt = TemplatePrompt(
            name="intro",
            template="{{ greeting }} {{ name }}, welcome to {{ place }}!",
        )
        result = prompt.render(
            greeting="Hello",
            name="Alice",
            place="Wonderland",
        )
        assert result.rendered == "Hello Alice, welcome to Wonderland!"
    
    def test_template_compilation_at_init(self):
        """Test that templates are compiled during initialization."""
        prompt = TemplatePrompt(
            name="test",
            template="Hello {{ name }}!",
        )
        # Should have compiled template
        assert hasattr(prompt, "_compiled_template")
        assert "name" in prompt._compiled_template.required_vars


class TestDefaultValues:
    """Test default value handling."""
    
    def test_default_value_used(self):
        """Test that default values are applied."""
        prompt = TemplatePrompt(
            name="greeting",
            template="Hello {{ name }}!",
            args={
                "name": {"default": "World"},
            },
        )
        result = prompt.render()
        assert result.rendered == "Hello World!"
    
    def test_provided_value_overrides_default(self):
        """Test that provided values override defaults."""
        prompt = TemplatePrompt(
            name="greeting",
            template="Hello {{ name }}!",
            args={
                "name": {"default": "World"},
            },
        )
        result = prompt.render(name="Alice")
        assert result.rendered == "Hello Alice!"
    
    def test_multiple_defaults(self):
        """Test multiple default values."""
        prompt = TemplatePrompt(
            name="message",
            template="{{ greeting }} {{ name }}!",
            args={
                "greeting": {"default": "Hello"},
                "name": {"default": "World"},
            },
        )
        result = prompt.render()
        assert result.rendered == "Hello World!"
        
        result = prompt.render(greeting="Hi")
        assert result.rendered == "Hi World!"
        
        result = prompt.render(name="Alice")
        assert result.rendered == "Hello Alice!"


class TestRequiredVariables:
    """Test required variable validation."""
    
    def test_missing_required_variable_error(self):
        """Test that missing required variables raise error."""
        prompt = TemplatePrompt(
            name="greeting",
            template="Hello {{ name }}!",
        )
        with pytest.raises(PromptError) as exc_info:
            prompt.render()
        assert "Missing required variables" in str(exc_info.value)
        assert "name" in exc_info.value.missing_vars
    
    def test_get_required_variables(self):
        """Test getting required variable names."""
        prompt = TemplatePrompt(
            name="test",
            template="{{ a }} and {{ b }} and {{ c }}",
        )
        required = prompt.get_required_variables()
        assert required == {"a", "b", "c"}
    
    def test_required_with_defaults_not_required(self):
        """Test that variables with defaults are not required."""
        prompt = TemplatePrompt(
            name="test",
            template="{{ a }} and {{ b }}",
            args={
                "a": {"default": "A"},
                "b": {"required": True},
            },
        )
        # Should only fail on b
        with pytest.raises(PromptError) as exc_info:
            prompt.render()
        assert "b" in exc_info.value.missing_vars


class TestConditionals:
    """Test conditional rendering in templates."""
    
    def test_if_block(self):
        """Test if block rendering."""
        prompt = TemplatePrompt(
            name="conditional",
            template="{% if show %}Visible{% endif %}",
        )
        
        result = prompt.render(show=True)
        assert result.rendered == "Visible"
        
        result = prompt.render(show=False)
        assert result.rendered == ""
    
    def test_if_else_block(self):
        """Test if-else rendering."""
        prompt = TemplatePrompt(
            name="conditional",
            template="{% if premium %}Premium User{% else %}Free User{% endif %}",
        )
        
        result = prompt.render(premium=True)
        assert result.rendered == "Premium User"
        
        result = prompt.render(premium=False)
        assert result.rendered == "Free User"
    
    def test_complex_conditional(self):
        """Test complex conditional logic."""
        prompt = TemplatePrompt(
            name="score",
            template="""
Grade: {% if score >= 90 %}A{% elif score >= 80 %}B{% elif score >= 70 %}C{% else %}F{% endif %}
""",
        )
        
        assert "A" in prompt.render(score=95).rendered
        assert "B" in prompt.render(score=85).rendered
        assert "C" in prompt.render(score=75).rendered
        assert "F" in prompt.render(score=65).rendered


class TestLoops:
    """Test loop rendering in templates."""
    
    def test_for_loop_list(self):
        """Test for loop over list."""
        prompt = TemplatePrompt(
            name="list",
            template="""
Items:
{% for item in items %}
- {{ item }}
{% endfor %}
""",
        )
        result = prompt.render(items=["apple", "banana", "cherry"])
        assert "- apple" in result.rendered
        assert "- banana" in result.rendered
        assert "- cherry" in result.rendered
    
    def test_for_loop_with_index(self):
        """Test for loop with index."""
        prompt = TemplatePrompt(
            name="numbered",
            template="""
{% for item in items %}
{{ loop.index }}. {{ item }}
{% endfor %}
""",
        )
        result = prompt.render(items=["first", "second", "third"])
        assert "1. first" in result.rendered
        assert "2. second" in result.rendered
        assert "3. third" in result.rendered
    
    def test_nested_loops(self):
        """Test nested loops."""
        prompt = TemplatePrompt(
            name="matrix",
            template="""
{% for row in matrix %}
{% for col in row %}{{ col }} {% endfor %}
{% endfor %}
""",
        )
        result = prompt.render(matrix=[[1, 2], [3, 4], [5, 6]])
        assert "1 2" in result.rendered
        assert "3 4" in result.rendered
        assert "5 6" in result.rendered


class TestFilters:
    """Test filters in templates."""
    
    def test_truncate_filter(self):
        """Test truncate filter."""
        prompt = TemplatePrompt(
            name="summary",
            template="Summary: {{ text|truncate(20) }}",
        )
        result = prompt.render(text="This is a very long piece of text that needs truncation")
        assert len(result.rendered.split(": ")[1]) <= 20
    
    def test_uppercase_filter(self):
        """Test uppercase filter."""
        prompt = TemplatePrompt(
            name="upper",
            template="{{ text|uppercase }}",
        )
        result = prompt.render(text="hello world")
        assert result.rendered == "HELLO WORLD"
    
    def test_list_join_filter(self):
        """Test list join filter."""
        prompt = TemplatePrompt(
            name="joined",
            template="Tags: {{ tags|list_join(', ') }}",
        )
        result = prompt.render(tags=["python", "ai", "template"])
        assert result.rendered == "Tags: python, ai, template"
    
    def test_json_encode_filter(self):
        """Test JSON encoding filter."""
        prompt = TemplatePrompt(
            name="json",
            template="Data: {{ data|json_encode }}",
        )
        result = prompt.render(data={"key": "value", "number": 42})
        assert "key" in result.rendered
        assert "value" in result.rendered
        assert "42" in result.rendered
    
    def test_chained_filters(self):
        """Test chaining multiple filters."""
        prompt = TemplatePrompt(
            name="chained",
            template="{{ text|strip|uppercase|truncate(10) }}",
        )
        result = prompt.render(text="  hello world  ")
        assert len(result.rendered) <= 10
        assert result.rendered.isupper() or result.rendered.endswith("...")


class TestNestedObjects:
    """Test nested object access."""
    
    def test_nested_dict_access(self):
        """Test accessing nested dictionary values."""
        prompt = TemplatePrompt(
            name="profile",
            template="Name: {{ user.profile.name }}, Age: {{ user.profile.age }}",
        )
        result = prompt.render(
            user={
                "profile": {
                    "name": "Alice",
                    "age": 30,
                }
            }
        )
        assert result.rendered == "Name: Alice, Age: 30"
    
    def test_list_access(self):
        """Test accessing list items."""
        prompt = TemplatePrompt(
            name="items",
            template="First: {{ items[0] }}, Last: {{ items[-1] }}",
        )
        result = prompt.render(items=["apple", "banana", "cherry"])
        assert "First: apple" in result.rendered
        assert "Last: cherry" in result.rendered
    
    def test_deeply_nested_access(self):
        """Test deeply nested object access."""
        prompt = TemplatePrompt(
            name="deep",
            template="{{ data.level1.level2.level3.value }}",
        )
        result = prompt.render(
            data={
                "level1": {
                    "level2": {
                        "level3": {
                            "value": "deep_value"
                        }
                    }
                }
            }
        )
        assert result.rendered == "deep_value"


class TestErrorHandling:
    """Test error handling and messages."""
    
    def test_compilation_error_at_init(self):
        """Test that template syntax errors are caught at initialization."""
        with pytest.raises(PromptError) as exc_info:
            TemplatePrompt(
                name="invalid",
                template="{% if True %}unclosed",
            )
        assert "compile" in str(exc_info.value).lower()
    
    def test_render_error_with_name(self):
        """Test that render errors include prompt name."""
        prompt = TemplatePrompt(
            name="test_prompt",
            template="{{ undefined }}",
        )
        with pytest.raises(PromptError) as exc_info:
            prompt.render()
        assert exc_info.value.prompt_name == "test_prompt"
    
    def test_metadata_includes_engine(self):
        """Test that result metadata indicates engine used."""
        prompt = TemplatePrompt(
            name="test",
            template="Hello {{ name }}!",
        )
        result = prompt.render(name="World")
        assert result.metadata.get("engine") == "jinja2"


class TestComplexAIPrompts:
    """Test complex AI prompt scenarios."""
    
    def test_rag_context_prompt(self):
        """Test RAG-style prompt with document context."""
        prompt = TemplatePrompt(
            name="rag_qa",
            template="""
You are a helpful assistant. Answer the question based on the context below.

Context:
{% for doc in documents %}
Document {{ loop.index }}:
{{ doc.content }}
{% if doc.metadata %}Metadata: {{ doc.metadata|json_encode }}{% endif %}

{% endfor %}

Question: {{ question }}

Provide a detailed answer based on the context.
""",
        )
        result = prompt.render(
            documents=[
                {
                    "content": "Python is a programming language.",
                    "metadata": {"source": "doc1.txt"},
                },
                {
                    "content": "It was created by Guido van Rossum.",
                    "metadata": {"source": "doc2.txt"},
                },
            ],
            question="What is Python?",
        )
        assert "Document 1" in result.rendered
        assert "Document 2" in result.rendered
        assert "Python is a programming language" in result.rendered
        assert "What is Python?" in result.rendered
    
    def test_few_shot_prompt(self):
        """Test few-shot learning prompt."""
        prompt = TemplatePrompt(
            name="few_shot",
            template="""
Task: {{ task }}

{% if examples %}
Examples:
{% for example in examples %}
Input: {{ example.input }}
Output: {{ example.output }}

{% endfor %}
{% endif %}

Now perform the task:
Input: {{ input }}
Output:
""",
        )
        result = prompt.render(
            task="Classify sentiment",
            examples=[
                {"input": "I love this!", "output": "positive"},
                {"input": "This is terrible", "output": "negative"},
            ],
            input="This is amazing!",
        )
        assert "Classify sentiment" in result.rendered
        assert "I love this!" in result.rendered
        assert "positive" in result.rendered
        assert "This is amazing!" in result.rendered
    
    def test_conditional_instructions(self):
        """Test prompt with conditional instructions."""
        prompt = TemplatePrompt(
            name="conditional_instruct",
            template="""
Generate a response for: {{ request }}

Instructions:
{% if formal %}Use formal, professional language.{% else %}Use casual, friendly language.{% endif %}
{% if max_length %}Limit response to {{ max_length }} words.{% endif %}
{% if include_examples %}Include specific examples.{% endif %}
{% if tone %}Tone: {{ tone }}{% endif %}
""",
            args={
                "formal": {"default": False},
                "include_examples": {"default": False},
                "tone": {"required": False, "default": None},
            },
        )
        result = prompt.render(
            request="Explain quantum computing",
            formal=True,
            max_length=100,
        )
        assert "formal, professional" in result.rendered
        assert "100 words" in result.rendered
    
    def test_system_user_prompt(self):
        """Test system + user message prompt."""
        prompt = TemplatePrompt(
            name="chat",
            template="""
System: {{ system_message }}

{% if chat_history %}
Conversation History:
{% for msg in chat_history %}
{{ msg.role }}: {{ msg.content }}
{% endfor %}
{% endif %}

User: {{ user_message }}
""",
        )
        result = prompt.render(
            system_message="You are a helpful assistant",
            chat_history=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
            user_message="How are you?",
        )
        assert "You are a helpful assistant" in result.rendered
        assert "Hi" in result.rendered
        assert "Hello!" in result.rendered
        assert "How are you?" in result.rendered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
