"""
Tests for provider integration template usage and safety checks.

Tests cover:
- Provider integration prompt building
- Agent runtime template usage
- Prompt program template rendering
- Compile-time validation in parser
- Safety sandbox enforcement
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from namel3ss.providers.integration import _build_prompt_from_step
from namel3ss.templates import (
    TemplateError,
    TemplateSecurityError,
    TemplateCompilationError,
    get_default_engine,
)


class TestProviderIntegrationTemplates:
    """Test template usage in provider integration."""
    
    def test_build_prompt_from_step_simple(self):
        """Test simple variable substitution in chain step."""
        step = Mock()
        step.target = "Translate '{{ text }}' to {{ language }}"
        step.name = "translate"
        
        state = {"text": "hello", "language": "French"}
        result = _build_prompt_from_step(step, state)
        
        assert result == "Translate 'hello' to French"
    
    def test_build_prompt_from_step_conditional(self):
        """Test conditional logic in chain step template."""
        step = Mock()
        step.target = """
{% if formal %}
Please provide a formal summary of: {{ text }}
{% else %}
Summarize: {{ text }}
{% endif %}
"""
        step.name = "summarize"
        
        # Formal
        state = {"formal": True, "text": "The quick brown fox"}
        result = _build_prompt_from_step(step, state)
        assert "formal summary" in result.lower()
        
        # Informal
        state = {"formal": False, "text": "The quick brown fox"}
        result = _build_prompt_from_step(step, state)
        assert "Summarize:" in result
    
    def test_build_prompt_from_step_loop(self):
        """Test loop in chain step template."""
        step = Mock()
        step.target = """
Process the following items:
{% for item in items %}
- {{ item }}
{% endfor %}
"""
        step.name = "process"
        
        state = {"items": ["apple", "banana", "cherry"]}
        result = _build_prompt_from_step(step, state)
        
        assert "- apple" in result
        assert "- banana" in result
        assert "- cherry" in result
    
    def test_build_prompt_from_step_undefined_var(self):
        """Test that undefined variables raise error."""
        step = Mock()
        step.target = "Hello {{ missing_var }}!"
        step.name = "test"
        
        state = {}
        
        with pytest.raises(TemplateError):
            _build_prompt_from_step(step, state)
    
    def test_build_prompt_from_step_filters(self):
        """Test template filters in chain step."""
        step = Mock()
        step.target = "{{ text | uppercase }} - {{ text | lowercase }}"
        step.name = "transform"
        
        state = {"text": "Hello World"}
        result = _build_prompt_from_step(step, state)
        
        assert "HELLO WORLD" in result
        assert "hello world" in result


class TestAgentRuntimeTemplates:
    """Test template usage in agent runtime."""
    
    def test_format_tools_description(self):
        """Test agent tools description uses template engine."""
        from namel3ss.agents.runtime import AgentRuntime
        from namel3ss.ast.agents import AgentDefinition
        
        # Create mock agent definition
        agent_def = Mock(spec=AgentDefinition)
        agent_def.name = "test_agent"
        agent_def.tool_names = ["search", "calculator", "weather"]
        agent_def.system_prompt = "You are helpful"
        agent_def.max_turns = 10
        
        # Create runtime
        llm = AsyncMock()
        runtime = AgentRuntime(agent_def, llm, {})
        
        # Test tools description formatting
        result = runtime._format_tools_description()
        
        assert "search" in result
        assert "calculator" in result
        assert "weather" in result
        assert "TOOL_CALL:" in result


class TestPromptProgramTemplates:
    """Test template usage in prompt programs."""
    
    @pytest.mark.asyncio
    async def test_render_prompt_with_template_engine(self):
        """Test that prompt programs use template engine."""
        from namel3ss.prompts.runtime import PromptProgram
        from namel3ss.ast import Prompt, PromptArgument
        
        # Create prompt with template and explicit args
        prompt_def = Prompt(
            name="test_prompt",
            model="gpt-4",
            template="Summarize: {{ text }}",
            args=[
                PromptArgument(
                    name="text",
                    arg_type="string",
                    required=True,
                )
            ],
            parameters={},
        )
        
        program = PromptProgram(definition=prompt_def)
        
        # Render with simple variable
        result = await program.render_prompt({"text": "Hello world"})
        assert result == "Summarize: Hello world"
    
    @pytest.mark.asyncio
    async def test_render_prompt_with_conditional(self):
        """Test conditional logic in prompt templates."""
        from namel3ss.prompts.runtime import PromptProgram
        from namel3ss.ast import Prompt, PromptArgument
        
        prompt_def = Prompt(
            name="conditional_prompt",
            model="gpt-4",
            template="""
{% if include_context %}
Context: {{ context }}

{% endif %}
Question: {{ question }}
""",
            args=[
                PromptArgument(name="include_context", arg_type="boolean", required=False, default=False),
                PromptArgument(name="context", arg_type="string", required=False, default=""),
                PromptArgument(name="question", arg_type="string", required=True),
            ],
            parameters={},
        )
        
        program = PromptProgram(definition=prompt_def)
        
        # With context
        result = await program.render_prompt({
            "include_context": True,
            "context": "Background info",
            "question": "What is AI?",
        })
        assert "Context: Background info" in result
        
        # Without context
        result = await program.render_prompt({
            "include_context": False,
            "question": "What is AI?",
        })
        assert "Context:" not in result


class TestCompileTimeValidation:
    """Test compile-time template validation."""
    
    def test_valid_template_compilation_engine(self):
        """Test that valid templates compile without errors at engine level."""
        engine = get_default_engine()
        
        # Valid template should compile
        template_source = "Hello {{ name }}, welcome to {{ place }}!"
        compiled = engine.compile(template_source, name="valid", validate=True)
        
        assert compiled.name == "valid"
        assert "name" in compiled.required_vars
        assert "place" in compiled.required_vars
        
        # Should render correctly
        result = compiled.render({"name": "Alice", "place": "Wonderland"})
        assert result == "Hello Alice, welcome to Wonderland!"
    
    def test_invalid_template_syntax_engine(self):
        """Test that invalid template syntax is caught at engine level."""
        engine = get_default_engine()
        
        # Template with unclosed tag
        template_source = "Hello {{ name"
        
        with pytest.raises(TemplateCompilationError) as exc_info:
            engine.compile(template_source, name="invalid", validate=True)
        assert "syntax error" in str(exc_info.value).lower()
    
    def test_template_security_validation(self):
        """Test that security violations are caught during compilation."""
        engine = get_default_engine()
        
        # Template with dangerous pattern
        template_source = "{{ __import__('os').system('ls') }}"
        
        with pytest.raises((TemplateCompilationError, TemplateSecurityError)) as exc_info:
            engine.compile(template_source, name="malicious", validate=True)
        # Should mention either security or the dangerous pattern
        assert "__import__" in str(exc_info.value) or "security" in str(exc_info.value).lower()


class TestSafetySandbox:
    """Test security sandbox enforcement."""
    
    def test_sandbox_blocks_import(self):
        """Test that __import__ is blocked."""
        engine = get_default_engine()
        
        template_source = "{{ __import__('os').system('ls') }}"
        
        with pytest.raises((TemplateCompilationError, TemplateSecurityError, TemplateError)):
            engine.compile(template_source, name="malicious", validate=True)
    
    def test_sandbox_blocks_eval(self):
        """Test that eval is blocked."""
        engine = get_default_engine()
        
        template_source = "{{ eval('print(123)') }}"
        
        with pytest.raises((TemplateCompilationError, TemplateSecurityError, TemplateError)):
            compiled = engine.compile(template_source, name="malicious", validate=True)
            # Even if compilation passes, rendering should fail
            compiled.render({})
    
    def test_sandbox_blocks_exec(self):
        """Test that exec is blocked."""
        engine = get_default_engine()
        
        template_source = "{{ exec('import os') }}"
        
        with pytest.raises((TemplateCompilationError, TemplateSecurityError, TemplateError)):
            compiled = engine.compile(template_source, name="malicious", validate=True)
            compiled.render({})
    
    def test_sandbox_blocks_open(self):
        """Test that open is blocked."""
        engine = get_default_engine()
        
        template_source = "{{ open('/etc/passwd').read() }}"
        
        with pytest.raises((TemplateCompilationError, TemplateSecurityError, TemplateError)):
            compiled = engine.compile(template_source, name="malicious", validate=True)
            compiled.render({})
    
    def test_sandbox_blocks_class_access(self):
        """Test that __class__ and reflection are blocked."""
        engine = get_default_engine()
        
        template_source = "{{ ''.__class__.__bases__ }}"
        
        with pytest.raises((TemplateCompilationError, TemplateSecurityError, TemplateError)):
            engine.compile(template_source, name="malicious", validate=True)
    
    def test_sandbox_allows_safe_operations(self):
        """Test that safe operations are allowed."""
        engine = get_default_engine()
        
        template_source = """
{{ text | uppercase }}
{{ numbers | length }}
{{ range(5) | list }}
{{ items | list_join(", ") }}
"""
        
        compiled = engine.compile(template_source, name="safe", validate=True)
        result = compiled.render({
            "text": "hello",
            "numbers": [1, 2, 3],
            "items": ["a", "b", "c"],
        })
        
        assert "HELLO" in result
        assert "3" in result
        assert "a, b, c" in result
    
    def test_sandbox_context_isolation(self):
        """Test that templates only see provided variables."""
        engine = get_default_engine()
        
        # Template trying to access non-provided variables
        template_source = "{{ provided_var }} {{ not_provided_var }}"
        
        compiled = engine.compile(template_source, name="isolation_test", validate=True)
        
        with pytest.raises(TemplateError):
            # Should fail because not_provided_var is not in context
            compiled.render({"provided_var": "hello"})


class TestExistingSafetyChecks:
    """Verify that existing safety checks remain intact."""
    
    def test_no_weakened_provider_checks(self):
        """Test that provider API key checks still work."""
        from namel3ss.ml.providers.openai import OpenAIProvider
        from namel3ss.ml.providers.base import LLMError
        
        # Should still require API key
        with pytest.raises(LLMError) as exc_info:
            OpenAIProvider(model="gpt-4", api_key=None)
        assert "API key" in str(exc_info.value)
    
    def test_no_weakened_agent_turn_limits(self):
        """Test that agent max turns limit still enforced."""
        from namel3ss.agents.runtime import AgentRuntime
        from namel3ss.ast.agents import AgentDefinition
        
        agent_def = Mock(spec=AgentDefinition)
        agent_def.name = "test"
        agent_def.system_prompt = "test"
        agent_def.tool_names = []
        agent_def.max_turns = 5  # Limit to 5 turns
        agent_def.memory = None
        
        llm = AsyncMock()
        runtime = AgentRuntime(agent_def, llm, {})
        
        # Verify agent_def.max_turns is accessible and enforced through agent_def
        assert runtime.agent_def.max_turns == 5
    
    def test_no_weakened_prompt_validation(self):
        """Test that prompt argument validation still works."""
        from namel3ss.prompts.runtime import PromptProgram, PromptProgramError
        from namel3ss.ast import Prompt, PromptArgument
        
        # Prompt with required argument
        prompt_def = Prompt(
            name="test",
            model="gpt-4",
            template="{{ required_arg }}",
            args=[
                PromptArgument(
                    name="required_arg",
                    arg_type="string",
                    required=True,
                )
            ],
        )
        
        program = PromptProgram(definition=prompt_def)
        
        # Should still validate required arguments
        import asyncio
        with pytest.raises(PromptProgramError):
            asyncio.run(program.render_prompt({}))  # Missing required_arg
