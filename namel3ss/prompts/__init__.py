"""
Prompt subsystem for Namel3ss - first-class prompt templates with variable substitution.

This package provides:
- BasePrompt: Abstract interface for prompt templates
- PromptRegistry: Central registry for prompt instances
- Prompt factory: create_prompt() for instantiating prompts
- Template rendering with variable substitution
- PromptProgram: Structured prompts with typed args and output schemas
- OutputValidator: Validation of LLM outputs against schemas

Example usage:
    from namel3ss.prompts import create_prompt, get_registry
    
    # Create a prompt
    summarize = create_prompt(
        name="summarize",
        template="Summarize the following text in {max_length} words: {text}",
        model="gpt-4"
    )
    
    # Render prompt
    rendered = summarize.render(text="Long article...", max_length=100)
    print(rendered)
    
    # Structured prompts
    from namel3ss.prompts.runtime import PromptProgram
    from namel3ss.prompts.validator import OutputValidator
    
    program = PromptProgram(prompt_definition)
    rendered = program.render_prompt({"text": "Hello"})
    schema = program.get_output_schema()
"""

from .base import BasePrompt, PromptResult, PromptError
from .registry import PromptRegistry, get_registry, reset_registry
from .factory import create_prompt, register_provider
from .runtime import PromptProgram, PromptProgramError, create_prompt_program
from .validator import OutputValidator, ValidationError, ValidationResult, validate_output
from .executor import (
    StructuredPromptError,
    StructuredPromptResult,
    execute_structured_prompt,
    execute_structured_prompt_sync,
)

__all__ = [
    "BasePrompt",
    "PromptResult",
    "PromptError",
    "PromptRegistry",
    "get_registry",
    "reset_registry",
    "create_prompt",
    "register_provider",
    "PromptProgram",
    "PromptProgramError",
    "create_prompt_program",
    "OutputValidator",
    "ValidationError",
    "ValidationResult",
    "validate_output",
    "StructuredPromptError",
    "StructuredPromptResult",
    "execute_structured_prompt",
    "execute_structured_prompt_sync",
]
