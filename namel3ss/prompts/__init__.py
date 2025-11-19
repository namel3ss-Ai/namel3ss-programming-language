"""
Prompt subsystem for Namel3ss - first-class prompt templates with variable substitution.

This package provides:
- BasePrompt: Abstract interface for prompt templates
- PromptRegistry: Central registry for prompt instances
- Prompt factory: create_prompt() for instantiating prompts
- Template rendering with variable substitution

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
"""

from .base import BasePrompt, PromptResult, PromptError
from .registry import PromptRegistry, get_registry, reset_registry
from .factory import create_prompt, register_provider

__all__ = [
    "BasePrompt",
    "PromptResult",
    "PromptError",
    "PromptRegistry",
    "get_registry",
    "reset_registry",
    "create_prompt",
    "register_provider",
]
