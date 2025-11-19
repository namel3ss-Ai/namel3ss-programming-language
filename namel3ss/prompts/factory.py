"""Prompt factory for creating prompt instances."""

from typing import Any, Dict, Optional, Type

from .base import BasePrompt, PromptError
from .registry import get_registry


# Registry of prompt provider classes
_PROMPT_PROVIDERS: Dict[str, Type[BasePrompt]] = {}


def register_provider(prompt_type: str, provider_class: Type[BasePrompt]) -> None:
    """
    Register a prompt provider class for a given type.
    
    Args:
        prompt_type: Prompt type identifier (e.g., "template", "few_shot")
        provider_class: Class implementing BasePrompt
    """
    _PROMPT_PROVIDERS[prompt_type] = provider_class


def get_provider_class(prompt_type: str = "template") -> Type[BasePrompt]:
    """
    Get the provider class for a prompt type.
    
    Args:
        prompt_type: Prompt type identifier
        
    Returns:
        Provider class
        
    Raises:
        PromptError: If provider not found
    """
    # Try lazy loading for built-in providers
    if prompt_type not in _PROMPT_PROVIDERS:
        if prompt_type == "template":
            from .template_prompt import TemplatePrompt
            register_provider("template", TemplatePrompt)
        # Add more built-in types as needed
    
    provider_class = _PROMPT_PROVIDERS.get(prompt_type)
    if provider_class is None:
        raise PromptError(f"Unknown prompt type: {prompt_type}")
    
    return provider_class


def create_prompt(
    name: str,
    template: str,
    *,
    model: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
    prompt_type: str = "template",
    register: bool = False,
    **config: Any,
) -> BasePrompt:
    """
    Create a prompt instance using the factory pattern.
    
    Args:
        name: Prompt identifier
        template: Template string with {variable} placeholders
        model: Target LLM model name
        args: Argument specifications
        prompt_type: Type of prompt (default: "template")
        register: If True, register in global registry
        **config: Additional configuration
        
    Returns:
        Instantiated prompt
        
    Raises:
        PromptError: If prompt type unknown or creation fails
        
    Example:
        prompt = create_prompt(
            name="summarize",
            template="Summarize {text} in {max_length} words",
            model="gpt-4",
            register=True
        )
    """
    provider_class = get_provider_class(prompt_type)
    
    # Build config dict
    full_config = {
        "name": name,
        "template": template,
        "model": model,
        "args": args,
        **config,
    }
    
    try:
        prompt = provider_class(**full_config)
    except Exception as e:
        raise PromptError(
            f"Failed to create prompt '{name}': {e}",
            prompt_name=name,
            original_error=e,
        )
    
    if register:
        registry = get_registry()
        registry.update(name, prompt)
    
    return prompt


def register_prompt(name: str, prompt: BasePrompt) -> None:
    """
    Register a prompt instance in the global registry.
    
    Args:
        name: Prompt identifier
        prompt: Prompt instance
        
    Raises:
        ValueError: If prompt already registered
    """
    registry = get_registry()
    registry.register(name, prompt)
