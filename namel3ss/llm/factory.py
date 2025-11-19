"""Factory for creating and registering LLM instances."""

import os
from typing import Dict, Any, Optional, Type
from .base import BaseLLM, LLMError
from .registry import get_registry


# Provider class registry - populated when provider modules are imported
_PROVIDER_CLASSES: Dict[str, Type[BaseLLM]] = {}


def register_provider(name: str, provider_class: Type[BaseLLM]) -> None:
    """
    Register a provider class.
    
    Args:
        name: Provider name (e.g., 'openai', 'anthropic')
        provider_class: The provider class to register
    """
    _PROVIDER_CLASSES[name.lower()] = provider_class


def get_provider_class(provider: str) -> Type[BaseLLM]:
    """
    Get a provider class by name.
    
    Args:
        provider: Provider name (e.g., 'openai', 'anthropic')
    
    Returns:
        The provider class
    
    Raises:
        LLMError: If the provider is not registered
    """
    provider_key = provider.lower()
    
    # Lazy import providers on first use
    if not _PROVIDER_CLASSES:
        _load_providers()
    
    if provider_key not in _PROVIDER_CLASSES:
        available = ', '.join(sorted(_PROVIDER_CLASSES.keys()))
        raise LLMError(
            f"Unknown LLM provider '{provider}'. "
            f"Available providers: {available or 'none'}"
        )
    
    return _PROVIDER_CLASSES[provider_key]


def _load_providers() -> None:
    """Lazy load all provider modules."""
    # Import each provider module to trigger registration
    try:
        from . import openai_llm  # noqa: F401
    except ImportError:
        pass
    
    try:
        from . import anthropic_llm  # noqa: F401
    except ImportError:
        pass
    
    try:
        from . import vertex_llm  # noqa: F401
    except ImportError:
        pass
    
    try:
        from . import azure_openai_llm  # noqa: F401
    except ImportError:
        pass
    
    try:
        from . import ollama_llm  # noqa: F401
    except ImportError:
        pass


def create_llm(
    name: str,
    provider: str,
    model: str,
    config: Optional[Dict[str, Any]] = None,
    register: bool = True,
) -> BaseLLM:
    """
    Create an LLM instance.
    
    Args:
        name: Logical name for the LLM
        provider: Provider name ('openai', 'anthropic', 'vertex', 'azure_openai', 'ollama')
        model: Model identifier (e.g., 'gpt-4', 'claude-3-opus')
        config: Additional configuration (temperature, max_tokens, etc.)
        register: Whether to register the LLM in the global registry
    
    Returns:
        The created LLM instance
    
    Raises:
        LLMError: If the provider is unknown or configuration is invalid
    
    Example:
        >>> llm = create_llm('my_gpt4', 'openai', 'gpt-4', {'temperature': 0.7})
        >>> response = llm.generate('Hello!')
    """
    provider_class = get_provider_class(provider)
    
    # Merge default config with provided config
    merged_config = config or {}
    
    # Resolve environment variables for common config keys
    if 'api_key' not in merged_config:
        api_key = _resolve_api_key(provider)
        if api_key:
            merged_config['api_key'] = api_key
    
    # Create the LLM instance
    try:
        llm = provider_class(name=name, model=model, config=merged_config)
    except Exception as e:
        raise LLMError(
            f"Failed to create {provider} LLM '{name}' with model '{model}': {e}",
            provider=provider,
            model=model
        ) from e
    
    # Register if requested
    if register:
        registry = get_registry()
        registry.update(llm)
    
    return llm


def _resolve_api_key(provider: str) -> Optional[str]:
    """
    Resolve API key from environment variables.
    
    Args:
        provider: Provider name
    
    Returns:
        The API key, or None if not found
    """
    provider_lower = provider.lower()
    
    # Map provider to environment variable name
    env_var_map = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'vertex': 'GOOGLE_CLOUD_PROJECT',  # Vertex uses project ID
        'azure_openai': 'AZURE_OPENAI_API_KEY',
        'ollama': None,  # Ollama doesn't require an API key
    }
    
    env_var = env_var_map.get(provider_lower)
    if env_var:
        return os.environ.get(env_var)
    
    return None


def register_llm(llm: BaseLLM) -> None:
    """
    Register an LLM instance in the global registry.
    
    Args:
        llm: The LLM instance to register
    
    Raises:
        ValueError: If an LLM with the same name is already registered
    
    Example:
        >>> from namel3ss.llm.openai_llm import OpenAILLM
        >>> llm = OpenAILLM('my_gpt4', 'gpt-4')
        >>> register_llm(llm)
    """
    registry = get_registry()
    registry.register(llm)
