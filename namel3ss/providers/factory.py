"""Provider factory for creating N3Provider instances.

Central factory for instantiating providers from DSL specifications
and managing provider instances through a registry.

Key Functions:
    - create_provider_from_spec: Main entry point for creating providers
    - register_provider_class: Register a provider implementation
    - get_provider_class: Get a provider class by type name
    - register_provider_instance: Register a provider instance for reuse
    - get_provider_instance: Get a registered provider instance

Example:
    >>> from namel3ss.providers.factory import create_provider_from_spec
    >>> 
    >>> provider = create_provider_from_spec(
    ...     name="my_gpt4",
    ...     provider_type="openai",
    ...     model="gpt-4",
    ...     config={"temperature": 0.7, "max_tokens": 1000}
    ... )
"""

from typing import Dict, Any, Optional, Type, TYPE_CHECKING
from .base import N3Provider, ProviderError
from .config import load_config_for_provider, ProviderConfigError

if TYPE_CHECKING:
    from .validation import validate_provider_name, validate_model_name


# Provider class registry - populated lazily when provider modules are imported
_PROVIDER_CLASSES: Dict[str, Type[N3Provider]] = {}

# Provider instance registry - for reusing providers across chains/agents
_PROVIDER_REGISTRY: Dict[str, N3Provider] = {}


def register_provider_class(name: str, provider_class: Type[N3Provider]) -> None:
    """
    Register a provider implementation class.
    
    Called by provider modules on import to register themselves.
    
    Args:
        name: Provider type name (e.g., 'openai', 'anthropic')
        provider_class: The N3Provider subclass
    """
    _PROVIDER_CLASSES[name.lower()] = provider_class


def get_provider_class(provider_type: str) -> Type[N3Provider]:
    """
    Get a provider class by type name.
    
    Args:
        provider_type: Provider type (e.g., 'openai', 'anthropic')
    
    Returns:
        The N3Provider subclass
    
    Raises:
        ProviderError: If the provider type is not registered
    """
    provider_key = provider_type.lower()
    
    # Lazy load providers on first use
    if not _PROVIDER_CLASSES:
        _load_provider_modules()
    
    if provider_key not in _PROVIDER_CLASSES:
        available = ', '.join(sorted(_PROVIDER_CLASSES.keys()))
        raise ProviderError(
            f"Unknown provider type '{provider_type}'. "
            f"Available providers: {available or 'none (check installation)'}"
        )
    
    return _PROVIDER_CLASSES[provider_key]


def _load_provider_modules() -> None:
    """Lazy load all provider implementation modules."""
    # Import each provider module to trigger registration
    # Use try/except to handle missing optional dependencies gracefully
    
    try:
        from . import openai_provider  # noqa: F401
    except ImportError as e:
        # Log but don't fail - provider may not be needed
        pass
    
    try:
        from . import anthropic_provider  # noqa: F401
    except ImportError:
        pass
    
    try:
        from . import google_provider  # noqa: F401
    except ImportError:
        pass
    
    try:
        from . import azure_openai_provider  # noqa: F401
    except ImportError:
        pass
    
    try:
        from . import local_provider  # noqa: F401
    except ImportError:
        pass
    
    # Import new local model providers
    try:
        from .local import vllm  # noqa: F401
    except ImportError:
        pass
        
    try:
        from .local import ollama  # noqa: F401
    except ImportError:
        pass
        
    try:
        from .local import local_ai  # noqa: F401
    except ImportError:
        pass
    
    try:
        from . import http_provider  # noqa: F401
    except ImportError:
        pass


def create_provider_from_spec(
    name: str,
    provider_type: str,
    model: str,
    config: Optional[Dict[str, Any]] = None,
) -> N3Provider:
    """
    Create an N3Provider instance from a DSL specification.
    
    This is the main entry point for creating providers from parsed DSL `llm` blocks.
    
    Process:
    1. Validate provider name and model name
    2. Load base configuration from NAMEL3SS_PROVIDER_* environment variables
    3. Merge with DSL-level config (DSL overrides environment)
    4. Instantiate the appropriate provider class
    5. Return the configured provider
    
    Args:
        name: Logical name for the provider instance (e.g., "chat_gpt_4o")
        provider_type: Provider type ('openai', 'anthropic', 'google', 'azure_openai', 'local', 'http')
        model: Model identifier (e.g., 'gpt-4', 'claude-3-opus')
        config: DSL-level configuration overrides including:
            - temperature: Sampling temperature (0-2)
            - max_tokens: Maximum tokens to generate (must be positive)
            - top_p: Nucleus sampling parameter (0-1)
            - api_key: API key override
            - timeout: Request timeout in seconds
    
    Returns:
        Configured N3Provider instance
    
    Raises:
        ProviderValidationError: If name, model, or config is invalid
        ProviderError: If provider type is unknown or instantiation fails
        ProviderConfigError: If required configuration is missing
    
    Example:
        >>> # From DSL: llm chat_model: provider: openai, model: gpt-4
        >>> provider = create_provider_from_spec(
        ...     name="chat_model",
        ...     provider_type="openai",
        ...     model="gpt-4",
        ...     config={"temperature": 0.7, "max_tokens": 1000}
        ... )
    """
    # Lazy import to avoid circular dependency
    from .validation import validate_provider_name, validate_model_name
    
    # Validate inputs
    try:
        validate_provider_name(name)
        validate_model_name(model)
    except Exception as e:
        raise ProviderError(
            f"Invalid provider specification: {e}"
        ) from e
    
    # Load configuration (environment + DSL)
    try:
        merged_config = load_config_for_provider(provider_type, config)
    except ProviderConfigError as e:
        raise ProviderError(
            f"Failed to load configuration for provider '{provider_type}': {e}"
        ) from e
    
    # Get the provider class
    provider_class = get_provider_class(provider_type)
    
    # Instantiate the provider
    try:
        provider = provider_class(
            name=name,
            model=model,
            config=merged_config,
        )
    except Exception as e:
        raise ProviderError(
            f"Failed to create provider '{name}' (type={provider_type}, model={model}): {e}"
        ) from e
    
    return provider


def register_provider_instance(name: str, provider: N3Provider) -> None:
    """
    Register a provider instance in the global registry.
    
    This allows reusing provider instances across chains, agents, and other consumers
    instead of recreating them for each use.
    
    Args:
        name: Unique name for this provider instance
        provider: The N3Provider instance to register
    """
    _PROVIDER_REGISTRY[name] = provider


def get_provider_instance(name: str) -> Optional[N3Provider]:
    """
    Get a registered provider instance by name.
    
    Args:
        name: Provider instance name
    
    Returns:
        The registered N3Provider, or None if not found
    """
    return _PROVIDER_REGISTRY.get(name)


def clear_provider_registry() -> None:
    """Clear all registered provider instances."""
    _PROVIDER_REGISTRY.clear()


def list_registered_providers() -> Dict[str, N3Provider]:
    """
    Get all registered provider instances.
    
    Returns:
        Dictionary mapping provider names to instances
    """
    return _PROVIDER_REGISTRY.copy()


class ProviderRegistry:
    """
    Provider registry for managing provider lifecycle.
    
    This class provides a context manager and explicit lifecycle management
    for providers used in a specific execution context (e.g., a chain run).
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        self._providers: Dict[str, N3Provider] = {}
    
    def create_and_register(
        self,
        name: str,
        provider_type: str,
        model: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> N3Provider:
        """
        Create a provider and register it in this registry.
        
        Args:
            name: Provider instance name
            provider_type: Provider type
            model: Model identifier
            config: Configuration overrides
        
        Returns:
            The created and registered provider
        """
        provider = create_provider_from_spec(name, provider_type, model, config)
        self._providers[name] = provider
        return provider
    
    def register(self, name: str, provider: N3Provider) -> None:
        """Register an existing provider."""
        self._providers[name] = provider
    
    def get(self, name: str) -> Optional[N3Provider]:
        """Get a provider by name."""
        return self._providers.get(name)
    
    def get_or_create(
        self,
        name: str,
        provider_type: str,
        model: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> N3Provider:
        """
        Get an existing provider or create and register a new one.
        
        Args:
            name: Provider instance name
            provider_type: Provider type
            model: Model identifier
            config: Configuration overrides
        
        Returns:
            The provider instance
        """
        provider = self.get(name)
        if provider is None:
            provider = self.create_and_register(name, provider_type, model, config)
        return provider
    
    def clear(self) -> None:
        """Clear all providers from this registry."""
        self._providers.clear()
    
    def list_providers(self) -> Dict[str, N3Provider]:
        """Get all registered providers."""
        return self._providers.copy()
    
    def __enter__(self) -> "ProviderRegistry":
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and cleanup."""
        self.clear()


__all__ = [
    "register_provider_class",
    "get_provider_class",
    "create_provider_from_spec",
    "register_provider_instance",
    "get_provider_instance",
    "clear_provider_registry",
    "list_registered_providers",
    "ProviderRegistry",
]
