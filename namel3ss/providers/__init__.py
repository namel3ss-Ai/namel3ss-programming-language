"""N3Provider system for unified LLM backend abstraction.

This package provides a production-grade provider abstraction layer for all
LLM backends used in Namel3ss applications.

Main Components:
    - N3Provider: Unified async interface for all LLM backends
    - ProviderResponse: Normalized response format
    - Factory: Create providers from DSL specifications
    - Config: Environment-based configuration (NAMEL3SS_PROVIDER_*)
    - Implementations: OpenAI, Anthropic, Google, Azure, Local, HTTP

Usage:
    >>> from namel3ss.providers import create_provider_from_spec
    >>> 
    >>> # Create provider from DSL config
    >>> provider = create_provider_from_spec(
    ...     name="chat_model",
    ...     provider_type="openai",
    ...     model="gpt-4",
    ...     config={"temperature": 0.7}
    ... )
    >>> 
    >>> # Generate response
    >>> from namel3ss.providers import ProviderMessage
    >>> messages = [ProviderMessage(role="user", content="Hello!")]
    >>> response = await provider.generate(messages)
    >>> print(response.output_text)

Environment Configuration:
    Set NAMEL3SS_PROVIDER_{TYPE}_{KEY} environment variables:
    
    - NAMEL3SS_PROVIDER_OPENAI_API_KEY
    - NAMEL3SS_PROVIDER_ANTHROPIC_API_KEY
    - NAMEL3SS_PROVIDER_GOOGLE_PROJECT
    - NAMEL3SS_PROVIDER_AZURE_OPENAI_ENDPOINT
    - NAMEL3SS_PROVIDER_LOCAL_ENGINE_URL
    - NAMEL3SS_PROVIDER_HTTP_BASE_URL

Provider Types:
    - openai: OpenAI GPT models
    - anthropic: Anthropic Claude models
    - google/vertex: Google Gemini and Vertex AI models
    - azure_openai: Azure OpenAI deployments
    - local/ollama/vllm: Local LLM engines
    - http: Generic HTTP LLM endpoints

Validation:
    Centralized validation functions for provider configs, messages, and parameters.
    
    >>> from namel3ss.providers import validate_provider_config
    >>> from namel3ss.providers import ProviderValidationError
    >>> 
    >>> try:
    ...     validate_provider_config(
    ...         name="gpt4",
    ...         model="gpt-4",
    ...         temperature=0.7,
    ...         max_tokens=1000
    ...     )
    ... except ProviderValidationError as e:
    ...     print(f"[{e.code}] {e.message}")

Error Handling:
    Domain-specific exceptions for provider errors with error codes.
    
    >>> from namel3ss.providers import ProviderAuthError, ProviderRateLimitError
    >>> 
    >>> try:
    ...     # Provider operation
    ...     pass
    ... except ProviderAuthError as e:
    ...     print(f"Auth failed: {e.format()}")
    ... except ProviderRateLimitError as e:
    ...     print(f"Rate limited, retry after {e.retry_after}s")
"""

from .base import (
    N3Provider,
    ProviderMessage,
    ProviderResponse,
    ProviderError,
    BaseLLMAdapter,
)

from .errors import (
    ProviderValidationError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderAPIError,
    ProviderTimeoutError,
)

from .validation import (
    validate_provider_name,
    validate_model_name,
    validate_temperature,
    validate_max_tokens,
    validate_top_p,
    validate_api_key,
    validate_endpoint,
    validate_message,
    validate_messages,
    validate_provider_config,
)

from .config import (
    ProviderConfigError,
    load_provider_config,
    load_config_for_provider,
    load_openai_config,
    load_anthropic_config,
    load_google_config,
    load_azure_openai_config,
    load_local_config,
    load_http_config,
)

from .factory import (
    create_provider_from_spec,
    register_provider_class,
    get_provider_class,
    register_provider_instance,
    get_provider_instance,
    clear_provider_registry,
    list_registered_providers,
    ProviderRegistry,
)

# Import provider implementations to trigger registration
# Use try/except to handle missing optional dependencies gracefully
try:
    from .openai_provider import OpenAIProvider
except ImportError:
    OpenAIProvider = None

try:
    from .anthropic_provider import AnthropicProvider
except ImportError:
    AnthropicProvider = None

try:
    from .google_provider import GoogleProvider
except ImportError:
    GoogleProvider = None

try:
    from .azure_openai_provider import AzureOpenAIProvider
except ImportError:
    AzureOpenAIProvider = None

try:
    from .local_provider import LocalProvider
except ImportError:
    LocalProvider = None

try:
    from .http_provider import HttpProvider
except ImportError:
    HttpProvider = None

# Import integration adapters
try:
    from .integration import (
        ProviderLLMBridge,
        run_chain_with_provider,
        run_agent_with_provider,
    )
except ImportError:
    ProviderLLMBridge = None
    run_chain_with_provider = None
    run_agent_with_provider = None


__all__ = [
    # Base types
    "N3Provider",
    "ProviderMessage",
    "ProviderResponse",
    "ProviderError",
    "BaseLLMAdapter",
    # Errors
    "ProviderValidationError",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderAPIError",
    "ProviderTimeoutError",
    # Validation
    "validate_provider_name",
    "validate_model_name",
    "validate_temperature",
    "validate_max_tokens",
    "validate_top_p",
    "validate_api_key",
    "validate_endpoint",
    "validate_message",
    "validate_messages",
    "validate_provider_config",
    # Configuration
    "ProviderConfigError",
    "load_provider_config",
    "load_config_for_provider",
    "load_openai_config",
    "load_anthropic_config",
    "load_google_config",
    "load_azure_openai_config",
    "load_local_config",
    "load_http_config",
    # Factory
    "create_provider_from_spec",
    "register_provider_class",
    "get_provider_class",
    "register_provider_instance",
    "get_provider_instance",
    "clear_provider_registry",
    "list_registered_providers",
    "ProviderRegistry",
    # Provider implementations
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "AzureOpenAIProvider",
    "LocalProvider",
    "HttpProvider",
    # Integration
    "ProviderLLMBridge",
    "run_chain_with_provider",
    "run_agent_with_provider",
]


__version__ = "1.0.0"
