"""Provider configuration management.

Centralized configuration for all LLM providers using NAMEL3SS_PROVIDER_* environment variables.
"""

import os
from typing import Dict, Any, Optional


# Environment variable prefix
ENV_PREFIX = "NAMEL3SS_PROVIDER_"


class ProviderConfigError(Exception):
    """Raised when provider configuration is invalid or missing."""
    pass


def load_provider_config(provider_type: str) -> Dict[str, Any]:
    """
    Load configuration for a given provider type from environment variables.
    
    Environment variable conventions:
        - NAMEL3SS_PROVIDER_{TYPE}_{KEY}
        
    Examples:
        - NAMEL3SS_PROVIDER_OPENAI_API_KEY
        - NAMEL3SS_PROVIDER_OPENAI_BASE_URL
        - NAMEL3SS_PROVIDER_ANTHROPIC_API_KEY
        - NAMEL3SS_PROVIDER_GOOGLE_PROJECT
        - NAMEL3SS_PROVIDER_GOOGLE_LOCATION
        - NAMEL3SS_PROVIDER_AZURE_OPENAI_ENDPOINT
        - NAMEL3SS_PROVIDER_AZURE_OPENAI_API_KEY
        - NAMEL3SS_PROVIDER_LOCAL_ENGINE_URL
        - NAMEL3SS_PROVIDER_HTTP_BASE_URL
    
    Args:
        provider_type: Provider type (e.g., 'openai', 'anthropic', 'google', 'azure_openai', 'local', 'http')
    
    Returns:
        Configuration dictionary with keys normalized to lowercase
    
    Raises:
        ProviderConfigError: If required configuration is missing
    """
    provider_type = provider_type.upper()
    prefix = f"{ENV_PREFIX}{provider_type}_"
    
    config: Dict[str, Any] = {}
    
    # Scan environment for matching variables
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Extract the config key (everything after the prefix)
            config_key = key[len(prefix):].lower()
            config[config_key] = value
    
    return config


def get_config_value(
    provider_type: str,
    key: str,
    default: Optional[str] = None,
    required: bool = False,
) -> Optional[str]:
    """
    Get a single configuration value for a provider.
    
    Args:
        provider_type: Provider type (e.g., 'openai', 'anthropic')
        key: Configuration key (e.g., 'api_key', 'base_url')
        default: Default value if not found
        required: If True, raise error if not found and no default
    
    Returns:
        Configuration value or default
    
    Raises:
        ProviderConfigError: If required and not found
    """
    env_key = f"{ENV_PREFIX}{provider_type.upper()}_{key.upper()}"
    value = os.environ.get(env_key, default)
    
    if required and value is None:
        raise ProviderConfigError(
            f"Required configuration '{env_key}' not found for provider '{provider_type}'. "
            f"Set the environment variable or provide it in the DSL config."
        )
    
    return value


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Merge base configuration with overrides.
    
    DSL-level config overrides environment-level config.
    
    Args:
        base_config: Base configuration (typically from environment)
        override_config: Override configuration (typically from DSL)
    
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    if override_config:
        result.update(override_config)
    
    return result


# Provider-specific configuration helpers


def load_openai_config(dsl_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load OpenAI provider configuration.
    
    Environment variables:
        - NAMEL3SS_PROVIDER_OPENAI_API_KEY (required)
        - NAMEL3SS_PROVIDER_OPENAI_BASE_URL (optional, default: https://api.openai.com/v1)
        - NAMEL3SS_PROVIDER_OPENAI_ORGANIZATION (optional)
        - NAMEL3SS_PROVIDER_OPENAI_TIMEOUT (optional, default: 60)
    
    Args:
        dsl_config: Optional DSL-level configuration overrides
    
    Returns:
        Merged configuration dictionary
    """
    base_config = load_provider_config('openai')
    
    # Set defaults
    if 'base_url' not in base_config:
        base_config['base_url'] = 'https://api.openai.com/v1'
    if 'timeout' not in base_config:
        base_config['timeout'] = 60
    
    return merge_configs(base_config, dsl_config)


def load_anthropic_config(dsl_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load Anthropic provider configuration.
    
    Environment variables:
        - NAMEL3SS_PROVIDER_ANTHROPIC_API_KEY (required)
        - NAMEL3SS_PROVIDER_ANTHROPIC_BASE_URL (optional, default: https://api.anthropic.com)
        - NAMEL3SS_PROVIDER_ANTHROPIC_VERSION (optional, default: 2023-06-01)
        - NAMEL3SS_PROVIDER_ANTHROPIC_TIMEOUT (optional, default: 60)
    
    Args:
        dsl_config: Optional DSL-level configuration overrides
    
    Returns:
        Merged configuration dictionary
    """
    base_config = load_provider_config('anthropic')
    
    # Set defaults
    if 'base_url' not in base_config:
        base_config['base_url'] = 'https://api.anthropic.com'
    if 'timeout' not in base_config:
        base_config['timeout'] = 60
    if 'version' not in base_config:
        base_config['version'] = '2023-06-01'
    
    return merge_configs(base_config, dsl_config)


def load_google_config(dsl_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load Google (Vertex AI / Gemini) provider configuration.
    
    Environment variables:
        - NAMEL3SS_PROVIDER_GOOGLE_PROJECT (required for Vertex AI)
        - NAMEL3SS_PROVIDER_GOOGLE_LOCATION (optional, default: us-central1)
        - NAMEL3SS_PROVIDER_GOOGLE_API_KEY (optional, for Gemini API)
        - NAMEL3SS_PROVIDER_GOOGLE_CREDENTIALS_PATH (optional)
        - NAMEL3SS_PROVIDER_GOOGLE_TIMEOUT (optional, default: 60)
    
    Args:
        dsl_config: Optional DSL-level configuration overrides
    
    Returns:
        Merged configuration dictionary
    """
    base_config = load_provider_config('google')
    
    # Set defaults
    if 'location' not in base_config:
        base_config['location'] = 'us-central1'
    if 'timeout' not in base_config:
        base_config['timeout'] = 60
    
    return merge_configs(base_config, dsl_config)


def load_azure_openai_config(dsl_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load Azure OpenAI provider configuration.
    
    Environment variables:
        - NAMEL3SS_PROVIDER_AZURE_OPENAI_API_KEY (required)
        - NAMEL3SS_PROVIDER_AZURE_OPENAI_ENDPOINT (required)
        - NAMEL3SS_PROVIDER_AZURE_OPENAI_DEPLOYMENT (optional, can be set per-model)
        - NAMEL3SS_PROVIDER_AZURE_OPENAI_API_VERSION (optional, default: 2023-05-15)
        - NAMEL3SS_PROVIDER_AZURE_OPENAI_TIMEOUT (optional, default: 60)
    
    Args:
        dsl_config: Optional DSL-level configuration overrides
    
    Returns:
        Merged configuration dictionary
    """
    base_config = load_provider_config('azure_openai')
    
    # Set defaults
    if 'api_version' not in base_config:
        base_config['api_version'] = '2023-05-15'
    if 'timeout' not in base_config:
        base_config['timeout'] = 60
    
    return merge_configs(base_config, dsl_config)


def load_local_config(dsl_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load local engine (vLLM/Ollama) provider configuration.
    
    Environment variables:
        - NAMEL3SS_PROVIDER_LOCAL_ENGINE_URL (required)
        - NAMEL3SS_PROVIDER_LOCAL_ENGINE_TYPE (optional: 'ollama', 'vllm', default: auto-detect)
        - NAMEL3SS_PROVIDER_LOCAL_TIMEOUT (optional, default: 120)
    
    Args:
        dsl_config: Optional DSL-level configuration overrides
    
    Returns:
        Merged configuration dictionary
    """
    base_config = load_provider_config('local')
    
    # Set defaults
    if 'timeout' not in base_config:
        base_config['timeout'] = 120  # Longer timeout for local engines
    
    return merge_configs(base_config, dsl_config)


def load_http_config(dsl_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load generic HTTP provider configuration.
    
    Environment variables:
        - NAMEL3SS_PROVIDER_HTTP_BASE_URL (required)
        - NAMEL3SS_PROVIDER_HTTP_API_KEY (optional)
        - NAMEL3SS_PROVIDER_HTTP_TIMEOUT (optional, default: 60)
        - NAMEL3SS_PROVIDER_HTTP_REQUEST_PATH (optional, default: /v1/chat/completions)
        - NAMEL3SS_PROVIDER_HTTP_AUTH_HEADER (optional, default: Authorization)
    
    Args:
        dsl_config: Optional DSL-level configuration overrides
    
    Returns:
        Merged configuration dictionary
    """
    base_config = load_provider_config('http')
    
    # Set defaults
    if 'timeout' not in base_config:
        base_config['timeout'] = 60
    if 'request_path' not in base_config:
        base_config['request_path'] = '/v1/chat/completions'
    if 'auth_header' not in base_config:
        base_config['auth_header'] = 'Authorization'
    
    return merge_configs(base_config, dsl_config)


# Provider type registry with config loaders
PROVIDER_CONFIG_LOADERS = {
    'openai': load_openai_config,
    'anthropic': load_anthropic_config,
    'google': load_google_config,
    'vertex': load_google_config,  # Alias for google
    'azure_openai': load_azure_openai_config,
    'local': load_local_config,
    'ollama': load_local_config,  # Alias for local
    'vllm': load_local_config,  # Alias for local
    'http': load_http_config,
}


def load_config_for_provider(
    provider_type: str,
    dsl_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load configuration for any provider type.
    
    This is the main entry point for loading provider configuration.
    
    Args:
        provider_type: Provider type (e.g., 'openai', 'anthropic', 'google')
        dsl_config: Optional DSL-level configuration overrides
    
    Returns:
        Fully merged configuration dictionary
    
    Raises:
        ProviderConfigError: If provider type is unknown
    """
    provider_type_lower = provider_type.lower()
    
    if provider_type_lower not in PROVIDER_CONFIG_LOADERS:
        available = ', '.join(sorted(PROVIDER_CONFIG_LOADERS.keys()))
        raise ProviderConfigError(
            f"Unknown provider type '{provider_type}'. "
            f"Available types: {available}"
        )
    
    loader = PROVIDER_CONFIG_LOADERS[provider_type_lower]
    return loader(dsl_config)


__all__ = [
    "ProviderConfigError",
    "load_provider_config",
    "get_config_value",
    "merge_configs",
    "load_openai_config",
    "load_anthropic_config",
    "load_google_config",
    "load_azure_openai_config",
    "load_local_config",
    "load_http_config",
    "load_config_for_provider",
]
