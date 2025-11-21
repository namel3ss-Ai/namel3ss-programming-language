"""
Centralized validation functions for the Namel3ss providers subsystem.

This module provides pure validation functions for all provider-related operations:
- Provider configuration validation (names, types, API keys, endpoints)
- Message validation (role, content)
- Generation parameter validation (temperature, max_tokens, top_p, etc.)
- Model name validation

All validators are pure functions (no I/O, no side effects) that raise
ProviderValidationError when validation fails.

Example:
    from namel3ss.providers.validation import validate_temperature, validate_api_key
    from namel3ss.providers.errors import ProviderValidationError
    
    try:
        validate_temperature(0.7, provider="openai", model="gpt-4")
        validate_api_key("sk-...", provider="openai")
    except ProviderValidationError as e:
        print(f"Validation failed: {e.format()}")
"""

from typing import Any, Dict, List, Optional

from .base import ProviderMessage
from .errors import ProviderValidationError


def validate_provider_name(
    name: str,
    *,
    provider_type: Optional[str] = None,
) -> None:
    """
    Validate provider name is non-empty and well-formed.
    
    Args:
        name: Provider name to validate
        provider_type: Optional provider type for context
        
    Raises:
        ProviderValidationError: If name is invalid
    """
    if not name:
        raise ProviderValidationError(
            "Provider name cannot be empty",
            code="PROV001",
            provider=provider_type,
            field="name",
        )
    
    if not isinstance(name, str):
        raise ProviderValidationError(
            "Provider name must be a string",
            code="PROV002",
            provider=provider_type,
            field="name",
            value=type(name).__name__,
            expected="str",
        )


def validate_model_name(
    model: str,
    *,
    provider: Optional[str] = None,
) -> None:
    """
    Validate model name is non-empty.
    
    Args:
        model: Model name to validate
        provider: Optional provider type for context
        
    Raises:
        ProviderValidationError: If model is invalid
    """
    if not model:
        raise ProviderValidationError(
            "Model name cannot be empty",
            code="PROV003",
            provider=provider,
            field="model",
        )
    
    if not isinstance(model, str):
        raise ProviderValidationError(
            "Model name must be a string",
            code="PROV004",
            provider=provider,
            field="model",
            value=type(model).__name__,
            expected="str",
        )


def validate_temperature(
    temperature: float,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """
    Validate temperature parameter is in valid range.
    
    Args:
        temperature: Temperature value (typically 0.0 to 2.0)
        provider: Optional provider for context
        model: Optional model for context
        
    Raises:
        ProviderValidationError: If temperature is invalid
    """
    if not isinstance(temperature, (int, float)):
        raise ProviderValidationError(
            "Temperature must be a number",
            code="PROV005",
            provider=provider,
            model=model,
            field="temperature",
            value=type(temperature).__name__,
            expected="float or int",
        )
    
    if temperature < 0 or temperature > 2:
        raise ProviderValidationError(
            f"Temperature must be between 0 and 2, got {temperature}",
            code="PROV006",
            provider=provider,
            model=model,
            field="temperature",
            value=temperature,
            expected="0.0 to 2.0",
        )


def validate_max_tokens(
    max_tokens: int,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """
    Validate max_tokens parameter is positive.
    
    Args:
        max_tokens: Maximum tokens to generate
        provider: Optional provider for context
        model: Optional model for context
        
    Raises:
        ProviderValidationError: If max_tokens is invalid
    """
    if not isinstance(max_tokens, int):
        raise ProviderValidationError(
            "max_tokens must be an integer",
            code="PROV007",
            provider=provider,
            model=model,
            field="max_tokens",
            value=type(max_tokens).__name__,
            expected="int",
        )
    
    if max_tokens <= 0:
        raise ProviderValidationError(
            f"max_tokens must be positive, got {max_tokens}",
            code="PROV008",
            provider=provider,
            model=model,
            field="max_tokens",
            value=max_tokens,
            expected="> 0",
        )


def validate_top_p(
    top_p: float,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """
    Validate top_p parameter is in valid range.
    
    Args:
        top_p: Nucleus sampling parameter (0.0 to 1.0)
        provider: Optional provider for context
        model: Optional model for context
        
    Raises:
        ProviderValidationError: If top_p is invalid
    """
    if not isinstance(top_p, (int, float)):
        raise ProviderValidationError(
            "top_p must be a number",
            code="PROV009",
            provider=provider,
            model=model,
            field="top_p",
            value=type(top_p).__name__,
            expected="float or int",
        )
    
    if top_p < 0 or top_p > 1:
        raise ProviderValidationError(
            f"top_p must be between 0 and 1, got {top_p}",
            code="PROV010",
            provider=provider,
            model=model,
            field="top_p",
            value=top_p,
            expected="0.0 to 1.0",
        )


def validate_api_key(
    api_key: str,
    *,
    provider: Optional[str] = None,
    min_length: int = 10,
) -> None:
    """
    Validate API key is non-empty and well-formed.
    
    Args:
        api_key: API key to validate
        provider: Optional provider for context
        min_length: Minimum key length (default 10)
        
    Raises:
        ProviderValidationError: If API key is invalid
    """
    if not api_key:
        raise ProviderValidationError(
            f"API key for {provider} is missing or empty",
            code="PROV011",
            provider=provider,
            field="api_key",
        )
    
    if not isinstance(api_key, str):
        raise ProviderValidationError(
            "API key must be a string",
            code="PROV012",
            provider=provider,
            field="api_key",
            value=type(api_key).__name__,
            expected="str",
        )
    
    if len(api_key) < min_length:
        raise ProviderValidationError(
            f"API key too short (minimum {min_length} characters)",
            code="PROV013",
            provider=provider,
            field="api_key",
            value=f"{len(api_key)} chars",
            expected=f">= {min_length} chars",
        )


def validate_endpoint(
    endpoint: str,
    *,
    provider: Optional[str] = None,
    require_https: bool = True,
) -> None:
    """
    Validate endpoint URL is well-formed.
    
    Args:
        endpoint: API endpoint URL
        provider: Optional provider for context
        require_https: Whether to require HTTPS (default True)
        
    Raises:
        ProviderValidationError: If endpoint is invalid
    """
    if not endpoint:
        raise ProviderValidationError(
            "Endpoint cannot be empty",
            code="PROV014",
            provider=provider,
            field="endpoint",
        )
    
    if not isinstance(endpoint, str):
        raise ProviderValidationError(
            "Endpoint must be a string",
            code="PROV015",
            provider=provider,
            field="endpoint",
            value=type(endpoint).__name__,
            expected="str",
        )
    
    if require_https and not endpoint.startswith("https://"):
        if endpoint.startswith("http://"):
            raise ProviderValidationError(
                "Endpoint must use HTTPS for security",
                code="PROV015",
                provider=provider,
                field="endpoint",
                value=endpoint,
                expected="URL starting with https://",
            )
        else:
            raise ProviderValidationError(
                "Endpoint must be a valid URL",
                code="PROV017",
                provider=provider,
                field="endpoint",
                value=endpoint,
                expected="URL starting with https://",
            )


def validate_message(
    message: ProviderMessage,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """
    Validate a single provider message.
    
    Accepts both dict-like objects and objects with attributes.
    
    Args:
        message: Message to validate (dict or object with role/content)
        provider: Optional provider for context
        model: Optional model for context
        
    Raises:
        ProviderValidationError: If message is invalid
    """
    # Handle both dict and object formats
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
        
        if role is None:
            raise ProviderValidationError(
                "Message must have a 'role' key",
                code="PROV018",
                provider=provider,
                model=model,
                field="message.role",
            )
        
        if content is None:
            raise ProviderValidationError(
                "Message must have a 'content' key",
                code="PROV019",
                provider=provider,
                model=model,
                field="message.content",
            )
    else:
        # Object with attributes
        if not hasattr(message, "role"):
            raise ProviderValidationError(
                "Message must have a 'role' attribute",
                code="PROV018",
                provider=provider,
                model=model,
                field="message.role",
            )
        
        if not hasattr(message, "content"):
            raise ProviderValidationError(
                "Message must have a 'content' attribute",
                code="PROV019",
                provider=provider,
                model=model,
                field="message.content",
            )
        
        role = message.role
        content = message.content
    
    valid_roles = {"system", "user", "assistant", "function", "tool"}
    if role not in valid_roles:
        raise ProviderValidationError(
            f"Invalid message role: {role}",
            code="PROV020",
            provider=provider,
            model=model,
            field="message.role",
            value=role,
            expected=f"One of: {valid_roles}",
        )
    
    if not content:
        raise ProviderValidationError(
            f"Message content cannot be empty for role '{role}'",
            code="PROV021",
            provider=provider,
            model=model,
            field="message.content",
        )


def validate_messages(
    messages: List[ProviderMessage],
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """
    Validate a list of provider messages.
    
    Args:
        messages: List of messages to validate
        provider: Optional provider for context
        model: Optional model for context
        
    Raises:
        ProviderValidationError: If messages are invalid
    """
    if not messages:
        raise ProviderValidationError(
            "Messages list cannot be empty",
            code="PROV022",
            provider=provider,
            model=model,
            field="messages",
        )
    
    if not isinstance(messages, list):
        raise ProviderValidationError(
            "Messages must be a list",
            code="PROV023",
            provider=provider,
            model=model,
            field="messages",
            value=type(messages).__name__,
            expected="list",
        )
    
    # Validate each message
    for i, message in enumerate(messages):
        try:
            validate_message(message, provider=provider, model=model)
        except ProviderValidationError as e:
            # Add index to error
            e.field = f"messages[{i}].{e.field or 'unknown'}"
            raise


def validate_provider_config(
    *,
    name: str,
    model: str,
    provider_type: Optional[str] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
) -> None:
    """
    Validate complete provider configuration.
    
    Runs all applicable validators based on provided parameters.
    
    Args:
        name: Provider name
        model: Model name
        provider_type: Provider type (openai, anthropic, etc.)
        api_key: API key (if provided)
        endpoint: Endpoint URL (if provided)
        temperature: Temperature parameter (if provided)
        max_tokens: Max tokens parameter (if provided)
        top_p: Top-p parameter (if provided)
        
    Raises:
        ProviderValidationError: If any validation fails
    """
    # Validate required fields
    validate_provider_name(name, provider_type=provider_type)
    validate_model_name(model, provider=provider_type)
    
    # Validate optional fields
    if api_key is not None:
        validate_api_key(api_key, provider=provider_type)
    
    if endpoint is not None:
        validate_endpoint(endpoint, provider=provider_type)
    
    if temperature is not None:
        validate_temperature(temperature, provider=provider_type, model=model)
    
    if max_tokens is not None:
        validate_max_tokens(max_tokens, provider=provider_type, model=model)
    
    if top_p is not None:
        validate_top_p(top_p, provider=provider_type, model=model)


__all__ = [
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
]
