"""
Provider-specific exception types for the Namel3ss providers subsystem.

This module provides a hierarchy of exceptions for provider-related errors:
- ProviderValidationError: Validation failures during provider creation or config
- ProviderAuthError: Authentication and authorization failures
- ProviderRateLimitError: Rate limiting errors with retry information
- ProviderAPIError: API-specific errors from provider services
- ProviderTimeoutError: Request timeout errors

All exceptions extend the base ProviderError from base.py and include contextual
information for debugging (error codes, provider names, model names, etc.).

Example:
    from namel3ss.providers.errors import ProviderAuthError
    
    raise ProviderAuthError(
        "Invalid API key",
        code="PROV020",
        provider="openai",
        model="gpt-4"
    )
"""

from typing import Any, Optional

from .base import ProviderError as BaseProviderError


class ProviderValidationError(BaseProviderError):
    """
    Exception raised when provider validation fails.
    
    Used for:
    - Invalid provider configuration (empty API keys, invalid endpoints)
    - Message validation failures (empty content, invalid roles)
    - Parameter validation errors (temperature out of range, negative max_tokens)
    - Model name validation (unsupported models)
    
    Attributes:
        message: Human-readable error message
        code: Error code (PROV001-PROV015)
        provider: Provider type (openai, anthropic, google, etc.)
        model: Model identifier
        field: Field name that failed validation
        value: The invalid value
        expected: Expected value or type
    
    Example:
        >>> from namel3ss.providers.errors import ProviderValidationError
        >>> raise ProviderValidationError(
        ...     "Temperature must be between 0 and 2",
        ...     code="PROV005",
        ...     provider="openai",
        ...     model="gpt-4",
        ...     field="temperature",
        ...     value=3.5,
        ...     expected="0.0 to 2.0"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        *,
        code: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        field: Optional[str] = None,
        value: Any = None,
        expected: Optional[str] = None,
    ):
        """
        Initialize validation error.
        
        Args:
            message: Error description
            code: Error code (PROV001-PROV015)
            provider: Provider type
            model: Model identifier
            field: Field that failed validation
            value: The invalid value
            expected: Expected value or type description
        """
        super().__init__(message, provider=provider, model=model)
        self.message = message
        self.code = code
        self.field = field
        self.value = value
        self.expected = expected
    
    def format(self) -> str:
        """
        Format error with full context.
        
        Returns:
            Formatted error message with code and context
        """
        parts = [f"[{self.code}] Provider Validation Error"]
        
        if self.provider:
            parts.append(f"Provider: {self.provider}")
        
        if self.model:
            parts.append(f"Model: {self.model}")
        
        if self.field:
            parts.append(f"Field: {self.field}")
        
        if self.expected:
            parts.append(f"Expected: {self.expected}")
        
        if self.value is not None:
            parts.append(f"Got: {self.value!r}")
        
        parts.append(f"Message: {self.message}")
        
        return "\n".join(parts)
    
    def __str__(self) -> str:
        """Return formatted error message."""
        return self.format()


class ProviderAuthError(BaseProviderError):
    """
    Exception raised when provider authentication fails.
    
    Used for:
    - Missing API keys
    - Invalid API keys
    - Expired credentials
    - Insufficient permissions
    - Region/quota restrictions
    
    Attributes:
        message: Human-readable error message
        code: Error code (PROV016-PROV025)
        provider: Provider type
        model: Model identifier
        auth_type: Type of authentication (api_key, oauth, service_account)
        status_code: HTTP status code if applicable
    
    Example:
        >>> from namel3ss.providers.errors import ProviderAuthError
        >>> raise ProviderAuthError(
        ...     "API key not found in environment",
        ...     code="PROV016",
        ...     provider="openai",
        ...     auth_type="api_key"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        *,
        code: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        auth_type: Optional[str] = None,
        status_code: Optional[int] = None,
    ):
        """
        Initialize authentication error.
        
        Args:
            message: Error description
            code: Error code (PROV016-PROV025)
            provider: Provider type
            model: Model identifier
            auth_type: Authentication type
            status_code: HTTP status code
        """
        super().__init__(message, provider=provider, model=model)
        self.message = message
        self.code = code
        self.auth_type = auth_type
        self.status_code = status_code
    
    def format(self) -> str:
        """Format error with full context."""
        parts = [f"[{self.code}] Provider Authentication Error"]
        
        if self.provider:
            parts.append(f"Provider: {self.provider}")
        
        if self.model:
            parts.append(f"Model: {self.model}")
        
        if self.auth_type:
            parts.append(f"Auth Type: {self.auth_type}")
        
        if self.status_code:
            parts.append(f"Status Code: {self.status_code}")
        
        parts.append(f"Message: {self.message}")
        
        return "\n".join(parts)
    
    def __str__(self) -> str:
        return self.format()


class ProviderRateLimitError(BaseProviderError):
    """
    Exception raised when provider rate limit is exceeded.
    
    Used for:
    - Request rate limits (requests per minute/second)
    - Token rate limits (tokens per minute)
    - Quota exhaustion (daily/monthly limits)
    - Concurrent request limits
    
    Attributes:
        message: Human-readable error message
        code: Error code (PROV026-PROV032)
        provider: Provider type
        model: Model identifier
        retry_after: Seconds to wait before retrying
        limit_type: Type of limit (requests, tokens, quota)
        status_code: HTTP status code (usually 429)
    
    Example:
        >>> from namel3ss.providers.errors import ProviderRateLimitError
        >>> raise ProviderRateLimitError(
        ...     "Rate limit exceeded",
        ...     code="PROV026",
        ...     provider="openai",
        ...     model="gpt-4",
        ...     retry_after=60,
        ...     limit_type="requests"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        *,
        code: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        retry_after: Optional[int] = None,
        limit_type: Optional[str] = None,
        status_code: int = 429,
    ):
        """
        Initialize rate limit error.
        
        Args:
            message: Error description
            code: Error code (PROV026-PROV032)
            provider: Provider type
            model: Model identifier
            retry_after: Seconds to wait before retry
            limit_type: Type of limit (requests, tokens, quota)
            status_code: HTTP status code (default 429)
        """
        super().__init__(message, provider=provider, model=model)
        self.message = message
        self.code = code
        self.retry_after = retry_after
        self.limit_type = limit_type
        self.status_code = status_code
    
    def format(self) -> str:
        """Format error with full context."""
        parts = [f"[{self.code}] Provider Rate Limit Error"]
        
        if self.provider:
            parts.append(f"Provider: {self.provider}")
        
        if self.model:
            parts.append(f"Model: {self.model}")
        
        if self.limit_type:
            parts.append(f"Limit Type: {self.limit_type}")
        
        if self.retry_after:
            parts.append(f"Retry After: {self.retry_after}s")
        
        parts.append(f"Status Code: {self.status_code}")
        parts.append(f"Message: {self.message}")
        
        return "\n".join(parts)
    
    def __str__(self) -> str:
        return self.format()


class ProviderAPIError(BaseProviderError):
    """
    Exception raised when provider API returns an error.
    
    Used for:
    - HTTP errors (4xx, 5xx)
    - API-specific error responses
    - Service unavailable errors
    - Invalid request formats
    - Content policy violations
    
    Attributes:
        message: Human-readable error message
        code: Error code (PROV033-PROV045)
        provider: Provider type
        model: Model identifier
        status_code: HTTP status code
        error_type: Provider-specific error type
        retryable: Whether the error is retryable
    
    Example:
        >>> from namel3ss.providers.errors import ProviderAPIError
        >>> raise ProviderAPIError(
        ...     "Content filtered by safety system",
        ...     code="PROV040",
        ...     provider="openai",
        ...     model="gpt-4",
        ...     status_code=400,
        ...     error_type="content_filter",
        ...     retryable=False
        ... )
    """
    
    def __init__(
        self,
        message: str,
        *,
        code: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        error_type: Optional[str] = None,
        retryable: bool = False,
    ):
        """
        Initialize API error.
        
        Args:
            message: Error description
            code: Error code (PROV033-PROV045)
            provider: Provider type
            model: Model identifier
            status_code: HTTP status code
            error_type: Provider-specific error type
            retryable: Whether error is retryable
        """
        super().__init__(message, provider=provider, model=model)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.error_type = error_type
        self.retryable = retryable
    
    def format(self) -> str:
        """Format error with full context."""
        parts = [f"[{self.code}] Provider API Error"]
        
        if self.provider:
            parts.append(f"Provider: {self.provider}")
        
        if self.model:
            parts.append(f"Model: {self.model}")
        
        if self.status_code:
            parts.append(f"Status Code: {self.status_code}")
        
        if self.error_type:
            parts.append(f"Error Type: {self.error_type}")
        
        parts.append(f"Retryable: {'Yes' if self.retryable else 'No'}")
        parts.append(f"Message: {self.message}")
        
        return "\n".join(parts)
    
    def __str__(self) -> str:
        return self.format()


class ProviderTimeoutError(BaseProviderError):
    """
    Exception raised when provider request times out.
    
    Used for:
    - Connection timeouts
    - Read timeouts
    - Generation timeouts (long-running requests)
    - Network issues
    
    Attributes:
        message: Human-readable error message
        code: Error code (PROV046-PROV050)
        provider: Provider type
        model: Model identifier
        timeout_seconds: Configured timeout value
        operation: Operation that timed out
    
    Example:
        >>> from namel3ss.providers.errors import ProviderTimeoutError
        >>> raise ProviderTimeoutError(
        ...     "Request timed out after 30 seconds",
        ...     code="PROV046",
        ...     provider="openai",
        ...     model="gpt-4",
        ...     timeout_seconds=30,
        ...     operation="generate"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        *,
        code: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
    ):
        """
        Initialize timeout error.
        
        Args:
            message: Error description
            code: Error code (PROV046-PROV050)
            provider: Provider type
            model: Model identifier
            timeout_seconds: Configured timeout
            operation: Operation that timed out
        """
        super().__init__(message, provider=provider, model=model)
        self.message = message
        self.code = code
        self.timeout_seconds = timeout_seconds
        self.operation = operation
    
    def format(self) -> str:
        """Format error with full context."""
        parts = [f"[{self.code}] Provider Timeout Error"]
        
        if self.provider:
            parts.append(f"Provider: {self.provider}")
        
        if self.model:
            parts.append(f"Model: {self.model}")
        
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        
        if self.timeout_seconds:
            parts.append(f"Timeout: {self.timeout_seconds}s")
        
        parts.append(f"Message: {self.message}")
        
        return "\n".join(parts)
    
    def __str__(self) -> str:
        return self.format()


__all__ = [
    "ProviderValidationError",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderAPIError",
    "ProviderTimeoutError",
]
