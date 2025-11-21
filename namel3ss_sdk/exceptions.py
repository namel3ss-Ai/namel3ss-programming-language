"""Exception hierarchy for N3 SDK.

Provides comprehensive, context-rich exceptions for all failure modes:
- Network errors (timeout, connection, rate limiting)
- Authentication/authorization failures
- Runtime execution errors
- Schema validation errors

All exceptions include request ID and relevant context for debugging.
"""

from typing import Any, Dict, Optional


class N3Error(Exception):
    """Base exception for all N3 SDK errors.
    
    All SDK exceptions inherit from this base class, making it easy to catch
    all SDK-related errors with a single except clause.
    
    Attributes:
        message: Human-readable error description
        request_id: Optional request ID for tracing
        context: Additional error context (status codes, etc.)
    """
    
    def __init__(
        self,
        message: str,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.request_id = request_id
        self.context = context or {}
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message with context."""
        parts = [self.message]
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"context=({ctx_str})")
        return " | ".join(parts)


class N3ClientError(N3Error):
    """Client-side error (4xx status codes).
    
    Raised when the request is invalid due to:
    - Missing required parameters
    - Invalid parameter values
    - Malformed requests
    - Resource not found
    
    These errors typically cannot be resolved by retrying.
    
    Example:
        >>> try:
        ...     client.chains.run("nonexistent_chain")
        ... except N3ClientError as e:
        ...     print(f"Chain not found: {e.message}")
        ...     print(f"Status: {e.context['status_code']}")
    """
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        context = context or {}
        if status_code:
            context["status_code"] = status_code
        super().__init__(message, request_id, context)
        self.status_code = status_code


class N3ServerError(N3Error):
    """Server-side error (5xx status codes).
    
    Raised when the N3 server encounters an internal error:
    - Runtime exceptions
    - Database failures
    - Unexpected errors
    
    These errors may be transient and retrying may succeed.
    """
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        context = context or {}
        if status_code:
            context["status_code"] = status_code
        super().__init__(message, request_id, context)
        self.status_code = status_code


class N3TimeoutError(N3Error):
    """Request timed out.
    
    Raised when a request exceeds the configured timeout.
    May be transient - consider retrying with backoff.
    
    Example:
        >>> try:
        ...     client.chains.run("slow_chain", timeout=5.0)
        ... except N3TimeoutError:
        ...     # Retry with longer timeout
        ...     client.chains.run("slow_chain", timeout=30.0)
    """
    
    def __init__(
        self,
        message: str = "Request timed out",
        timeout_seconds: Optional[float] = None,
        request_id: Optional[str] = None,
    ):
        context = {}
        if timeout_seconds is not None:
            context["timeout_seconds"] = timeout_seconds
        super().__init__(message, request_id, context)
        self.timeout_seconds = timeout_seconds


class N3AuthError(N3Error):
    """Authentication or authorization failed.
    
    Raised when:
    - API token is missing
    - API token is invalid or expired
    - User lacks permission for the operation
    
    Check your credentials and permissions.
    
    Example:
        >>> try:
        ...     client.chains.run("admin_only_chain")
        ... except N3AuthError:
        ...     print("Access denied - check your token")
    """
    pass


class N3ConnectionError(N3Error):
    """Network connection failed.
    
    Raised when unable to connect to the N3 server:
    - DNS resolution failed
    - Connection refused
    - TLS handshake failed
    - Network unreachable
    
    May be transient - consider retrying.
    """
    pass


class N3RateLimitError(N3ClientError):
    """Rate limit exceeded.
    
    Raised when too many requests are made in a time window.
    Wait before retrying.
    
    Attributes:
        retry_after: Seconds to wait before retrying (if provided by server)
    
    Example:
        >>> try:
        ...     client.chains.run("popular_chain")
        ... except N3RateLimitError as e:
        ...     time.sleep(e.retry_after or 60)
        ...     client.chains.run("popular_chain")
    """
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        request_id: Optional[str] = None,
    ):
        context = {}
        if retry_after is not None:
            context["retry_after"] = retry_after
        super().__init__(message, status_code=429, request_id=request_id, context=context)
        self.retry_after = retry_after


class N3RuntimeError(N3Error):
    """Runtime execution error.
    
    Raised when N3 runtime encounters an error during execution:
    - Chain execution failed
    - Agent error
    - Tool invocation failed
    - Prompt rendering error
    
    Contains execution context for debugging.
    
    Example:
        >>> try:
        ...     runtime.chains.run("buggy_chain")
        ... except N3RuntimeError as e:
        ...     print(f"Execution failed: {e.message}")
        ...     print(f"Step: {e.context.get('failed_step')}")
    """
    
    def __init__(
        self,
        message: str,
        chain: Optional[str] = None,
        step: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        context = context or {}
        if chain:
            context["chain"] = chain
        if step:
            context["step"] = step
        super().__init__(message, context=context)


class N3SchemaError(N3ClientError):
    """Schema validation error.
    
    Raised when inputs don't match expected schema:
    - Missing required fields
    - Wrong field types
    - Invalid field values
    - Extra unknown fields (in strict mode)
    
    Contains validation details for debugging.
    
    Example:
        >>> try:
        ...     client.chains.run("typed_chain", invalid_param=123)
        ... except N3SchemaError as e:
        ...     print(f"Invalid input: {e.message}")
        ...     print(f"Errors: {e.validation_errors}")
    """
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[list] = None,
        request_id: Optional[str] = None,
    ):
        context = {}
        if validation_errors:
            context["validation_errors"] = validation_errors
        super().__init__(message, status_code=422, request_id=request_id, context=context)
        self.validation_errors = validation_errors or []


class N3CircuitBreakerError(N3Error):
    """Circuit breaker is open.
    
    Raised when circuit breaker prevents request due to repeated failures.
    The service may be down or experiencing issues. Wait before retrying.
    
    Example:
        >>> try:
        ...     client.chains.run("unreliable_chain")
        ... except N3CircuitBreakerError:
        ...     # Service is down, wait before retrying
        ...     time.sleep(60)
    """
    pass
