"""
Tool-specific exception types for the Namel3ss tools subsystem.

This module provides a hierarchy of exceptions for tool-related errors:
- ToolValidationError: Validation failures during tool creation or input checking
- ToolRegistrationError: Errors when registering tools in the registry
- ToolExecutionError: Runtime errors during tool execution

All exceptions extend the base ToolError and include contextual information
for debugging (error codes, tool names, field names, etc.).

Example:
    from namel3ss.tools.errors import ToolValidationError
    
    raise ToolValidationError(
        "Tool name cannot be empty",
        code="TOOL001",
        tool_name="",
        field="name"
    )
"""

from typing import Any, Optional

from .base import ToolError


class ToolValidationError(ToolError):
    """
    Exception raised when tool validation fails.
    
    Used for:
    - Invalid tool configuration (empty names, invalid types)
    - Schema validation failures (missing required fields, type mismatches)
    - Input validation errors (missing arguments, invalid values)
    - Tool creation parameter validation
    
    Attributes:
        message: Human-readable error message
        code: Error code (TOOL001-TOOL020)
        tool_name: Name of the tool being validated
        field: Field name that failed validation
        value: The invalid value
        tool_type: Type of tool (http, python, etc.)
        expected: Expected value or type
    
    Example:
        >>> from namel3ss.tools.errors import ToolValidationError
        >>> raise ToolValidationError(
        ...     "Tool name cannot be empty",
        ...     code="TOOL001",
        ...     tool_name="",
        ...     field="name"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        *,
        code: str,
        tool_name: Optional[str] = None,
        field: Optional[str] = None,
        value: Any = None,
        tool_type: Optional[str] = None,
        expected: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize validation error.
        
        Args:
            message: Error description
            code: Error code (TOOL001-TOOL020)
            tool_name: Name of the tool
            field: Field that failed validation
            value: The invalid value
            tool_type: Tool type identifier
            expected: Expected value or type description
            original_error: Original exception if wrapped
        """
        super().__init__(message, tool_name=tool_name, original_error=original_error)
        self.message = message
        self.code = code
        self.field = field
        self.value = value
        self.tool_type = tool_type
        self.expected = expected
    
    def format(self) -> str:
        """
        Format error with full context.
        
        Returns:
            Formatted error message with code and context
            
        Example:
            >>> error = ToolValidationError(
            ...     "Tool name cannot be empty",
            ...     code="TOOL001",
            ...     tool_name="",
            ...     field="name"
            ... )
            >>> print(error.format())
            [TOOL001] Tool Validation Error
            Tool: ''
            Field: name
            Message: Tool name cannot be empty
        """
        parts = [f"[{self.code}] Tool Validation Error"]
        
        if self.tool_name is not None:
            parts.append(f"Tool: {self.tool_name!r}")
        
        if self.tool_type:
            parts.append(f"Type: {self.tool_type}")
        
        if self.field:
            parts.append(f"Field: {self.field}")
        
        if self.expected:
            parts.append(f"Expected: {self.expected}")
        
        if self.value is not None:
            parts.append(f"Got: {self.value!r}")
        
        parts.append(f"Message: {self.message}")
        
        if self.original_error:
            parts.append(f"Caused by: {self.original_error}")
        
        return "\n".join(parts)
    
    def __str__(self) -> str:
        """Return formatted error message."""
        return self.format()


class ToolRegistrationError(ToolError):
    """
    Exception raised when tool registration fails.
    
    Used for:
    - Duplicate tool registration attempts
    - Invalid tool instances
    - Registry state errors
    - Tool lookup failures
    
    Attributes:
        message: Human-readable error message
        code: Error code (TOOL021-TOOL030)
        tool_name: Name of the tool
        registry_state: Current registry state description
        conflict: Description of conflicting registration
    
    Example:
        >>> from namel3ss.tools.errors import ToolRegistrationError
        >>> raise ToolRegistrationError(
        ...     "Tool 'weather' is already registered",
        ...     code="TOOL021",
        ...     tool_name="weather",
        ...     conflict="Existing tool of type 'http'"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        *,
        code: str,
        tool_name: Optional[str] = None,
        registry_state: Optional[str] = None,
        conflict: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize registration error.
        
        Args:
            message: Error description
            code: Error code (TOOL021-TOOL030)
            tool_name: Name of the tool
            registry_state: Description of registry state
            conflict: Description of conflict
            original_error: Original exception if wrapped
        """
        super().__init__(message, tool_name=tool_name, original_error=original_error)
        self.message = message
        self.code = code
        self.registry_state = registry_state
        self.conflict = conflict
    
    def format(self) -> str:
        """
        Format error with full context.
        
        Returns:
            Formatted error message with code and context
        """
        parts = [f"[{self.code}] Tool Registration Error"]
        
        if self.tool_name:
            parts.append(f"Tool: {self.tool_name!r}")
        
        parts.append(f"Message: {self.message}")
        
        if self.conflict:
            parts.append(f"Conflict: {self.conflict}")
        
        if self.registry_state:
            parts.append(f"Registry State: {self.registry_state}")
        
        if self.original_error:
            parts.append(f"Caused by: {self.original_error}")
        
        return "\n".join(parts)
    
    def __str__(self) -> str:
        """Return formatted error message."""
        return self.format()


class ToolExecutionError(ToolError):
    """
    Exception raised when tool execution fails at runtime.
    
    Used for:
    - HTTP request failures (timeouts, connection errors)
    - Python code execution errors
    - External API errors
    - Timeout errors
    - Resource unavailability
    
    Attributes:
        message: Human-readable error message
        code: Error code (TOOL031-TOOL050)
        tool_name: Name of the tool
        operation: Operation being performed
        status_code: HTTP status code (for HTTP tools)
        timeout: Whether the error was due to timeout
        retryable: Whether the operation can be retried
    
    Example:
        >>> from namel3ss.tools.errors import ToolExecutionError
        >>> raise ToolExecutionError(
        ...     "HTTP request timed out",
        ...     code="TOOL033",
        ...     tool_name="weather_api",
        ...     operation="GET /v1/current",
        ...     timeout=True,
        ...     retryable=True
        ... )
    """
    
    def __init__(
        self,
        message: str,
        *,
        code: str,
        tool_name: Optional[str] = None,
        operation: Optional[str] = None,
        status_code: Optional[int] = None,
        timeout: bool = False,
        retryable: bool = False,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize execution error.
        
        Args:
            message: Error description
            code: Error code (TOOL031-TOOL050)
            tool_name: Name of the tool
            operation: Operation being performed
            status_code: HTTP status code if applicable
            timeout: Whether error was due to timeout
            retryable: Whether operation can be retried
            original_error: Original exception if wrapped
        """
        super().__init__(
            message,
            tool_name=tool_name,
            status_code=status_code,
            original_error=original_error,
        )
        self.message = message
        self.code = code
        self.operation = operation
        self.timeout = timeout
        self.retryable = retryable
    
    def format(self) -> str:
        """
        Format error with full context.
        
        Returns:
            Formatted error message with code and context
        """
        parts = [f"[{self.code}] Tool Execution Error"]
        
        if self.tool_name:
            parts.append(f"Tool: {self.tool_name!r}")
        
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        
        if self.status_code:
            parts.append(f"Status Code: {self.status_code}")
        
        parts.append(f"Message: {self.message}")
        
        if self.timeout:
            parts.append("Reason: Request timed out")
        
        if self.retryable:
            parts.append("Retryable: Yes")
        
        if self.original_error:
            parts.append(f"Caused by: {type(self.original_error).__name__}: {self.original_error}")
        
        return "\n".join(parts)
    
    def __str__(self) -> str:
        """Return formatted error message."""
        return self.format()


class ToolTimeoutError(ToolExecutionError):
    """
    Exception raised when tool execution exceeds timeout.
    
    Specialized subclass of ToolExecutionError for timeout scenarios.
    Always has timeout=True and retryable=True by default.
    
    Attributes:
        timeout_seconds: The timeout value that was exceeded
        elapsed_seconds: Actual elapsed time before timeout
    
    Example:
        >>> raise ToolTimeoutError(
        ...     "Tool execution exceeded 30 second timeout",
        ...     code="TOOL034",
        ...     tool_name="slow_api",
        ...     timeout_seconds=30.0,
        ...     elapsed_seconds=30.5
        ... )
    """
    
    def __init__(
        self,
        message: str,
        *,
        code: str = "TOOL034",
        tool_name: Optional[str] = None,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
        original_error: Optional[Exception] = None,
    ):
        """Initialize timeout error."""
        super().__init__(
            message,
            code=code,
            tool_name=tool_name,
            operation=operation,
            timeout=True,
            retryable=True,
            original_error=original_error,
        )
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
    
    def format(self) -> str:
        """Format error with timeout details."""
        parts = [f"[{self.code}] Tool Timeout Error"]
        
        if self.tool_name:
            parts.append(f"Tool: {self.tool_name!r}")
        
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        
        parts.append(f"Message: {self.message}")
        
        if self.timeout_seconds is not None:
            parts.append(f"Timeout: {self.timeout_seconds}s")
        
        if self.elapsed_seconds is not None:
            parts.append(f"Elapsed: {self.elapsed_seconds}s")
        
        if self.original_error:
            parts.append(f"Caused by: {self.original_error}")
        
        return "\n".join(parts)


class ToolConfigurationError(ToolError):
    """
    Exception raised when tool configuration is invalid.
    
    Used for:
    - Missing required configuration
    - Invalid configuration values
    - Configuration conflicts
    - Credential/authentication errors
    
    Attributes:
        code: Error code (TOOL051-TOOL060)
        config_field: Configuration field that is invalid
        config_value: The invalid value
        expected: Expected value or format
    
    Example:
        >>> raise ToolConfigurationError(
        ...     "Missing required auth_token",
        ...     code="TOOL051",
        ...     tool_name="api_tool",
        ...     config_field="auth_token",
        ...     expected="Non-empty string"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        *,
        code: str,
        tool_name: Optional[str] = None,
        config_field: Optional[str] = None,
        config_value: Any = None,
        expected: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        """Initialize configuration error."""
        super().__init__(message, tool_name=tool_name, original_error=original_error)
        self.message = message
        self.code = code
        self.config_field = config_field
        self.config_value = config_value
        self.expected = expected
    
    def format(self) -> str:
        """Format error with configuration details."""
        parts = [f"[{self.code}] Tool Configuration Error"]
        
        if self.tool_name:
            parts.append(f"Tool: {self.tool_name!r}")
        
        if self.config_field:
            parts.append(f"Config Field: {self.config_field}")
        
        if self.expected:
            parts.append(f"Expected: {self.expected}")
        
        if self.config_value is not None:
            parts.append(f"Got: {self.config_value!r}")
        
        parts.append(f"Message: {self.message}")
        
        if self.original_error:
            parts.append(f"Caused by: {self.original_error}")
        
        return "\n".join(parts)
    
    def __str__(self) -> str:
        """Return formatted error message."""
        return self.format()


class ToolAuthenticationError(ToolConfigurationError):
    """
    Exception raised when tool authentication fails.
    
    Specialized subclass for authentication/authorization failures.
    
    Example:
        >>> raise ToolAuthenticationError(
        ...     "Invalid API key",
        ...     code="TOOL052",
        ...     tool_name="api_tool",
        ...     config_field="auth_token"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        *,
        code: str = "TOOL052",
        tool_name: Optional[str] = None,
        config_field: Optional[str] = "auth_token",
        original_error: Optional[Exception] = None,
    ):
        """Initialize authentication error."""
        super().__init__(
            message,
            code=code,
            tool_name=tool_name,
            config_field=config_field,
            expected="Valid authentication credentials",
            original_error=original_error,
        )


def serialize_tool_error(error: ToolError) -> Dict[str, Any]:
    """
    Serialize ToolError to dictionary for logging/API responses.
    
    Args:
        error: ToolError instance
    
    Returns:
        Dictionary with error details
    
    Example:
        >>> try:
        ...     # tool execution
        ...     pass
        ... except ToolError as e:
        ...     error_dict = serialize_tool_error(e)
        ...     logger.error("Tool error", extra=error_dict)
    """
    from namel3ss.tools.schemas import ToolErrorModel
    
    error_dict = {
        "code": getattr(error, "code", "UNKNOWN"),
        "message": str(error),
        "tool_name": error.tool_name,
        "type": type(error).__name__,
    }
    
    # Add specific fields based on error type
    if isinstance(error, ToolValidationError):
        error_dict.update({
            "field": error.field,
            "value": error.value,
            "tool_type": error.tool_type,
            "expected": error.expected,
        })
    elif isinstance(error, ToolExecutionError):
        error_dict.update({
            "operation": error.operation,
            "status_code": error.status_code,
            "timeout": error.timeout,
            "retryable": error.retryable,
        })
        if isinstance(error, ToolTimeoutError):
            error_dict.update({
                "timeout_seconds": error.timeout_seconds,
                "elapsed_seconds": error.elapsed_seconds,
            })
    elif isinstance(error, ToolConfigurationError):
        error_dict.update({
            "config_field": error.config_field,
            "config_value": error.config_value,
            "expected": error.expected,
        })
    
    if error.original_error:
        error_dict["original_error"] = {
            "type": type(error.original_error).__name__,
            "message": str(error.original_error),
        }
    
    return {k: v for k, v in error_dict.items() if v is not None}


def deserialize_tool_error(error_dict: Dict[str, Any]) -> ToolError:
    """
    Deserialize dictionary to ToolError.
    
    Args:
        error_dict: Dictionary from serialize_tool_error()
    
    Returns:
        ToolError instance
    
    Example:
        >>> error_dict = serialize_tool_error(error)
        >>> restored_error = deserialize_tool_error(error_dict)
    """
    error_type = error_dict.get("type", "ToolError")
    code = error_dict.get("code", "UNKNOWN")
    message = error_dict.get("message", "Unknown error")
    tool_name = error_dict.get("tool_name")
    
    if error_type == "ToolValidationError":
        return ToolValidationError(
            message,
            code=code,
            tool_name=tool_name,
            field=error_dict.get("field"),
            value=error_dict.get("value"),
            tool_type=error_dict.get("tool_type"),
            expected=error_dict.get("expected"),
        )
    elif error_type == "ToolTimeoutError":
        return ToolTimeoutError(
            message,
            code=code,
            tool_name=tool_name,
            operation=error_dict.get("operation"),
            timeout_seconds=error_dict.get("timeout_seconds"),
            elapsed_seconds=error_dict.get("elapsed_seconds"),
        )
    elif error_type == "ToolExecutionError":
        return ToolExecutionError(
            message,
            code=code,
            tool_name=tool_name,
            operation=error_dict.get("operation"),
            status_code=error_dict.get("status_code"),
            timeout=error_dict.get("timeout", False),
            retryable=error_dict.get("retryable", False),
        )
    elif error_type == "ToolAuthenticationError":
        return ToolAuthenticationError(
            message,
            code=code,
            tool_name=tool_name,
            config_field=error_dict.get("config_field"),
        )
    elif error_type == "ToolConfigurationError":
        return ToolConfigurationError(
            message,
            code=code,
            tool_name=tool_name,
            config_field=error_dict.get("config_field"),
            config_value=error_dict.get("config_value"),
            expected=error_dict.get("expected"),
        )
    else:
        return ToolError(
            message,
            tool_name=tool_name,
        )


__all__ = [
    "ToolValidationError",
    "ToolRegistrationError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolConfigurationError",
    "ToolAuthenticationError",
    "serialize_tool_error",
    "deserialize_tool_error",
]
