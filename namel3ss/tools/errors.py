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


__all__ = [
    "ToolValidationError",
    "ToolRegistrationError",
    "ToolExecutionError",
]
