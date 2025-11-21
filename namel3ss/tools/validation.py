"""
Centralized validation functions for the Namel3ss tools subsystem.

This module provides pure validation functions for all tool-related operations:
- Tool configuration validation (names, types, parameters)
- Schema validation (input/output schemas)
- Registry operations validation
- Tool creation parameter validation
- Execution input validation

All validators are pure functions (no I/O, no side effects) that raise
ToolValidationError when validation fails.

Example:
    from namel3ss.tools.validation import validate_tool_name, validate_tool_type
    from namel3ss.tools.errors import ToolValidationError
    
    try:
        validate_tool_name("my-tool", tool_type="http")
        validate_tool_type("http")
    except ToolValidationError as e:
        print(f"Validation failed: {e.format()}")
"""

from typing import Any, Dict, Optional

from .base import BaseTool
from .errors import ToolValidationError


def validate_tool_name(
    name: str,
    *,
    tool_type: Optional[str] = None,
) -> None:
    """
    Validate tool name is non-empty and well-formed.
    
    Args:
        name: Tool name to validate
        tool_type: Optional tool type for context
        
    Raises:
        ToolValidationError: If name is invalid
        
    Example:
        >>> validate_tool_name("weather_api", tool_type="http")
        >>> validate_tool_name("")  # Raises ToolValidationError
    """
    if not name:
        raise ToolValidationError(
            "Tool name cannot be empty",
            code="TOOL001",
            tool_name=name,
            field="name",
            tool_type=tool_type,
        )
    
    if not isinstance(name, str):
        raise ToolValidationError(
            "Tool name must be a string",
            code="TOOL002",
            tool_name=str(name),
            field="name",
            value=type(name).__name__,
            expected="str",
            tool_type=tool_type,
        )
    
    # Check for reasonable length
    if len(name) > 200:
        raise ToolValidationError(
            f"Tool name too long (max 200 characters, got {len(name)})",
            code="TOOL003",
            tool_name=name,
            field="name",
            value=len(name),
            expected="<= 200 characters",
            tool_type=tool_type,
        )


def validate_tool_type(
    tool_type: str,
    *,
    tool_name: Optional[str] = None,
) -> None:
    """
    Validate tool type is non-empty and well-formed.
    
    Args:
        tool_type: Tool type to validate
        tool_name: Optional tool name for context
        
    Raises:
        ToolValidationError: If tool_type is invalid
        
    Example:
        >>> validate_tool_type("http", tool_name="weather")
        >>> validate_tool_type("")  # Raises ToolValidationError
    """
    if not tool_type:
        raise ToolValidationError(
            "Tool type cannot be empty",
            code="TOOL004",
            tool_name=tool_name,
            field="tool_type",
            tool_type=tool_type,
        )
    
    if not isinstance(tool_type, str):
        raise ToolValidationError(
            "Tool type must be a string",
            code="TOOL005",
            tool_name=tool_name,
            field="tool_type",
            value=type(tool_type).__name__,
            expected="str",
            tool_type=str(tool_type),
        )


def validate_timeout(
    timeout: float,
    *,
    tool_name: Optional[str] = None,
    tool_type: Optional[str] = None,
) -> None:
    """
    Validate timeout is positive number.
    
    Args:
        timeout: Timeout value in seconds
        tool_name: Optional tool name for context
        tool_type: Optional tool type for context
        
    Raises:
        ToolValidationError: If timeout is invalid
        
    Example:
        >>> validate_timeout(30.0)
        >>> validate_timeout(-5.0)  # Raises ToolValidationError
    """
    if not isinstance(timeout, (int, float)):
        raise ToolValidationError(
            "Timeout must be a number",
            code="TOOL006",
            tool_name=tool_name,
            field="timeout",
            value=type(timeout).__name__,
            expected="float or int",
            tool_type=tool_type,
        )
    
    if timeout <= 0:
        raise ToolValidationError(
            "Timeout must be positive",
            code="TOOL007",
            tool_name=tool_name,
            field="timeout",
            value=timeout,
            expected="> 0",
            tool_type=tool_type,
        )
    
    # Warn about very long timeouts (could indicate misconfiguration)
    if timeout > 3600:  # 1 hour
        raise ToolValidationError(
            f"Timeout suspiciously long: {timeout} seconds (max recommended: 3600)",
            code="TOOL008",
            tool_name=tool_name,
            field="timeout",
            value=timeout,
            expected="<= 3600 seconds",
            tool_type=tool_type,
        )


def validate_schema(
    schema: Dict[str, Any],
    *,
    schema_type: str = "input",
    tool_name: Optional[str] = None,
    tool_type: Optional[str] = None,
) -> None:
    """
    Validate tool input/output schema is well-formed.
    
    Args:
        schema: Schema dictionary to validate
        schema_type: "input" or "output"
        tool_name: Optional tool name for context
        tool_type: Optional tool type for context
        
    Raises:
        ToolValidationError: If schema is invalid
        
    Example:
        >>> validate_schema({"location": {"type": "string", "required": True}})
        >>> validate_schema("not a dict")  # Raises ToolValidationError
    """
    if not isinstance(schema, dict):
        raise ToolValidationError(
            f"Tool {schema_type} schema must be a dictionary",
            code="TOOL009",
            tool_name=tool_name,
            field=f"{schema_type}_schema",
            value=type(schema).__name__,
            expected="dict",
            tool_type=tool_type,
        )
    
    # Validate each field in schema
    for field_name, field_spec in schema.items():
        if not isinstance(field_name, str):
            raise ToolValidationError(
                f"Schema field name must be string, got {type(field_name).__name__}",
                code="TOOL010",
                tool_name=tool_name,
                field=f"{schema_type}_schema.{field_name}",
                value=type(field_name).__name__,
                expected="str",
                tool_type=tool_type,
            )
        
        # Field spec can be dict or simple type
        if isinstance(field_spec, dict):
            # Validate known keys
            allowed_keys = {"type", "required", "default", "description", "enum", "format"}
            unknown_keys = set(field_spec.keys()) - allowed_keys
            if unknown_keys:
                raise ToolValidationError(
                    f"Unknown schema keys: {unknown_keys}",
                    code="TOOL011",
                    tool_name=tool_name,
                    field=f"{schema_type}_schema.{field_name}",
                    value=list(unknown_keys),
                    expected=f"One of: {allowed_keys}",
                    tool_type=tool_type,
                )


def validate_http_method(
    method: str,
    *,
    tool_name: Optional[str] = None,
) -> None:
    """
    Validate HTTP method is supported.
    
    Args:
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        tool_name: Optional tool name for context
        
    Raises:
        ToolValidationError: If method is invalid
        
    Example:
        >>> validate_http_method("GET", tool_name="api")
        >>> validate_http_method("INVALID")  # Raises ToolValidationError
    """
    valid_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
    method_upper = method.upper() if isinstance(method, str) else ""
    
    if method_upper not in valid_methods:
        raise ToolValidationError(
            f"Invalid HTTP method: {method}",
            code="TOOL012",
            tool_name=tool_name,
            field="method",
            value=method,
            expected=f"One of: {valid_methods}",
            tool_type="http",
        )


def validate_http_endpoint(
    endpoint: str,
    *,
    tool_name: Optional[str] = None,
) -> None:
    """
    Validate HTTP endpoint URL is non-empty and well-formed.
    
    Args:
        endpoint: Endpoint URL
        tool_name: Optional tool name for context
        
    Raises:
        ToolValidationError: If endpoint is invalid
        
    Example:
        >>> validate_http_endpoint("https://api.example.com/v1")
        >>> validate_http_endpoint("")  # Raises ToolValidationError
    """
    if not endpoint:
        raise ToolValidationError(
            "HTTP endpoint cannot be empty",
            code="TOOL013",
            tool_name=tool_name,
            field="endpoint",
            tool_type="http",
        )
    
    if not isinstance(endpoint, str):
        raise ToolValidationError(
            "HTTP endpoint must be a string",
            code="TOOL014",
            tool_name=tool_name,
            field="endpoint",
            value=type(endpoint).__name__,
            expected="str",
            tool_type="http",
        )
    
    # Basic URL format check
    if not (endpoint.startswith("http://") or endpoint.startswith("https://")):
        raise ToolValidationError(
            "HTTP endpoint must start with http:// or https://",
            code="TOOL015",
            tool_name=tool_name,
            field="endpoint",
            value=endpoint,
            expected="URL starting with http:// or https://",
            tool_type="http",
        )


def validate_http_headers(
    headers: Dict[str, str],
    *,
    tool_name: Optional[str] = None,
) -> None:
    """
    Validate HTTP headers dictionary.
    
    Args:
        headers: Headers dictionary
        tool_name: Optional tool name for context
        
    Raises:
        ToolValidationError: If headers are invalid
        
    Example:
        >>> validate_http_headers({"Content-Type": "application/json"})
        >>> validate_http_headers("not a dict")  # Raises ToolValidationError
    """
    if not isinstance(headers, dict):
        raise ToolValidationError(
            "HTTP headers must be a dictionary",
            code="TOOL016",
            tool_name=tool_name,
            field="headers",
            value=type(headers).__name__,
            expected="dict",
            tool_type="http",
        )
    
    for key, value in headers.items():
        if not isinstance(key, str):
            raise ToolValidationError(
                f"Header name must be string, got {type(key).__name__}",
                code="TOOL017",
                tool_name=tool_name,
                field=f"headers.{key}",
                value=type(key).__name__,
                expected="str",
                tool_type="http",
            )
        
        if not isinstance(value, str):
            raise ToolValidationError(
                f"Header value must be string, got {type(value).__name__}",
                code="TOOL018",
                tool_name=tool_name,
                field=f"headers.{key}",
                value=type(value).__name__,
                expected="str",
                tool_type="http",
            )


def validate_python_code(
    code: str,
    *,
    tool_name: Optional[str] = None,
) -> None:
    """
    Validate Python code string is non-empty.
    
    Args:
        code: Python code to validate
        tool_name: Optional tool name for context
        
    Raises:
        ToolValidationError: If code is invalid
        
    Note:
        This performs basic validation only. Syntax checking happens at execution.
        
    Example:
        >>> validate_python_code("print('hello')")
        >>> validate_python_code("")  # Raises ToolValidationError
    """
    if not code:
        raise ToolValidationError(
            "Python code cannot be empty",
            code="TOOL019",
            tool_name=tool_name,
            field="code",
            tool_type="python",
        )
    
    if not isinstance(code, str):
        raise ToolValidationError(
            "Python code must be a string",
            code="TOOL020",
            tool_name=tool_name,
            field="code",
            value=type(code).__name__,
            expected="str",
            tool_type="python",
        )


def validate_tool_instance(
    tool: Any,
    *,
    tool_name: Optional[str] = None,
) -> None:
    """
    Validate object is a valid BaseTool instance.
    
    Args:
        tool: Object to validate
        tool_name: Optional tool name for context
        
    Raises:
        ToolValidationError: If not a valid tool instance
        
    Example:
        >>> from namel3ss.tools import HttpTool
        >>> tool = HttpTool(name="api", endpoint="https://api.com")
        >>> validate_tool_instance(tool)
        >>> validate_tool_instance("not a tool")  # Raises ToolValidationError
    """
    if not isinstance(tool, BaseTool):
        raise ToolValidationError(
            f"Object is not a BaseTool instance (got {type(tool).__name__})",
            code="TOOL021",
            tool_name=tool_name,
            field="tool",
            value=type(tool).__name__,
            expected="BaseTool subclass",
        )
    
    # Validate tool has required attributes
    required_attrs = ["name", "tool_type", "execute"]
    for attr in required_attrs:
        if not hasattr(tool, attr):
            raise ToolValidationError(
                f"Tool instance missing required attribute: {attr}",
                code="TOOL022",
                tool_name=tool_name or getattr(tool, "name", None),
                field=attr,
                expected=f"Tool with {attr} attribute",
            )


def validate_execution_inputs(
    inputs: Dict[str, Any],
    *,
    schema: Optional[Dict[str, Any]] = None,
    tool_name: Optional[str] = None,
    tool_type: Optional[str] = None,
) -> None:
    """
    Validate tool execution inputs against schema.
    
    Args:
        inputs: Input dictionary to validate
        schema: Optional input schema to validate against
        tool_name: Optional tool name for context
        tool_type: Optional tool type for context
        
    Raises:
        ToolValidationError: If inputs are invalid
        
    Example:
        >>> schema = {"location": {"type": "string", "required": True}}
        >>> validate_execution_inputs({"location": "NYC"}, schema=schema)
        >>> validate_execution_inputs({}, schema=schema)  # Raises ToolValidationError
    """
    if not isinstance(inputs, dict):
        raise ToolValidationError(
            "Tool execution inputs must be a dictionary",
            code="TOOL023",
            tool_name=tool_name,
            field="inputs",
            value=type(inputs).__name__,
            expected="dict",
            tool_type=tool_type,
        )
    
    # If schema provided, validate against it
    if schema:
        for field_name, field_spec in schema.items():
            if isinstance(field_spec, dict):
                # Check required fields
                if field_spec.get("required", True):
                    if field_name not in inputs:
                        raise ToolValidationError(
                            f"Missing required input field: {field_name}",
                            code="TOOL024",
                            tool_name=tool_name,
                            field=field_name,
                            expected=f"Required field {field_name}",
                            tool_type=tool_type,
                        )


def validate_tool_config(
    *,
    name: str,
    tool_type: str,
    timeout: float = 30.0,
    input_schema: Optional[Dict[str, Any]] = None,
    output_schema: Optional[Dict[str, Any]] = None,
    endpoint: Optional[str] = None,
    method: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    code: Optional[str] = None,
    function: Any = None,
) -> None:
    """
    Validate complete tool configuration before creation.
    
    Runs all applicable validators based on tool type.
    
    Args:
        name: Tool name
        tool_type: Tool type
        timeout: Execution timeout
        input_schema: Input schema
        output_schema: Output schema
        endpoint: HTTP endpoint (for HTTP tools)
        method: HTTP method (for HTTP tools)
        headers: HTTP headers (for HTTP tools)
        code: Python code (for Python tools)
        function: Python function (for Python tools)
        
    Raises:
        ToolValidationError: If any validation fails
        
    Example:
        >>> validate_tool_config(
        ...     name="weather",
        ...     tool_type="http",
        ...     endpoint="https://api.weather.com",
        ...     method="GET"
        ... )
    """
    # Validate common fields
    validate_tool_name(name, tool_type=tool_type)
    validate_tool_type(tool_type, tool_name=name)
    validate_timeout(timeout, tool_name=name, tool_type=tool_type)
    
    # Validate schemas if provided
    if input_schema:
        validate_schema(input_schema, schema_type="input", tool_name=name, tool_type=tool_type)
    
    if output_schema:
        validate_schema(output_schema, schema_type="output", tool_name=name, tool_type=tool_type)
    
    # Type-specific validation
    if tool_type == "http":
        if endpoint:
            validate_http_endpoint(endpoint, tool_name=name)
        if method:
            validate_http_method(method, tool_name=name)
        if headers:
            validate_http_headers(headers, tool_name=name)
    
    elif tool_type == "python":
        if code:
            validate_python_code(code, tool_name=name)
        if function is None and code is None:
            raise ToolValidationError(
                "Python tool requires either 'function' or 'code' parameter",
                code="TOOL025",
                tool_name=name,
                field="function/code",
                expected="Either function or code",
                tool_type="python",
            )


__all__ = [
    "validate_tool_name",
    "validate_tool_type",
    "validate_timeout",
    "validate_schema",
    "validate_http_method",
    "validate_http_endpoint",
    "validate_http_headers",
    "validate_python_code",
    "validate_tool_instance",
    "validate_execution_inputs",
    "validate_tool_config",
]
