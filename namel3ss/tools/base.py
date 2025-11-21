"""
Base tool interface and common types for the Namel3ss tools subsystem.

This module defines the core abstractions for tools:
- BaseTool: Abstract base class for all tool implementations
- ToolResult: Standardized result format for tool execution
- ToolError: Base exception class for tool-related errors

Tools represent external capabilities that can be invoked from Namel3ss programs.
Each tool has typed inputs/outputs and can be executed with validation.

Architecture:
    Tool implementations extend BaseTool and implement the execute() method.
    Results are returned as ToolResult objects with success/error information.
    Errors are raised as ToolError subclasses with context.

Example:
    from namel3ss.tools.base import BaseTool, ToolResult
    
    class CustomTool(BaseTool):
        def execute(self, **inputs):
            result = perform_operation(inputs)
            return ToolResult(output=result, success=True)
    
    tool = CustomTool(name="custom", tool_type="custom")
    result = tool.execute(param="value")
    print(result.output)

Best Practices:
    - Always validate inputs before execution
    - Return ToolResult with success=False rather than raising exceptions for expected failures
    - Raise ToolError for unexpected failures or configuration issues
    - Include metadata in ToolResult for debugging and observability
    - Set appropriate timeouts to prevent hanging operations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """
    Result from tool execution.
    
    Standardized format for all tool execution results, including both
    successful and failed operations. Contains output data, success status,
    error messages, and additional metadata.
    
    Attributes:
        output: The actual result data (can be any type)
        success: True if execution succeeded, False otherwise
        error: Error message if execution failed (None on success)
        metadata: Additional context (status codes, headers, timing, etc.)
    
    Examples:
        Successful execution:
        >>> result = ToolResult(
        ...     output={"temperature": 72, "condition": "sunny"},
        ...     success=True,
        ...     metadata={"status_code": 200, "cache_hit": True}
        ... )
        >>> print(result.output["temperature"])
        72
        
        Failed execution:
        >>> result = ToolResult(
        ...     output=None,
        ...     success=False,
        ...     error="Connection timeout",
        ...     metadata={"timeout_seconds": 30}
        ... )
        >>> if not result.success:
        ...     print(f"Error: {result.error}")
        Error: Connection timeout
    
    Design Notes:
        - Use success=False for expected failures (network errors, invalid input)
        - Include relevant metadata for debugging (status codes, timestamps)
        - Keep output type flexible to support various tool types
        - Metadata dict allows extensibility without changing the interface
    """
    
    output: Any
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "success" if self.success else "error"
        return f"ToolResult(status={status}, output={self.output!r})"


class ToolError(Exception):
    """
    Base exception for tool errors.
    
    All tool-related exceptions extend this base class. Provides context
    about the tool, error type, and original exception if wrapped.
    
    Attributes:
        message: Human-readable error description
        tool_name: Name of the tool that raised the error
        status_code: HTTP status code (for HTTP tool errors)
        original_error: Wrapped exception if this error wraps another
    
    Subclasses:
        ToolValidationError: Validation failures during tool creation/input checking
        ToolRegistrationError: Errors during tool registration in registry
        ToolExecutionError: Runtime errors during tool execution
    
    Example:
        >>> from namel3ss.tools.base import ToolError
        >>> try:
        ...     raise ToolError(
        ...         "Tool execution failed",
        ...         tool_name="weather_api",
        ...         status_code=500
        ...     )
        ... except ToolError as e:
        ...     print(f"Error in {e.tool_name}: {e}")
        Error in weather_api: Tool execution failed
    
    Best Practices:
        - Always include tool_name for context
        - Wrap original exceptions to preserve stack traces
        - Use specific subclasses (ToolValidationError, etc.) when appropriate
        - Include status_code for HTTP tools
    """
    
    def __init__(
        self,
        message: str,
        *,
        tool_name: Optional[str] = None,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.tool_name = tool_name
        self.status_code = status_code
        self.original_error = original_error


class BaseTool(ABC):
    """
    Abstract base class for all tool implementations.
    
    Tools represent external capabilities that can be invoked from Namel3ss programs.
    Each tool has typed inputs/outputs, configurable timeouts, and schema-based validation.
    
    Architecture:
        Subclasses must implement the execute() method to define tool behavior.
        Input/output schemas define the tool's contract using JSON Schema format.
        Validation is performed automatically before execution.
    
    Attributes:
        name: Unique identifier for the tool instance
        tool_type: Type identifier (http, python, database, etc.)
        input_schema: JSON Schema defining expected inputs
        output_schema: JSON Schema defining output format
        timeout: Maximum execution time in seconds
        config: Tool-specific configuration dictionary
    
    Built-in Tool Types:
        - http: REST/HTTP API calls with configurable methods and headers
        - python: Execute Python code or call Python functions
        - database: Query databases (PostgreSQL, MySQL, MongoDB, etc.)
        - vector_search: Search vector databases for semantic similarity
    
    Example Implementation:
        >>> from namel3ss.tools.base import BaseTool, ToolResult
        >>> 
        >>> class CalculatorTool(BaseTool):
        ...     def execute(self, **inputs):
        ...         expression = inputs.get("expression")
        ...         result = eval(expression)  # Note: Use safely in production!
        ...         return ToolResult(output=result, success=True)
        >>> 
        >>> calc = CalculatorTool(
        ...     name="calculator",
        ...     tool_type="python",
        ...     input_schema={
        ...         "expression": {"type": "string", "required": True}
        ...     },
        ...     timeout=5.0
        ... )
        >>> result = calc.execute(expression="2 + 2")
        >>> print(result.output)
        4
    
    Lifecycle:
        1. Tool is created with configuration (name, type, schemas, timeout)
        2. Tool is optionally registered in global registry
        3. execute() is called with input dictionary
        4. validate_inputs() checks inputs against schema
        5. Tool-specific logic runs
        6. ToolResult is returned
    
    Best Practices:
        - Define clear input/output schemas for type safety
        - Set appropriate timeouts (default 30s, adjust per tool)
        - Validate all inputs before executing operations
        - Return ToolResult with success=False for expected failures
        - Raise ToolError for configuration or unexpected errors
        - Include metadata in results for debugging
        - Use descriptive tool names (e.g., "weather_api" not "api")
    
    Error Handling:
        - Configuration errors: Raise ToolError in __init__
        - Input validation errors: Raise ToolError in validate_inputs()
        - Execution failures: Return ToolResult(success=False) or raise ToolError
        - Timeout errors: Automatically handled by execution framework
    
    Thread Safety:
        Tool instances should be thread-safe if used concurrently.
        Avoid mutable shared state in instance variables.
    """
    
    def __init__(
        self,
        *,
        name: str,
        tool_type: str,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        **config: Any,
    ):
        """
        Initialize the tool.
        
        Args:
            name: Tool identifier
            tool_type: Type of tool (http, python, database, etc.)
            input_schema: JSON schema describing expected inputs
            output_schema: JSON schema describing outputs
            timeout: Execution timeout in seconds
            **config: Tool-specific configuration
        """
        self.name = name
        self.tool_type = tool_type
        self.input_schema = input_schema or {}
        self.output_schema = output_schema or {}
        self.timeout = timeout
        self.config = config
    
    @abstractmethod
    def execute(self, **inputs: Any) -> ToolResult:
        """
        Execute the tool with given inputs.
        
        Args:
            **inputs: Tool inputs matching input_schema
            
        Returns:
            ToolResult with output data
            
        Raises:
            ToolError: If execution fails
        """
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Validate inputs against schema.
        
        Args:
            inputs: Input dictionary to validate
            
        Raises:
            ToolError: If validation fails
        """
        if not self.input_schema:
            return
        
        # Basic validation - check required fields
        for field_name, field_spec in self.input_schema.items():
            if isinstance(field_spec, dict) and field_spec.get("required", True):
                if field_name not in inputs:
                    raise ToolError(
                        f"Missing required input '{field_name}'",
                        tool_name=self.name,
                    )
    
    def get_tool_type(self) -> str:
        """Return the tool type identifier."""
        return self.tool_type
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, type={self.tool_type!r})"
