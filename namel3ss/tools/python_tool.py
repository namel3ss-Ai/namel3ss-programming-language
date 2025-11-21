"""Python tool implementation for executing Python code or functions."""

from typing import Any, Callable, Dict, Optional

from .base import BaseTool, ToolError, ToolResult
from .errors import ToolExecutionError, ToolValidationError
from .validation import validate_python_code


class PythonTool(BaseTool):
    """
    Tool for executing Python code or calling Python functions.
    
    Can execute either a code string or call a provided function.
    Supports passing inputs as function arguments or code variables.
    
    Features:
        - Execute arbitrary Python code strings
        - Call Python functions with arguments
        - Capture return values or result variable
        - Safe execution with timeout support
        - Error context in results
    
    Configuration:
        name: Tool identifier
        function: Python callable to execute (optional)
        code: Python code string to execute (optional)
        input_schema: Schema for function arguments/code variables
        timeout: Execution timeout in seconds
    
    One of function or code must be provided.
    
    Example with function:
        >>> def calculate_tax(amount, rate):
        ...     return amount * rate
        >>> 
        >>> tool = create_tool(
        ...     name="tax_calc",
        ...     tool_type="python",
        ...     function=calculate_tax,
        ...     input_schema={
        ...         "amount": {"type": "number", "required": True},
        ...         "rate": {"type": "number", "required": True}
        ...     }
        ... )
        >>> result = tool.execute(amount=100, rate=0.08)
        >>> print(result.output)
        8.0
    
    Example with code:
        >>> code = '''
        ... total = price * quantity
        ... discount = total * 0.1
        ... result = total - discount
        ... '''
        >>> tool = create_tool(
        ...     name="discount_calc",
        ...     tool_type="python",
        ...     code=code
        ... )
        >>> result = tool.execute(price=50, quantity=3)
        >>> print(result.output['result'])
        135.0
    
    Security Warnings:
        - Do NOT execute untrusted code strings (arbitrary code execution risk)
        - Validate and sanitize all inputs
        - Use functions instead of code strings when possible
        - Consider sandboxing for untrusted code
        - Set timeouts to prevent infinite loops
    
    Best Practices:
        - Prefer function parameter over code for security
        - Define clear input schemas
        - Use 'result' variable in code for return values
        - Handle exceptions in code/functions
        - Set appropriate timeouts
    """
    
    def __init__(
        self,
        *,
        name: str,
        tool_type: str = "python",
        function: Optional[Callable] = None,
        code: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        **config: Any,
    ):
        """
        Initialize Python tool.
        
        Args:
            name: Tool identifier
            tool_type: Always "python"
            function: Python callable to execute
            code: Python code string to execute
            input_schema: Schema for inputs
            output_schema: Schema for outputs
            timeout: Execution timeout in seconds
            **config: Additional configuration
        """
        super().__init__(
            name=name,
            tool_type=tool_type,
            input_schema=input_schema,
            output_schema=output_schema,
            timeout=timeout,
            **config,
        )
        
        if function is None and code is None:
            raise ToolValidationError(
                "Python tool requires either 'function' or 'code' parameter",
                code="TOOL025",
                tool_name=name,
                field="function/code",
                expected="Either function or code",
                tool_type="python",
            )
        
        # Validate code if provided
        if code:
            validate_python_code(code, tool_name=name)
        
        self.function = function
        self.code = code
    
    def execute(self, **inputs: Any) -> ToolResult:
        """
        Execute the Python function or code.
        
        Args:
            **inputs: Function arguments or variables for code execution
            
        Returns:
            ToolResult with execution result
            
        Raises:
            ToolError: If execution fails
        """
        try:
            self.validate_inputs(inputs)
            
            if self.function is not None:
                # Call the function
                result = self.function(**inputs)
                return ToolResult(
                    output=result,
                    success=True,
                    metadata={"execution_type": "function"},
                )
            
            elif self.code is not None:
                # Execute code string
                local_vars = dict(inputs)
                exec(self.code, {}, local_vars)
                
                # Return the result variable if present
                result = local_vars.get("result", local_vars)
                return ToolResult(
                    output=result,
                    success=True,
                    metadata={"execution_type": "code"},
                )
            
            else:
                raise ToolExecutionError(
                    "No function or code to execute",
                    code="TOOL032",
                    tool_name=self.name,
                    operation="execute",
                )
        
        except Exception as e:
            # Return result with error rather than raising
            # This allows callers to handle errors gracefully
            return ToolResult(
                output=None,
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                metadata={
                    "error_type": type(e).__name__,
                    "execution_type": "function" if self.function else "code",
                },
            )
    
    def get_tool_type(self) -> str:
        """Return 'python'."""
        return "python"
