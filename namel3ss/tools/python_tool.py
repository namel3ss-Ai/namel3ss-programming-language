"""Python tool implementation for executing Python code or functions."""

from typing import Any, Callable, Dict, Optional

from .base import BaseTool, ToolError, ToolResult


class PythonTool(BaseTool):
    """
    Tool for executing Python code or calling Python functions.
    
    Can execute either a code string or call a provided function.
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
            raise ToolError(
                "Python tool requires either 'function' or 'code' parameter",
                tool_name=name,
            )
        
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
                raise ToolError("No function or code to execute", tool_name=self.name)
        
        except Exception as e:
            return ToolResult(
                output=None,
                success=False,
                error=str(e),
                metadata={"error_type": type(e).__name__},
            )
    
    def get_tool_type(self) -> str:
        """Return 'python'."""
        return "python"
