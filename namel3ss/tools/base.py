"""Base tool interface and common types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """Result from tool execution."""
    
    output: Any
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "success" if self.success else "error"
        return f"ToolResult(status={status}, output={self.output!r})"


class ToolError(Exception):
    """Base exception for tool errors."""
    
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
    Each tool has typed inputs/outputs and can be executed with validation.
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
