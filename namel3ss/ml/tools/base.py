"""Base tool interface and types."""

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


class ToolError(Exception):
    """Base exception for tool errors."""
    
    def __init__(self, message: str, *, tool_name: Optional[str] = None,
                 status_code: Optional[int] = None, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.tool_name = tool_name
        self.status_code = status_code
        self.original_error = original_error


class Tool(ABC):
    """
    Abstract base class for tools.
    
    Tools represent external capabilities that can be invoked from Namel3ss programs.
    """
    
    def __init__(self, *, name: str, input_schema: Optional[Dict] = None,
                 output_schema: Optional[Dict] = None, **config):
        """
        Initialize the tool.
        
        Args:
            name: Tool identifier
            input_schema: JSON schema describing expected inputs
            output_schema: JSON schema describing outputs
            **config: Tool-specific configuration
        """
        self.name = name
        self.input_schema = input_schema or {}
        self.output_schema = output_schema or {}
        self.config = config
    
    @abstractmethod
    def execute(self, **inputs) -> ToolResult:
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
        required = self.input_schema.get("required", [])
        for field in required:
            if field not in inputs:
                raise ToolError(
                    f"Missing required field '{field}'",
                    tool_name=self.name
                )
        
        # Type validation for known fields
        properties = self.input_schema.get("properties", {})
        for field_name, field_value in inputs.items():
            if field_name not in properties:
                continue
            
            expected_type = properties[field_name].get("type")
            if not expected_type:
                continue
            
            # Simple type checking
            type_map = {
                "string": str,
                "number": (int, float),
                "integer": int,
                "boolean": bool,
                "object": dict,
                "array": list,
            }
            
            python_type = type_map.get(expected_type)
            if python_type and not isinstance(field_value, python_type):
                raise ToolError(
                    f"Field '{field_name}' must be of type {expected_type}, "
                    f"got {type(field_value).__name__}",
                    tool_name=self.name
                )
