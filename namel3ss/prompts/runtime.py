"""
Runtime support for structured prompts with typed args and output schemas.

This module provides the PromptProgram abstraction for executing prompts
with argument validation, template rendering, and structured output handling.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from namel3ss.ast import Prompt, PromptArgument, OutputSchema, OutputField, OutputFieldType
from namel3ss.errors import N3Error


class PromptProgramError(N3Error):
    """Errors related to prompt program execution."""
    pass


@dataclass
class PromptProgram:
    """
    Runtime abstraction for a structured prompt.
    
    Handles:
    - Argument validation and coercion
    - Template rendering with placeholders
    - Output schema generation for LLM providers
    - Integration with validation layer
    
    Example:
        program = PromptProgram(prompt_definition)
        rendered = program.render_prompt({"text": "Hello", "max_words": 50})
        schema = program.get_output_schema()
    """
    
    definition: Prompt
    
    def render_prompt(self, args: Dict[str, Any]) -> str:
        """
        Render the prompt template with validated arguments.
        
        Args:
            args: Dictionary of argument values
            
        Returns:
            Rendered prompt string
            
        Raises:
            PromptProgramError: If arguments are invalid or missing
        """
        # Validate and apply defaults
        validated_args = self._validate_and_apply_defaults(args)
        
        # Render template
        template = self.definition.template
        
        try:
            # Simple placeholder substitution: {arg_name}
            rendered = template
            for arg_name, arg_value in validated_args.items():
                placeholder = f"{{{arg_name}}}"
                # Convert value to string
                str_value = self._value_to_string(arg_value)
                rendered = rendered.replace(placeholder, str_value)
            
            return rendered
        except Exception as e:
            raise PromptProgramError(
                f"Failed to render prompt '{self.definition.name}': {e}"
            ) from e
    
    def _validate_and_apply_defaults(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate provided arguments and apply defaults.
        
        Args:
            args: User-provided arguments
            
        Returns:
            Complete argument dictionary with defaults applied
            
        Raises:
            PromptProgramError: If required arguments are missing or types are invalid
        """
        if not self.definition.args:
            # No args defined - accept empty or reject non-empty
            if args:
                raise PromptProgramError(
                    f"Prompt '{self.definition.name}' does not accept arguments, but received: {list(args.keys())}"
                )
            return {}
        
        validated: Dict[str, Any] = {}
        
        # Check for unknown arguments
        arg_names = {arg.name for arg in self.definition.args}
        unknown = set(args.keys()) - arg_names
        if unknown:
            raise PromptProgramError(
                f"Prompt '{self.definition.name}' received unknown arguments: {', '.join(sorted(unknown))}"
            )
        
        # Validate and collect all arguments
        for arg_def in self.definition.args:
            arg_name = arg_def.name
            
            if arg_name in args:
                # User provided value - validate type
                value = args[arg_name]
                validated[arg_name] = self._coerce_argument(arg_name, value, arg_def.arg_type)
            elif arg_def.default is not None:
                # Use default value
                validated[arg_name] = arg_def.default
            elif arg_def.required:
                # Required but not provided
                raise PromptProgramError(
                    f"Prompt '{self.definition.name}' missing required argument: {arg_name}"
                )
        
        return validated
    
    def _coerce_argument(self, name: str, value: Any, expected_type: str) -> Any:
        """
        Coerce and validate an argument value to the expected type.
        
        Args:
            name: Argument name
            value: Provided value
            expected_type: Expected type string (string, int, float, bool, list, object)
            
        Returns:
            Coerced value
            
        Raises:
            PromptProgramError: If coercion fails
        """
        try:
            if expected_type == 'string':
                return str(value)
            elif expected_type == 'int':
                if isinstance(value, bool):
                    raise ValueError("Cannot coerce bool to int")
                return int(value)
            elif expected_type == 'float':
                if isinstance(value, bool):
                    raise ValueError("Cannot coerce bool to float")
                return float(value)
            elif expected_type == 'bool':
                if isinstance(value, bool):
                    return value
                # Accept common boolean representations
                if isinstance(value, str):
                    lower = value.lower()
                    if lower in {'true', '1', 'yes', 'y'}:
                        return True
                    elif lower in {'false', '0', 'no', 'n'}:
                        return False
                raise ValueError(f"Cannot coerce '{value}' to bool")
            elif expected_type == 'list':
                if isinstance(value, list):
                    return value
                raise ValueError(f"Expected list, got {type(value).__name__}")
            elif expected_type == 'object':
                if isinstance(value, dict):
                    return value
                raise ValueError(f"Expected object/dict, got {type(value).__name__}")
            else:
                # Unknown type - pass through
                return value
        except (ValueError, TypeError) as e:
            raise PromptProgramError(
                f"Prompt '{self.definition.name}' argument '{name}': cannot coerce value to type '{expected_type}': {e}"
            ) from e
    
    def _value_to_string(self, value: Any) -> str:
        """Convert an argument value to a string for template substitution."""
        if isinstance(value, str):
            return value
        elif isinstance(value, bool):
            return 'true' if value else 'false'
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            # Format lists as comma-separated
            return ', '.join(str(item) for item in value)
        elif isinstance(value, dict):
            # Format dicts as key:value pairs
            pairs = [f"{k}:{v}" for k, v in value.items()]
            return ', '.join(pairs)
        else:
            return str(value)
    
    def get_output_schema(self) -> Optional[Dict[str, Any]]:
        """
        Get the JSON Schema representation of the output schema.
        
        Returns:
            JSON Schema dict if output_schema is defined, None otherwise
        """
        if not self.definition.output_schema:
            return None
        
        return self.definition.output_schema.to_json_schema()
    
    def get_structured_output_format(self) -> Optional[str]:
        """
        Get a human-readable description of the expected output format.
        
        Useful for providers that don't support JSON Schema but need
        format instructions in the prompt.
        
        Returns:
            Format description string or None
        """
        if not self.definition.output_schema:
            return None
        
        lines = ["Expected output format (JSON):"]
        lines.append("{")
        
        for field in self.definition.output_schema.fields:
            field_desc = self._format_output_field(field)
            lines.append(f"  {field_desc}")
        
        lines.append("}")
        return "\n".join(lines)
    
    def _format_output_field(self, field: OutputField) -> str:
        """Format an output field for human-readable description."""
        type_str = self._format_output_type(field.field_type)
        required_marker = "" if field.required else " (optional)"
        desc = f" // {field.description}" if field.description else ""
        return f'"{field.name}": {type_str}{required_marker}{desc}'
    
    def _format_output_type(self, field_type: OutputFieldType) -> str:
        """Format an output field type for display."""
        if field_type.base_type == 'enum' and field_type.enum_values:
            vals = ' | '.join(f'"{v}"' for v in field_type.enum_values)
            return vals
        elif field_type.base_type == 'list' and field_type.element_type:
            elem_type = self._format_output_type(field_type.element_type)
            return f"[{elem_type}]"
        else:
            return field_type.base_type
    
    def has_structured_output(self) -> bool:
        """Check if this prompt has a structured output schema."""
        return self.definition.output_schema is not None
    
    def get_arg_names(self) -> List[str]:
        """Get list of argument names."""
        return [arg.name for arg in self.definition.args]
    
    def get_required_arg_names(self) -> List[str]:
        """Get list of required argument names."""
        return [arg.name for arg in self.definition.args if arg.required]


def create_prompt_program(prompt_definition: Prompt) -> PromptProgram:
    """
    Factory function to create a PromptProgram from a Prompt AST node.
    
    Args:
        prompt_definition: Prompt AST node
        
    Returns:
        PromptProgram instance
    """
    return PromptProgram(definition=prompt_definition)


__all__ = [
    "PromptProgram",
    "PromptProgramError",
    "create_prompt_program",
]
