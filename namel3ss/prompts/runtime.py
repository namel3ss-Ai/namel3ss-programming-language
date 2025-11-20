"""
Runtime support for structured prompts with typed args and output schemas.

This module provides the PromptProgram abstraction for executing prompts
with argument validation, template rendering, and structured output handling.
Supports memory integration for stateful LLM interactions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

from namel3ss.ast import Prompt, PromptArgument, OutputSchema, OutputField, OutputFieldType
from namel3ss.errors import N3Error
from namel3ss.templates import get_default_engine, TemplateError

if TYPE_CHECKING:
    from namel3ss.codegen.backend.core.runtime.memory import MemoryRegistry


class PromptProgramError(N3Error):
    """Errors related to prompt program execution."""
    
    def __init__(self, message: str, *, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


@dataclass
class PromptProgram:
    """
    Runtime abstraction for a structured prompt.
    
    Handles:
    - Argument validation and coercion
    - Template rendering with placeholders
    - Output schema generation for LLM providers
    - Integration with validation layer
    - Memory access for stateful interactions
    
    Example:
        program = PromptProgram(prompt_definition)
        rendered = program.render_prompt({"text": "Hello", "max_words": 50})
        schema = program.get_output_schema()
    """
    
    definition: Prompt
    memory_registry: Optional['MemoryRegistry'] = None
    scope_context: Optional[Dict[str, str]] = None
    
    async def render_prompt(self, args: Dict[str, Any]) -> str:
        """
        Render the prompt template with validated arguments and memory context.
        
        Uses the unified template engine for secure, production-grade rendering.
        Supports Jinja2 syntax with variables, conditionals, loops, and filters.
        
        Args:
            args: Dictionary of argument values
            
        Returns:
            Rendered prompt string with memory references resolved
            
        Raises:
            PromptProgramError: If arguments are invalid or missing
        """
        # Validate and apply defaults
        validated_args = self._validate_and_apply_defaults(args)
        
        # Resolve memory references in template
        template = self.definition.template
        
        # Find and resolve memory placeholders: {memory.name}
        if self.memory_registry and '{memory.' in template:
            template = await self._resolve_memory_placeholders(template)
        
        try:
            # Use template engine for rendering
            engine = get_default_engine()
            compiled = engine.compile(
                template,
                name=f"prompt_{self.definition.name}",
                validate=True
            )
            
            # Convert arguments to template context
            context = self._args_to_context(validated_args)
            
            return compiled.render(context)
            
        except TemplateError as e:
            raise PromptProgramError(
                f"Failed to render prompt '{self.definition.name}': {e}",
                original_error=e
            ) from e
        except Exception as e:
            raise PromptProgramError(
                f"Failed to render prompt '{self.definition.name}': {e}"
            ) from e
    
    def _args_to_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert validated arguments to template context.
        
        Converts values to template-friendly formats while preserving types
        that Jinja2 can handle directly (str, int, float, bool, list, dict).
        
        Args:
            args: Validated argument values
            
        Returns:
            Template context dictionary
        """
        context = {}
        for key, value in args.items():
            # Keep Jinja2-compatible types as-is
            if isinstance(value, (str, int, float, bool, type(None), list, dict)):
                context[key] = value
            else:
                # Convert other types to strings
                context[key] = self._value_to_string(value)
        return context
    
    async def _resolve_memory_placeholders(self, template: str) -> str:
        """
        Resolve memory references in template: {memory.conversation_history}
        
        Args:
            template: Template string with memory placeholders
            
        Returns:
            Template with memory placeholders replaced by actual content
        """
        # Pattern: {memory.name} or {memory.name:limit}
        pattern = r'\{memory\.([a-zA-Z_][a-zA-Z0-9_]*)(?::(\d+))?\}'
        
        import re
        matches = list(re.finditer(pattern, template))
        
        if not matches:
            return template
        
        # Replace from end to start to preserve indices
        for match in reversed(matches):
            memory_name = match.group(1)
            limit_str = match.group(2)
            limit = int(limit_str) if limit_str else None
            
            try:
                handle = self.memory_registry.get(
                    memory_name,
                    scope_context=self.scope_context
                )
                content = await handle.read(limit=limit, reverse=False)
                replacement = self._memory_content_to_string(content)
            except Exception as e:
                # Log warning but don't fail the render
                import logging
                logging.warning(
                    f"Failed to resolve memory '{memory_name}' in prompt: {e}"
                )
                replacement = f"[memory:{memory_name}:unavailable]"
            
            template = template[:match.start()] + replacement + template[match.end():]
        
        return template
    
    def _memory_content_to_string(self, content: Any) -> str:
        """
        Convert memory content to string for template inclusion.
        
        Args:
            content: Memory content (list, dict, or scalar)
            
        Returns:
            Formatted string representation
        """
        if content is None:
            return ""
        
        if isinstance(content, list):
            # Format list items as numbered entries
            if not content:
                return ""
            
            lines = []
            for i, item in enumerate(content, 1):
                if isinstance(item, dict):
                    # Assume conversation format: {role, content}
                    role = item.get('role', 'unknown')
                    text = item.get('content', str(item))
                    lines.append(f"{i}. [{role}] {text}")
                else:
                    lines.append(f"{i}. {item}")
            
            return "\n".join(lines)
        
        elif isinstance(content, dict):
            # Format dict as key: value pairs
            lines = [f"{k}: {v}" for k, v in content.items()]
            return "\n".join(lines)
        
        else:
            return str(content)
    
    async def read_memory(self, name: str, *, limit: Optional[int] = None) -> Any:
        """
        Read from a memory store.
        
        Args:
            name: Memory name
            limit: Optional limit on items returned
            
        Returns:
            Memory contents
            
        Raises:
            PromptProgramError: If memory registry not available
        """
        if not self.memory_registry:
            raise PromptProgramError(
                f"Cannot read memory '{name}': memory registry not available"
            )
        
        try:
            handle = self.memory_registry.get(name, scope_context=self.scope_context)
            return await handle.read(limit=limit)
        except Exception as e:
            raise PromptProgramError(
                f"Failed to read memory '{name}': {e}"
            ) from e
    
    async def write_memory(self, name: str, value: Any) -> None:
        """
        Write to a memory store.
        
        Args:
            name: Memory name
            value: Value to write
            
        Raises:
            PromptProgramError: If memory registry not available
        """
        if not self.memory_registry:
            raise PromptProgramError(
                f"Cannot write memory '{name}': memory registry not available"
            )
        
        try:
            handle = self.memory_registry.get(name, scope_context=self.scope_context)
            await handle.write(value)
        except Exception as e:
            raise PromptProgramError(
                f"Failed to write memory '{name}': {e}"
            ) from e
    
    async def append_memory(self, name: str, item: Any) -> None:
        """
        Append item to a list-type memory store.
        
        Args:
            name: Memory name
            item: Item to append
            
        Raises:
            PromptProgramError: If memory registry not available
        """
        if not self.memory_registry:
            raise PromptProgramError(
                f"Cannot append to memory '{name}': memory registry not available"
            )
        
        try:
            handle = self.memory_registry.get(name, scope_context=self.scope_context)
            await handle.append(item)
        except Exception as e:
            raise PromptProgramError(
                f"Failed to append to memory '{name}': {e}"
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
