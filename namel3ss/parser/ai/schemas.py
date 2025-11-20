"""Schema parsing for structured prompts and outputs."""

from __future__ import annotations

import ast as py_ast
import textwrap
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from namel3ss.ast import (
    EnumType,
    OutputField,
    OutputFieldType,
    OutputSchema,
    PromptArgument,
    PromptField,
)

if TYPE_CHECKING:
    from ..base import ParserBase

# Import template engine for compile-time validation
try:
    from namel3ss.templates import get_default_engine, TemplateCompilationError
    TEMPLATE_VALIDATION_AVAILABLE = True
except ImportError:
    TEMPLATE_VALIDATION_AVAILABLE = False


class SchemaParserMixin:
    """Mixin for parsing prompt schemas, arguments, and output specifications."""
    
    def _parse_prompt_schema_block(self: 'ParserBase', parent_indent: int) -> List[PromptField]:
        """
        Parse schema block defining input or output fields for prompts.
        
        Schema blocks define typed fields with descriptions, constraints,
        and metadata for validation and documentation.
        
        Syntax:
            input:
                text: string required
                    description: "Input text to analyze"
                max_length: int
                    default: 100
                    description: "Maximum output length"
            
            output:
                category: string required
                    enum: ["billing", "technical", "account"]
                confidence: float
                    description: "Confidence score 0-1"
        """
        fields: List[PromptField] = []
        while self.pos < len(self.lines):
            line = self._peek()
            if line is None:
                break
            indent = self._indent(line)
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            name_part, _, remainder = stripped.partition(':')
            field_name = self._strip_quotes(name_part.strip())
            field_type_text = remainder.strip() or "text"
            self._advance()
            config: Dict[str, Any] = {}
            nxt = self._peek()
            if nxt is not None and self._indent(nxt) > indent:
                config = self._parse_kv_block(indent)
            dtype_override = config.pop('type', config.pop('dtype', None))
            if dtype_override is not None:
                field_type_text = str(dtype_override)
            required_value = config.pop('required', None)
            optional_value = config.pop('optional', None)
            nullable_value = config.pop('nullable', None)
            required = True
            if required_value is not None:
                required = self._to_bool(required_value, True)
            elif optional_value is not None:
                required = not self._to_bool(optional_value, False)
            if nullable_value is not None and self._to_bool(nullable_value, False):
                required = False
            default_value = config.pop('default', None)
            description_raw = config.pop('description', config.pop('desc', None))
            description = str(description_raw) if description_raw is not None else None
            enum_override = config.pop('enum', None)
            field_type, enum_values = self._parse_prompt_field_type(field_type_text)
            if enum_override is not None:
                if isinstance(enum_override, list):
                    enum_values = [self._stringify_value(item) for item in enum_override if item is not None]
                else:
                    enum_values = [self._stringify_value(enum_override)]
            metadata_raw = config.pop('metadata', {})
            metadata = self._coerce_options_dict(metadata_raw)
            if config:
                metadata.update(config)
            fields.append(
                PromptField(
                    name=field_name,
                    field_type=field_type,
                    required=required,
                    description=description,
                    default=default_value,
                    enum=enum_values,
                    metadata=metadata,
                )
            )
        return fields

    def _parse_prompt_args(self: 'ParserBase', parent_indent: int) -> List[PromptArgument]:
        """
        Parse args block for structured prompts.
        
        Syntax:
            args: {
                text: string,
                max_length: int = 100,
                style: string = "concise"
            }
        """
        args: List[PromptArgument] = []
        
        while self.pos < len(self.lines):
            line = self._peek()
            if line is None:
                break
            indent = self._indent(line)
            stripped = line.strip()
            
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            
            # Parse: name: type [= default]
            if '=' in stripped:
                name_type_part, _, default_part = stripped.partition('=')
                has_default = True
            else:
                name_type_part = stripped
                default_part = None
                has_default = False
            
            # Remove trailing comma
            name_type_part = name_type_part.rstrip(',').strip()
            
            # Split name: type
            if ':' not in name_type_part:
                raise self._error(f"Expected 'name: type' in args block", self.pos + 1, line)
            
            arg_name, _, type_str = name_type_part.partition(':')
            arg_name = arg_name.strip()
            type_str = type_str.strip()
            
            # Parse default value if present
            default_value = None
            if has_default and default_part:
                default_str = default_part.rstrip(',').strip()
                default_value = self._coerce_scalar(default_str)
            
            # Normalize type names
            arg_type = self._normalize_arg_type(type_str)
            
            args.append(PromptArgument(
                name=arg_name,
                arg_type=arg_type,
                required=not has_default,
                default=default_value,
            ))
            
            self._advance()
        
        return args
    
    def _normalize_arg_type(self: 'ParserBase', type_str: str) -> str:
        """Normalize argument type strings to canonical forms."""
        type_lower = type_str.lower().strip()
        
        # Map common variations
        type_map = {
            'str': 'string',
            'text': 'string',
            'int': 'int',
            'integer': 'int',
            'number': 'float',
            'float': 'float',
            'bool': 'bool',
            'boolean': 'bool',
            'array': 'list',
            'dict': 'object',
            'map': 'object',
        }
        
        # Handle list[T] syntax
        if type_lower.startswith('list['):
            return type_str  # Keep as-is for now
        
        return type_map.get(type_lower, type_str)
    
    def _parse_output_schema(self: 'ParserBase', parent_indent: int) -> OutputSchema:
        """
        Parse output_schema block for structured prompts.
        
        Syntax:
            output_schema: {
                category: enum["billing", "technical", "account"],
                urgency: enum["low", "medium", "high"],
                needs_handoff: bool,
                confidence: float,
                tags: list[string],
                user: {
                    name: string,
                    email: string,
                    roles: list[string]
                }
            }
        """
        fields: List[OutputField] = []
        
        while self.pos < len(self.lines):
            line = self._peek()
            if line is None:
                break
            indent = self._indent(line)
            stripped = line.strip()
            
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            
            # Parse: field_name: type
            if ':' not in stripped:
                raise self._error(
                    f"Expected 'field_name: type' in output_schema",
                    self.pos + 1,
                    line,
                    hint='Each field must have format: field_name: type (e.g., status: string)'
                )
            
            field_name, _, type_part = stripped.partition(':')
            field_name = field_name.strip().rstrip(',')
            type_part = type_part.strip().rstrip(',')
            
            # Check if this is the start of a nested object
            if type_part == '{':
                # The nested object fields follow on subsequent lines
                self._advance()
                field_type = OutputFieldType(
                    base_type="object",
                    nested_fields=self._parse_nested_object_fields(indent)
                )
            elif type_part.startswith('list[{'):
                # list of objects: list[{ ... }]
                # The object fields follow on subsequent lines
                self._advance()
                element_type = OutputFieldType(
                    base_type="object",
                    nested_fields=self._parse_nested_object_fields(indent)
                )
                field_type = OutputFieldType(
                    base_type="list",
                    element_type=element_type
                )
            else:
                # Parse the field type normally
                field_type = self._parse_output_field_type(type_part, self.pos + 1, line)
                self._advance()
            
            fields.append(OutputField(
                name=field_name,
                field_type=field_type,
                required=True,  # Default to required
            ))
        
        if not fields:
            raise self._error(
                "output_schema cannot be empty",
                self.pos,
                "",
                hint='Define at least one output field, e.g., result: string'
            )
        
        # Consume the closing brace if present
        if self.pos < len(self.lines):
            line = self._peek()
            if line and line.strip() in ['}', '},']:
                self._advance()
        
        return OutputSchema(fields=fields)
    
    def _parse_output_field_type(self: 'ParserBase', type_str: str, line_no: int, line: str, parent_indent: Optional[int] = None) -> OutputFieldType:
        """
        Parse a field type specification into OutputFieldType.
        
        Supports:
        - Primitives: string, int, float, bool
        - Enums: enum["val1", "val2", "val3"]
        - Lists: list[string], list[int], list[object]
        - Nested objects: { field: type, ... }
        """
        type_str = type_str.strip()
        
        # Handle nested object type: { ... }
        if type_str == '{' or (type_str.startswith('{') and len(type_str) == 1):
            # This is an inline nested object, parse its fields from subsequent lines
            if parent_indent is None:
                raise self._error("Cannot parse nested object without parent indent context", line_no, line)
            nested_fields = self._parse_nested_object_fields(parent_indent)
            return OutputFieldType(
                base_type="object",
                nested_fields=nested_fields
            )
        
        # Handle enum["val1", "val2"]
        if type_str.startswith('enum['):
            if not type_str.endswith(']'):
                raise self._error(
                    "Malformed enum type, expected closing ]",
                    line_no,
                    line,
                    hint='Enum syntax: enum["value1", "value2"]'
                )
            
            inner = type_str[5:-1].strip()
            enum_values = self._parse_enum_values(inner, line_no, line)
            
            return OutputFieldType(
                base_type="enum",
                enum_values=enum_values
            )
        
        # Handle list[T]
        if type_str.startswith('list['):
            if not type_str.endswith(']'):
                raise self._error(
                    "Malformed list type, expected closing ]",
                    line_no,
                    line,
                    hint='List syntax: list[string] or list[int]'
                )
            
            inner_type_str = type_str[5:-1].strip()
            element_type = self._parse_output_field_type(inner_type_str, line_no, line, parent_indent)
            
            return OutputFieldType(
                base_type="list",
                element_type=element_type
            )
        
        # Handle optional types (trailing ?)
        nullable = False
        if type_str.endswith('?'):
            nullable = True
            type_str = type_str[:-1].strip()
        
        # Normalize primitive types
        type_lower = type_str.lower()
        type_map = {
            'str': 'string',
            'text': 'string',
            'string': 'string',
            'int': 'int',
            'integer': 'int',
            'number': 'float',
            'float': 'float',
            'bool': 'bool',
            'boolean': 'bool',
        }
        
        base_type = type_map.get(type_lower)
        if not base_type:
            raise self._error(
                f"Unknown output field type: {type_str}",
                line_no,
                line,
                hint='Supported types: string, int, float, bool, enum[...], list[...], or nested object'
            )
        
        return OutputFieldType(
            base_type=base_type,
            nullable=nullable
        )
    
    def _parse_nested_object_fields(self: 'ParserBase', parent_indent: int) -> List[OutputField]:
        """
        Parse nested object fields for structured output schemas.
        
        Recursively parses object structures within output schemas,
        supporting deeply nested objects and lists of objects.
        
        Syntax:
            user: {
                name: string,
                email: string,
                roles: list[string],
                profile: {
                    age: int,
                    location: string
                }
            }
        """
        nested_fields: List[OutputField] = []
        
        while self.pos < len(self.lines):
            line = self._peek()
            if line is None:
                break
            
            indent = self._indent(line)
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            
            # Check for closing brace
            if stripped == '}' or stripped.startswith('}'):
                self._advance()
                break
            
            # Stop if we've dedented back to or before parent level
            if indent <= parent_indent:
                break
            
            # Parse: field_name: type
            if ':' not in stripped:
                raise self._error(
                    f"Expected 'field_name: type' in nested object",
                    self.pos + 1,
                    line,
                    hint='Each field must have format: field_name: type'
                )
            
            field_name, _, type_part = stripped.partition(':')
            field_name = field_name.strip().rstrip(',')
            type_part = type_part.strip().rstrip(',')
            
            # Check if this is the start of a nested object
            if type_part == '{':
                # The nested object fields follow on subsequent lines
                self._advance()
                field_type = OutputFieldType(
                    base_type="object",
                    nested_fields=self._parse_nested_object_fields(indent)
                )
            elif type_part.startswith('list[{'):
                # list of objects: list[{ ... }]
                # The object fields follow on subsequent lines
                self._advance()
                element_type = OutputFieldType(
                    base_type="object",
                    nested_fields=self._parse_nested_object_fields(indent)
                )
                field_type = OutputFieldType(
                    base_type="list",
                    element_type=element_type
                )
            else:
                # Parse the field type normally
                field_type = self._parse_output_field_type(type_part, self.pos + 1, line)
                self._advance()
            
            nested_fields.append(OutputField(
                name=field_name,
                field_type=field_type,
                required=True,
            ))
        
        if not nested_fields:
            raise self._error(
                "Nested object must have at least one field",
                self.pos,
                "",
                hint='Define at least one field in the nested object'
            )
        
        return nested_fields
    
    def _parse_enum_values(self: 'ParserBase', inner: str, line_no: int, line: str) -> List[str]:
        """
        Parse enumeration values from type specification.
        
        Extracts and validates string values from enum declarations,
        ensuring all values are properly quoted strings.
        
        Syntax:
            enum["value1", "value2", "value3"]
        """
        if not inner:
            raise self._error(
                "Enum must have at least one value",
                line_no,
                line,
                hint='Add enum values, e.g., enum["option1", "option2"]'
            )
        
        # Try to use Python's ast.literal_eval for safety
        expr = f"[{inner}]"
        try:
            parsed = py_ast.literal_eval(expr)
            if isinstance(parsed, (list, tuple)):
                values = []
                for item in parsed:
                    if not isinstance(item, str):
                        raise self._error(f"Enum values must be strings, got: {type(item).__name__}", line_no, line)
                    values.append(item)
                if not values:
                    raise self._error("Enum must have at least one value", line_no, line)
                return values
        except (ValueError, SyntaxError) as e:
            raise self._error(f"Invalid enum syntax: {e}", line_no, line)
        
        raise self._error("Failed to parse enum values", line_no, line)

    def _parse_prompt_template_block(self: 'ParserBase', parent_indent: int) -> str:
        """
        Parse multi-line template block with compile-time validation.
        
        Validates template syntax using the template engine during compilation
        to catch errors early with proper file/line/column information.
        """
        start_line = self.pos + 1  # Track line number for error reporting
        lines: List[str] = []
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            if indent <= parent_indent:
                break
            lines.append(nxt[parent_indent:])
            self._advance()
        raw = "\n".join(lines).rstrip("\n")
        text = textwrap.dedent(raw).strip("\n")
        stripped = text.strip()
        if stripped.startswith('"""') and stripped.endswith('"""') and len(stripped) >= 6:
            text = stripped[3:-3].strip("\n")
        elif stripped.startswith("'''") and stripped.endswith("'''") and len(stripped) >= 6:
            text = stripped[3:-3].strip("\n")
        if not text:
            raise self._error("Prompt template cannot be empty", self.pos, "")
        
        # Compile-time template validation
        if TEMPLATE_VALIDATION_AVAILABLE:
            try:
                engine = get_default_engine()
                # Compile the template to catch syntax errors
                # Use validation=True to also check for security issues
                engine.compile(text, name=f"<template at line {start_line}>", validate=True)
            except TemplateCompilationError as e:
                # Re-raise as parser error with proper context
                error_line = start_line + (e.line_number or 1) - 1
                raise self._error(
                    f"Template compilation error: {str(e)}",
                    error_line,
                    self.lines[error_line - 1] if 0 < error_line <= len(self.lines) else ""
                )
        
        return text

    def _parse_prompt_field_type(self: 'ParserBase', raw: Optional[str]) -> Tuple[str, List[str]]:
        """
        Parse prompt field type specification into type and enum values.
        
        Normalizes type names and extracts enumeration values for
        constrained string fields.
        
        Supported Types:
            text/string: Text field
            int/integer: Integer field
            float/number: Floating point field
            bool/boolean: Boolean field
            json/object: JSON object
            list/array: Array field
            one_of(...): Enum with specific values
        
        Returns:
            Tuple of (normalized_type, enum_values_list)
        """
        if not raw:
            return "text", []
        text = str(raw).strip()
        lowered = text.lower()
        if lowered.startswith("one_of"):
            start = text.find("(")
            end = text.rfind(")")
            if start != -1 and end != -1 and end > start:
                inner = text[start + 1 : end]
                return "enum", self._parse_prompt_enum_values(inner)
            return "enum", []
        if lowered in {"string", "text"}:
            return "text", []
        if lowered in {"int", "integer"}:
            return "int", []
        if lowered in {"float", "number"}:
            return "number", []
        if lowered in {"bool", "boolean"}:
            return "boolean", []
        if lowered in {"json", "object"}:
            return "json", []
        if lowered in {"list", "array"}:
            return "list", []
        return text, []

    def _parse_prompt_enum_values(self: 'ParserBase', inner: str) -> List[str]:
        """
        Parse enumeration values from one_of() specification.
        
        Safely extracts string values using Python AST literal_eval,
        falling back to comma-separated parsing if needed.
        
        Example:
            one_of("low", "medium", "high") -> ["low", "medium", "high"]
        """
        expr = f"[{inner}]"
        try:
            parsed = py_ast.literal_eval(expr)
            if isinstance(parsed, (list, tuple)):
                return [self._stringify_value(item) for item in parsed if item is not None]
        except Exception:
            pass
        tokens = [token.strip() for token in inner.split(',') if token.strip()]
        return [self._strip_quotes(token) for token in tokens]
