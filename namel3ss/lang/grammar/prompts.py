"""Legacy prompt parsing with args and output schema."""

from __future__ import annotations
import re
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Tuple

if TYPE_CHECKING:
    from .helpers import _Line

from namel3ss.ast.ai import PromptArgument, Prompt, OutputSchema, OutputField, OutputFieldType

# Regex pattern for parsing prompt declarations  
_PROMPT_HEADER_RE = re.compile(r'^prompt\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:{]')
_PROMPT_HEADER_QUOTED_RE = re.compile(r'^prompt\s+"([^"]+)"\s*[:{]')


class PromptsParserMixin:
    """Mixin providing legacy prompt parsing with args and output schema."""

    def _parse_prompt_legacy(self, line: _Line) -> None:
        """
        Parse a prompt definition block with typed arguments and output schema.
        
        Grammar:
            prompt <name>:
                args:
                    <arg_name>: <type> [= <default>]
                    ...
                output_schema:
                    <field_name>: <type>
                    ...
                template: <string>
                model: <llm_name>
        """
        stripped = line.text.strip()
        
        # Match prompt header (accepts both : and { for backward compatibility)
        match = _PROMPT_HEADER_RE.match(stripped) or _PROMPT_HEADER_QUOTED_RE.match(stripped)
        if not match:
            error_msg = (
                "Invalid prompt declaration syntax.\n"
                "Expected: prompt <name>: or prompt \"name\":\n"
                f"Found: {stripped}"
            )
            raise self._error(error_msg, line)
        
        name = match.group(1)
        base_indent = self._indent(line.text)
        self._advance()
        
        # Initialize fields
        args: List[PromptArgument] = []
        output_schema: Optional[OutputSchema] = None
        template: Optional[str] = None
        model: Optional[str] = None
        config: Dict[str, Any] = {}
        
        # Parse block content
        while True:
            nxt = self._peek_line()
            if nxt is None:
                break
            indent = self._indent(nxt.text)
            stripped = nxt.text.strip()
            
            # Stop if we've dedented
            if stripped and indent <= base_indent:
                break
            
            # Skip empty lines and comments
            if self._should_skip_comment(stripped, nxt.number, nxt.text):
                self._advance()
                continue
            
            lowered = stripped.lower()
            
            # Handle args: block
            if lowered.startswith('args:'):
                self._advance()
                args = self._parse_prompt_args_block(indent)
            # Handle output_schema: block
            elif lowered.startswith('output_schema:'):
                self._advance()
                output_schema = self._parse_output_schema_block(indent)
            # Handle template: field
            elif lowered.startswith('template:'):
                self._advance()
                template = self._parse_prompt_template(indent, stripped)
            # Handle model: field
            elif lowered.startswith('model:'):
                model = stripped.split(':', 1)[1].strip()
                self._advance()
            # Handle other config fields
            else:
                if ':' in stripped:
                    key, val = stripped.split(':', 1)
                    config[key.strip()] = val.strip()
                self._advance()
        
        if not template:
            raise self._error("prompt block requires 'template' field", line)
        
        prompt = Prompt(
            name=name,
            model=model or '',
            template=template,
            args=args,
            output_schema=output_schema,
            parameters=config,
        )
        
        self._ensure_app(line)
        self._app.prompts.append(prompt)

    def _parse_prompt_args_block(self, parent_indent: int) -> List[PromptArgument]:
        """Parse the args: block for a structured prompt."""
        args: List[PromptArgument] = []
        
        while True:
            nxt = self._peek_line()
            if nxt is None:
                break
            
            indent = self._indent(nxt.text)
            stripped = nxt.text.strip()
            
            # Stop if dedented
            if stripped and indent <= parent_indent:
                break
            
            # Skip empty lines and comments
            if self._should_skip_comment(stripped, nxt.number, nxt.text):
                self._advance()
                continue
            
            # Parse arg line: name: type [= default]
            if ':' in stripped:
                parts = stripped.split(':', 1)
                arg_name = parts[0].strip()
                arg_spec = parts[1].strip()
                
                # Check for default value
                arg_type = 'string'
                default = None
                required = True
                
                if '=' in arg_spec:
                    type_part, default_part = arg_spec.split('=', 1)
                    arg_type = type_part.strip()
                    default = default_part.strip()
                    # Remove quotes from default if present
                    if default.startswith('"') and default.endswith('"'):
                        default = default[1:-1]
                    elif default.startswith("'") and default.endswith("'"):
                        default = default[1:-1]
                    required = False
                else:
                    arg_type = arg_spec
                
                args.append(PromptArgument(
                    name=arg_name,
                    arg_type=arg_type,
                    required=required,
                    default=default,
                ))
            
            self._advance()
        
        return args

    def _parse_output_schema_block(self, parent_indent: int) -> OutputSchema:
        """Parse the output_schema: block for a structured prompt."""
        fields: List[OutputField] = []
        
        while True:
            nxt = self._peek_line()
            if nxt is None:
                break
            
            indent = self._indent(nxt.text)
            stripped = nxt.text.strip()
            
            # Stop if dedented
            if stripped and indent <= parent_indent:
                break
            
            # Skip empty lines and comments
            if self._should_skip_comment(stripped, nxt.number, nxt.text):
                self._advance()
                continue
            
            # Parse field line: name: type
            if ':' in stripped:
                parts = stripped.split(':', 1)
                field_name = parts[0].strip()
                type_spec = parts[1].strip().rstrip(',')
                
                # Check for optional marker (?)
                required = True
                if field_name.endswith('?'):
                    field_name = field_name[:-1]
                    required = False
                
                # Check if this is a nested object or list of objects
                if type_spec == '{':
                    # Nested object definition follows
                    self._advance()
                    nested_fields = self._parse_nested_object_fields_grammar(indent)
                    field_type = OutputFieldType(base_type='object', nested_fields=nested_fields)
                elif type_spec.startswith('list[{'):
                    # List of objects
                    self._advance()
                    nested_fields = self._parse_nested_object_fields_grammar(indent)
                    element_type = OutputFieldType(base_type='object', nested_fields=nested_fields)
                    field_type = OutputFieldType(base_type='list', element_type=element_type)
                else:
                    # Parse the type specification normally
                    field_type = self._parse_output_field_type(type_spec)
                    self._advance()
                
                fields.append(OutputField(
                    name=field_name,
                    field_type=field_type,
                    required=required,
                ))
                continue
            
            self._advance()
        
        return OutputSchema(fields=fields)
    
    def _parse_nested_object_fields_grammar(self, parent_indent: int) -> List[OutputField]:
        """Parse nested object fields for grammar-based parsing."""
        nested_fields: List[OutputField] = []
        
        while True:
            nxt = self._peek_line()
            if nxt is None:
                break
            
            indent = self._indent(nxt.text)
            stripped = nxt.text.strip()
            
            # Skip empty lines and comments
            if self._should_skip_comment(stripped, nxt.number, nxt.text):
                self._advance()
                continue
            
            # Check for closing brace
            if stripped == '}' or stripped.startswith('}'):
                self._advance()
                break
            
            # Stop if dedented
            if indent <= parent_indent:
                break
            
            # Parse field
            if ':' in stripped:
                parts = stripped.split(':', 1)
                field_name = parts[0].strip().rstrip(',')
                type_spec = parts[1].strip().rstrip(',')
                
                required = True
                if field_name.endswith('?'):
                    field_name = field_name[:-1]
                    required = False
                
                # Check for nested object
                if type_spec == '{':
                    self._advance()
                    sub_nested_fields = self._parse_nested_object_fields_grammar(indent)
                    field_type = OutputFieldType(base_type='object', nested_fields=sub_nested_fields)
                elif type_spec.startswith('list[{'):
                    self._advance()
                    sub_nested_fields = self._parse_nested_object_fields_grammar(indent)
                    element_type = OutputFieldType(base_type='object', nested_fields=sub_nested_fields)
                    field_type = OutputFieldType(base_type='list', element_type=element_type)
                else:
                    field_type = self._parse_output_field_type(type_spec)
                    self._advance()
                
                nested_fields.append(OutputField(
                    name=field_name,
                    field_type=field_type,
                    required=required,
                ))
                continue
            
            self._advance()
        
        if not nested_fields:
            raise self._error("Nested object must have at least one field", 0, "")
        
        return nested_fields

    def _parse_output_field_type(self, type_spec: str, line_no: int = None, line: str = None) -> OutputFieldType:
        """
        Parse a type specification like 'string', 'enum(\"a\", \"b\")', 'list[string]', 'object {...}'.
        
        Args compatible with both Grammar (1 arg) and AIParserMixin (3 args) call patterns.
        """
        type_spec = type_spec.strip()
        
        # Special case: list[{ without closing ] should not be parsed here
        # This is handled separately in _parse_output_schema_block and _parse_nested_object_fields_grammar
        if type_spec == 'list[{' or type_spec.startswith('list[{') and not type_spec.endswith(']'):
            raise self._error(f"Incomplete list of objects syntax: {type_spec}. Should be handled at higher level.", line_no or 0, line or "")
        
        # Handle enum: enum(\"val1\", \"val2\", ...)
        if type_spec.startswith('enum(') or type_spec.startswith('enum['):
            # Extract enum values
            start_char = '(' if '(' in type_spec else '['
            end_char = ')' if start_char == '(' else ']'
            start_idx = type_spec.index(start_char) + 1
            end_idx = type_spec.rindex(end_char)
            values_str = type_spec[start_idx:end_idx]
            
            # Split and clean values
            enum_values = []
            for val in values_str.split(','):
                val = val.strip()
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                elif val.startswith("'") and val.endswith("'"):
                    val = val[1:-1]
                if val:
                    enum_values.append(val)
            
            return OutputFieldType(base_type='enum', enum_values=enum_values)
        
        # Handle list: list[element_type]
        if type_spec.startswith('list[') or type_spec.startswith('list<'):
            start_idx = type_spec.index('[') if '[' in type_spec else type_spec.index('<')
            end_idx = type_spec.rindex(']') if '[' in type_spec else type_spec.rindex('>')
            element_type_str = type_spec[start_idx+1:end_idx].strip()
            element_type = self._parse_output_field_type(element_type_str)
            return OutputFieldType(base_type='list', element_type=element_type)
        
        # Handle object: object { ... } (simplified - just mark as object for now)
        if type_spec.startswith('object'):
            # For now, we'll create a simple object type
            # Full nested object parsing would require more complex logic
            return OutputFieldType(base_type='object')
        
        # Handle primitives: string, int, float, bool
        if type_spec in ('string', 'int', 'float', 'bool'):
            return OutputFieldType(base_type=type_spec)
        
        # Default to string
        return OutputFieldType(base_type='string')

    def _parse_prompt_template(self, parent_indent: int, first_line: str) -> str:
        """Parse template field - can be inline or multiline block."""
        # Check if there's inline content after template:
        if 'template:' in first_line.lower():
            inline = first_line.split(':', 1)[1].strip()
            if inline:
                # Remove quotes if present
                if inline.startswith('"""') or inline.startswith("'''"):
                    # Multiline string starts on same line - collect until end marker
                    quote = inline[:3]
                    content = inline[3:]
                    lines = [content]
                    
                    while True:
                        nxt = self._peek_line()
                        if nxt is None:
                            break
                        line_text = nxt.text.rstrip()
                        self._advance()
                        
                        if quote in line_text:
                            # Found end quote
                            end_idx = line_text.index(quote)
                            lines.append(line_text[:end_idx])
                            break
                        else:
                            lines.append(line_text)
                    
                    return '\n'.join(lines)
                elif inline.startswith('"') and inline.endswith('"'):
                    return inline[1:-1]
                elif inline.startswith("'") and inline.endswith("'"):
                    return inline[1:-1]
                else:
                    return inline
        
        # Otherwise, expect a multiline block
        lines = []
        while True:
            nxt = self._peek_line()
            if nxt is None:
                break
            
            indent = self._indent(nxt.text)
            stripped = nxt.text.strip()
            
            # Stop if dedented
            if stripped and indent <= parent_indent:
                break
            
            # Add line content (preserving indentation within the block)
            if stripped:
                lines.append(nxt.text[parent_indent+2:] if len(nxt.text) > parent_indent+2 else stripped)
            else:
                lines.append('')
            
            self._advance()
        
        return '\n'.join(lines)


__all__ = ['PromptsParserMixin']
