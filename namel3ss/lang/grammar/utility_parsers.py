"""Utility parsing methods for KV blocks, lists, schemas."""

from __future__ import annotations
import re
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Tuple

if TYPE_CHECKING:
    from .helpers import _Line

from namel3ss.ast.agents import GraphEdge


class UtilityParsersMixin:
    """Mixin providing utility parsing methods for kv blocks, lists, schemas."""

    def _parse_kv_block_braces(self, parent_indent: int) -> dict[str, any]:
        """Parse a key-value block enclosed in braces."""
        entries: dict[str, any] = {}
        depth = 1  # Start with one open brace
        
        while True:
            line = self._peek_line()
            if line is None:
                break
            stripped = line.text.strip()
            
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                self._advance()
                continue
            
            # Check for closing brace
            if stripped == '}':
                depth -= 1
                self._advance()
                if depth == 0:
                    break
                continue
            
            # Parse key-value pair
            match = re.match(r'^([A-Za-z0-9_\-\s]+):\s*(.*)$', stripped)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                
                # Check if value is a multi-line list
                if value.startswith('[') and not value.endswith(']'):
                    # Collect multi-line list (this will advance past all list lines)
                    entries[key] = self._collect_multiline_list()
                    continue  # Skip the normal advance at the end
                # Parse value (handle lists, strings, numbers)
                elif value.startswith('['):
                    # Single-line list
                    entries[key] = self._parse_list_value(value)
                elif value.startswith('"') and value.endswith('"'):
                    entries[key] = value[1:-1]
                elif value.replace('.', '').replace('-', '').isdigit():
                    entries[key] = float(value) if '.' in value else int(value)
                elif value == '':
                    # Empty value, might be multi-line, skip for now
                    pass
                else:
                    entries[key] = value
            
            self._advance()
        
        return entries

    def _collect_multiline_list(self) -> str:
        """Collect a multi-line list value into a single string."""
        lines = []
        bracket_depth = 1  # Already saw opening [
        
        self._advance()  # Move past the key: [ line
        
        while True:
            line = self._peek_line()
            if line is None:
                break
            
            stripped = line.text.strip()
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                self._advance()
                continue
            
            lines.append(stripped)
            
            # Track bracket depth
            bracket_depth += stripped.count('[')
            bracket_depth -= stripped.count(']')
            
            self._advance()
            
            if bracket_depth == 0:
                break
        
        return '[' + ' '.join(lines) + ']'

    def _parse_list_field(self, value) -> List[str]:
        """Parse a list field like [item1, item2, item3]."""
        # If already a list, return it
        if isinstance(value, list):
            return value
        if not isinstance(value, str):
            return []
        if not value.startswith('['):
            return []
        value = value.strip()[1:-1]  # Remove [ ]
        if not value:
            return []
        items = [item.strip().strip('"').strip("'") for item in value.split(',')]
        return [item for item in items if item]

    def _parse_list_value(self, value: str) -> List[any]:
        """Parse a list value, potentially spanning multiple lines."""
        # Simple implementation - in production would need full bracket matching
        return self._parse_list_field(value)

    def _parse_graph_edges(self, edges_raw: str) -> List[GraphEdge]:
        """Parse graph edges from list of edge dictionaries."""
        if not edges_raw or edges_raw == '[]':
            return []
        
        # Simple parser for edge dictionaries
        # In production, would use proper JSON/dict parsing
        edges = []
        
        # Extract individual edge blocks
        edge_pattern = re.findall(r'\{([^}]+)\}', edges_raw)
        for edge_str in edge_pattern:
            edge_dict = {}
            # Parse key-value pairs within edge
            pairs = re.findall(r'([A-Za-z_]+):\s*["\']?([^,"\']+)["\']?', edge_str)
            for key, value in pairs:
                edge_dict[key.strip()] = value.strip()
            
            if 'from' in edge_dict and 'to' in edge_dict:
                edges.append(GraphEdge(
                    from_agent=edge_dict['from'],
                    to_agent=edge_dict['to'],
                    condition=edge_dict.get('when', 'default'),
                ))
        
        return edges

    def _parse_schema_field(self, value):
        """Helper to parse schema fields (dict or string)."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            # Simple string-to-dict conversion for now
            # In production, this would use proper JSON/dict parsing
            return {'_raw': value}
        return {}

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _ensure_app(self, line: _Line) -> None:
        if self._app is None:
            fallback_name = self._module_name or self._module_name_override
            fallback = fallback_name.split('.')[-1] if fallback_name else 'app'
            self._app = App(name=fallback)

    def _peek_line(self) -> Optional[_Line]:
        """Return the current line as a _Line object for Grammar parsing."""
        if self._cursor < len(self._lines):
            return self._lines[self._cursor]
        return None

    @staticmethod
    def _indent(text: str) -> int:
        """
        Compute indent for either a raw string or a _Line wrapper.

        AIParserMixin sometimes passes _Line objects back into the grammar
        helpers, so we normalize here to avoid type errors.
        """
        if isinstance(text, _Line):
            text = text.text
        return len(text) - len(text.lstrip(' '))

    def _error(self, message: str, line_or_line_no=None, line_text: str = None) -> N3SyntaxError:
        """
        Create a syntax error. Supports two call patterns:
        1. Grammar style: _error(message, line: _Line)
        2. AIParserMixin style: _error(message, line_no: int, line: str)
        """
        # Pattern 1: Grammar style with _Line object
        if isinstance(line_or_line_no, _Line):
            line = line_or_line_no
            return N3SyntaxError(
                f"Syntax error: {message}",
                path=self._path or None,
                line=line.number,
                code="SYNTAX_GRAMMAR",
                hint=line.text.strip() or None,
            )
        # Pattern 2: AIParserMixin style with line_no and line_text
        elif isinstance(line_or_line_no, int):
            line_no = line_or_line_no
            return N3SyntaxError(
                f"Syntax error: {message}",
                path=self._path or None,
                line=line_no,
                code="SYNTAX_GRAMMAR",
                hint=line_text.strip() if line_text else None,
            )
        # Fallback for no line info
        else:
            return N3SyntaxError(
                f"Syntax error: {message}",
                path=self._path or None,
                line=None,
                code="SYNTAX_GRAMMAR",
                hint=None,
            )
    
    def _parse_string_list(self, value: str) -> List[str]:
        """Parse a string representation of a list like '["a", "b", "c"]'."""
        import json
        try:
            # Try JSON parsing first
            result = json.loads(value)
            if isinstance(result, list):
                return [str(item) for item in result]
            return [str(result)]
        except (json.JSONDecodeError, ValueError):
            # Fall back to basic parsing
            if value.startswith('[') and value.endswith(']'):
                content = value[1:-1]
                items = []
                for item in content.split(','):
                    item = item.strip().strip('"').strip("'")
                    if item:
                        items.append(item)
                return items
            return [value.strip().strip('"').strip("'")]
    
    def _parse_function_def(self, line: _Line) -> None:
        """Function definition parser - moved to functions.py mixin."""
        pass

__all__ = ['UtilityParsersMixin']
