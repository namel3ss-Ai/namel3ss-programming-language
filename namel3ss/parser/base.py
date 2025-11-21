from __future__ import annotations

import ast
import io
import re
import tokenize
from tokenize import TokenInfo
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union, Literal as TypingLiteral
from pathlib import Path
from dataclasses import dataclass

_CONTEXT_SENTINEL = "__n3_ctx__"
_WHITESPACE_TOKENS: Set[int] = {
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
}
_BOOL_NORMALISATIONS: Dict[str, str] = {
    "true": "True",
    "false": "False",
    "null": "None",
    "none": "None",
}
_LIKE_TOKEN_MAP: Dict[str, str] = {
    "like": "<<",
    "ilike": ">>",
}
_EFFECT_KEYWORDS: Set[str] = {"pure", "ai"}

from namel3ss.ast import (
    AttributeRef,
    BinaryOp,
    CallExpression,
    CachePolicy,
    ContextValue,
    Expression,
    LayoutMeta,
    LayoutSpec,
    Literal,
    Import,
    NameRef,
    PaginationPolicy,
    StreamingPolicy,
    UnaryOp,
    WindowFrame,
    WindowOp,
)
from namel3ss.errors import N3SyntaxError
from namel3ss.parser.expression_builder import _ExpressionBuilder


# ============================================================================
# Indentation Analysis Data Structures
# ============================================================================

@dataclass
class IndentationInfo:
    """
    Detailed information about a line's indentation.
    
    This data structure provides comprehensive analysis of leading whitespace,
    supporting robust detection of mixed tabs/spaces and inconsistent indentation.
    
    Attributes:
        spaces: Number of leading spaces
        tabs: Number of leading tabs
        mixed: True if line has both tabs and spaces in leading whitespace
        indent_style: Detected indentation style
        effective_level: Effective indentation level for comparison (spaces + tabs*4)
    
    Examples:
        >>> info = IndentationInfo(spaces=4, tabs=0, mixed=False, 
        ...                        indent_style='spaces', effective_level=4)
        >>> info.spaces
        4
        >>> info.indent_style
        'spaces'
    """
    spaces: int
    tabs: int
    mixed: bool
    indent_style: TypingLiteral['spaces', 'tabs', 'mixed', 'none']
    effective_level: int


class ParserBase:
    """Shared parser state and helper utilities."""

    def __init__(self, source: str, *, module_name: Optional[str] = None, path: str = ""):
        self.lines: List[str] = source.splitlines()
        self.pos: int = 0
        self.app = None
        self._loop_depth: int = 0
        self._module_name_override: Optional[str] = module_name
        self.module_name: Optional[str] = module_name
        self.module_imports: List[Import] = []
        self._import_aliases: Dict[str, str] = {}
        self._named_imports: Dict[str, str] = {}
        self._module_declared = False
        self._explicit_app_declared = False
        self.language_version: Optional[str] = None
        self._allow_implicit_app = module_name is not None
        self.source_path: str = path

    # ------------------------------------------------------------------
    # Cursor helpers
    # ------------------------------------------------------------------
    def _peek(self) -> Optional[str]:
        """Return the current line without consuming it."""
        if self.pos < len(self.lines):
            return self.lines[self.pos]
        return None

    def _advance(self) -> Optional[str]:
        """Return the current line and move the cursor forward."""
        line = self._peek()
        self.pos += 1
        return line

    # ========================================================================
    # Indentation Analysis and Validation
    # ========================================================================

    def _compute_indent_details(self, line: str) -> IndentationInfo:
        """
        Analyze line indentation in comprehensive detail.
        
        This method performs robust analysis of leading whitespace, detecting:
        - Number of spaces and tabs
        - Mixed indentation (tabs + spaces)
        - Predominant indentation style
        
        Args:
            line: The line to analyze (must include leading whitespace)
            
        Returns:
            IndentationInfo with complete indentation analysis
            
        Examples:
            >>> info = parser._compute_indent_details("    code")
            >>> info.spaces
            4
            >>> info.indent_style
            'spaces'
            
            >>> info = parser._compute_indent_details("\\t\\tcode")
            >>> info.tabs
            2
            >>> info.indent_style
            'tabs'
            
            >>> info = parser._compute_indent_details("\\t  code")
            >>> info.mixed
            True
        """
        if not line:
            return IndentationInfo(
                spaces=0,
                tabs=0,
                mixed=False,
                indent_style='none',
                effective_level=0
            )
        
        # Count leading whitespace
        spaces = 0
        tabs = 0
        for char in line:
            if char == ' ':
                spaces += 1
            elif char == '\t':
                tabs += 1
            else:
                break  # Hit non-whitespace
        
        # Determine if mixed
        mixed = spaces > 0 and tabs > 0
        
        # Determine style
        if tabs > 0 and spaces == 0:
            indent_style = 'tabs'
        elif spaces > 0 and tabs == 0:
            indent_style = 'spaces'
        elif mixed:
            indent_style = 'mixed'
        else:
            indent_style = 'none'
        
        # Compute effective level (tabs count as 4 spaces for comparison)
        effective_level = spaces + (tabs * 4)
        
        return IndentationInfo(
            spaces=spaces,
            tabs=tabs,
            mixed=mixed,
            indent_style=indent_style,
            effective_level=effective_level
        )

    def _expect_indent_greater_than(
        self,
        line: str,
        base_indent: int,
        line_no: int,
        context: str = "block"
    ) -> IndentationInfo:
        """
        Validate that line is indented more than base_indent.
        
        This method ensures that a line is properly indented for a nested block,
        providing helpful error messages if indentation is insufficient.
        
        Args:
            line: The line to check (with leading whitespace)
            base_indent: The parent indentation level
            line_no: Line number for error reporting
            context: Human-readable context (e.g., "if block", "page body")
            
        Returns:
            IndentationInfo for the line if validation passes
            
        Raises:
            N3SyntaxError: If line is not properly indented with helpful hint
            
        Examples:
            # Valid - indented more than parent
            >>> info = parser._expect_indent_greater_than("    code", 0, 10, "page body")
            >>> info.effective_level > 0
            True
            
            # Invalid - not indented
            >>> parser._expect_indent_greater_than("code", 0, 10, "page body")
            N3SyntaxError: Expected indented block for page body...
        """
        info = self._compute_indent_details(line)
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            return info
        
        # Check for mixed indentation
        if info.mixed:
            message = f"Mixed tabs and spaces in indentation"
            hint = (
                "Use either tabs or spaces consistently throughout your file.\n"
                "Recommended: Use 4 spaces for indentation."
            )
            raise self._error(message, line_no, line, hint=hint)
        
        # Check if indented more than base
        if info.effective_level <= base_indent:
            message = f"Expected indented block for {context}"
            if info.effective_level == 0:
                hint = f"This line should be indented (e.g., 4 spaces) to be part of the {context}."
            else:
                hint = (
                    f"This line has {info.effective_level} spaces, but needs more than {base_indent} "
                    f"to be part of the {context}."
                )
            raise self._error(message, line_no, line, hint=hint)
        
        return info

    def _validate_block_indent(
        self,
        line: str,
        expected_indent: int,
        line_no: int,
        block_start_line: int,
        context: str = "block"
    ) -> IndentationInfo:
        """
        Validate that line matches expected block indentation.
        
        This method ensures consistent indentation within a block, detecting
        when lines have inconsistent indentation compared to the block start.
        
        Args:
            line: The line to check
            expected_indent: Expected indentation level
            line_no: Current line number
            block_start_line: Line where block started (for error messages)
            context: Human-readable context
            
        Returns:
            IndentationInfo for the line if validation passes
            
        Raises:
            N3SyntaxError: If indentation doesn't match with helpful hint
            
        Examples:
            # Consistent indentation - OK
            >>> info = parser._validate_block_indent("    stmt1", 4, 11, 10, "if body")
            >>> info.effective_level
            4
            
            # Inconsistent - error
            >>> parser._validate_block_indent("  stmt2", 4, 12, 10, "if body")
            N3SyntaxError: Inconsistent indentation in if body...
        """
        info = self._compute_indent_details(line)
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            return info
        
        # Check for mixed indentation
        if info.mixed:
            message = f"Mixed tabs and spaces in indentation"
            hint = (
                "Use either tabs or spaces consistently throughout your file.\n"
                "Recommended: Use 4 spaces for indentation."
            )
            raise self._error(message, line_no, line, hint=hint)
        
        # Check if indentation matches expected
        if info.effective_level != expected_indent:
            if info.effective_level < expected_indent:
                # Less indentation might mean end of block - don't error here
                # Caller will handle this as block termination
                return info
            
            # More indentation than expected - inconsistent
            message = f"Inconsistent indentation in {context}"
            hint = (
                f"This line uses {info.effective_level} spaces, "
                f"but the {context} started with {expected_indent} spaces at line {block_start_line}.\n"
                f"Hint: Use consistent indentation throughout each block."
            )
            raise self._error(message, line_no, line, hint=hint)
        
        return info

    def _detect_indentation_issues(self, lines: Optional[List[str]] = None) -> Optional[str]:
        """
        Scan lines for common indentation problems.
        
        This diagnostic method scans source lines to detect common indentation
        issues that might cause parsing problems:
        - Mixed tabs and spaces across file
        - Inconsistent indentation increments (mixing 2-space and 4-space)
        - Tab usage (generally discouraged)
        
        Args:
            lines: Lines to scan (defaults to self.lines)
            
        Returns:
            Warning message string if issues found, None otherwise
            
        Note:
            This is typically called during parser initialization for early
            detection of indentation problems. It returns warnings, not errors.
            
        Examples:
            >>> warning = parser._detect_indentation_issues()
            >>> if warning:
            ...     print(f"Warning: {warning}")
        """
        if lines is None:
            lines = self.lines
        
        if not lines:
            return None
        
        has_tabs = False
        has_spaces = False
        indent_levels: Set[int] = set()
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            info = self._compute_indent_details(line)
            
            if info.tabs > 0:
                has_tabs = True
            if info.spaces > 0:
                has_spaces = True
            
            if info.mixed:
                return (
                    "Mixed tabs and spaces detected in file. "
                    "Use either tabs or spaces consistently (recommended: 4 spaces)."
                )
            
            if info.effective_level > 0:
                indent_levels.add(info.effective_level)
        
        # Check for mixed tab/space usage across file
        if has_tabs and has_spaces:
            return (
                "File mixes tab and space indentation on different lines. "
                "Use either tabs or spaces consistently (recommended: 4 spaces)."
            )
        
        # Check for inconsistent indentation increments
        if len(indent_levels) > 1:
            sorted_levels = sorted(indent_levels)
            # Check if increments are consistent (all multiples of smallest)
            smallest = sorted_levels[0]
            if smallest > 0:
                inconsistent = any(level % smallest != 0 for level in sorted_levels if level != smallest)
                if inconsistent:
                    return (
                        f"Inconsistent indentation increments detected (found levels: {sorted(indent_levels)}). "
                        f"Consider using consistent {smallest}-space indentation throughout."
                    )
        
        # Warn if tabs are used (spaces are more portable)
        if has_tabs and not has_spaces:
            return (
                "File uses tab indentation. "
                "Consider using spaces for better portability (recommended: 4 spaces)."
            )
        
        return None

    def _indent(self, line: str) -> int:
        """
        Compute the indentation level (leading spaces) for *line*.
        
        **BACKWARD COMPATIBILITY METHOD**
        
        This method is preserved for backward compatibility with existing parser code.
        Returns effective indentation level (spaces count, with tabs counted as 4 spaces).
        
        New code should use _compute_indent_details() for robust indentation handling
        with tab/space detection and better error messages.
        
        Args:
            line: The line to analyze
            
        Returns:
            Effective indentation level (number of spaces, tabs count as 4)
            
        Examples:
            >>> parser._indent("    code")
            4
            >>> parser._indent("\\tcode")  # Tab counts as 4
            4
        """
        return self._compute_indent_details(line).effective_level

    def _error(
        self,
        message: str,
        line_no: Optional[int] = None,
        line: Optional[str] = None,
        hint: Optional[str] = None,
    ) -> N3SyntaxError:
        """
        Create a comprehensive N3SyntaxError instance.
        
        Args:
            message: Primary error message
            line_no: Line number (1-indexed, defaults to current position)
            line: The problematic line text (defaults to line at pos-1)
            hint: Optional helpful hint for fixing the error
            
        Returns:
            N3SyntaxError instance with all context
            
        Examples:
            # Simple error
            raise self._error("Expected ':' after if condition", line_no, line)
            
            # Error with hint
            raise self._error(
                "Inconsistent indentation detected",
                line_no,
                line,
                hint="Use consistent indentation throughout each block"
            )
        """
        if line_no is None:
            line_no = min(self.pos, len(self.lines))
        if line is None and 0 <= self.pos - 1 < len(self.lines):
            line = self.lines[self.pos - 1]
        elif line is None:
            line = ""
        
        # Use provided hint or fall back to line content
        error_hint = hint if hint is not None else (line.strip() or None)
        
        return N3SyntaxError(
            f"Syntax error: {message}",
            path=self.source_path or None,
            line=line_no,
            code="SYNTAX_ERROR",
            hint=error_hint,
        )

    def _default_app_name(self) -> str:
        if self.module_name:
            return self.module_name.split('.')[-1]
        if self.source_path:
            stem = Path(self.source_path).stem
            if stem:
                return stem
        return "app"

    # ------------------------------------------------------------------
    # Scalar coercion helpers
    # ------------------------------------------------------------------
    def _parse_bool(self, raw: str) -> bool:
        """Parse a boolean-like string into a bool."""
        value = raw.strip().lower()
        if value in {"true", "yes", "1", "on"}:
            return True
        if value in {"false", "no", "0", "off"}:
            return False
        raise self._error(f"Expected boolean value, found '{raw}'")

    def _parse_context_reference(self, token: str) -> Optional[ContextValue]:
        match = re.match(r'^(ctx|env):([A-Za-z0-9_\.]+)$', token)
        if not match:
            return None
        scope = match.group(1)
        path_text = match.group(2)
        if not path_text:
            return None
        path = [segment for segment in path_text.split('.') if segment]
        if not path:
            return None
        return ContextValue(scope=scope, path=path)

    def _coerce_scalar(self, raw: str) -> Any:
        """Attempt to coerce a scalar configuration value."""
        text = raw.strip()
        if not text:
            return text

        context_ref = self._parse_context_reference(text)
        if context_ref is not None:
            return context_ref
        lower = text.lower()
        if lower in {"true", "false", "null", "none"}:
            if lower in {"true", "false"}:
                return self._parse_bool(text)
            return None
        if re.fullmatch(r"[-+]?\d+", text):
            try:
                return int(text)
            except ValueError:
                pass
        if re.fullmatch(r"[-+]?\d*\.\d+", text):
            try:
                return float(text)
            except ValueError:
                pass
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            inner = text[1:-1]
            context_ref_inner = self._parse_context_reference(inner)
            if context_ref_inner is not None:
                return context_ref_inner
            return inner
        if text.startswith('[') or text.startswith('{') or text.startswith('('):
            try:
                parsed = ast.literal_eval(text)
                return parsed
            except (SyntaxError, ValueError):
                pass
        return text

    def _coerce_expression(self, value: Any) -> Expression:
        if isinstance(value, Expression):
            return value
        if isinstance(value, str):
            try:
                return self._parse_expression(value)
            except N3SyntaxError:
                return Literal(value)
        return Literal(value)

    def _coerce_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        try:
            text = str(value).strip()
            if not text:
                return None
            match = re.match(r"(-?\d+)", text)
            if not match:
                return None
            return int(match.group(1))
        except (ValueError, TypeError):
            return None

    def _stringify_value(self, value: Any) -> str:
        if value is None:
            return ''
        if isinstance(value, ContextValue):
            path = '.'.join(value.path)
            return f"{value.scope}:{path}" if path else value.scope
        if isinstance(value, Literal):
            inner = value.value
            return '' if inner is None else str(inner)
        if isinstance(value, Expression):
            return str(value)
        return str(value)

    # ------------------------------------------------------------------
    # Enhanced coercion helpers with contextual error messages
    # ------------------------------------------------------------------

    def _coerce_scalar_with_context(
        self,
        raw: Any,
        field_name: str,
        expected_type: Optional[str] = None,
        line_no: Optional[int] = None,
        line: Optional[str] = None
    ) -> Any:
        """
        Coerce scalar value with contextual error messages.
        
        This enhanced version of _coerce_scalar provides better error messages
        by including the field name and expected type in errors.
        
        Args:
            raw: The raw value to coerce
            field_name: Name of the field being parsed (for error messages)
            expected_type: Expected type hint ('int', 'float', 'bool', 'string', None for any)
            line_no: Line number for error reporting
            line: Line text for error reporting
            
        Returns:
            Coerced value
            
        Raises:
            N3SyntaxError: If coercion fails with helpful context
            
        Examples:
            # Good coercion
            value = self._coerce_scalar_with_context("42", "page_size", "int", line_no, line)
            # value = 42
            
            # Error with context
            value = self._coerce_scalar_with_context("abc", "page_size", "int", line_no, line)
            # Raises: Invalid value for page_size
            #   → Expected int, got "abc"
            #   Hint: page_size must be a positive integer
        """
        try:
            result = self._coerce_scalar(raw)
            
            # Validate type if specified
            if expected_type == 'int':
                if not isinstance(result, int):
                    raise ValueError(f"Expected integer, got {type(result).__name__}")
            elif expected_type == 'float':
                if not isinstance(result, (int, float)):
                    raise ValueError(f"Expected number, got {type(result).__name__}")
            elif expected_type == 'bool':
                if not isinstance(result, bool):
                    raise ValueError(f"Expected boolean, got {type(result).__name__}")
            elif expected_type == 'string':
                if not isinstance(result, str):
                    raise ValueError(f"Expected string, got {type(result).__name__}")
            
            return result
            
        except (ValueError, TypeError) as exc:
            message = f"Invalid value for {field_name}"
            if expected_type:
                message += f"\n  → Expected {expected_type}, got {repr(raw)}"
            else:
                message += f"\n  → Got {repr(raw)}"
            
            hint = self._coercion_hint(field_name, expected_type)
            
            raise self._error(message, line_no, line, hint=hint)

    def _coerce_int_with_context(
        self,
        raw: Any,
        field_name: str,
        line_no: Optional[int] = None,
        line: Optional[str] = None,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> int:
        """
        Coerce value to integer with validation and context.
        
        Args:
            raw: The raw value to coerce
            field_name: Name of the field being parsed
            line_no: Line number for error reporting
            line: Line text for error reporting
            min_value: Optional minimum allowed value
            max_value: Optional maximum allowed value
            
        Returns:
            Coerced integer value
            
        Raises:
            N3SyntaxError: If coercion or validation fails
            
        Examples:
            # Valid integer
            value = self._coerce_int_with_context("42", "page_size", line_no, line, min_value=1)
            # value = 42
            
            # Invalid - not a number
            value = self._coerce_int_with_context("abc", "page_size", line_no, line)
            # Raises: Invalid value for page_size
            #   → Expected integer, got "abc"
            
            # Invalid - out of range
            value = self._coerce_int_with_context("-5", "page_size", line_no, line, min_value=1)
            # Raises: Invalid value for page_size
            #   → Value -5 is less than minimum 1
        """
        result = self._coerce_int(raw)
        
        if result is None:
            message = f"Invalid value for {field_name}"
            message += f"\n  → Expected integer, got {repr(raw)}"
            hint = self._coercion_hint(field_name, 'int')
            raise self._error(message, line_no, line, hint=hint)
        
        # Validate range
        if min_value is not None and result < min_value:
            message = f"Invalid value for {field_name}"
            message += f"\n  → Value {result} is less than minimum {min_value}"
            hint = f"{field_name} must be at least {min_value}"
            raise self._error(message, line_no, line, hint=hint)
        
        if max_value is not None and result > max_value:
            message = f"Invalid value for {field_name}"
            message += f"\n  → Value {result} is greater than maximum {max_value}"
            hint = f"{field_name} must be at most {max_value}"
            raise self._error(message, line_no, line, hint=hint)
        
        return result

    def _coerce_bool_with_context(
        self,
        raw: Any,
        field_name: str,
        line_no: Optional[int] = None,
        line: Optional[str] = None
    ) -> bool:
        """
        Coerce value to boolean with context.
        
        Args:
            raw: The raw value to coerce
            field_name: Name of the field being parsed
            line_no: Line number for error reporting
            line: Line text for error reporting
            
        Returns:
            Coerced boolean value
            
        Raises:
            N3SyntaxError: If coercion fails
        """
        try:
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str):
                return self._parse_bool(raw)
            # Try to coerce through scalar
            result = self._coerce_scalar(raw)
            if isinstance(result, bool):
                return result
            raise ValueError("Not a boolean")
        except (ValueError, N3SyntaxError):
            message = f"Invalid value for {field_name}"
            message += f"\n  → Expected boolean, got {repr(raw)}"
            hint = "Use true/false, yes/no, 1/0, or on/off"
            raise self._error(message, line_no, line, hint=hint)

    def _coercion_hint(self, field_name: str, expected_type: Optional[str]) -> Optional[str]:
        """
        Generate helpful hint for coercion errors.
        
        Provides field-specific or type-specific hints to help users
        understand what values are acceptable.
        
        Args:
            field_name: Name of field being coerced
            expected_type: Expected type ('int', 'float', 'bool', 'string')
            
        Returns:
            Hint string or None
            
        Examples:
            >>> parser._coercion_hint('page_size', 'int')
            'page_size must be a positive integer (e.g., 10, 20, 50)'
            
            >>> parser._coercion_hint('unknown_field', 'int')
            'Must be a whole number (e.g., 1, 42, 100)'
        """
        # Field-specific hints for common configuration fields
        field_hints = {
            'page_size': 'page_size must be a positive integer (e.g., 10, 20, 50)',
            'max_entries': 'max_entries must be a positive integer',
            'max_pages': 'max_pages must be a positive integer',
            'chunk_size': 'chunk_size must be a positive integer',
            'ttl_seconds': 'ttl_seconds must be a positive integer (time in seconds)',
            'ttl': 'ttl must be a positive integer (time in seconds)',
            'temperature': 'temperature must be a number between 0.0 and 2.0',
            'top_p': 'top_p must be a number between 0.0 and 1.0',
            'max_tokens': 'max_tokens must be a positive integer',
            'width': 'width must be a positive integer (pixels or grid units)',
            'height': 'height must be a positive integer (pixels or grid units)',
            'batch_size': 'batch_size must be a positive integer',
            'epochs': 'epochs must be a positive integer',
            'learning_rate': 'learning_rate must be a positive number (e.g., 0.001, 0.01)',
            'timeout': 'timeout must be a positive number (seconds)',
            'retry_count': 'retry_count must be a non-negative integer',
            'port': 'port must be an integer between 1 and 65535',
        }
        
        if field_name in field_hints:
            return field_hints[field_name]
        
        # Generic type hints
        if expected_type == 'int':
            return 'Must be a whole number (e.g., 1, 42, 100)'
        elif expected_type == 'float':
            return 'Must be a number (e.g., 1.5, 3.14, 0.5)'
        elif expected_type == 'bool':
            return 'Must be true/false, yes/no, or 1/0'
        elif expected_type == 'string':
            return 'Must be a text value (optionally in quotes)'
        
        return None

    # ------------------------------------------------------------------
    # Block helpers
    # ------------------------------------------------------------------
    def _parse_kv_block(self, parent_indent: int) -> Dict[str, Any]:
        """Parse an indented key/value block and return a dictionary."""
        config: Dict[str, Any] = {}
        while self.pos < len(self.lines):
            line = self._peek()
            if line is None:
                break
            stripped = line.strip()
            
            # Handle closing brace for blocks
            if stripped == '}':
                self._advance()
                break
            
            indent = self._indent(line)
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                self._advance()
                continue
            if indent <= parent_indent or stripped.startswith('-'):
                break
            match = re.match(r'([\w\.\s]+):\s*(.*)$', stripped)
            if not match:
                raise self._error("Expected 'key: value' inside block", self.pos + 1, line)
            key = match.group(1).strip()
            remainder = match.group(2)
            self._advance()
            if remainder == "":
                nested = self._parse_kv_block(indent)
                config[key] = nested
            else:
                value = self._coerce_scalar(remainder)
                config[key] = value
        return config

    def _parse_string_list(self, parent_indent: int) -> List[str]:
        """Parse a bullet list (``- entry``) into a list of strings."""

        values: List[str] = []
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
            if not stripped.startswith('-'):
                break
            entry = stripped[1:].strip()
            self._advance()
            if not entry:
                raise self._error("List entry cannot be empty", self.pos, line)
            scalar = self._coerce_scalar(entry)
            text = self._strip_quotes(str(scalar)) if scalar is not None else ""
            if not text:
                raise self._error("List entry cannot be blank", self.pos, line)
            values.append(text)
        return values

    def _peek_next_content_line(self) -> Optional[str]:
        """Return the next non-empty, non-comment line without advancing."""

        idx = self.pos
        while idx < len(self.lines):
            line = self.lines[idx]
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                idx += 1
                continue
            return line
        return None

    def _coerce_options_dict(self, raw: Any) -> Dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return {key: self._transform_config(value) for key, value in raw.items()}
        return {"value": self._transform_config(raw)}

    def _strip_quotes(self, value: str) -> str:
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        return value

    def _parse_effect_annotation(
        self,
        token: Optional[str],
        line_no: int,
        line: str,
    ) -> Optional[str]:
        if token is None:
            return None
        value = token.strip().lower()
        if value not in _EFFECT_KEYWORDS:
            allowed = ", ".join(sorted(_EFFECT_KEYWORDS))
            raise self._error(f"Unknown effect '{token}'. Allowed effects: {allowed}", line_no, line)
        return value

    # ------------------------------------------------------------------
    # Windowing & layout helpers
    # ------------------------------------------------------------------
    def _parse_window_frame(self, raw: str) -> WindowFrame:
        """Parse textual window specification into a WindowFrame."""
        text = raw.strip()
        lower = text.lower()
        mode: TypingLiteral["rolling", "expanding", "cumulative"] = "rolling"
        interval_value: Optional[int] = None
        interval_unit: Optional[str] = None

        match_last = re.match(r'last\s+(\d+)\s+([\w]+)', lower)
        if match_last:
            interval_value = int(match_last.group(1))
            interval_unit = match_last.group(2)
        elif lower.startswith('over all') or lower in {'all', 'overall', 'cumulative'}:
            mode = "cumulative"
        elif lower.startswith('expanding'):
            mode = "expanding"

        return WindowFrame(mode=mode, interval_value=interval_value, interval_unit=interval_unit)

    def _build_cache_policy(self, data: Dict[str, Any]) -> CachePolicy:
        if not data:
            return CachePolicy(strategy="none")
        strategy = str(data.get('strategy', 'memory') or 'memory').lower()
        ttl_raw = data.get('ttl_seconds') or data.get('ttl') or data.get('ttl_s')
        ttl_seconds: Optional[int] = None
        if ttl_raw is not None:
            if isinstance(ttl_raw, (int, float)):
                ttl_seconds = int(ttl_raw)
            else:
                ttl_clean = str(ttl_raw).strip()
                match_val = re.match(r'(\d+)', ttl_clean)
                if match_val:
                    ttl_seconds = int(match_val.group(1))
        max_entries = data.get('max_entries') or data.get('max rows') or data.get('max')
        if max_entries is not None and not isinstance(max_entries, int):
            try:
                max_entries = int(str(max_entries))
            except ValueError:
                max_entries = None
        return CachePolicy(strategy=strategy, ttl_seconds=ttl_seconds, max_entries=max_entries)

    def _build_pagination_policy(self, data: Dict[str, Any]) -> PaginationPolicy:
        if not data:
            return PaginationPolicy(enabled=False)
        enabled = self._parse_bool(str(data.get('enabled', 'true'))) if 'enabled' in data else True
        page_size = data.get('page_size') or data.get('page size') or data.get('size')
        if page_size is not None and not isinstance(page_size, int):
            try:
                page_size = int(str(page_size))
            except ValueError:
                page_size = None
        max_pages = data.get('max_pages') or data.get('max pages')
        if max_pages is not None and not isinstance(max_pages, int):
            try:
                max_pages = int(str(max_pages))
            except ValueError:
                max_pages = None
        return PaginationPolicy(enabled=enabled, page_size=page_size, max_pages=max_pages)

    def _build_streaming_policy(self, data: Dict[str, Any]) -> StreamingPolicy:
        if not data:
            return StreamingPolicy(enabled=True)
        enabled = self._parse_bool(str(data.get('enabled', 'true'))) if 'enabled' in data else True
        chunk_size = data.get('chunk_size') or data.get('chunk size') or data.get('batch')
        if chunk_size is not None and not isinstance(chunk_size, int):
            try:
                chunk_size = int(str(chunk_size))
            except ValueError:
                chunk_size = None
        return StreamingPolicy(enabled=enabled, chunk_size=chunk_size)

    def _build_layout_spec(self, data: Dict[str, Any]) -> LayoutSpec:
        layout = LayoutSpec()

        def _to_int(val: Any) -> Optional[int]:
            if isinstance(val, int):
                return val
            if isinstance(val, float):
                return int(val)
            try:
                if isinstance(val, str) and not val.strip():
                    return None
                return int(val)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                try:
                    return int(float(str(val)))
                except (ValueError, TypeError):
                    return None

        for key, value in data.items():
            lower = key.replace(' ', '_').lower()
            if lower == 'width':
                maybe = _to_int(value)
                if maybe is not None:
                    layout.width = maybe
            elif lower == 'height':
                maybe = _to_int(value)
                if maybe is not None:
                    layout.height = maybe
            elif lower == 'variant':
                layout.variant = str(value)
            elif lower == 'order':
                maybe = _to_int(value)
                if maybe is not None:
                    layout.order = maybe
            elif lower == 'area':
                layout.area = str(value)
            elif lower == 'breakpoint':
                layout.breakpoint = str(value)
            else:
                layout.props[key] = value
        return layout

    def _build_layout_meta(self, data: Dict[str, Any]) -> LayoutMeta:
        meta = LayoutMeta()

        def _to_int(val: Any) -> Optional[int]:
            if isinstance(val, int):
                return val
            if isinstance(val, float):
                return int(val)
            try:
                if isinstance(val, str) and not val.strip():
                    return None
                return int(val)
            except (ValueError, TypeError):
                try:
                    return int(float(str(val)))
                except (ValueError, TypeError):
                    return None

        extras: Dict[str, Any] = {}
        for key, value in data.items():
            lower = key.replace(' ', '_').lower()
            if lower == 'width':
                maybe = _to_int(value)
                if maybe is not None:
                    meta.width = maybe
            elif lower == 'height':
                maybe = _to_int(value)
                if maybe is not None:
                    meta.height = maybe
            elif lower == 'variant':
                meta.variant = str(value)
            elif lower == 'align':
                meta.align = str(value)
            elif lower == 'emphasis':
                meta.emphasis = str(value)
            else:
                extras[key] = value
        meta.extras = extras
        return meta

    # Placeholder methods expected in subclasses/mixins
    def _parse_expression(self, text: str) -> Expression:
        """Parse an expression string into an Expression AST node."""

        source = (text or "").strip()
        if not source:
            raise self._error("Expression cannot be empty", self.pos, text)

        def _raise(message: str) -> None:
            raise self._error(message, self.pos, text)

        prepared = _prepare_expression_source(source, _raise)
        try:
            parsed = ast.parse(prepared, mode="eval")
        except SyntaxError as exc:
            details = exc.msg or "Invalid expression syntax"
            if exc.offset is not None:
                details = f"{details} (column {exc.offset})"
            _raise(details)

        builder = _ExpressionBuilder(_raise)
        try:
            expression = builder.convert(parsed)
        except N3SyntaxError:
            raise
        except Exception as exc:  # pragma: no cover - defensive safeguard
            _raise(f"Unsupported expression element: {exc}")

        if not isinstance(expression, Expression):  # pragma: no cover - defensive
            _raise("Parsed expression did not produce an Expression node")
        return expression

def _prepare_expression_source(source: str, raise_error: Callable[[str], None]) -> str:
    """Convert N3 expression syntax into Python-compatible source for AST parsing."""

    reader = io.StringIO(source).readline
    try:
        tokens = list(tokenize.generate_tokens(reader))
    except tokenize.TokenError as exc:
        raise_error(f"Invalid expression: {exc}")

    result: List[TokenInfo] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        token_type = token.type
        token_str = token.string

        if token_type == tokenize.ENDMARKER:
            result.append(token)
            break

        if token_type == tokenize.NAME:
            lowered = token_str.lower()

            if lowered in _BOOL_NORMALISATIONS:
                normalised = _BOOL_NORMALISATIONS[lowered]
                result.append(TokenInfo(tokenize.NAME, normalised, token.start, token.end, token.line))
                i += 1
                continue

            if lowered in {"ctx", "env"}:
                j = i + 1
                while j < len(tokens) and tokens[j].type in _WHITESPACE_TOKENS:
                    j += 1
                if j < len(tokens) and tokens[j].type == tokenize.OP and tokens[j].string == ":":
                    j += 1
                    path_parts: List[str] = []
                    expect_segment = True
                    while j < len(tokens):
                        lookahead = tokens[j]
                        if lookahead.type in _WHITESPACE_TOKENS:
                            j += 1
                            continue
                        if expect_segment and lookahead.type in {tokenize.NAME, tokenize.NUMBER}:
                            path_parts.append(lookahead.string)
                            j += 1
                            expect_segment = False
                            continue
                        if not expect_segment and lookahead.type == tokenize.OP and lookahead.string == '.':
                            expect_segment = True
                            j += 1
                            continue
                        break
                    if expect_segment:
                        raise_error("Expected context path after prefix")
                    if not path_parts:
                        raise_error("Context path cannot be empty")
                    result.extend(_build_context_tokens(lowered, path_parts, token))
                    i = j
                    continue

            if lowered in _LIKE_TOKEN_MAP:
                j = i + 1
                while j < len(tokens) and tokens[j].type in _WHITESPACE_TOKENS:
                    j += 1
                if j < len(tokens) and tokens[j].type == tokenize.OP and tokens[j].string == '(':
                    result.append(token)
                    i += 1
                    continue
                replacement = _LIKE_TOKEN_MAP[lowered]
                result.append(TokenInfo(tokenize.OP, replacement, token.start, token.end, token.line))
                i += 1
                continue

            result.append(token)
            i += 1
            continue

        if token_type == tokenize.OP:
            if token_str == '=':
                result.append(TokenInfo(tokenize.OP, '==', token.start, token.end, token.line))
                i += 1
                continue
            if token_str == '<>':
                result.append(TokenInfo(tokenize.OP, '!=', token.start, token.end, token.line))
                i += 1
                continue
            if token_str == '<' and i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if nxt.type == tokenize.OP and nxt.string == '>':
                    result.append(TokenInfo(tokenize.OP, '!=', token.start, nxt.end, token.line))
                    i += 2
                    continue

        result.append(token)
        i += 1

    return tokenize.untokenize(result)


def _build_context_tokens(scope: str, path: List[str], template: TokenInfo) -> List[TokenInfo]:
    """Create the token sequence representing a context reference call."""

    line = template.line
    tokens: List[TokenInfo] = [
        TokenInfo(tokenize.NAME, _CONTEXT_SENTINEL, template.start, template.end, line),
        TokenInfo(tokenize.OP, '(', template.start, template.end, line),
        TokenInfo(tokenize.STRING, repr(scope), template.start, template.end, line),
    ]
    for segment in path:
        tokens.append(TokenInfo(tokenize.OP, ',', template.start, template.end, line))
        tokens.append(TokenInfo(tokenize.STRING, repr(segment), template.start, template.end, line))
    tokens.append(TokenInfo(tokenize.OP, ')', template.start, template.end, line))
    return tokens
