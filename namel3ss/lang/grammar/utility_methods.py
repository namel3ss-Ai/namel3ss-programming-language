"""Core utility methods for grammar parsing."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .helpers import _Line

from namel3ss.ast import App
from namel3ss.parser.base import N3SyntaxError


class UtilityMethodsMixin:
    """Mixin providing core utility methods for grammar parsing."""

    def _ensure_app(self, line: _Line) -> None:
        """Ensure an App object exists, creating one if necessary."""
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
        from .helpers import _Line
        if isinstance(text, _Line):
            text = text.text
        return len(text) - len(text.lstrip(' '))

    def _error(self, message: str, line_or_line_no=None, line_text: str = None) -> N3SyntaxError:
        """
        Create a syntax error. Supports two call patterns:
        1. Grammar style: _error(message, line: _Line)
        2. AIParserMixin style: _error(message, line_no: int, line: str)
        """
        from .helpers import _Line
        
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

    def _unsupported(self, line: _Line, feature: str) -> None:
        """Raise error for unsupported grammar feature."""
        from .helpers import GrammarUnsupportedError
        location = f"{self._path}:{line.number}" if self._path else f"line {line.number}"
        raise GrammarUnsupportedError(f"Unsupported {feature} near {location}")
    
    def _transform_config(self, value: Any) -> Any:
        """Transform configuration values for compatibility with legacy parser."""
        # Simple passthrough - extend as needed for type conversions
        if isinstance(value, str):
            # Handle common type conversions
            if value.lower() in ('true', 'false'):
                return value.lower() == 'true'
            try:
                if '.' in value:
                    return float(value)
                return int(value)
            except ValueError:
                return value
        return value


__all__ = ['UtilityMethodsMixin']
