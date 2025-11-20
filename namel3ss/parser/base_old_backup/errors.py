"""Error handling and diagnostics."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...errors import N3SyntaxError


class ErrorMixin:
    """Mixin providing error creation and diagnostics."""
    
    def _error(
        self,
        message: str,
        line_no: Optional[int] = None,
        line: Optional[str] = None,
        hint: Optional[str] = None,
    ) -> "N3SyntaxError":
        """
        Create a comprehensive N3SyntaxError instance.
        
        Args:
            message: Primary error message
            line_no: Line number (1-indexed, defaults to current position)
            line: The problematic line text
            hint: Optional helpful hint for fixing the error
            
        Returns:
            N3SyntaxError instance with context
        """
        from namel3ss.errors import N3SyntaxError
        
        if line_no is None:
            line_no = min(self.pos, len(self.lines))
        if line is None and 0 <= self.pos - 1 < len(self.lines):
            line = self.lines[self.pos - 1]
        elif line is None:
            line = ""
        
        error_hint = hint if hint is not None else (line.strip() or None)
        
        return N3SyntaxError(
            f"Syntax error: {message}",
            path=self.source_path or None,
            line=line_no,
            code="SYNTAX_ERROR",
            hint=error_hint,
        )
