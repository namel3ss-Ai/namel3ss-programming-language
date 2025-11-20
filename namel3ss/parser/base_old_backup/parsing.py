"""Core parsing cursor operations."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING


class ParsingMixin:
    """Mixin providing core parsing cursor operations."""
    
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
