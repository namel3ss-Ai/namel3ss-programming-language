"""Source location information for AST nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SourceLocation:
    """
    Represents a location in source code.
    
    Used to track where AST nodes originated from for error reporting
    and debugging purposes.
    """
    file: str
    line: int
    column: int = 0
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    
    def __str__(self) -> str:
        """Return human-readable location string."""
        if self.end_line and self.end_line != self.line:
            return f"{self.file}:{self.line}-{self.end_line}"
        return f"{self.file}:{self.line}:{self.column}"


__all__ = ["SourceLocation"]
