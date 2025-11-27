"""Structured metadata for Namel3ss comments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Comment:
    """Represents a single line comment in source code.

    Comments are ignored by the parser/IR but preserved for documentation,
    hover tooltips, and structure outlines.
    """

    raw: str
    text: str
    emoji: Optional[str]
    line: int
    column: int = 1


__all__ = ["Comment"]
