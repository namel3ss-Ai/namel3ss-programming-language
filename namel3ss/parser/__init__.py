from __future__ import annotations

from .base import N3SyntaxError
from .program import ProgramParserMixin


class Parser(ProgramParserMixin):
    """Public parser entry point for Namel3ss programs."""

    def __init__(self, source: str):
        super().__init__(source)


__all__ = ["Parser", "N3SyntaxError"]
