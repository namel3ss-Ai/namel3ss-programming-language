"""Helper classes and exceptions for grammar parsing."""

from __future__ import annotations
from dataclasses import dataclass


class GrammarUnsupportedError(Exception):
    """Raised when a grammar feature is recognized but not yet implemented."""
    pass


@dataclass
class _Line:
    """Wrapper for a line of source code with its line number."""
    text: str
    number: int


__all__ = ['GrammarUnsupportedError', '_Line']
