"""Backward compatibility wrapper for dataset parser."""

from __future__ import annotations

# Import the refactored implementation
from .datasets import DatasetParserMixin

# Re-export for backward compatibility
__all__ = ['DatasetParserMixin']
