"""
Semantic linter for Namel3ss.

This module provides production-grade semantic analysis and linting
that goes beyond syntax checking to analyze semantic correctness,
best practices, and potential issues in Namel3ss applications.
"""

from __future__ import annotations

__all__ = ["SemanticLinter", "LintRule", "LintResult", "LintSeverity"]

from .core import SemanticLinter, LintResult, LintSeverity
from .rules import LintRule
from .builtin_rules import get_default_rules