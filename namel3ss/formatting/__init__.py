"""
Production-grade AST-based formatter for Namel3ss.

This module provides a sophisticated formatter that:
1. Parses .ai files into AST
2. Applies consistent formatting rules
3. Preserves semantic meaning while standardizing style
4. Integrates with both CLI and LSP
"""

from __future__ import annotations

__all__ = ["ASTFormatter", "FormattingOptions", "FormattedResult", "IndentStyle", "DefaultFormattingRules"]

from .core import ASTFormatter, FormattingOptions, FormattedResult, IndentStyle

# Simple default rules to get started
class DefaultFormattingRules:
    @staticmethod
    def standard():
        return FormattingOptions(
            indent_style=IndentStyle.SPACES,
            indent_size=4,
            trim_trailing_whitespace=True,
            insert_final_newline=True
        )