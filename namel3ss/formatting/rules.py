"""Default formatting rules for Namel3ss."""

from __future__ import annotations

from .core import FormattingOptions, IndentStyle


class DefaultFormattingRules:
    """Default formatting rules following Namel3ss style guide."""
    
    @classmethod
    def standard(cls) -> FormattingOptions:
        """Standard Namel3ss formatting rules."""
        return FormattingOptions(
            indent_style=IndentStyle.SPACES,
            indent_size=4,
            tab_size=4,
            max_line_length=100,
            insert_final_newline=True,
            trim_trailing_whitespace=True,
            preserve_empty_lines=False,
            max_empty_lines=2,
            wrap_long_expressions=True,
            align_multiline_parameters=True,
            prefer_single_quotes=False,
            normalize_quotes=True,
            preserve_comments=True,
            align_comments=True,
        )
    
    @classmethod
    def compact(cls) -> FormattingOptions:
        """Compact formatting with minimal whitespace."""
        return FormattingOptions(
            indent_style=IndentStyle.SPACES,
            indent_size=2,
            tab_size=2,
            max_line_length=120,
            insert_final_newline=True,
            trim_trailing_whitespace=True,
            preserve_empty_lines=False,
            max_empty_lines=1,
            wrap_long_expressions=False,
            align_multiline_parameters=False,
            prefer_single_quotes=False,
            normalize_quotes=True,
            preserve_comments=True,
            align_comments=False,
        )
    
    @classmethod
    def expanded(cls) -> FormattingOptions:
        """Expanded formatting with generous whitespace."""
        return FormattingOptions(
            indent_style=IndentStyle.SPACES,
            indent_size=4,
            tab_size=4,
            max_line_length=80,
            insert_final_newline=True,
            trim_trailing_whitespace=True,
            preserve_empty_lines=True,
            max_empty_lines=3,
            wrap_long_expressions=True,
            align_multiline_parameters=True,
            prefer_single_quotes=False,
            normalize_quotes=True,
            preserve_comments=True,
            align_comments=True,
        )