"""Indentation analysis and validation."""

from __future__ import annotations

from typing import Optional, Set, List, TYPE_CHECKING, Literal as TypingLiteral
from dataclasses import dataclass

if TYPE_CHECKING:
    pass


@dataclass
class IndentationInfo:
    """
    Detailed information about a line's indentation.
    
    Attributes:
        spaces: Number of leading spaces
        tabs: Number of leading tabs
        mixed: True if line has both tabs and spaces
        indent_style: Detected indentation style
        effective_level: Effective indentation level (spaces + tabs*4)
    """
    spaces: int
    tabs: int
    mixed: bool
    indent_style: TypingLiteral['spaces', 'tabs', 'mixed', 'none']
    effective_level: int


class IndentationMixin:
    """Mixin providing indentation analysis and validation."""
    
    def _compute_indent_details(self, line: str) -> IndentationInfo:
        """Analyze line indentation in comprehensive detail."""
        if not line:
            return IndentationInfo(
                spaces=0, tabs=0, mixed=False,
                indent_style='none', effective_level=0
            )
        
        spaces = tabs = 0
        for char in line:
            if char == ' ':
                spaces += 1
            elif char == '\t':
                tabs += 1
            else:
                break
        
        mixed = spaces > 0 and tabs > 0
        if tabs > 0 and spaces == 0:
            indent_style = 'tabs'
        elif spaces > 0 and tabs == 0:
            indent_style = 'spaces'
        elif mixed:
            indent_style = 'mixed'
        else:
            indent_style = 'none'
        
        effective_level = spaces + (tabs * 4)
        
        return IndentationInfo(
            spaces=spaces, tabs=tabs, mixed=mixed,
            indent_style=indent_style, effective_level=effective_level
        )

    def _expect_indent_greater_than(
        self, base_indent: int, line: str, line_no: int,
        context: str = "block", hint: Optional[str] = None
    ) -> IndentationInfo:
        """Validate that line is indented more than base_indent."""
        info = self._compute_indent_details(line)
        stripped = line.strip()
        
        if not stripped or stripped.startswith('#'):
            return info
        
        if info.mixed:
            raise self._error(
                "Mixed tabs and spaces in indentation",
                line_no, line,
                hint="Use either tabs or spaces consistently (recommended: 4 spaces)."
            )
        
        if info.effective_level <= base_indent:
            message = f"Expected indented block for {context}"
            if info.effective_level == 0:
                error_hint = hint or f"This line should be indented (e.g., 4 spaces) to be part of the {context}."
            else:
                error_hint = hint or (
                    f"This line has {info.effective_level} spaces, but needs more than {base_indent} "
                    f"to be part of the {context}."
                )
            raise self._error(message, line_no, line, hint=error_hint)
        
        return info

    def _validate_block_indent(
        self, expected_indent: int, line: str, line_no: int,
        block_start_line: int, context: str = "block"
    ) -> IndentationInfo:
        """Validate that line matches expected block indentation."""
        info = self._compute_indent_details(line)
        stripped = line.strip()
        
        if not stripped or stripped.startswith('#'):
            return info
        
        if info.mixed:
            raise self._error(
                "Mixed tabs and spaces in indentation",
                line_no, line,
                hint="Use either tabs or spaces consistently (recommended: 4 spaces)."
            )
        
        if info.effective_level != expected_indent:
            if info.effective_level < expected_indent:
                return info
            
            message = f"Inconsistent indentation in {context}"
            hint = (
                f"This line uses {info.effective_level} spaces, "
                f"but the {context} started with {expected_indent} spaces at line {block_start_line}.\n"
                f"Hint: Use consistent indentation throughout each block."
            )
            raise self._error(message, line_no, line, hint=hint)
        
        return info

    def _detect_indentation_issues(self, lines: Optional[List[str]] = None) -> Optional[str]:
        """Scan lines for common indentation problems."""
        if lines is None:
            lines = self.lines
        
        if not lines:
            return None
        
        has_tabs = has_spaces = False
        indent_levels: Set[int] = set()
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            info = self._compute_indent_details(line)
            
            if info.tabs > 0:
                has_tabs = True
            if info.spaces > 0:
                has_spaces = True
            
            if info.mixed:
                return (
                    "Mixed tabs and spaces detected. "
                    "Use either tabs or spaces consistently (recommended: 4 spaces)."
                )
            
            if info.effective_level > 0:
                indent_levels.add(info.effective_level)
        
        if has_tabs and has_spaces:
            return (
                "File mixes tab and space indentation on different lines. "
                "Use consistently (recommended: 4 spaces)."
            )
        
        if len(indent_levels) > 1:
            sorted_levels = sorted(indent_levels)
            smallest = sorted_levels[0]
            if smallest > 0:
                inconsistent = any(level % smallest != 0 for level in sorted_levels if level != smallest)
                if inconsistent:
                    return (
                        f"Inconsistent indentation increments (levels: {sorted(indent_levels)}). "
                        f"Use consistent {smallest}-space indentation."
                    )
        
        if has_tabs and not has_spaces:
            return "File uses tab indentation. Consider 4 spaces for portability."
        
        return None

    def _indent(self, line: str) -> int:
        """
        Compute indentation level (spaces count, tabs as 4).
        
        BACKWARD COMPATIBILITY: Returns effective level for legacy code.
        New code should use _compute_indent_details() for full analysis.
        """
        return self._compute_indent_details(line).effective_level
