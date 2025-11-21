"""Unified error handling for N3 parser.

This module provides structured error types with:
- Line numbers and column positions
- Expected vs. found token information
- Human-readable suggestions
- Error codes for programmatic handling
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class N3Error(Exception):
    """Base class for all N3 errors."""
    
    message: str
    path: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    code: str = "N3_ERROR"
    
    def __str__(self) -> str:
        """Format error message with location."""
        parts = []
        
        if self.path:
            parts.append(f"File: {self.path}")
        
        if self.line is not None:
            if self.column is not None:
                parts.append(f"Line {self.line}:{self.column}")
            else:
                parts.append(f"Line {self.line}")
        
        parts.append(f"[{self.code}] {self.message}")
        
        return " | ".join(parts)


@dataclass
class N3SyntaxError(N3Error):
    """Syntax error with detailed context."""
    
    expected: List[str] = field(default_factory=list)
    found: Optional[str] = None
    suggestion: Optional[str] = None
    code: str = "SYNTAX_ERROR"
    
    def __str__(self) -> str:
        """Format syntax error with expectations and suggestions."""
        base = super().__str__()
        details = []
        
        if self.expected:
            if len(self.expected) == 1:
                details.append(f"Expected: {self.expected[0]}")
            else:
                details.append(f"Expected one of: {', '.join(self.expected)}")
        
        if self.found:
            details.append(f"Found: {self.found}")
        
        if self.suggestion:
            details.append(f"Suggestion: {self.suggestion}")
        
        if details:
            return base + "\n  " + "\n  ".join(details)
        
        return base


@dataclass
class N3SemanticError(N3Error):
    """Semantic validation error."""
    
    context: Optional[str] = None
    code: str = "SEMANTIC_ERROR"
    
    def __str__(self) -> str:
        """Format semantic error with context."""
        base = super().__str__()
        if self.context:
            return f"{base}\n  Context: {self.context}"
        return base


@dataclass
class N3TypeError(N3Error):
    """Type error in expressions or declarations."""
    
    expected_type: Optional[str] = None
    found_type: Optional[str] = None
    code: str = "TYPE_ERROR"
    
    def __str__(self) -> str:
        """Format type error."""
        base = super().__str__()
        details = []
        
        if self.expected_type:
            details.append(f"Expected type: {self.expected_type}")
        
        if self.found_type:
            details.append(f"Found type: {self.found_type}")
        
        if details:
            return base + "\n  " + "\n  ".join(details)
        
        return base


@dataclass
class N3IndentationError(N3SyntaxError):
    """Indentation error."""
    
    expected_indent: Optional[int] = None
    found_indent: Optional[int] = None
    code: str = "INDENTATION_ERROR"
    
    def __str__(self) -> str:
        """Format indentation error."""
        base = N3Error.__str__(self)
        details = []
        
        if self.expected_indent is not None:
            details.append(f"Expected indentation: {self.expected_indent} spaces")
        
        if self.found_indent is not None:
            details.append(f"Found indentation: {self.found_indent} spaces")
        
        if self.suggestion:
            details.append(f"Suggestion: {self.suggestion}")
        
        if details:
            return base + "\n  " + "\n  ".join(details)
        
        return base


@dataclass
class N3DuplicateDeclarationError(N3SemanticError):
    """Duplicate declaration error."""
    
    name: Optional[str] = None
    first_line: Optional[int] = None
    code: str = "DUPLICATE_DECLARATION"
    
    def __str__(self) -> str:
        """Format duplicate declaration error."""
        base = N3Error.__str__(self)
        details = []
        
        if self.name:
            details.append(f"Duplicate name: {self.name}")
        
        if self.first_line:
            details.append(f"First declared at line: {self.first_line}")
        
        if details:
            return base + "\n  " + "\n  ".join(details)
        
        return base


@dataclass
class N3ReferenceError(N3SemanticError):
    """Undefined reference error."""
    
    name: Optional[str] = None
    available: List[str] = field(default_factory=list)
    code: str = "REFERENCE_ERROR"
    
    def __str__(self) -> str:
        """Format reference error with available names."""
        base = N3Error.__str__(self)
        details = []
        
        if self.name:
            details.append(f"Undefined: {self.name}")
        
        if self.available:
            # Find similar names for suggestion
            similar = self._find_similar(self.name, self.available) if self.name else []
            if similar:
                details.append(f"Did you mean: {', '.join(similar[:3])}?")
            else:
                details.append(f"Available names: {', '.join(self.available[:5])}")
        
        if details:
            return base + "\n  " + "\n  ".join(details)
        
        return base
    
    @staticmethod
    def _find_similar(target: str, candidates: List[str], max_distance: int = 2) -> List[str]:
        """Find similar names using Levenshtein distance."""
        def levenshtein(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]
        
        similar = [
            (name, levenshtein(target.lower(), name.lower()))
            for name in candidates
        ]
        similar.sort(key=lambda x: x[1])
        return [name for name, dist in similar if dist <= max_distance]


def create_syntax_error(
    message: str,
    *,
    path: Optional[str] = None,
    line: Optional[int] = None,
    column: Optional[int] = None,
    expected: Optional[List[str]] = None,
    found: Optional[str] = None,
    suggestion: Optional[str] = None,
) -> N3SyntaxError:
    """Create a syntax error with context."""
    return N3SyntaxError(
        message=message,
        path=path,
        line=line,
        column=column,
        expected=expected or [],
        found=found,
        suggestion=suggestion,
    )


def create_semantic_error(
    message: str,
    *,
    path: Optional[str] = None,
    line: Optional[int] = None,
    context: Optional[str] = None,
) -> N3SemanticError:
    """Create a semantic error with context."""
    return N3SemanticError(
        message=message,
        path=path,
        line=line,
        context=context,
    )


def create_type_error(
    message: str,
    *,
    path: Optional[str] = None,
    line: Optional[int] = None,
    expected_type: Optional[str] = None,
    found_type: Optional[str] = None,
) -> N3TypeError:
    """Create a type error."""
    return N3TypeError(
        message=message,
        path=path,
        line=line,
        expected_type=expected_type,
        found_type=found_type,
    )


__all__ = [
    "N3Error",
    "N3SyntaxError",
    "N3SemanticError",
    "N3TypeError",
    "N3IndentationError",
    "N3DuplicateDeclarationError",
    "N3ReferenceError",
    "create_syntax_error",
    "create_semantic_error",
    "create_type_error",
]
