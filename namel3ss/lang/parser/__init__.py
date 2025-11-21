"""Unified N3 parser package.

This is the single, canonical parser for Namel3ss language. It replaces all
legacy parser mechanisms with a clean, deterministic recursive descent parser.

Public API:
    parse_module(source, path, module_name) -> Module
    N3Parser - The main parser class
    
Error types:
    N3Error, N3SyntaxError, N3SemanticError, N3TypeError
    N3IndentationError, N3DuplicateDeclarationError, N3ReferenceError
"""

from typing import Optional
from namel3ss.ast.program import Module

from .parse import N3Parser
from .errors import (
    N3Error,
    N3SyntaxError,
    N3SemanticError,
    N3TypeError,
    N3IndentationError,
    N3DuplicateDeclarationError,
    N3ReferenceError,
)


def parse_module(source: str, path: str = "", module_name: Optional[str] = None) -> Module:
    """
    Parse N3 source code into a Module AST.
    
    This is the ONLY entry point for parsing N3 code. No fallback mechanisms,
    no dual-parser logic. All code must go through this single parser.
    
    Args:
        source: N3 source code to parse
        path: Optional file path for error reporting
        module_name: Optional module name override
    
    Returns:
        Module AST node
    
    Raises:
        N3SyntaxError: If source has syntax errors
        N3SemanticError: If source has semantic errors
        N3TypeError: If source has type errors
    
    Example:
        ```python
        source = '''
        app "My App" {
          description: "A sample app"
        }
        '''
        
        module = parse_module(source)
        print(module.name)  # "My App"
        ```
    """
    parser = N3Parser(source, path=path, module_name=module_name)
    return parser.parse()


__all__ = [
    "parse_module",
    "N3Parser",
    "N3Error",
    "N3SyntaxError",
    "N3SemanticError",
    "N3TypeError",
    "N3IndentationError",
    "N3DuplicateDeclarationError",
    "N3ReferenceError",
]
