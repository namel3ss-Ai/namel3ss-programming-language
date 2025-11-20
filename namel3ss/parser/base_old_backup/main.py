"""Main parser base composition."""

from __future__ import annotations

from typing import Dict, List, Optional

from .parsing import ParsingMixin
from .indentation import IndentationMixin, IndentationInfo
from .errors import ErrorMixin
from .coercion import CoercionMixin
from .context import ContextMixin
from .utilities import UtilitiesMixin


class ParserBase(
    ParsingMixin,
    IndentationMixin,
    ErrorMixin,
    CoercionMixin,
    ContextMixin,
    UtilitiesMixin,
):
    """
    Base parser with core functionality.
    
    Provides fundamental parsing operations including:
    - Cursor management (peek, advance)
    - Indentation analysis and validation
    - Error handling with diagnostics
    - Value coercion (bool, int, scalar, expression)
    - Context reference parsing (ctx:, env:)
    - Policy builders (cache, pagination, streaming)
    
    This is a modular refactored version maintaining 100% backward compatibility.
    """
    
    def __init__(self, source: str, *, module_name: Optional[str] = None, path: str = ""):
        """Initialize parser with source code."""
        from ...ast import Import
        
        self.lines: List[str] = source.splitlines()
        self.pos: int = 0
        self.app = None
        self._loop_depth: int = 0
        self._module_name_override: Optional[str] = module_name
        self.module_name: Optional[str] = module_name
        self.module_imports: List[Import] = []
        self._import_aliases: Dict[str, str] = {}
        self._named_imports: Dict[str, str] = {}
        self._module_declared = False
        self._explicit_app_declared = False
        self.language_version: Optional[str] = None
        self._allow_implicit_app = module_name is not None
        self.source_path: str = path


# Re-export IndentationInfo for backward compatibility
__all__ = ['ParserBase', 'IndentationInfo']
