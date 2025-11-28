"""Grammar package for Namel3ss language parsing.

DEPRECATED: This module now redirects to the unified parser.
The new canonical parser is in namel3ss.lang.parser.

This backward compatibility layer will be removed in a future version.
"""

from typing import Optional
from namel3ss.ast.program import Module

# Import from the new unified parser
from namel3ss.lang.parser import parse_module as unified_parse_module
from namel3ss.lang.parser import N3SyntaxError as GrammarUnsupportedError

# Legacy fallback for logic-first syntax (knowledge/query blocks with colons)
from namel3ss.parser.program import LegacyProgramParser
from namel3ss.lang.parser import N3SyntaxError as ModernSyntaxError
from namel3ss.errors import N3SyntaxError as LegacySyntaxError


def parse_module(source: str, path: str = "", module_name: Optional[str] = None) -> Module:
    """
    Parse *source* using the unified parser and return a Module AST.
    
    This function now uses the canonical unified parser. The legacy fallback
    mechanism has been removed to ensure deterministic parsing behavior.
    """
    try:
        return unified_parse_module(source, path=path, module_name=module_name)
    except ModernSyntaxError as modern_error:
        # Maintain backward compatibility for colon-indented logic syntax
        legacy_parser = LegacyProgramParser(source, module_name=module_name, path=path)
        try:
            return legacy_parser.parse()
        except Exception as legacy_error:
            # If the legacy parser produced a syntax error, surface the more
            # specific message; otherwise, preserve the modern parser failure.
            if isinstance(legacy_error, (ModernSyntaxError, LegacySyntaxError)):
                raise legacy_error
            raise modern_error


__all__ = ['parse_module', 'GrammarUnsupportedError']
