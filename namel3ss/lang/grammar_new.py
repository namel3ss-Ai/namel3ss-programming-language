"""Backward compatibility wrapper for grammar module.

The grammar parser has been refactored into a modular package structure:
    namel3ss/lang/grammar/
        __init__.py - Package composition
        constants.py - Regex patterns (43 lines)
        helpers.py - Helper classes (20 lines)
        directives.py - Module/import parsing (95 lines)
        declarations.py - App/theme/dataset/frame parsing (185 lines)
        pages.py - Page statement parsing (180 lines)
        ai_components.py - LLM/tool parsing (243 lines)
        prompts.py - Prompt parsing (384 lines)
        rag.py - RAG parsing (143 lines)
        agents.py - Agent/graph parsing (146 lines)
        policy.py - Policy parsing (84 lines)
        utility_parsers.py - Utility parsing methods (224 lines)
        functions.py - Function/rule parsing (56 lines)
        utility_methods.py - Core utilities (88 lines)
        parser.py - Main parser class (261 lines)

Original file: 1,993 lines monolithic
New structure: 14 modules, ~2,152 lines total with module headers
Wrapper reduction: 1,993 → 62 lines (97% reduction)
"""

from __future__ import annotations

from typing import Optional

from namel3ss.ast.program import Module

# Re-export main classes for backward compatibility
from .grammar import GrammarUnsupportedError, _GrammarModuleParser


def parse_module(source: str, path: str = "", module_name: Optional[str] = None) -> Module:
    """Parse *source* using the grammar-backed parser and return a Module AST."""
    
    parser = _GrammarModuleParser(source=source, path=path, module_name=module_name)
    try:
        return parser.parse()
    except GrammarUnsupportedError:
        from namel3ss.parser.program import LegacyProgramParser
        
        legacy = LegacyProgramParser(source, module_name=module_name, path=path)
        return legacy.parse()


__all__ = ['parse_module', 'GrammarUnsupportedError', '_GrammarModuleParser']
