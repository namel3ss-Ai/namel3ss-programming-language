"""Grammar package for Namel3ss language parsing.

This package provides a modular grammar parser organized into focused components:
- constants: Regular expression patterns for grammar matching
- helpers: Helper classes (_Line, GrammarUnsupportedError)
- directives: Module, import, and language_version parsing
- declarations: App, theme, dataset, frame parsing
- pages: Page and page statement parsing with control flow
- ai_components: LLM, tool, and AI wrapper parsing
- prompts: Structured prompt parsing with args and output schema
- rag: RAG index and pipeline parsing
- agents: Agent and graph parsing
- policy: Policy parsing
- utility_parsers: KV blocks, lists, and schema parsing utilities
- functions: Function and rule definition parsing
- utility_methods: Core utility methods (_ensure_app, _indent, _error, etc.)
- parser: Main _GrammarModuleParser class composing all mixins
"""

from typing import Optional
from namel3ss.ast.program import Module

from .helpers import GrammarUnsupportedError, _Line
from .parser import _GrammarModuleParser


def parse_module(source: str, path: str = "", module_name: Optional[str] = None) -> Module:
    """Parse *source* using the grammar-backed parser and return a Module AST."""
    
    parser = _GrammarModuleParser(source=source, path=path, module_name=module_name)
    try:
        return parser.parse()
    except GrammarUnsupportedError:
        from namel3ss.parser.program import LegacyProgramParser
        
        legacy = LegacyProgramParser(source, module_name=module_name, path=path)
        return legacy.parse()


__all__ = ['parse_module', 'GrammarUnsupportedError', '_Line', '_GrammarModuleParser']
