"""Function and rule definition parsing."""

from __future__ import annotations
import re
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Tuple

if TYPE_CHECKING:
    from .helpers import _Line

from namel3ss.ast import App


class FunctionsParserMixin:
    """Mixin providing function and rule definition parsing."""

    def _parse_function_def(self, line: _Line) -> None:
        """Parse function definition: fn name(params) => body"""
        from namel3ss.parser.symbolic import SymbolicExpressionParser
        from namel3ss.ast.expressions import FunctionDef
        
        # For now, only handle single-line function definitions
        func_text = line.text
        
        # Create symbolic parser
        parser = SymbolicExpressionParser(func_text, path=self._path)
        
        try:
            func_def = parser.parse_function_def()
            
            # Functions are attached to the active app; create a default if needed.
            self._ensure_app(line)
            
            # Add function to app functions collection
            if self._app:
                self._app.functions.append(func_def)
            self._extra_nodes.append(func_def)

            # Move past this line so the main loop can continue
            self._advance()
            
        except Exception as e:
            raise self._error(f"Failed to parse function definition: {e}", line)
    
    def _parse_rule_def(self, line: _Line) -> None:
        """Parse rule definition: rule head :- body."""
        from namel3ss.parser.symbolic import SymbolicExpressionParser
        from namel3ss.ast.expressions import RuleDef
        
        # For now, only handle single-line rule definitions
        rule_text = line.text
        
        # Create symbolic parser
        parser = SymbolicExpressionParser(rule_text, path=self._path)
        
        try:
            rule_def = parser.parse_rule_def()
            
            # Ensure app exists
            if self._app is None:
                self._app = App(name="", body=[])
            
            # Add rule to app rules collection
            self._app.rules.append(rule_def)
            self._extra_nodes.append(rule_def)
            
        except Exception as e:
            raise self._error(f"Failed to parse rule definition: {e}", line)

    def _unsupported(self, line: _Line, feature: str) -> None:
        location = f"{self._path}:{line.number}" if self._path else f"line {line.number}"
        raise GrammarUnsupportedError(f"Unsupported {feature} near {location}")

__all__ = ['FunctionsParserMixin']
