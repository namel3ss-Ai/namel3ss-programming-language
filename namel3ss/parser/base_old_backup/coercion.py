"""Value coercion utilities."""

from __future__ import annotations

import ast
import re
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...ast import Expression, ContextValue, Literal


class CoercionMixin:
    """Mixin providing value coercion utilities."""
    
    def _parse_bool(self, raw: str) -> bool:
        """Parse a boolean-like string into a bool."""
        value = raw.strip().lower()
        if value in {"true", "yes", "1", "on"}:
            return True
        if value in {"false", "no", "0", "off"}:
            return False
        raise self._error(f"Expected boolean value, found '{raw}'")

    def _coerce_scalar(self, raw: str) -> Any:
        """Attempt to coerce a scalar configuration value."""
        text = raw.strip()
        if not text:
            return text

        context_ref = self._parse_context_reference(text)
        if context_ref is not None:
            return context_ref
        
        lower = text.lower()
        if lower in {"true", "false", "null", "none"}:
            if lower in {"true", "false"}:
                return self._parse_bool(text)
            return None
        
        if re.fullmatch(r"[-+]?\d+", text):
            try:
                return int(text)
            except ValueError:
                pass
        
        if re.fullmatch(r"[-+]?\d*\.\d+", text):
            try:
                return float(text)
            except ValueError:
                pass
        
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            inner = text[1:-1]
            context_ref_inner = self._parse_context_reference(inner)
            if context_ref_inner is not None:
                return context_ref_inner
            return inner
        
        if text.startswith('[') or text.startswith('{') or text.startswith('('):
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                pass
        
        return text

    def _coerce_expression(self, value: Any) -> "Expression":
        """Coerce value to Expression AST node."""
        from namel3ss.ast import Expression, Literal
        
        if isinstance(value, Expression):
            return value
        if isinstance(value, str):
            try:
                return self._parse_expression(value)
            except Exception:
                return Literal(value)
        return Literal(value)

    def _coerce_int(self, value: Any) -> Optional[int]:
        """Coerce value to integer."""
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        try:
            text = str(value).strip()
            if not text:
                return None
            match = re.match(r"(-?\d+)", text)
            if not match:
                return None
            return int(match.group(1))
        except (ValueError, TypeError):
            return None

    def _stringify_value(self, value: Any) -> str:
        """Convert value to string representation."""
        from namel3ss.ast import ContextValue, Literal, Expression
        
        if value is None:
            return ''
        if isinstance(value, ContextValue):
            path = '.'.join(value.path)
            return f"{value.scope}:{path}" if path else value.scope
        if isinstance(value, Literal):
            inner = value.value
            return '' if inner is None else str(inner)
        if isinstance(value, Expression):
            return str(value)
        return str(value)
    
    def _strip_quotes(self, value: str) -> str:
        """Remove surrounding quotes from a string."""
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        return value
