"""Context reference parsing."""

from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...ast import ContextValue


class ContextMixin:
    """Mixin for parsing context references."""
    
    def _parse_context_reference(self, token: str) -> Optional["ContextValue"]:
        """Parse context reference like 'ctx:variable' or 'env:VAR'."""
        from namel3ss.ast import ContextValue
        
        match = re.match(r'^(ctx|env):([A-Za-z0-9_\.]+)$', token)
        if not match:
            return None
        
        scope = match.group(1)
        path_text = match.group(2)
        if not path_text:
            return None
        
        path = [segment for segment in path_text.split('.') if segment]
        if not path:
            return None
        
        return ContextValue(scope=scope, path=path)
