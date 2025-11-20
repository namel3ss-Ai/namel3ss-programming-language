"""Tokenization for symbolic expression parser."""

from __future__ import annotations

import re
from typing import List, Optional


class TokenizerMixin:
    """Mixin providing tokenization for symbolic expressions."""
    
    def _tokenize(self, source: str) -> List[str]:
        """Tokenize source code into meaningful tokens."""
        # Pattern matches: identifiers, numbers, operators, parens, strings, etc
        pattern = r'(\d+\.\d+|\d+|[a-zA-Z_][a-zA-Z0-9_]*|=>|==|!=|<=|>=|->|[+\-*/%<>=!(){}[\],.:]|"[^"]*"|\'[^\']*\')'
        tokens = []
        for match in re.finditer(pattern, source):
            token = match.group(0)
            if token.strip():  # Ignore pure whitespace
                tokens.append(token)
        return tokens
