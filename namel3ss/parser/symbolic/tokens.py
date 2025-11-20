"""Token operations for symbolic expression parser."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...errors import N3SyntaxError


class TokenOperationsMixin:
    """Mixin providing token manipulation operations."""
    
    def current_token(self) -> Optional[str]:
        """Get current token without consuming it."""
        if self.token_pos < len(self.tokens):
            return self.tokens[self.token_pos]
        return None
    
    def peek(self) -> Optional[str]:
        """Peek at current token."""
        return self.current_token()
    
    def consume(self) -> str:
        """Consume and return current token."""
        from namel3ss.errors import N3SyntaxError
        
        if self.token_pos >= len(self.tokens):
            raise N3SyntaxError(
                "Unexpected end of input",
                path=None,
                line=self.pos,
                hint="Check for incomplete expressions or missing closing brackets"
            )
        token = self.tokens[self.token_pos]
        self.token_pos += 1
        return token
    
    def expect(self, expected: str) -> None:
        """Expect a specific token."""
        from namel3ss.errors import N3SyntaxError
        
        token = self.consume()
        if token != expected:
            raise N3SyntaxError(
                f"Expected '{expected}' but got '{token}'",
                path=None,
                line=self.pos,
                hint=f"Ensure proper syntax for the current construct"
            )
    
    def try_consume(self, expected: str) -> bool:
        """Try to consume a token, return True if successful."""
        if self.current_token() == expected:
            self.consume()
            return True
        return False
    
    def word(self) -> str:
        """Parse an identifier/word."""
        from namel3ss.errors import N3SyntaxError
        
        token = self.consume()
        if not token or not (token[0].isalpha() or token[0] == '_'):
            raise N3SyntaxError(
                f"Expected identifier but got '{token}'",
                path=None,
                line=self.pos,
                hint="Identifiers must start with a letter or underscore"
            )
        return token
    
    def peek_word(self) -> Optional[str]:
        """Peek at current token if it's a word/identifier."""
        token = self.current_token()
        if token and (token[0].isalpha() or token[0] == '_'):
            return token
        return None
    
    def string(self) -> str:
        """Parse a string literal."""
        from namel3ss.errors import N3SyntaxError
        
        token = self.consume()
        if not token or not (token.startswith('"') or token.startswith("'")):
            raise N3SyntaxError(
                f"Expected string but got '{token}'",
                path=None,
                line=self.pos,
                hint='String literals must be enclosed in quotes'
            )
        # Remove quotes
        return token[1:-1] if len(token) >= 2 else token
    
    def number(self):
        """Parse a number literal."""
        from namel3ss.errors import N3SyntaxError
        
        token = self.consume()
        try:
            if '.' in token:
                return float(token)
            return int(token)
        except ValueError:
            raise N3SyntaxError(
                f"Expected number but got '{token}'",
                path=None,
                line=self.pos,
                hint='Numeric literals must contain only digits and optionally a decimal point'
            )
