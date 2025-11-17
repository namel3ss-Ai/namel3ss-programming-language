from __future__ import annotations

from typing import List, Optional, Union

from namel3ss.ast import (
    AttributeRef,
    BinaryOp,
    CallExpression,
    ContextValue,
    Expression,
    Literal,
    NameRef,
    UnaryOp,
)

from .base import N3SyntaxError, ParserBase


class ExpressionParserMixin(ParserBase):
    """Expression parsing utilities."""

    _expr_text: str
    _expr_pos: int

    def _parse_expression(self, text: str) -> Expression:
        """Parse an expression from text using recursive descent with precedence."""
        self._expr_text = text.strip()
        self._expr_pos = 0
        result = self._parse_logical_or()
        self._expr_skip_whitespace()
        if self._expr_pos != len(self._expr_text):
            raise self._error(
                "Unexpected trailing characters in expression",
                self.pos,
                self._expr_text,
            )
        return result

    def _expr_peek(self) -> Optional[str]:
        if self._expr_pos < len(self._expr_text):
            return self._expr_text[self._expr_pos]
        return None

    def _expr_skip_whitespace(self) -> None:
        while self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos].isspace():
            self._expr_pos += 1

    def _expr_match(self, token: str) -> bool:
        self._expr_skip_whitespace()
        if self._expr_text[self._expr_pos : self._expr_pos + len(token)] == token:
            if self._expr_pos + len(token) < len(self._expr_text):
                next_char = self._expr_text[self._expr_pos + len(token)]
                if token.isalnum() and (next_char.isalnum() or next_char == '_'):
                    return False
            self._expr_pos += len(token)
            return True
        return False

    def _expr_consume(self, token: str) -> None:
        if not self._expr_match(token):
            raise self._error(f"Expected '{token}' in expression", self.pos, self._expr_text)

    def _parse_logical_or(self) -> Expression:
        left = self._parse_logical_and()
        while self._expr_match('or'):
            right = self._parse_logical_and()
            left = BinaryOp(left=left, op='or', right=right)
        return left

    def _parse_logical_and(self) -> Expression:
        left = self._parse_comparison()
        while self._expr_match('and'):
            right = self._parse_comparison()
            left = BinaryOp(left=left, op='and', right=right)
        return left

    def _parse_comparison(self) -> Expression:
        left = self._parse_additive()
        self._expr_skip_whitespace()
        if self._expr_match('=='):
            right = self._parse_additive()
            return BinaryOp(left=left, op='==', right=right)
        if self._expr_match('!='):
            right = self._parse_additive()
            return BinaryOp(left=left, op='!=', right=right)
        if self._expr_match('>='):
            right = self._parse_additive()
            return BinaryOp(left=left, op='>=', right=right)
        if self._expr_match('<='):
            right = self._parse_additive()
            return BinaryOp(left=left, op='<=', right=right)
        if self._expr_match('>'):
            right = self._parse_additive()
            return BinaryOp(left=left, op='>', right=right)
        if self._expr_match('<'):
            right = self._parse_additive()
            return BinaryOp(left=left, op='<', right=right)
        return left

    def _parse_additive(self) -> Expression:
        left = self._parse_multiplicative()
        while True:
            if self._expr_match('+'):
                right = self._parse_multiplicative()
                left = BinaryOp(left=left, op='+', right=right)
            elif self._expr_match('-'):
                right = self._parse_multiplicative()
                left = BinaryOp(left=left, op='-', right=right)
            else:
                break
        return left

    def _parse_multiplicative(self) -> Expression:
        left = self._parse_unary()
        while True:
            if self._expr_match('*'):
                right = self._parse_unary()
                left = BinaryOp(left=left, op='*', right=right)
            elif self._expr_match('/'):
                right = self._parse_unary()
                left = BinaryOp(left=left, op='/', right=right)
            else:
                break
        return left

    def _parse_unary(self) -> Expression:
        self._expr_skip_whitespace()
        if self._expr_match('not'):
            operand = self._parse_unary()
            return UnaryOp(op='not', operand=operand)
        if self._expr_match('-'):
            operand = self._parse_unary()
            return UnaryOp(op='-', operand=operand)
        return self._parse_primary()

    def _parse_primary(self) -> Expression:
        self._expr_skip_whitespace()
        if self._expr_match('('):
            expr = self._parse_logical_or()
            self._expr_consume(')')
            return expr

        if self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] in ('"', "'"):
            quote = self._expr_text[self._expr_pos]
            self._expr_pos += 1
            start = self._expr_pos
            while self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] != quote:
                if self._expr_text[self._expr_pos] == '\\':
                    self._expr_pos += 2
                else:
                    self._expr_pos += 1
            if self._expr_pos >= len(self._expr_text):
                raise self._error("Unterminated string in expression", self.pos, self._expr_text)
            value = self._expr_text[start:self._expr_pos]
            self._expr_pos += 1
            return Literal(value=value)

        if self._expr_match('true'):
            return Literal(value=True)
        if self._expr_match('false'):
            return Literal(value=False)
        if self._expr_match('null') or self._expr_match('None'):
            return Literal(value=None)

        for prefix in ('ctx:', 'env:'):
            if self._expr_text.startswith(prefix, self._expr_pos):
                scope = prefix[:-1]
                self._expr_pos += len(prefix)
                start = self._expr_pos
                while self._expr_pos < len(self._expr_text):
                    ch = self._expr_text[self._expr_pos]
                    if ch.isalnum() or ch in {'_', '.'}:
                        self._expr_pos += 1
                    else:
                        break
                path_text = self._expr_text[start:self._expr_pos]
                if not path_text:
                    raise self._error("Expected context path after prefix", self.pos, self._expr_text)
                path = [segment for segment in path_text.split('.') if segment]
                if not path:
                    raise self._error("Context path cannot be empty", self.pos, self._expr_text)
                return ContextValue(scope=scope, path=path)

        if self._expr_pos < len(self._expr_text) and (
            self._expr_text[self._expr_pos].isdigit()
            or (
                self._expr_text[self._expr_pos] == '.'
                and self._expr_pos + 1 < len(self._expr_text)
                and self._expr_text[self._expr_pos + 1].isdigit()
            )
        ):
            start = self._expr_pos
            has_dot = False
            while self._expr_pos < len(self._expr_text):
                ch = self._expr_text[self._expr_pos]
                if ch.isdigit():
                    self._expr_pos += 1
                elif ch == '.' and not has_dot:
                    has_dot = True
                    self._expr_pos += 1
                else:
                    break
            num_str = self._expr_text[start:self._expr_pos]
            value = float(num_str) if has_dot else int(num_str)
            return Literal(value=value)

        if self._expr_pos < len(self._expr_text) and (
            self._expr_text[self._expr_pos].isalpha()
            or self._expr_text[self._expr_pos] == '_'
        ):
            start = self._expr_pos
            while self._expr_pos < len(self._expr_text):
                ch = self._expr_text[self._expr_pos]
                if ch.isalnum() or ch == '_':
                    self._expr_pos += 1
                else:
                    break
            name = self._expr_text[start:self._expr_pos]
            current: Union[NameRef, AttributeRef] = NameRef(name=name)

            while True:
                self._expr_skip_whitespace()
                if self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] == '.':
                    self._expr_pos += 1
                    self._expr_skip_whitespace()
                    attr_start = self._expr_pos
                    if self._expr_pos < len(self._expr_text) and (
                        self._expr_text[self._expr_pos].isalpha()
                        or self._expr_text[self._expr_pos] == '_'
                    ):
                        while self._expr_pos < len(self._expr_text):
                            ch = self._expr_text[self._expr_pos]
                            if ch.isalnum() or ch == '_':
                                self._expr_pos += 1
                            else:
                                break
                        attr = self._expr_text[attr_start:self._expr_pos]
                        base_name = current.name if isinstance(current, NameRef) else f"{current.base}.{current.attr}"
                        current = AttributeRef(base=base_name, attr=attr)
                        continue
                    raise self._error("Expected attribute name after '.'", self.pos, self._expr_text)
                break

            while True:
                self._expr_skip_whitespace()
                if self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] == '(':
                    self._expr_pos += 1
                    args: List[Expression] = []
                    self._expr_skip_whitespace()
                    if self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] == ')':
                        self._expr_pos += 1
                        current = CallExpression(function=current, arguments=args)
                        continue
                    while True:
                        arg = self._parse_logical_or()
                        args.append(arg)
                        self._expr_skip_whitespace()
                        if (
                            self._expr_pos < len(self._expr_text)
                            and self._expr_text[self._expr_pos] == ','
                        ):
                            self._expr_pos += 1
                            continue
                        if (
                            self._expr_pos < len(self._expr_text)
                            and self._expr_text[self._expr_pos] == ')'
                        ):
                            self._expr_pos += 1
                            break
                        raise self._error("Expected ',' or ')' in function call", self.pos, self._expr_text)
                    current = CallExpression(function=current, arguments=args)
                    continue
                break

            return current

        raise self._error(
            f"Unexpected character in expression: '{self._expr_peek()}'",
            self.pos,
            self._expr_text,
        )
