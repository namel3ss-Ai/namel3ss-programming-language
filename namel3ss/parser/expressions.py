from __future__ import annotations

from typing import Dict, List, Optional, Union

from namel3ss.ast import (
    AttributeRef,
    BinaryOp,
    CallExpression,
    ContextValue,
    Expression,
    FrameExpression,
    FrameFilter,
    FrameGroupBy,
    FrameJoin,
    FrameOrderBy,
    FrameRef,
    FrameSelect,
    FrameSummarise,
    Literal,
    NameRef,
    UnaryOp,
)

from .base import N3SyntaxError, ParserBase
# KeywordRegistry import removed - not used in this module


class ExpressionParserMixin(ParserBase):
    """
    Mixin for parsing N3 expressions using recursive descent with operator precedence.
    
    This parser handles both traditional expressions (arithmetic, logical, comparisons)
    and advanced frame-based data transformations. It integrates with the symbolic
    expression parser for functional programming constructs.
    
    Expression Types:
        Literals:
            - Numbers: 42, 3.14
            - Strings: "text", 'value'
            - Booleans: true, false
            - Null: null, None
        
        Context References:
            - ctx:user.name (context scope)
            - env:API_KEY (environment scope)
        
        Arithmetic Operations:
            - Addition/Subtraction: price + tax, total - discount
            - Multiplication/Division: quantity * price, amount / count
            - Unary negation: -value
        
        Logical Operations:
            - Conjunction: condition1 and condition2
            - Disjunction: condition1 or condition2
            - Negation: not condition
        
        Comparisons:
            - Equality: value == target, value != other
            - Ordering: age > 18, score <= 100, price >= 50, count < 10
        
        Frame Operations (Data Transformations):
            - Filter: sales.filter(revenue > 1000)
            - Select: customers.select(name, email, city)
            - Order: products.order_by(price, descending=true)
            - Group: orders.group_by(customer_id)
            - Summarize: data.summarise(total=sum(amount), avg_price=mean(price))
            - Join: orders.join(customers, on=customer_id, how=left)
        
        Function Calls:
            - sum(values), count(items), mean(scores)
            - format(value, pattern), round(number, 2)
        
        Symbolic Constructs (delegated to SymbolicExpressionParser):
            - fn: Anonymous functions
            - match: Pattern matching
            - let: Variable binding
            - if: Conditional expressions
            - query: Logic queries
    
    Operator Precedence (highest to lowest):
        1. Primary (literals, names, calls, parentheses)
        2. Unary (not, -)
        3. Multiplicative (*, /)
        4. Additive (+, -)
        5. Comparison (==, !=, <, >, <=, >=)
        6. Logical AND
        7. Logical OR
    
    Features:
        - Recursive descent parsing with proper precedence handling
        - String escape sequence support
        - Attribute access chaining (obj.attr1.attr2)
        - Function call argument parsing
        - Frame DSL method chaining for data pipelines
        - Integration with symbolic expression parser
        - Comprehensive error messages with position tracking
    """

    _expr_text: str
    _expr_pos: int

    _FRAME_DSL_METHODS = {"filter", "select", "order_by", "group_by", "summarise", "join"}

    def _parse_expression(self, text: str) -> Expression:
        """
        Parse an expression from text using recursive descent with precedence.
        
        Automatically detects symbolic constructs (fn, match, let, if, query, =>)
        and delegates to the symbolic expression parser when found. Otherwise
        uses the legacy parser for backward compatibility with standard expressions.
        
        Args:
            text: Expression text to parse
        
        Returns:
            Expression AST node
        
        Raises:
            N3SyntaxError: If expression syntax is invalid
        """
        stripped = text.strip()
        
        # Detect unsupported ternary usage early to give a clearer error
        if '?' in stripped and ':' in stripped:
            raise self._error(
                "Ternary operator ('? :') is not supported. Use an if/else block instead.",
                self.pos,
                stripped,
            )
        
        # Check if this expression uses symbolic constructs
        symbolic_keywords = ['fn', 'match', 'let', 'if', 'query', 'rule', '=>', '~']
        uses_symbolic = any(kw in stripped for kw in symbolic_keywords)
        
        if uses_symbolic:
            # Use symbolic expression parser
            from namel3ss.parser.symbolic import SymbolicExpressionParser
            from namel3ss.ast.expressions import Expression as SymbolicExpression
            
            parser = SymbolicExpressionParser(stripped)
            try:
                return parser.parse_extended_expression()
            except Exception as e:
                raise self._error(f"Failed to parse symbolic expression: {e}", self.pos, text)
        
        # Use legacy expression parser for backward compatibility
        self._expr_text = stripped
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
        
        # Check for ternary operator (not supported)
        if self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] == '?':
            raise self._error(
                "Ternary operators (? :) are not supported. Use if/else blocks instead. Example: if condition:\n    value1\nelse:\n    value2",
                self.pos,
                self._expr_text
            )
        
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
                raise self._error(
                    "Unterminated string in expression",
                    self.pos,
                    self._expr_text,
                    hint='Strings must be closed with matching quotes'
                )
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
                    raise self._error(
                        "Expected context path after prefix",
                        self.pos,
                        self._expr_text,
                        hint='Use context references like ctx:user.name or env:API_KEY'
                    )
                path = [segment for segment in path_text.split('.') if segment]
                if not path:
                    raise self._error(
                        "Context path cannot be empty",
                        self.pos,
                        self._expr_text,
                        hint='Provide a valid path after the colon, e.g., ctx:user.id'
                    )
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
            frame_chain = self._try_parse_frame_chain(name)
            if frame_chain is not None:
                return frame_chain
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
                    raise self._error(
                        "Expected attribute name after '.'",
                        self.pos,
                        self._expr_text,
                        hint='Attribute names must start with a letter or underscore'
                    )
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
                        raise self._error(
                            "Expected ',' or ')' in function call",
                            self.pos,
                            self._expr_text,
                            hint='Separate function arguments with commas'
                        )
                    current = CallExpression(function=current, arguments=args)
                    continue
                break

            return current

        raise self._error(
            f"Unexpected character in expression: '{self._expr_peek()}'",
            self.pos,
            self._expr_text,
            hint='Check for invalid characters or missing operators/operands'
        )

    def _try_parse_frame_chain(self, root_name: str) -> Optional[FrameExpression]:
        self._expr_skip_whitespace()
        if self._expr_pos >= len(self._expr_text) or self._expr_text[self._expr_pos] != '.':
            return None
        chain_start = self._expr_pos
        current: FrameExpression = FrameRef(name=root_name)
        consumed = False
        while True:
            self._expr_skip_whitespace()
            if self._expr_pos >= len(self._expr_text) or self._expr_text[self._expr_pos] != '.':
                break
            self._expr_pos += 1
            method_start = self._expr_pos
            method = self._expr_parse_identifier_token()
            if method not in self._FRAME_DSL_METHODS:
                if consumed:
                    raise self._error(
                        f"Unknown frame operation '{method}'",
                        self.pos,
                        self._expr_text,
                        hint='Valid frame operations: filter, select, order_by, group_by, summarise, join'
                    )
                self._expr_pos = chain_start
                return None
            consumed = True
            if method == 'filter':
                current = FrameFilter(source=current, predicate=self._parse_frame_predicate())
            elif method == 'select':
                current = FrameSelect(source=current, columns=self._parse_frame_column_list())
            elif method == 'order_by':
                columns, descending = self._parse_frame_order_by_args()
                current = FrameOrderBy(source=current, columns=columns, descending=descending)
            elif method == 'group_by':
                columns = self._parse_frame_column_list()
                current = FrameGroupBy(source=current, columns=columns)
            elif method == 'summarise':
                aggregations = self._parse_frame_aggregation_block()
                current = FrameSummarise(source=current, aggregations=aggregations)
            elif method == 'join':
                target, on_columns, how = self._parse_frame_join_args()
                current = FrameJoin(left=current, right=target, on=on_columns, how=how)
        if consumed:
            return current
        self._expr_pos = chain_start
        return None

    def _parse_frame_predicate(self) -> Expression:
        self._expr_skip_whitespace()
        self._expr_consume('(')
        predicate = self._parse_logical_or()
        self._expr_skip_whitespace()
        self._expr_consume(')')
        return predicate

    def _parse_frame_column_list(self) -> List[str]:
        self._expr_skip_whitespace()
        self._expr_consume('(')
        columns: List[str] = []
        self._expr_skip_whitespace()
        if self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] == ')':
            raise self._error(
                "Column list cannot be empty",
                self.pos,
                self._expr_text,
                hint='Provide at least one column name in select()'
            )
        while True:
            column = self._expr_parse_identifier_or_string()
            columns.append(column)
            self._expr_skip_whitespace()
            if self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] == ',':
                self._expr_pos += 1
                self._expr_skip_whitespace()
                continue
            break
        self._expr_consume(')')
        return columns

    def _parse_frame_order_by_args(self) -> tuple[List[str], bool]:
        self._expr_skip_whitespace()
        self._expr_consume('(')
        columns: List[str] = []
        descending = False
        parsed_any = False
        while True:
            self._expr_skip_whitespace()
            if self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] == ')':
                break
            token = self._expr_parse_identifier_or_string()
            self._expr_skip_whitespace()
            if self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] == '=':
                keyword = token
                self._expr_pos += 1
                self._expr_skip_whitespace()
                value_expr = self._parse_logical_or()
                if keyword != 'descending':
                    raise self._error(
                        "Only 'descending' keyword is supported in order_by",
                        self.pos,
                        self._expr_text,
                        hint='Use: .order_by(column1, column2, descending=true)'
                    )
                descending = self._expr_eval_bool_literal(value_expr)
            else:
                columns.append(token)
            parsed_any = True
            self._expr_skip_whitespace()
            if self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] == ',':
                self._expr_pos += 1
                continue
            break
        if not parsed_any or not columns:
            raise self._error(
                "order_by requires at least one column",
                self.pos,
                self._expr_text,
                hint='Specify columns to sort by, e.g., .order_by(date, priority)'
            )
        self._expr_consume(')')
        return columns, descending

    def _parse_frame_aggregation_block(self) -> Dict[str, Expression]:
        self._expr_skip_whitespace()
        self._expr_consume('(')
        aggregations: Dict[str, Expression] = {}
        self._expr_skip_whitespace()
        if self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] == ')':
            raise self._error(
                "summarise requires at least one aggregation",
                self.pos,
                self._expr_text,
                hint='Define aggregations like: .summarise(total=sum(amount), count=count())'
            )
        while True:
            name = self._expr_parse_identifier_token()
            self._expr_skip_whitespace()
            self._expr_consume('=')
            self._expr_skip_whitespace()
            aggregations[name] = self._parse_logical_or()
            self._expr_skip_whitespace()
            if self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] == ',':
                self._expr_pos += 1
                continue
            break
        self._expr_consume(')')
        return aggregations

    def _parse_frame_join_args(self) -> tuple[str, List[str], str]:
        self._expr_skip_whitespace()
        self._expr_consume('(')
        target = self._expr_parse_identifier_or_string()
        on_columns: List[str] = []
        how = 'inner'
        while True:
            self._expr_skip_whitespace()
            if self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] == ')':
                break
            if self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] == ',':
                self._expr_pos += 1
            self._expr_skip_whitespace()
            if self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] == ')':
                break
            keyword = self._expr_parse_identifier_token()
            self._expr_skip_whitespace()
            self._expr_consume('=')
            self._expr_skip_whitespace()
            value_expr = self._parse_logical_or()
            if keyword == 'on':
                on_columns = [self._expr_eval_identifier_like(value_expr)]
            elif keyword == 'how':
                how_value = self._expr_eval_identifier_like(value_expr)
                how_value_lower = how_value.lower()
                if how_value_lower not in {'inner', 'left', 'right', 'outer'}:
                    raise self._error(
                        "join how must be inner|left|right|outer",
                        self.pos,
                        self._expr_text,
                        hint='Valid join types: inner (default), left, right, outer'
                    )
                how = how_value_lower
            else:
                raise self._error(
                    f"Unsupported join keyword '{keyword}'",
                    self.pos,
                    self._expr_text,
                    hint='Valid join parameters: on (required), how (optional)'
                )
        if not on_columns:
            raise self._error(
                "join requires 'on' column",
                self.pos,
                self._expr_text,
                hint='Specify join column with: .join(other_table, on=column_name)'
            )
        self._expr_consume(')')
        return target, on_columns, how

    def _expr_parse_identifier_token(self) -> str:
        self._expr_skip_whitespace()
        if self._expr_pos >= len(self._expr_text):
            raise self._error("Expected identifier", self.pos, self._expr_text)
        ch = self._expr_text[self._expr_pos]
        if not (ch.isalpha() or ch == '_'):
            raise self._error("Expected identifier", self.pos, self._expr_text)
        start = self._expr_pos
        while self._expr_pos < len(self._expr_text):
            ch = self._expr_text[self._expr_pos]
            if ch.isalnum() or ch == '_':
                self._expr_pos += 1
            else:
                break
        return self._expr_text[start:self._expr_pos]

    def _expr_parse_identifier_or_string(self) -> str:
        self._expr_skip_whitespace()
        if self._expr_pos >= len(self._expr_text):
            raise self._error("Expected identifier or string literal", self.pos, self._expr_text)
        ch = self._expr_text[self._expr_pos]
        if ch in {'"', "'"}:
            return self._expr_parse_string_literal()
        if ch.isalpha() or ch == '_':
            return self._expr_parse_identifier_token()
        raise self._error("Expected identifier or string literal", self.pos, self._expr_text)

    def _expr_parse_string_literal(self) -> str:
        quote = self._expr_text[self._expr_pos]
        self._expr_pos += 1
        start = self._expr_pos
        while self._expr_pos < len(self._expr_text) and self._expr_text[self._expr_pos] != quote:
            if self._expr_text[self._expr_pos] == '\\':
                self._expr_pos += 2
            else:
                self._expr_pos += 1
        if self._expr_pos >= len(self._expr_text):
            raise self._error("Unterminated string literal", self.pos, self._expr_text)
        value = self._expr_text[start:self._expr_pos]
        self._expr_pos += 1
        return value

    def _expr_eval_bool_literal(self, expr: Expression) -> bool:
        if isinstance(expr, Literal) and isinstance(expr.value, bool):
            return expr.value
        raise self._error("Expected boolean literal", self.pos, self._expr_text)

    def _expr_eval_identifier_like(self, expr: Expression) -> str:
        if isinstance(expr, Literal) and isinstance(expr.value, str):
            return expr.value
        if isinstance(expr, NameRef):
            return expr.name
        raise self._error("Expected identifier or string literal", self.pos, self._expr_text)
