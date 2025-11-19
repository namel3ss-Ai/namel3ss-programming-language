"""Extended expression parser for symbolic programming constructs."""

from __future__ import annotations

from typing import List, Optional, Tuple

from namel3ss.ast.expressions import (
    CallExpr,
    ConstructorPattern,
    DictExpr,
    DictPattern,
    FunctionDef,
    IfExpr,
    IndexExpr,
    LambdaExpr,
    LetExpr,
    ListExpr,
    ListPattern,
    LiteralExpr,
    LiteralPattern,
    MatchCase,
    MatchExpr,
    Parameter,
    Pattern,
    QueryExpr,
    RuleBody,
    RuleClause,
    RuleDef,
    RuleHead,
    SliceExpr,
    TupleExpr,
    TuplePattern,
    UnifyExpr,
    VarExpr,
    VarPattern,
    WildcardPattern,
)
from namel3ss.ast.base import Expression

from .base import N3SyntaxError, ParserBase
import re


class SymbolicExpressionParser(ParserBase):
    """Parser for symbolic expressions, functions, pattern matching, and rules."""
    
    def __init__(self, source: str, *, path: str = ""):
        super().__init__(source, path=path)
        self.tokens = self._tokenize(source)
        self.token_pos = 0
    
    def _tokenize(self, source: str) -> list:
        """Tokenize source code into meaningful tokens."""
        # Pattern matches: identifiers, numbers, operators, parens, strings, etc
        pattern = r'(\d+\.\d+|\d+|[a-zA-Z_][a-zA-Z0-9_]*|=>|==|!=|<=|>=|->|[+\-*/%<>=!(){}[\],.:]|"[^"]*"|\'[^\']*\')'
        tokens = []
        for match in re.finditer(pattern, source):
            token = match.group(0)
            if token.strip():  # Ignore pure whitespace
                tokens.append(token)
        return tokens
    
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
        if self.token_pos >= len(self.tokens):
            raise N3SyntaxError("Unexpected end of input", path=None, line=self.pos)
        token = self.tokens[self.token_pos]
        self.token_pos += 1
        return token
    
    def expect(self, expected: str) -> None:
        """Expect a specific token."""
        token = self.consume()
        if token != expected:
            raise N3SyntaxError(
                f"Expected '{expected}' but got '{token}'",
                path=None,
                line=self.pos
            )
    
    def try_consume(self, expected: str) -> bool:
        """Try to consume a token, return True if successful."""
        if self.current_token() == expected:
            self.consume()
            return True
        return False
    
    def word(self) -> str:
        """Parse an identifier/word."""
        token = self.consume()
        if not token or not (token[0].isalpha() or token[0] == '_'):
            raise N3SyntaxError(
                f"Expected identifier but got '{token}'",
                path=None,
                line=self.pos
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
        token = self.consume()
        if not token or not (token.startswith('"') or token.startswith("'")):
            raise N3SyntaxError(f"Expected string but got '{token}'", path=None, line=self.pos)
        # Remove quotes
        return token[1:-1] if len(token) >= 2 else token
    
    def number(self):
        """Parse a number literal."""
        token = self.consume()
        try:
            if '.' in token:
                return float(token)
            return int(token)
        except ValueError:
            raise N3SyntaxError(f"Expected number but got '{token}'", path=None, line=self.pos)

    def parse_function_def(self) -> FunctionDef:
        """Parse: fn name(params) { body } or fn name(params) => expr"""
        self.expect("fn")
        name = self.word()
        self.expect("(")
        
        params: List[Parameter] = []
        while not self.try_consume(")"):
            if params:
                self.expect(",")
            param_name = self.word()
            default = None
            type_hint = None
            
            if self.try_consume(":"):
                type_hint = self.word()
            
            if self.try_consume("="):
                default = self.parse_extended_expression()
            
            params.append(Parameter(name=param_name, default=default, type_hint=type_hint))
        
        # Check for arrow function or block
        if self.try_consume("=>"):
            body = self.parse_extended_expression()
        else:
            self.expect("{")
            body = self.parse_extended_expression()
            self.expect("}")
        
        return FunctionDef(name=name, params=params, body=body)
    
    def parse_lambda(self) -> LambdaExpr:
        """Parse: fn(params) => expr or fn(params) { body }"""
        self.expect("fn")
        self.expect("(")
        
        params: List[Parameter] = []
        while not self.try_consume(")"):
            if params:
                self.expect(",")
            param_name = self.word()
            default = None
            type_hint = None
            
            if self.try_consume(":"):
                type_hint = self.word()
            
            if self.try_consume("="):
                default = self.parse_extended_expression()
            
            params.append(Parameter(name=param_name, default=default, type_hint=type_hint))
        
        if self.try_consume("=>"):
            body = self.parse_extended_expression()
        else:
            self.expect("{")
            body = self.parse_extended_expression()
            self.expect("}")
        
        return LambdaExpr(params=params, body=body)
    
    def parse_if_expr(self) -> IfExpr:
        """Parse: if cond then_expr else else_expr"""
        self.expect("if")
        condition = self.parse_extended_expression()
        
        self.expect("then")
        then_expr = self.parse_extended_expression()
        
        else_expr = None
        if self.try_consume("else"):
            else_expr = self.parse_extended_expression()
        
        return IfExpr(condition=condition, then_expr=then_expr, else_expr=else_expr)
    
    def parse_let_expr(self) -> LetExpr:
        """Parse: let x = value, y = value2 in body"""
        self.expect("let")
        
        bindings: List[Tuple[str, Expression]] = []
        while True:
            var_name = self.word()
            self.expect("=")
            value = self.parse_extended_expression()
            bindings.append((var_name, value))
            
            if not self.try_consume(","):
                break
        
        self.expect("in")
        body = self.parse_extended_expression()
        
        return LetExpr(bindings=bindings, body=body)
    
    def parse_match_expr(self) -> MatchExpr:
        """Parse: match expr { case pat => body, ... }"""
        self.expect("match")
        expr = self.parse_extended_expression()
        
        self.expect("{")
        cases: List[MatchCase] = []
        
        while not self.try_consume("}"):
            if cases:
                self.expect(",")
            
            self.expect("case")
            pattern = self.parse_pattern()
            
            guard = None
            if self.try_consume("if"):
                guard = self.parse_extended_expression()
            
            self.expect("=>")
            body = self.parse_extended_expression()
            
            cases.append(MatchCase(pattern=pattern, guard=guard, body=body))
        
        return MatchExpr(expr=expr, cases=cases)
    
    def parse_pattern(self) -> Pattern:
        """Parse a pattern for pattern matching."""
        # Wildcard
        if self.try_consume("_"):
            return WildcardPattern()
        
        # List pattern
        if self.try_consume("["):
            elements: List[Pattern] = []
            rest_var = None
            
            while not self.try_consume("]"):
                if elements:
                    self.expect(",")
                
                # Check for rest pattern: ...rest
                if self.try_consume("..."):
                    rest_var = self.word()
                    self.expect("]")
                    break
                
                elements.append(self.parse_pattern())
            
            return ListPattern(elements=elements, rest_var=rest_var)
        
        # Dict pattern
        if self.try_consume("{"):
            pairs: List[Tuple[str, Pattern]] = []
            rest_var = None
            
            while not self.try_consume("}"):
                if pairs:
                    self.expect(",")
                
                # Check for rest pattern
                if self.try_consume("..."):
                    rest_var = self.word()
                    self.expect("}")
                    break
                
                key = self.word()
                self.expect(":")
                value_pattern = self.parse_pattern()
                pairs.append((key, value_pattern))
            
            return DictPattern(pairs=pairs, rest_var=rest_var)
        
        # Tuple pattern
        if self.try_consume("("):
            elements: List[Pattern] = []
            
            while not self.try_consume(")"):
                if elements:
                    self.expect(",")
                elements.append(self.parse_pattern())
            
            # Single element in parens is not a tuple
            if len(elements) == 1:
                return elements[0]
            
            return TuplePattern(elements=elements)
        
        # Literal or variable or constructor
        # Try to parse as literal first
        if self.peek() and self.peek() in ('"', "'"):
            value = self.string()
            return LiteralPattern(value=value)
        
        # Number
        if self.peek() and (self.peek().isdigit() or self.peek() == "-"):
            value = self.number()
            return LiteralPattern(value=value)
        
        # Could be variable or constructor
        name = self.word()
        
        # Check for constructor pattern: Constructor(args)
        if self.try_consume("("):
            args: List[Pattern] = []
            
            while not self.try_consume(")"):
                if args:
                    self.expect(",")
                args.append(self.parse_pattern())
            
            return ConstructorPattern(name=name, args=args)
        
        # Variable pattern (lowercase) or literal constant (True/False/None)
        if name in ("True", "true"):
            return LiteralPattern(value=True)
        if name in ("False", "false"):
            return LiteralPattern(value=False)
        if name in ("None", "null"):
            return LiteralPattern(value=None)
        
        return VarPattern(name=name)
    
    def parse_rule_def(self) -> RuleDef:
        """Parse: rule head :- body1, body2, ... or rule head. (fact)"""
        self.expect("rule")
        
        # Parse head: predicate(args)
        predicate = self.word()
        self.expect("(")
        
        head_args: List[Expression] = []
        while not self.try_consume(")"):
            if head_args:
                self.expect(",")
            head_args.append(self.parse_extended_expression())
        
        head = RuleHead(predicate=predicate, args=head_args)
        
        # Check for fact (ending with .)
        if self.try_consume("."):
            return RuleDef(head=head, body=None)
        
        # Parse body
        self.expect(":-")
        clauses: List[RuleClause] = []
        
        while True:
            negated = self.try_consume("not")
            
            # Could be a predicate call or an expression
            checkpoint = self.token_pos
            try:
                pred_name = self.word()
                if self.try_consume("("):
                    # It's a predicate call
                    args: List[Expression] = []
                    while not self.try_consume(")"):
                        if args:
                            self.expect(",")
                        args.append(self.parse_extended_expression())
                    
                    clauses.append(RuleClause(predicate=pred_name, args=args, negated=negated))
                else:
                    # Backtrack and parse as expression
                    self.token_pos = checkpoint
                    expr = self.parse_extended_expression()
                    clauses.append(RuleClause(expr=expr, negated=negated))
            except N3SyntaxError:
                # Parse as expression
                self.token_pos = checkpoint
                expr = self.parse_extended_expression()
                clauses.append(RuleClause(expr=expr, negated=negated))
            
            if not self.try_consume(","):
                break
        
        self.expect(".")
        return RuleDef(head=head, body=RuleBody(clauses=clauses))
    
    def parse_query_expr(self) -> QueryExpr:
        """Parse: query predicate(args) or query predicate(args) limit n"""
        self.expect("query")
        predicate = self.word()
        self.expect("(")
        
        args: List[Expression] = []
        while not self.try_consume(")"):
            if args:
                self.expect(",")
            args.append(self.parse_extended_expression())
        
        limit = None
        if self.try_consume("limit"):
            limit = int(self.word())
        
        return QueryExpr(predicate=predicate, args=args, limit=limit)
    
    def parse_extended_expression(self) -> Expression:
        """Parse any extended expression including new constructs."""
        # Check for keywords
        if self.peek_word() == "if":
            return self.parse_if_expr()
        
        if self.peek_word() == "let":
            return self.parse_let_expr()
        
        if self.peek_word() == "match":
            return self.parse_match_expr()
        
        if self.peek_word() == "fn":
            # Could be lambda or named function
            # Look ahead to distinguish
            checkpoint = self.token_pos
            self.expect("fn")
            if self.peek() == "(":
                # Lambda
                self.token_pos = checkpoint
                return self.parse_lambda()
            else:
                # Named function - but this should be at statement level
                # For now, allow it as expression
                self.token_pos = checkpoint
                return self.parse_function_def()
        
        if self.peek_word() == "query":
            return self.parse_query_expr()
        
        # List literal
        if self.try_consume("["):
            elements: List[Expression] = []
            while not self.try_consume("]"):
                if elements:
                    self.expect(",")
                elements.append(self.parse_extended_expression())
            return ListExpr(elements=elements)
        
        # Dict literal
        if self.try_consume("{"):
            pairs: List[Tuple[Expression, Expression]] = []
            while not self.try_consume("}"):
                if pairs:
                    self.expect(",")
                key = self.parse_extended_expression()
                self.expect(":")
                value = self.parse_extended_expression()
                pairs.append((key, value))
            return DictExpr(pairs=pairs)
        
        # Tuple or grouped expression
        if self.try_consume("("):
            elements: List[Expression] = []
            while not self.try_consume(")"):
                if elements:
                    self.expect(",")
                elements.append(self.parse_extended_expression())
            
            if len(elements) == 1:
                return elements[0]  # Grouped expression
            return TupleExpr(elements=elements)
        
        # Literals
        if self.peek() and self.peek() in ('"', "'"):
            return LiteralExpr(value=self.string())
        
        if self.peek() and (self.peek().isdigit() or self.peek() == "-"):
            return LiteralExpr(value=self.number())
        
        # Variables and calls
        name = self.word()
        
        # Check for special literals
        if name in ("True", "true"):
            return LiteralExpr(value=True)
        if name in ("False", "false"):
            return LiteralExpr(value=False)
        if name in ("None", "null"):
            return LiteralExpr(value=None)
        
        # Variable or function call
        expr: Expression = VarExpr(name=name)
        
        # Check for indexing, slicing, or calls
        while True:
            if self.try_consume("["):
                # Could be indexing or slicing
                if self.try_consume(":"):
                    # [:end]
                    end = self.parse_extended_expression() if not self.peek() == "]" else None
                    self.expect("]")
                    expr = SliceExpr(base=expr, start=None, end=end)
                else:
                    start = self.parse_extended_expression()
                    if self.try_consume(":"):
                        # [start:end] or [start:]
                        end = self.parse_extended_expression() if not self.peek() == "]" else None
                        self.expect("]")
                        expr = SliceExpr(base=expr, start=start, end=end)
                    else:
                        # [index]
                        self.expect("]")
                        expr = IndexExpr(base=expr, index=start)
            elif self.try_consume("("):
                # Function call
                args: List[Expression] = []
                kwargs: dict[str, Expression] = {}
                
                while not self.try_consume(")"):
                    if args or kwargs:
                        self.expect(",")
                    
                    # Check for keyword argument
                    checkpoint = self.token_pos
                    try:
                        name_part = self.word()
                        if self.try_consume("="):
                            # Keyword argument
                            kwargs[name_part] = self.parse_extended_expression()
                        else:
                            # Positional argument
                            self.token_pos = checkpoint
                            args.append(self.parse_extended_expression())
                    except:
                        self.token_pos = checkpoint
                        args.append(self.parse_extended_expression())
                
                expr = CallExpr(func=expr, args=args, kwargs=kwargs)
            elif self.try_consume("~"):
                # Unification operator
                right = self.parse_extended_expression()
                expr = UnifyExpr(left=expr, right=right)
            else:
                break
        
        return expr
