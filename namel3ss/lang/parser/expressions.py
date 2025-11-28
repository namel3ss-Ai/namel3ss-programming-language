"""Expression and value parsing methods for N3Parser.

Handles expressions, literals, operators, and complex values.
"""

from typing import Any, Dict, List, Optional
from namel3ss.ast import VariableAssignment
from .grammar.lexer import TokenType
from .errors import create_syntax_error


class ExpressionParsingMixin:
    """Mixin with expression and value parsing methods."""
    
    # Structural tokens that cannot be used as field names in blocks
    _STRUCTURAL_TOKENS = frozenset({
        TokenType.LBRACE, TokenType.RBRACE, TokenType.COLON, TokenType.COMMA,
        TokenType.NEWLINE, TokenType.INDENT, TokenType.DEDENT, TokenType.EOF
    })
    
    def _parse_config_block(
        self,
        allow_any_keyword: bool = False,
        special_handlers: Optional[Dict[str, Any]] = None,
        break_on_dedent: bool = False,
    ) -> Dict[str, Any]:
        """
        Parse a configuration block with configurable keyword handling.
        
        This is a shared helper for parse_block() and prompt block parsing.
        It implements the generic block loop: token skipping, key detection,
        and key: value parsing.
        
        Args:
            allow_any_keyword: If True, any non-structural token can be a key.
                              If False, only IDENTIFIER and a limited set of
                              keywords (MODEL, FILTER, INDEX, MEMORY, CHAIN) are allowed.
            special_handlers: Optional dict mapping field names to custom parser
                            functions. If a key matches, the handler is called
                            instead of parse_value(). The handler should consume
                            tokens and return the parsed value.
        
        Returns:
            Dict mapping field names to their parsed values.
        
        Grammar:
            Block = "{" , { BlockStatement } , "}" ;
            BlockStatement = KeyValuePair ;
            KeyValuePair = Key , ":" , Value ;
        """
        # Note: Opening brace should already be consumed by caller
        self.skip_newlines()
        
        # Skip indent if present (lexer may insert it after opening brace)
        self.consume_if(TokenType.INDENT)
        
        config = {}
        special_handlers = special_handlers or {}
        
        while True:
            if break_on_dedent and self.match(TokenType.DEDENT):
                self.advance()
                break
            if break_on_dedent and self.match(TokenType.PAGE, TokenType.SHOW):
                break
            if self.match(TokenType.RBRACE) or self.match(TokenType.EOF):
                break
            # Skip any dedent/indent tokens between lines
            while self.consume_if(TokenType.DEDENT):
                pass
            
            # Check for closing brace after consuming dedents
            if self.match(TokenType.RBRACE):
                break
                
            # Skip indent tokens
            while self.consume_if(TokenType.INDENT):
                pass
            
            # Check again for closing brace
            if self.match(TokenType.RBRACE):
                break
            
            # Parse key - strategy depends on allow_any_keyword flag
            key_token = self.current()
            if not key_token:
                raise self.error("Unexpected end of file in block")
            
            # Check if this is a structural token (not allowed as key)
            if key_token.type == TokenType.COLON:
                # Skip stray colons that can appear in loose YAML-ish layouts
                self.advance()
                self.skip_newlines()
                continue
            if key_token.type in self._STRUCTURAL_TOKENS:
                raise create_syntax_error(
                    "Expected field name in block",
                    path=self.path,
                    line=key_token.line,
                    column=key_token.column,
                    expected="identifier or keyword",
                    found=key_token.type.name.lower()
                )
            
            # Collect key tokens up to the colon (supports composite keys like "using model foo")
            key_parts = []
            while True:
                tok = self.current()
                if tok is None:
                    raise self.error("Unexpected end of file in block")
                if tok.type == TokenType.COLON:
                    break
                if tok.type in self._STRUCTURAL_TOKENS and tok.type not in {TokenType.STRING, TokenType.NUMBER}:
                    # Stop if we hit structural punctuation unexpectedly
                    if tok.type == TokenType.NEWLINE:
                        break
                    raise create_syntax_error(
                        "Expected field name in block",
                        path=self.path,
                        line=tok.line,
                        column=tok.column,
                        expected="identifier or keyword",
                        found=tok.type.name.lower(),
                    )
                key_parts.append(str(tok.value if hasattr(tok, "value") else tok.type.name.lower()))
                self.advance()
            key = " ".join(key_parts).strip()
            
            if self.match(TokenType.COLON):
                self.advance()
            else:
                # Tolerate newline/dedent before colon or implicit true
                if self.match(TokenType.NEWLINE):
                    self.advance()
                    self.skip_newlines()
                if self.match(TokenType.COLON):
                    self.advance()
                else:
                    # Implicit boolean flag
                    config[key] = True
                    self.skip_newlines()
                    continue
            
            # Parse value - use special handler if one is registered for this key
            if key in special_handlers:
                value = special_handlers[key]()
            else:
                value = self.parse_value(allow_dedent_skip=not break_on_dedent)
            
            config[key] = value
            
            self.skip_newlines()
        
        # Skip dedent if present before closing brace
        self.consume_if(TokenType.DEDENT)
        
        return config
    
    def parse_block(self) -> Dict[str, Any]:
        """
        Parse a configuration block.
        
        Grammar:
            Block = "{" , "\n" , { BlockStatement } , "}" ;
            BlockStatement = KeyValuePair | NestedDecl ;
            
        Note: Allows keywords as field names for forward compatibility.
        """
        self.expect(TokenType.LBRACE)
        config = self._parse_config_block(allow_any_keyword=True)
        self.expect(TokenType.RBRACE)
        self.skip_newlines()
        
        return config
    
    def parse_value(self, allow_dedent_skip: bool = True) -> Any:
        """
        Parse a value (literal, array, object, expression).
        
        Grammar:
            Value = Literal | ArrayValue | ObjectValue | Expression ;
        """
        token = self.current()
        
        if not token:
            raise self.error("Expected value")
        if not allow_dedent_skip and token.type in {TokenType.DEDENT, TokenType.PAGE, TokenType.SHOW, TokenType.DATASET, TokenType.MEMORY}:
            return {}
        
        # Inline show statement used inside component configs (e.g., on submit)
        if token.type == TokenType.SHOW:
            return self.parse_show_statement()
        
        # Skip structural tokens that can precede a value in indented YAML-style blocks
        while token and token.type in (TokenType.NEWLINE, TokenType.INDENT, TokenType.DEDENT):
            if token.type == TokenType.DEDENT and not allow_dedent_skip:
                break
            self.advance()
            token = self.current()
            if not token:
                raise self.error("Expected value")
        
        # String literal
        if token.type == TokenType.STRING:
            return self.advance().value
        
        # Number literal
        if token.type == TokenType.NUMBER:
            value = self.advance().value
            if '.' in value or 'e' in value.lower():
                return float(value)
            return int(value)
        
        # Boolean literals
        if token.type in (TokenType.TRUE, TokenType.FALSE):
            self.advance()
            return token.type == TokenType.TRUE
        
        # YAML-style dash list (e.g., - item)
        if token.type == TokenType.MINUS:
            return self._parse_dash_list_value()
        
        # Null literal
        if token.type == TokenType.NULL:
            self.advance()
            return None
        
        # Array literal
        if token.type == TokenType.LBRACKET:
            return self.parse_array_literal()
        
        # Object literal
        if token.type == TokenType.LBRACE:
            return self.parse_object_literal()
        
        # Environment variable reference
        if token.type == TokenType.ENV:
            return self.parse_env_ref()
        
        # Lambda expression
        if token.type == TokenType.FN:
            return self.parse_lambda_expression()
        
        # Inline Python block: python { ... }
        if token.type == TokenType.PYTHON:
            return self.parse_inline_python_block()
        
        # Inline React block: react { ... }
        if token.type == TokenType.REACT:
            return self.parse_inline_react_block()
        
        # Identifier (could be function ref or start of expression)
        if token.type == TokenType.IDENTIFIER:
            # Try to parse as full expression
            return self.parse_expression()
        
        # Try parsing as expression
        return self.parse_expression()
    
    def parse_array_literal(self) -> List[Any]:
        """Parse array literal: [item1, item2, ...]"""
        self.expect(TokenType.LBRACKET)
        self.skip_newlines()
        
        items = []
        
        while not self.match(TokenType.RBRACKET):
            items.append(self.parse_value())
            self.skip_newlines()
            
            if not self.match(TokenType.RBRACKET):
                self.expect(TokenType.COMMA)
                self.skip_newlines()
        
        self.expect(TokenType.RBRACKET)
        return items
    
    def parse_object_literal(self, allow_keyword_keys: bool = False) -> Dict[str, Any]:
        """
        Parse object literal: {key: value, ...}
        
        Args:
            allow_keyword_keys: If True, allows keywords as object keys (for schemas).
        """
        self.expect(TokenType.LBRACE)
        self.skip_newlines()
        
        # Skip indent if present (lexer may insert it after opening brace)
        self.consume_if(TokenType.INDENT)
        
        obj = {}
        
        while not self.match(TokenType.RBRACE):
            # Skip any dedent/indent tokens between lines
            while self.consume_if(TokenType.DEDENT):
                pass
            
            if self.match(TokenType.RBRACE):
                break
            
            while self.consume_if(TokenType.INDENT):
                pass
            
            if self.match(TokenType.RBRACE):
                break
            
            # Key can be identifier, string, or (if allowed) keyword
            if self.match(TokenType.IDENTIFIER):
                key = self.advance().value
            elif self.match(TokenType.STRING):
                key = self.advance().value
            elif allow_keyword_keys:
                # In schema contexts, allow keywords as keys
                key_token = self.current()
                if key_token.type in self._STRUCTURAL_TOKENS:
                    raise self.error("Expected object key (identifier or string)")
                key = key_token.value.lower() if hasattr(key_token, 'value') and key_token.value else key_token.type.name.lower()
                self.advance()
            else:
                raise self.error("Expected object key (identifier or string)")
            
            self.expect(TokenType.COLON)
            value = self.parse_value()
            obj[key] = value
            
            self.skip_newlines()
            
            # Optional comma (newlines can also separate entries)
            if not self.match(TokenType.RBRACE):
                if self.match(TokenType.COMMA):
                    self.advance()
                    self.skip_newlines()
        
        # Skip dedent if present before closing brace
        self.consume_if(TokenType.DEDENT)
        
        self.expect(TokenType.RBRACE)
        return obj
    
    def _parse_dash_list_value(self):
        """Parse YAML-style dash-prefixed lists."""
        items = []
        while True:
            self.skip_newlines()
            if not self.match(TokenType.MINUS):
                break
            self.advance()  # consume '-'
            self.skip_newlines()
            
            # Inline key:value after '-'
            if self.match(TokenType.IDENTIFIER) and self.peek(1) and self.peek(1).type == TokenType.COLON:
                key = self.advance().value
                self.expect(TokenType.COLON)
                self.skip_newlines()
                value = self.parse_value()
                entry = {key: value}
            else:
                entry = self.parse_value()
            
            # Consume nested indented key/values belonging to this dash item
            self.skip_newlines()
            if self.consume_if(TokenType.INDENT):
                self.skip_newlines()
                while not self.match(TokenType.DEDENT) and not self.match(TokenType.EOF):
                    if not self.match(TokenType.IDENTIFIER):
                        break
                    nested_key = self.advance().value
                    self.expect(TokenType.COLON)
                    self.skip_newlines()
                    entry[nested_key] = self.parse_value()
                    self.skip_newlines()
                self.consume_if(TokenType.DEDENT)
            
            items.append(entry)
            self.skip_newlines()
        return items
    
    def parse_env_ref(self) -> Dict[str, str]:
        """Parse environment variable reference: env.VAR_NAME"""
        self.expect(TokenType.ENV)
        self.expect(TokenType.DOT)
        var_name = self.expect(TokenType.IDENTIFIER).value
        
        return {"type": "env_ref", "variable": var_name}
    
    def parse_lambda_expression(self) -> Dict[str, Any]:
        """
        Parse lambda expression.
        
        Grammar:
            LambdaExpr = "fn" , "(" , [ ParameterList ] , ")" , "=>" , Expression ;
        """
        self.expect(TokenType.FN)
        self.expect(TokenType.LPAREN)
        
        params = self.parse_parameter_list()
        
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.FAT_ARROW)
        
        body = self.parse_expression()
        
        return {
            "type": "lambda",
            "params": params,
            "body": body,
        }
    
    def parse_parameter_list(self) -> List[Dict[str, Any]]:
        """Parse function parameter list."""
        params = []
        
        if self.match(TokenType.RPAREN):
            return params
        
        # First parameter
        params.append(self.parse_parameter())
        
        # Additional parameters
        while self.consume_if(TokenType.COMMA):
            params.append(self.parse_parameter())
        
        return params
    
    def parse_parameter(self) -> Dict[str, Any]:
        """Parse single function parameter."""
        name = self.expect(TokenType.IDENTIFIER).value
        
        # Optional type annotation
        param_type = None
        if self.consume_if(TokenType.COLON):
            param_type = self.expect(TokenType.IDENTIFIER).value
        
        return {"name": name, "type": param_type}
    
    def parse_expression(self) -> Any:
        """
        Parse expression with operator precedence.
        
        Uses Pratt parsing (precedence climbing) for operators.
        """
        return self.parse_logical_or()
    
    def parse_logical_or(self) -> Any:
        """Parse logical OR expression."""
        left = self.parse_logical_and()
        
        while self.consume_if(TokenType.OR):
            op = "||"
            right = self.parse_logical_and()
            left = {"type": "binary_op", "op": op, "left": left, "right": right}
        
        return left
    
    def parse_logical_and(self) -> Any:
        """Parse logical AND expression."""
        left = self.parse_equality()
        
        while self.consume_if(TokenType.AND):
            op = "&&"
            right = self.parse_equality()
            left = {"type": "binary_op", "op": op, "left": left, "right": right}
        
        return left
    
    def parse_equality(self) -> Any:
        """Parse equality expression (==, !=)."""
        left = self.parse_relational()
        
        while True:
            if self.consume_if(TokenType.EQ):
                op = "=="
            elif self.consume_if(TokenType.NE):
                op = "!="
            else:
                break
            
            right = self.parse_relational()
            left = {"type": "binary_op", "op": op, "left": left, "right": right}
        
        return left
    
    def parse_relational(self) -> Any:
        """Parse relational expression (<, >, <=, >=)."""
        left = self.parse_additive()
        
        while True:
            if self.consume_if(TokenType.LT):
                op = "<"
            elif self.consume_if(TokenType.GT):
                op = ">"
            elif self.consume_if(TokenType.LE):
                op = "<="
            elif self.consume_if(TokenType.GE):
                op = ">="
            else:
                break
            
            right = self.parse_additive()
            left = {"type": "binary_op", "op": op, "left": left, "right": right}
        
        return left
    
    def parse_additive(self) -> Any:
        """Parse additive expression (+, -)."""
        left = self.parse_multiplicative()
        
        while True:
            if self.consume_if(TokenType.PLUS):
                op = "+"
            elif self.consume_if(TokenType.MINUS):
                op = "-"
            else:
                break
            
            right = self.parse_multiplicative()
            left = {"type": "binary_op", "op": op, "left": left, "right": right}
        
        return left
    
    def parse_multiplicative(self) -> Any:
        """Parse multiplicative expression (*, /, %)."""
        left = self.parse_exponential()
        
        while True:
            if self.consume_if(TokenType.STAR):
                op = "*"
            elif self.consume_if(TokenType.SLASH):
                op = "/"
            elif self.consume_if(TokenType.PERCENT):
                op = "%"
            else:
                break
            
            right = self.parse_exponential()
            left = {"type": "binary_op", "op": op, "left": left, "right": right}
        
        return left
    
    def parse_exponential(self) -> Any:
        """Parse exponential expression (**)."""
        left = self.parse_unary()
        
        if self.consume_if(TokenType.POWER):
            # Right-associative
            right = self.parse_exponential()
            return {"type": "binary_op", "op": "**", "left": left, "right": right}
        
        return left
    
    def parse_unary(self) -> Any:
        """Parse unary expression (!, -, +)."""
        if self.consume_if(TokenType.NOT):
            operand = self.parse_unary()
            return {"type": "unary_op", "op": "!", "operand": operand}
        
        if self.consume_if(TokenType.MINUS):
            operand = self.parse_unary()
            return {"type": "unary_op", "op": "-", "operand": operand}
        
        if self.consume_if(TokenType.PLUS):
            operand = self.parse_unary()
            return {"type": "unary_op", "op": "+", "operand": operand}
        
        return self.parse_postfix()
    
    def parse_postfix(self) -> Any:
        """Parse postfix expression (function call, member access, index)."""
        expr = self.parse_primary()
        
        while True:
            # Function call
            if self.match(TokenType.LPAREN):
                expr = self.parse_function_call(expr)
            # Member access
            elif self.consume_if(TokenType.DOT):
                member = self.expect(TokenType.IDENTIFIER).value
                expr = {"type": "member_access", "object": expr, "member": member}
            # Index access
            elif self.match(TokenType.LBRACKET):
                self.advance()
                index = self.parse_expression()
                self.expect(TokenType.RBRACKET)
                expr = {"type": "index_access", "object": expr, "index": index}
            else:
                break
        
        return expr
    
    def parse_function_call(self, callee: Any) -> Dict[str, Any]:
        """Parse function call arguments."""
        self.expect(TokenType.LPAREN)
        
        args = []
        while not self.match(TokenType.RPAREN):
            args.append(self.parse_expression())
            if not self.match(TokenType.RPAREN):
                self.expect(TokenType.COMMA)
        
        self.expect(TokenType.RPAREN)
        
        return {"type": "function_call", "callee": callee, "args": args}
    
    def parse_primary(self) -> Any:
        """Parse primary expression."""
        token = self.current()
        
        if not token:
            raise self.error("Expected expression")
        
        # Literals
        if token.type == TokenType.STRING:
            return self.advance().value
        
        if token.type == TokenType.NUMBER:
            value = self.advance().value
            if '.' in value or 'e' in value.lower():
                return float(value)
            return int(value)
        
        if token.type in (TokenType.TRUE, TokenType.FALSE):
            self.advance()
            return token.type == TokenType.TRUE
        
        if token.type == TokenType.NULL:
            self.advance()
            return None
        
        # Identifier
        if token.type == TokenType.IDENTIFIER:
            if self.peek(1) and self.peek(1).type == TokenType.COLON:
                # Interpret as inline object mapping (YAML-style) until dedent
                return self._parse_config_block(allow_any_keyword=True, break_on_dedent=True)
            return self.advance().value
        
        # Treat certain keyword tokens as identifier-like values (e.g., template/model names)
        keyword_ident_tokens = {
            TokenType.TEMPLATE, TokenType.MODEL, TokenType.LLM, TokenType.CHAIN,
            TokenType.AGENT, TokenType.TOOL, TokenType.MEMORY, TokenType.DATASET,
            TokenType.FRAME, TokenType.INDEX, TokenType.CONNECTOR, TokenType.TABLE,
        }
        if token.type in keyword_ident_tokens:
            value = token.value if hasattr(token, "value") else token.type.name.lower()
            self.advance()
            return value
        
        # Parenthesized expression
        if token.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        
        # Array literal
        if token.type == TokenType.LBRACKET:
            return self.parse_array_literal()
        
        # Object literal
        if token.type == TokenType.LBRACE:
            return self.parse_object_literal()
        
        # Lambda
        if token.type == TokenType.FN:
            return self.parse_lambda_expression()
        
        # Match expression
        if token.type == TokenType.MATCH:
            return self.parse_match_expression()
        
        # Let expression
        if token.type == TokenType.LET:
            return self.parse_let_expression()
        
        # Env reference
        if token.type == TokenType.ENV:
            return self.parse_env_ref()
        
        raise create_syntax_error(
            "Expected expression",
            path=self.path,
            line=token.line,
            found=token.value,
        )
    
    def parse_match_expression(self) -> Dict[str, Any]:
        """Parse match expression."""
        self.expect(TokenType.MATCH)
        scrutinee = self.parse_expression()
        
        self.expect(TokenType.LBRACE)
        self.skip_newlines()
        
        cases = []
        while not self.match(TokenType.RBRACE):
            self.expect(TokenType.CASE)
            pattern = self.parse_pattern()
            self.expect(TokenType.FAT_ARROW)
            result = self.parse_expression()
            
            cases.append({"pattern": pattern, "result": result})
            self.skip_newlines()
        
        self.expect(TokenType.RBRACE)
        
        return {
            "type": "match",
            "scrutinee": scrutinee,
            "cases": cases,
        }
    
    def parse_pattern(self) -> Any:
        """Parse match pattern."""
        token = self.current()
        
        # Array pattern
        if token.type == TokenType.LBRACKET:
            return self.parse_array_pattern()
        
        # Literal pattern
        if token.type in (TokenType.STRING, TokenType.NUMBER, TokenType.TRUE, TokenType.FALSE, TokenType.NULL):
            return self.parse_value()
        
        # Identifier pattern (binding)
        if token.type == TokenType.IDENTIFIER:
            return self.advance().value
        
        raise self.error("Invalid pattern")
    
    def parse_array_pattern(self) -> Dict[str, Any]:
        """Parse array pattern: [], [x], [x, y], [x, ...rest]"""
        self.expect(TokenType.LBRACKET)
        
        elements = []
        rest = None
        
        while not self.match(TokenType.RBRACKET):
            if self.consume_if(TokenType.ELLIPSIS):
                rest = self.expect(TokenType.IDENTIFIER).value
                break
            
            elements.append(self.parse_pattern())
            
            if not self.match(TokenType.RBRACKET):
                self.expect(TokenType.COMMA)
        
        self.expect(TokenType.RBRACKET)
        
        return {
            "type": "array_pattern",
            "elements": elements,
            "rest": rest,
        }
    
    def parse_let_expression(self) -> Dict[str, Any]:
        """Parse let expression: let x = expr in expr"""
        self.expect(TokenType.LET)
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ASSIGN)
        value = self.parse_expression()
        self.expect(TokenType.IN)
        body = self.parse_expression()
        
        return {
            "type": "let",
            "name": name,
            "value": value,
            "body": body,
        }
    
    def parse_schema_definition(self) -> List[Dict[str, Any]]:
        """Parse schema definition for input/output."""
        self.skip_newlines()
        
        if self.match(TokenType.LBRACKET):
            # Array-style schema
            self.advance()
            self.skip_newlines()
            
            fields = []
            while not self.match(TokenType.RBRACKET):
                # Expect: - field_name: type (constraint)
                if self.consume_if(TokenType.MINUS):
                    field_name = self.expect(TokenType.IDENTIFIER).value
                    self.expect(TokenType.COLON)
                    field_type = self.expect(TokenType.IDENTIFIER).value
                    
                    constraint = None
                    if self.consume_if(TokenType.LPAREN):
                        constraint = self.expect(TokenType.IDENTIFIER).value
                        self.expect(TokenType.RPAREN)
                    
                    fields.append({
                        "name": field_name,
                        "type": field_type,
                        "constraint": constraint,
                    })
                
                self.skip_newlines()
            
            self.expect(TokenType.RBRACKET)
            return fields
        
        # Object-style schema - allow keywords as keys
        return self.parse_object_literal(allow_keyword_keys=True)
    
    # Helper methods
    
    def parse_page_statement(self) -> Any:
        """Parse a page statement (show, if, for, log, etc.)."""
        token = self.current()
        
        # Skip leading newlines only
        while token and token.type == TokenType.NEWLINE:
            self.advance()
            token = self.current()
        if not token:
            return None
        
        # Show statement
        if token.type == TokenType.SHOW:
            return self.parse_show_statement()
        
        # Control flow
        if token.type == TokenType.IF:
            return self.parse_if_statement()
        
        if token.type == TokenType.FOR:
            return self.parse_for_statement()
        
        # Set statement (e.g., set session.var to expr)
        if token.type == TokenType.IDENTIFIER and token.value == "set":
            return self.parse_set_statement()
            
        # Log statement
        if token.type == TokenType.IDENTIFIER and token.value == "log":
            return self.parse_log_statement()
        
        # Action statement inside pages
        if token.type == TokenType.IDENTIFIER and token.value == "action":
            return self.parse_action_statement()
        
        # Generic container-style blocks (stack, grid, tabs, etc.)
        if token.type == TokenType.IDENTIFIER and self._has_colon_ahead():
            container_name = token.value
            # Consume tokens up to colon on this line
            while not self.match(TokenType.COLON) and not self.match(TokenType.EOF):
                self.advance()
            self.expect(TokenType.COLON)
            self.skip_newlines()
            self.consume_if(TokenType.INDENT)
            body = []
            while not self.match(TokenType.DEDENT) and not self.match(TokenType.EOF):
                stmt = self.parse_page_statement()
                if stmt:
                    body.append(stmt)
                self.skip_newlines()
            self.consume_if(TokenType.DEDENT)
            return {"type": container_name, "body": body}
        
        # Simple assignment within page blocks: name: <value>
        next_token = self.peek(1)
        if token.type == TokenType.IDENTIFIER and next_token and next_token.type == TokenType.COLON:
            name = token.value
            self.advance()  # consume name
            self.expect(TokenType.COLON)
            self.skip_newlines()
            value = self.parse_value()
            return VariableAssignment(name=name, value=value)
        
        # Expression statement
        return self.parse_expression()

    def parse_set_statement(self):
        """Parse simple set statements: set target to value."""
        self.expect(TokenType.IDENTIFIER)  # 'set'
        target_tokens = []
        while self.current() and not (self.match(TokenType.IDENTIFIER) and self.current().value == "to"):
            if self.match(TokenType.NEWLINE):
                break
            tok = self.advance()
            target_tokens.append(str(tok.value if hasattr(tok, "value") else tok.type.name.lower()))
        target = " ".join(target_tokens).strip()
        if self.match(TokenType.IDENTIFIER) and self.current().value == "to":
            self.advance()
        value = None
        if not self.match(TokenType.NEWLINE, TokenType.DEDENT, TokenType.EOF):
            value = self.parse_expression()
        return {"type": "set", "target": target, "value": value}
    
    def parse_show_statement(self) -> Dict[str, Any]:
        """Parse show statement."""
        from namel3ss.ast.pages import ShowText, ShowForm, FormField
        from namel3ss.lang.parser.errors import create_syntax_error
        
        self.expect(TokenType.SHOW)
        
        # Component type can be identifier or certain keywords
        component_token = self.current()
        if component_token.type == TokenType.IDENTIFIER:
            component_type = component_token.value
            self.advance()
        else:
            # Allow any token as component type (for flexibility)
            component_type = str(component_token.value if hasattr(component_token, 'value') and component_token.value else component_token.type.name.lower())
            self.advance()
        
        # Capture simple inline payloads like: show text "hello"
        inline_value = None
        if self.match(TokenType.STRING):
            inline_value = self.advance().value
        
        # Explicitly reject unsupported components with guidance
        unsupported = {
            "progress_bar": "Use stat_summary or a simple text indicator instead.",
            "code_block": "Use show text with syntax highlighting client-side.",
            "json_view": "Use show text or data_table as an alternative.",
            "tree_view": "Use accordion or data_list as an alternative.",
        }
        if component_type in unsupported:
            raise create_syntax_error(
                "Unsupported component",
                path=self.path,
                line=component_token.line,
                column=component_token.column,
                suggestion=unsupported[component_type],
            )
        
        # Optional source clause (e.g., show table "Users" from dataset users)
        source_type = None
        source_name = None
        if self.consume_if(TokenType.FROM):
            if self.match(TokenType.DATASET):
                self.advance()
                source_type = "dataset"
                source_name = self.expect_name()
            elif self.match(TokenType.TABLE):
                self.advance()
                source_type = "table"
                source_name = self.expect_name()
            elif self.match(TokenType.IDENTIFIER):
                source_type = self.advance().value
                if self.match(TokenType.IDENTIFIER) or self.match(TokenType.STRING):
                    source_name = self.expect_name()
        
        has_inline_block = False
        
        config = {}
        if self.consume_if(TokenType.COLON):
            has_inline_block = True
            self.skip_newlines()
            # If we see an indented block, parse it as a config map instead of a single value
            if self.match(TokenType.INDENT):
                config = self._parse_config_block(allow_any_keyword=True, break_on_dedent=True)
            else:
                config = self.parse_value()
        elif self.match(TokenType.LBRACE):
            config = self.parse_block()
        
        # Propagate source info into config for downstream use
        if source_type or source_name:
            if isinstance(config, dict):
                config.setdefault("from", {})
                if isinstance(config["from"], dict):
                    if source_type:
                        config["from"]["type"] = source_type
                    if source_name:
                        config["from"]["source"] = source_name
                else:
                    config["from"] = {"type": source_type, "source": source_name}
            else:
                config = {"from": {"type": source_type, "source": source_name}, "value": config} if config else {"from": {"type": source_type, "source": source_name}}
        
        # Special-case common components
        if component_type == "text":
            text_value = inline_value if inline_value is not None else (config if isinstance(config, str) else None)
            if text_value is not None:
                return ShowText(text=text_value)
        if component_type == "form":
            fields_cfg = config.pop("fields", []) if isinstance(config, dict) else []
            if isinstance(config, dict) and isinstance(fields_cfg, list) and fields_cfg:
                shared_keys = {}
                for k in ("component", "label", "multiple", "accept", "max_file_size"):
                    if k in config:
                        shared_keys[k] = config.pop(k)
                if shared_keys:
                    if isinstance(fields_cfg[0], dict):
                        fields_cfg[0].update(shared_keys)
                    else:
                        fields_cfg[0] = {**shared_keys, "name": str(fields_cfg[0])}
            # Merge flattened dict entries into field objects
            merged_fields = []
            current = {}
            for entry in fields_cfg if isinstance(fields_cfg, list) else []:
                if isinstance(entry, dict):
                    if ("name" in entry and current) or ("component" in entry and "name" in current):
                        merged_fields.append(current)
                        current = {}
                    current.update(entry)
                elif isinstance(entry, str):
                    if current:
                        merged_fields.append(current)
                    current = {"name": entry}
            if current:
                merged_fields.append(current)
            
            form_fields = []
            for f in merged_fields:
                if isinstance(f, dict):
                    form_fields.append(FormField(
                        name=str(f.get("name", "")),
                        component=str(f.get("component", f.get("field_type", "text_input"))),
                        label=f.get("label"),
                        placeholder=f.get("placeholder"),
                        help_text=f.get("help_text"),
                        required=bool(f.get("required", False)),
                        multiple=bool(f.get("multiple", False)),
                        accept=f.get("accept"),
                        max_file_size=f.get("max_file_size"),
                    ))
                elif isinstance(f, str):
                    form_fields.append(FormField(name=f))
            on_submit_cfg = None
            if isinstance(config, dict):
                on_submit_cfg = config.pop("on submit", config.pop("onsubmit", None))
            return ShowForm(
                title=inline_value or config.get("title") if isinstance(config, dict) else inline_value,
                fields=form_fields,
                on_submit_ops=on_submit_cfg if isinstance(on_submit_cfg, list) else [],
            )
        
        # Consume remaining inline tokens on this line to avoid stray parsing (e.g., type="success")
        while self.current() and self.current().type not in (TokenType.NEWLINE, TokenType.DEDENT, TokenType.RBRACE, TokenType.EOF):
            self.advance()
        
        # If we captured an inline value but don't have a structured config,
        # carry it forward in the config for downstream consumers.
        if inline_value is not None and not config:
            config = inline_value
        
        return {
            "type": "show",
            "component": component_type,
            "config": config,
        }

    def _has_colon_ahead(self) -> bool:
        """Look ahead on the current line for a colon before newline/dedent."""
        i = 0
        while True:
            tok = self.peek(i)
            if not tok:
                return False
            if tok.type == TokenType.COLON:
                return True
            if tok.type in (TokenType.NEWLINE, TokenType.DEDENT, TokenType.RBRACE, TokenType.EOF):
                return False
            i += 1
    
    def parse_log_statement(self) -> Dict[str, Any]:
        """Parse log statement: log [level] "message" """
        from namel3ss.ast import LogLevel, LogStatement, Literal
        from namel3ss.ast.source_location import SourceLocation
        
        # Consume 'log' token
        self.expect(TokenType.IDENTIFIER)  # 'log'
        
        # Check for optional level
        level = LogLevel.INFO  # default
        level_token = self.current()
        
        if (level_token and level_token.type == TokenType.IDENTIFIER and 
            level_token.value in {'debug', 'info', 'warn', 'error'}):
            level = LogLevel(level_token.value)
            self.advance()
        
        # Expect message string
        message_token = self.expect(TokenType.STRING)
        message_text = message_token.value
        
        # Create AST node
        message = Literal(message_text)
        source_location = SourceLocation(
            file=getattr(self, 'filename', '<unknown>'),
            line=message_token.line,
            column=message_token.column
        )
        
        return LogStatement(
            level=level,
            message=message,
            source_location=source_location
        )
    
    def parse_action_statement(self):
        """Parse page-level action blocks (action "...": when ...)."""
        from namel3ss.ast import (
            Action,
            ToastOperation,
            RunPromptOperation,
            RunChainOperation,
            GoToPageOperation,
            AskConnectorOperation,
            CallPythonOperation,
        )
        
        self.expect(TokenType.IDENTIFIER)  # 'action'
        name = self.expect_name()
        declared_effect = None
        if self.match(TokenType.IDENTIFIER) and self.current().value == "effect":
            self.advance()
            declared_effect = self.expect_name()
        
        # Support both ':' and '{' but tests rely on ':' with indentation
        if self.consume_if(TokenType.COLON):
            pass
        elif self.consume_if(TokenType.LBRACE):
            # Allow brace syntax as well
            pass
        else:
            raise create_syntax_error(
                "Action declaration must end with ':' or '{'",
                path=self.path,
                line=self.current().line if self.current() else None,
                suggestion="Use: action \"Name\":",
            )
        
        self.skip_newlines()
        self.consume_if(TokenType.INDENT)
        
        trigger = ""
        # Optional 'when' trigger line
        if self.match(TokenType.IDENTIFIER) and self.current().value == "when":
            self.advance()
            trigger_parts = []
            while self.current() and self.current().type not in (TokenType.COLON, TokenType.NEWLINE):
                tok = self.advance()
                trigger_parts.append(str(tok.value if hasattr(tok, "value") else tok.type.name.lower()))
            trigger = " ".join(trigger_parts).strip()
            self.consume_if(TokenType.COLON)
            self.skip_newlines()
            self.consume_if(TokenType.INDENT)
        
        operations = []
        while not self.match(TokenType.DEDENT) and not self.match(TokenType.RBRACE) and not self.match(TokenType.EOF):
            self.skip_newlines()
            if self.match(TokenType.DEDENT) or self.match(TokenType.RBRACE) or self.match(TokenType.EOF):
                break
            
            op = None
            # show toast "msg"
            if self.match(TokenType.SHOW):
                op = self._parse_action_show_operation()
            # run prompt foo with:
            elif self.match(TokenType.IDENTIFIER) and self.current().value == "run" and self.peek(1) and self.peek(1).type == TokenType.PROMPT:
                op = self._parse_action_run_prompt()
            # run chain foo with:
            elif self.match(TokenType.IDENTIFIER) and self.current().value == "run" and self.peek(1) and self.peek(1).type == TokenType.CHAIN:
                op = self._parse_action_run_chain()
            # go to page "X"
            elif self.match(TokenType.IDENTIFIER) and self.current().value == "go":
                op = self._parse_action_go_to_page()
            # ask connector foo with:
            elif self.match(TokenType.IDENTIFIER) and self.current().value == "ask":
                op = self._parse_action_ask_connector()
            # call python "module" method "fn" with:
            elif self.match(TokenType.IDENTIFIER) and self.current().value == "call":
                op = self._parse_action_call_python()
            
            # Unknown operation: skip the line gracefully
            if op is None:
                while self.current() and self.current().type not in (TokenType.NEWLINE, TokenType.DEDENT, TokenType.RBRACE, TokenType.EOF):
                    self.advance()
            else:
                operations.append(op)
            
            self.skip_newlines()
            while self.consume_if(TokenType.DEDENT):
                pass
        
        # Consume closing dedent/brace if present
        self.consume_if(TokenType.DEDENT)
        if self.match(TokenType.RBRACE):
            self.advance()
        
        return Action(
            name=name,
            trigger=trigger or "",
            operations=operations,
            declared_effect=declared_effect,
        )
    
    def _parse_action_show_operation(self):
        """Parse simple show toast operation used in actions."""
        from namel3ss.ast import ToastOperation
        
        self.expect(TokenType.SHOW)
        if self.match(TokenType.IDENTIFIER):
            component = self.advance().value
        else:
            component = str(self.advance().value if self.current() else "")
        message = ""
        if self.match(TokenType.STRING):
            message = self.advance().value
        # Treat any show toast as ToastOperation; otherwise fallback to generic message
        if component == "toast":
            op = ToastOperation(message=message)
        else:
            op = ToastOperation(message=message or component)
        # Consume rest of line to avoid stray tokens (e.g., type="success")
        while self.current() and self.current().type not in (TokenType.NEWLINE, TokenType.DEDENT, TokenType.RBRACE, TokenType.EOF):
            self.advance()
        return op
    
    def _parse_action_run_prompt(self):
        """Parse run prompt <name> with: block."""
        from namel3ss.ast import RunPromptOperation
        
        self.expect(TokenType.IDENTIFIER)  # run
        self.expect(TokenType.PROMPT)
        prompt_name = self.expect_name()
        
        arguments = {}
        if self.match(TokenType.IDENTIFIER) and self.current().value == "with":
            self.advance()
        if self.consume_if(TokenType.COLON):
            self.skip_newlines()
            self.consume_if(TokenType.INDENT)
            while not self.match(TokenType.DEDENT) and not self.match(TokenType.EOF):
                if not self.match(TokenType.IDENTIFIER):
                    break
                arg_name = self.advance().value
                self.expect(TokenType.ASSIGN)
                arg_value = self.parse_expression()
                arguments[arg_name] = arg_value
                self.skip_newlines()
            self.consume_if(TokenType.DEDENT)
        return RunPromptOperation(prompt_name=prompt_name, arguments=arguments)
    
    def _parse_action_run_chain(self):
        """Parse run chain <name> with: block."""
        from namel3ss.ast import RunChainOperation
        
        self.expect(TokenType.IDENTIFIER)  # run
        self.expect(TokenType.CHAIN)
        chain_name = self.expect_name()
        
        inputs = {}
        if self.match(TokenType.IDENTIFIER) and self.current().value == "with":
            self.advance()
        if self.consume_if(TokenType.COLON):
            self.skip_newlines()
            self.consume_if(TokenType.INDENT)
            while not self.match(TokenType.DEDENT) and not self.match(TokenType.EOF):
                if not self.match(TokenType.IDENTIFIER):
                    break
                input_name = self.advance().value
                self.expect(TokenType.ASSIGN)
                input_value = self.parse_expression()
                inputs[input_name] = input_value
                self.skip_newlines()
            self.consume_if(TokenType.DEDENT)
        return RunChainOperation(chain_name=chain_name, inputs=inputs)
    
    def _parse_action_ask_connector(self):
        """Parse ask connector <name> with: block."""
        from namel3ss.ast import AskConnectorOperation
        
        self.expect(TokenType.IDENTIFIER)  # ask
        if self.match(TokenType.IDENTIFIER) and self.current().value == "connector":
            self.advance()
        connector_name = self.expect_name()
        
        arguments = {}
        if self.match(TokenType.IDENTIFIER) and self.current().value == "with":
            self.advance()
        if self.consume_if(TokenType.COLON):
            self.skip_newlines()
            self.consume_if(TokenType.INDENT)
            while not self.match(TokenType.DEDENT) and not self.match(TokenType.EOF):
                if not self.match(TokenType.IDENTIFIER):
                    break
                arg_name = self.advance().value
                self.expect(TokenType.ASSIGN)
                arg_value = self.parse_expression()
                arguments[arg_name] = arg_value
                self.skip_newlines()
            self.consume_if(TokenType.DEDENT)
        return AskConnectorOperation(connector_name=connector_name, arguments=arguments)
    
    def _parse_action_call_python(self):
        """Parse call python \"module\" method \"fn\" with: block."""
        from namel3ss.ast import CallPythonOperation
        
        self.expect(TokenType.IDENTIFIER)  # call
        if self.match(TokenType.IDENTIFIER) and self.current().value == "python":
            self.advance()
        module_name = self.expect_name()
        if self.match(TokenType.IDENTIFIER) and self.current().value == "method":
            self.advance()
        method_name = self.expect_name()
        
        arguments = {}
        if self.match(TokenType.IDENTIFIER) and self.current().value == "with":
            self.advance()
        if self.consume_if(TokenType.COLON):
            self.skip_newlines()
            self.consume_if(TokenType.INDENT)
            while not self.match(TokenType.DEDENT) and not self.match(TokenType.EOF):
                if not self.match(TokenType.IDENTIFIER):
                    break
                arg_name = self.advance().value
                self.expect(TokenType.ASSIGN)
                arg_value = self.parse_expression()
                arguments[arg_name] = arg_value
                self.skip_newlines()
            self.consume_if(TokenType.DEDENT)
        return CallPythonOperation(module=module_name, method=method_name, arguments=arguments)
    
    def _parse_action_go_to_page(self):
        """Parse go to page \"Page\" operation."""
        from namel3ss.ast import GoToPageOperation
        
        self.expect(TokenType.IDENTIFIER)  # go
        if self.match(TokenType.TO):
            self.advance()
        if self.match(TokenType.PAGE):
            self.advance()
        page_name = self.expect_name()
        return GoToPageOperation(page_name=page_name)
    
    def parse_if_statement(self) -> Dict[str, Any]:
        """Parse if statement."""
        self.expect(TokenType.IF)
        condition = self.parse_expression()
        
        # Support both brace and colon syntax
        if self.match(TokenType.LBRACE):
            # New brace syntax
            self.advance()
            self.skip_newlines()
            self.consume_if(TokenType.INDENT)
            
            then_statements = []
            while not self.match(TokenType.RBRACE):
                while self.consume_if(TokenType.DEDENT):
                    pass
                if self.match(TokenType.RBRACE):
                    break
                while self.consume_if(TokenType.INDENT):
                    pass
                
                stmt = self.parse_page_statement()
                if stmt:
                    then_statements.append(stmt)
                self.skip_newlines()
            
            self.consume_if(TokenType.DEDENT)
            self.expect(TokenType.RBRACE)
            self.skip_newlines()
        else:
            # Legacy colon syntax
            self.expect(TokenType.COLON)
            self.skip_newlines()
            
            then_statements = [self.parse_page_statement()]
            self.skip_newlines()
        
        # Handle else clause
        else_statements = None
        if self.match(TokenType.ELSE):
            self.advance()
            
            if self.match(TokenType.LBRACE):
                # Brace syntax for else
                self.advance()
                self.skip_newlines()
                self.consume_if(TokenType.INDENT)
                
                else_statements = []
                while not self.match(TokenType.RBRACE):
                    while self.consume_if(TokenType.DEDENT):
                        pass
                    if self.match(TokenType.RBRACE):
                        break
                    while self.consume_if(TokenType.INDENT):
                        pass
                    
                    stmt = self.parse_page_statement()
                    if stmt:
                        else_statements.append(stmt)
                    self.skip_newlines()
                
                self.consume_if(TokenType.DEDENT)
                self.expect(TokenType.RBRACE)
                self.skip_newlines()
            else:
                # Colon syntax for else
                self.expect(TokenType.COLON)
                self.skip_newlines()
                else_statements = [self.parse_page_statement()]
                self.skip_newlines()
        
        return {
            "type": "if",
            "condition": condition,
            "then": then_statements,
            "else": else_statements,
        }
        then_block = self.parse_indented_statements()
        
        # Optional else
        else_block = None
        if self.match(TokenType.ELSE):
            self.advance()
            self.expect(TokenType.COLON)
            self.skip_newlines()
            else_block = self.parse_indented_statements()
        
        return {
            "type": "if",
            "condition": condition,
            "then": then_block,
            "else": else_block,
        }
    
    def parse_for_statement(self) -> Dict[str, Any]:
        """Parse for statement."""
        self.expect(TokenType.FOR)
        var = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.IN)
        iterable = self.parse_expression()
        self.expect(TokenType.COLON)
        self.skip_newlines()
        
        body = self.parse_indented_statements()
        
        return {
            "type": "for",
            "variable": var,
            "iterable": iterable,
            "body": body,
        }
    
    def parse_indented_statements(self) -> List[Any]:
        """Parse indented block of statements."""
        self.expect(TokenType.INDENT)
        
        statements = []
        while not self.match(TokenType.DEDENT):
            stmt = self.parse_page_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()
        
        self.expect(TokenType.DEDENT)
        return statements
    
    def parse_indented_config(self) -> Dict[str, Any]:
        """Parse indented configuration block."""
        self.expect(TokenType.INDENT)
        
        config = {}
        while not self.match(TokenType.DEDENT):
            key = self.expect(TokenType.IDENTIFIER).value
            
            if self.consume_if(TokenType.COLON):
                value = self.parse_value()
                config[key] = value
            
            self.skip_newlines()
        
        self.expect(TokenType.DEDENT)
        return config
    
    def parse_dataset_source(self) -> Dict[str, Any]:
        """Parse dataset source (table, file, query)."""
        # Database table
        if self.consume_if(TokenType.TABLE):
            table_name = self.expect(TokenType.IDENTIFIER).value
            return {"type": "table", "name": table_name}
        
        # Postgres table
        if self.consume_if(TokenType.POSTGRES):
            self.expect(TokenType.TABLE)
            table_name = self.expect(TokenType.IDENTIFIER).value
            return {"type": "postgres_table", "name": table_name}
        
        # Generic identifier
        table_name = self.expect(TokenType.IDENTIFIER).value
        return {"type": "table", "name": table_name}
    
    def parse_chain_step(self) -> Dict[str, Any]:
        """
        Parse a chain step definition (DEPRECATED - use parse.py's _parse_step_block).
        
        This method is kept for backward compatibility but should not be used
        for new chain definitions. Chain steps are now parsed as part of
        parse_chain_declaration using the 'step' keyword and proper ChainStep AST nodes.
        
        Legacy behavior: parse simple identifier reference or expression.
        """
        # Simple identifier reference (legacy)
        if self.match(TokenType.IDENTIFIER):
            step_ref = self.advance().value
            
            # Check for arrow/pipe connector
            connector = None
            if self.consume_if(TokenType.ARROW):
                connector = "->"
            elif self.consume_if(TokenType.PIPE):
                connector = "|"
            
            return {
                "type": "step_ref",
                "ref": step_ref,
                "connector": connector,
            }
        
        raise self.error("Expected chain step")
    
    def parse_inline_python_block(self) -> Dict[str, Any]:
        """
        Parse inline Python code block.
        
        Syntax: python { <python code> }
        
        Returns AST representation of InlinePythonBlock.
        The code is extracted verbatim, preserving whitespace and indentation.
        Nested braces are handled correctly.
        """
        from namel3ss.ast import InlinePythonBlock
        from namel3ss.ast.source_location import SourceLocation
        
        # Consume 'python' keyword
        python_token = self.expect(TokenType.PYTHON)
        start_line = python_token.line
        start_column = python_token.column
        
        # Expect opening brace
        self.expect(TokenType.LBRACE)
        
        # Extract code until matching closing brace
        code = self._extract_inline_code_block()
        
        # Create source location
        end_token = self.current() or python_token
        location = SourceLocation(
            file=self.path,
            line=start_line,
            column=start_column,
            end_line=end_token.line,
            end_column=getattr(end_token, 'column', 0),
        )
        
        # Return InlinePythonBlock node
        return InlinePythonBlock(
            code=code,
            location=location,
        )
    
    def parse_inline_react_block(self) -> Dict[str, Any]:
        """
        Parse inline React/JSX code block.
        
        Syntax: react { <jsx code> }
        
        Returns AST representation of InlineReactBlock.
        The JSX code is extracted verbatim, preserving formatting.
        Nested braces are handled correctly.
        """
        from namel3ss.ast import InlineReactBlock
        from namel3ss.ast.source_location import SourceLocation
        
        # Consume 'react' keyword
        react_token = self.expect(TokenType.REACT)
        start_line = react_token.line
        start_column = react_token.column
        
        # Expect opening brace
        self.expect(TokenType.LBRACE)
        
        # Extract code until matching closing brace
        code = self._extract_inline_code_block()
        
        # Create source location
        end_token = self.current() or react_token
        location = SourceLocation(
            file=self.path,
            line=start_line,
            column=start_column,
            end_line=end_token.line,
            end_column=getattr(end_token, 'column', 0),
        )
        
        # Return InlineReactBlock node
        return InlineReactBlock(
            code=code,
            location=location,
        )
    
    def _extract_inline_code_block(self) -> str:
        """
        Extract raw code from inline block, handling nested braces.
        
        Assumes opening brace has already been consumed.
        Consumes tokens until matching closing brace is found.
        Preserves original source text including whitespace and formatting.
        
        Returns:
            Raw code string without leading/trailing braces.
        
        Raises:
            SyntaxError if closing brace is never found.
        """
        # Track brace depth to handle nested braces
        brace_depth = 1
        code_tokens = []
        prev_token_type = None
        
        while brace_depth > 0:
            token = self.current()
            
            if not token or token.type == TokenType.EOF:
                raise self.error("Unclosed inline block: expected '}'")
            
            # Track brace depth
            if token.type == TokenType.LBRACE:
                brace_depth += 1
                code_tokens.append('{')
                prev_token_type = TokenType.LBRACE
                self.advance()
            elif token.type == TokenType.RBRACE:
                brace_depth -= 1
                if brace_depth > 0:
                    code_tokens.append('}')
                    prev_token_type = TokenType.RBRACE
                self.advance()
            else:
                # Add space between tokens unless they are structural punctuation
                no_space_prev = {
                    TokenType.NEWLINE,
                    TokenType.INDENT,
                    TokenType.DEDENT,
                    TokenType.LBRACE,
                    TokenType.LBRACKET,
                    TokenType.LPAREN,
                    TokenType.LT,
                    TokenType.SLASH,
                    TokenType.DOT,
                }
                no_space_current = {
                    TokenType.RBRACE,
                    TokenType.RBRACKET,
                    TokenType.RPAREN,
                    TokenType.COMMA,
                    TokenType.DOT,
                    TokenType.COLON,
                    TokenType.GT,
                    TokenType.SLASH,
                    TokenType.LPAREN,
                }
                token_value = token.value if hasattr(token, "value") else None
                if (
                    prev_token_type is not None
                    and prev_token_type not in no_space_prev
                    and token.type not in no_space_current
                    and token_value not in ("!", "?")
                ):
                    code_tokens.append(' ')
                
                # Preserve token value
                if token.type == TokenType.STRING:
                    code_tokens.append(f"\"{token.value}\"")
                elif hasattr(token, 'value') and token.value is not None:
                    code_tokens.append(str(token.value))
                elif token.type == TokenType.NEWLINE:
                    code_tokens.append('\n')
                elif token.type == TokenType.INDENT:
                    code_tokens.append('    ')  # 4 spaces
                elif token.type == TokenType.DEDENT:
                    pass  # Skip dedent tokens in inline code
                else:
                    # For operators/keywords, use token name or symbol
                    token_map = {
                        TokenType.COLON: ':',
                        TokenType.COMMA: ',',
                        TokenType.DOT: '.',
                        TokenType.LPAREN: '(',
                        TokenType.RPAREN: ')',
                        TokenType.LBRACKET: '[',
                        TokenType.RBRACKET: ']',
                        TokenType.ARROW: '->',
                        TokenType.FAT_ARROW: '=>',
                        TokenType.PLUS: '+',
                        TokenType.MINUS: '-',
                        TokenType.STAR: '*',
                        TokenType.SLASH: '/',
                        TokenType.ASSIGN: '=',
                        TokenType.LT: '<',
                        TokenType.GT: '>',
                    }
                    code_tokens.append(token_map.get(token.type, token.type.name.lower()))
                
                prev_token_type = token.type
                self.advance()
        
        # Join tokens and strip leading/trailing whitespace from each line
        # but preserve relative indentation
        code = ''.join(code_tokens)
        
        # Strip leading/trailing blank lines
        lines = code.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        # Find minimum indentation (excluding blank lines)
        min_indent = float('inf')
        for line in lines:
            stripped = line.lstrip()
            if stripped:  # Non-blank line
                indent = len(line) - len(stripped)
                min_indent = min(min_indent, indent)
        
        # Remove common leading indentation
        if min_indent != float('inf') and min_indent > 0:
            lines = [line[min_indent:] if line.strip() else line for line in lines]
        
        return '\n'.join(lines)


__all__ = ["ExpressionParsingMixin"]
