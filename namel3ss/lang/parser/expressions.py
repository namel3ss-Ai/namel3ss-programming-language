"""Expression and value parsing methods for N3Parser.

Handles expressions, literals, operators, and complex values.
"""

from typing import Any, Dict, List, Optional
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
        special_handlers: Optional[Dict[str, Any]] = None
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
        
        while not self.match(TokenType.RBRACE):
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
            if key_token.type in self._STRUCTURAL_TOKENS:
                raise create_syntax_error(
                    "Expected field name in block",
                    path=self.path,
                    line=key_token.line,
                    column=key_token.column,
                    expected="identifier or keyword",
                    found=key_token.type.name.lower()
                )
            
            # Handle various token types as keys
            if key_token.type == TokenType.IDENTIFIER:
                key = key_token.value
                self.advance()
            elif allow_any_keyword:
                # In allow_any_keyword mode, any non-structural token is valid
                key = key_token.value.lower() if hasattr(key_token, 'value') and key_token.value else key_token.type.name.lower()
                self.advance()
            elif key_token.type in (TokenType.MODEL, TokenType.FILTER, TokenType.INDEX, 
                                     TokenType.MEMORY, TokenType.CHAIN):
                # In strict mode, only specific keywords are allowed
                key = key_token.value.lower() if hasattr(key_token, 'value') and key_token.value else key_token.type.name.lower()
                self.advance()
            else:
                # In strict mode, reject other keywords
                raise create_syntax_error(
                    "Expected field name in block",
                    path=self.path,
                    line=key_token.line,
                    column=key_token.column,
                    expected="identifier or keyword",
                    found=key_token.type.name.lower()
                )
            
            self.expect(TokenType.COLON)
            
            # Parse value - use special handler if one is registered for this key
            if key in special_handlers:
                value = special_handlers[key]()
            else:
                value = self.parse_value()
            
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
    
    def parse_value(self) -> Any:
        """
        Parse a value (literal, array, object, expression).
        
        Grammar:
            Value = Literal | ArrayValue | ObjectValue | Expression ;
        """
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
            return self.advance().value
        
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
        """Parse a page statement (show, if, for, etc.)."""
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
        
        # Expression statement
        return self.parse_expression()
    
    def parse_show_statement(self) -> Dict[str, Any]:
        """Parse show statement."""
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
        
        config = {}
        if self.consume_if(TokenType.COLON):
            self.skip_newlines()
            config = self.parse_value()
        elif self.match(TokenType.LBRACE):
            config = self.parse_block()
        
        return {
            "type": "show",
            "component": component_type,
            "config": config,
        }
    
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
        
        # Could be an inline expression
        return self.parse_expression()


__all__ = ["ExpressionParsingMixin"]
