"""Pattern matching parser for match expressions."""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ...ast.expressions import MatchExpr, MatchCase, Pattern
    from ...ast.base import Expression


class PatternParserMixin:
    """Mixin for parsing pattern matching expressions."""
    
    def parse_match_expr(self) -> "MatchExpr":
        """
        Parse match expression.
        
        Syntax: 
            match value {
                pattern1 => result1,
                pattern2 if guard => result2,
                _ => default
            }
        
        Pattern matching provides structural decomposition and conditional
        branching based on data shape and content.
        """
        from namel3ss.ast.expressions import MatchExpr, MatchCase
        
        self.expect("match")
        value = self.parse_extended_expression()
        
        self.expect("{")
        
        cases: List[MatchCase] = []
        while not self.try_consume("}"):
            if cases:
                self.expect(",")
            
            pattern = self.parse_pattern()
            
            # Optional guard
            guard = None
            if self.try_consume("if"):
                guard = self.parse_extended_expression()
            
            self.expect("=>")
            result = self.parse_extended_expression()
            
            cases.append(MatchCase(pattern=pattern, guard=guard, result=result))
        
        return MatchExpr(value=value, cases=cases)
    
    def parse_pattern(self) -> "Pattern":
        """
        Parse a pattern for destructuring.
        
        Supports:
            - Wildcard: _
            - Variable: x
            - Literal: 42, "hello", True
            - List: [x, y, ...rest]
            - Dict: {key: value, ...rest}
            - Tuple: (x, y, z)
            - Constructor: Point(x, y)
        """
        from namel3ss.ast.expressions import Pattern
        
        # Wildcard
        if self.try_consume("_"):
            return Pattern(kind="wildcard")
        
        # List pattern
        if self.try_consume("["):
            elements: List[Pattern] = []
            rest_var = None
            
            while not self.try_consume("]"):
                if elements or rest_var:
                    self.expect(",")
                
                if self.try_consume("..."):
                    rest_var = self.word()
                    self.expect("]")
                    break
                
                elements.append(self.parse_pattern())
            
            return Pattern(kind="list", elements=elements, rest=rest_var)
        
        # Dict pattern
        if self.try_consume("{"):
            pairs: dict[str, Pattern] = {}
            rest_var = None
            
            while not self.try_consume("}"):
                if pairs or rest_var:
                    self.expect(",")
                
                if self.try_consume("..."):
                    rest_var = self.word()
                    self.expect("}")
                    break
                
                key = self.word()
                self.expect(":")
                value = self.parse_pattern()
                pairs[key] = value
            
            return Pattern(kind="dict", pairs=pairs, rest=rest_var)
        
        # Tuple pattern
        if self.try_consume("("):
            elements: List[Pattern] = []
            while not self.try_consume(")"):
                if elements:
                    self.expect(",")
                elements.append(self.parse_pattern())
            
            if len(elements) == 1:
                return elements[0]  # Grouped pattern
            return Pattern(kind="tuple", elements=elements)
        
        # Literal patterns
        if self.peek() and self.peek() in ('"', "'"):
            return Pattern(kind="literal", value=self.string())
        
        if self.peek() and (self.peek().isdigit() or self.peek() == "-"):
            return Pattern(kind="literal", value=self.number())
        
        # Variable or constructor
        name = self.word()
        
        # Check for boolean/null literals
        if name in ("True", "true"):
            return Pattern(kind="literal", value=True)
        if name in ("False", "false"):
            return Pattern(kind="literal", value=False)
        if name in ("None", "null"):
            return Pattern(kind="literal", value=None)
        
        # Check for constructor pattern
        if self.try_consume("("):
            args: List[Pattern] = []
            while not self.try_consume(")"):
                if args:
                    self.expect(",")
                args.append(self.parse_pattern())
            return Pattern(kind="constructor", name=name, args=args)
        
        # Simple variable pattern
        return Pattern(kind="variable", name=name)
