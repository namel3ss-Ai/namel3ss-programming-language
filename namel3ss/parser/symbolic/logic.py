"""Logic programming constructs for rules and queries."""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ...ast.expressions import RuleDef, RuleHead, RuleBody, RuleClause, QueryExpr
    from ...ast.base import Expression

from ..base import N3SyntaxError


class LogicParserMixin:
    """Mixin for parsing logic programming rules and queries."""
    
    def parse_rule_def(self) -> "RuleDef":
        """
        Parse logic programming rule definition.
        
        Syntax: 
            - Facts: rule parent(alice, bob).
            - Rules: rule ancestor(X, Y) :- parent(X, Y), parent(Y, Z).
        
        Rules define logical relationships and inference patterns for
        declarative knowledge representation.
        """
        from namel3ss.ast.expressions import RuleDef, RuleHead, RuleBody, RuleClause
        
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
    
    def parse_query_expr(self) -> "QueryExpr":
        """
        Parse logic query expression.
        
        Syntax: query predicate(args) or query predicate(args) limit n
        
        Queries search for solutions that satisfy logical predicates,
        optionally limited to a maximum number of results.
        """
        from namel3ss.ast.expressions import QueryExpr
        
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
