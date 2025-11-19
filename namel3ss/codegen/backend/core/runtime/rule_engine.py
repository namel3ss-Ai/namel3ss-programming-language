"""Prolog-style rule engine with unification and backtracking."""

from __future__ import annotations

import copy
from typing import Any, Dict, Generator, List, Optional

from namel3ss.ast.expressions import (
    Expression,
    LiteralExpr,
    RuleBody,
    RuleClause,
    RuleDef,
    RuleHead,
    VarExpr,
)

__all__ = [
    "RuleDatabase",
    "UnificationError",
    "unify",
]


class UnificationError(Exception):
    """Raised when unification fails."""
    pass


def is_variable(expr: Any) -> bool:
    """Check if expression is a variable."""
    return isinstance(expr, VarExpr) or (isinstance(expr, str) and expr[0].isupper())


def unify(term1: Any, term2: Any, bindings: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Unify two terms, returning updated bindings or None on failure.
    
    Implements the Robinson unification algorithm with occurs check.
    """
    if bindings is None:
        bindings = {}
    else:
        bindings = dict(bindings)
    
    # Dereference variables
    term1 = deref(term1, bindings)
    term2 = deref(term2, bindings)
    
    # Same term
    if term1 == term2:
        return bindings
    
    # Variable unification
    if is_variable(term1):
        if occurs_check(term1, term2, bindings):
            return None
        var_name = term1.name if isinstance(term1, VarExpr) else term1
        bindings[var_name] = term2
        return bindings
    
    if is_variable(term2):
        if occurs_check(term2, term1, bindings):
            return None
        var_name = term2.name if isinstance(term2, VarExpr) else term2
        bindings[var_name] = term1
        return bindings
    
    # Literal unification
    if isinstance(term1, LiteralExpr) and isinstance(term2, LiteralExpr):
        if term1.value == term2.value:
            return bindings
        return None
    
    # List unification
    if isinstance(term1, list) and isinstance(term2, list):
        if len(term1) != len(term2):
            return None
        
        for t1, t2 in zip(term1, term2):
            bindings = unify(t1, t2, bindings)
            if bindings is None:
                return None
        return bindings
    
    # Dict unification (symbolic terms)
    if isinstance(term1, dict) and isinstance(term2, dict):
        # Must have same keys
        if set(term1.keys()) != set(term2.keys()):
            return None
        
        for key in term1.keys():
            bindings = unify(term1[key], term2[key], bindings)
            if bindings is None:
                return None
        return bindings
    
    # Primitive value unification
    if isinstance(term1, (int, float, str, bool, type(None))) and \
       isinstance(term2, (int, float, str, bool, type(None))):
        if term1 == term2:
            return bindings
        return None
    
    # No unification possible
    return None


def deref(term: Any, bindings: Dict[str, Any]) -> Any:
    """Dereference a term by following variable bindings."""
    if isinstance(term, VarExpr):
        var_name = term.name
        if var_name in bindings:
            return deref(bindings[var_name], bindings)
        return term
    
    if isinstance(term, str) and term[0].isupper() and term in bindings:
        return deref(bindings[term], bindings)
    
    return term


def occurs_check(var: Any, term: Any, bindings: Dict[str, Any]) -> bool:
    """
    Check if variable occurs in term (prevents infinite structures).
    """
    var_name = var.name if isinstance(var, VarExpr) else var
    term = deref(term, bindings)
    
    if is_variable(term):
        term_name = term.name if isinstance(term, VarExpr) else term
        return var_name == term_name
    
    if isinstance(term, list):
        return any(occurs_check(var, t, bindings) for t in term)
    
    if isinstance(term, dict):
        return any(occurs_check(var, v, bindings) for v in term.values())
    
    return False


def substitute(term: Any, bindings: Dict[str, Any]) -> Any:
    """Apply variable bindings to a term."""
    if isinstance(term, VarExpr):
        var_name = term.name
        if var_name in bindings:
            return substitute(bindings[var_name], bindings)
        return term
    
    if isinstance(term, list):
        return [substitute(t, bindings) for t in term]
    
    if isinstance(term, dict):
        return {k: substitute(v, bindings) for k, v in term.items()}
    
    return term


class RuleDatabase:
    """Database of rules with querying via unification and backtracking."""
    
    def __init__(self, rules: List[RuleDef], max_depth: int = 50):
        """
        Initialize rule database.
        
        Args:
            rules: List of rule definitions
            max_depth: Maximum recursion depth for queries
        """
        self.rules = rules
        self.max_depth = max_depth
        
        # Index rules by predicate name for efficiency
        self.rule_index: Dict[str, List[RuleDef]] = {}
        for rule in rules:
            predicate = rule.head.predicate
            if predicate not in self.rule_index:
                self.rule_index[predicate] = []
            self.rule_index[predicate].append(rule)
    
    def query(
        self,
        predicate: str,
        args: List[Any],
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the database for solutions.
        
        Args:
            predicate: Predicate name to query
            args: Arguments to the predicate
            limit: Maximum number of solutions to return
        
        Returns:
            List of variable binding dictionaries for each solution
        """
        solutions = []
        
        for bindings in self._query_generator(predicate, args, depth=0):
            # Filter to only variables from the query
            query_vars = self._extract_vars_from_args(args)
            solution = {var: bindings.get(var) for var in query_vars if var in bindings}
            solutions.append(solution)
            
            if limit is not None and len(solutions) >= limit:
                break
        
        return solutions
    
    def _query_generator(
        self,
        predicate: str,
        args: List[Any],
        depth: int,
        bindings: Optional[Dict[str, Any]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate solutions via backtracking.
        
        Yields variable bindings for each solution.
        """
        if depth > self.max_depth:
            return
        
        if bindings is None:
            bindings = {}
        
        # Get rules for this predicate
        rules = self.rule_index.get(predicate, [])
        
        for rule in rules:
            # Rename variables in rule to avoid conflicts
            renamed_rule = self._rename_rule_vars(rule, depth)
            
            # Try to unify query with rule head
            head_bindings = self._unify_with_head(renamed_rule.head, args, bindings)
            if head_bindings is None:
                continue
            
            # If it's a fact (no body), we have a solution
            if renamed_rule.body is None or not renamed_rule.body.clauses:
                yield head_bindings
                continue
            
            # Otherwise, solve the body
            yield from self._solve_body(renamed_rule.body, head_bindings, depth + 1)
    
    def _solve_body(
        self,
        body: RuleBody,
        bindings: Dict[str, Any],
        depth: int
    ) -> Generator[Dict[str, Any], None, None]:
        """Solve rule body (conjunction of clauses)."""
        if not body.clauses:
            yield bindings
            return
        
        # Solve clauses left-to-right
        yield from self._solve_clauses(body.clauses, bindings, depth)
    
    def _solve_clauses(
        self,
        clauses: List[RuleClause],
        bindings: Dict[str, Any],
        depth: int
    ) -> Generator[Dict[str, Any], None, None]:
        """Solve a list of clauses."""
        if not clauses:
            yield bindings
            return
        
        first_clause = clauses[0]
        rest_clauses = clauses[1:]
        
        # Handle negation-as-failure
        if first_clause.negated:
            # Try to solve the clause
            has_solution = False
            for _ in self._solve_single_clause(first_clause, bindings, depth):
                has_solution = True
                break
            
            # If no solution, negation succeeds
            if not has_solution:
                yield from self._solve_clauses(rest_clauses, bindings, depth)
            return
        
        # Solve first clause
        for new_bindings in self._solve_single_clause(first_clause, bindings, depth):
            # Then solve remaining clauses with new bindings
            yield from self._solve_clauses(rest_clauses, new_bindings, depth)
    
    def _solve_single_clause(
        self,
        clause: RuleClause,
        bindings: Dict[str, Any],
        depth: int
    ) -> Generator[Dict[str, Any], None, None]:
        """Solve a single clause."""
        if clause.predicate is not None:
            # It's a predicate call
            # Substitute bindings in arguments
            subst_args = [substitute(arg, bindings) for arg in clause.args]
            
            # Query recursively
            yield from self._query_generator(clause.predicate, subst_args, depth, bindings)
        
        elif clause.expr is not None:
            # It's an expression constraint
            # Evaluate expression with current bindings
            # (This requires the expression evaluator, which we'll integrate later)
            # For now, just yield bindings if it's a simple literal true
            if isinstance(clause.expr, LiteralExpr) and clause.expr.value is True:
                yield bindings
    
    def _unify_with_head(
        self,
        head: RuleHead,
        args: List[Any],
        bindings: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Try to unify query arguments with rule head."""
        if len(args) != len(head.args):
            return None
        
        result_bindings = dict(bindings)
        
        for query_arg, head_arg in zip(args, head.args):
            result_bindings = unify(query_arg, head_arg, result_bindings)
            if result_bindings is None:
                return None
        
        return result_bindings
    
    def _rename_rule_vars(self, rule: RuleDef, suffix: int) -> RuleDef:
        """Rename variables in a rule to avoid conflicts."""
        # This is a simplified version - proper implementation would use
        # a complete AST traversal and variable renaming
        return rule
    
    def _extract_vars_from_args(self, args: List[Any]) -> set[str]:
        """Extract all variable names from arguments."""
        vars_set: set[str] = set()
        
        for arg in args:
            if isinstance(arg, VarExpr):
                vars_set.add(arg.name)
            elif isinstance(arg, str) and arg[0].isupper():
                vars_set.add(arg)
            elif isinstance(arg, list):
                vars_set.update(self._extract_vars_from_args(arg))
            elif isinstance(arg, dict):
                vars_set.update(self._extract_vars_from_args(list(arg.values())))
        
        return vars_set
