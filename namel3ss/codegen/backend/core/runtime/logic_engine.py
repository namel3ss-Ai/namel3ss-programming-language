"""
Logic engine for Namel3ss knowledge and inference system.

Provides Prolog-style unification and backtracking search with safety limits.
This is a production-grade implementation with proper error handling and
configurable resource limits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Set, Union

from namel3ss.ast.logic import (
    LogicAtom,
    LogicFact,
    LogicList,
    LogicNumber,
    LogicRule,
    LogicString,
    LogicStruct,
    LogicTerm,
    LogicVar,
)


# ============================================================================
# Substitution (Variable Bindings)
# ============================================================================

@dataclass
class Substitution:
    """
    Represents a set of variable bindings.
    
    Maps variable names to their bound terms.
    Immutable substitutions enable backtracking without mutation.
    """
    bindings: Dict[str, LogicTerm] = field(default_factory=dict)
    
    def bind(self, var_name: str, term: LogicTerm) -> Substitution:
        """Create a new substitution with an additional binding."""
        new_bindings = self.bindings.copy()
        new_bindings[var_name] = term
        return Substitution(new_bindings)
    
    def lookup(self, var_name: str) -> Optional[LogicTerm]:
        """Look up a variable's binding."""
        return self.bindings.get(var_name)
    
    def apply(self, term: LogicTerm) -> LogicTerm:
        """Apply substitution to a term, resolving variable references."""
        if isinstance(term, LogicVar):
            binding = self.lookup(term.name)
            if binding is not None:
                # Follow chains of bindings
                return self.apply(binding)
            return term
        elif isinstance(term, LogicStruct):
            # Apply to all arguments
            new_args = [self.apply(arg) for arg in term.args]
            result = LogicStruct(
                functor=term.functor,
                args=new_args,
                line=term.line,
                column=term.column
            )
            return result
        elif isinstance(term, LogicList):
            # Apply to all elements and tail
            new_elements = [self.apply(elem) for elem in term.elements]
            new_tail = self.apply(term.tail) if term.tail else None
            return LogicList(
                elements=new_elements,
                tail=new_tail,
                line=term.line,
                column=term.column
            )
        else:
            # Atoms, numbers, strings are ground terms
            return term
    
    def __str__(self) -> str:
        if not self.bindings:
            return "{}"
        items = [f"{k}={v}" for k, v in self.bindings.items()]
        return "{" + ", ".join(items) + "}"


# ============================================================================
# Unification
# ============================================================================

def occurs_check(var: LogicVar, term: LogicTerm, subst: Substitution) -> bool:
    """
    Check if a variable occurs in a term (prevents infinite structures).
    
    Returns True if the variable occurs in the term.
    """
    term = subst.apply(term)
    
    if isinstance(term, LogicVar):
        return term.name == var.name
    elif isinstance(term, LogicStruct):
        return any(occurs_check(var, arg, subst) for arg in term.args)
    elif isinstance(term, LogicList):
        for elem in term.elements:
            if occurs_check(var, elem, subst):
                return True
        if term.tail and occurs_check(var, term.tail, subst):
            return True
        return False
    else:
        return False


def unify(term1: LogicTerm, term2: LogicTerm, subst: Substitution) -> Optional[Substitution]:
    """
    Unify two terms under the given substitution.
    
    Returns an updated substitution if unification succeeds, None otherwise.
    Implements the classic unification algorithm with occurs check.
    """
    # Apply current substitution to both terms
    term1 = subst.apply(term1)
    term2 = subst.apply(term2)
    
    # Case 1: Both are the same term
    if term1 is term2:
        return subst
    
    # Case 2: Variable unification
    if isinstance(term1, LogicVar):
        if occurs_check(term1, term2, subst):
            return None  # Would create infinite structure
        return subst.bind(term1.name, term2)
    
    if isinstance(term2, LogicVar):
        if occurs_check(term2, term1, subst):
            return None
        return subst.bind(term2.name, term1)
    
    # Case 3: Atom unification
    if isinstance(term1, LogicAtom) and isinstance(term2, LogicAtom):
        return subst if term1.value == term2.value else None
    
    # Case 4: Number unification
    if isinstance(term1, LogicNumber) and isinstance(term2, LogicNumber):
        return subst if term1.value == term2.value else None
    
    # Case 5: String unification
    if isinstance(term1, LogicString) and isinstance(term2, LogicString):
        return subst if term1.value == term2.value else None
    
    # Case 6: Structure unification
    if isinstance(term1, LogicStruct) and isinstance(term2, LogicStruct):
        # Must have same functor and arity
        if term1.functor != term2.functor or len(term1.args) != len(term2.args):
            return None
        
        # Unify arguments left to right
        current_subst = subst
        for arg1, arg2 in zip(term1.args, term2.args):
            current_subst = unify(arg1, arg2, current_subst)
            if current_subst is None:
                return None
        return current_subst
    
    # Case 7: List unification
    if isinstance(term1, LogicList) and isinstance(term2, LogicList):
        # Handle list patterns [H|T]
        if term1.tail or term2.tail:
            # TODO: Implement full list pattern matching
            # For now, require both to be patterns or neither
            if bool(term1.tail) != bool(term2.tail):
                return None
            
            # Unify elements
            if len(term1.elements) != len(term2.elements):
                return None
            
            current_subst = subst
            for elem1, elem2 in zip(term1.elements, term2.elements):
                current_subst = unify(elem1, elem2, current_subst)
                if current_subst is None:
                    return None
            
            # Unify tails if present
            if term1.tail and term2.tail:
                current_subst = unify(term1.tail, term2.tail, current_subst)
            
            return current_subst
        
        # Regular lists: must have same length
        if len(term1.elements) != len(term2.elements):
            return None
        
        current_subst = subst
        for elem1, elem2 in zip(term1.elements, term2.elements):
            current_subst = unify(elem1, elem2, current_subst)
            if current_subst is None:
                return None
        return current_subst
    
    # Different types don't unify
    return None


# ============================================================================
# Logic Engine
# ============================================================================

@dataclass
class LogicEngineConfig:
    """Configuration for logic engine safety limits."""
    max_depth: int = 100  # Maximum recursion depth
    max_steps: int = 10000  # Maximum inference steps
    timeout_seconds: float = 10.0  # Timeout for queries


class LogicEngineError(Exception):
    """Base exception for logic engine errors."""
    pass


class LogicEngineTimeout(LogicEngineError):
    """Raised when query execution exceeds timeout."""
    pass


class LogicEngineDepthLimit(LogicEngineError):
    """Raised when recursion depth limit is exceeded."""
    pass


class LogicEngineStepLimit(LogicEngineError):
    """Raised when inference step limit is exceeded."""
    pass


class LogicEngine:
    """
    Production-grade logic engine with unification and backtracking.
    
    Implements depth-first search with backtracking over facts and rules.
    Includes safety limits to prevent infinite loops and resource exhaustion.
    """
    
    def __init__(self, config: Optional[LogicEngineConfig] = None):
        """Initialize the logic engine with optional configuration."""
        self.config = config or LogicEngineConfig()
        self._step_count = 0
        self._start_time = 0.0
    
    def solve(
        self,
        goals: List[LogicStruct],
        facts: List[LogicFact],
        rules: List[LogicRule],
        initial_subst: Optional[Substitution] = None,
    ) -> Iterator[Substitution]:
        """
        Solve a conjunction of goals using the given facts and rules.
        
        Yields substitutions that satisfy all goals.
        Uses depth-first search with backtracking.
        """
        self._step_count = 0
        self._start_time = time.time()
        
        if initial_subst is None:
            initial_subst = Substitution()
        
        # Solve goal conjunction
        yield from self._solve_goals(goals, facts, rules, initial_subst, depth=0)
    
    def _solve_goals(
        self,
        goals: List[LogicStruct],
        facts: List[LogicFact],
        rules: List[LogicRule],
        subst: Substitution,
        depth: int,
    ) -> Iterator[Substitution]:
        """
        Solve a conjunction of goals (all must succeed).
        
        Uses depth-first search with backtracking.
        """
        # Check safety limits
        self._check_limits(depth)
        
        # Base case: no more goals
        if not goals:
            yield subst
            return
        
        # Recursive case: solve first goal, then remaining goals
        first_goal = goals[0]
        remaining_goals = goals[1:]
        
        # Try to solve first goal
        for new_subst in self._solve_goal(first_goal, facts, rules, subst, depth):
            # Recursively solve remaining goals with new substitution
            yield from self._solve_goals(remaining_goals, facts, rules, new_subst, depth)
    
    def _solve_goal(
        self,
        goal: LogicStruct,
        facts: List[LogicFact],
        rules: List[LogicRule],
        subst: Substitution,
        depth: int,
    ) -> Iterator[Substitution]:
        """
        Solve a single goal by unifying with facts and rules.
        
        Tries facts first, then rules.
        """
        # Apply current substitution to goal
        goal = subst.apply(goal)
        
        # Try to unify with facts
        for fact in facts:
            self._step_count += 1
            self._check_limits(depth)
            
            # Rename variables in fact to avoid conflicts
            renamed_fact = self._rename_term(fact.head, {}, depth)
            
            # Try to unify goal with fact
            new_subst = unify(goal, renamed_fact, subst)
            if new_subst is not None:
                yield new_subst
        
        # Try to unify with rules
        for rule in rules:
            self._step_count += 1
            self._check_limits(depth)
            
            # Rename variables in rule to avoid conflicts
            var_map: Dict[str, str] = {}
            renamed_head = self._rename_term(rule.head, var_map, depth)
            renamed_body = [self._rename_term(g, var_map, depth) for g in rule.body]
            
            # Try to unify goal with rule head
            new_subst = unify(goal, renamed_head, subst)
            if new_subst is not None:
                # Recursively solve rule body
                yield from self._solve_goals(renamed_body, facts, rules, new_subst, depth + 1)
    
    def _rename_term(
        self,
        term: LogicTerm,
        var_map: Dict[str, str],
        depth: int,
    ) -> LogicTerm:
        """
        Rename variables in a term to avoid conflicts.
        
        Creates fresh variable names based on depth level.
        """
        if isinstance(term, LogicVar):
            if term.name not in var_map:
                var_map[term.name] = f"{term.name}_{depth}"
            return LogicVar(name=var_map[term.name], line=term.line, column=term.column)
        elif isinstance(term, LogicStruct):
            return LogicStruct(
                functor=term.functor,
                args=[self._rename_term(arg, var_map, depth) for arg in term.args],
                line=term.line,
                column=term.column,
            )
        elif isinstance(term, LogicList):
            new_elements = [self._rename_term(elem, var_map, depth) for elem in term.elements]
            new_tail = self._rename_term(term.tail, var_map, depth) if term.tail else None
            return LogicList(
                elements=new_elements,
                tail=new_tail,
                line=term.line,
                column=term.column,
            )
        else:
            # Ground terms don't need renaming
            return term
    
    def _check_limits(self, depth: int) -> None:
        """Check if safety limits have been exceeded."""
        # Check depth limit
        if depth > self.config.max_depth:
            raise LogicEngineDepthLimit(
                f"Recursion depth limit exceeded: {depth} > {self.config.max_depth}"
            )
        
        # Check step limit
        if self._step_count > self.config.max_steps:
            raise LogicEngineStepLimit(
                f"Inference step limit exceeded: {self._step_count} > {self.config.max_steps}"
            )
        
        # Check timeout
        elapsed = time.time() - self._start_time
        if elapsed > self.config.timeout_seconds:
            raise LogicEngineTimeout(
                f"Query timeout: {elapsed:.2f}s > {self.config.timeout_seconds}s"
            )


__all__ = [
    "Substitution",
    "unify",
    "occurs_check",
    "LogicEngine",
    "LogicEngineConfig",
    "LogicEngineError",
    "LogicEngineTimeout",
    "LogicEngineDepthLimit",
    "LogicEngineStepLimit",
]
