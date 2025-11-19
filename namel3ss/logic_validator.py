"""
Validation for logic programming constructs in Namel3ss.

Validates knowledge modules and queries for:
- Predicate arity consistency
- Variable scoping
- Undefined knowledge module references
- Singleton variables (variables that appear only once)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

from namel3ss.ast.logic import (
    KnowledgeModule,
    LogicAtom,
    LogicFact,
    LogicList,
    LogicQuery,
    LogicRule,
    LogicStruct,
    LogicTerm,
    LogicVar,
)
from namel3ss.errors import N3Error


# ============================================================================
# Validation Errors
# ============================================================================

class LogicValidationError(N3Error):
    """Base exception for logic validation errors."""
    pass


class ArityMismatchError(LogicValidationError):
    """Raised when a predicate is used with inconsistent arity."""
    pass


class UndefinedPredicateError(LogicValidationError):
    """Raised when a predicate is used but never defined."""
    pass


class SingletonVariableWarning(LogicValidationError):
    """Warning for variables that appear only once (often typos)."""
    pass


class UndefinedModuleError(LogicValidationError):
    """Raised when a query references an undefined knowledge module."""
    pass


# ============================================================================
# Predicate Registry
# ============================================================================

class PredicateInfo:
    """Information about a predicate's usage."""
    
    def __init__(self, name: str, arity: int):
        self.name = name
        self.arity = arity
        self.definitions: List[int] = []  # Line numbers where defined
        self.usages: List[int] = []  # Line numbers where used
    
    def __repr__(self) -> str:
        return f"{self.name}/{self.arity}"


class PredicateRegistry:
    """Tracks predicate definitions and usages for validation."""
    
    def __init__(self):
        # Map: (predicate_name, arity) -> PredicateInfo
        self.predicates: Dict[tuple, PredicateInfo] = {}
    
    def register_definition(self, name: str, arity: int, line: int) -> None:
        """Register a predicate definition (fact or rule head)."""
        key = (name, arity)
        if key not in self.predicates:
            self.predicates[key] = PredicateInfo(name, arity)
        self.predicates[key].definitions.append(line)
    
    def register_usage(self, name: str, arity: int, line: int) -> None:
        """Register a predicate usage (in rule body or query)."""
        key = (name, arity)
        if key not in self.predicates:
            self.predicates[key] = PredicateInfo(name, arity)
        self.predicates[key].usages.append(line)
    
    def get_predicate(self, name: str, arity: int) -> PredicateInfo:
        """Get predicate info."""
        return self.predicates.get((name, arity))
    
    def check_arity_consistency(self) -> List[str]:
        """Check for predicates used with multiple arities."""
        errors = []
        
        # Group by predicate name
        by_name: Dict[str, List[PredicateInfo]] = defaultdict(list)
        for pred_info in self.predicates.values():
            by_name[pred_info.name].append(pred_info)
        
        # Check for inconsistent arities
        for name, pred_list in by_name.items():
            if len(pred_list) > 1:
                arities = sorted(set(p.arity for p in pred_list))
                lines = []
                for pred in pred_list:
                    lines.extend(pred.definitions)
                    lines.extend(pred.usages)
                
                errors.append(
                    f"Predicate '{name}' used with inconsistent arities: "
                    f"{arities} (lines: {sorted(set(lines))})"
                )
        
        return errors
    
    def check_undefined_predicates(self) -> List[str]:
        """Check for predicates used but never defined."""
        warnings = []
        
        for pred_info in self.predicates.values():
            if pred_info.usages and not pred_info.definitions:
                warnings.append(
                    f"Predicate '{pred_info}' used but never defined "
                    f"(lines: {pred_info.usages})"
                )
        
        return warnings


# ============================================================================
# Logic Validator
# ============================================================================

class LogicValidator:
    """
    Validator for logic programming constructs.
    
    Performs static analysis on knowledge modules and queries to detect
    common errors and potential issues.
    """
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.predicate_registry = PredicateRegistry()
        self.module_names: Set[str] = set()
    
    def validate_knowledge_module(self, module: KnowledgeModule) -> None:
        """Validate a knowledge module."""
        self.module_names.add(module.name)
        
        # Validate facts
        for fact in module.facts:
            self._validate_fact(fact)
        
        # Validate rules
        for rule in module.rules:
            self._validate_rule(rule)
    
    def validate_query(self, query: LogicQuery) -> None:
        """Validate a query."""
        # Check knowledge sources exist
        for source in query.knowledge_sources:
            if source not in self.module_names:
                self.errors.append(
                    f"Query '{query.name}' references undefined knowledge module: {source}"
                )
        
        # Validate goals
        for goal in query.goals:
            self._register_struct_usage(goal)
        
        # Check that projected variables actually appear in goals
        if query.variables:
            goal_vars = self._collect_variables_from_goals(query.goals)
            for var in query.variables:
                if var not in goal_vars:
                    self.warnings.append(
                        f"Query '{query.name}' projects variable '{var}' "
                        f"that doesn't appear in any goal"
                    )
    
    def _validate_fact(self, fact: LogicFact) -> None:
        """Validate a fact."""
        # Register predicate definition
        self.predicate_registry.register_definition(
            fact.head.functor,
            fact.head.arity,
            fact.line or 0
        )
        
        # Check that fact has no variables (should be ground)
        vars_in_fact = self._collect_variables_from_struct(fact.head)
        if vars_in_fact:
            self.warnings.append(
                f"Fact {fact.head} contains variables: {sorted(vars_in_fact)} "
                f"(line {fact.line}). Facts should be ground."
            )
    
    def _validate_rule(self, rule: LogicRule) -> None:
        """Validate a rule."""
        # Register head predicate definition
        self.predicate_registry.register_definition(
            rule.head.functor,
            rule.head.arity,
            rule.line or 0
        )
        
        # Register body predicate usages
        for goal in rule.body:
            self._register_struct_usage(goal)
        
        # Check variable safety: all head variables must appear in body
        head_vars = self._collect_variables_from_struct(rule.head)
        body_vars = set()
        for goal in rule.body:
            body_vars.update(self._collect_variables_from_struct(goal))
        
        unsafe_vars = head_vars - body_vars
        if unsafe_vars:
            self.errors.append(
                f"Rule {rule.head} has unsafe variables in head: {sorted(unsafe_vars)} "
                f"(line {rule.line}). All head variables must appear in the body."
            )
        
        # Check for singleton variables (appear only once)
        all_vars = head_vars | body_vars
        var_counts: Dict[str, int] = defaultdict(int)
        
        for var in self._collect_all_variable_occurrences_rule(rule):
            var_counts[var] += 1
        
        singleton_vars = [v for v in all_vars if var_counts[v] == 1 and not v.startswith('_')]
        if singleton_vars:
            self.warnings.append(
                f"Rule {rule.head} has singleton variables: {sorted(singleton_vars)} "
                f"(line {rule.line}). These may be typos."
            )
    
    def _register_struct_usage(self, struct: LogicStruct) -> None:
        """Register a structure as a predicate usage."""
        self.predicate_registry.register_usage(
            struct.functor,
            struct.arity,
            struct.line or 0
        )
    
    def _collect_variables_from_struct(self, struct: LogicStruct) -> Set[str]:
        """Collect all variable names from a structure."""
        variables = set()
        self._collect_vars_recursive(struct, variables)
        return variables
    
    def _collect_variables_from_goals(self, goals: List[LogicStruct]) -> Set[str]:
        """Collect all variable names from a list of goals."""
        variables = set()
        for goal in goals:
            variables.update(self._collect_variables_from_struct(goal))
        return variables
    
    def _collect_vars_recursive(self, term: LogicTerm, variables: Set[str]) -> None:
        """Recursively collect variable names from a term."""
        if isinstance(term, LogicVar):
            variables.add(term.name)
        elif isinstance(term, LogicStruct):
            for arg in term.args:
                self._collect_vars_recursive(arg, variables)
        elif isinstance(term, LogicList):
            for elem in term.elements:
                self._collect_vars_recursive(elem, variables)
            if term.tail:
                self._collect_vars_recursive(term.tail, variables)
    
    def _collect_all_variable_occurrences_rule(self, rule: LogicRule) -> List[str]:
        """Collect all variable occurrences (with duplicates) from a rule."""
        occurrences = []
        
        # Head
        self._collect_var_occurrences(rule.head, occurrences)
        
        # Body
        for goal in rule.body:
            self._collect_var_occurrences(goal, occurrences)
        
        return occurrences
    
    def _collect_var_occurrences(self, term: LogicTerm, occurrences: List[str]) -> None:
        """Recursively collect variable occurrences from a term."""
        if isinstance(term, LogicVar):
            occurrences.append(term.name)
        elif isinstance(term, LogicStruct):
            for arg in term.args:
                self._collect_var_occurrences(arg, occurrences)
        elif isinstance(term, LogicList):
            for elem in term.elements:
                self._collect_var_occurrences(elem, occurrences)
            if term.tail:
                self._collect_var_occurrences(term.tail, occurrences)
    
    def finalize(self) -> None:
        """Perform final cross-module validation checks."""
        # Check arity consistency
        arity_errors = self.predicate_registry.check_arity_consistency()
        self.errors.extend(arity_errors)
        
        # Check undefined predicates
        undefined_warnings = self.predicate_registry.check_undefined_predicates()
        self.warnings.extend(undefined_warnings)
    
    def get_results(self) -> tuple[List[str], List[str]]:
        """Get validation errors and warnings."""
        return self.errors, self.warnings
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0


# ============================================================================
# Validation Function
# ============================================================================

def validate_logic_constructs(
    knowledge_modules: List[KnowledgeModule],
    queries: List[LogicQuery],
) -> tuple[List[str], List[str]]:
    """
    Validate knowledge modules and queries.
    
    Returns:
        Tuple of (errors, warnings)
    """
    validator = LogicValidator()
    
    # Validate knowledge modules
    for module in knowledge_modules:
        validator.validate_knowledge_module(module)
    
    # Validate queries
    for query in queries:
        validator.validate_query(query)
    
    # Finalize validation
    validator.finalize()
    
    return validator.get_results()


__all__ = [
    "LogicValidator",
    "LogicValidationError",
    "ArityMismatchError",
    "UndefinedPredicateError",
    "UndefinedModuleError",
    "validate_logic_constructs",
]
