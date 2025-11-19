"""
AST nodes for logic programming constructs in Namel3ss.

Provides first-class support for:
- Logic terms (variables, atoms, numbers, strings, compound structures)
- Facts and rules (knowledge representation)
- Goals and queries (inference and reasoning)
- Knowledge modules (namespaced collections of facts and rules)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# ============================================================================
# Logic Terms
# ============================================================================

@dataclass
class LogicTerm:
    """Base class for all logic terms."""
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class LogicVar(LogicTerm):
    """A logic variable (e.g., X, Y, _Result)."""
    name: str = ""
    
    def __str__(self) -> str:
        return self.name


@dataclass
class LogicAtom(LogicTerm):
    """An atom (symbolic constant, e.g., alice, true, nil)."""
    value: str = ""
    
    def __str__(self) -> str:
        return self.value


@dataclass
class LogicNumber(LogicTerm):
    """A numeric literal."""
    value: Union[int, float] = 0
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass
class LogicString(LogicTerm):
    """A string literal."""
    value: str = ""
    
    def __str__(self) -> str:
        return f'"{self.value}"'


@dataclass
class LogicStruct(LogicTerm):
    """
    A compound term / structure (e.g., parent(alice, bob), point(1, 2, 3)).
    
    functor: The function symbol (predicate name)
    args: List of argument terms
    """
    functor: str = ""
    args: List[LogicTerm] = field(default_factory=list)
    
    @property
    def arity(self) -> int:
        """Return the arity (number of arguments) of this structure."""
        return len(self.args)
    
    def __str__(self) -> str:
        if not self.args:
            return self.functor
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.functor}({args_str})"


@dataclass
class LogicList(LogicTerm):
    """
    A list term (e.g., [1, 2, 3] or [H|T]).
    
    elements: List of element terms
    tail: Optional tail variable for list patterns like [H|T]
    """
    elements: List[LogicTerm] = field(default_factory=list)
    tail: Optional[LogicVar] = None
    
    def __str__(self) -> str:
        if self.tail:
            elements_str = ", ".join(str(e) for e in self.elements)
            return f"[{elements_str}|{self.tail}]"
        elements_str = ", ".join(str(e) for e in self.elements)
        return f"[{elements_str}]"


# ============================================================================
# Knowledge Representation
# ============================================================================

@dataclass
class LogicFact:
    """
    A ground fact (e.g., parent(alice, bob)).
    
    This represents an assertion of knowledge without conditions.
    """
    head: LogicStruct = field(default_factory=lambda: LogicStruct())
    metadata: Dict[str, Any] = field(default_factory=dict)
    line: Optional[int] = None
    column: Optional[int] = None
    
    def __str__(self) -> str:
        return f"{self.head}."


@dataclass
class LogicRule:
    """
    A rule with head and body (e.g., ancestor(X, Y) :- parent(X, Y)).
    
    head: The consequent (what is concluded)
    body: List of goals (conditions that must be satisfied)
    """
    head: LogicStruct = field(default_factory=lambda: LogicStruct())
    body: List[LogicStruct] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    line: Optional[int] = None
    column: Optional[int] = None
    
    def __str__(self) -> str:
        if not self.body:
            return f"{self.head}."
        body_str = ", ".join(str(goal) for goal in self.body)
        return f"{self.head} :- {body_str}."


# ============================================================================
# Goals and Queries
# ============================================================================

@dataclass
class LogicGoal:
    """
    A goal to be satisfied (single or conjunction of predicates).
    
    goals: List of predicates that form a conjunction (all must succeed)
    """
    goals: List[LogicStruct] = field(default_factory=list)
    line: Optional[int] = None
    column: Optional[int] = None
    
    def __str__(self) -> str:
        return ", ".join(str(g) for g in self.goals)


@dataclass
class LogicQuery:
    """
    A top-level query construct for executing logic programs.
    
    name: Query identifier
    knowledge_sources: Names of knowledge modules to use
    goals: Goals to satisfy
    limit: Optional limit on number of solutions
    variables: Variables to project in results (if None, all vars)
    """
    name: str = ""
    knowledge_sources: List[str] = field(default_factory=list)
    goals: List[LogicStruct] = field(default_factory=list)
    limit: Optional[int] = None
    variables: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    line: Optional[int] = None
    column: Optional[int] = None
    
    def __str__(self) -> str:
        sources = ", ".join(self.knowledge_sources)
        goals_str = ", ".join(str(g) for g in self.goals)
        return f"query {self.name} from [{sources}]: {goals_str}"


# ============================================================================
# Knowledge Modules
# ============================================================================

@dataclass
class KnowledgeModule:
    """
    A collection of facts and rules forming a knowledge base.
    
    name: Module identifier
    facts: Ground facts
    rules: Inference rules
    imports: Other knowledge modules to import
    """
    name: str = ""
    facts: List[LogicFact] = field(default_factory=list)
    rules: List[LogicRule] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    
    def __str__(self) -> str:
        return f"knowledge {self.name} ({len(self.facts)} facts, {len(self.rules)} rules)"


# ============================================================================
# Logic Program (Collection)
# ============================================================================

@dataclass
class LogicProgram:
    """
    A complete logic program consisting of knowledge modules.
    
    This is used at runtime to combine multiple knowledge sources
    for query execution.
    """
    modules: Dict[str, KnowledgeModule] = field(default_factory=dict)
    
    def add_module(self, module: KnowledgeModule) -> None:
        """Add a knowledge module to this program."""
        self.modules[module.name] = module
    
    def get_module(self, name: str) -> Optional[KnowledgeModule]:
        """Retrieve a knowledge module by name."""
        return self.modules.get(name)
    
    def all_facts(self) -> List[LogicFact]:
        """Get all facts from all modules."""
        facts = []
        for module in self.modules.values():
            facts.extend(module.facts)
        return facts
    
    def all_rules(self) -> List[LogicRule]:
        """Get all rules from all modules."""
        rules = []
        for module in self.modules.values():
            rules.extend(module.rules)
        return rules


# ============================================================================
# Type Aliases and Utilities
# ============================================================================

# Union type for any logic term
AnyLogicTerm = Union[LogicVar, LogicAtom, LogicNumber, LogicString, LogicStruct, LogicList]

# Union type for knowledge elements
KnowledgeElement = Union[LogicFact, LogicRule]
