"""
Query compiler for translating N3 query blocks into logic engine calls.

Compiles high-level query syntax into executable logic queries with proper
integration of knowledge modules, adapters, and safety limits.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from namel3ss.ast.logic import (
    KnowledgeModule,
    LogicFact,
    LogicQuery,
    LogicRule,
    LogicStruct,
    LogicVar,
)
from namel3ss.codegen.backend.core.runtime.logic_adapters import AdapterRegistry
from namel3ss.codegen.backend.core.runtime.logic_engine import (
    LogicEngine,
    LogicEngineConfig,
    Substitution,
)


# ============================================================================
# Query Compilation Errors
# ============================================================================

class QueryCompilationError(Exception):
    """Base exception for query compilation errors."""
    pass


class UndefinedKnowledgeModule(QueryCompilationError):
    """Raised when a query references an undefined knowledge module."""
    pass


class InvalidQueryGoal(QueryCompilationError):
    """Raised when a query goal is malformed."""
    pass


# ============================================================================
# Query Context
# ============================================================================

class QueryContext:
    """
    Context for query execution containing knowledge bases and adapters.
    
    Provides unified access to facts and rules from:
    - Explicitly defined knowledge modules
    - Dataset adapters
    - Model adapters
    """
    
    def __init__(
        self,
        knowledge_modules: Optional[Dict[str, KnowledgeModule]] = None,
        adapter_registry: Optional[AdapterRegistry] = None,
    ):
        """
        Initialize query context.
        
        Args:
            knowledge_modules: Map of module name -> KnowledgeModule
            adapter_registry: Registry of data adapters
        """
        self.knowledge_modules = knowledge_modules or {}
        self.adapter_registry = adapter_registry or AdapterRegistry()
    
    def get_knowledge_module(self, name: str) -> Optional[KnowledgeModule]:
        """Get a knowledge module by name."""
        return self.knowledge_modules.get(name)
    
    def collect_facts(self, module_names: List[str]) -> List[LogicFact]:
        """
        Collect all facts from the specified knowledge modules.
        
        Also includes facts from adapters.
        """
        facts = []
        
        # Collect from knowledge modules
        for name in module_names:
            module = self.get_knowledge_module(name)
            if module is None:
                raise UndefinedKnowledgeModule(
                    f"Knowledge module not found: {name}"
                )
            facts.extend(module.facts)
        
        # Add adapter facts
        facts.extend(self.adapter_registry.get_all_facts())
        
        return facts
    
    def collect_rules(self, module_names: List[str]) -> List[LogicRule]:
        """Collect all rules from the specified knowledge modules."""
        rules = []
        
        for name in module_names:
            module = self.get_knowledge_module(name)
            if module is None:
                raise UndefinedKnowledgeModule(
                    f"Knowledge module not found: {name}"
                )
            rules.extend(module.rules)
        
        return rules


# ============================================================================
# Query Compiler
# ============================================================================

class QueryCompiler:
    """
    Compiles and executes N3 query blocks.
    
    Translates high-level query syntax into logic engine calls with proper
    integration of knowledge sources and safety limits.
    """
    
    def __init__(
        self,
        context: QueryContext,
        engine_config: Optional[LogicEngineConfig] = None,
    ):
        """
        Initialize query compiler.
        
        Args:
            context: Query execution context
            engine_config: Logic engine configuration (for safety limits)
        """
        self.context = context
        self.engine_config = engine_config or LogicEngineConfig()
    
    def compile_query(self, query: LogicQuery) -> CompiledQuery:
        """
        Compile a query into an executable form.
        
        Args:
            query: The query to compile
            
        Returns:
            CompiledQuery object ready for execution
        """
        # Validate query
        if not query.goals:
            raise InvalidQueryGoal("Query must have at least one goal")
        
        # Collect facts and rules from knowledge sources
        facts = self.context.collect_facts(query.knowledge_sources)
        rules = self.context.collect_rules(query.knowledge_sources)
        
        # Compile goals
        goals = self._compile_goals(query.goals)
        
        return CompiledQuery(
            name=query.name,
            goals=goals,
            facts=facts,
            rules=rules,
            limit=query.limit,
            variables=query.variables,
            engine_config=self.engine_config,
        )
    
    def _compile_goals(self, goals: List[LogicStruct]) -> List[LogicStruct]:
        """
        Compile query goals.
        
        Currently just validates and returns goals as-is.
        Future: Could optimize, reorder, or transform goals here.
        """
        for goal in goals:
            if not isinstance(goal, LogicStruct):
                raise InvalidQueryGoal(
                    f"Goal must be a structure, got: {type(goal).__name__}"
                )
        
        return goals


# ============================================================================
# Compiled Query
# ============================================================================

class CompiledQuery:
    """
    A compiled query ready for execution.
    
    Contains all the facts, rules, and configuration needed to execute
    the query through the logic engine.
    """
    
    def __init__(
        self,
        name: str,
        goals: List[LogicStruct],
        facts: List[LogicFact],
        rules: List[LogicRule],
        limit: Optional[int] = None,
        variables: Optional[List[str]] = None,
        engine_config: Optional[LogicEngineConfig] = None,
    ):
        """Initialize compiled query."""
        self.name = name
        self.goals = goals
        self.facts = facts
        self.rules = rules
        self.limit = limit
        self.variables = variables
        self.engine_config = engine_config or LogicEngineConfig()
    
    def execute(self) -> Iterator[Dict[str, Any]]:
        """
        Execute the query and yield result bindings.
        
        Yields dictionaries mapping variable names to their bound values.
        If variables list is specified, only those variables are included.
        """
        engine = LogicEngine(config=self.engine_config)
        
        solution_count = 0
        for subst in engine.solve(self.goals, self.facts, self.rules):
            # Apply limit if specified
            if self.limit is not None and solution_count >= self.limit:
                break
            
            # Extract variable bindings
            result = self._extract_bindings(subst)
            yield result
            
            solution_count += 1
    
    def execute_all(self) -> List[Dict[str, Any]]:
        """Execute query and return all results as a list."""
        return list(self.execute())
    
    def _extract_bindings(self, subst: Substitution) -> Dict[str, Any]:
        """
        Extract variable bindings from a substitution.
        
        Returns dictionary mapping variable names to their bound values.
        If self.variables is specified, only extract those variables.
        """
        result = {}
        
        # Determine which variables to extract
        if self.variables:
            var_names = self.variables
        else:
            # Extract all variables from goals
            var_names = self._collect_variables(self.goals)
        
        # Extract bindings
        for var_name in var_names:
            # Apply substitution to resolve variable
            term = subst.apply(LogicVar(name=var_name))
            # Convert to Python value
            result[var_name] = self._term_to_python(term)
        
        return result
    
    def _collect_variables(self, goals: List[LogicStruct]) -> List[str]:
        """Collect all variable names from goals."""
        variables = set()
        
        for goal in goals:
            self._collect_vars_from_struct(goal, variables)
        
        return list(variables)
    
    def _collect_vars_from_struct(
        self,
        struct: LogicStruct,
        variables: set,
    ) -> None:
        """Recursively collect variables from a structure."""
        from namel3ss.ast.logic import LogicList, LogicVar
        
        for arg in struct.args:
            if isinstance(arg, LogicVar):
                variables.add(arg.name)
            elif isinstance(arg, LogicStruct):
                self._collect_vars_from_struct(arg, variables)
            elif isinstance(arg, LogicList):
                for elem in arg.elements:
                    if isinstance(elem, LogicVar):
                        variables.add(elem.name)
                    elif isinstance(elem, LogicStruct):
                        self._collect_vars_from_struct(elem, variables)
    
    def _term_to_python(self, term: Any) -> Any:
        """Convert a logic term to a Python value."""
        from namel3ss.ast.logic import (
            LogicAtom,
            LogicList,
            LogicNumber,
            LogicString,
            LogicStruct,
            LogicVar,
        )
        
        if isinstance(term, LogicVar):
            # Unbound variable
            return None
        elif isinstance(term, LogicAtom):
            # Handle special atoms
            if term.value == "true":
                return True
            elif term.value == "false":
                return False
            elif term.value == "null":
                return None
            else:
                return term.value
        elif isinstance(term, LogicNumber):
            return term.value
        elif isinstance(term, LogicString):
            return term.value
        elif isinstance(term, LogicStruct):
            # Convert structure to dictionary
            return {
                "functor": term.functor,
                "args": [self._term_to_python(arg) for arg in term.args],
            }
        elif isinstance(term, LogicList):
            return [self._term_to_python(elem) for elem in term.elements]
        else:
            return str(term)


__all__ = [
    "QueryContext",
    "QueryCompiler",
    "CompiledQuery",
    "QueryCompilationError",
    "UndefinedKnowledgeModule",
    "InvalidQueryGoal",
]
