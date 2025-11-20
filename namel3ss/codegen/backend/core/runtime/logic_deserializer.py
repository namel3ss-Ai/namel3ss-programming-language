"""
Deserialize logic AST structures from runtime dictionaries.

This module reconstructs LogicQuery, KnowledgeModule, and related AST nodes
from the serialized dictionary format stored in the runtime.
"""

from typing import Any, Dict, List, Optional

from namel3ss.ast.logic import (
    LogicAtom,
    LogicFact,
    LogicList,
    LogicNumber,
    LogicQuery,
    LogicRule,
    LogicString,
    LogicStruct,
    LogicTerm,
    LogicVar,
    KnowledgeModule,
)


def deserialize_logic_term(term_dict: Dict[str, Any]) -> LogicTerm:
    """Reconstruct a LogicTerm from its serialized dict."""
    term_type = term_dict.get("type")
    
    if term_type == "var":
        return LogicVar(name=term_dict["name"])
    
    elif term_type == "atom":
        return LogicAtom(value=term_dict["value"])
    
    elif term_type == "number":
        return LogicNumber(value=term_dict["value"])
    
    elif term_type == "string":
        return LogicString(value=term_dict["value"])
    
    elif term_type == "struct":
        args = [deserialize_logic_term(arg) for arg in term_dict["args"]]
        return LogicStruct(functor=term_dict["functor"], args=args)
    
    elif term_type == "list":
        elements = [deserialize_logic_term(el) for el in term_dict["elements"]]
        tail = (
            deserialize_logic_term(term_dict["tail"])
            if term_dict.get("tail")
            else None
        )
        if tail and not isinstance(tail, LogicVar):
            tail = None  # tail must be a variable
        return LogicList(elements=elements, tail=tail)
    
    else:
        # Unknown term type, return as atom
        return LogicAtom(value=str(term_dict.get("value", "unknown")))


def deserialize_logic_fact(fact_dict: Dict[str, Any]) -> LogicFact:
    """Reconstruct a LogicFact from its serialized dict."""
    head = deserialize_logic_term(fact_dict["head"])
    if not isinstance(head, LogicStruct):
        # Facts must have struct heads
        head = LogicStruct(functor="unknown", args=[])
    
    return LogicFact(
        head=head,
        metadata=fact_dict.get("metadata", {}),
    )


def deserialize_logic_rule(rule_dict: Dict[str, Any]) -> LogicRule:
    """Reconstruct a LogicRule from its serialized dict."""
    head = deserialize_logic_term(rule_dict["head"])
    if not isinstance(head, LogicStruct):
        head = LogicStruct(functor="unknown", args=[])
    
    body = []
    for goal_dict in rule_dict.get("body", []):
        goal = deserialize_logic_term(goal_dict)
        if isinstance(goal, LogicStruct):
            body.append(goal)
    
    return LogicRule(
        head=head,
        body=body,
        metadata=rule_dict.get("metadata", {}),
    )


def deserialize_knowledge_module(module_dict: Dict[str, Any]) -> KnowledgeModule:
    """Reconstruct a KnowledgeModule from its serialized dict."""
    facts = [deserialize_logic_fact(f) for f in module_dict.get("facts", [])]
    rules = [deserialize_logic_rule(r) for r in module_dict.get("rules", [])]
    
    return KnowledgeModule(
        name=module_dict["name"],
        facts=facts,
        rules=rules,
        imports=list(module_dict.get("imports", [])),
        metadata=module_dict.get("metadata", {}),
        description=module_dict.get("description"),
    )


def deserialize_logic_query(query_dict: Dict[str, Any]) -> LogicQuery:
    """Reconstruct a LogicQuery from its serialized dict."""
    goals = []
    for goal_dict in query_dict.get("goals", []):
        goal = deserialize_logic_term(goal_dict)
        if isinstance(goal, LogicStruct):
            goals.append(goal)
    
    return LogicQuery(
        name=query_dict["name"],
        knowledge_sources=list(query_dict.get("knowledge_sources", [])),
        goals=goals,
        limit=query_dict.get("limit"),
        variables=list(query_dict["variables"]) if query_dict.get("variables") else None,
        metadata=query_dict.get("metadata", {}),
    )


def load_queries_from_runtime(queries_dict: Dict[str, Dict[str, Any]]) -> List[LogicQuery]:
    """Load all queries from the APP_QUERIES runtime dict."""
    return [deserialize_logic_query(q) for q in queries_dict.values()]


def load_knowledge_modules_from_runtime(
    modules_dict: Dict[str, Dict[str, Any]]
) -> List[KnowledgeModule]:
    """Load all knowledge modules from the KNOWLEDGE_MODULES runtime dict."""
    return [deserialize_knowledge_module(m) for m in modules_dict.values()]


__all__ = [
    "deserialize_logic_term",
    "deserialize_logic_fact",
    "deserialize_logic_rule",
    "deserialize_knowledge_module",
    "deserialize_logic_query",
    "load_queries_from_runtime",
    "load_knowledge_modules_from_runtime",
]
