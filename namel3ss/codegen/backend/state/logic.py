"""Logic programming construct encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .expressions import _encode_value, _expression_to_runtime, _expression_to_source

if TYPE_CHECKING:
    from ....ast import LogicFact, LogicKnowledgeModule, LogicQuery, LogicRule, LogicTerm


def _encode_logic_term(term: "LogicTerm", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a logic term for backend state."""
    if isinstance(term, str):
        return {"type": "atom", "value": term}
    if isinstance(term, (int, float, bool)):
        return {"type": "literal", "value": term}
    if isinstance(term, list):
        return {
            "type": "list",
            "elements": [_encode_logic_term(item, env_keys) for item in term],
        }
    if isinstance(term, dict):
        return {
            "type": "struct",
            "functor": term.get("functor"),
            "args": [_encode_logic_term(arg, env_keys) for arg in term.get("args", [])],
        }
    # Fallback for expression-based terms
    return {
        "type": "expression",
        "source": _expression_to_source(term),
        "expr": _expression_to_runtime(term),
    }


def _encode_logic_fact(fact: "LogicFact", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a logic fact for backend state."""
    return {
        "predicate": fact.predicate,
        "args": [_encode_logic_term(arg, env_keys) for arg in fact.args],
        "confidence": fact.confidence if hasattr(fact, "confidence") else 1.0,
        "metadata": _encode_value(getattr(fact, "metadata", {}), env_keys),
    }


def _encode_logic_rule(rule: "LogicRule", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a logic rule for backend state."""
    head = _encode_logic_fact(rule.head, env_keys) if hasattr(rule, "head") else {}
    body: List[Dict[str, Any]] = []
    if hasattr(rule, "body") and rule.body:
        for clause in rule.body:
            if hasattr(clause, "predicate"):
                body.append(_encode_logic_fact(clause, env_keys))
            else:
                body.append(_encode_logic_term(clause, env_keys))
    return {
        "head": head,
        "body": body,
        "confidence": rule.confidence if hasattr(rule, "confidence") else 1.0,
        "metadata": _encode_value(getattr(rule, "metadata", {}), env_keys),
    }


def _encode_knowledge_module(module: "LogicKnowledgeModule", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a logic knowledge module for backend state."""
    facts: List[Dict[str, Any]] = []
    if hasattr(module, "facts") and module.facts:
        facts = [_encode_logic_fact(fact, env_keys) for fact in module.facts]
    rules: List[Dict[str, Any]] = []
    if hasattr(module, "rules") and module.rules:
        rules = [_encode_logic_rule(rule, env_keys) for rule in module.rules]
    return {
        "name": module.name,
        "facts": facts,
        "rules": rules,
        "imports": list(getattr(module, "imports", [])),
        "metadata": _encode_value(getattr(module, "metadata", {}), env_keys),
    }


def _encode_logic_query(query: "LogicQuery", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a logic query for backend state."""
    goals: List[Dict[str, Any]] = []
    if hasattr(query, "goals") and query.goals:
        for goal in query.goals:
            if hasattr(goal, "predicate"):
                goals.append(_encode_logic_fact(goal, env_keys))
            else:
                goals.append(_encode_logic_term(goal, env_keys))
    return {
        "name": query.name if hasattr(query, "name") else None,
        "goals": goals,
        "limit": query.limit if hasattr(query, "limit") else None,
        "metadata": _encode_value(getattr(query, "metadata", {}), env_keys),
    }
