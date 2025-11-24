"""Agent and multi-agent graph encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, Set, TYPE_CHECKING

from .expressions import _encode_value

if TYPE_CHECKING:
    from ....ast import AgentDefinition, GraphDefinition


def _encode_agent(agent: "AgentDefinition", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode an agent definition for backend state."""
    metadata_value = _encode_value(getattr(agent, "metadata", {}), env_keys)
    if not isinstance(metadata_value, dict):
        metadata_value = {"value": metadata_value} if metadata_value is not None else {}
    
    # Extract security metadata
    capabilities = list(getattr(agent, "capabilities", []) or [])
    permission_level = getattr(agent, "permission_level", None)
    security_config = getattr(agent, "security_config", None)
    
    return {
        "name": agent.name,
        "llm": getattr(agent, "llm_name", None) or getattr(agent, "llm", None),
        "tools": list(getattr(agent, "tool_names", []) or []),
        "system_prompt": getattr(agent, "system_prompt", None),
        "memory": getattr(agent, "memory_config", None),
        "goal": getattr(agent, "goal", None),
        "max_iterations": getattr(agent, "max_turns", None),
        "max_tokens": getattr(agent, "max_tokens", None),
        "temperature": getattr(agent, "temperature", None),
        "top_p": getattr(agent, "top_p", None),
        # Security metadata
        "capabilities": capabilities,
        "permission_level": permission_level,
        "security_config": security_config,
        "config": _encode_value(getattr(agent, "config", {}), env_keys),
        "metadata": metadata_value,
    }


def _encode_graph(graph: "GraphDefinition", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a multi-agent graph definition for backend state."""
    node_edges = {
        source: list(edges) for source, edges in (getattr(graph, "edges", None) or {}).items()
    }
    return {
        "name": graph.name,
        "nodes": list(getattr(graph, "nodes", []) or getattr(graph, "agents", []) or []),
        "edges": node_edges,
        "entry_point": getattr(graph, "start_agent", None),
        "conditional_edges": dict(getattr(graph, "conditional_edges", {}) or {}),
        "description": getattr(graph, "description", None),
        "metadata": _encode_value(getattr(graph, "metadata", {}), env_keys),
    }
