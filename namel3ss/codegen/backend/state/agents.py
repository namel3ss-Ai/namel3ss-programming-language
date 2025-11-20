"""Agent and multi-agent graph encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, Set, TYPE_CHECKING

from .expressions import _encode_value

if TYPE_CHECKING:
    from ....ast import Agent, MultiAgentGraph


def _encode_agent(agent: "Agent", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode an agent definition for backend state."""
    return {
        "name": agent.name,
        "llm": agent.llm,
        "tools": list(agent.tools),
        "system_prompt": agent.system_prompt,
        "memory": agent.memory,
        "guardrails": list(agent.guardrails or []),
        "max_iterations": agent.max_iterations,
        "description": agent.description,
        "metadata": _encode_value(agent.metadata, env_keys),
    }


def _encode_graph(graph: "MultiAgentGraph", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a multi-agent graph definition for backend state."""
    node_edges = {
        source: list(edges) for source, edges in (graph.edges or {}).items()
    }
    return {
        "name": graph.name,
        "nodes": list(graph.nodes),
        "edges": node_edges,
        "entry_point": graph.entry_point,
        "conditional_edges": dict(graph.conditional_edges or {}),
        "description": graph.description,
        "metadata": _encode_value(graph.metadata, env_keys),
    }
