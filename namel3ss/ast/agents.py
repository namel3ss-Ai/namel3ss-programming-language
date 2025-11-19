"""Agent and Graph AST nodes for multi-agent workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .base import Expression


@dataclass
class MemoryConfig:
    """Configuration for agent memory."""
    
    policy: str = "none"  # none, conversation, window, custom
    max_items: Optional[int] = None
    window_size: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDefinition:
    """
    First-class agent definition for multi-agent workflows.
    
    Example DSL syntax:
        agent researcher {
            llm: chat_gpt_4o
            tools: [search_docs, support_rag]
            memory: "conversation"
            goal: "Gather accurate information and propose options."
            system_prompt: "You are a research assistant..."  # optional
            max_turns: 10  # optional
            temperature: 0.7  # optional
        }
    """
    name: str
    llm_name: str
    tool_names: List[str] = field(default_factory=list)
    memory_config: Optional[Union[MemoryConfig, str]] = None
    goal: str = ""
    system_prompt: Optional[str] = None
    max_turns: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[Any] = None  # SourceLocation when available


@dataclass
class GraphEdge:
    """
    Represents a directed edge in a multi-agent graph.
    
    Example DSL syntax:
        { from: researcher, to: decider, when: "done_research" }
    """
    from_agent: str
    to_agent: str
    condition: Union[str, Expression]  # Simple string or expression for routing
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphDefinition:
    """
    First-class graph definition for multi-agent workflows.
    
    Example DSL syntax:
        graph support_flow {
            start: researcher
            edges: [
                { from: researcher, to: decider, when: "done_research" },
                { from: decider, to: researcher, when: "needs_more_info" }
            ]
            termination: decider
            max_hops: 32  # optional
            timeout_ms: 60000  # optional
        }
    """
    name: str
    start_agent: str
    edges: List[GraphEdge] = field(default_factory=list)
    termination_agents: List[str] = field(default_factory=list)
    termination_condition: Optional[Union[str, Expression]] = None
    max_hops: Optional[int] = None
    timeout_ms: Optional[int] = None
    timeout_s: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[Any] = None  # SourceLocation when available


__all__ = [
    "MemoryConfig",
    "AgentDefinition",
    "GraphEdge",
    "GraphDefinition",
]
