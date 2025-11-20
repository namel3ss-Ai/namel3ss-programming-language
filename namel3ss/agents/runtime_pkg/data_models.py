from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from namel3ss.llm.base import LLMResponse


@dataclass
class AgentMessage:
    """A message in agent conversation history."""
    
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTurn:
    """A complete turn of agent execution (prompt + response + tool calls)."""
    
    messages: List[AgentMessage]
    llm_response: Optional[LLMResponse] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from agent execution."""
    
    status: str  # "success", "error", "max_turns", "goal_achieved"
    final_response: str
    turns: List[AgentTurn]
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
