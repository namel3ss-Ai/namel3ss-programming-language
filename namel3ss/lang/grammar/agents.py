"""Agent and graph parsing."""

from __future__ import annotations
import re
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Tuple

if TYPE_CHECKING:
    from .helpers import _Line

from namel3ss.ast.agents import AgentDefinition, GraphDefinition, GraphEdge, MemoryConfig


class AgentsParserMixin:
    """Mixin providing agent and graph parsing."""

    def _parse_agent(self, line: _Line) -> None:
        """
        Parse an agent definition block.
        
        Grammar:
            agent <name> {
                llm: <llm_name>
                tools: [<tool1>, <tool2>, ...]
                memory: "<policy>" or {config}
                goal: "<description>"
                system_prompt: "<prompt>"  # optional
                max_turns: <int>  # optional
                temperature: <float>  # optional
            }
        """
        match = _AGENT_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "agent declaration")
        name = match.group(1)
        base_indent = self._indent(line.text)
        self._advance()
        
        # Parse key-value properties within braces
        properties = self._parse_kv_block_braces(base_indent)
        
        # Extract required fields
        llm_name = properties.get('llm')
        if not llm_name:
            raise self._error("agent block requires 'llm' field", line)
        
        # Parse tools list
        tools_raw = properties.get('tools', '[]')
        tool_names = self._parse_list_field(tools_raw)
        
        # Parse memory config
        memory_raw = properties.get('memory')
        memory_config = None
        if memory_raw:
            if isinstance(memory_raw, str):
                # Simple string policy like "conversation" or "none"
                memory_config = memory_raw
            elif isinstance(memory_raw, dict):
                memory_config = MemoryConfig(**memory_raw)
        
        goal = properties.get('goal', '')
        system_prompt = properties.get('system_prompt')
        max_turns = int(properties['max_turns']) if 'max_turns' in properties else None
        max_tokens = int(properties['max_tokens']) if 'max_tokens' in properties else None
        temperature = float(properties['temperature']) if 'temperature' in properties else None
        top_p = float(properties['top_p']) if 'top_p' in properties else None
        
        # Build config from remaining properties
        config = {k: v for k, v in properties.items() 
                  if k not in {'llm', 'tools', 'memory', 'goal', 'system_prompt', 
                               'max_turns', 'max_tokens', 'temperature', 'top_p'}}
        
        agent = AgentDefinition(
            name=name,
            llm_name=llm_name,
            tool_names=tool_names,
            memory_config=memory_config,
            goal=goal,
            system_prompt=system_prompt,
            max_turns=max_turns,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            config=config,
        )
        
        self._ensure_app(line)
        self._app.agents.append(agent)

    def _parse_graph(self, line: _Line) -> None:
        """
        Parse a graph definition block.
        
        Grammar:
            graph <name> {
                start: <agent_name>
                edges: [
                    { from: <agent1>, to: <agent2>, when: "<condition>" },
                    ...
                ]
                termination: <agent_name> or [<agent1>, <agent2>]
                max_hops: <int>  # optional
                timeout_ms: <int>  # optional
            }
        """
        match = _GRAPH_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "graph declaration")
        name = match.group(1)
        base_indent = self._indent(line.text)
        self._advance()
        
        # Parse key-value properties within braces
        properties = self._parse_kv_block_braces(base_indent)
        
        # Extract required fields
        start_agent = properties.get('start')
        if not start_agent:
            raise self._error("graph block requires 'start' field", line)
        
        # Parse edges list
        edges_raw = properties.get('edges', '[]')
        edges = self._parse_graph_edges(edges_raw)
        
        # Parse termination (can be single agent or list)
        termination_raw = properties.get('termination')
        termination_agents = []
        termination_condition = None
        if termination_raw:
            if isinstance(termination_raw, str):
                # Check if it's a condition expression or agent name
                if termination_raw.startswith('"') or '==' in termination_raw or 'and' in termination_raw:
                    termination_condition = termination_raw.strip('"')
                else:
                    termination_agents = [termination_raw]
            elif isinstance(termination_raw, list):
                termination_agents = termination_raw
        
        max_hops = int(properties['max_hops']) if 'max_hops' in properties else None
        timeout_ms = int(properties['timeout_ms']) if 'timeout_ms' in properties else None
        timeout_s = float(properties['timeout_s']) if 'timeout_s' in properties else None
        
        # Build config from remaining properties
        config = {k: v for k, v in properties.items() 
                  if k not in {'start', 'edges', 'termination', 'max_hops', 'timeout_ms', 'timeout_s'}}
        
        graph = GraphDefinition(
            name=name,
            start_agent=start_agent,
            edges=edges,
            termination_agents=termination_agents,
            termination_condition=termination_condition,
            max_hops=max_hops,
            timeout_ms=timeout_ms,
            timeout_s=timeout_s,
            config=config,
        )
        
        self._ensure_app(line)
        self._app.graphs.append(graph)

    def _parse_policy(self, line: _Line) -> None:
        """Policy parser - moved to policy.py mixin."""
        pass

__all__ = ['AgentsParserMixin']
