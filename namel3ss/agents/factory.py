"""Factory for building graph executors from backend state."""

from typing import Any, Callable, Dict, Optional

from namel3ss.agents import GraphExecutor
from namel3ss.ast.agents import AgentDefinition, GraphDefinition
from namel3ss.llm.base import BaseLLM


def build_graph_executor(
    graph_def: GraphDefinition,
    agent_registry: Dict[str, AgentDefinition],
    llm_registry: Dict[str, BaseLLM],
    tool_registry: Optional[Dict[str, Callable]] = None,
) -> GraphExecutor:
    """
    Build a GraphExecutor from definitions.
    
    This factory is used by the generated backend runtime to create
    graph executors on demand.
    
    Args:
        graph_def: GraphDefinition from AST
        agent_registry: Dict mapping agent names to AgentDefinition
        llm_registry: Dict mapping LLM names to BaseLLM instances
        tool_registry: Dict mapping tool names to callable functions
    
    Returns:
        Configured GraphExecutor instance
    """
    return GraphExecutor(
        graph_def,
        agent_registry,
        llm_registry,
        tool_registry,
    )


def run_graph_from_state(
    graph_name: str,
    graphs: Dict[str, Dict[str, Any]],
    agents: Dict[str, Dict[str, Any]],
    llms: Dict[str, Any],
    tools: Dict[str, Any],
    user_input: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a graph from encoded backend state.
    
    This is the runtime entry point called from generated backend code.
    It reconstructs the graph and agent definitions from dictionaries,
    creates the executor, and runs it.
    
    Args:
        graph_name: Name of graph to execute
        graphs: Encoded graph definitions
        agents: Encoded agent definitions
        llms: LLM instances or definitions
        tools: Tool callable functions
        user_input: User's input message
        context: Additional context variables
    
    Returns:
        Dict with status, response, hops, and metadata
    """
    if graph_name not in graphs:
        return {
            "status": "error",
            "error": f"Graph '{graph_name}' not found",
            "final_response": "",
            "hops": [],
            "metadata": {},
        }
    
    # Import here to avoid circular dependencies in generated code
    from namel3ss.ast.agents import GraphDefinition, GraphEdge, AgentDefinition, MemoryConfig
    from namel3ss.llm.base import BaseLLM
    
    # Reconstruct graph definition from dict
    graph_data = graphs[graph_name]
    
    edges = []
    for edge_data in graph_data.get("edges", []):
        edges.append(GraphEdge(
            from_agent=edge_data["from_agent"],
            to_agent=edge_data["to_agent"],
            condition=edge_data.get("condition"),
        ))
    
    graph_def = GraphDefinition(
        name=graph_name,
        start_agent=graph_data["start_agent"],
        edges=edges,
        termination_agents=graph_data.get("termination_agents", []),
        termination_condition=graph_data.get("termination_condition"),
        max_hops=graph_data.get("max_hops", 32),
        timeout_ms=graph_data.get("timeout_ms"),
    )
    
    # Reconstruct agent definitions
    agent_registry = {}
    for agent_name, agent_data in agents.items():
        memory_config = None
        if agent_data.get("memory_config"):
            mem = agent_data["memory_config"]
            memory_config = MemoryConfig(
                policy=mem.get("policy", "conversation_window"),
                max_items=mem.get("max_items"),
                window_size=mem.get("window_size"),
                config=mem.get("config", {}),
            )
        
        agent_registry[agent_name] = AgentDefinition(
            name=agent_name,
            llm_name=agent_data["llm_name"],
            tool_names=agent_data.get("tool_names", []),
            memory_config=memory_config,
            goal=agent_data.get("goal", ""),
            system_prompt=agent_data.get("system_prompt"),
            max_turns=agent_data.get("max_turns"),
            temperature=agent_data.get("temperature"),
            config=agent_data.get("config", {}),
        )
    
    # LLM registry should contain BaseLLM instances
    llm_registry = llms if isinstance(llms, dict) else {}
    
    # Tool registry should contain callable functions
    tool_registry = tools if isinstance(tools, dict) else {}
    
    # Build and execute graph
    executor = build_graph_executor(
        graph_def,
        agent_registry,
        llm_registry,
        tool_registry,
    )
    
    result = executor.execute(
        user_input,
        context=context,
        max_hops=graph_data.get("max_hops"),
        timeout_ms=graph_data.get("timeout_ms"),
    )
    
    # Convert result to dict
    return {
        "status": result.status,
        "final_response": result.final_response,
        "hops": [
            {
                "agent_name": hop.agent_name,
                "response": hop.agent_result.final_response,
                "next_agent": hop.next_agent,
                "routing_decision": hop.routing_decision,
                "turns": len(hop.agent_result.turns),
                "metadata": hop.metadata,
            }
            for hop in result.hops
        ],
        "start_agent": result.start_agent,
        "end_agent": result.end_agent,
        "metadata": result.metadata,
        "error": result.error,
    }
