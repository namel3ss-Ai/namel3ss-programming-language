"""Graph executor for multi-agent orchestration with routing and state management."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import time
import logging

from namel3ss.ast.agents import GraphDefinition, GraphEdge, AgentDefinition
from namel3ss.agents.runtime import AgentRuntime, AgentResult
from namel3ss.llm.base import BaseLLM
from namel3ss.observability.metrics import record_metric

logger = logging.getLogger(__name__)


@dataclass
class GraphHop:
    """A single hop in graph execution (one agent execution)."""
    
    agent_name: str
    agent_result: AgentResult
    next_agent: Optional[str] = None
    routing_decision: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphResult:
    """Result from graph execution."""
    
    status: str  # "success", "error", "max_hops", "timeout", "no_path"
    final_response: str
    hops: List[GraphHop]
    start_agent: str
    end_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class GraphExecutor:
    """
    Multi-agent orchestration engine.
    
    Executes graph workflows with:
    - Agent-to-agent routing based on edges and conditions
    - Max hops limit to prevent infinite loops
    - Timeout enforcement
    - State passing between agents
    - Termination condition evaluation
    """
    
    def __init__(
        self,
        graph_def: GraphDefinition,
        agent_registry: Dict[str, AgentDefinition],
        llm_registry: Dict[str, BaseLLM],
        tool_registry: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize graph executor.
        
        Args:
            graph_def: GraphDefinition from AST
            agent_registry: Dict mapping agent names to AgentDefinition
            llm_registry: Dict mapping LLM names to BaseLLM instances
            tool_registry: Dict mapping tool names to callable functions
        """
        self.graph_def = graph_def
        self.agent_registry = agent_registry
        self.llm_registry = llm_registry
        self.tool_registry = tool_registry or {}
        
        # Build agent runtimes
        self.agent_runtimes: Dict[str, AgentRuntime] = {}
        self._build_agent_runtimes()
        
        # Build routing graph
        self.routing_graph: Dict[str, List[GraphEdge]] = {}
        self._build_routing_graph()
    
    def _build_agent_runtimes(self) -> None:
        """Build AgentRuntime instances for all agents in the graph."""
        # Collect all agents referenced in graph
        agent_names = {self.graph_def.start_agent}
        agent_names.update(self.graph_def.termination_agents)
        for edge in self.graph_def.edges:
            agent_names.add(edge.from_agent)
            agent_names.add(edge.to_agent)
        
        # Create runtime for each agent
        for agent_name in agent_names:
            if agent_name not in self.agent_registry:
                logger.warning(f"Agent {agent_name} not found in registry")
                continue
            
            agent_def = self.agent_registry[agent_name]
            
            # Get LLM instance
            llm_name = agent_def.llm_name
            if llm_name not in self.llm_registry:
                logger.warning(f"LLM {llm_name} not found in registry for agent {agent_name}")
                continue
            
            llm_instance = self.llm_registry[llm_name]
            
            # Create runtime
            self.agent_runtimes[agent_name] = AgentRuntime(
                agent_def,
                llm_instance,
                self.tool_registry,
            )
    
    def _build_routing_graph(self) -> None:
        """Build adjacency list for routing."""
        for edge in self.graph_def.edges:
            if edge.from_agent not in self.routing_graph:
                self.routing_graph[edge.from_agent] = []
            self.routing_graph[edge.from_agent].append(edge)
    
    def execute(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        max_hops: Optional[int] = None,
        timeout_ms: Optional[int] = None,
    ) -> GraphResult:
        """
        Execute graph workflow starting from start_agent.
        
        Args:
            user_input: User's initial input
            context: Additional context variables
            max_hops: Maximum hops (overrides graph default)
            timeout_ms: Timeout in milliseconds (overrides graph default)
        
        Returns:
            GraphResult with execution trace and final response
        """
        context = context or {}
        max_hops = max_hops or self.graph_def.max_hops or 32
        timeout_ms = timeout_ms or self.graph_def.timeout_ms
        
        start_time = time.time()
        hops: List[GraphHop] = []
        
        # Record graph execution start
        record_metric("graph.execution.start", 1, tags={"graph": self.graph_def.name})
        logger.info(f"Graph {self.graph_def.name} starting execution from agent {self.graph_def.start_agent}")
        
        try:
            current_agent = self.graph_def.start_agent
            current_input = user_input
            
            for hop_num in range(max_hops):
                # Check timeout
                if timeout_ms:
                    elapsed_ms = (time.time() - start_time) * 1000
                    if elapsed_ms > timeout_ms:
                        record_metric("graph.execution.timeout", 1, tags={
                            "graph": self.graph_def.name,
                            "hops": str(len(hops))
                        })
                        logger.warning(f"Graph {self.graph_def.name} timed out after {len(hops)} hops")
                        return GraphResult(
                            status="timeout",
                            final_response="",
                            hops=hops,
                            start_agent=self.graph_def.start_agent,
                            end_agent=current_agent,
                            metadata={
                                "total_hops": len(hops),
                                "elapsed_ms": elapsed_ms,
                                "timeout_ms": timeout_ms,
                            },
                            error=f"Graph execution timeout after {elapsed_ms:.0f}ms",
                        )
                
                logger.debug(f"Graph {self.graph_def.name} hop {hop_num + 1}: agent={current_agent}")
                
                # Record hop start
                record_metric("graph.hop.start", 1, tags={
                    "graph": self.graph_def.name,
                    "agent": current_agent,
                    "hop": str(hop_num + 1)
                })
                
                # Execute current agent
                if current_agent not in self.agent_runtimes:
                    record_metric("graph.hop.error", 1, tags={
                        "graph": self.graph_def.name,
                        "agent": current_agent,
                        "error": "agent_not_found"
                    })
                    return GraphResult(
                        status="error",
                        final_response="",
                        hops=hops,
                        start_agent=self.graph_def.start_agent,
                        end_agent=current_agent,
                        metadata={"total_hops": len(hops)},
                        error=f"Agent {current_agent} not available in runtime",
                    )
                
                agent_runtime = self.agent_runtimes[current_agent]
                agent_result = agent_runtime.act(current_input, context=context)
                
                # Record hop completion
                record_metric("graph.hop.complete", 1, tags={
                    "graph": self.graph_def.name,
                    "agent": current_agent,
                    "hop": str(hop_num + 1),
                    "status": agent_result.status
                })
                
                # Check termination
                is_termination_agent = current_agent in self.graph_def.termination_agents
                termination_condition_met = self._check_termination_condition(
                    agent_result,
                    context,
                )
                
                # Route to next agent
                next_agent = None
                routing_decision = None
                
                if not is_termination_agent and not termination_condition_met:
                    next_agent, routing_decision = self._route_to_next_agent(
                        current_agent,
                        agent_result,
                        context,
                    )
                    
                    if next_agent:
                        record_metric("graph.routing.decision", 1, tags={
                            "graph": self.graph_def.name,
                            "from_agent": current_agent,
                            "to_agent": next_agent
                        })
                        logger.debug(f"Graph {self.graph_def.name} routing: {current_agent} -> {next_agent}")
                    else:
                        record_metric("graph.routing.terminal", 1, tags={
                            "graph": self.graph_def.name,
                            "agent": current_agent
                        })
                        logger.debug(f"Graph {self.graph_def.name} reached terminal agent: {current_agent}")
                
                # Record hop
                hop = GraphHop(
                    agent_name=current_agent,
                    agent_result=agent_result,
                    next_agent=next_agent,
                    routing_decision=routing_decision,
                    metadata={
                        "hop_number": hop_num + 1,
                        "is_termination": is_termination_agent or termination_condition_met,
                    },
                )
                hops.append(hop)
                
                # Check if we should terminate
                if is_termination_agent or termination_condition_met or next_agent is None:
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    # Record successful completion
                    record_metric("graph.execution.complete", elapsed_ms, tags={
                        "graph": self.graph_def.name,
                        "hops": str(len(hops)),
                        "status": "success"
                    })
                    record_metric("graph.execution.hops", len(hops), tags={"graph": self.graph_def.name})
                    logger.info(f"Graph {self.graph_def.name} completed successfully in {len(hops)} hops ({elapsed_ms:.1f}ms)")
                    
                    return GraphResult(
                        status="success",
                        final_response=agent_result.final_response,
                        hops=hops,
                        start_agent=self.graph_def.start_agent,
                        end_agent=current_agent,
                        metadata={
                            "total_hops": len(hops),
                            "elapsed_ms": elapsed_ms,
                            "termination_reason": "agent" if is_termination_agent else "condition" if termination_condition_met else "no_path",
                        },
                    )
                
                # Move to next agent
                current_agent = next_agent
                current_input = agent_result.final_response  # Pass output as input to next agent
            
            # Max hops reached
            elapsed_ms = (time.time() - start_time) * 1000
            final_response = hops[-1].agent_result.final_response if hops else ""
            
            record_metric("graph.execution.max_hops", 1, tags={
                "graph": self.graph_def.name,
                "hops": str(max_hops)
            })
            logger.warning(f"Graph {self.graph_def.name} reached max hops ({max_hops})")
            
            return GraphResult(
                status="max_hops",
                final_response=final_response,
                hops=hops,
                start_agent=self.graph_def.start_agent,
                end_agent=current_agent,
                metadata={
                    "total_hops": len(hops),
                    "max_hops": max_hops,
                    "elapsed_ms": elapsed_ms,
                },
            )
        
        except Exception as e:
            logger.error(f"Graph {self.graph_def.name} error: {e}", exc_info=True)
            record_metric("graph.execution.error", 1, tags={
                "graph": self.graph_def.name,
                "error_type": type(e).__name__
            })
            elapsed_ms = (time.time() - start_time) * 1000
            final_response = hops[-1].agent_result.final_response if hops else ""
            return GraphResult(
                status="error",
                final_response=final_response,
                hops=hops,
                start_agent=self.graph_def.start_agent,
                end_agent=current_agent if 'current_agent' in locals() else None,
                metadata={
                    "total_hops": len(hops),
                    "elapsed_ms": elapsed_ms,
                },
                error=str(e),
            )
    
    def _route_to_next_agent(
        self,
        current_agent: str,
        agent_result: AgentResult,
        context: Dict[str, Any],
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Determine next agent based on routing rules.
        
        Returns:
            Tuple of (next_agent_name, routing_decision)
        """
        # Get outgoing edges for current agent
        edges = self.routing_graph.get(current_agent, [])
        
        if not edges:
            return None, "no_edges"
        
        # Evaluate edge conditions
        for edge in edges:
            if edge.condition is None:
                # Unconditional edge
                return edge.to_agent, "unconditional"
            
            # Evaluate condition
            if self._evaluate_condition(edge.condition, agent_result, context):
                return edge.to_agent, f"condition:{edge.condition}"
        
        # No matching condition - take first edge as fallback
        if edges:
            return edges[0].to_agent, "fallback"
        
        return None, "no_match"
    
    def _evaluate_condition(
        self,
        condition: str,
        agent_result: AgentResult,
        context: Dict[str, Any],
    ) -> bool:
        """
        Evaluate routing condition.
        
        Supports simple conditions like:
        - "status == 'success'"
        - "contains('keyword')"
        - Boolean expressions
        """
        try:
            # Create evaluation context
            eval_context = {
                "status": agent_result.status,
                "response": agent_result.final_response,
                "turns": len(agent_result.turns),
                "context": context,
                "result": agent_result,
            }
            
            # Helper functions
            def contains(text: str) -> bool:
                return text.lower() in agent_result.final_response.lower()
            
            eval_context["contains"] = contains
            
            # Evaluate condition
            # Note: Using eval is simple but not safe for untrusted input
            # In production, use a safe expression evaluator
            result = eval(condition, {"__builtins__": {}}, eval_context)
            return bool(result)
        
        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False
    
    def _check_termination_condition(
        self,
        agent_result: AgentResult,
        context: Dict[str, Any],
    ) -> bool:
        """Check if global termination condition is met."""
        if not self.graph_def.termination_condition:
            return False
        
        return self._evaluate_condition(
            self.graph_def.termination_condition,
            agent_result,
            context,
        )
    
    def reset(self) -> None:
        """Reset all agent memories."""
        for runtime in self.agent_runtimes.values():
            runtime.reset()
