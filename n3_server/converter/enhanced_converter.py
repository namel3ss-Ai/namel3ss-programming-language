"""
Production-grade N3 AST Converter with full validation and error handling.

Converts visual graph JSON to N3 AST with:
- Strict Pydantic v2 validation
- Comprehensive error messages
- Idempotent conversion
- Extensible node type handlers
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass

from n3_server.converter.models import (
    GraphJSON,
    GraphNode,
    GraphEdge,
    NodeType,
    AgentNodeData,
    PromptNodeData,
    RagNodeData,
    ToolNodeData,
    ConditionNodeData,
    ConversionError,
)

# Import N3 AST types
from namel3ss.ast.ai_workflows import Chain, ChainStep
from namel3ss.ast.agents import AgentDefinition, GraphDefinition, GraphEdge as N3GraphEdge
from namel3ss.ast.ai.prompts import Prompt, PromptField, PromptArgument
from namel3ss.ast.rag import RagPipelineDefinition
from namel3ss.ast.ai.tools import ToolDefinition

logger = logging.getLogger(__name__)


@dataclass
class ConversionContext:
    """Context for graph to AST conversion."""
    project_id: str
    graph_name: str
    visited_nodes: Set[str]
    agent_registry: Dict[str, AgentDefinition]
    prompt_registry: Dict[str, Prompt]
    rag_registry: Dict[str, RagPipelineDefinition]
    tool_registry: Dict[str, ToolDefinition]
    
    def mark_visited(self, node_id: str) -> None:
        """Mark a node as visited to detect cycles."""
        if node_id in self.visited_nodes:
            raise ConversionError(
                f"Cycle detected: node {node_id} already visited",
                node_id=node_id
            )
        self.visited_nodes.add(node_id)
    
    def is_visited(self, node_id: str) -> bool:
        """Check if node was visited."""
        return node_id in self.visited_nodes


class EnhancedN3ASTConverter:
    """
    Production-grade converter between graph JSON and N3 AST.
    
    Features:
    - Full Pydantic v2 validation
    - Cycle detection
    - Comprehensive error messages
    - Extensible node handlers
    - Idempotent conversion
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    # ============= Graph JSON to N3 AST =============
    
    def convert_graph_to_chain(
        self,
        graph_json: Union[Dict[str, Any], GraphJSON],
        entry_node_id: Optional[str] = None,
    ) -> Tuple[Chain, ConversionContext]:
        """
        Convert graph JSON to N3 Chain AST with full validation.
        
        Args:
            graph_json: Raw graph JSON data or validated GraphJSON
            entry_node_id: Optional entry point (defaults to start node)
        
        Returns:
            Tuple of (Chain AST, ConversionContext with registries)
        
        Raises:
            ConversionError: If conversion fails with detailed context
        """
        try:
            # Validate graph structure if dict, otherwise use validated model
            if isinstance(graph_json, dict):
                validated_graph = GraphJSON(**graph_json)
            else:
                validated_graph = graph_json
        except Exception as e:
            raise ConversionError(f"Graph validation failed: {str(e)}")
        
        # Create conversion context
        context = ConversionContext(
            project_id=validated_graph.projectId,
            graph_name=validated_graph.name,
            visited_nodes=set(),
            agent_registry={},
            prompt_registry={},
            rag_registry={},
            tool_registry={},
        )
        
        # Find entry node
        if entry_node_id:
            entry_node = validated_graph.get_node_by_id(entry_node_id)
            if not entry_node:
                raise ConversionError(f"Entry node not found: {entry_node_id}")
        else:
            # Use first start node
            start_nodes = [n for n in validated_graph.nodes if n.type == NodeType.START]
            if not start_nodes:
                raise ConversionError("No START node found in graph")
            entry_node = start_nodes[0]
        
        # Build registries from standalone nodes
        self._build_registries(validated_graph, context)
        
        # Convert graph to chain steps
        chain_steps = self._traverse_and_convert(
            validated_graph,
            entry_node,
            context
        )
        
        # Build Chain AST
        chain = Chain(
            name=validated_graph.name or "unnamed_chain",
            steps=chain_steps,
            input_key="input",
        )
        
        self.logger.info(
            f"Successfully converted graph to chain: {chain.name} "
            f"with {len(chain_steps)} steps"
        )
        
        return chain, context
    
    def _build_registries(self, graph: GraphJSON, context: ConversionContext) -> None:
        """Build agent, prompt, RAG, and tool registries from graph nodes."""
        for node in graph.nodes:
            try:
                if node.type == NodeType.AGENT and isinstance(node.data, AgentNodeData):
                    agent = self._convert_agent_node(node)
                    context.agent_registry[agent.name] = agent
                
                elif node.type == NodeType.PROMPT and isinstance(node.data, PromptNodeData):
                    prompt = self._convert_prompt_node(node)
                    context.prompt_registry[prompt.name] = prompt
                
                elif node.type == NodeType.RAG_DATASET and isinstance(node.data, RagNodeData):
                    rag = self._convert_rag_node(node)
                    context.rag_registry[rag.name] = rag
                
                elif node.type == NodeType.PYTHON_HOOK and isinstance(node.data, ToolNodeData):
                    tool = self._convert_tool_node(node)
                    context.tool_registry[tool.name] = tool
            
            except Exception as e:
                self.logger.warning(
                    f"Failed to register node {node.id} ({node.type}): {e}"
                )
    
    def _traverse_and_convert(
        self,
        graph: GraphJSON,
        current_node: GraphNode,
        context: ConversionContext,
    ) -> List[ChainStep]:
        """
        Traverse graph from entry node and convert to chain steps.
        
        Uses depth-first traversal with cycle detection.
        """
        steps = []
        node_queue = [current_node]
        
        while node_queue:
            node = node_queue.pop(0)
            
            # Skip if visited (prevent cycles)
            if context.is_visited(node.id):
                continue
            
            context.mark_visited(node.id)
            
            # Skip start/end nodes
            if node.type in (NodeType.START, NodeType.END):
                # Get next nodes
                outgoing = graph.get_edges_from_node(node.id)
                for edge in outgoing:
                    next_node = graph.get_node_by_id(edge.target)
                    if next_node:
                        node_queue.append(next_node)
                continue
            
            # Convert node to chain step
            try:
                step = self._convert_node_to_step(node, context)
                if step:
                    steps.append(step)
            except Exception as e:
                raise ConversionError(
                    f"Failed to convert node to step: {e}",
                    node_id=node.id,
                    details={"node_type": node.type, "label": node.label}
                )
            
            # Add connected nodes to queue
            outgoing = graph.get_edges_from_node(node.id)
            for edge in outgoing:
                next_node = graph.get_node_by_id(edge.target)
                if next_node and not context.is_visited(next_node.id):
                    node_queue.append(next_node)
        
        return steps
    
    def _convert_node_to_step(
        self,
        node: GraphNode,
        context: ConversionContext,
    ) -> Optional[ChainStep]:
        """Convert a graph node to a ChainStep."""
        if node.type == NodeType.PROMPT:
            return self._prompt_node_to_step(node, context)
        
        elif node.type == NodeType.AGENT:
            return self._agent_node_to_step(node, context)
        
        elif node.type == NodeType.RAG_DATASET:
            return self._rag_node_to_step(node, context)
        
        elif node.type == NodeType.PYTHON_HOOK:
            return self._tool_node_to_step(node, context)
        
        elif node.type == NodeType.CONDITION:
            # Conditions are handled as edges, not steps
            return None
        
        else:
            self.logger.warning(f"Unsupported node type for conversion: {node.type}")
            return None
    
    # ============= Node to AST Converters =============
    
    def _convert_agent_node(self, node: GraphNode) -> AgentDefinition:
        """Convert agent node to AgentDefinition."""
        if not isinstance(node.data, AgentNodeData):
            raise ConversionError(
                f"Invalid agent node data type: {type(node.data)}",
                node_id=node.id
            )
        
        data = node.data
        
        return AgentDefinition(
            name=data.name,
            llm_name=data.llm,
            tool_names=data.tools,
            memory_config=data.memory,
            goal=data.goal or "",
            system_prompt=data.systemPrompt,
            max_turns=data.maxTurns or 10,
            temperature=data.temperature,
            config=data.config,
        )
    
    def _convert_prompt_node(self, node: GraphNode) -> Prompt:
        """Convert prompt node to Prompt."""
        if not isinstance(node.data, PromptNodeData):
            raise ConversionError(
                f"Invalid prompt node data type: {type(node.data)}",
                node_id=node.id
            )
        
        data = node.data
        
        # Convert arguments to PromptArgument list
        args = []
        for arg_name in data.arguments:
            args.append(PromptArgument(
                name=arg_name,
                arg_type="string",  # Default type
                required=True,
            ))
        
        return Prompt(
            name=data.name,
            model=data.model or "",
            template=data.text,
            args=args,
            output_schema=data.outputSchema,
            parameters={
                "temperature": data.temperature,
                "max_tokens": data.maxTokens,
            } if data.temperature or data.maxTokens else {},
            metadata={},
            description=None,
            effects=set(),
        )
    
    def _convert_rag_node(self, node: GraphNode) -> RagPipelineDefinition:
        """Convert RAG node to RagPipelineDefinition."""
        if not isinstance(node.data, RagNodeData):
            raise ConversionError(
                f"Invalid RAG node data type: {type(node.data)}",
                node_id=node.id
            )
        
        data = node.data
        
        return RagPipelineDefinition(
            name=data.name,
            query_encoder=data.queryEncoder or "",
            index=data.index or "",
            top_k=data.topK,
            reranker=data.reranker,
            distance_metric=data.distanceMetric or "cosine",
            enable_hybrid=data.enableHybrid or False,
        )
    
    def _convert_tool_node(self, node: GraphNode) -> ToolDefinition:
        """Convert tool node to ToolDefinition."""
        if not isinstance(node.data, ToolNodeData):
            raise ConversionError(
                f"Invalid tool node data type: {type(node.data)}",
                node_id=node.id
            )
        
        data = node.data
        
        return ToolDefinition(
            name=data.name,
            source="python",
            function_name=data.target,
            description=f"Tool: {data.name}",
            parameters={},
        )
    
    # ============= Node to ChainStep Converters =============
    
    def _prompt_node_to_step(
        self,
        node: GraphNode,
        context: ConversionContext,
    ) -> ChainStep:
        """Convert prompt node to ChainStep."""
        if not isinstance(node.data, PromptNodeData):
            raise ConversionError(
                f"Invalid prompt node data",
                node_id=node.id
            )
        
        data = node.data
        
        # Ensure prompt is in registry
        if data.name not in context.prompt_registry:
            prompt = self._convert_prompt_node(node)
            context.prompt_registry[prompt.name] = prompt
        
        return ChainStep(
            kind="prompt",
            target=data.name,
            name=node.label or data.name,
            options={},
            stop_on_error=True,
        )
    
    def _agent_node_to_step(
        self,
        node: GraphNode,
        context: ConversionContext,
    ) -> ChainStep:
        """Convert agent node to ChainStep."""
        if not isinstance(node.data, AgentNodeData):
            raise ConversionError(
                f"Invalid agent node data",
                node_id=node.id
            )
        
        data = node.data
        
        # Ensure agent is in registry
        if data.name not in context.agent_registry:
            agent = self._convert_agent_node(node)
            context.agent_registry[agent.name] = agent
        
        return ChainStep(
            kind="agent",
            target=data.name,
            name=node.label or data.name,
            options={
                "goal": data.goal or "Execute task",
                "max_turns": data.maxTurns or 10,
            },
            stop_on_error=True,
        )
    
    def _rag_node_to_step(
        self,
        node: GraphNode,
        context: ConversionContext,
    ) -> ChainStep:
        """Convert RAG node to ChainStep."""
        if not isinstance(node.data, RagNodeData):
            raise ConversionError(
                f"Invalid RAG node data",
                node_id=node.id
            )
        
        data = node.data
        
        # Ensure RAG is in registry
        if data.name not in context.rag_registry:
            rag = self._convert_rag_node(node)
            context.rag_registry[rag.name] = rag
        
        return ChainStep(
            kind="knowledge_query",
            target=data.name,
            name=node.label or data.name,
            options={
                "top_k": data.topK,
                "reranker": data.reranker,
            },
            stop_on_error=True,
        )
    
    def _tool_node_to_step(
        self,
        node: GraphNode,
        context: ConversionContext,
    ) -> ChainStep:
        """Convert tool node to ChainStep."""
        if not isinstance(node.data, ToolNodeData):
            raise ConversionError(
                f"Invalid tool node data",
                node_id=node.id
            )
        
        data = node.data
        
        # Ensure tool is in registry
        if data.name not in context.tool_registry:
            tool = self._convert_tool_node(node)
            context.tool_registry[tool.name] = tool
        
        return ChainStep(
            kind="tool",
            target=data.target,
            name=node.label or data.name,
            options=data.options,
            stop_on_error=True,
        )
    
    # ============= Utility Methods =============
    
    def validate_graph(self, graph_json: Union[Dict[str, Any], GraphJSON]) -> GraphJSON:
        """
        Validate graph JSON structure.
        
        Args:
            graph_json: Raw graph JSON or validated GraphJSON
        
        Returns:
            Validated GraphJSON model
        
        Raises:
            ConversionError: If validation fails
        """
        try:
            if isinstance(graph_json, dict):
                return GraphJSON(**graph_json)
            else:
                return graph_json
        except Exception as e:
            raise ConversionError(f"Graph validation failed: {str(e)}")
    
    def get_conversion_summary(self, context: ConversionContext) -> Dict[str, Any]:
        """Get summary of conversion results."""
        total_components = (
            len(context.agent_registry) +
            len(context.prompt_registry) +
            len(context.rag_registry) +
            len(context.tool_registry)
        )
        return {
            "project_id": context.project_id,
            "graph_name": context.graph_name,
            "nodes_converted": len(context.visited_nodes),
            "total_nodes": total_components,
            "agents": len(context.agent_registry),
            "prompts": len(context.prompt_registry),
            "rag_pipelines": len(context.rag_registry),
            "tools": len(context.tool_registry),
        }


__all__ = [
    "EnhancedN3ASTConverter",
    "ConversionContext",
    "ConversionError",
]
