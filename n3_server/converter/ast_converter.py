"""
N3 AST to Graph JSON Converter

Bidirectional converter between N3 compiler AST and graph editor JSON format.
Maps N3 constructs (agents, chains, prompts, RAG, etc.) to visual graph nodes.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from nanoid import generate

# Import N3 AST nodes
from namel3ss.ast.agents import AgentDefinition, GraphDefinition, GraphEdge as N3GraphEdge
from namel3ss.ast.ai_workflows import Chain, ChainStep, WorkflowIfBlock, WorkflowForBlock, WorkflowWhileBlock
from namel3ss.ast.ai_prompts import Prompt
from namel3ss.ast.rag import IndexDefinition, RagPipelineDefinition
from namel3ss.ast.ai_tools import ToolDefinition


@dataclass
class GraphNode:
    """Graph editor node format."""
    id: str
    type: str
    label: str
    data: Dict[str, Any]
    position: Optional[Dict[str, float]] = None


@dataclass
class GraphEdge:
    """Graph editor edge format."""
    id: str
    source: str
    target: str
    label: Optional[str] = None
    conditionExpr: Optional[str] = None


@dataclass
class GraphJSON:
    """Complete graph JSON structure."""
    projectId: str
    name: str
    chains: List[Dict[str, str]]
    agents: List[Dict[str, str]]
    activeRootId: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class N3ASTConverter:
    """Convert between N3 AST and graph editor JSON format."""
    
    def __init__(self):
        self.node_positions = {}  # Cache for layout positions
        self.x_offset = 100
        self.y_offset = 200
        self.x_spacing = 250
        self.y_spacing = 150
    
    def _generate_id(self, prefix: str = "node") -> str:
        """Generate unique node ID."""
        return f"{prefix}-{generate(size=8)}"
    
    def _get_position(self, node_id: str, depth: int = 0, index: int = 0) -> Dict[str, float]:
        """Calculate node position for layout."""
        if node_id in self.node_positions:
            return self.node_positions[node_id]
        
        position = {
            "x": self.x_offset + (depth * self.x_spacing),
            "y": self.y_offset + (index * self.y_spacing)
        }
        self.node_positions[node_id] = position
        return position
    
    # ============= AST to Graph JSON =============
    
    def agent_to_node(self, agent: AgentDefinition, depth: int = 0, index: int = 0) -> GraphNode:
        """Convert N3 AgentDefinition to graph node."""
        node_id = self._generate_id("agent")
        
        return GraphNode(
            id=node_id,
            type="agent",
            label=agent.name,
            data={
                "name": agent.name,
                "llm": agent.llm_name,
                "tools": agent.tool_names,
                "memory": str(agent.memory_config) if agent.memory_config else "none",
                "goal": agent.goal,
                "systemPrompt": agent.system_prompt,
                "maxTurns": agent.max_turns,
                "temperature": agent.temperature,
                "config": agent.config,
            },
            position=self._get_position(node_id, depth, index)
        )
    
    def prompt_to_node(self, prompt: Prompt, depth: int = 0, index: int = 0) -> GraphNode:
        """Convert N3 Prompt to graph node."""
        node_id = self._generate_id("prompt")
        
        # Extract template text
        template_text = ""
        if hasattr(prompt, 'template') and prompt.template:
            if isinstance(prompt.template, str):
                template_text = prompt.template
            elif hasattr(prompt.template, 'text'):
                template_text = prompt.template.text
        
        return GraphNode(
            id=node_id,
            type="prompt",
            label=prompt.name,
            data={
                "name": prompt.name,
                "text": template_text,
                "model": prompt.model if hasattr(prompt, 'model') else None,
                "temperature": prompt.temperature if hasattr(prompt, 'temperature') else None,
                "arguments": [arg.name for arg in (prompt.arguments or [])],
                "outputSchema": prompt.output_schema.to_dict() if hasattr(prompt, 'output_schema') and prompt.output_schema else None,
            },
            position=self._get_position(node_id, depth, index)
        )
    
    def rag_to_node(self, rag: RagPipelineDefinition, depth: int = 0, index: int = 0) -> GraphNode:
        """Convert N3 RAG pipeline to graph node."""
        node_id = self._generate_id("rag")
        
        return GraphNode(
            id=node_id,
            type="ragDataset",
            label=rag.name,
            data={
                "name": rag.name,
                "queryEncoder": rag.query_encoder,
                "index": rag.index,
                "topK": rag.top_k if hasattr(rag, 'top_k') else 5,
                "reranker": rag.reranker if hasattr(rag, 'reranker') else None,
                "distanceMetric": rag.distance_metric if hasattr(rag, 'distance_metric') else "cosine",
                "enableHybrid": getattr(rag, 'enable_hybrid', False),
            },
            position=self._get_position(node_id, depth, index)
        )
    
    def chain_step_to_nodes(self, step: ChainStep, depth: int = 0, index: int = 0) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Convert ChainStep to graph node(s) and edges."""
        nodes = []
        edges = []
        
        if step.kind == "prompt":
            node_id = self._generate_id("step")
            node = GraphNode(
                id=node_id,
                type="prompt",
                label=step.name or step.target,
                data={
                    "target": step.target,
                    "options": step.options,
                    "stopOnError": step.stop_on_error,
                    "evaluation": step.evaluation.__dict__ if step.evaluation else None,
                },
                position=self._get_position(node_id, depth, index)
            )
            nodes.append(node)
        
        elif step.kind == "tool":
            node_id = self._generate_id("tool")
            node = GraphNode(
                id=node_id,
                type="pythonHook",
                label=step.name or step.target,
                data={
                    "target": step.target,
                    "options": step.options,
                },
                position=self._get_position(node_id, depth, index)
            )
            nodes.append(node)
        
        elif step.kind == "knowledge_query":
            node_id = self._generate_id("knowledge")
            node = GraphNode(
                id=node_id,
                type="ragDataset",
                label=step.name or step.target,
                data={
                    "target": step.target,
                    "options": step.options,
                },
                position=self._get_position(node_id, depth, index)
            )
            nodes.append(node)
        
        return nodes, edges
    
    def condition_to_node(self, condition_expr: str, depth: int = 0, index: int = 0) -> GraphNode:
        """Convert conditional expression to condition node."""
        node_id = self._generate_id("condition")
        
        return GraphNode(
            id=node_id,
            type="condition",
            label="Condition",
            data={
                "expression": str(condition_expr),
            },
            position=self._get_position(node_id, depth, index)
        )
    
    def chain_to_graph(self, chain: Chain) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Convert N3 Chain to graph nodes and edges."""
        nodes = []
        edges = []
        
        # Create start node
        start_id = self._generate_id("start")
        start_node = GraphNode(
            id=start_id,
            type="start",
            label="START",
            data={"chainName": chain.name},
            position=self._get_position(start_id, 0, 0)
        )
        nodes.append(start_node)
        
        # Process steps
        prev_node_id = start_id
        step_index = 0
        
        for step in chain.steps:
            step_nodes, step_edges = self.chain_step_to_nodes(step, depth=step_index + 1, index=0)
            nodes.extend(step_nodes)
            edges.extend(step_edges)
            
            # Connect to previous node
            if step_nodes:
                edge_id = self._generate_id("edge")
                edge = GraphEdge(
                    id=edge_id,
                    source=prev_node_id,
                    target=step_nodes[0].id,
                )
                edges.append(edge)
                prev_node_id = step_nodes[-1].id
            
            step_index += 1
        
        # Create end node
        end_id = self._generate_id("end")
        end_node = GraphNode(
            id=end_id,
            type="end",
            label="END",
            data={},
            position=self._get_position(end_id, step_index + 1, 0)
        )
        nodes.append(end_node)
        
        # Connect last step to end
        edge_id = self._generate_id("edge")
        edge = GraphEdge(
            id=edge_id,
            source=prev_node_id,
            target=end_id,
        )
        edges.append(edge)
        
        return nodes, edges
    
    def agent_graph_to_graph(self, graph: GraphDefinition) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Convert N3 GraphDefinition (multi-agent) to graph nodes and edges."""
        nodes = []
        edges = []
        
        # Create start node pointing to start_agent
        start_id = self._generate_id("start")
        start_node = GraphNode(
            id=start_id,
            type="start",
            label="START",
            data={"graphName": graph.name},
            position=self._get_position(start_id, 0, 0)
        )
        nodes.append(start_node)
        
        # Create agent nodes (these reference AgentDefinition by name)
        agent_node_map = {}
        for idx, edge in enumerate(graph.edges):
            for agent_name in [edge.from_agent, edge.to_agent]:
                if agent_name not in agent_node_map:
                    node_id = self._generate_id("agentref")
                    agent_node = GraphNode(
                        id=node_id,
                        type="agent",
                        label=agent_name,
                        data={
                            "name": agent_name,
                            "reference": True,  # This references an AgentDefinition
                        },
                        position=self._get_position(node_id, 1, len(agent_node_map))
                    )
                    nodes.append(agent_node)
                    agent_node_map[agent_name] = node_id
        
        # Connect start to start_agent
        if graph.start_agent in agent_node_map:
            edge_id = self._generate_id("edge")
            edge = GraphEdge(
                id=edge_id,
                source=start_id,
                target=agent_node_map[graph.start_agent],
            )
            edges.append(edge)
        
        # Create edges between agents
        for n3_edge in graph.edges:
            if n3_edge.from_agent in agent_node_map and n3_edge.to_agent in agent_node_map:
                edge_id = self._generate_id("edge")
                edge = GraphEdge(
                    id=edge_id,
                    source=agent_node_map[n3_edge.from_agent],
                    target=agent_node_map[n3_edge.to_agent],
                    label=str(n3_edge.condition),
                    conditionExpr=str(n3_edge.condition),
                )
                edges.append(edge)
        
        # Create end node
        end_id = self._generate_id("end")
        end_node = GraphNode(
            id=end_id,
            type="end",
            label="END",
            data={},
            position=self._get_position(end_id, 2, 0)
        )
        nodes.append(end_node)
        
        # Connect termination agents to end
        for term_agent in graph.termination_agents:
            if term_agent in agent_node_map:
                edge_id = self._generate_id("edge")
                edge = GraphEdge(
                    id=edge_id,
                    source=agent_node_map[term_agent],
                    target=end_id,
                )
                edges.append(edge)
        
        return nodes, edges
    
    def ast_to_graph_json(
        self,
        project_id: str,
        name: str,
        chains: List[Chain] = None,
        agents: List[AgentDefinition] = None,
        agent_graphs: List[GraphDefinition] = None,
        prompts: List[Prompt] = None,
        rags: List[RagPipelineDefinition] = None,
    ) -> GraphJSON:
        """Convert N3 AST components to complete graph JSON."""
        all_nodes = []
        all_edges = []
        chain_info = []
        agent_info = []
        
        # Convert chains
        if chains:
            for chain in chains:
                nodes, edges = self.chain_to_graph(chain)
                all_nodes.extend(nodes)
                all_edges.extend(edges)
                chain_info.append({"id": self._generate_id("chain"), "name": chain.name})
        
        # Convert agent graphs
        if agent_graphs:
            for graph in agent_graphs:
                nodes, edges = self.agent_graph_to_graph(graph)
                all_nodes.extend(nodes)
                all_edges.extend(edges)
        
        # Convert standalone agents
        if agents:
            for idx, agent in enumerate(agents):
                node = self.agent_to_node(agent, depth=0, index=idx)
                all_nodes.append(node)
                agent_info.append({"id": node.id, "name": agent.name})
        
        # Convert standalone prompts
        if prompts:
            for idx, prompt in enumerate(prompts):
                node = self.prompt_to_node(prompt, depth=0, index=idx + len(agents or []))
                all_nodes.append(node)
        
        # Convert RAG pipelines
        if rags:
            for idx, rag in enumerate(rags):
                node = self.rag_to_node(rag, depth=0, index=idx + len(agents or []) + len(prompts or []))
                all_nodes.append(node)
        
        # Find or create root node
        start_nodes = [n for n in all_nodes if n.type == "start"]
        active_root_id = start_nodes[0].id if start_nodes else (all_nodes[0].id if all_nodes else "")
        
        return GraphJSON(
            projectId=project_id,
            name=name,
            chains=chain_info,
            agents=agent_info,
            activeRootId=active_root_id,
            nodes=[self._node_to_dict(n) for n in all_nodes],
            edges=[self._edge_to_dict(e) for e in all_edges],
            metadata={}
        )
    
    def _node_to_dict(self, node: GraphNode) -> Dict[str, Any]:
        """Convert GraphNode to dictionary."""
        return {
            "id": node.id,
            "type": node.type,
            "label": node.label,
            "data": node.data,
            "position": node.position,
        }
    
    def _edge_to_dict(self, edge: GraphEdge) -> Dict[str, Any]:
        """Convert GraphEdge to dictionary."""
        result = {
            "id": edge.id,
            "source": edge.source,
            "target": edge.target,
        }
        if edge.label:
            result["label"] = edge.label
        if edge.conditionExpr:
            result["conditionExpr"] = edge.conditionExpr
        return result
    
    # ============= Graph JSON to AST =============
    
    def graph_json_to_chain(self, nodes: List[Dict], edges: List[Dict], chain_name: str) -> Chain:
        """Convert graph JSON back to N3 Chain AST."""
        # Find start and end nodes
        start_nodes = [n for n in nodes if n["type"] == "start"]
        end_nodes = [n for n in nodes if n["type"] == "end"]
        
        if not start_nodes:
            raise ValueError("No start node found in graph")
        
        # Build adjacency list
        adjacency = {}
        for edge in edges:
            source = edge["source"]
            if source not in adjacency:
                adjacency[source] = []
            adjacency[source].append(edge)
        
        # Traverse from start to build steps
        steps = []
        current_id = start_nodes[0]["id"]
        visited = set()
        
        while current_id and current_id not in visited:
            visited.add(current_id)
            
            # Find current node
            current_node = next((n for n in nodes if n["id"] == current_id), None)
            if not current_node or current_node["type"] in ["start", "end"]:
                # Move to next
                if current_id in adjacency:
                    next_edge = adjacency[current_id][0]
                    current_id = next_edge["target"]
                else:
                    break
                continue
            
            # Convert node to ChainStep
            node_type = current_node["type"]
            node_data = current_node["data"]
            
            if node_type == "prompt":
                step = ChainStep(
                    kind="prompt",
                    target=node_data.get("target", node_data.get("name", "")),
                    options=node_data.get("options", {}),
                    name=current_node["label"],
                    stop_on_error=node_data.get("stopOnError", True),
                )
                steps.append(step)
            
            elif node_type == "pythonHook":
                step = ChainStep(
                    kind="tool",
                    target=node_data.get("target", node_data.get("name", "")),
                    options=node_data.get("options", {}),
                    name=current_node["label"],
                )
                steps.append(step)
            
            elif node_type == "ragDataset":
                step = ChainStep(
                    kind="knowledge_query",
                    target=node_data.get("target", node_data.get("name", "")),
                    options=node_data.get("options", {}),
                    name=current_node["label"],
                )
                steps.append(step)
            
            # Move to next node
            if current_id in adjacency:
                next_edge = adjacency[current_id][0]
                current_id = next_edge["target"]
            else:
                break
        
        return Chain(
            name=chain_name,
            steps=steps,
            input_key="input",
            output_key="output",
        )
    
    def graph_json_to_agent(self, node: Dict) -> AgentDefinition:
        """Convert graph node back to N3 AgentDefinition."""
        data = node["data"]
        
        return AgentDefinition(
            name=data.get("name", node["label"]),
            llm_name=data.get("llm", ""),
            tool_names=data.get("tools", []),
            memory_config=data.get("memory"),
            goal=data.get("goal", ""),
            system_prompt=data.get("systemPrompt"),
            max_turns=data.get("maxTurns"),
            temperature=data.get("temperature"),
            config=data.get("config", {}),
        )
