"""
Pydantic v2 validation models for graph JSON structure.

These models provide strict typing and validation for graph nodes and edges
to ensure data integrity before conversion to N3 AST.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class NodeType(str, Enum):
    """Valid node types in the graph editor."""
    START = "start"
    END = "end"
    AGENT = "agent"
    PROMPT = "prompt"
    RAG_DATASET = "ragDataset"
    PYTHON_HOOK = "pythonHook"
    CONDITION = "condition"
    LOOP = "loop"
    SUBGRAPH = "subgraph"


class Position(BaseModel):
    """Node position in the canvas."""
    x: float
    y: float


class AgentNodeData(BaseModel):
    """Data for agent node."""
    name: str
    llm: str
    tools: List[str] = Field(default_factory=list)
    memory: Optional[str] = None
    goal: Optional[str] = None
    systemPrompt: Optional[str] = Field(default=None, alias="system_prompt")
    maxTurns: Optional[int] = Field(default=10, alias="max_turns")
    temperature: Optional[float] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        populate_by_name = True


class PromptNodeData(BaseModel):
    """Data for prompt node."""
    name: str
    text: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    maxTokens: Optional[int] = Field(default=None, alias="max_tokens")
    arguments: List[str] = Field(default_factory=list)
    outputSchema: Optional[Dict[str, Any]] = Field(default=None, alias="output_schema")
    
    class Config:
        populate_by_name = True


class RagNodeData(BaseModel):
    """Data for RAG dataset node."""
    name: str
    queryEncoder: Optional[str] = Field(default=None, alias="query_encoder")
    index: Optional[str] = None
    topK: int = Field(default=5, alias="top_k")
    reranker: Optional[str] = None
    distanceMetric: str = Field(default="cosine", alias="distance_metric")
    enableHybrid: bool = Field(default=False, alias="enable_hybrid")
    
    class Config:
        populate_by_name = True


class ToolNodeData(BaseModel):
    """Data for tool/Python hook node."""
    name: str
    target: str
    options: Dict[str, Any] = Field(default_factory=dict)


class ConditionNodeData(BaseModel):
    """Data for condition node."""
    expression: str
    description: Optional[str] = None


class StartEndNodeData(BaseModel):
    """Data for start/end nodes."""
    chainName: Optional[str] = Field(default=None, alias="chain_name")
    graphName: Optional[str] = Field(default=None, alias="graph_name")
    
    class Config:
        populate_by_name = True


class GraphNode(BaseModel):
    """Graph editor node with validated data."""
    id: str
    type: NodeType
    label: str
    data: Union[
        AgentNodeData,
        PromptNodeData,
        RagNodeData,
        ToolNodeData,
        ConditionNodeData,
        StartEndNodeData,
        Dict[str, Any]  # Fallback for unknown node types
    ]
    position: Optional[Position] = None
    
    @field_validator('data', mode='before')
    @classmethod
    def validate_data(cls, v: Any, info) -> Any:
        """Validate data based on node type."""
        # If already validated, return as-is
        if isinstance(v, BaseModel):
            return v
        
        if not isinstance(v, dict):
            return v
        
        # Get node type from validation info
        node_type = info.data.get('type')
        
        # Map node type to data model
        type_map = {
            NodeType.AGENT: AgentNodeData,
            NodeType.PROMPT: PromptNodeData,
            NodeType.RAG_DATASET: RagNodeData,
            NodeType.PYTHON_HOOK: ToolNodeData,
            NodeType.CONDITION: ConditionNodeData,
            NodeType.START: StartEndNodeData,
            NodeType.END: StartEndNodeData,
        }
        
        data_model = type_map.get(node_type)
        if data_model:
            return data_model(**v)
        
        return v


class GraphEdge(BaseModel):
    """Graph editor edge with optional conditions."""
    id: str
    source: str
    target: str
    label: Optional[str] = None
    conditionExpr: Optional[str] = Field(default=None, alias="condition_expr")
    
    class Config:
        populate_by_name = True
    
    @field_validator('source', 'target')
    @classmethod
    def validate_node_refs(cls, v: str) -> str:
        """Ensure node references are non-empty."""
        if not v or not v.strip():
            raise ValueError("Node reference cannot be empty")
        return v


class ChainInfo(BaseModel):
    """Chain metadata."""
    id: str
    name: str


class AgentInfo(BaseModel):
    """Agent metadata."""
    id: str
    name: str


class GraphMetadata(BaseModel):
    """Graph metadata and settings."""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    author: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    max_hops: Optional[int] = Field(default=32, alias="maxHops")
    timeout_ms: Optional[int] = Field(default=None, alias="timeoutMs")
    
    class Config:
        populate_by_name = True


class GraphJSON(BaseModel):
    """Complete validated graph JSON structure."""
    projectId: str = Field(alias="project_id")
    name: str
    chains: List[ChainInfo] = Field(default_factory=list)
    agents: List[AgentInfo] = Field(default_factory=list)
    activeRootId: Optional[str] = Field(default=None, alias="active_root_id")
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: GraphMetadata = Field(default_factory=GraphMetadata)
    
    class Config:
        populate_by_name = True
    
    @model_validator(mode='after')
    def validate_graph_structure(self) -> 'GraphJSON':
        """Validate graph structure integrity."""
        # Collect node IDs
        node_ids = {node.id for node in self.nodes}
        
        # Validate edge references
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(f"Edge {edge.id} references unknown source node: {edge.source}")
            if edge.target not in node_ids:
                raise ValueError(f"Edge {edge.id} references unknown target node: {edge.target}")
        
        # Validate activeRootId
        if self.activeRootId and self.activeRootId not in node_ids:
            raise ValueError(f"Active root ID {self.activeRootId} does not exist in nodes")
        
        # Ensure at least one start node
        start_nodes = [n for n in self.nodes if n.type == NodeType.START]
        if not start_nodes:
            raise ValueError("Graph must have at least one START node")
        
        return self
    
    def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
        return next((n for n in self.nodes if n.id == node_id), None)
    
    def get_edges_from_node(self, node_id: str) -> List[GraphEdge]:
        """Get all outgoing edges from a node."""
        return [e for e in self.edges if e.source == node_id]
    
    def get_edges_to_node(self, node_id: str) -> List[GraphEdge]:
        """Get all incoming edges to a node."""
        return [e for e in self.edges if e.target == node_id]


class ConversionError(Exception):
    """Raised when graph to AST conversion fails."""
    
    def __init__(self, message: str, node_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.node_id = node_id
        self.details = details or {}
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format error message with context."""
        msg = f"Graph conversion error: {self.message}"
        if self.node_id:
            msg += f" (node: {self.node_id})"
        if self.details:
            msg += f" | Details: {self.details}"
        return msg


__all__ = [
    "NodeType",
    "Position",
    "AgentNodeData",
    "PromptNodeData",
    "RagNodeData",
    "ToolNodeData",
    "ConditionNodeData",
    "StartEndNodeData",
    "GraphNode",
    "GraphEdge",
    "ChainInfo",
    "AgentInfo",
    "GraphMetadata",
    "GraphJSON",
    "ConversionError",
]
