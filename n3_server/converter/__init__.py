"""
N3 AST Converter package.

Provides bidirectional conversion between visual graph JSON and N3 AST with:
- Legacy converter (N3ASTConverter)
- Enhanced converter with Pydantic v2 validation (EnhancedN3ASTConverter)
"""

from .ast_converter import N3ASTConverter, GraphNode, GraphEdge, GraphJSON as LegacyGraphJSON
from .enhanced_converter import (
    EnhancedN3ASTConverter,
    ConversionContext,
)
from .models import (
    GraphJSON,
    GraphNode as ValidatedGraphNode,
    GraphEdge as ValidatedGraphEdge,
    NodeType,
    ConversionError,
    AgentNodeData,
    PromptNodeData,
    RagNodeData,
    ToolNodeData,
)

__all__ = [
    # Legacy converter
    "N3ASTConverter",
    "GraphNode",
    "GraphEdge",
    "LegacyGraphJSON",
    # Enhanced converter
    "EnhancedN3ASTConverter",
    "ConversionContext",
    # Validated models
    "GraphJSON",
    "ValidatedGraphNode",
    "ValidatedGraphEdge",
    "NodeType",
    "ConversionError",
    "AgentNodeData",
    "PromptNodeData",
    "RagNodeData",
    "ToolNodeData",
]

