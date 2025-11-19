"""RAG and retrieval-centric AST nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import Expression


@dataclass
class IndexDefinition:
    """
    First-class index block defining a retrieval index built from a dataset.
    
    Example DSL syntax:
        index docs_index {
            source_dataset: "product_docs"
            embedding_model: "text-embedding-3-small"
            chunk_size: 512
            overlap: 64
            backend: "pgvector"
            namespace: "docs"
            metadata_fields: ["category", "product_id"]
        }
    """
    name: str
    source_dataset: str
    embedding_model: str
    chunk_size: int = 512
    overlap: int = 64
    backend: str = "pgvector"
    namespace: Optional[str] = None
    collection: Optional[str] = None
    table_name: Optional[str] = None
    metadata_fields: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[Any] = None  # SourceLocation


@dataclass
class RagPipelineDefinition:
    """
    First-class rag_pipeline block defining a declarative retrieval configuration.
    
    Example DSL syntax:
        rag_pipeline support_rag {
            query_encoder: "text-embedding-3-small"
            index: docs_index
            top_k: 8
            reranker: "bge-reranker"
            distance_metric: "cosine"
            filters: {category: "support"}
        }
    """
    name: str
    query_encoder: str
    index: str  # Reference to IndexDefinition by name
    top_k: int = 5
    reranker: Optional[str] = None
    distance_metric: str = "cosine"
    filters: Optional[Expression] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[Any] = None  # SourceLocation


__all__ = [
    "IndexDefinition",
    "RagPipelineDefinition",
]
