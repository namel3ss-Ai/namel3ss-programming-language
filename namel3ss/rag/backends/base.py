"""Base vector index backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class ScoredDocument:
    """A document with similarity score."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class VectorIndexBackend(ABC):
    """
    Abstract base class for vector database backends.
    
    Implementations must handle connection management, schema setup,
    and provide efficient vector similarity search.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    async def initialize(self):
        """Initialize the backend (create tables/collections, setup indices, etc.)."""
        pass
    
    @abstractmethod
    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        contents: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """
        Insert or update vectors in the index.
        
        Args:
            ids: Unique identifiers for each document
            embeddings: List of embedding vectors
            contents: List of text content
            metadatas: List of metadata dicts
        """
        pass
    
    @abstractmethod
    async def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        distance_metric: str = "cosine",
    ) -> List[ScoredDocument]:
        """
        Query the index for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters
            distance_metric: Distance metric ("cosine", "euclidean", "dot")
            
        Returns:
            List of scored documents
        """
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str]):
        """Delete documents by ID."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Return the number of documents in the index."""
        pass
    
    @abstractmethod
    async def close(self):
        """Close connections and cleanup resources."""
        pass


# Backend registry
_VECTOR_BACKENDS: Dict[str, type] = {}


def register_vector_backend(name: str, backend_class: type):
    """Register a vector backend implementation."""
    _VECTOR_BACKENDS[name] = backend_class


def get_vector_backend(backend_name: str, config: Optional[Dict[str, Any]] = None) -> VectorIndexBackend:
    """
    Get a vector backend instance.
    
    Args:
        backend_name: Backend identifier ("pgvector", "qdrant", "weaviate", etc.)
        config: Backend-specific configuration
        
    Returns:
        VectorIndexBackend instance
    """
    backend_class = _VECTOR_BACKENDS.get(backend_name)
    if backend_class is None:
        available = ", ".join(_VECTOR_BACKENDS.keys())
        raise ValueError(
            f"Unknown vector backend: {backend_name}. Available: {available}"
        )
    
    return backend_class(config)


__all__ = [
    "VectorIndexBackend",
    "ScoredDocument",
    "register_vector_backend",
    "get_vector_backend",
]
