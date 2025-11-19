"""RAG (Retrieval-Augmented Generation) runtime components."""

from .embeddings import EmbeddingProvider, get_embedding_provider
from .backends import VectorIndexBackend, get_vector_backend, ScoredDocument
from .chunking import chunk_text, TextChunk
from .pipeline import RagPipelineRuntime, RagResult, IndexBuildResult, build_index

__all__ = [
    "EmbeddingProvider",
    "get_embedding_provider",
    "VectorIndexBackend",
    "get_vector_backend",
    "chunk_text",
    "TextChunk",
    "RagPipelineRuntime",
    "RagResult",
    "IndexBuildResult",
    "build_index",
    "ScoredDocument",
]
