"""RAG (Retrieval-Augmented Generation) runtime components."""

from .embeddings import EmbeddingProvider, get_embedding_provider
from .backends import VectorIndexBackend, get_vector_backend, ScoredDocument
from .chunking import chunk_text, TextChunk
from .pipeline import RagPipelineRuntime, RagResult, IndexBuildResult, build_index
from .rerankers import (
    BaseReranker,
    get_reranker,
    register_reranker,
    SentenceTransformerReranker,
    CohereReranker,
    HTTPReranker,
)

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
    "BaseReranker",
    "get_reranker",
    "register_reranker",
    "SentenceTransformerReranker",
    "CohereReranker",
    "HTTPReranker",
]
