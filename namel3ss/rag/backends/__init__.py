"""Vector database backend abstractions."""

from .base import VectorIndexBackend, ScoredDocument, get_vector_backend
from .pgvector import PgVectorBackend

__all__ = [
    "VectorIndexBackend",
    "ScoredDocument",
    "PgVectorBackend",
    "get_vector_backend",
]
