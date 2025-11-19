"""Embedding provider abstractions and implementations."""

from __future__ import annotations

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding operation."""
    embeddings: List[List[float]]
    model: str
    tokens_used: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, model: str, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = config or {}
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Embed a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            EmbeddingResult containing embeddings and metadata
        """
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.
        
        Args:
            query: Query text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        pass
    
    def get_dimension(self) -> int:
        """Return the embedding dimension for this model."""
        # Default dimensions for common models
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
        }
        return dimensions.get(self.model, 1536)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using official API."""
    
    def __init__(self, model: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model, config or {})
        config = config or {}
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
    
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Embed texts using OpenAI API."""
        try:
            # Lazy import to avoid dependency issues
            import openai
            
            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            
            response = await client.embeddings.create(
                model=self.model,
                input=texts,
            )
            
            embeddings = [item.embedding for item in response.data]
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
            
            return EmbeddingResult(
                embeddings=embeddings,
                model=self.model,
                tokens_used=tokens_used,
                metadata={"provider": "openai"}
            )
            
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        result = await self.embed_texts([query])
        return result.embeddings[0]


class SentenceTransformerProvider(EmbeddingProvider):
    """Local embedding using sentence-transformers."""
    
    def __init__(self, model: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model, config or {})
        config = config or {}
        self.device = config.get("device", "cpu")
        self._model = None
    
    def _get_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model, device=self.device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
        return self._model
    
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Embed texts using local sentence transformer."""
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        return EmbeddingResult(
            embeddings=embeddings_list,
            model=self.model,
            tokens_used=0,  # Local models don't track tokens
            metadata={"provider": "sentence-transformers", "device": self.device}
        )
    
    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        result = await self.embed_texts([query])
        return result.embeddings[0]


# Registry of available providers
_EMBEDDING_PROVIDERS: Dict[str, type] = {
    "openai": OpenAIEmbeddingProvider,
    "sentence-transformers": SentenceTransformerProvider,
    "local": SentenceTransformerProvider,
}


def register_embedding_provider(name: str, provider_class: type):
    """Register a custom embedding provider."""
    _EMBEDDING_PROVIDERS[name] = provider_class


def get_embedding_provider(
    model: str,
    provider: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> EmbeddingProvider:
    """
    Get an embedding provider instance.
    
    Args:
        model: Model identifier (e.g., "text-embedding-3-small")
        provider: Provider name ("openai", "sentence-transformers", "local")
                 If None, infer from model name
        config: Optional configuration dict
        
    Returns:
        Embedding provider instance
    """
    config = config or {}
    
    # Infer provider from model name if not specified
    if provider is None:
        if "text-embedding" in model or model.startswith("gpt"):
            provider = "openai"
        else:
            provider = "sentence-transformers"
    
    provider_class = _EMBEDDING_PROVIDERS.get(provider)
    if provider_class is None:
        available = ", ".join(_EMBEDDING_PROVIDERS.keys())
        raise ValueError(
            f"Unknown embedding provider: {provider}. Available: {available}"
        )
    
    return provider_class(model, config)


__all__ = [
    "EmbeddingProvider",
    "EmbeddingResult",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerProvider",
    "get_embedding_provider",
    "register_embedding_provider",
]
