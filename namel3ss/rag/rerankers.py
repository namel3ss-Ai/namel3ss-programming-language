"""Production-grade document reranking for RAG pipelines."""

from __future__ import annotations

import os
import time
import asyncio
import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Protocol, Tuple
from collections import OrderedDict

from .backends.base import ScoredDocument

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from reranking operation."""
    documents: List[ScoredDocument]
    model: str
    query: str
    original_count: int
    reranked_count: int
    time_ms: float
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseReranker(Protocol):
    """Protocol for reranker implementations."""
    
    async def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: Optional[int] = None,
    ) -> List[ScoredDocument]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Query string
            documents: List of candidate documents to rerank
            top_k: Optional limit on number of results (defaults to len(documents))
        
        Returns:
            Reranked list of documents, sorted by relevance (highest first)
        """
        ...
    
    def get_model_name(self) -> str:
        """Return the model name/identifier."""
        ...


class SimpleCache:
    """Simple in-memory LRU cache for reranking results."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
    
    def _make_key(self, query: str, doc_ids: List[str]) -> str:
        """Create cache key from query and document IDs."""
        # Sort doc_ids for consistent keys regardless of order
        sorted_ids = sorted(doc_ids)
        key_str = f"{query}::{','.join(sorted_ids)}"
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, query: str, doc_ids: List[str]) -> Optional[List[ScoredDocument]]:
        """Get cached result if available and not expired."""
        key = self._make_key(query, doc_ids)
        
        if key in self._cache:
            result, timestamp = self._cache[key]
            
            # Check if expired
            if time.time() - timestamp < self.ttl_seconds:
                # Move to end (mark as recently used)
                self._cache.move_to_end(key)
                return result
            else:
                # Expired, remove it
                del self._cache[key]
        
        return None
    
    def put(self, query: str, doc_ids: List[str], result: List[ScoredDocument]) -> None:
        """Store result in cache."""
        key = self._make_key(query, doc_ids)
        
        # If at capacity, remove oldest entry
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        
        self._cache[key] = (result, time.time())
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()


class SentenceTransformerReranker(BaseReranker):
    """
    Reranker using sentence-transformers cross-encoder models.
    
    Uses models like 'cross-encoder/ms-marco-MiniLM-L-6-v2' for semantic reranking.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize sentence-transformers reranker.
        
        Args:
            model_name: HuggingFace model name/path
            config: Configuration dict with optional keys:
                - device: Device to run on ("cpu", "cuda", "mps")
                - batch_size: Batch size for scoring (default: 32)
                - max_length: Max sequence length (default: 512)
                - normalize: Whether to normalize scores (default: True)
                - cache_enabled: Whether to enable result caching (default: True)
                - cache_size: Max cache entries (default: 1000)
                - cache_ttl: Cache TTL in seconds (default: 3600)
        """
        self.model_name = model_name
        self.config = config or {}
        
        self.device = self.config.get("device", "cpu")
        self.batch_size = self.config.get("batch_size", 32)
        self.max_length = self.config.get("max_length", 512)
        self.normalize = self.config.get("normalize", True)
        
        # Cache configuration
        self.cache_enabled = self.config.get("cache_enabled", True)
        cache_size = self.config.get("cache_size", 1000)
        cache_ttl = self.config.get("cache_ttl", 3600)
        
        self._model = None
        self._cache = SimpleCache(max_size=cache_size, ttl_seconds=cache_ttl) if self.cache_enabled else None
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                
                logger.info(f"Loading cross-encoder model: {self.model_name}")
                self._model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device=self.device,
                )
                logger.info(f"Cross-encoder model loaded successfully on {self.device}")
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for SentenceTransformerReranker. "
                    "Install it with: pip install sentence-transformers"
                ) from e
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model {self.model_name}: {e}")
                raise
        
        return self._model
    
    async def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: Optional[int] = None,
    ) -> List[ScoredDocument]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: Query string
            documents: Candidate documents
            top_k: Number of top documents to return (defaults to all)
        
        Returns:
            Reranked documents sorted by relevance
        """
        if not documents:
            return documents
        
        start_time = time.time()
        original_count = len(documents)
        k = top_k if top_k is not None else original_count
        
        # Check cache first
        cache_hit = False
        if self._cache:
            doc_ids = [doc.id for doc in documents]
            cached_result = self._cache.get(query, doc_ids)
            if cached_result is not None:
                cache_hit = True
                logger.debug(
                    f"Cache hit for query (length={len(query)}) with {original_count} documents"
                )
                return cached_result[:k]
        
        # Load model if needed
        model = self._load_model()
        
        # Prepare query-document pairs for batch scoring
        pairs = [(query, doc.content) for doc in documents]
        
        # Score in batches to handle large document sets efficiently
        try:
            # Run synchronous model in thread pool to avoid blocking
            import asyncio
            scores = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            )
            
            # Optionally normalize scores to [0, 1]
            if self.normalize:
                import numpy as np
                scores = np.array(scores)
                if scores.max() != scores.min():
                    scores = (scores - scores.min()) / (scores.max() - scores.min())
                scores = scores.tolist()
            
            # Create new ScoredDocument instances with rerank scores
            reranked_docs = []
            for doc, score in zip(documents, scores):
                # Keep original metadata but add rerank score
                new_metadata = {**doc.metadata, "original_score": doc.score, "rerank_score": float(score)}
                reranked_doc = ScoredDocument(
                    id=doc.id,
                    content=doc.content,
                    score=float(score),  # Replace score with rerank score
                    metadata=new_metadata,
                    embedding=doc.embedding,
                )
                reranked_docs.append(reranked_doc)
            
            # Sort by new scores (descending)
            reranked_docs.sort(key=lambda d: d.score, reverse=True)
            
            # Take top_k
            result = reranked_docs[:k]
            
            # Cache the full reranked result
            if self._cache:
                doc_ids = [doc.id for doc in documents]
                self._cache.put(query, doc_ids, reranked_docs)
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"Reranked {original_count} documents to {len(result)} "
                f"using {self.model_name} in {elapsed_ms:.2f}ms "
                f"(batch_size={self.batch_size})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Reranking failed with {self.model_name}: {e}")
            raise
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model_name


class CohereReranker(BaseReranker):
    """
    Reranker using Cohere's Rerank API.
    
    Requires COHERE_API_KEY environment variable or api_key in config.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Cohere reranker.
        
        Args:
            config: Configuration dict with optional keys:
                - api_key: Cohere API key (or use COHERE_API_KEY env var)
                - model: Model name (default: "rerank-english-v2.0")
                - base_url: API base URL (default: Cohere default)
                - timeout: Request timeout in seconds (default: 30)
                - max_retries: Max retry attempts (default: 3)
                - cache_enabled: Whether to enable caching (default: True)
                - cache_size: Max cache entries (default: 1000)
                - cache_ttl: Cache TTL in seconds (default: 3600)
        """
        self.config = config or {}
        
        self.api_key = self.config.get("api_key") or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Cohere API key not found. Set COHERE_API_KEY environment variable "
                "or pass api_key in config."
            )
        
        self.model = self.config.get("model", "rerank-english-v2.0")
        self.base_url = self.config.get("base_url")
        self.timeout = self.config.get("timeout", 30)
        self.max_retries = self.config.get("max_retries", 3)
        
        # Cache configuration
        self.cache_enabled = self.config.get("cache_enabled", True)
        cache_size = self.config.get("cache_size", 1000)
        cache_ttl = self.config.get("cache_ttl", 3600)
        
        self._client = None
        self._cache = SimpleCache(max_size=cache_size, ttl_seconds=cache_ttl) if self.cache_enabled else None
    
    def _get_client(self):
        """Lazy load Cohere client."""
        if self._client is None:
            try:
                import cohere
                
                client_kwargs = {
                    "api_key": self.api_key,
                    "timeout": self.timeout,
                }
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                
                self._client = cohere.AsyncClient(**client_kwargs)
                logger.info(f"Cohere client initialized with model: {self.model}")
            except ImportError as e:
                raise ImportError(
                    "cohere is required for CohereReranker. "
                    "Install it with: pip install cohere"
                ) from e
        
        return self._client
    
    async def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: Optional[int] = None,
    ) -> List[ScoredDocument]:
        """
        Rerank documents using Cohere Rerank API.
        
        Args:
            query: Query string
            documents: Candidate documents
            top_k: Number of top documents to return (defaults to all)
        
        Returns:
            Reranked documents sorted by relevance
        """
        if not documents:
            return documents
        
        start_time = time.time()
        original_count = len(documents)
        k = top_k if top_k is not None else original_count
        
        # Check cache first
        cache_hit = False
        if self._cache:
            doc_ids = [doc.id for doc in documents]
            cached_result = self._cache.get(query, doc_ids)
            if cached_result is not None:
                cache_hit = True
                logger.debug(
                    f"Cache hit for query (length={len(query)}) with {original_count} documents"
                )
                return cached_result[:k]
        
        client = self._get_client()
        
        # Prepare documents for Cohere API
        doc_texts = [doc.content for doc in documents]
        
        try:
            # Call Cohere rerank API
            response = await client.rerank(
                query=query,
                documents=doc_texts,
                model=self.model,
                top_n=k,  # Cohere returns top_n results
                return_documents=False,  # We already have the documents
            )
            
            # Map results back to original documents
            reranked_docs = []
            for result in response.results:
                original_doc = documents[result.index]
                
                # Preserve metadata and add rerank information
                new_metadata = {
                    **original_doc.metadata,
                    "original_score": original_doc.score,
                    "rerank_score": result.relevance_score,
                    "rerank_index": result.index,
                }
                
                reranked_doc = ScoredDocument(
                    id=original_doc.id,
                    content=original_doc.content,
                    score=result.relevance_score,
                    metadata=new_metadata,
                    embedding=original_doc.embedding,
                )
                reranked_docs.append(reranked_doc)
            
            # Cache the result
            if self._cache:
                doc_ids = [doc.id for doc in documents]
                self._cache.put(query, doc_ids, reranked_docs)
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"Reranked {original_count} documents to {len(reranked_docs)} "
                f"using Cohere {self.model} in {elapsed_ms:.2f}ms"
            )
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            raise
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return f"cohere:{self.model}"


class HTTPReranker(BaseReranker):
    """
    Generic HTTP-based reranker for custom APIs.
    
    Sends POST requests to a reranking endpoint with query and documents,
    expects JSON response with reranked results.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HTTP reranker.
        
        Args:
            config: Configuration dict with required keys:
                - endpoint: API endpoint URL
                - headers: Optional HTTP headers dict
                - timeout: Request timeout in seconds (default: 30)
                - max_retries: Max retry attempts (default: 3)
                - request_format: Format of request payload (default: "standard")
                - response_format: Format of response (default: "standard")
                - cache_enabled: Whether to enable caching (default: True)
                - cache_size: Max cache entries (default: 1000)
                - cache_ttl: Cache TTL in seconds (default: 3600)
        """
        self.config = config
        
        if "endpoint" not in config:
            raise ValueError("HTTPReranker requires 'endpoint' in config")
        
        self.endpoint = config["endpoint"]
        self.headers = config.get("headers", {})
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.request_format = config.get("request_format", "standard")
        self.response_format = config.get("response_format", "standard")
        
        # Cache configuration
        self.cache_enabled = config.get("cache_enabled", True)
        cache_size = config.get("cache_size", 1000)
        cache_ttl = config.get("cache_ttl", 3600)
        
        self._cache = SimpleCache(max_size=cache_size, ttl_seconds=cache_ttl) if self.cache_enabled else None
    
    async def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: Optional[int] = None,
    ) -> List[ScoredDocument]:
        """
        Rerank documents using HTTP API.
        
        Args:
            query: Query string
            documents: Candidate documents
            top_k: Number of top documents to return (defaults to all)
        
        Returns:
            Reranked documents sorted by relevance
        """
        if not documents:
            return documents
        
        start_time = time.time()
        original_count = len(documents)
        k = top_k if top_k is not None else original_count
        
        # Check cache first
        cache_hit = False
        if self._cache:
            doc_ids = [doc.id for doc in documents]
            cached_result = self._cache.get(query, doc_ids)
            if cached_result is not None:
                cache_hit = True
                logger.debug(
                    f"Cache hit for query (length={len(query)}) with {original_count} documents"
                )
                return cached_result[:k]
        
        try:
            import httpx
            
            # Prepare request payload
            payload = self._format_request(query, documents, k)
            
            # Make HTTP request with retries
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for attempt in range(self.max_retries):
                    try:
                        response = await client.post(
                            self.endpoint,
                            json=payload,
                            headers=self.headers,
                        )
                        response.raise_for_status()
                        break
                    except (httpx.HTTPStatusError, httpx.RequestError) as e:
                        if attempt == self.max_retries - 1:
                            raise
                        logger.warning(f"HTTP reranker attempt {attempt + 1} failed: {e}")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                # Parse response
                result_data = response.json()
                reranked_docs = self._parse_response(result_data, documents)
                
                # Take top_k
                result = reranked_docs[:k]
                
                # Cache the result
                if self._cache:
                    doc_ids = [doc.id for doc in documents]
                    self._cache.put(query, doc_ids, reranked_docs)
                
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"Reranked {original_count} documents to {len(result)} "
                    f"via HTTP endpoint in {elapsed_ms:.2f}ms"
                )
                
                return result
                
        except ImportError as e:
            raise ImportError(
                "httpx is required for HTTPReranker. "
                "Install it with: pip install httpx"
            ) from e
        except Exception as e:
            logger.error(f"HTTP reranking failed: {e}")
            raise
    
    def _format_request(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: int,
    ) -> Dict[str, Any]:
        """Format request payload based on configured format."""
        if self.request_format == "standard":
            return {
                "query": query,
                "documents": [
                    {"id": doc.id, "content": doc.content, "metadata": doc.metadata}
                    for doc in documents
                ],
                "top_k": top_k,
            }
        else:
            # Allow custom formatting via config
            raise ValueError(f"Unsupported request_format: {self.request_format}")
    
    def _parse_response(
        self,
        response_data: Dict[str, Any],
        original_documents: List[ScoredDocument],
    ) -> List[ScoredDocument]:
        """Parse response data based on configured format."""
        if self.response_format == "standard":
            # Expected format: {"results": [{"id": "...", "score": 0.95}, ...]}
            results = response_data.get("results", [])
            
            # Create document ID to original doc mapping
            doc_map = {doc.id: doc for doc in original_documents}
            
            reranked_docs = []
            for result in results:
                doc_id = result["id"]
                score = result["score"]
                
                if doc_id in doc_map:
                    original_doc = doc_map[doc_id]
                    new_metadata = {
                        **original_doc.metadata,
                        "original_score": original_doc.score,
                        "rerank_score": score,
                    }
                    
                    reranked_doc = ScoredDocument(
                        id=original_doc.id,
                        content=original_doc.content,
                        score=score,
                        metadata=new_metadata,
                        embedding=original_doc.embedding,
                    )
                    reranked_docs.append(reranked_doc)
            
            return reranked_docs
        else:
            raise ValueError(f"Unsupported response_format: {self.response_format}")
    
    def get_model_name(self) -> str:
        """Return the endpoint as model identifier."""
        return f"http:{self.endpoint}"


# Registry for reranker types
_RERANKER_REGISTRY: Dict[str, type] = {
    "sentence_transformers": SentenceTransformerReranker,
    "cross_encoder": SentenceTransformerReranker,  # Alias
    "cohere": CohereReranker,
    "http": HTTPReranker,
}


def register_reranker(name: str, reranker_class: type) -> None:
    """
    Register a custom reranker implementation.
    
    Args:
        name: Name to register under
        reranker_class: Reranker class implementing BaseReranker protocol
    """
    _RERANKER_REGISTRY[name] = reranker_class
    logger.info(f"Registered reranker: {name}")


def get_reranker(name: str, config: Optional[Dict[str, Any]] = None) -> BaseReranker:
    """
    Factory function to get reranker instance.
    
    Args:
        name: Reranker type name. Supported values:
            - "sentence_transformers" or "cross_encoder": Local cross-encoder models
            - "cohere": Cohere Rerank API
            - "http": Generic HTTP reranking endpoint
            - Or any custom registered name
        config: Configuration dict for the reranker. Contents vary by type:
            
            For sentence_transformers:
                - model_name: HuggingFace model (default: "cross-encoder/ms-marco-MiniLM-L-6-v2")
                - device: "cpu", "cuda", or "mps"
                - batch_size: Batch size for scoring
                - max_length: Max sequence length
                - normalize: Whether to normalize scores
                - cache_enabled, cache_size, cache_ttl: Caching options
            
            For cohere:
                - api_key: Cohere API key (or use COHERE_API_KEY env var)
                - model: Model name (default: "rerank-english-v2.0")
                - timeout: Request timeout
                - max_retries: Max retry attempts
                - cache_enabled, cache_size, cache_ttl: Caching options
            
            For http:
                - endpoint: API endpoint URL (required)
                - headers: HTTP headers dict
                - timeout: Request timeout
                - max_retries: Max retry attempts
                - request_format, response_format: Payload formats
                - cache_enabled, cache_size, cache_ttl: Caching options
    
    Returns:
        Reranker instance
    
    Raises:
        ValueError: If reranker name is not registered
    """
    config = config or {}
    
    if name not in _RERANKER_REGISTRY:
        available = ", ".join(_RERANKER_REGISTRY.keys())
        raise ValueError(
            f"Unknown reranker: '{name}'. Available rerankers: {available}. "
            f"Register custom rerankers with register_reranker()."
        )
    
    reranker_class = _RERANKER_REGISTRY[name]
    
    # Instantiate based on reranker type
    if name in ("sentence_transformers", "cross_encoder"):
        model_name = config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        return reranker_class(model_name=model_name, config=config)
    elif name == "cohere":
        return reranker_class(config=config)
    elif name == "http":
        return reranker_class(config=config)
    else:
        # Custom reranker - try to instantiate with config
        return reranker_class(config=config)


__all__ = [
    "BaseReranker",
    "RerankResult",
    "SentenceTransformerReranker",
    "CohereReranker",
    "HTTPReranker",
    "SimpleCache",
    "get_reranker",
    "register_reranker",
]
