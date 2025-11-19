"""RAG pipeline runtime execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .embeddings import get_embedding_provider
from .backends import get_vector_backend, ScoredDocument

logger = logging.getLogger(__name__)


@dataclass
class RagResult:
    """
    Result from RAG pipeline query execution.
    
    Attributes:
        documents: Retrieved documents with scores
        query: Original query string
        query_embedding: Query embedding vector (for debugging)
        metadata: Additional metadata (timing, token counts, etc.)
    """
    documents: List[ScoredDocument]
    query: str
    query_embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RagPipelineRuntime:
    """
    Runtime executor for RAG pipelines.
    
    Orchestrates the full RAG flow:
    1. Embed query using query_encoder
    2. Search vector index
    3. Optionally rerank results
    4. Return scored documents
    """
    
    def __init__(
        self,
        name: str,
        query_encoder: str,
        index_backend: Any,  # VectorIndexBackend instance
        top_k: int = 5,
        reranker: Optional[str] = None,
        distance_metric: str = "cosine",
        filters: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RAG pipeline runtime.
        
        Args:
            name: Pipeline name
            query_encoder: Embedding model name for query encoding
            index_backend: Vector backend instance
            top_k: Number of documents to retrieve
            reranker: Optional reranker model name
            distance_metric: Distance metric ("cosine", "euclidean", "dot")
            filters: Metadata filters for retrieval
            config: Additional configuration
        """
        self.name = name
        self.query_encoder = query_encoder
        self.index_backend = index_backend
        self.top_k = top_k
        self.reranker = reranker
        self.distance_metric = distance_metric
        self.filters = filters or {}
        self.config = config or {}
        
        # Initialize embedding provider
        self.embedding_provider = get_embedding_provider(query_encoder)
    
    async def execute_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
    ) -> RagResult:
        """
        Execute RAG query.
        
        Args:
            query: Query string
            top_k: Override default top_k
            filters: Override or extend default filters
            include_embeddings: Whether to include embeddings in result
        
        Returns:
            RagResult with retrieved documents and metadata
        """
        import time
        start_time = time.time()
        
        # Merge filters
        merged_filters = {**self.filters}
        if filters:
            merged_filters.update(filters)
        
        # Use provided top_k or default
        k = top_k if top_k is not None else self.top_k
        
        # Step 1: Embed query
        embed_start = time.time()
        try:
            embed_result = await self.embedding_provider.embed_query(query)
            query_embedding = embed_result.embeddings[0]
            embed_time = time.time() - embed_start
            
            logger.debug(
                f"RAG pipeline '{self.name}': Query embedded in {embed_time:.3f}s "
                f"(tokens: {embed_result.tokens_used})"
            )
        except Exception as e:
            logger.error(f"RAG pipeline '{self.name}': Query embedding failed: {e}")
            raise
        
        # Step 2: Search vector index
        search_start = time.time()
        try:
            documents = await self.index_backend.query(
                query_vector=query_embedding,
                top_k=k,
                filters=merged_filters if merged_filters else None,
                distance_metric=self.distance_metric,
            )
            search_time = time.time() - search_start
            
            logger.debug(
                f"RAG pipeline '{self.name}': Retrieved {len(documents)} documents "
                f"in {search_time:.3f}s"
            )
        except Exception as e:
            logger.error(f"RAG pipeline '{self.name}': Vector search failed: {e}")
            raise
        
        # Step 3: Optional reranking
        if self.reranker:
            rerank_start = time.time()
            try:
                documents = await self._rerank_documents(query, documents)
                rerank_time = time.time() - rerank_start
                logger.debug(
                    f"RAG pipeline '{self.name}': Reranked {len(documents)} documents "
                    f"in {rerank_time:.3f}s"
                )
            except Exception as e:
                logger.warning(
                    f"RAG pipeline '{self.name}': Reranking failed, "
                    f"using original ranking: {e}"
                )
        
        total_time = time.time() - start_time
        
        # Build result
        result = RagResult(
            documents=documents,
            query=query,
            query_embedding=query_embedding if include_embeddings else None,
            metadata={
                "pipeline_name": self.name,
                "top_k": k,
                "retrieved_count": len(documents),
                "query_encoder": self.query_encoder,
                "distance_metric": self.distance_metric,
                "filters": merged_filters,
                "timings": {
                    "embedding_ms": round(embed_time * 1000, 2),
                    "search_ms": round(search_time * 1000, 2),
                    "total_ms": round(total_time * 1000, 2),
                },
            },
        )
        
        return result
    
    async def _rerank_documents(
        self,
        query: str,
        documents: List[ScoredDocument],
    ) -> List[ScoredDocument]:
        """
        Rerank documents using reranker model.
        
        This is a placeholder for reranker integration.
        Real implementations could use:
        - Cross-encoder models (sentence-transformers)
        - Cohere Rerank API
        - Custom reranking logic
        
        For now, returns documents unchanged.
        """
        # TODO: Implement reranking when reranker models are added
        logger.warning(
            f"Reranker '{self.reranker}' specified but reranking not yet implemented"
        )
        return documents


@dataclass
class IndexBuildResult:
    """
    Result from index building operation.
    
    Attributes:
        index_name: Name of the index built
        documents_processed: Number of documents processed
        chunks_created: Number of chunks created
        chunks_indexed: Number of chunks successfully indexed
        tokens_used: Total embedding tokens used
        errors: List of error messages encountered
        metadata: Additional metadata (timing, etc.)
    """
    index_name: str
    documents_processed: int
    chunks_created: int
    chunks_indexed: int
    tokens_used: int
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


async def build_index(
    index_name: str,
    documents: List[Dict[str, Any]],
    embedding_model: str,
    vector_backend: Any,
    chunk_size: int = 512,
    overlap: int = 64,
    batch_size: int = 32,
) -> IndexBuildResult:
    """
    Build vector index from documents.
    
    Args:
        index_name: Name of the index
        documents: List of documents (each with 'id', 'content', 'metadata')
        embedding_model: Embedding model name
        vector_backend: Vector backend instance
        chunk_size: Chunk size in characters
        overlap: Overlap between chunks
        batch_size: Batch size for embedding
    
    Returns:
        IndexBuildResult with statistics
    """
    import time
    from .chunking import chunk_text
    
    start_time = time.time()
    
    # Import embedding provider
    embedding_provider = get_embedding_provider(embedding_model)
    
    chunks_created = 0
    chunks_indexed = 0
    tokens_used = 0
    errors = []
    
    # Process documents in batches
    all_chunk_ids = []
    all_chunk_embeddings = []
    all_chunk_contents = []
    all_chunk_metadatas = []
    
    for doc in documents:
        try:
            doc_id = doc["id"]
            content = doc["content"]
            doc_metadata = doc.get("metadata", {})
            
            # Chunk document
            chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
            chunks_created += len(chunks)
            
            for chunk in chunks:
                chunk_id = f"{doc_id}#{chunk.chunk_id}"
                chunk_metadata = {
                    **doc_metadata,
                    "doc_id": doc_id,
                    "chunk_id": chunk.chunk_id,
                    "start_pos": chunk.start_pos,
                    "end_pos": chunk.end_pos,
                }
                
                all_chunk_ids.append(chunk_id)
                all_chunk_contents.append(chunk.content)
                all_chunk_metadatas.append(chunk_metadata)
                
                # Batch embedding calls
                if len(all_chunk_contents) >= batch_size:
                    await _embed_and_upsert_batch(
                        embedding_provider,
                        vector_backend,
                        all_chunk_ids,
                        all_chunk_contents,
                        all_chunk_metadatas,
                    )
                    chunks_indexed += len(all_chunk_ids)
                    all_chunk_ids = []
                    all_chunk_contents = []
                    all_chunk_metadatas = []
        
        except Exception as e:
            error_msg = f"Error processing document {doc.get('id', 'unknown')}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    # Process remaining chunks
    if all_chunk_contents:
        try:
            embed_result = await _embed_and_upsert_batch(
                embedding_provider,
                vector_backend,
                all_chunk_ids,
                all_chunk_contents,
                all_chunk_metadatas,
            )
            chunks_indexed += len(all_chunk_ids)
            tokens_used += embed_result.tokens_used if hasattr(embed_result, 'tokens_used') else 0
        except Exception as e:
            error_msg = f"Error processing final batch: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    total_time = time.time() - start_time
    
    return IndexBuildResult(
        index_name=index_name,
        documents_processed=len(documents),
        chunks_created=chunks_created,
        chunks_indexed=chunks_indexed,
        tokens_used=tokens_used,
        errors=errors,
        metadata={
            "build_time_seconds": round(total_time, 2),
            "chunks_per_second": round(chunks_indexed / total_time, 2) if total_time > 0 else 0,
        },
    )


async def _embed_and_upsert_batch(
    embedding_provider,
    vector_backend,
    chunk_ids: List[str],
    chunk_contents: List[str],
    chunk_metadatas: List[Dict[str, Any]],
):
    """Helper to embed and upsert a batch of chunks."""
    # Embed batch
    embed_result = await embedding_provider.embed_texts(chunk_contents)
    
    # Upsert to backend
    await vector_backend.upsert(
        ids=chunk_ids,
        embeddings=embed_result.embeddings,
        contents=chunk_contents,
        metadatas=chunk_metadatas,
    )
    
    return embed_result


__all__ = [
    "RagResult",
    "RagPipelineRuntime",
    "IndexBuildResult",
    "build_index",
]
