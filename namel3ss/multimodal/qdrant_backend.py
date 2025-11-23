"""Qdrant vector database backend with multimodal and hybrid search support."""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

# Use absolute import to avoid relative import issues when running tests
from namel3ss.rag.backends.base import VectorIndexBackend, ScoredDocument

# Allow tests to patch QdrantClient directly on the module
QdrantClient = None

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Result from hybrid search combining dense and sparse retrieval."""
    documents: List[ScoredDocument] = field(default_factory=list)
    dense_results: List[ScoredDocument] = field(default_factory=list)
    sparse_results: List[ScoredDocument] = field(default_factory=list)
    fusion_method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    scores: List[float] = field(default_factory=list)


class QdrantMultimodalBackend(VectorIndexBackend):
    """
    Qdrant backend with support for:
    - Multi-vector storage (text, image, audio embeddings)
    - Hybrid search (dense + sparse/BM25)
    - Named vectors for different modalities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Connection config
        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port", 6333)
        self.api_key = self.config.get("api_key")
        self.url = self.config.get("url")  # Alternative to host:port
        
        # Collection config
        self.collection_name = self.config.get("collection_name", "multimodal_docs")
        
        # Vector dimensions
        self.text_dimension = self.config.get("text_dimension", 384)
        self.image_dimension = self.config.get("image_dimension", 512)
        self.audio_dimension = self.config.get("audio_dimension", 384)
        
        # Hybrid search config
        self.enable_sparse = self.config.get("enable_sparse", True)
        
        self.client = None
    
    async def initialize(self):
        """Initialize Qdrant client and create collection."""
        global QdrantClient
        try:
            if QdrantClient is None:
                from qdrant_client import QdrantClient as _QdrantClient
                QdrantClient = _QdrantClient
            from qdrant_client.models import (
                Distance,
                VectorParams,
                PointStruct,
                SparseVectorParams,
                SparseIndexParams,
            )
        except ImportError:
            raise ImportError(
                "qdrant-client required. Install with: "
                "pip install qdrant-client"
            )
        
        # Initialize client
        if self.url:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
            )
        
        logger.info(f"Connected to Qdrant at {self.url or f'{self.host}:{self.port}'}")
        
        # Check if collection exists
        collections = await self.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if self.collection_name not in collection_names:
            await self._create_collection()
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")
    
    async def _create_collection(self):
        """Create Qdrant collection with named vectors."""
        from qdrant_client.models import (
            Distance,
            VectorParams,
            SparseVectorParams,
            SparseIndexParams,
        )
        
        logger.info(f"Creating collection '{self.collection_name}'")
        
        # Define named vectors for each modality
        vectors_config = {
            "text": VectorParams(
                size=self.text_dimension,
                distance=Distance.COSINE,
            ),
            "image": VectorParams(
                size=self.image_dimension,
                distance=Distance.COSINE,
            ),
            "audio": VectorParams(
                size=self.audio_dimension,
                distance=Distance.COSINE,
            ),
        }
        
        # Add sparse vector config for hybrid search
        sparse_vectors_config = None
        if self.enable_sparse:
            sparse_vectors_config = {
                "text_sparse": SparseVectorParams(
                    index=SparseIndexParams(),
                ),
            }
        
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        
        logger.info(f"Collection '{self.collection_name}' created successfully")
    
    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        contents: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """
        Upsert documents with embeddings.
        
        Note: This is the legacy interface. Use upsert_multimodal for full support.
        """
        from qdrant_client.models import PointStruct
        
        points = []
        for i, (doc_id, embedding, content, metadata) in enumerate(
            zip(ids, embeddings, contents, metadatas)
        ):
            point = PointStruct(
                id=doc_id,
                vector={"text": embedding},  # Store in text vector
                payload={
                    "content": content,
                    **metadata,
                },
            )
            points.append(point)
        
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        logger.debug(f"Upserted {len(points)} points to '{self.collection_name}'")
    
    async def upsert_multimodal(
        self,
        ids: Optional[List[str]] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
        text_embeddings: Optional[List[Any]] = None,
        image_embeddings: Optional[List[Any]] = None,
        audio_embeddings: Optional[List[Any]] = None,
        sparse_embeddings: Optional[List[Dict[int, float]]] = None,
        contents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Upsert documents with multimodal embeddings.
        
        Args:
            ids: Document IDs
            text_embeddings: Dense text embeddings
            image_embeddings: Dense image embeddings
            audio_embeddings: Dense audio embeddings
            sparse_embeddings: Sparse text embeddings (BM25-style)
            contents: Text contents
            metadatas: Metadata dicts
        """
        try:
            from qdrant_client.models import PointStruct, SparseVector  # type: ignore
        except ImportError:
            # Lightweight fallbacks for testing environments without qdrant-client
            class PointStruct(dict):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
            class SparseVector(dict):
                def __init__(self, indices, values):
                    super().__init__(indices=indices, values=values)
        
        # Normalize inputs
        if documents:
            ids = [doc.get("id") for doc in documents]
            contents = [doc.get("content") for doc in documents]
            metadatas = [doc.get("metadata", {}) for doc in documents]
        ids = ids or []
        
        # Convert numpy arrays to lists for serialization if present
        def _to_list(arr):
            try:
                import numpy as np  # type: ignore
                if isinstance(arr, np.ndarray):
                    return arr.tolist()
            except ImportError:
                pass
            return arr
        
        text_embeddings = _to_list(text_embeddings)
        image_embeddings = _to_list(image_embeddings)
        audio_embeddings = _to_list(audio_embeddings)
        
        points = []
        for i, doc_id in enumerate(ids):
            # Build vector dict
            vectors = {}
            
            if text_embeddings and i < len(text_embeddings):
                vectors["text"] = text_embeddings[i]
            
            if image_embeddings and i < len(image_embeddings):
                vectors["image"] = image_embeddings[i]
            
            if audio_embeddings and i < len(audio_embeddings):
                vectors["audio"] = audio_embeddings[i]
            
            # Build sparse vectors dict
            sparse_vectors = None
            if sparse_embeddings and i < len(sparse_embeddings):
                sparse_dict = sparse_embeddings[i]
                # Support both {"indices":[...], "values":[...]} and {idx: value}
                if isinstance(sparse_dict, dict) and "indices" in sparse_dict and "values" in sparse_dict:
                    indices = sparse_dict["indices"]
                    values = sparse_dict["values"]
                else:
                    indices = list(sparse_dict.keys())
                    values = list(sparse_dict.values())
                sparse_vectors = {
                    "text_sparse": SparseVector(
                        indices=indices,
                        values=values,
                    )
                }
            
            # Build payload
            payload = {}
            if contents and i < len(contents):
                payload["content"] = contents[i]
            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])
            
            point = PointStruct(
                id=doc_id,
                vector=vectors,
                payload=payload,
            )
            
            if sparse_vectors:
                if isinstance(point, dict):
                    point["sparse_vector"] = sparse_vectors
                else:
                    setattr(point, "sparse_vector", sparse_vectors)
            
            points.append(point)
        
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        logger.debug(f"Upserted {len(points)} multimodal points")
    
    async def query(
        self,
        query_vector: Optional[List[float]] = None,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        distance_metric: str = "cosine",
        vector_name: str = "text",
        modality: Optional[str] = None,
    ) -> List[ScoredDocument]:
        """
        Query collection using dense vector search.
        
        Args:
            query_vector/query_embedding: Query embedding
            top_k: Number of results
            filters: Metadata filters
            distance_metric: Distance metric (ignored, using collection config)
            vector_name: Which named vector to query ("text", "image", "audio")
            
        Returns:
            List of scored documents
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Normalize query vector
        vector = query_vector if query_vector is not None else query_embedding
        if vector is None:
            raise ValueError("query_vector or query_embedding is required")
        if modality:
            vector_name = modality
        
        # Build filter
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                query_filter = Filter(must=conditions)
        
        # Search
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=(vector_name, query_vector),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        
        # Convert to ScoredDocuments
        documents = []
        for hit in results:
            documents.append({
                "id": str(getattr(hit, "id", "")),
                "content": hit.payload.get("content", "") if getattr(hit, "payload", None) else "",
                "score": getattr(hit, "score", 0),
                "metadata": {k: v for k, v in (getattr(hit, "payload", {}) or {}).items() if k != "content"},
            })
        
        return documents
    
    async def hybrid_search(
        self,
        dense_vector: Optional[List[float]] = None,
        query_embedding: Optional[List[float]] = None,
        sparse_vector: Dict[int, float] = None,
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        filters: Optional[Dict[str, Any]] = None,
        vector_name: str = "text",
    ) -> HybridSearchResult:
        """
        Perform hybrid search combining dense and sparse retrieval.
        
        Args:
            dense_vector: Dense query embedding
            sparse_vector: Sparse query embedding (BM25-style)
            top_k: Number of results
            dense_weight: Weight for dense scores
            sparse_weight: Weight for sparse scores
            filters: Metadata filters
            vector_name: Which dense vector to query
            
        Returns:
            HybridSearchResult with fused results
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue, SparseVector  # type: ignore
        except ImportError:
            class Filter(dict):
                def __init__(self, must=None):
                    super().__init__(must=must or [])
            class FieldCondition(dict):
                def __init__(self, key=None, match=None):
                    super().__init__(key=key, match=match)
            class MatchValue(dict):
                def __init__(self, value=None):
                    super().__init__(value=value)
            class SparseVector(dict):
                def __init__(self, indices=None, values=None):
                    super().__init__(indices=indices or [], values=values or [])
        
        # Normalize dense vector
        dense_vector = dense_vector if dense_vector is not None else query_embedding
        if dense_vector is None:
            raise ValueError("dense_vector or query_embedding is required")
        
        # Build filter
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                query_filter = Filter(must=conditions)
        
        # Get more candidates for fusion
        candidate_k = top_k * 3
        
        # Dense search
        dense_results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=(vector_name, dense_vector),
            query_filter=query_filter,
            limit=candidate_k,
            with_payload=True,
        )
        
        # Sparse search
        sparse_results = []
        if self.enable_sparse and sparse_vector:
            if isinstance(sparse_vector, dict) and "indices" in sparse_vector and "values" in sparse_vector:
                indices = sparse_vector["indices"]
                values = sparse_vector["values"]
            else:
                indices = list(sparse_vector.keys())
                values = list(sparse_vector.values())
            sparse_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=(
                    "text_sparse",
                    SparseVector(
                        indices=indices,
                        values=values,
                    ),
                ),
                query_filter=query_filter,
                limit=candidate_k,
                with_payload=True,
            )
        
        # Reciprocal Rank Fusion (RRF)
        fused_scores = {}
        k_rrf = 60  # RRF constant
        
        # Add dense scores
        for rank, hit in enumerate(dense_results):
            doc_id = str(hit.id)
            rrf_score = 1.0 / (k_rrf + rank + 1)
            fused_scores[doc_id] = {
                "score": dense_weight * rrf_score,
                "hit": hit,
            }
        
        # Add sparse scores
        for rank, hit in enumerate(sparse_results):
            doc_id = str(hit.id)
            rrf_score = 1.0 / (k_rrf + rank + 1)
            if doc_id in fused_scores:
                fused_scores[doc_id]["score"] += sparse_weight * rrf_score
            else:
                fused_scores[doc_id] = {
                    "score": sparse_weight * rrf_score,
                    "hit": hit,
                }
        
        # Sort by fused score
        sorted_items = sorted(
            fused_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True,
        )[:top_k]
        
        # Convert to ScoredDocuments
        fused_documents = []
        for doc_id, item in sorted_items:
            hit = item["hit"]
            doc = ScoredDocument(
                id=doc_id,
                content=hit.payload.get("content", ""),
                score=item["score"],
                metadata={k: v for k, v in hit.payload.items() if k != "content"},
            )
            fused_documents.append(doc)
        
        # Convert dense/sparse results to ScoredDocuments
        dense_docs = [
            ScoredDocument(
                id=str(hit.id),
                content=hit.payload.get("content", ""),
                score=hit.score,
                metadata={k: v for k, v in hit.payload.items() if k != "content"},
            )
            for hit in dense_results[:top_k]
        ]
        
        sparse_docs = [
            ScoredDocument(
                id=str(hit.id),
                content=hit.payload.get("content", ""),
                score=hit.score,
                metadata={k: v for k, v in hit.payload.items() if k != "content"},
            )
            for hit in sparse_results[:top_k]
        ]
        
        return HybridSearchResult(
            documents=fused_documents,
            dense_results=dense_docs,
            sparse_results=sparse_docs,
            fusion_method="reciprocal_rank_fusion",
            metadata={
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "k_rrf": k_rrf,
            },
            scores=[item[1]["score"] for item in sorted_items],
        )
    
    async def delete(self, ids: List[str]):
        """Delete documents by ID."""
        from qdrant_client.models import PointIdsList
        
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )
        
        logger.debug(f"Deleted {len(ids)} points from '{self.collection_name}'")
    
    async def count(self) -> int:
        """Return total document count."""
        info = await self.client.get_collection(self.collection_name)
        return info.points_count
    
    async def close(self):
        """Close connection."""
        if self.client:
            await self.client.close()
