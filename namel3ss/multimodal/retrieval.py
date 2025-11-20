"""Hybrid retrieval combining dense, sparse, and reranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from hybrid search."""
    documents: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.documents[idx]


class BM25Encoder:
    """BM25 sparse encoder for keyword-based retrieval."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 encoder.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.vocab = {}
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.corpus_size = 0
        self._fitted = False
    
    def fit(self, corpus: List[str]):
        """
        Fit BM25 on corpus.
        
        Args:
            corpus: List of documents
        """
        import math
        from collections import Counter
        
        self.corpus_size = len(corpus)
        self.doc_len = []
        
        # Tokenize and build vocabulary
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        
        # Calculate document frequencies
        df = Counter()
        for tokens in tokenized_corpus:
            unique_tokens = set(tokens)
            df.update(unique_tokens)
            self.doc_len.append(len(tokens))
        
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        
        # Build vocabulary and IDF
        for idx, (token, freq) in enumerate(df.items()):
            self.vocab[token] = idx
            # IDF calculation
            self.idf[idx] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )
        
        self._fitted = True
        logger.info(f"BM25 fitted on {self.corpus_size} documents, vocab size: {len(self.vocab)}")
    
    def encode_query(self, query: str) -> Dict[int, float]:
        """
        Encode query to sparse vector.
        
        Args:
            query: Query text
            
        Returns:
            Sparse vector as dict {term_id: weight}
        """
        if not self._fitted:
            raise ValueError("BM25 not fitted. Call fit() first.")
        
        tokens = self._tokenize(query)
        
        # Count term frequencies
        from collections import Counter
        tf = Counter(tokens)
        
        # Build sparse vector
        sparse_vector = {}
        for token, count in tf.items():
            if token in self.vocab:
                term_id = self.vocab[token]
                # BM25 query term weight (simplified)
                weight = self.idf[term_id] * count
                sparse_vector[term_id] = weight
        
        return sparse_vector
    
    def encode_documents(self, documents: List[str]) -> List[Dict[int, float]]:
        """
        Encode documents to sparse vectors.
        
        Args:
            documents: List of documents
            
        Returns:
            List of sparse vectors
        """
        if not self._fitted:
            raise ValueError("BM25 not fitted. Call fit() first.")
        
        sparse_vectors = []
        
        for doc in documents:
            tokens = self._tokenize(doc)
            doc_len = len(tokens)
            
            # Count term frequencies
            from collections import Counter
            tf = Counter(tokens)
            
            # Build sparse vector
            sparse_vector = {}
            for token, count in tf.items():
                if token in self.vocab:
                    term_id = self.vocab[token]
                    # BM25 term weight
                    numerator = count * (self.k1 + 1)
                    denominator = count + self.k1 * (
                        1 - self.b + self.b * doc_len / self.avgdl
                    )
                    weight = self.idf[term_id] * (numerator / denominator)
                    sparse_vector[term_id] = weight
            
            sparse_vectors.append(sparse_vector)
        
        return sparse_vectors
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        # Basic tokenization - can be improved with nltk/spacy
        return text.lower().split()


class ColBERTReranker:
    """ColBERTv2 reranker for late-interaction scoring."""
    
    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """
        Initialize ColBERT reranker.
        
        Args:
            model_name: Model identifier
            device: Device to use
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
    
    async def initialize(self):
        """Initialize ColBERT model."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("transformers required for ColBERT")
        
        logger.info(f"Loading ColBERT model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        logger.info(f"ColBERT reranker initialized on {self.device}")
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[tuple]:
        """
        Rerank documents using ColBERT late interaction.
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of top results to return
            
        Returns:
            List of (index, score) tuples sorted by score
        """
        if self.model is None:
            await self.initialize()
        
        import torch
        
        # Encode query
        query_inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        with torch.no_grad():
            query_embeddings = self.model(**query_inputs).last_hidden_state
        
        # Compute scores for each document
        scores = []
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            
            # Encode documents
            doc_inputs = self.tokenizer(
                batch_docs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            
            with torch.no_grad():
                doc_embeddings = self.model(**doc_inputs).last_hidden_state
            
            # Late interaction: MaxSim
            for doc_emb in doc_embeddings:
                # Compute similarity matrix
                similarity = torch.matmul(
                    query_embeddings[0],  # [query_len, dim]
                    doc_emb.t(),  # [dim, doc_len]
                )
                # MaxSim: for each query token, max similarity to any doc token
                max_sim = similarity.max(dim=1).values.mean().item()
                scores.append(max_sim)
        
        # Create ranked list
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        if top_k:
            ranked = ranked[:top_k]
        
        return ranked


class CrossEncoderReranker:
    """Cross-encoder reranker for pairwise scoring."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model name
            device: Device to use
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
    
    async def initialize(self):
        """Initialize cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("sentence-transformers required")
        
        logger.info(f"Loading CrossEncoder: {self.model_name}")
        
        self.model = CrossEncoder(self.model_name, device=self.device)
        
        logger.info(f"Cross-encoder reranker initialized")
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[tuple]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of top results to return
            
        Returns:
            List of (index, score) tuples sorted by score
        """
        if self.model is None:
            await self.initialize()
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Compute scores
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        # Create ranked list
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        if top_k:
            ranked = ranked[:top_k]
        
        return ranked


class HybridRetriever:
    """
    Hybrid retriever combining dense vectors, sparse (BM25), and reranking.
    """
    
    def __init__(
        self,
        vector_backend,
        embedding_provider,
        enable_sparse: bool = True,
        enable_reranking: bool = True,
        reranker_type: str = "cross_encoder",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        device: str = "cpu",
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_backend: Vector database backend (Qdrant)
            embedding_provider: Multimodal embedding provider
            enable_sparse: Enable BM25 sparse retrieval
            enable_reranking: Enable reranking
            reranker_type: "colbert" or "cross_encoder"
            reranker_model: Reranker model name
            dense_weight: Weight for dense scores
            sparse_weight: Weight for sparse scores
            device: Device for models
        """
        self.vector_backend = vector_backend
        self.embedding_provider = embedding_provider
        self.enable_sparse = enable_sparse
        self.enable_reranking = enable_reranking
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.device = device
        
        # Initialize BM25 encoder
        self.bm25_encoder = BM25Encoder() if enable_sparse else None
        
        # Initialize reranker
        self.reranker = None
        if enable_reranking:
            if reranker_type == "colbert":
                self.reranker = ColBERTReranker(
                    model_name=reranker_model,
                    device=device,
                )
            else:
                self.reranker = CrossEncoderReranker(
                    model_name=reranker_model,
                    device=device,
                )
    
    async def initialize(self):
        """Initialize all components."""
        await self.embedding_provider.initialize()
        if self.reranker:
            await self.reranker.initialize()
        logger.info("Hybrid retriever initialized")
    
    async def index_corpus(self, documents: List[str]):
        """
        Index corpus for BM25.
        
        Args:
            documents: List of document texts
        """
        if self.bm25_encoder:
            self.bm25_encoder.fit(documents)
            logger.info("BM25 encoder fitted on corpus")
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        modality: str = "text",
    ) -> SearchResult:
        """
        Perform hybrid search.
        
        Args:
            query: Query text
            top_k: Number of results before reranking
            rerank_top_k: Number of results after reranking
            filters: Metadata filters
            modality: Which modality to query ("text", "image", "audio")
            
        Returns:
            SearchResult with documents and scores
        """
        # Encode query
        dense_result = await self.embedding_provider.embed_text([query])
        dense_vector = dense_result.embeddings[0].tolist()
        
        # Encode sparse if enabled
        sparse_vector = None
        if self.enable_sparse and self.bm25_encoder and self.bm25_encoder._fitted:
            sparse_vector = self.bm25_encoder.encode_query(query)
        
        # Retrieve candidates
        if self.enable_sparse and sparse_vector:
            # Hybrid search
            hybrid_result = await self.vector_backend.hybrid_search(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=top_k,
                dense_weight=self.dense_weight,
                sparse_weight=self.sparse_weight,
                filters=filters,
                vector_name=modality,
            )
            candidates = hybrid_result.documents
        else:
            # Dense-only search
            candidates = await self.vector_backend.query(
                query_vector=dense_vector,
                top_k=top_k,
                filters=filters,
                vector_name=modality,
            )
        
        # Rerank if enabled
        if self.enable_reranking and self.reranker and len(candidates) > 0:
            # Extract document texts
            doc_texts = [doc.content for doc in candidates]
            
            # Rerank
            ranked = await self.reranker.rerank(
                query=query,
                documents=doc_texts,
                top_k=rerank_top_k,
            )
            
            # Reorder candidates
            reranked_candidates = []
            reranked_scores = []
            for idx, score in ranked:
                reranked_candidates.append(candidates[idx])
                reranked_scores.append(score)
            
            candidates = reranked_candidates
            scores = reranked_scores
        else:
            scores = [doc.score for doc in candidates]
        
        # Convert to result format
        documents = [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
            }
            for doc in candidates
        ]
        
        return SearchResult(
            documents=documents,
            scores=scores,
            metadata={
                "query": query,
                "modality": modality,
                "hybrid_enabled": self.enable_sparse,
                "reranking_enabled": self.enable_reranking,
                "num_results": len(documents),
            },
        )
