"""Unit tests for hybrid retrieval components."""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from namel3ss.multimodal.retrieval import (
    BM25Encoder,
    ColBERTReranker,
    CrossEncoderReranker,
    HybridRetriever,
    SearchResult,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_corpus():
    """Sample document corpus for BM25."""
    return [
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing deals with text analysis",
        "Computer vision enables image understanding",
        "Deep learning uses neural networks",
        "Reinforcement learning learns from rewards",
    ]


@pytest.fixture
def sample_query():
    """Sample search query."""
    return "artificial intelligence and machine learning"


class TestBM25Encoder:
    """Test suite for BM25 sparse encoder."""
    
    def test_init(self):
        """Test BM25 encoder initialization."""
        encoder = BM25Encoder()
        assert encoder.vocabulary == {}
        assert encoder.idf == {}
    
    def test_fit(self, sample_corpus):
        """Test fitting BM25 on corpus."""
        encoder = BM25Encoder()
        encoder.fit(sample_corpus)
        
        assert len(encoder.vocabulary) > 0
        assert len(encoder.idf) > 0
        # Check that common words have lower IDF
        assert "the" in encoder.idf or "of" in encoder.idf
    
    def test_encode_query(self, sample_corpus, sample_query):
        """Test encoding query to sparse vector."""
        encoder = BM25Encoder()
        encoder.fit(sample_corpus)
        
        sparse_vector = encoder.encode_query(sample_query)
        
        assert "indices" in sparse_vector
        assert "values" in sparse_vector
        assert len(sparse_vector["indices"]) == len(sparse_vector["values"])
        assert all(isinstance(idx, int) for idx in sparse_vector["indices"])
        assert all(isinstance(val, float) for val in sparse_vector["values"])
    
    def test_encode_documents(self, sample_corpus):
        """Test encoding multiple documents."""
        encoder = BM25Encoder()
        encoder.fit(sample_corpus)
        
        sparse_vectors = encoder.encode_documents(sample_corpus[:2])
        
        assert len(sparse_vectors) == 2
        for vec in sparse_vectors:
            assert "indices" in vec
            assert "values" in vec
    
    def test_empty_corpus(self):
        """Test handling empty corpus."""
        encoder = BM25Encoder()
        with pytest.raises((ValueError, ZeroDivisionError)):
            encoder.fit([])
    
    def test_unknown_terms_in_query(self, sample_corpus):
        """Test query with terms not in vocabulary."""
        encoder = BM25Encoder()
        encoder.fit(sample_corpus)
        
        # Query with completely new terms
        sparse_vector = encoder.encode_query("xyz abc def")
        
        # Should return empty or minimal sparse vector
        assert isinstance(sparse_vector, dict)


class TestColBERTReranker:
    """Test suite for ColBERT late-interaction reranker."""
    
    async def test_init(self):
        """Test ColBERT reranker initialization."""
        reranker = ColBERTReranker(
            model_name="colbert-ir/colbertv2.0",
            device="cpu"
        )
        await reranker.initialize()
        
        assert reranker.model is not None
        assert reranker.tokenizer is not None
    
    @patch("namel3ss.multimodal.retrieval.AutoModel")
    @patch("namel3ss.multimodal.retrieval.AutoTokenizer")
    async def test_rerank(self, mock_tokenizer_class, mock_model_class):
        """Test reranking documents."""
        # Mock model and tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.return_value = Mock(
            last_hidden_state=np.random.rand(1, 3, 128).astype(np.float32)
        )
        
        reranker = ColBERTReranker(model_name="test-model", device="cpu")
        reranker.model = mock_model
        reranker.tokenizer = mock_tokenizer
        
        query = "test query"
        documents = ["doc 1", "doc 2", "doc 3"]
        
        scores = await reranker.rerank(query, documents, top_k=2)
        
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)


class TestCrossEncoderReranker:
    """Test suite for Cross-Encoder reranker."""
    
    async def test_init(self):
        """Test Cross-Encoder initialization."""
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        await reranker.initialize()
        
        assert reranker.model is not None
    
    @patch("namel3ss.multimodal.retrieval.CrossEncoder")
    async def test_rerank(self, mock_cross_encoder_class):
        """Test reranking with cross-encoder."""
        mock_model = Mock()
        mock_cross_encoder_class.return_value = mock_model
        mock_model.predict.return_value = np.array([0.9, 0.7, 0.5])
        
        reranker = CrossEncoderReranker(model_name="test-model")
        reranker.model = mock_model
        
        query = "test query"
        documents = ["doc 1", "doc 2", "doc 3"]
        
        scores = await reranker.rerank(query, documents, top_k=2)
        
        assert len(scores) == 2
        assert scores[0] >= scores[1]  # Should be sorted descending
        mock_model.predict.assert_called_once()


class TestHybridRetriever:
    """Test suite for HybridRetriever."""
    
    async def test_init(self):
        """Test hybrid retriever initialization."""
        mock_backend = Mock()
        mock_provider = Mock()
        
        retriever = HybridRetriever(
            vector_backend=mock_backend,
            embedding_provider=mock_provider,
            enable_sparse=True,
            enable_reranking=True,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        
        assert retriever.vector_backend == mock_backend
        assert retriever.embedding_provider == mock_provider
        assert retriever.enable_sparse is True
        assert retriever.enable_reranking is True
    
    @patch("namel3ss.multimodal.retrieval.BM25Encoder")
    @patch("namel3ss.multimodal.retrieval.CrossEncoderReranker")
    async def test_initialize(self, mock_reranker_class, mock_bm25_class):
        """Test retriever initialization."""
        mock_backend = Mock()
        mock_provider = Mock()
        mock_bm25 = Mock()
        mock_reranker = AsyncMock()
        
        mock_bm25_class.return_value = mock_bm25
        mock_reranker_class.return_value = mock_reranker
        
        retriever = HybridRetriever(
            vector_backend=mock_backend,
            embedding_provider=mock_provider,
            enable_sparse=True,
            enable_reranking=True,
        )
        
        await retriever.initialize()
        
        assert retriever.bm25_encoder is not None
        assert retriever.reranker is not None
        mock_reranker.initialize.assert_called_once()
    
    @patch("namel3ss.multimodal.retrieval.HybridRetriever.initialize")
    async def test_search_dense_only(self, mock_init):
        """Test search with dense vectors only."""
        mock_backend = AsyncMock()
        mock_provider = AsyncMock()
        
        # Mock embedding
        mock_provider.embed_text.return_value = Mock(
            embeddings=np.random.rand(1, 384)
        )
        
        # Mock search results
        mock_backend.query.return_value = [
            {"id": "doc_1", "content": "Result 1", "score": 0.9},
            {"id": "doc_2", "content": "Result 2", "score": 0.8},
        ]
        
        retriever = HybridRetriever(
            vector_backend=mock_backend,
            embedding_provider=mock_provider,
            enable_sparse=False,
            enable_reranking=False,
        )
        
        result = await retriever.search(
            query="test query",
            top_k=2,
        )
        
        assert isinstance(result, SearchResult)
        assert len(result.documents) == 2
        assert len(result.scores) == 2
        mock_provider.embed_text.assert_called_once()
        mock_backend.query.assert_called_once()


class TestSearchResult:
    """Test SearchResult dataclass."""
    
    def test_create_result(self):
        """Test creating search result."""
        docs = [
            {"id": "1", "content": "doc 1"},
            {"id": "2", "content": "doc 2"},
        ]
        scores = [0.95, 0.87]
        metadata = {"query": "test", "hybrid": True}
        
        result = SearchResult(
            documents=docs,
            scores=scores,
            metadata=metadata,
        )
        
        assert len(result.documents) == 2
        assert result.scores[0] > result.scores[1]
        assert result.metadata["hybrid"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
