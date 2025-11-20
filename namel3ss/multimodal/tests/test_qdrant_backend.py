"""Unit tests for Qdrant multimodal backend."""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from namel3ss.multimodal.qdrant_backend import (
    QdrantMultimodalBackend,
    HybridSearchResult,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def backend_config():
    """Qdrant backend configuration."""
    return {
        "host": "localhost",
        "port": 6333,
        "collection_name": "test_multimodal",
    }


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {"id": "doc_1", "content": "First document", "metadata": {"category": "A"}},
        {"id": "doc_2", "content": "Second document", "metadata": {"category": "B"}},
        {"id": "doc_3", "content": "Third document", "metadata": {"category": "A"}},
    ]


@pytest.fixture
def sample_text_embeddings():
    """Sample text embeddings."""
    return np.random.rand(3, 384).astype(np.float32)


@pytest.fixture
def sample_image_embeddings():
    """Sample image embeddings."""
    return np.random.rand(2, 512).astype(np.float32)


class TestQdrantMultimodalBackend:
    """Test suite for QdrantMultimodalBackend."""
    
    async def test_init(self, backend_config):
        """Test backend initialization."""
        backend = QdrantMultimodalBackend(config=backend_config)
        
        assert backend.collection_name == "test_multimodal"
        assert backend.host == "localhost"
        assert backend.port == 6333
    
    @patch("namel3ss.multimodal.qdrant_backend.QdrantClient")
    async def test_initialize_creates_collection(self, mock_client_class, backend_config):
        """Test that initialize creates collection."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists = AsyncMock(return_value=False)
        mock_client.create_collection = AsyncMock()
        
        backend = QdrantMultimodalBackend(config=backend_config)
        await backend.initialize()
        
        mock_client.create_collection.assert_called_once()
    
    @patch("namel3ss.multimodal.qdrant_backend.QdrantClient")
    async def test_upsert_multimodal(
        self,
        mock_client_class,
        backend_config,
        sample_documents,
        sample_text_embeddings,
    ):
        """Test upserting multimodal documents."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_client.upsert = AsyncMock()
        
        backend = QdrantMultimodalBackend(config=backend_config)
        backend.client = mock_client
        
        await backend.upsert_multimodal(
            documents=sample_documents,
            text_embeddings=sample_text_embeddings,
            image_embeddings=None,
            audio_embeddings=None,
        )
        
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert call_args[1]["collection_name"] == "test_multimodal"
        assert len(call_args[1]["points"]) == 3
    
    @patch("namel3ss.multimodal.qdrant_backend.QdrantClient")
    async def test_upsert_with_images(
        self,
        mock_client_class,
        backend_config,
        sample_documents,
        sample_text_embeddings,
        sample_image_embeddings,
    ):
        """Test upserting documents with both text and image embeddings."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_client.upsert = AsyncMock()
        
        backend = QdrantMultimodalBackend(config=backend_config)
        backend.client = mock_client
        
        # Only 2 documents have images
        docs_with_images = sample_documents[:2]
        
        await backend.upsert_multimodal(
            documents=docs_with_images,
            text_embeddings=sample_text_embeddings[:2],
            image_embeddings=sample_image_embeddings,
            audio_embeddings=None,
        )
        
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        points = call_args[1]["points"]
        
        # Check that image vectors are included
        assert len(points) == 2
        for point in points:
            assert "text" in point.vector
            assert "image" in point.vector
    
    @patch("namel3ss.multimodal.qdrant_backend.QdrantClient")
    async def test_query_dense_only(self, mock_client_class, backend_config):
        """Test dense-only query."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists = AsyncMock(return_value=True)
        
        # Mock search results
        mock_result = MagicMock()
        mock_result.id = "doc_1"
        mock_result.score = 0.95
        mock_result.payload = {"content": "Test doc"}
        mock_client.search = AsyncMock(return_value=[mock_result])
        
        backend = QdrantMultimodalBackend(config=backend_config)
        backend.client = mock_client
        
        query_embedding = np.random.rand(384).astype(np.float32)
        results = await backend.query(
            query_embedding=query_embedding,
            top_k=5,
            modality="text",
        )
        
        assert len(results) == 1
        assert results[0]["id"] == "doc_1"
        assert results[0]["score"] == 0.95
        mock_client.search.assert_called_once()
    
    @patch("namel3ss.multimodal.qdrant_backend.QdrantClient")
    async def test_hybrid_search(self, mock_client_class, backend_config):
        """Test hybrid search with dense and sparse vectors."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists = AsyncMock(return_value=True)
        
        # Mock dense search results
        dense_result = MagicMock()
        dense_result.id = "doc_1"
        dense_result.score = 0.9
        dense_result.payload = {"content": "Dense match"}
        
        # Mock sparse search results
        sparse_result = MagicMock()
        sparse_result.id = "doc_2"
        sparse_result.score = 0.85
        sparse_result.payload = {"content": "Sparse match"}
        
        mock_client.search = AsyncMock(side_effect=[
            [dense_result],
            [sparse_result],
        ])
        
        backend = QdrantMultimodalBackend(config=backend_config)
        backend.client = mock_client
        
        query_embedding = np.random.rand(384).astype(np.float32)
        sparse_vector = {"indices": [0, 5, 10], "values": [0.8, 0.6, 0.4]}
        
        result = await backend.hybrid_search(
            query_embedding=query_embedding,
            sparse_vector=sparse_vector,
            top_k=10,
            dense_weight=0.7,
            sparse_weight=0.3,
        )
        
        assert isinstance(result, HybridSearchResult)
        assert len(result.documents) > 0
        assert len(result.scores) == len(result.documents)
        assert mock_client.search.call_count == 2
    
    @patch("namel3ss.multimodal.qdrant_backend.QdrantClient")
    async def test_delete_documents(self, mock_client_class, backend_config):
        """Test deleting documents."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.delete = AsyncMock()
        
        backend = QdrantMultimodalBackend(config=backend_config)
        backend.client = mock_client
        
        await backend.delete(["doc_1", "doc_2"])
        
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert call_args[1]["collection_name"] == "test_multimodal"


class TestHybridSearchResult:
    """Test HybridSearchResult dataclass."""
    
    def test_create_result(self):
        """Test creating hybrid search result."""
        docs = [{"id": "1", "content": "test"}]
        scores = [0.95]
        metadata = {"fusion_method": "rrf"}
        
        result = HybridSearchResult(
            documents=docs,
            scores=scores,
            metadata=metadata,
        )
        
        assert len(result.documents) == 1
        assert result.scores[0] == 0.95
        assert result.metadata["fusion_method"] == "rrf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
