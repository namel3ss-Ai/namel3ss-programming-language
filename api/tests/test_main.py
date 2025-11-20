"""Integration tests for FastAPI multimodal RAG service."""

import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

# Import the FastAPI app
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.main import app, startup_event


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_file():
    """Create a sample image file."""
    img = Image.new('RGB', (100, 100), color='blue')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return ("test.png", buf, "image/png")


@pytest.fixture
def sample_pdf_file():
    """Create a mock PDF file."""
    content = b"%PDF-1.4\nMock PDF content"
    buf = io.BytesIO(content)
    return ("test.pdf", buf, "application/pdf")


class TestHealthEndpoint:
    """Test /health endpoint."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "components" in data
        # Components might not be initialized in test
        assert isinstance(data["components"], dict)


class TestConfigEndpoint:
    """Test /config endpoint."""
    
    def test_get_config(self, client):
        """Test getting configuration."""
        response = client.get("/config")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check expected config fields
        assert "text_model" in data
        assert "extract_images" in data
        assert "enable_hybrid_search" in data
        assert "device" in data
        assert "collection_name" in data


class TestIngestEndpoint:
    """Test /ingest endpoint."""
    
    @patch("api.main.ingester")
    @patch("api.main.embedding_provider")
    @patch("api.main.vector_backend")
    def test_ingest_text_file(
        self,
        mock_backend,
        mock_provider,
        mock_ingester,
        client,
    ):
        """Test ingesting a text file."""
        # Mock ingestion result
        mock_ingester.ingest_bytes = AsyncMock(return_value=MagicMock(
            document_id="test.txt",
            text_contents=[MagicMock(content="Test content", metadata={})],
            image_contents=[],
            audio_contents=[],
            video_contents=[],
        ))
        
        # Mock embeddings
        mock_provider.embed_text = AsyncMock(return_value=MagicMock(
            embeddings=[[0.1] * 384],
            metadata={},
        ))
        
        # Mock backend upsert
        mock_backend.upsert_multimodal = AsyncMock()
        
        # Create text file
        text_content = b"Hello, world!"
        files = {"file": ("test.txt", io.BytesIO(text_content), "text/plain")}
        
        response = client.post("/ingest", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "document_id" in data
        assert "num_chunks" in data
    
    @patch("api.main.ingester")
    def test_ingest_with_extraction_params(
        self,
        mock_ingester,
        client,
        sample_image_file,
    ):
        """Test ingestion with extraction parameters."""
        mock_ingester.ingest_bytes = AsyncMock(return_value=MagicMock(
            document_id="test.png",
            text_contents=[],
            image_contents=[MagicMock(content=b"image", metadata={})],
            audio_contents=[],
            video_contents=[],
        ))
        
        files = {"file": sample_image_file}
        params = {"extract_images": True, "extract_audio": False}
        
        response = client.post("/ingest", files=files, params=params)
        
        assert response.status_code == 200
    
    def test_ingest_without_file(self, client):
        """Test ingestion without file."""
        response = client.post("/ingest")
        
        assert response.status_code == 422  # Unprocessable entity


class TestSearchEndpoint:
    """Test /search endpoint."""
    
    @patch("api.main.retriever")
    @patch("api.main.embedding_provider")
    def test_search_basic(
        self,
        mock_provider,
        mock_retriever,
        client,
    ):
        """Test basic search."""
        # Mock search results
        mock_retriever.search = AsyncMock(return_value=MagicMock(
            documents=[
                {"id": "doc1", "content": "Result 1", "metadata": {}},
                {"id": "doc2", "content": "Result 2", "metadata": {}},
            ],
            scores=[0.95, 0.87],
            metadata={"query": "test", "hybrid": False},
        ))
        
        search_request = {
            "query": "test query",
            "top_k": 5,
        }
        
        response = client.post("/search", json=search_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "scores" in data
        assert len(data["results"]) == 2
        assert data["scores"][0] > data["scores"][1]
    
    @patch("api.main.retriever")
    def test_search_with_filters(
        self,
        mock_retriever,
        client,
    ):
        """Test search with filters."""
        mock_retriever.search = AsyncMock(return_value=MagicMock(
            documents=[{"id": "doc1", "content": "Filtered result"}],
            scores=[0.92],
            metadata={},
        ))
        
        search_request = {
            "query": "test",
            "top_k": 10,
            "filters": {"category": "technology"},
        }
        
        response = client.post("/search", json=search_request)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) > 0
    
    @patch("api.main.retriever")
    def test_search_with_reranking(
        self,
        mock_retriever,
        client,
    ):
        """Test search with reranking enabled."""
        mock_retriever.search = AsyncMock(return_value=MagicMock(
            documents=[{"id": "doc1", "content": "Reranked result"}],
            scores=[0.98],
            metadata={"reranked": True},
        ))
        
        search_request = {
            "query": "test",
            "top_k": 20,
            "rerank_top_k": 5,
            "enable_reranking": True,
        }
        
        response = client.post("/search", json=search_request)
        
        assert response.status_code == 200
    
    def test_search_missing_query(self, client):
        """Test search without query."""
        response = client.post("/search", json={})
        
        assert response.status_code == 422  # Validation error


class TestStartupEvent:
    """Test application startup."""
    
    @patch("api.main.MultimodalIngester")
    @patch("api.main.MultimodalEmbeddingProvider")
    @patch("api.main.QdrantMultimodalBackend")
    @patch("api.main.HybridRetriever")
    @pytest.mark.asyncio
    async def test_startup_initialization(
        self,
        mock_retriever_class,
        mock_backend_class,
        mock_provider_class,
        mock_ingester_class,
    ):
        """Test that startup initializes all components."""
        # Mock all components
        mock_ingester = MagicMock()
        mock_provider = AsyncMock()
        mock_backend = AsyncMock()
        mock_retriever = AsyncMock()
        
        mock_ingester_class.return_value = mock_ingester
        mock_provider_class.return_value = mock_provider
        mock_backend_class.return_value = mock_backend
        mock_retriever_class.return_value = mock_retriever
        
        # Run startup
        await startup_event()
        
        # Verify initialization calls
        mock_provider.initialize.assert_called_once()
        mock_backend.initialize.assert_called_once()
        mock_retriever.initialize.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
