"""Shared pytest fixtures for multimodal tests."""

import pytest
import numpy as np
from PIL import Image
import io


@pytest.fixture
def sample_text_documents():
    """Sample text documents for testing."""
    return [
        "This is the first test document about machine learning.",
        "Second document discusses natural language processing.",
        "Third document covers computer vision and image recognition.",
        "Fourth document explains multimodal AI systems.",
        "Fifth document talks about vector databases and embeddings.",
    ]


@pytest.fixture
def sample_image_rgb():
    """Create a sample RGB image."""
    img = Image.new('RGB', (224, 224), color='red')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()


@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors."""
    return np.random.rand(10, 384).astype(np.float32)


@pytest.fixture
def sample_metadata():
    """Sample document metadata."""
    return [
        {"doc_id": f"doc_{i}", "category": "test", "page": i}
        for i in range(5)
    ]


@pytest.fixture(scope="session")
def mock_qdrant_available():
    """Check if Qdrant is available for testing."""
    try:
        from qdrant_client import QdrantClient
        return True
    except ImportError:
        return False
