"""Shared pytest fixtures and configuration for all tests."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_embeddings_384():
    """Sample 384-dimensional embeddings (SentenceTransformers default)."""
    return np.random.rand(10, 384).astype(np.float32)


@pytest.fixture
def sample_embeddings_512():
    """Sample 512-dimensional embeddings (CLIP default)."""
    return np.random.rand(10, 512).astype(np.float32)


@pytest.fixture
def sample_documents():
    """Sample document collection."""
    return [
        {
            "id": f"doc_{i}",
            "content": f"This is test document number {i} about various topics.",
            "metadata": {"index": i, "category": f"cat_{i % 3}"}
        }
        for i in range(20)
    ]


@pytest.fixture
def sample_queries():
    """Sample search queries."""
    return [
        "What is machine learning?",
        "Explain neural networks",
        "How does RAG work?",
        "Vector database performance",
        "Multimodal AI systems",
    ]


@pytest.fixture(scope="session")
def mock_environment(monkeypatch_session):
    """Mock environment variables for testing."""
    env_vars = {
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "DEVICE": "cpu",
        "OPENAI_API_KEY": "test-api-key",
    }
    for key, value in env_vars.items():
        monkeypatch_session.setenv(key, value)


@pytest.fixture
def evaluation_dataset():
    """Sample evaluation dataset."""
    return [
        {
            "query": "What is AI?",
            "relevant_docs": ["doc_1", "doc_3", "doc_5"],
            "relevance_scores": {"doc_1": 3.0, "doc_3": 2.0, "doc_5": 3.0},
            "ground_truth_answer": "AI is artificial intelligence.",
        },
        {
            "query": "Explain machine learning",
            "relevant_docs": ["doc_2", "doc_4"],
            "relevance_scores": {"doc_2": 3.0, "doc_4": 2.0},
            "ground_truth_answer": "Machine learning is a subset of AI.",
        },
    ]


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch for environment variables."""
    from _pytest.monkeypatch import MonkeyPatch
    m = MonkeyPatch()
    yield m
    m.undo()


# Skip markers for optional dependencies
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_qdrant: test requires Qdrant client"
    )
    config.addinivalue_line(
        "markers", "requires_transformers: test requires transformers library"
    )
    config.addinivalue_line(
        "markers", "requires_openai: test requires OpenAI API access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add skip markers."""
    # Check for optional dependencies
    try:
        import qdrant_client
        has_qdrant = True
    except ImportError:
        has_qdrant = False
    
    try:
        import transformers
        has_transformers = True
    except ImportError:
        has_transformers = False
    
    # Add skip markers
    skip_qdrant = pytest.mark.skip(reason="Qdrant client not installed")
    skip_transformers = pytest.mark.skip(reason="Transformers not installed")
    
    for item in items:
        if "requires_qdrant" in item.keywords and not has_qdrant:
            item.add_marker(skip_qdrant)
        if "requires_transformers" in item.keywords and not has_transformers:
            item.add_marker(skip_transformers)
