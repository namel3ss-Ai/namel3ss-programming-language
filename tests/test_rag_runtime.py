"""Unit tests for RAG runtime components."""

import pytest
from namel3ss.rag.chunking import chunk_text, TextChunk
from namel3ss.rag.embeddings import EmbeddingProvider, get_embedding_provider


def test_chunk_text_basic():
    """Test basic text chunking."""
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    
    assert len(chunks) > 0
    assert all(isinstance(c, TextChunk) for c in chunks)
    assert all(c.content for c in chunks)
    
    # Check that chunks have proper IDs (format is "chunk_0", "chunk_1", etc.)
    assert chunks[0].chunk_id == "chunk_0"
    if len(chunks) > 1:
        assert chunks[1].chunk_id == "chunk_1"


def test_chunk_text_with_overlap():
    """Test that overlapping chunks share content."""
    text = "A" * 100 + "B" * 100 + "C" * 100
    chunks = chunk_text(text, chunk_size=150, overlap=50)
    
    assert len(chunks) >= 2
    
    # Check overlap: end of first chunk should appear in second chunk
    if len(chunks) >= 2:
        # Some content from the end of chunk 0 should be in chunk 1
        overlap_region = chunks[0].content[-50:]
        assert overlap_region in chunks[1].content


def test_chunk_text_empty():
    """Test chunking empty text."""
    chunks = chunk_text("", chunk_size=100, overlap=20)
    assert len(chunks) == 0


def test_chunk_text_validation():
    """Test input validation for chunking."""
    # Overlap must be less than chunk_size
    with pytest.raises(ValueError) as exc_info:
        chunk_text("test text", chunk_size=100, overlap=150)
    assert "overlap" in str(exc_info.value).lower()
    
    # Chunk size must be positive
    with pytest.raises(ValueError) as exc_info:
        chunk_text("test text", chunk_size=0, overlap=0)
    assert "chunk_size" in str(exc_info.value).lower()


def test_chunk_text_preserves_content():
    """Test that chunking preserves all content."""
    text = "A" * 500
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    
    # Reconstruct text from non-overlapping parts
    # First chunk fully, then skip overlap for subsequent chunks
    reconstructed = chunks[0].content
    for i in range(1, len(chunks)):
        # Add the non-overlapping part
        overlap = min(50, len(chunks[i-1].content), len(chunks[i].content))
        reconstructed += chunks[i].content[overlap:]
    
    # Should have most of the content (some might be in final overlap)
    assert len(reconstructed) >= len(text) - 50


def test_chunk_text_metadata():
    """Test that chunks include proper metadata."""
    text = "Test content for metadata checking"
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    
    assert len(chunks) > 0
    
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"chunk_{i}"
        assert chunk.start_pos >= 0
        assert chunk.end_pos > chunk.start_pos
        assert chunk.content == text[chunk.start_pos:chunk.end_pos]


def test_embedding_provider_registry():
    """Test embedding provider registry and retrieval."""
    import os
    
    # Skip OpenAI tests if API key is not available
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    # Test OpenAI provider detection
    provider = get_embedding_provider("text-embedding-3-small")
    assert provider is not None
    assert hasattr(provider, "embed_texts")
    assert hasattr(provider, "embed_query")
    
    # Test another OpenAI model
    provider2 = get_embedding_provider("text-embedding-ada-002")
    assert provider2 is not None
    
    # Test Sentence Transformers provider detection
    provider3 = get_embedding_provider("all-MiniLM-L6-v2")
    assert provider3 is not None


def test_embedding_provider_interface():
    """Test that embedding providers implement the required interface."""
    import os
    from namel3ss.rag.embeddings import OpenAIEmbeddingProvider, SentenceTransformerProvider
    
    # Check OpenAI provider only if API key is available
    if os.getenv("OPENAI_API_KEY"):
        openai_provider = OpenAIEmbeddingProvider("text-embedding-ada-002")
        assert hasattr(openai_provider, "embed_texts")
        assert hasattr(openai_provider, "embed_query")
        assert hasattr(openai_provider, "get_dimension")
        assert callable(openai_provider.embed_texts)
        assert callable(openai_provider.embed_query)
        assert callable(openai_provider.get_dimension)
    
    # Check Sentence Transformers provider
    st_provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
    assert hasattr(st_provider, "embed_texts")
    assert hasattr(st_provider, "embed_query")
    assert hasattr(st_provider, "get_dimension")
    assert callable(st_provider.embed_texts)
    assert callable(st_provider.embed_query)
    assert callable(st_provider.get_dimension)


def test_sentence_transformer_dimension():
    """Test getting embedding dimension from Sentence Transformers."""
    from namel3ss.rag.embeddings import SentenceTransformerProvider
    
    provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
    dim = provider.get_dimension()
    
    # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    assert dim == 384


def test_chunk_text_with_separators():
    """Test chunking with different separators."""
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = chunk_text(text, chunk_size=20, overlap=5, separator="\n\n")
    
    assert len(chunks) > 0
    # Each chunk should ideally start with a paragraph
    for chunk in chunks:
        assert chunk.content.strip()  # No empty chunks


def test_chunk_text_large_document():
    """Test chunking a large document."""
    # Create a large document
    text = " ".join([f"Sentence {i}." for i in range(1000)])
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    
    assert len(chunks) > 5
    assert all(len(c.content) <= 700 for c in chunks)  # chunk_size + some buffer
    
    # Check that chunks are in order
    for i in range(len(chunks) - 1):
        assert chunks[i].start_pos < chunks[i+1].start_pos


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
