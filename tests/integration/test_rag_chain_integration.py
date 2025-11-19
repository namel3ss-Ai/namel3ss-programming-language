"""Test RAG integration with chain definitions."""

import pytest
from namel3ss.parser import Parser
from namel3ss.types import check_app
from namel3ss.codegen.backend.state import build_backend_state


def test_rag_in_chain_parsing():
    """Test parsing a chain that uses a RAG pipeline."""
    source = """
app "RAG Chain Test"

llm gpt4:
    provider: openai
    model: gpt-4

dataset docs:
    source: "data/*.txt"

index doc_idx:
    source_dataset: docs
    embedding_model: text-embedding-3-small
    chunk_size: 512
    overlap: 64
    backend: pgvector

rag_pipeline retrieval:
    query_encoder: text-embedding-3-small
    index: doc_idx
    top_k: 5

prompt qa:
    input:
        - question: text
        - context: text
    template: "Q: {question} Context: {context}"

chain rag_qa:
    input -> rag retrieval -> prompt qa -> llm gpt4
"""
    parser = Parser(source, path="test.n3")
    module = parser.parse()
    
    # Should parse without errors
    assert module is not None
    assert len(module.body) == 1
    
    app = module.body[0]
    assert app.name == "RAG Chain Test"
    assert len(app.indices) == 1
    assert len(app.rag_pipelines) == 1
    assert app.indices[0].name == "doc_idx"
    assert app.rag_pipelines[0].name == "retrieval"


def test_rag_chain_type_checking():
    """Test that type checker validates RAG chains correctly."""
    source = """
app "Type Check RAG"

llm gpt4:
    provider: openai
    model: gpt-4

dataset docs:
    source: "data/*.txt"

index doc_idx:
    source_dataset: docs
    embedding_model: text-embedding-3-small
    chunk_size: 512
    overlap: 64
    backend: pgvector

rag_pipeline retrieval:
    query_encoder: text-embedding-3-small
    index: doc_idx
    top_k: 5

prompt qa:
    input:
        - question: text
        - context: text
    template: "Q: {question}"

chain rag_qa:
    input -> rag retrieval -> prompt qa -> llm gpt4
"""
    parser = Parser(source, path="test.n3")
    module = parser.parse()
    app = module.body[0]
    
    errors = check_app(app)
    
    # Should have no errors (all references valid)
    assert len(errors) == 0


def test_rag_chain_invalid_reference():
    """Test that invalid RAG pipeline references are caught."""
    source = """
app "Invalid RAG Chain"

llm gpt4:
    provider: openai
    model: gpt-4

prompt qa:
    input:
        - question: text
    template: "Q: {question}"

chain bad_rag:
    input -> rag nonexistent_pipeline -> prompt qa -> llm gpt4
"""
    parser = Parser(source, path="test.n3")
    module = parser.parse()
    app = module.body[0]
    
    errors = check_app(app)
    
    # Should have error about undefined RAG pipeline
    # Note: This will fail until chain parsing includes RAG step support
    # but documents the expected behavior
    assert len(errors) > 0
    # Would expect error about 'nonexistent_pipeline' not being defined


def test_rag_backend_state_with_chain():
    """Test that backend state encoding includes RAG components used in chains."""
    source = """
app "RAG Backend State"

llm gpt4:
    provider: openai
    model: gpt-4

dataset docs:
    source: "data/*.txt"

index doc_idx:
    source_dataset: docs
    embedding_model: text-embedding-3-small
    chunk_size: 512
    overlap: 64
    backend: pgvector
    table_name: documents

rag_pipeline retrieval:
    query_encoder: text-embedding-3-small
    index: doc_idx
    top_k: 3
    distance_metric: cosine

prompt qa:
    input:
        - question: text
    template: "Q: {question}"

chain rag_qa:
    input -> rag retrieval -> prompt qa -> llm gpt4
"""
    parser = Parser(source, path="test.n3")
    module = parser.parse()
    app = module.body[0]
    
    state = build_backend_state(app)
    
    # Verify indices are encoded
    assert "indices" in state
    assert len(state["indices"]) == 1
    
    idx = state["indices"][0]
    assert idx["name"] == "doc_idx"
    assert idx["source_dataset"] == "docs"
    assert idx["chunk_size"] == 512
    assert idx["overlap"] == 64
    assert idx["table_name"] == "documents"
    
    # Verify RAG pipelines are encoded
    assert "rag_pipelines" in state
    assert len(state["rag_pipelines"]) == 1
    
    pipeline = state["rag_pipelines"][0]
    assert pipeline["name"] == "retrieval"
    assert pipeline["query_encoder"] == "text-embedding-3-small"
    assert pipeline["index"] == "doc_idx"
    assert pipeline["top_k"] == 3
    assert pipeline["distance_metric"] == "cosine"


def test_multiple_rag_pipelines_in_chain():
    """Test encoding multiple RAG pipelines that could be used in chains."""
    source = """
app "Multi-RAG"

dataset docs:
    source: "docs/*.txt"

dataset code:
    source: "src/*.py"

index doc_idx:
    source_dataset: docs
    embedding_model: text-embedding-3-small
    chunk_size: 512
    overlap: 64
    backend: pgvector
    table_name: doc_vectors

index code_idx:
    source_dataset: code
    embedding_model: text-embedding-3-small
    chunk_size: 256
    overlap: 32
    backend: pgvector
    table_name: code_vectors

rag_pipeline doc_search:
    query_encoder: text-embedding-3-small
    index: doc_idx
    top_k: 5

rag_pipeline code_search:
    query_encoder: text-embedding-3-small
    index: code_idx
    top_k: 3
"""
    parser = Parser(source, path="test.n3")
    module = parser.parse()
    app = module.body[0]
    
    state = build_backend_state(app)
    
    # Should have both indices and pipelines
    assert len(state["indices"]) == 2
    assert len(state["rag_pipelines"]) == 2
    
    # Verify doc pipeline
    doc_pipeline = next(p for p in state["rag_pipelines"] if p["name"] == "doc_search")
    assert doc_pipeline["index"] == "doc_idx"
    assert doc_pipeline["top_k"] == 5
    
    # Verify code pipeline
    code_pipeline = next(p for p in state["rag_pipelines"] if p["name"] == "code_search")
    assert code_pipeline["index"] == "code_idx"
    assert code_pipeline["top_k"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
