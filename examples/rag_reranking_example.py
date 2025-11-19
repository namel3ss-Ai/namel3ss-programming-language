"""
Example: Using Document Reranking in RAG Pipelines

This example demonstrates how to use production-grade document reranking
to improve retrieval quality in RAG applications.
"""

import asyncio
import os
from namel3ss.rag import (
    RagPipelineRuntime,
    ScoredDocument,
    get_reranker,
)


async def example_basic_reranking():
    """
    Example 1: Basic reranking with sentence-transformers cross-encoder.
    
    This is the recommended approach for local/on-premise deployments
    where you want high-quality reranking without API costs.
    """
    print("\n" + "="*60)
    print("Example 1: Sentence-Transformers Cross-Encoder Reranking")
    print("="*60)
    
    # Sample documents (simulating vector search results)
    documents = [
        ScoredDocument(
            id="doc1",
            content="Machine learning is a method of data analysis that automates analytical model building.",
            score=0.82,
            metadata={"source": "intro.txt"},
        ),
        ScoredDocument(
            id="doc2",
            content="Python is a high-level programming language known for its simplicity.",
            score=0.78,
            metadata={"source": "python.txt"},
        ),
        ScoredDocument(
            id="doc3",
            content="Deep learning is part of machine learning methods based on neural networks.",
            score=0.85,
            metadata={"source": "deep_learning.txt"},
        ),
        ScoredDocument(
            id="doc4",
            content="Natural language processing enables computers to understand human language.",
            score=0.80,
            metadata={"source": "nlp.txt"},
        ),
    ]
    
    query = "What is machine learning?"
    
    print(f"\nQuery: {query}")
    print(f"Initial documents: {len(documents)}")
    print("\nOriginal ranking (by vector similarity):")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. [{doc.score:.3f}] {doc.id}: {doc.content[:60]}...")
    
    try:
        # Initialize reranker with sentence-transformers
        # Uses cross-encoder/ms-marco-MiniLM-L-6-v2 by default
        reranker = get_reranker(
            "sentence_transformers",
            config={
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "device": "cpu",  # Use "cuda" for GPU
                "batch_size": 32,
                "normalize": True,
                "cache_enabled": True,
            }
        )
        
        # Rerank documents
        reranked = await reranker.rerank(query, documents, top_k=3)
        
        print("\nAfter reranking (top 3):")
        for i, doc in enumerate(reranked, 1):
            print(f"  {i}. [{doc.score:.3f}] {doc.id}: {doc.content[:60]}...")
            print(f"      Original score: {doc.metadata['original_score']:.3f}")
    except ImportError as e:
        print(f"\nSkipping: {e}")
        print("Install with: pip install sentence-transformers")


async def example_cohere_reranking():
    """
    Example 2: Using Cohere's Rerank API.
    
    Best for production deployments where you want state-of-the-art
    reranking without managing models. Requires COHERE_API_KEY.
    """
    print("\n" + "="*60)
    print("Example 2: Cohere Rerank API")
    print("="*60)
    
    # Check if API key is available
    if not os.getenv("COHERE_API_KEY"):
        print("\nSkipping Cohere example - COHERE_API_KEY not set")
        print("Set it with: export COHERE_API_KEY='your-key-here'")
        return
    
    documents = [
        ScoredDocument(
            id="doc1",
            content="The latest iPhone features include advanced camera systems and faster processors.",
            score=0.75,
            metadata={},
        ),
        ScoredDocument(
            id="doc2",
            content="Smartphone battery technology has improved significantly with new lithium polymer cells.",
            score=0.80,
            metadata={},
        ),
        ScoredDocument(
            id="doc3",
            content="Mobile app development requires knowledge of iOS and Android platforms.",
            score=0.78,
            metadata={},
        ),
    ]
    
    query = "How has smartphone battery life improved?"
    
    print(f"\nQuery: {query}")
    
    # Initialize Cohere reranker
    reranker = get_reranker(
        "cohere",
        config={
            "model": "rerank-english-v2.0",
            "timeout": 30,
            "cache_enabled": True,
        }
    )
    
    # Rerank
    reranked = await reranker.rerank(query, documents)
    
    print("\nReranked results:")
    for i, doc in enumerate(reranked, 1):
        print(f"  {i}. [{doc.score:.3f}] {doc.content[:70]}...")


async def example_rag_pipeline_with_reranking():
    """
    Example 3: Full RAG pipeline with integrated reranking.
    
    Shows how reranking is integrated into RagPipelineRuntime for
    production use.
    """
    print("\n" + "="*60)
    print("Example 3: RAG Pipeline with Reranking")
    print("="*60)
    
    # Note: This is a conceptual example
    # In production, you would have a real vector backend initialized
    
    print("\nIn a production RAG pipeline:")
    print("""
    pipeline = RagPipelineRuntime(
        name="my_rag_pipeline",
        query_encoder="text-embedding-3-small",
        index_backend=my_vector_backend,
        top_k=20,  # Retrieve 20 candidates
        reranker="sentence_transformers",  # Enable reranking
        config={
            "reranker_config": {
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "batch_size": 32,
                "cache_enabled": True,
            }
        }
    )
    
    # Execute query - reranking happens automatically
    result = await pipeline.execute_query("What is machine learning?")
    
    # The result.documents are already reranked
    for doc in result.documents:
        print(f"Score: {doc.score}, Content: {doc.content[:50]}...")
    
    # Check timing metadata
    print(f"Embedding: {result.metadata['timings']['embedding_ms']}ms")
    print(f"Search: {result.metadata['timings']['search_ms']}ms")
    print(f"Rerank: {result.metadata['timings']['rerank_ms']}ms")
    """)


async def example_custom_http_reranker():
    """
    Example 4: Using a custom HTTP reranking endpoint.
    
    Useful when you have your own reranking service or want to use
    a provider not directly supported.
    """
    print("\n" + "="*60)
    print("Example 4: Custom HTTP Reranker")
    print("="*60)
    
    print("\nFor custom reranking APIs:")
    print("""
    reranker = get_reranker(
        "http",
        config={
            "endpoint": "https://my-reranker.example.com/api/rerank",
            "headers": {
                "Authorization": "Bearer YOUR_TOKEN",
                "Content-Type": "application/json",
            },
            "timeout": 60,
            "max_retries": 3,
            "request_format": "standard",
            "response_format": "standard",
            "cache_enabled": True,
        }
    )
    
    # The HTTP reranker expects your API to accept:
    # POST /api/rerank
    # {
    #   "query": "...",
    #   "documents": [
    #     {"id": "doc1", "content": "...", "metadata": {...}},
    #     ...
    #   ],
    #   "top_k": 5
    # }
    #
    # And return:
    # {
    #   "results": [
    #     {"id": "doc3", "score": 0.95},
    #     {"id": "doc1", "score": 0.87},
    #     ...
    #   ]
    # }
    
    reranked = await reranker.rerank(query, documents, top_k=5)
    """)


async def example_performance_optimization():
    """
    Example 5: Performance optimization with caching and batching.
    """
    print("\n" + "="*60)
    print("Example 5: Performance Optimization")
    print("="*60)
    
    print("\nOptimizing reranking performance:")
    print("""
    # 1. Enable caching for repeated queries
    reranker = get_reranker(
        "sentence_transformers",
        config={
            "cache_enabled": True,
            "cache_size": 2000,      # Cache 2000 query-document combinations
            "cache_ttl": 3600,       # 1 hour TTL
        }
    )
    
    # 2. Adjust batch size for your hardware
    reranker = get_reranker(
        "sentence_transformers",
        config={
            "batch_size": 64,        # Larger batches for GPU
            "device": "cuda",        # Use GPU if available
        }
    )
    
    # 3. For large document sets, consider two-stage retrieval:
    pipeline = RagPipelineRuntime(
        name="two_stage_rag",
        query_encoder="text-embedding-3-small",
        index_backend=backend,
        top_k=100,              # Retrieve many candidates cheaply
        reranker="sentence_transformers",
        config={
            "reranker_config": {
                "batch_size": 64,
                # Reranker will use top_k from pipeline (100 -> 100)
                # Or specify different top_k in execute_query()
            }
        }
    )
    
    # Retrieve 100, rerank all, return top 5
    result = await pipeline.execute_query(query, top_k=5)
    """)


async def example_error_handling():
    """
    Example 6: Robust error handling in production.
    """
    print("\n" + "="*60)
    print("Example 6: Error Handling")
    print("="*60)
    
    print("\nThe pipeline gracefully handles reranker failures:")
    print("""
    # If reranker fails to initialize, pipeline still works without it
    pipeline = RagPipelineRuntime(
        name="robust_pipeline",
        query_encoder="text-embedding-3-small",
        index_backend=backend,
        reranker="sentence_transformers",  # If this fails...
    )
    # ...the pipeline logs a warning and continues without reranking
    
    # If reranking fails during query:
    result = await pipeline.execute_query(query)
    # Pipeline logs warning and returns original vector search results
    
    # Check if reranking was used:
    if "rerank_ms" in result.metadata["timings"]:
        print("Reranking was applied")
    else:
        print("Using original ranking")
    """)


async def example_compare_rerankers():
    """
    Example 7: Comparing different reranker backends.
    """
    print("\n" + "="*60)
    print("Example 7: Comparing Reranker Backends")
    print("="*60)
    
    documents = [
        ScoredDocument(
            id="doc1",
            content="Climate change is causing global temperature rise.",
            score=0.80,
            metadata={},
        ),
        ScoredDocument(
            id="doc2",
            content="Renewable energy sources include solar and wind power.",
            score=0.85,
            metadata={},
        ),
        ScoredDocument(
            id="doc3",
            content="Electric vehicles help reduce carbon emissions.",
            score=0.75,
            metadata={},
        ),
    ]
    
    query = "How can we reduce greenhouse gas emissions?"
    
    print(f"\nQuery: {query}")
    print("\nOriginal ranking:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. [{doc.score:.2f}] {doc.id}")
    
    # Compare different models
    rerankers = [
        ("sentence_transformers", {
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        }),
        # Add more rerankers as needed
    ]
    
    for name, config in rerankers:
        try:
            reranker = get_reranker(name, config)
            reranked = await reranker.rerank(query, documents)
            
            print(f"\n{reranker.get_model_name()}:")
            for i, doc in enumerate(reranked, 1):
                print(f"  {i}. [{doc.score:.3f}] {doc.id}")
        except (ImportError, Exception) as e:
            print(f"\n{name}: Skipped - {str(e)[:80]}")


async def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("  NAMEL3SS RAG DOCUMENT RERANKING EXAMPLES")
    print("="*70)
    
    await example_basic_reranking()
    await example_cohere_reranking()
    await example_rag_pipeline_with_reranking()
    await example_custom_http_reranker()
    await example_performance_optimization()
    await example_error_handling()
    await example_compare_rerankers()
    
    print("\n" + "="*70)
    print("  Examples complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • Use sentence_transformers for local, high-quality reranking")
    print("  • Use cohere for state-of-the-art cloud-based reranking")
    print("  • Use http for custom reranking services")
    print("  • Enable caching for performance with repeated queries")
    print("  • Pipelines gracefully handle reranker failures")
    print("  • Two-stage retrieval (large top_k + rerank) gives best results")
    print()


if __name__ == "__main__":
    asyncio.run(main())
