# Document Reranking in Namel3ss RAG Pipelines

## Overview

Namel3ss provides production-grade document reranking capabilities for RAG (Retrieval-Augmented Generation) pipelines. Reranking improves retrieval quality by using more sophisticated models to re-score and re-order documents after initial vector similarity search.

## Why Reranking?

Vector similarity search (using embeddings) is fast but limited:
- **Semantic limitations**: Embeddings capture general meaning but miss fine-grained relevance
- **Context insensitivity**: Can't model complex query-document interactions
- **Fixed representation**: Single vector per document/query

Reranking solves these issues by:
- **Cross-attention models**: Analyze query-document pairs jointly for better relevance scoring
- **Contextual understanding**: Consider full query and document context together
- **Higher accuracy**: State-of-the-art cross-encoders significantly improve ranking quality

**Best practice**: Use fast vector search for candidate retrieval (top 100-200), then rerank to get top results (top 5-10).

## Architecture

### Components

1. **BaseReranker Protocol**: Abstract interface for all reranker implementations
2. **Reranker Factory**: `get_reranker()` function for instantiating rerankers
3. **Built-in Implementations**:
   - `SentenceTransformerReranker`: Local cross-encoder models
   - `CohereReranker`: Cohere Rerank API
   - `HTTPReranker`: Generic HTTP endpoint wrapper
4. **Caching**: Built-in LRU cache with TTL for performance
5. **Pipeline Integration**: Seamless integration with `RagPipelineRuntime`

### Data Flow

```
Query + Candidate Documents (from vector search)
        ↓
    Reranker
        ↓
Query-Document Pairs → Batch Scoring → Sort by Score → Top-K
        ↓
Reranked Documents (higher quality ranking)
```

## Quick Start

### 1. Basic Usage with Sentence-Transformers

```python
from namel3ss.rag import get_reranker, ScoredDocument

# Initialize reranker
reranker = get_reranker(
    "sentence_transformers",
    config={
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "device": "cpu",  # or "cuda" for GPU
        "batch_size": 32,
    }
)

# Rerank documents
documents = [...]  # List of ScoredDocument
reranked = await reranker.rerank(
    query="What is machine learning?",
    documents=documents,
    top_k=5
)
```

### 2. Integrated with RAG Pipeline

```python
from namel3ss.rag import RagPipelineRuntime

pipeline = RagPipelineRuntime(
    name="my_pipeline",
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

# Reranking happens automatically
result = await pipeline.execute_query("What is machine learning?")
print(f"Reranking took: {result.metadata['timings']['rerank_ms']}ms")
```

## Reranker Backends

### Sentence-Transformers (Local Cross-Encoders)

**Best for**: Local/on-premise deployments, GPU available, privacy requirements

```python
reranker = get_reranker(
    "sentence_transformers",
    config={
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Or other HF model
        "device": "cpu",           # "cpu", "cuda", or "mps"
        "batch_size": 32,          # Larger for GPU
        "max_length": 512,         # Max sequence length
        "normalize": True,         # Normalize scores to [0, 1]
        "cache_enabled": True,     # Enable caching
        "cache_size": 1000,        # Max cached entries
        "cache_ttl": 3600,         # Cache TTL in seconds
    }
)
```

**Popular Models**:
- `cross-encoder/ms-marco-MiniLM-L-6-v2`: Fast, good quality (default)
- `cross-encoder/ms-marco-TinyBERT-L-2-v2`: Faster, lower quality
- `cross-encoder/ms-marco-MiniLM-L-12-v2`: Slower, higher quality

**Requirements**: `pip install sentence-transformers`

### Cohere Rerank API

**Best for**: Cloud deployments, state-of-the-art quality, managed service

```python
import os
os.environ["COHERE_API_KEY"] = "your-key"

reranker = get_reranker(
    "cohere",
    config={
        "model": "rerank-english-v2.0",  # Or "rerank-multilingual-v2.0"
        "timeout": 30,
        "max_retries": 3,
        "cache_enabled": True,
    }
)
```

**Requirements**: `pip install cohere`

**Pricing**: See [Cohere Pricing](https://cohere.com/pricing)

### HTTP Reranker (Custom Endpoints)

**Best for**: Custom reranking services, proprietary models

```python
reranker = get_reranker(
    "http",
    config={
        "endpoint": "https://my-service.com/rerank",
        "headers": {
            "Authorization": "Bearer YOUR_TOKEN",
        },
        "timeout": 60,
        "max_retries": 3,
        "request_format": "standard",  # See API format below
        "response_format": "standard",
    }
)
```

**Expected API Format**:

Request (POST):
```json
{
  "query": "...",
  "documents": [
    {"id": "doc1", "content": "...", "metadata": {...}},
    ...
  ],
  "top_k": 5
}
```

Response:
```json
{
  "results": [
    {"id": "doc1", "score": 0.95},
    {"id": "doc3", "score": 0.87},
    ...
  ]
}
```

## Performance Optimization

### 1. Enable Caching

Cache reranking results for repeated queries:

```python
config = {
    "cache_enabled": True,
    "cache_size": 2000,      # Cache 2000 query-document combinations
    "cache_ttl": 3600,       # 1 hour TTL
}
```

**When to use**: Applications with repeated or similar queries (chatbots, search interfaces)

### 2. Batch Size Tuning

```python
# For CPU
config = {"batch_size": 32}

# For GPU
config = {
    "batch_size": 128,       # Larger batches
    "device": "cuda",
}
```

**Guidelines**:
- CPU: 16-32
- GPU: 64-256 depending on model size and VRAM
- Monitor memory usage and adjust

### 3. Two-Stage Retrieval

Retrieve many candidates cheaply, rerank fewer:

```python
pipeline = RagPipelineRuntime(
    name="two_stage",
    top_k=100,               # Retrieve 100 candidates (fast)
    reranker="sentence_transformers",
)

# Rerank all 100, return top 5 (accurate)
result = await pipeline.execute_query(query, top_k=5)
```

**Typical configuration**:
- Retrieve: 50-200 candidates
- Rerank & return: 5-20 results

### 4. Model Selection

Choose model based on latency/quality tradeoff:

| Model | Latency | Quality | Use Case |
|-------|---------|---------|----------|
| TinyBERT-L-2 | ~5ms | Good | High QPS services |
| MiniLM-L-6 | ~10ms | Better | Balanced (default) |
| MiniLM-L-12 | ~20ms | Best | Quality-critical |

## Error Handling

The pipeline gracefully handles reranker failures:

```python
# If reranker initialization fails
pipeline = RagPipelineRuntime(
    reranker="sentence_transformers",  # Missing library, etc.
)
# Logs warning, continues without reranking

# If reranking fails during query
result = await pipeline.execute_query(query)
# Logs warning, returns original vector search ranking

# Check if reranking was applied
if "rerank_ms" in result.metadata["timings"]:
    print("Reranking succeeded")
else:
    print("Using original ranking")
```

## Advanced Usage

### Custom Reranker Implementation

Implement your own reranker:

```python
from namel3ss.rag.rerankers import BaseReranker, register_reranker

class MyCustomReranker(BaseReranker):
    def __init__(self, config=None):
        self.config = config or {}
        # Initialize your model
    
    async def rerank(self, query, documents, top_k=None):
        # Your reranking logic
        # Must return List[ScoredDocument] sorted by score
        ...
    
    def get_model_name(self):
        return "my_custom_reranker"

# Register it
register_reranker("custom", MyCustomReranker)

# Use it
reranker = get_reranker("custom", config={...})
```

### Monitoring and Observability

```python
result = await pipeline.execute_query(query)

# Check timings
timings = result.metadata["timings"]
print(f"Embedding: {timings['embedding_ms']}ms")
print(f"Search: {timings['search_ms']}ms")
print(f"Rerank: {timings.get('rerank_ms', 'N/A')}ms")
print(f"Total: {timings['total_ms']}ms")

# Check which reranker was used
print(f"Reranker: {result.metadata.get('reranker', 'none')}")

# Inspect document scores
for doc in result.documents:
    if "original_score" in doc.metadata:
        print(f"Doc {doc.id}:")
        print(f"  Original: {doc.metadata['original_score']:.3f}")
        print(f"  Reranked: {doc.score:.3f}")
```

## Testing

The reranking system includes comprehensive tests:

```bash
# Run all reranker tests
pytest tests/test_rerankers.py -v

# Run specific test categories
pytest tests/test_rerankers.py::TestSimpleCache -v
pytest tests/test_rerankers.py::TestRerankerFactory -v
pytest tests/test_rerankers.py::TestRerankerIntegration -v
```

Test coverage includes:
- Cache operations (LRU, TTL, consistency)
- Factory and registration
- Order changes and score updates
- Metadata preservation
- Error handling and timeouts
- Performance with large document sets
- Configuration validation

## Production Deployment

### Checklist

- [ ] Choose appropriate reranker backend for your use case
- [ ] Configure batch size for your hardware
- [ ] Enable caching for repeated queries
- [ ] Set up monitoring for reranking timing
- [ ] Test graceful degradation (what happens if reranker fails)
- [ ] Tune two-stage retrieval (retrieve vs. rerank counts)
- [ ] Load test to verify latency under production load

### Environment Variables

```bash
# For Cohere
export COHERE_API_KEY="your-key"

# For custom HTTP reranker (if using env vars for auth)
export RERANKER_API_KEY="your-key"
export RERANKER_ENDPOINT="https://..."
```

### Resource Requirements

**Sentence-Transformers (CPU)**:
- Memory: ~1-2GB per model
- CPU: 2-4 cores recommended
- Latency: 10-50ms per batch of 32

**Sentence-Transformers (GPU)**:
- VRAM: 2-4GB depending on model
- GPU: Any CUDA-compatible GPU
- Latency: 1-5ms per batch of 128

**Cohere API**:
- No local resources
- Rate limits apply (check Cohere docs)
- Latency: 100-500ms depending on batch size and location

## Troubleshooting

### "sentence-transformers is required"

```bash
pip install sentence-transformers
```

### "Cohere API key not found"

```bash
export COHERE_API_KEY="your-key"
# Or pass in config:
config = {"api_key": "your-key"}
```

### Slow reranking performance

1. Check batch size - increase for GPU, decrease if OOM
2. Enable caching for repeated queries
3. Use smaller/faster model (TinyBERT instead of MiniLM-L-12)
4. Reduce number of documents to rerank

### Reranking doesn't improve results

1. Ensure query and documents are in English (for English models)
2. Try different reranker model
3. Increase number of candidates retrieved before reranking
4. Check if reranking is actually running (check timing metadata)

### Memory errors with large batches

```python
config = {
    "batch_size": 16,  # Reduce batch size
    "cache_size": 500,  # Reduce cache size
}
```

## Examples

See `examples/rag_reranking_example.py` for comprehensive examples of:
- Basic reranking
- Pipeline integration
- Performance optimization
- Error handling
- Comparing different rerankers

Run examples:
```bash
python examples/rag_reranking_example.py
```

## API Reference

### get_reranker()

```python
def get_reranker(
    name: str,
    config: Optional[Dict[str, Any]] = None
) -> BaseReranker
```

Factory function to create reranker instances.

**Parameters**:
- `name`: Reranker type ("sentence_transformers", "cohere", "http", or custom)
- `config`: Configuration dict (varies by reranker type)

**Returns**: Reranker instance implementing BaseReranker protocol

**Raises**: ValueError if reranker type is unknown

### BaseReranker.rerank()

```python
async def rerank(
    self,
    query: str,
    documents: List[ScoredDocument],
    top_k: Optional[int] = None,
) -> List[ScoredDocument]
```

Rerank documents based on relevance to query.

**Parameters**:
- `query`: Query string
- `documents`: List of candidate documents to rerank
- `top_k`: Optional limit on results (defaults to len(documents))

**Returns**: Reranked list of documents sorted by relevance (descending)

### register_reranker()

```python
def register_reranker(name: str, reranker_class: type) -> None
```

Register a custom reranker implementation.

**Parameters**:
- `name`: Name to register under
- `reranker_class`: Class implementing BaseReranker protocol

## Further Reading

- [Sentence-Transformers Cross-Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Cohere Rerank API](https://docs.cohere.com/docs/reranking)
- [Two-Stage Retrieval Pattern](https://www.pinecone.io/learn/series/rag/rerankers/)

## Support

For issues or questions:
- GitHub Issues: [namel3ss-programming-language/issues](https://github.com/SsebowaDisan/namel3ss-programming-language/issues)
- See tests: `tests/test_rerankers.py` for usage examples
- Run examples: `examples/rag_reranking_example.py`
