# Multimodal RAG System - API Reference

## FastAPI Endpoints

### Health Check

**GET** `/health`

Check service health status.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "ingester": "ok",
    "embedding_provider": "ok",
    "vector_backend": "ok",
    "retriever": "ok"
  }
}
```

---

### Ingest Document

**POST** `/ingest`

Ingest a document and extract multimodal content.

**Parameters:**
- `file` (form-data, required): Document file (PDF, image, audio, text, Word doc)
- `extract_images` (query, optional): Extract images from document (default: true)
- `extract_audio` (query, optional): Extract audio from video (default: false)

**Example:**
```bash
curl -X POST "http://localhost:8000/ingest?extract_images=true" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "document_id": "document.pdf",
  "num_chunks": 45,
  "modalities": ["text", "image"],
  "status": "success",
  "error": null
}
```

---

### Search Documents

**POST** `/search`

Perform hybrid search with optional reranking.

**Request Body:**
```json
{
  "query": "How do I configure authentication?",
  "top_k": 10,
  "rerank_top_k": 5,
  "enable_hybrid": true,
  "enable_reranking": true,
  "filters": {"category": "security"},
  "modality": "text"
}
```

**Response:**
```json
{
  "query": "How do I configure authentication?",
  "results": [
    {
      "id": "doc_123_chunk_5",
      "content": "To configure authentication...",
      "metadata": {
        "document_id": "doc_123",
        "filename": "security_guide.pdf",
        "modality": "text",
        "chunk_index": 5
      }
    }
  ],
  "scores": [0.92, 0.87, 0.81, 0.75, 0.68],
  "metadata": {
    "query": "How do I configure authentication?",
    "modality": "text",
    "hybrid_enabled": true,
    "reranking_enabled": true,
    "num_results": 5
  }
}
```

---

### Get Configuration

**GET** `/config`

Get current system configuration.

**Response:**
```json
{
  "text_model": "all-MiniLM-L6-v2",
  "image_model": "openai/clip-vit-base-patch32",
  "extract_images": true,
  "extract_audio": false,
  "enable_hybrid_search": true,
  "enable_reranking": true,
  "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "device": "cpu",
  "vector_db_type": "qdrant",
  "collection_name": "multimodal_docs"
}
```

---

## CLI Commands

### Evaluate RAG System

```bash
n3 eval rag <dataset_path> [OPTIONS]
```

**Arguments:**
- `dataset_path`: Path to evaluation dataset (JSON or CSV)

**Options:**
- `--output, -o`: Output file for markdown results (default: evaluation_results.md)
- `--collection`: Qdrant collection name (default: multimodal_docs)
- `--qdrant-host`: Qdrant host (default: localhost)
- `--qdrant-port`: Qdrant port (default: 6333)
- `--top-k`: Number of documents to retrieve (default: 10)
- `--k-values`: Comma-separated k values for metrics (default: 1,3,5,10)
- `--use-llm-judge`: Use LLM to judge answer quality (flag)
- `--llm-model`: LLM model for judging (default: gpt-4)
- `--llm-api-key`: OpenAI API key (or set OPENAI_API_KEY env var)
- `--csv-output`: Optional CSV file for detailed results
- `--device`: Device for models (default: cpu)

**Example:**
```bash
n3 eval rag eval_dataset.json \
  --collection my_docs \
  --top-k 20 \
  --k-values 1,5,10 \
  --use-llm-judge \
  --output results.md \
  --csv-output results.csv
```

---

### Batch Search

```bash
n3 eval rag batch-search <queries_file> [OPTIONS]
```

**Arguments:**
- `queries_file`: File with queries (one per line or JSON array)

**Options:**
- `--collection`: Qdrant collection name
- `--qdrant-host`: Qdrant host
- `--qdrant-port`: Qdrant port
- `--top-k`: Number of results per query
- `--device`: Device for models

**Example:**
```bash
n3 eval rag batch-search queries.txt \
  --collection my_docs \
  --top-k 10 > results.json
```

---

## Python API

### Multimodal Ingester

```python
from namel3ss.multimodal import MultimodalIngester

ingester = MultimodalIngester(
    extract_images=True,
    extract_audio=False,
    max_image_size=(1024, 1024),
)

# Ingest file
result = await ingester.ingest_file("document.pdf")

# Access extracted contents
text_contents = result.text_contents
image_contents = result.image_contents
audio_contents = result.audio_contents
```

---

### Multimodal Embeddings

```python
from namel3ss.multimodal import MultimodalEmbeddingProvider

provider = MultimodalEmbeddingProvider(
    text_model="all-MiniLM-L6-v2",
    image_model="openai/clip-vit-base-patch32",
    audio_model="openai/whisper-base",
    device="cpu",
)

await provider.initialize()

# Embed text
text_result = await provider.embed_text(["example text"])
# text_result.embeddings: np.ndarray of shape (1, 384)

# Embed images
image_result = await provider.embed_images([image_bytes])
# image_result.embeddings: np.ndarray of shape (1, 512)

# Embed audio
audio_result = await provider.embed_audio([audio_bytes])
# audio_result.embeddings: np.ndarray of shape (1, 384)
# audio_result.metadata["transcripts"]: ["transcribed text"]
```

---

### Hybrid Retriever

```python
from namel3ss.multimodal import HybridRetriever
from namel3ss.multimodal.qdrant_backend import QdrantMultimodalBackend

# Initialize backend
backend = QdrantMultimodalBackend(config={
    "host": "localhost",
    "port": 6333,
    "collection_name": "my_docs",
})
await backend.initialize()

# Initialize retriever
retriever = HybridRetriever(
    vector_backend=backend,
    embedding_provider=provider,
    enable_sparse=True,
    enable_reranking=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
)
await retriever.initialize()

# Search
result = await retriever.search(
    query="example query",
    top_k=10,
    rerank_top_k=5,
)

# Access results
for doc, score in zip(result.documents, result.scores):
    print(f"Score: {score}, Content: {doc['content']}")
```

---

### RAG Evaluator

```python
from namel3ss.eval.rag_eval import RAGEvaluator

evaluator = RAGEvaluator(
    k_values=[1, 3, 5, 10],
    use_llm_judge=True,
    llm_judge_model="gpt-4",
)

# Evaluate single query
result = await evaluator.evaluate_query(
    query="example query",
    retrieved_doc_ids=["doc1", "doc2", "doc3"],
    relevant_doc_ids=["doc2", "doc5"],
)

print(f"Precision@5: {result.precision_at_k[5]}")
print(f"NDCG@5: {result.ndcg_at_k[5]}")
print(f"MRR: {result.mrr}")

# Evaluate dataset
aggregated = await evaluator.evaluate_dataset(
    eval_examples=dataset,
    retriever_fn=my_retriever_function,
)

# Generate report
report = evaluator.format_results(aggregated)
print(report)
```

---

## Environment Variables

```bash
# Qdrant connection
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export COLLECTION_NAME=multimodal_docs

# Model selection
export TEXT_MODEL=all-MiniLM-L6-v2
export IMAGE_MODEL=openai/clip-vit-base-patch32
export AUDIO_MODEL=openai/whisper-base

# Feature flags
export EXTRACT_IMAGES=true
export EXTRACT_AUDIO=false
export ENABLE_HYBRID=true
export ENABLE_RERANKING=true

# Reranker
export RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Device
export DEVICE=cpu  # or cuda, mps

# OpenAI (for LLM judge)
export OPENAI_API_KEY=sk-...
```

---

## Error Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad request (invalid parameters) |
| 500 | Internal server error |
| 503 | Service unavailable (not initialized) |
