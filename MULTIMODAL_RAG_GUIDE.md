# Multimodal RAG System - Comprehensive Documentation

## Overview

The Namel3ss Multimodal RAG system provides production-ready support for:

- **Multimodal Ingestion**: Extract text, images, and audio from documents (PDF, Word, images, audio files)
- **Multimodal Embeddings**: Text (SentenceTransformers/OpenAI), Images (CLIP), Audio (Whisper → text embeddings)
- **Hybrid Search**: Dense vector search + sparse BM25 retrieval with reciprocal rank fusion
- **Advanced Reranking**: ColBERTv2 late-interaction or Cross-Encoder reranking
- **Vector Database**: Qdrant with multi-vector and sparse vector support
- **Evaluation**: Precision@k, Recall@k, NDCG@k, MRR, Hit Rate, LLM-based faithfulness metrics

## Installation

```bash
# Core dependencies
pip install sentence-transformers transformers torch pillow

# PDF processing
pip install PyMuPDF

# Audio processing  
pip install whisper librosa soundfile

# Vector database
pip install qdrant-client

# API service
pip install fastapi uvicorn

# CLI and evaluation
pip install click pandas

# Optional: Word document support
pip install python-docx

# Optional: video processing
# Requires ffmpeg: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)
```

## Quick Start

### 1. N3 Language Syntax

Define a multimodal RAG pipeline in your `.n3` file:

```n3
app "Multimodal Documentation Assistant"

// Dataset of documents
dataset product_docs:
    source: "docs/"
    format: directory

// Multimodal index with image/audio extraction
index docs_index:
    source_dataset: product_docs
    embedding_model: "all-MiniLM-L6-v2"
    chunk_size: 512
    overlap: 64
    backend: qdrant
    collection: multimodal_docs
    
    # Multimodal features
    extract_images: true
    extract_audio: false
    image_model: "openai/clip-vit-base-patch32"
    audio_model: "openai/whisper-base"

// Hybrid RAG pipeline with reranking
rag_pipeline multimodal_rag:
    query_encoder: "all-MiniLM-L6-v2"
    index: docs_index
    top_k: 20
    
    # Hybrid search
    enable_hybrid: true
    sparse_model: "bm25"
    dense_weight: 0.7
    sparse_weight: 0.3
    
    # Reranking
    reranker: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_type: "cross_encoder"  # or "colbert"
    distance_metric: cosine
```

### 2. Using the FastAPI Service

#### Start the API Server

```bash
# Set environment variables
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export COLLECTION_NAME=multimodal_docs
export DEVICE=cpu  # or cuda/mps

# Optional: Enable features
export EXTRACT_IMAGES=true
export EXTRACT_AUDIO=false
export ENABLE_HYBRID=true
export ENABLE_RERANKING=true

# Start server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

#### Ingest Documents

```bash
# Upload a PDF with images
curl -X POST "http://localhost:8000/ingest?extract_images=true" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# Response
{
  "document_id": "document.pdf",
  "num_chunks": 45,
  "modalities": ["text", "image"],
  "status": "success"
}
```

#### Search Documents

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I configure authentication?",
    "top_k": 10,
    "rerank_top_k": 5,
    "enable_hybrid": true,
    "enable_reranking": true,
    "modality": "text"
  }'
```

### 3. Using the CLI for Evaluation

#### Prepare Evaluation Dataset

Create `eval_dataset.json`:

```json
[
  {
    "query": "What is machine learning?",
    "relevant_docs": ["doc_123", "doc_456"],
    "relevance_scores": {
      "doc_123": 1.0,
      "doc_456": 0.8
    },
    "ground_truth_answer": "Machine learning is a subset of AI..."
  }
]
```

#### Run Evaluation

```bash
# Basic retrieval evaluation
n3 eval rag eval_dataset.json \
  --collection multimodal_docs \
  --top-k 10 \
  --k-values 1,3,5,10 \
  --output results.md \
  --csv-output results.csv

# With LLM judge for answer quality
n3 eval rag eval_dataset.json \
  --use-llm-judge \
  --llm-model gpt-4 \
  --llm-api-key $OPENAI_API_KEY \
  --output results_with_judge.md
```

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ FastAPI      │  │ CLI Tool     │  │ N3 Compiler  │      │
│  │ Service      │  │ (Evaluation) │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                    Multimodal RAG Core                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          MultimodalIngester                           │  │
│  │  • PDF (PyMuPDF)  • Images (PIL)  • Audio (Whisper) │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │       MultimodalEmbeddingProvider                     │  │
│  │  • Text (SentenceTransformers/OpenAI)                │  │
│  │  • Images (CLIP)                                      │  │
│  │  • Audio (Whisper → Text embeddings)                 │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            HybridRetriever                            │  │
│  │  • Dense (vector search)                              │  │
│  │  • Sparse (BM25)                                      │  │
│  │  • Reranking (ColBERT/CrossEncoder)                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                   Vector Database (Qdrant)                   │
│  • Named vectors (text, image, audio)                        │
│  • Sparse vectors (BM25)                                     │
│  • Hybrid search with RRF                                    │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Ingestion**: Document → Multimodal extraction → Embeddings → Vector DB
2. **Retrieval**: Query → Dense + Sparse embeddings → Hybrid search → Reranking → Results
3. **Evaluation**: Queries → Retrieval → Metrics computation → Report

## Configuration

### Embedding Models

#### Text Embeddings

- **SentenceTransformers** (default): `all-MiniLM-L6-v2` (384 dim), `all-mpnet-base-v2` (768 dim)
- **OpenAI**: `text-embedding-3-small` (1536 dim), `text-embedding-3-large` (3072 dim)

#### Image Embeddings

- **CLIP**: `openai/clip-vit-base-patch32` (512 dim), `openai/clip-vit-large-patch14` (768 dim)
- **ImageBind**: Can be integrated via custom embedder

#### Audio Embeddings

- **Whisper**: `openai/whisper-base`, `openai/whisper-large`
- Audio is transcribed then embedded as text

### Vector Databases

#### Qdrant (Recommended for Multimodal)

```python
{
    "host": "localhost",
    "port": 6333,
    "collection_name": "multimodal_docs",
    "text_dimension": 384,
    "image_dimension": 512,
    "audio_dimension": 384,
    "enable_sparse": true,
}
```

#### PgVector (Text Only)

```python
{
    "dsn": "postgresql://user:pass@localhost/dbname",
    "table_name": "embeddings",
    "dimension": 384,
}
```

### Hybrid Search Weights

Adjust dense/sparse weights based on your use case:

- **Semantic-heavy** (technical docs): dense=0.8, sparse=0.2
- **Keyword-heavy** (product names): dense=0.5, sparse=0.5
- **Balanced** (general content): dense=0.7, sparse=0.3

### Reranker Models

#### Cross-Encoder (Fast, Good Quality)

- `cross-encoder/ms-marco-MiniLM-L-6-v2` (40M params)
- `cross-encoder/ms-marco-MiniLM-L-12-v2` (Better, slower)

#### ColBERTv2 (Best Quality, Slower)

- `colbert-ir/colbertv2.0`

## Evaluation Metrics

### Retrieval Metrics

- **Precision@K**: Fraction of retrieved docs that are relevant
- **Recall@K**: Fraction of relevant docs that are retrieved  
- **NDCG@K**: Normalized discounted cumulative gain (position-aware)
- **MRR**: Mean reciprocal rank of first relevant doc
- **Hit Rate**: Whether any relevant doc was retrieved

### Generation Metrics (LLM Judge)

- **Faithfulness**: Is answer grounded in retrieved contexts?
- **Relevance**: Does answer address the query?
- **Correctness**: Is answer factually correct vs ground truth?

## Best Practices

### 1. Chunking Strategy

```python
chunk_size: 512      # Good for most content
overlap: 64          # 12.5% overlap prevents context loss
```

For code: smaller chunks (256)
For narrative: larger chunks (1024)

### 2. Top-K Selection

```
Retrieval top_k: 20    # Get enough candidates
Reranking top_k: 5     # Return best results
```

### 3. Hybrid Search Tuning

Start with defaults (0.7/0.3), then:
- Monitor precision/recall on eval set
- Increase sparse weight if missing keyword matches
- Increase dense weight if missing semantic matches

### 4. Performance Optimization

- Use `batch_size=32` for embeddings
- Enable GPU: `device="cuda"` (10-50x faster)
- Cache embeddings for repeated queries
- Use smaller models for CPU deployment

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Missing PyMuPDF
pip install PyMuPDF

# Missing Whisper
pip install openai-whisper

# Missing transformers
pip install transformers torch
```

**Qdrant Connection**
```bash
# Start Qdrant with Docker
docker run -p 6333:6333 qdrant/qdrant

# Or install locally
# See: https://qdrant.tech/documentation/quick-start/
```

**CUDA Out of Memory**
```python
# Reduce batch size
config.batch_size = 16  # default: 32

# Use smaller models
config.image_model = "openai/clip-vit-base-patch32"  # not large

# Or use CPU
config.device = "cpu"
```

**Slow Reranking**
```python
# Use smaller reranker
reranker: "cross-encoder/ms-marco-MiniLM-L-6-v2"  # not L-12

# Or disable reranking
enable_reranking: false
```

## Examples

See `examples/multimodal_rag.n3` for complete working example.

## Testing

```bash
# Run unit tests
pytest namel3ss/multimodal/tests/

# Run integration tests  
pytest namel3ss/eval/tests/

# Test API
pytest api/tests/
```

## Docker Deployment

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Expose API port
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Performance Benchmarks

On M1 Mac (8-core):

| Operation | CPU Time | GPU Time |
|-----------|----------|----------|
| Text embedding (batch=32) | 120ms | 15ms |
| Image embedding (batch=8) | 800ms | 80ms |
| Audio transcription | 2s | 400ms |
| BM25 encoding | 50ms | N/A |
| Cross-encoder reranking (10 docs) | 300ms | 40ms |
| ColBERT reranking (10 docs) | 800ms | 100ms |

## License

MIT License - see LICENSE file

## Contributing

See CONTRIBUTING.md

## Support

- GitHub Issues: https://github.com/SsebowaDisan/namel3ss-programming-language/issues
- Documentation: https://namel3ss.readthedocs.io
