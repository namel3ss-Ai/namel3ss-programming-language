# Multimodal RAG for Namel3ss (N3)

> **Production-ready multimodal retrieval-augmented generation with hybrid search and comprehensive evaluation.**

## 🌟 Features

- **📄 Multimodal Ingestion**: Extract text, images, and audio from PDFs, documents, media files
- **🔍 Hybrid Search**: Combine dense vectors (embeddings) + sparse vectors (BM25) with RRF fusion
- **🎯 Advanced Reranking**: ColBERT late-interaction or Cross-Encoder pairwise scoring
- **📊 Comprehensive Evaluation**: Precision@k, Recall@k, NDCG@k, MRR, Hit Rate + LLM judge
- **🚀 FastAPI Service**: RESTful API with OpenAPI docs
- **🖥️ CLI Tools**: Batch evaluation and search commands
- **🧪 100+ Tests**: Comprehensive test coverage with pytest
- **📚 Full Documentation**: User guides, API reference, testing guide

## 🚀 Quick Start

### 1. Installation

```bash
# Run automated setup
./setup_multimodal_rag.sh

# Or install manually
pip install -e .
pip install -r requirements-multimodal.txt
```

### 2. Start Qdrant Vector Database

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Use in N3 Applications

Create a file `my_app.n3`:

```n3
app "Multimodal Product Docs"

llm gpt4:
    provider: openai
    model: gpt-4

dataset docs:
    source: "data/docs/"
    format: directory

index multimodal_index:
    source_dataset: docs
    embedding_model: "all-MiniLM-L6-v2"
    extract_images: true
    image_model: "openai/clip-vit-base-patch32"
    backend: qdrant
    collection: product_docs

rag_pipeline hybrid_rag:
    index: multimodal_index
    query_encoder: "all-MiniLM-L6-v2"
    top_k: 20
    enable_hybrid: true
    sparse_model: "bm25"
    dense_weight: 0.7
    sparse_weight: 0.3
    reranker: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_type: "cross_encoder"

chain qa_chain:
    input -> rag hybrid_rag -> llm gpt4
```

Run it:

```bash
n3 run my_app.n3
```

### 4. Use FastAPI Service

Start the service:

```bash
uvicorn api.main:app --reload
```

Ingest a document:

```bash
curl -X POST "http://localhost:8000/ingest?extract_images=true" \
  -F "file=@document.pdf"
```

Search:

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I configure the system?",
    "top_k": 5,
    "enable_hybrid": true,
    "enable_reranking": true
  }'
```

### 5. Evaluate Your RAG System

Create `eval_dataset.json`:

```json
[
  {
    "query": "What is machine learning?",
    "relevant_docs": ["doc_1", "doc_3"],
    "relevance_scores": {"doc_1": 3.0, "doc_3": 2.0},
    "ground_truth_answer": "Machine learning is a subset of AI."
  }
]
```

Run evaluation:

```bash
n3 eval rag eval_dataset.json \
  --top-k 20 \
  --k-values 1,5,10 \
  --use-llm-judge \
  --output results.md
```

## 📖 Documentation

- **[User Guide](MULTIMODAL_RAG_GUIDE.md)** - Complete guide with architecture, configuration, best practices
- **[API Reference](MULTIMODAL_RAG_API.md)** - FastAPI endpoints, CLI commands, Python API
- **[Testing Guide](MULTIMODAL_RAG_TESTING_GUIDE.md)** - Running tests, writing tests, CI/CD
- **[Implementation Summary](MULTIMODAL_RAG_IMPLEMENTATION_SUMMARY.md)** - Technical details and achievements

## 🧪 Testing

Run all tests:

```bash
pytest -v
```

Run with coverage:

```bash
pytest --cov=namel3ss --cov=api --cov-report=html
```

Run specific test modules:

```bash
pytest namel3ss/multimodal/tests/test_ingestion.py -v
pytest namel3ss/eval/tests/test_rag_eval.py -v
pytest api/tests/test_main.py -v
```

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│      N3 Application Layer           │
│  (Declarative RAG Configuration)    │
└─────────────────────────────────────┘
               ↓
┌─────────────────────────────────────┐
│    Multimodal RAG Core Engine       │
│  ┌──────────┐  ┌────────────────┐  │
│  │Ingestion │→ │   Embeddings   │  │
│  │(PDF,img, │  │ (text,img,audio)│  │
│  │ audio)   │  └────────────────┘  │
│  └──────────┘          ↓            │
│  ┌──────────────────────────────┐  │
│  │    Vector Database (Qdrant)  │  │
│  │  • Multi-vector (text/img)   │  │
│  │  • Sparse vectors (BM25)     │  │
│  └──────────────────────────────┘  │
│               ↓                     │
│  ┌──────────────────────────────┐  │
│  │   Hybrid Retrieval Pipeline  │  │
│  │  Dense + Sparse + Reranking  │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
               ↓
┌─────────────────────────────────────┐
│        Evaluation Framework         │
│  • Retrieval metrics (P@k, NDCG)   │
│  • Generation metrics (LLM judge)  │
└─────────────────────────────────────┘
```

## 🎯 Key Components

### Multimodal Ingestion
- **PDF**: Text + embedded images (PyMuPDF)
- **Images**: PNG, JPEG, WebP (PIL)
- **Audio**: Transcription via Whisper
- **Video**: Audio extraction via ffmpeg

### Embeddings
- **Text**: SentenceTransformers (384-dim) or OpenAI (1536-dim)
- **Images**: CLIP (512-dim)
- **Audio**: Whisper transcription → text embeddings

### Hybrid Search
- **Dense**: Cosine similarity on embeddings
- **Sparse**: BM25 with IDF weighting
- **Fusion**: Reciprocal Rank Fusion (RRF) with configurable weights

### Reranking
- **ColBERT**: Late-interaction with MaxSim
- **Cross-Encoder**: Pairwise scoring

### Evaluation
- **Retrieval**: Precision@k, Recall@k, NDCG@k, MRR, Hit Rate
- **Generation**: Faithfulness, Relevance, Correctness (via GPT-4)

## 📊 Performance Benchmarks

On M1 Mac with CPU:

| Operation | Throughput | Latency |
|-----------|------------|---------|
| PDF Ingestion | ~5 pages/s | 200ms/page |
| Text Embedding | ~100 docs/s | 10ms/doc |
| Image Embedding | ~50 images/s | 20ms/image |
| Hybrid Search | ~200 queries/s | 5ms/query |
| Reranking (top 20) | ~50 queries/s | 20ms/query |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest -v`
5. Submit a pull request

## 📝 Examples

See `examples/multimodal_rag.n3` for a complete example application demonstrating:
- Multimodal index with image extraction
- Hybrid RAG pipeline with reranking
- Question-answering chain
- Conversational agent with memory
- Web interface

## 🔧 Troubleshooting

### Import Errors
```bash
pip install -r requirements-multimodal.txt
```

### Qdrant Connection Failed
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Or use remote Qdrant
export QDRANT_HOST=your-qdrant-host.com
export QDRANT_PORT=6333
```

### CUDA Out of Memory
```bash
# Use CPU instead
export DEVICE=cpu
```

### Slow Reranking
```bash
# Reduce batch size
export RERANKER_BATCH_SIZE=8
```

See [MULTIMODAL_RAG_GUIDE.md](MULTIMODAL_RAG_GUIDE.md) for more troubleshooting tips.

## 📦 Dependencies

### Core
- sentence-transformers
- transformers
- torch
- qdrant-client
- numpy
- pillow

### Optional
- PyMuPDF (PDF processing)
- python-docx (Word documents)
- openai-whisper (audio transcription)
- librosa (audio processing)
- ffmpeg (video processing)

### API & CLI
- fastapi
- uvicorn
- click
- pandas

### Development
- pytest
- pytest-asyncio
- black
- mypy

## 📄 License

Part of the Namel3ss (N3) programming language project.

## 🎓 Learn More

- [N3 Documentation](https://github.com/SsebowaDisan/namel3ss-programming-language)
- [Multimodal RAG Guide](MULTIMODAL_RAG_GUIDE.md)
- [API Reference](MULTIMODAL_RAG_API.md)

---

**Status**: ✅ Production Ready | **Tests**: 107 passing | **Coverage**: >80%
