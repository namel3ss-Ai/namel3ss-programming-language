# RAG (Retrieval-Augmented Generation) in Namel3ss

Namel3ss provides first-class support for RAG workflows with declarative syntax for vector indices and retrieval pipelines.

## Overview

RAG enhances LLM responses by retrieving relevant context from your data before generating answers. Namel3ss makes this pattern simple with two main constructs:

1. **`index`** - Define vector indices for your documents
2. **`rag_pipeline`** - Configure retrieval and ranking strategies

## Quick Start

```n3
app "Documentation Assistant"

// Define an LLM for responses
llm gpt4:
    provider: openai
    model: gpt-4
    temperature: 0.7

// Create a vector index from your documentation dataset
index docs_index:
    source_dataset: documentation
    embedding_model: text-embedding-3-small
    chunk_size: 512
    overlap: 64
    backend: pgvector
    table_name: doc_embeddings

// Configure retrieval pipeline
rag_pipeline support_rag:
    query_encoder: text-embedding-3-small
    index: docs_index
    top_k: 5
    distance_metric: cosine
```

## Index Configuration

### Required Fields

- **`source_dataset`**: Name of the dataset containing your documents
- **`embedding_model`**: Model for generating embeddings (e.g., `text-embedding-3-small`, `text-embedding-ada-002`)
- **`chunk_size`**: Maximum characters per chunk (recommended: 512-1024)
- **`overlap`**: Character overlap between chunks (recommended: 50-128)
- **`backend`**: Vector database backend (`pgvector`, `qdrant`, `weaviate`, `chroma`)

### Optional Fields

- **`namespace`**: Namespace for organizing indices (backend-specific)
- **`collection`**: Collection name (for Qdrant, Weaviate)
- **`table_name`**: Table name (for pgvector)
- **`metadata_fields`**: List of metadata fields to extract and store

### Example with All Options

```n3
index knowledge_base:
    source_dataset: wiki_docs
    embedding_model: text-embedding-3-large
    chunk_size: 1024
    overlap: 128
    backend: pgvector
    table_name: kb_embeddings
    metadata_fields: [category, author, date]
```

## RAG Pipeline Configuration

### Required Fields

- **`query_encoder`**: Embedding model for queries (usually same as index)
- **`index`**: Name of the index to query
- **`top_k`**: Number of documents to retrieve (recommended: 3-10)
- **`distance_metric`**: Similarity metric (`cosine`, `euclidean`, `dot`)

### Optional Fields

- **`reranker`**: Reranker model for improving relevance (e.g., `cross-encoder-ms-marco`)
- **`filters`**: Metadata filters to narrow search

### Example with Reranking

```n3
rag_pipeline advanced_retrieval:
    query_encoder: text-embedding-3-small
    index: docs_index
    top_k: 20
    distance_metric: cosine
    reranker: cross-encoder-ms-marco
    filters: {category: "technical"}
```

## Vector Database Setup

### PostgreSQL + pgvector

1. Install pgvector extension:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

2. Set connection string:
```bash
export NAMEL3SS_PG_DSN="postgresql://user:password@localhost:5432/mydb"
```

3. The index will auto-create tables on first build.

### Recommended Settings

- **For general docs**: `chunk_size: 512`, `overlap: 64`
- **For long-form content**: `chunk_size: 1024`, `overlap: 128`
- **For code snippets**: `chunk_size: 256`, `overlap: 32`
- **For Q&A pairs**: `chunk_size: 512`, `overlap: 0` (no overlap needed)

## Building Indices

Use the CLI to build indices from your datasets:

```bash
# Build a specific index
namel3ss build-index app.ai docs_index --verbose

# Override the source dataset
namel3ss build-index app.ai docs_index --dataset alternative_docs
```

The build process:
1. Loads documents from the dataset
2. Chunks text with configured overlap
3. Generates embeddings (tracks token usage)
4. Stores vectors in the backend
5. Reports statistics and errors

## Using RAG in Chains

RAG pipelines integrate seamlessly with Namel3ss chains using the flow syntax:

```n3
// Define a prompt that uses retrieved context
prompt doc_qa:
    input:
        - question: text (required)
        - context: text (required)
    
    template: |
        Context: {context}
        Question: {question}
        Answer based on the context above.
    
    output:
        - answer: text

// Chain that combines RAG retrieval with LLM
chain rag_qa_chain:
    input -> rag doc_retrieval -> prompt doc_qa -> llm gpt4
```

The `rag` step in a chain:
- Takes the input query and performs vector similarity search
- Returns a list of `ScoredDocument` objects with content, metadata, and relevance scores
- Can be followed by prompts that format the retrieved context
- Results are automatically passed to the next step in the chain

**Retrieved Documents Structure:**

Each document returned by a RAG step contains:
- `content`: The text chunk
- `score`: Similarity score (0.0 to 1.0)
- `metadata`: Custom fields from your index
- `id`: Unique document identifier

You can access these in templates using standard Namel3ss expressions.


## Best Practices

### Chunk Size Selection

- **Too small**: Loss of context, more API calls
- **Too large**: Irrelevant content, exceeds model context
- **Sweet spot**: 512-1024 characters for most use cases

### Overlap Strategy

- Use overlap to prevent context loss at boundaries
- Recommended: 10-20% of chunk_size
- For critical content: 25% overlap

### Top-K Selection

- Start with `top_k: 5` and adjust based on results
- More is not always better (noise increases)
- Use rerankers to retrieve more, then filter

### Distance Metrics

- **`cosine`**: Best for most cases, normalized similarity
- **`euclidean`**: When absolute distances matter
- **`dot`**: Fast, good for large-scale retrieval

### Metadata Filtering

Use filters to narrow search scope:

```n3
rag_pipeline filtered_search:
    query_encoder: text-embedding-3-small
    index: docs_index
    top_k: 5
    filters: {
        category: "api-docs",
        version: "v2"
    }
```

## Embedding Models

### OpenAI (Production)

Requires `OPENAI_API_KEY` environment variable.

- `text-embedding-3-small`: Fast, cost-effective (1536 dimensions)
- `text-embedding-3-large`: Higher quality (3072 dimensions)
- `text-embedding-ada-002`: Legacy, still supported (1536 dimensions)

### Sentence Transformers (Local)

No API key needed, runs locally.

- `all-MiniLM-L6-v2`: Fast, good quality (384 dimensions)
- `all-mpnet-base-v2`: Higher quality (768 dimensions)

## Troubleshooting

### "Dataset not found" Error

Ensure the `source_dataset` exists in your app definition.

### "API key not found" Error

Set the appropriate environment variable:
```bash
export OPENAI_API_KEY="your-api-key"
```

### "Overlap must be less than chunk_size" Error

Reduce overlap or increase chunk_size. Overlap should be 10-25% of chunk_size.

### Poor Retrieval Quality

1. Try increasing `top_k`
2. Add a reranker model
3. Adjust chunk_size (larger for more context)
4. Use metadata filters to narrow scope

### Slow Index Building

1. Increase batch_size in build_index()
2. Use local embeddings for development
3. Enable connection pooling for vector backend

## Performance Tips

- **Batch embed**: Process multiple documents at once
- **Cache embeddings**: Store in backend, avoid recomputing
- **Use async**: All RAG operations are async-first
- **Monitor tokens**: Track embedding API usage
- **Index incrementally**: Build indices in stages for large datasets

## Examples

### Complete RAG Application

Here's a full example showing all components working together:

```n3
app "Documentation Assistant"

// Dataset of documentation
dataset documentation:
    source: "docs/*.md"
    format: markdown

// LLM for generating answers
llm gpt4:
    provider: openai
    model: gpt-4
    temperature: 0.7

// Vector index
index docs_index:
    source_dataset: documentation
    embedding_model: text-embedding-3-small
    chunk_size: 512
    overlap: 64
    backend: pgvector
    table_name: doc_embeddings

// RAG pipeline
rag_pipeline doc_retrieval:
    query_encoder: text-embedding-3-small
    index: docs_index
    top_k: 5
    distance_metric: cosine

// Prompt template
prompt qa_prompt:
    input:
        - question: text (required)
        - context: text (required)
    
    template: |
        You are a helpful documentation assistant.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer based only on the context above.
    
    output:
        - answer: text

// Chain combining RAG + LLM
chain rag_qa:
    input -> rag doc_retrieval -> prompt qa_prompt -> llm gpt4

// API endpoint
page "Ask Documentation" at "/ask":
    show form:
        fields: question
        on submit:
            run chain rag_qa with:
                question = form.question
            show response chain.answer
```

**Building the index:**

```bash
# Set up environment
export OPENAI_API_KEY="your-key"
export NAMEL3SS_PG_DSN="postgresql://user:pass@localhost/dbname"

# Build the index
namel3ss build-index app.ai docs_index --verbose
```

See `examples/rag_demo.ai` for the reference implementation.

