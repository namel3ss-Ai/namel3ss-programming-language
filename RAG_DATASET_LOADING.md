# RAG Dataset Loading - Usage Examples

This document provides examples of how to use the production-grade dataset loading system for RAG indexing.

## Overview

The RAG CLI now supports:
- ✅ Real document loading from multiple sources (CSV, JSON, JSONL, inline, SQL, custom)
- ✅ Streaming for large datasets (no memory limits)
- ✅ Progress reporting with ETA and throughput
- ✅ Resumable indexing with checkpoints
- ✅ Metadata filtering
- ✅ Document limits
- ✅ Configurable field mappings

## Basic Usage

### 1. CSV Dataset

```n3
app my_docs {
    dataset "product_docs" {
        source_type: "csv"
        source: "data/products.csv"
        metadata: {
            content_field: "description"
            id_field: "product_id"
            metadata_fields: ["name", "category", "price"]
        }
    }
    
    index docs_index {
        source_dataset: "product_docs"
        embedding_model: "text-embedding-3-small"
        chunk_size: 512
        overlap: 64
        backend: "pgvector"
    }
}
```

**Build the index:**
```bash
namel3ss build-index my_app.n3 docs_index --verbose
```

### 2. JSON/JSONL Dataset

```n3
dataset "articles" {
    source_type: "jsonl"  # or "json" for array format
    source: "data/articles.jsonl"
    metadata: {
        content_field: "body"
        id_field: "article_id"
        metadata_fields: ["title", "author", "published_date", "tags"]
    }
}
```

### 3. Inline Dataset (for testing)

```n3
dataset "faq" {
    source_type: "inline"
    source: ""
    metadata: {
        content_field: "answer"
        id_field: "question_id"
        records: [
            {
                question_id: "q1"
                question: "What is Namel3ss?"
                answer: "Namel3ss is an AI-native programming language..."
                category: "general"
            }
            {
                question_id: "q2"
                question: "How do I build an index?"
                answer: "Use the build-index command..."
                category: "howto"
            }
        ]
    }
}
```

### 4. Database Dataset

```n3
dataset "user_tickets" {
    source_type: "sql"
    source: "SELECT ticket_id, subject, description, tags, created_at FROM support_tickets WHERE status = 'resolved'"
    metadata: {
        content_field: "description"
        id_field: "ticket_id"
        metadata_fields: ["subject", "tags", "created_at"]
        connector: "default"
        query_params: {}
    }
}
```

## CLI Options

### Limit Number of Documents
```bash
namel3ss build-index app.n3 my_index --max-documents 1000
# or short form:
namel3ss build-index app.n3 my_index -n 1000
```

### Filter by Metadata
```bash
# Single filter
namel3ss build-index app.n3 my_index --filter category=support

# Multiple filters
namel3ss build-index app.n3 my_index --filter category=support --filter lang=en --filter priority=high
```

### Resume from Checkpoint
```bash
# Start indexing
namel3ss build-index app.n3 my_index --verbose

# If interrupted, resume:
namel3ss build-index app.n3 my_index --resume --verbose
```

### Force Rebuild
```bash
# Delete checkpoint and rebuild from scratch
namel3ss build-index app.n3 my_index --force-rebuild --verbose
```

### Verbose Output with Progress
```bash
namel3ss build-index app.n3 my_index --verbose

# Output includes:
# - Documents processed per second
# - Chunks created per second
# - Total embedding tokens used
# - Estimated time remaining (if tqdm installed)
```

## Advanced: Custom Field Mappings

### CSV with Custom Delimiter
```n3
dataset "tsv_data" {
    source_type: "csv"
    source: "data/records.tsv"
    metadata: {
        content_field: "text"
        id_field: "id"
        delimiter: "\t"  # Tab-separated
    }
}
```

### Selecting Specific Metadata Fields
```n3
dataset "selective_metadata" {
    source_type: "json"
    source: "data/articles.json"
    metadata: {
        content_field: "body"
        id_field: "id"
        # Only include these fields in metadata (not all fields)
        metadata_fields: ["title", "author", "published_date"]
    }
}
```

### Auto-generated IDs
If you don't specify `id_field`, IDs will be auto-generated as `{dataset_name}_{counter}`:
```n3
dataset "no_ids" {
    source_type: "json"
    source: "data/documents.json"
    metadata: {
        content_field: "text"
        # id_field not specified - will auto-generate
    }
}
```

## Advanced: Custom Loaders

For specialized data sources, implement a custom loader:

```python
# my_project/custom_loader.py
from namel3ss.rag.loaders import BaseDatasetLoader, LoadedDocument
from typing import AsyncIterator, Optional, Dict, Any

class S3DatasetLoader(BaseDatasetLoader):
    async def iter_documents(
        self,
        *,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
    ) -> AsyncIterator[LoadedDocument]:
        # Your S3 loading logic here
        bucket = self.config["bucket"]
        prefix = self.config["prefix"]
        
        # Stream objects from S3
        for obj in s3_client.list_objects(bucket, prefix):
            content = s3_client.get_object(bucket, obj.key)
            doc = self._create_document({
                "id": obj.key,
                "content": content,
                "metadata": {"size": obj.size, "modified": obj.last_modified}
            })
            if doc:
                yield doc
```

Use it in your dataset:
```n3
dataset "s3_docs" {
    source_type: "custom"
    source: ""
    connector: {
        connector_type: "my_project.custom_loader.S3DatasetLoader"
        options: {
            bucket: "my-docs-bucket"
            prefix: "documents/"
            content_field: "content"
        }
    }
}
```

## Progress Reporting

Install `tqdm` for enhanced progress bars:
```bash
pip install tqdm
```

Progress output includes:
- **Documents/sec**: Throughput of document processing
- **Chunks/sec**: Throughput of chunk creation
- **Tokens used**: Total embedding API tokens consumed
- **ETA**: Estimated time remaining (when using tqdm)

Example output:
```
Loading program from app.n3...

Building index 'docs_index':
  Source dataset: product_docs
  Embedding model: text-embedding-3-small
  Chunk size: 512
  Overlap: 64
  Backend: pgvector

Vector backend initialized
Dataset loader created for 'product_docs'
Loading and indexing documents...

Indexing documents: 100%|██████████| 1000/1000 [00:45<00:00, 22.1 docs/s]

📊 Final: 1000 docs, 5234 chunks, 2,451,023 tokens in 45.3s (22.1 docs/s, 115.5 chunks/s)

✓ Index 'docs_index' built successfully!
  Documents processed: 1000
  Chunks created: 5234
  Chunks indexed: 5234
  Embedding tokens: 2,451,023
  Build time: 45.32s
```

## Resumable Indexing

State is automatically saved in `~/.namel3ss/index_states/{index_name}_{dataset_name}.json`

The state tracks:
- ✅ All processed document IDs
- ✅ Total documents, chunks, and tokens
- ✅ Start and update timestamps
- ✅ Completion status
- ✅ Index configuration (model, chunk size, etc.)

To resume after interruption:
```bash
namel3ss build-index app.n3 my_index --resume
```

To force rebuild (ignoring previous state):
```bash
namel3ss build-index app.n3 my_index --force-rebuild
```

## Error Handling

The system gracefully handles errors:
- **Missing files**: Logs error, returns no documents
- **Malformed records**: Logs error, skips record, continues
- **Empty content**: Skips document with warning
- **Invalid JSON**: Logs error line number, skips line
- **Database errors**: Logs error, exits with clear message

All errors are logged with context (file path, line number, document ID, etc.)

## Best Practices

1. **Start small**: Test with `--max-documents 100` first
2. **Use filters**: Index subsets with `--filter` for faster iteration
3. **Resume builds**: Use `--resume` for large datasets
4. **Monitor progress**: Use `--verbose` to track throughput
5. **Check logs**: Review error messages for data quality issues
6. **Optimize chunks**: Adjust `chunk_size` and `overlap` based on content
7. **Batch size**: Default batch size (32) works well for most cases
8. **Clean restarts**: Use `--force-rebuild` when changing index config

## Troubleshooting

### "Dataset not found"
Ensure dataset name matches exactly (case-sensitive):
```bash
namel3ss build-index app.n3 my_index --dataset my_dataset
```

### "File not found"
Use absolute paths or paths relative to current directory:
```n3
source: "/absolute/path/to/data.csv"
# or
source: "./relative/path/to/data.csv"
```

### "No documents loaded"
Check:
- File exists and is readable
- Content field name is correct
- Records have non-empty content field
- Filters aren't too restrictive

### Performance issues
- Install `tqdm` for progress tracking
- Reduce batch size if memory constrained
- Use `--max-documents` for testing
- Check network/database latency for remote sources

## Migration from Placeholder

If you have existing code using the old placeholder:
```python
# OLD (placeholder):
documents = [{"id": "doc_1", "content": "...", "metadata": {}}]

# NEW (production):
# Just define your dataset in .n3 file and run CLI
# No code changes needed!
```

The CLI now automatically:
1. Loads the dataset definition
2. Creates the appropriate loader
3. Streams documents efficiently
4. Handles all edge cases
5. Reports progress
6. Saves checkpoints

No more TODOs, no more placeholders, no more warnings! 🎉
