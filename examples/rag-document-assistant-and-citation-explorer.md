# RAG Document Assistant & Citation Explorer

> **Production-grade RAG system demonstrating document ingestion, vector search, citation tracking, and tool inspection**

## Overview

The **RAG Document Assistant & Citation Explorer** is a complete production example showcasing how to build a Retrieval-Augmented Generation (RAG) system using Namel3ss. This application enables users to:

- 📚 **Upload and manage document collections** (knowledge bases)
- 💬 **Chat with an AI assistant** that answers based solely on uploaded documents
- 🔍 **Inspect citations** to see exactly where answers came from
- 🛠️ **View tool calls** to understand retrieval, reranking, and summarization operations
- 📊 **Compare answers** across different configurations or time periods using diff views
- 📈 **Analyze query patterns** with built-in analytics

This example is designed for:
- **AI Engineers** building RAG systems in production
- **Knowledge Management Teams** creating internal Q&A systems
- **Product Teams** evaluating RAG architectures
- **Namel3ss Developers** learning best practices for AI-native applications

## Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Library    │  │  Assistant   │  │   History    │         │
│  │   (Upload)   │  │   (Chat)     │  │  (Compare)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     RAG Pipeline                                │
│                                                                  │
│  1. Document Ingestion → 2. Chunking → 3. Embedding            │
│                   ↓                                              │
│  4. Vector Store (pgvector) ← 5. Similarity Search             │
│                   ↓                                              │
│  6. Reranking → 7. Context Assembly → 8. LLM Generation        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Data Layer                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Collections  │  │  Documents   │  │    Chunks    │         │
│  │   (Metadata) │  │   (Files)    │  │ (Embeddings) │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │   Queries    │  │     Logs     │                            │
│  │  (History)   │  │  (Retrieval) │                            │
│  └──────────────┘  └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### RAG Pipeline Flow

```
User Question
     ↓
[1] Vector Search (search_collection tool)
     ├─→ Query Embedding
     ├─→ Similarity Search (top_k=10)
     └─→ Retrieve Chunks with Scores
     ↓
[2] Reranking (rerank_results tool)
     ├─→ Cross-Encoder Scoring
     └─→ Top 5 Most Relevant Chunks
     ↓
[3] Context Assembly
     ├─→ Format Chunks with Metadata
     ├─→ Add Citation Markers
     └─→ Optional Summarization
     ↓
[4] LLM Generation (rag_assistant agent)
     ├─→ Grounded Answer
     ├─→ Inline Citations [Doc: X, Page: Y]
     └─→ Confidence Indicators
     ↓
User receives Answer + Citations + Tool Call Logs
```

## Data Model

### Core Entities

#### Collections (Knowledge Bases)
```sql
CREATE TABLE collections (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    document_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    size_mb FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);
```

#### Documents
```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    collection_id INTEGER REFERENCES collections(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),  -- pdf, txt, docx, md
    size_bytes BIGINT,
    page_count INTEGER,
    chunk_count INTEGER DEFAULT 0,
    uploaded_at TIMESTAMP DEFAULT NOW(),
    processed_at TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending'  -- pending, processing, completed, failed
);
```

#### Chunks (with Embeddings)
```sql
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    collection_id INTEGER REFERENCES collections(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),  -- pgvector type
    token_count INTEGER,
    page_number INTEGER,
    section_title VARCHAR(255),
    chunk_index INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Vector similarity index
CREATE INDEX chunks_embedding_idx ON chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

#### Queries (Conversation History)
```sql
CREATE TABLE queries (
    id SERIAL PRIMARY KEY,
    collection_id INTEGER REFERENCES collections(id),
    session_id UUID,
    user_message TEXT NOT NULL,
    assistant_answer TEXT NOT NULL,
    citations JSONB,  -- Array of citation objects
    tool_calls JSONB,  -- Array of tool call records
    model VARCHAR(100),
    temperature FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    response_time_ms INTEGER
);
```

#### Retrieval Logs
```sql
CREATE TABLE retrieval_logs (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES queries(id),
    collection_id INTEGER REFERENCES collections(id),
    query_text TEXT,
    chunks_retrieved INTEGER,
    avg_score FLOAT,
    reranking_enabled BOOLEAN,
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Key Namel3ss Features Demonstrated

### 1. **file_upload** Component
- Multi-file upload with progress tracking
- Supported formats: PDF, TXT, DOCX, MD
- Max file size: 50MB
- Automatic processing pipeline trigger

**Location**: Document Library page (`/library`)

```n3
show file_upload:
    id: "document_uploader"
    accept: ".pdf,.txt,.docx,.md"
    multiple: true
    max_size: 50mb
    on_upload: "process_documents"
    show_progress: true
```

### 2. **chat_thread** Component
- Real-time conversation with RAG assistant
- Inline citation display
- Markdown rendering for formatted responses
- Session persistence

**Location**: Assistant Workspace page (`/assistant/:collection_id`)

```n3
show chat_thread:
    agent_id: "rag_assistant"
    collection_id: "{{collection_id}}"
    show_citations: true
    citation_format: "inline"
    enable_markdown: true
```

### 3. **tool_call_view** Component
- Displays all tool invocations during conversation
- Shows inputs, outputs, timing, and status
- Group by tool name for analysis
- Expandable detailed view

**Location**: Assistant Workspace → Tools tab

```n3
show tool_call_view:
    session_id: "{{chat_session_id}}"
    show_timing: true
    show_inputs: true
    show_outputs: true
    group_by: "tool_name"
```

**Displayed Tools**:
- `search_collection` - Vector search operations
- `rerank_results` - Reranking tool calls
- `summarize_chunks` - Context compression
- `inspect_chunk` - Chunk detail fetches

### 4. **log_view** Component
- Real-time streaming logs from RAG pipeline
- Filter by level (info, warning, error)
- Auto-scroll with tail mode
- Full-text search

**Location**: Assistant Workspace → Logs tab

```n3
show log_view:
    source: "rag_retrieval"
    session_id: "{{chat_session_id}}"
    max_entries: 500
    auto_scroll: true
    filter_levels: [info, warning, error]
```

### 5. **diff_view** Component
- Side-by-side comparison of answers
- Highlight changes between versions
- Compare across:
  - Different models
  - Different configurations
  - Different time periods
  - Document updates

**Location**: Query History → Compare tab

```n3
show diff_view:
    left_content: "{{query_a.answer}}"
    left_label: "{{query_a.created_at}} - {{query_a.model}}"
    right_content: "{{query_b.answer}}"
    right_label: "{{query_b.created_at}} - {{query_b.model}}"
    mode: "split"
    highlight_changes: true
```

### 6. RAG-Specific Features

#### Vector Store Configuration
```n3
vector_store "rag_vectors":
    backend: pgvector
    dimensions: 1536
    similarity: cosine
    index_type: ivfflat
```

#### Embedding Model
```n3
embedding_model "chunk_embedder":
    provider: openai
    model: text-embedding-3-small
    dimensions: 1536
```

#### Agent with RAG Instructions
```n3
agent "rag_assistant":
    model: gpt-4o
    temperature: 0.2
    tools: [search_collection, rerank_results, summarize_chunks, inspect_chunk]
    system_prompt: """
    You are a RAG Document Assistant...
    RULES:
    1. Always use search_collection to find information
    2. Never hallucinate
    3. Always provide citations [Doc: X, Page: Y]
    4. Indicate "I don't have information" when uncertain
    """
```

## How to Run

### Prerequisites

1. **PostgreSQL with pgvector extension**
   ```bash
   # Install pgvector
   CREATE EXTENSION vector;
   ```

2. **Python 3.10+**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**
   Create `.env` file:
   ```bash
   # OpenAI API (for embeddings and chat)
   OPENAI_API_KEY=your_openai_api_key
   
   # Database
   DATABASE_URL=postgresql://user:password@localhost:5432/rag_db
   
   # Optional: Reranking model
   COHERE_API_KEY=your_cohere_key  # For reranking
   ```

### Setup Database Schema

```bash
# Run migrations
psql $DATABASE_URL < schema/rag_schema.sql

# Or use Alembic
alembic upgrade head
```

### Generate Application

```bash
# Generate backend and frontend
namel3ss generate examples/rag-document-assistant.ai output/rag_app

# Navigate to output
cd output/rag_app
```

### Start Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`

API endpoints:
- `POST /api/collections` - Create collection
- `POST /api/documents/upload` - Upload documents
- `POST /api/chat` - Send message to RAG assistant
- `GET /api/queries/{id}` - Get query details
- `GET /api/tool-calls/{session_id}` - Get tool calls for session

### Start Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Initialize Sample Data (Optional)

```bash
# Load sample documents
python scripts/load_sample_data.py

# Creates:
# - "Product Documentation" collection
# - Sample PDF/text files
# - Pre-chunked and embedded
```

## How to Use

### 1. Create a Collection

1. Navigate to **Document Library** (`/library`)
2. Click **"New Collection"**
3. Enter collection name and description
4. Click **"Create"**

### 2. Upload Documents

1. Select a collection from the table
2. Use the **file_upload** component:
   - Click "Upload Documents" or drag & drop files
   - Supported: PDF, TXT, DOCX, MD (up to 50MB each)
3. Wait for processing (chunking + embedding)
4. See progress indicators and completion alerts

### 3. Chat with Documents

1. Click **"Open"** on a collection
2. Navigate to Assistant Workspace (`/assistant/:collection_id`)
3. Type question in chat input
4. Observe:
   - **Answer with inline citations** [Doc: name, Page: X]
   - **Citations panel** on the right showing source chunks
   - **Tool Calls tab** displaying:
     - Vector search with retrieved chunk IDs
     - Reranking scores
     - Any summarization operations
   - **Logs tab** showing retrieval pipeline details

**Example Questions**:
- "What are the main features of the product?"
- "How do I configure authentication?"
- "What are the system requirements?"

### 4. Inspect Citations

- **In Chat**: Citations appear as `[Doc: filename, Page: 5]`
- **Citations Panel**: Click any citation to:
  - View full chunk text
  - See relevance score
  - Jump to document page
  - Inspect chunk metadata

### 5. Review Tool Calls

Switch to **Tool Calls tab** to see:

- **search_collection**:
  ```json
  Input: { "collection_id": 1, "query": "authentication", "top_k": 10 }
  Output: { "results": [...], "scores": [0.89, 0.85, ...], "count": 10 }
  Timing: 145ms
  ```

- **rerank_results**:
  ```json
  Input: { "query": "authentication", "chunks": [...], "top_k": 5 }
  Output: { "reranked": [...], "scores": [0.92, 0.88, ...] }
  Timing: 78ms
  ```

### 6. Compare Answers Over Time

1. Navigate to **History** (`/history`)
2. Go to **"Compare Answers"** tab
3. Select two queries from dropdowns:
   - Same question, different models
   - Same question, before/after document update
   - Different configurations (temperature, top_k)
4. View **diff_view** showing:
   - Text differences highlighted
   - Side-by-side or unified view
   - Citation differences in comparison panel

### 7. Analyze Query Patterns

1. Go to **Analytics tab** in History page
2. View metrics:
   - Total queries over time
   - Average citations per answer
   - Response time trends
   - Tool call frequency
   - Most queried collections

### 8. Configure RAG Settings

1. Navigate to **Settings** (`/settings`)
2. Adjust parameters:
   - **Retrieval**: top_k, similarity threshold, reranking
   - **Chunking**: chunk size, overlap, strategy
   - **Models**: embedding model, chat model, temperature
3. Changes apply to new queries immediately

## Extension Ideas

### 1. Quality Evaluation
Add `evaluation_result` component to assess answer quality:

```n3
show evaluation_result:
    query_id: "{{current_query_id}}"
    metrics:
        - name: "Relevance"
          score: "{{eval.relevance_score}}"
        - name: "Citation Accuracy"
          score: "{{eval.citation_accuracy}}"
        - name: "Completeness"
          score: "{{eval.completeness}}"
```

**Implementation**:
- Add evaluation agent that checks answers against retrieved chunks
- Store evaluation scores in `evaluations` table
- Display trends over time

### 2. Multi-User Collections
Add access control and sharing:

```n3
dataset "collection_permissions" from table collection_permissions

tool "share_collection":
    input:
        collection_id: int
        user_email: text
        permission_level: text  # read, write, admin
```

**Implementation**:
- Add `user_id` and `permission_level` to collections
- Filter collections by user permissions
- Add sharing UI in collection settings

### 3. Advanced Chunking Strategies
Support multiple chunking approaches:

```n3
tool "chunk_document":
    input:
        document_id: int
        strategy: text  # fixed_size, semantic, by_section, sliding_window
        params: object
```

**Strategies**:
- **Semantic**: Use sentence transformers to identify semantic boundaries
- **By Section**: Parse document structure (headings, paragraphs)
- **Sliding Window**: Overlapping chunks with configurable stride

### 4. Citation Confidence Scores
Add confidence indicators to citations:

```n3
show data_list:
    items: "{{citations}}"
    item_template:
        title: "{{item.document_name}}"
        badge:
            text: "{{item.confidence}}"
            color: "{{item.confidence > 0.8 ? 'green' : 'yellow'}}"
```

**Implementation**:
- Calculate confidence from retrieval + reranking scores
- Use LLM to verify citation accuracy
- Display confidence badges in UI

### 5. Document Version Control
Track document changes and re-embed on updates:

```n3
dataset "document_versions" from table document_versions

tool "diff_document_versions":
    input:
        document_id: int
        version_a: int
        version_b: int
    output:
        diff: text
        chunks_added: list
        chunks_removed: list
```

**Implementation**:
- Store document versions with timestamps
- Re-chunk and re-embed on update
- Compare answers before/after updates using `diff_view`

### 6. Hybrid Search (Vector + Keyword)
Combine vector search with BM25 keyword search:

```n3
tool "hybrid_search":
    input:
        collection_id: int
        query: text
        alpha: float = 0.7  # Weight for vector vs keyword
    output:
        results: list
        vector_scores: list
        keyword_scores: list
        combined_scores: list
```

**Implementation**:
- Run parallel vector and BM25 searches
- Combine scores with weighted average
- Compare hybrid vs vector-only in analytics

### 7. Auto-Tagging and Categorization
Automatically tag documents and chunks:

```n3
agent "document_tagger":
    model: gpt-4o-mini
    tools: [extract_entities, classify_content]
    system_prompt: "Extract tags, entities, and categories from documents"
```

**Implementation**:
- Run tagging agent on document upload
- Store tags in document metadata
- Enable tag-based filtering in library

### 8. Export and Sharing
Export conversations and citations:

```n3
tool "export_conversation":
    input:
        session_id: uuid
        format: text  # pdf, markdown, json
    output:
        file_url: text
```

**Implementation**:
- Generate PDF with questions, answers, and citations
- Include source document excerpts
- Add download button in navbar

## Testing

### Parser Tests
Verify the RAG app parses correctly:

```python
def test_parse_rag_app():
    parser = Parser(open('examples/rag-document-assistant.ai').read())
    module = parser.parse()
    app = module.body[0]
    
    assert app.name == "RAG Document Assistant"
    assert len(app.datasets) == 5  # collections, documents, chunks, queries, logs
    assert len(app.pages) == 4  # library, assistant, history, settings
```

### IR Generation Tests
Confirm IR contains RAG components:

```python
def test_rag_ir_generation():
    ir = build_frontend_ir(app)
    
    # Verify file_upload component
    library_page = next(p for p in ir.pages if p.route == "/library")
    assert any(c.type == "file_upload" for c in library_page.components)
    
    # Verify chat_thread component
    assistant_page = next(p for p in ir.pages if p.route.startswith("/assistant"))
    assert any(c.type == "chat_thread" for c in assistant_page.components)
```

### Component Tests
Test individual RAG components:

```python
def test_tool_call_view_config():
    """Test tool_call_view has correct configuration"""
    assistant_page = get_page(app, "/assistant/:collection_id")
    tool_view = find_component(assistant_page, "tool_call_view")
    
    assert tool_view.show_timing == True
    assert tool_view.show_inputs == True
    assert tool_view.group_by == "tool_name"

def test_diff_view_config():
    """Test diff_view configuration for answer comparison"""
    history_page = get_page(app, "/history")
    diff_view = find_component(history_page, "diff_view")
    
    assert diff_view.mode == "split"
    assert diff_view.highlight_changes == True
```

### Integration Tests
Test the full RAG pipeline:

```python
def test_rag_pipeline_integration():
    """Test document upload → chunking → embedding → search → answer"""
    # Upload document
    response = client.post("/api/documents/upload", files={"file": test_pdf})
    assert response.status_code == 200
    
    # Wait for processing
    document_id = response.json()["document_id"]
    wait_for_processing(document_id)
    
    # Query
    chat_response = client.post("/api/chat", json={
        "collection_id": 1,
        "message": "What is the main topic?"
    })
    
    # Verify answer has citations
    answer = chat_response.json()
    assert len(answer["citations"]) > 0
    assert "tool_calls" in answer
    assert any(tc["tool_name"] == "search_collection" for tc in answer["tool_calls"])
```

## Performance Considerations

### Vector Search Optimization
- **Index Type**: Use IVFFlat for collections < 100K chunks, HNSW for larger
- **Index Parameters**: Tune `lists` parameter based on collection size
- **Query Optimization**: Batch similarity searches when possible

### Chunking Strategy
- **Chunk Size**: 512 tokens balances context vs granularity
- **Overlap**: 50-100 token overlap prevents splitting mid-concept
- **Caching**: Cache embeddings to avoid recomputation

### Reranking
- **When to Use**: Collections > 1000 documents
- **Top K**: Retrieve 20-50, rerank to top 5-10
- **Model**: Cross-encoder models (e.g., ms-marco-MiniLM-L-6-v2)

### Database
- **Connection Pool**: Configure based on concurrent users
- **Vector Index**: Rebuild periodically as collection grows
- **Partitioning**: Consider partitioning large collections

## Troubleshooting

### Issue: Slow Retrieval
**Solution**:
- Check vector index exists: `\d chunks` in psql
- Rebuild index: `REINDEX INDEX chunks_embedding_idx`
- Reduce `top_k` parameter
- Enable query caching

### Issue: Poor Answer Quality
**Solution**:
- Review retrieval logs - are relevant chunks retrieved?
- Adjust similarity threshold (try 0.6-0.8)
- Enable reranking
- Increase `top_k` for broader context
- Check chunking strategy (semantic > fixed_size for some docs)

### Issue: Citations Not Appearing
**Solution**:
- Verify `show_citations: true` in chat_thread config
- Check query response includes citations array
- Ensure agent system prompt includes citation instructions
- Review tool_call_view to confirm search_collection executed

### Issue: File Upload Fails
**Solution**:
- Check file size < 50MB
- Verify accepted file types (.pdf, .txt, .docx, .md)
- Check backend logs for processing errors
- Ensure embedding API key is configured

## Conclusion

The RAG Document Assistant demonstrates production-grade patterns for building knowledge-based AI systems with Namel3ss. Key takeaways:

✅ **Use proper RAG architecture** - chunking, embedding, vector search, reranking
✅ **Provide full observability** - tool_call_view and log_view are essential
✅ **Ground answers in sources** - always include citations
✅ **Enable comparison** - diff_view helps evaluate answer quality
✅ **Design for iteration** - configurable settings, query history, analytics

This example serves as a reference implementation for teams building:
- Internal knowledge bases
- Customer support Q&A systems
- Document analysis tools
- Research assistants
- Compliance and legal document search

For questions or contributions, see [CONTRIBUTING.md](../CONTRIBUTING.md).
