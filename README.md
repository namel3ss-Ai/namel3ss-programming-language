# Namel3ss (N3) – The AI Programming Language

**Namel3ss** is a declarative AI programming language that compiles English-like specifications into production-ready applications with LLM integration, memory systems, multi-agent orchestration, and intelligent workflows built directly into the language.

Unlike frameworks that bolt AI onto existing languages, Namel3ss treats **prompts, memory, agents, chains, and RAG** as first-class language constructs—compiled, type-checked, and optimized into FastAPI backends and modern frontends.

## What Makes Namel3ss an AI Language?

Namel3ss is designed from the ground up for AI-first development:

- **Structured Prompts with Schemas**: Define typed prompt inputs/outputs with compile-time validation
- **Memory Systems**: Session, conversation, and global memory scopes as language primitives
- **Multi-Agent Orchestration**: Declarative agent graphs with routing, handoffs, and state management  
- **Chain Workflows**: Compose templates, connectors, and Python hooks into deterministic pipelines
- **RAG & Vector Search**: First-class dataset integration with semantic search and retrieval
- **Logic Engine**: Constraint-based reasoning compiled into backend execution
- **Observability Built-In**: Automatic tracing, metrics, and monitoring for all AI operations

All of these features compile down to optimized FastAPI backends with `/api/ai/*` endpoints, runtime type safety, and deterministic testing infrastructure—no library dependencies or framework gymnastics required.

## Quick Example: AI-Native Application

```text
app "AI Support" connects to postgres "SUPPORT_DB".

# First-class memory as a language construct
memory "chat_history":
  scope: conversation
  kind: conversation
  max_items: 50

# Structured prompts with typed schemas
prompt "ClassifyTicket":
  input:
    ticket: text
    history: conversation
  output:
    category: one_of("billing", "technical", "general")
    urgency: one_of("low", "medium", "high")
    summary: text
  using model "gpt-4o-mini":
    """
    You are a support agent. Classify this ticket based on history.
    
    Ticket: {{ticket}}
    History: {{history}}
    """

# Multi-step chain workflow
define chain "SupportFlow":
  steps:
    - step "classify":
        kind: prompt
        target: ClassifyTicket
        read_memory: chat_history
    - if ctx:steps.classify.result.urgency == "high":
        then:
          - step "escalate":
              kind: python
              module: ops.alerting
              arguments:
                ticket: ctx:payload
        else:
          - step "auto_reply":
              kind: template
              target: AutoReply
    - step "save":
        write_memory: chat_history

# Page with AI-powered form
page "Support Chat" at "/chat":
  show form "Ask Support":
    fields: message
    on submit:
      run chain SupportFlow with:
        ticket = form.message
      show toast "Your ticket has been processed"
```

This compiles to a complete FastAPI backend with:
- `/api/ai/chains/SupportFlow` endpoint with typed validation
- Conversation memory with configurable scope and TTL
- Prompt execution with schema enforcement
- Conditional workflow orchestration
- Deterministic testing stubs for CI/CD

## Language Features

### Core Application Structure
- **Apps** with database connections, theming, and configuration
- **Pages** with routes and component composition
- **Datasets** from SQL, REST, CSV with filters, joins, and caching
- **Components**: tables, charts, forms, actions with full styling

### AI-Native Constructs

#### Structured Prompts
```text
prompt "SummarizeDocument":
  input:
    content: text
    max_length: number = 500
  output:
    summary: text
    key_points: list<text>
  metadata:
    temperature: 0.2
  using model "gpt-4o-mini":
    """
    Summarize this document in under {{max_length}} characters:
    {{content}}
    """
```

#### Memory Systems
```text
memory "user_preferences":
  scope: user
  kind: key_value
  max_items: 100

memory "conversation":
  scope: session
  kind: conversation
  ttl: 3600

memory "knowledge_base":
  scope: global
  kind: vector
  dimensions: 1536
```

#### Multi-Agent Orchestration
```text
define agent "Researcher":
  model: "gpt-4o"
  tools: [web_search, read_document]
  instructions: "You research topics thoroughly"

define agent "Writer":
  model: "gpt-4o-mini"
  instructions: "You write clear summaries"

define agent_graph "ResearchPipeline":
  start: Researcher
  routing:
    - from: Researcher
      to: Writer
      condition: ctx:steps.Researcher.complete == true
```

#### RAG & Dataset Integration
```text
dataset "documentation" from vector_store DOCS:
  embeddings: openai
  dimensions: 1536
  similarity: cosine

prompt "AnswerQuestion":
  input:
    question: text
    context: retrieved_docs
  output:
    answer: text
    sources: list<text>
  using model "gpt-4o":
    """
    Answer based on these docs:
    {{context}}
    
    Question: {{question}}
    """

define chain "RAG_Query":
  input -> retrieve_similar_from documentation top_k 5 -> prompt AnswerQuestion
```

## Installation

Namel3ss requires **Python 3.10 or newer**.

### From PyPI (recommended)

```bash
pip install namel3ss
```

Verify installation:

```bash
namel3ss --help
```

### From source

```bash
git clone https://github.com/SsebowaDisan/namel3ss-programming-language.git
cd namel3ss-programming-language
pip install -e .
```

### Optional extras

```bash
# AI connectors for OpenAI, Anthropic, etc.
pip install namel3ss[ai-connectors]

# SQL database support
pip install namel3ss[sql]

# Redis for caching and memory
pip install namel3ss[redis]

# MongoDB datasets
pip install namel3ss[mongo]

# Observability with OpenTelemetry
pip install namel3ss[observability]

# All features
pip install namel3ss[all]
```

### Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
./scripts/run_tests.sh
```

## Quick Start

### 1. Create your first AI application

```bash
# Generate from example
namel3ss generate examples/ai_demo.n3 out

# Or create your own my_app.n3 file
namel3ss generate my_app.n3 out
```

### 2. Run the backend

```bash
cd out/backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Open the frontend

Open `out/frontend/index.html` in your browser or serve with:

```bash
cd out/frontend
python -m http.server 8080
```

## CLI Reference

### Core Commands

```bash
# Generate full application
namel3ss generate app.n3 output/

# Build with backend
namel3ss build app.n3 --build-backend --backend-out backend/

# Run development server
namel3ss run app.n3 --dev

# Execute a chain
namel3ss run chain SupportFlow --payload '{"ticket": "Help!"}'

# Run experiment with variants
namel3ss eval experiment ModelComparison
```

### AI-Specific Commands

```bash
# Test prompt execution
namel3ss run prompt ClassifyTicket --input '{"ticket": "Billing issue"}'

# Validate memory configuration
namel3ss doctor --check-memory

# Trace chain execution
namel3ss run chain RAG_Query --trace --payload '{"query": "..."}'

# Export agent graph visualization
namel3ss graph ResearchPipeline --output graph.json
```

### Configuration

```bash
# Use environment file
namel3ss run app.n3 --env .env.prod

# Check available integrations
namel3ss doctor

# Validate .n3 syntax
namel3ss validate my_app.n3
```

## Production Deployment

### Environment Variables

Set these before running your backend:

**Authentication:**
```bash
NAMEL3SS_AUTH_MODE=required
NAMEL3SS_JWT_SECRET=your-secret
NAMEL3SS_API_KEY=your-api-key
```

**Databases:**
```bash
NAMEL3SS_DATABASE_URL=postgresql://...
NAMEL3SS_REDIS_URL=redis://localhost:6379
MONGODB_URI=mongodb://...
```

**AI Providers:**
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**Runtime:**
```bash
NAMEL3SS_ALLOW_STUBS=0  # Disable test stubs in prod
NAMEL3SS_ENABLE_TRACING=1
```

### Run Production Server

```bash
cd out/backend
pip install -r requirements.txt

# With Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn
gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 4
```

### Observability Endpoints

Generated backends include:

- `GET /api/health` - Health check
- `GET /readyz` - Readiness probe (checks DB connections)
- `GET /metrics` - Prometheus metrics
- `GET /api/ai/chains` - List all chains
- `GET /api/ai/prompts` - List all prompts
- `GET /api/ai/memory/stats` - Memory system statistics

## Testing

Run the complete test suite:

```bash
./scripts/run_tests.sh
```

Run specific test categories:

```bash
# Language and parser tests
pytest tests/test_parser.py tests/test_codegen.py

# AI feature tests
pytest tests/test_prompts.py tests/test_chains.py tests/test_memory_system.py

# Backend integration tests
pytest tests/test_backend_integration.py

# Error message quality
pytest tests/test_error_messages.py
```

Run with coverage:

```bash
pytest --cov=namel3ss --cov-report=html
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Documentation Index](docs/INDEX.md)** - Central navigation hub
- **[Testing Guide](docs/TESTING.md)** - Complete testing reference
- **[Memory System](docs/MEMORY_SYSTEM.md)** - Memory scopes and patterns
- **[Control Flow](CONTROL_FLOW_SYNTAX.md)** - Conditionals and loops
- **[CLI Documentation](CLI_DOCUMENTATION.md)** - Full CLI reference

## Architecture

### Compilation Pipeline

```
.n3 source
    ↓
Parser (ANTLR4 grammar)
    ↓
AST (typed nodes)
    ↓
Resolver (semantic analysis)
    ↓
Code Generator
    ↓
FastAPI backend + Frontend assets
```

### Generated Backend Structure

```
out/backend/
├── main.py                 # FastAPI app entry
├── database.py            # SQLAlchemy models
├── schemas.py             # Pydantic schemas
├── requirements.txt       # Dependencies
├── generated/             # Generated routers
│   ├── pages.py
│   ├── datasets.py
│   └── ai/                # AI-specific endpoints
│       ├── prompts.py
│       ├── chains.py
│       ├── memory.py
│       └── agents.py
└── custom/                # User customizations (preserved)
    └── custom_api.py
```

### Runtime Features

- **Dataset Registry**: Cached execution with pagination
- **Memory Manager**: Scoped storage with TTL and eviction
- **Prompt Engine**: Schema validation and template rendering
- **Chain Orchestrator**: Step execution with context management
- **Agent Runtime**: Multi-agent graphs with tool calling
- **Observability**: OpenTelemetry tracing and Prometheus metrics

## Examples

### AI Examples

- `examples/ai_demo.n3` - Complete AI application
- `examples/text_classification.n3` - Classification with prompts
- `examples/rag_qa.n3` - RAG question answering
- `examples/experiment_comparison.n3` - A/B testing models
- `examples/memory_chat_demo.n3` - Stateful conversation

### Traditional Examples

- `examples/app.n3` - Full-stack CRUD application
- `examples/control_flow_demo.n3` - Conditionals and loops

Run any example:

```bash
namel3ss generate examples/ai_demo.n3 demo_output
cd demo_output/backend && uvicorn main:app --reload
```

## Why Namel3ss for AI Development?

### Compared to LangChain/LlamaIndex
- **Language, not library**: First-class syntax for prompts, memory, chains
- **Compile-time validation**: Catch errors before runtime
- **Type safety**: Structured inputs/outputs with schemas
- **Deterministic testing**: Built-in stubs and mocks

### Compared to prompt engineering tools
- **Full application generation**: Not just prompts—complete backends
- **Integrated memory**: Session, conversation, and global scopes built-in
- **Multi-agent orchestration**: Declarative agent graphs, not imperative code

### Compared to traditional web frameworks
- **AI-native constructs**: Prompts and chains are language primitives
- **Natural language syntax**: Write apps in structured English
- **Zero boilerplate**: No Flask/FastAPI setup, routes auto-generated

## Contributing

Contributions welcome! This is a production-ready AI programming language with:

- 500+ tests across language, backend, and AI features
- Comprehensive documentation and guides
- Type-safe compilation pipeline
- Production deployment patterns

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

[MIT License](LICENSE)

## Community

- **GitHub**: [github.com/SsebowaDisan/namel3ss-programming-language](https://github.com/SsebowaDisan/namel3ss-programming-language)
- **Issues**: Report bugs or request features
- **Discussions**: Share your AI applications built with Namel3ss

---

**Namel3ss** — The programming language where AI developers write in prompts, memory, agents, and chains.
