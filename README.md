# Namel3ss (N3) – The AI Programming Language

> **✨ Latest**: Syntax Highlighting & AI Semantic Components now available - [See Recent Additions](#recent-additions-november-2025)

**Namel3ss** is a declarative AI programming language that compiles English-like specifications into production-ready applications with LLM integration, memory systems, multi-agent orchestration, and intelligent workflows built directly into the language.

Unlike frameworks that bolt AI onto existing languages, Namel3ss treats **prompts, memory, agents, chains, and RAG** as first-class language constructs—compiled, type-checked, and optimized into FastAPI backends and modern frontends.

For all documentation, follow the shared [Style Guide](docs/STYLE_GUIDE.md) and link glossary terms on first use via [docs/reference/GLOSSARY.md](docs/reference/GLOSSARY.md). Documentation should target experienced engineers and use consistent terms such as [N3](docs/reference/GLOSSARY.md#n3), [Agent](docs/reference/GLOSSARY.md#agent), [Runtime](docs/reference/GLOSSARY.md#runtime), and [Compilation](docs/reference/GLOSSARY.md#compilation).

## What Makes Namel3ss an AI Language?

Namel3ss is designed from the ground up for AI-first development:

- **Structured Prompts with Schemas**: Define typed prompt inputs/outputs with compile-time validation
- **Memory Systems**: Session, conversation, and global memory scopes as language primitives
- **Multi-Agent Orchestration**: Declarative agent graphs with routing, handoffs, and state management  
- **Chain Workflows**: Compose templates, connectors, and Python hooks into deterministic pipelines
- **RAG & Vector Search**: First-class dataset integration with semantic search and retrieval
- **Production Local Models**: Full deployment management for vLLM, Ollama, and LocalAI with CLI-based operations
- **Professional UI Components**: Production-ready navigation (sidebar, navbar, breadcrumbs, command palette), data display (tables, charts, stats, timelines), and feedback components (modals, toasts)
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

# Professional dashboard with navigation and data display
page "Analytics Dashboard" at "/dashboard":
  # Navigation sidebar
  sidebar:
    item "Dashboard" at "/dashboard" icon "📊"
    item "Analytics" at "/analytics" icon "📈"
    item "Reports" at "/reports" icon "📋":
      item "Sales" at "/reports/sales"
      item "Revenue" at "/reports/revenue"
  
  # Top navbar with actions
  navbar:
    logo: "/assets/logo.png"
    title: "Analytics"
    action "Export" icon "📥" type "button"
    action "User" icon "👤" type "menu":
      item "Profile" at "/profile"
      item "Logout" action "logout"
  
  # Breadcrumb navigation
  breadcrumbs:
    item "Home" at "/"
    item "Dashboard"
  
  # Command palette for quick navigation
  command palette:
    shortcut: "Ctrl+K"
    source "pages" from "/api/search/pages" label "Jump to..."
  
  # KPI metrics with sparklines
  show stat_summary from dataset revenue:
    label: "Total Revenue"
    value:
      field: total_revenue
      format: currency
    delta:
      field: revenue_change
      format: percentage
    sparkline:
      data: daily_revenue
      color: "#10b981"
  
  # Multi-series chart
  show data_chart "Sales Trend" from dataset sales:
    chart:
      type: line
      x_axis: date
      series:
        - data_key: revenue
          label: "Revenue"
        - data_key: orders
          label: "Orders"
  
  # Interactive data table
  show data_table "Recent Orders" from dataset orders:
    columns:
      - field: order_number
        header: "Order #"
        sortable: true
      - field: amount
        header: "Amount"
        format: currency
    toolbar:
      searchable: true
      filters:
        - field: status
          label: "Status"
          type: select
    row_actions:
      - label: "View"
        action: view_order
        icon: eye
  
  # Feedback components for user interactions
  modal "confirm_delete":
    title: "Delete Order?"
    description: "This action cannot be undone"
    trigger: "show_delete_confirm"
    content:
      show text "The order will be permanently removed from the system."
    actions:
      action "Cancel" variant "ghost"
      action "Delete" variant "destructive" action "do_delete"
  
  toast "order_deleted":
    title: "Order Deleted"
    description: "The order has been removed"
    variant: success
    duration: 3000
    action_label: "Undo"
    action: "undo_delete"
    trigger: "show_delete_success"

# AI-powered form with chain workflow
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

### Local Model Example

For private, cost-effective AI without external APIs:

```text
# Deploy a local model for private AI
model "local_chat" using local_engine:
  engine_type: "ollama" 
  model_name: "llama3.2:latest"
  deployment:
    port: 11434
    gpu_layers: -1
    context_length: 4096

prompt "PrivateChat":
  input: message: text
  output: response: text
  using model "local_chat":
    "You are a helpful assistant. {{message}}"

# Deploy and use
# namel3ss deploy local start local_chat
# namel3ss run prompt PrivateChat --input '{"message": "Hello!"}'
```

## Recent Additions (November 2025)

### 🎨 Comprehensive Syntax Highlighting

**Python-like syntax highlighting** for `.ai` and `.n3` files across all major editors (VS Code, Vim, Neovim, Sublime Text). Keywords, components, properties, strings, and numbers are now color-coded for improved readability.

**Color Scheme:**
- Keywords (`app`, `page`, `if`, `llm`) in **blue/purple**
- UI Components (`show_text`, `modal`, `toast`) in **yellow**
- AI Components (`chat_thread`, `agent_panel`) in **orange**
- Properties (`title`, `messages_binding`) in **cyan**
- Strings, numbers, booleans, comments in distinct colors

**Example with colors:**
```namel3ss
# Chat interface demo
app "AI Dashboard"                    # comment in gray

page "Chat" at "/chat":               # keywords in blue
  chat_thread "conversation":         # AI component in orange
    messages_binding: agent.messages  # property in cyan, binding in default
    streaming_enabled: true           # property in cyan, boolean in purple
    show_tokens: 42                   # property in cyan, number in teal
```

**Installation:** See [SYNTAX_HIGHLIGHTING.md](SYNTAX_HIGHLIGHTING.md) for setup instructions and [SYNTAX_REFERENCE.md](SYNTAX_REFERENCE.md) for a complete color guide.

### 🤖 AI Semantic Components

**Six production-ready AI-specific UI components** for building intelligent interfaces:

- **`chat_thread`**: Conversation displays with real-time streaming, token counts, and message grouping
- **`agent_panel`**: Agent status dashboards showing metrics, model info, and tool availability
- **`tool_call_view`**: Tool invocation displays with expandable inputs/outputs and timing
- **`log_view`**: Structured log viewing with search, filtering, and syntax highlighting
- **`evaluation_result`**: Metrics display with deltas, trends, and comparisons
- **`diff_view`**: Side-by-side or unified diff viewer with syntax highlighting

```namel3ss
# Complete AI interface
page "AI Workspace":
  chat_thread "conversation":
    messages_binding: "agent_run.conversation"
    streaming_enabled: true
    show_tokens: true
    auto_scroll: true
  
  agent_panel "status":
    agent_binding: "agent_run.agent"
    show_metrics: true
    show_cost: true
    show_latency: true
  
  tool_call_view "tools":
    calls_binding: "agent_run.tool_calls"
    show_inputs: true
    show_outputs: true
    expandable: true
```

See [examples/ai_components_demo.ai](examples/ai_components_demo.ai) for complete examples.

### 🎉 Feedback Components (Modal Dialogs & Toast Notifications)

**Production-ready user feedback components** with comprehensive testing and documentation. Build professional UIs with confirmation dialogs, notifications, and interactive feedback patterns.

#### Modal Dialogs

Create rich dialog overlays for confirmations, forms, and important user interactions:

```text
# Confirmation dialog with actions
modal "confirm_delete":
  title: "Delete Order?"
  description: "This action cannot be undone"
  size: md  # sm (400px), md (600px), lg (800px), xl (1000px), full
  dismissible: true
  trigger: "show_delete_confirm"
  content:
    show text "The order will be permanently removed from the system."
    show text "This includes all related transactions and history."
  actions:
    action "Cancel" variant "ghost"
    action "Delete" variant "destructive" action "do_delete"

# Form validation modal
modal "edit_profile":
  title: "Edit Profile"
  size: lg
  actions:
    action "Cancel" variant "ghost"
    action "Validate" variant "primary" action "validate_profile" close false
    action "Save" variant "primary" action "save_profile"

# Info modal with custom size
modal "terms_conditions":
  title: "Terms & Conditions"
  description: "Please review our terms"
  size: xl
  dismissible: true
  content:
    show text "Full terms and conditions content here..."
  actions:
    action "Accept" variant "primary" action "accept_terms"
    action "Decline" variant "ghost" action "decline_terms"
```

**Modal Features**:
- **5 Size Variants**: `sm` (400px), `md` (600px), `lg` (800px), `xl` (1000px), `full` (full width)
- **5 Action Variants**: `default` (gray), `primary` (blue), `destructive` (red), `ghost` (transparent), `link` (text-only)
- **Nested Content**: Include multiple `show text` statements for rich content
- **Non-Closing Actions**: Set `close: false` for validation without closing modal
- **Dismissible Control**: Control ESC key and backdrop click behavior
- **Trigger-Based**: Show modals via events from backend or frontend
- **Accessibility**: Full ARIA support, keyboard navigation, focus management

#### Toast Notifications

Display temporary notifications for success, errors, warnings, and info:

```text
# Success notification with action
toast "order_created":
  title: "Order Created"
  description: "Your order has been placed successfully"
  variant: success
  duration: 3000
  position: top-right
  action_label: "View Order"
  action: "view_order"
  trigger: "show_order_created"

# Error notification (persistent)
toast "connection_error":
  title: "Connection Lost"
  description: "Unable to connect to server"
  variant: error
  duration: 0  # 0 = persistent (no auto-dismiss)
  position: top
  action_label: "Retry"
  action: "retry_connection"
  trigger: "show_connection_error"

# Warning notification
toast "low_stock":
  title: "Low Stock Warning"
  description: "Only 3 items remaining"
  variant: warning
  duration: 5000
  position: bottom-right
  trigger: "show_low_stock"

# Info notification (quick)
toast "sync_complete":
  title: "Sync Complete"
  variant: info
  duration: 2000
  position: bottom
  trigger: "show_sync_complete"
```

**Toast Features**:
- **5 Variants**: `default`, `success` (green, CheckCircle icon), `error` (red, XCircle icon), `warning` (yellow, AlertCircle icon), `info` (blue, Info icon)
- **6 Positions**: `top`, `top-right` (default), `top-left`, `bottom`, `bottom-right`, `bottom-left`
- **Flexible Duration**: 2000-5000ms for auto-dismiss, or 0 for persistent display
- **Optional Actions**: Add action button with label and handler
- **Trigger-Based**: Show toasts from backend responses or frontend events
- **Icons**: Automatic icon selection based on variant

#### Common Patterns

**Confirmation Flow** (Modal → Action → Toast):
```text
page "Orders" at "/orders":
  # User clicks delete button → triggers modal
  modal "confirm_delete":
    title: "Delete Order?"
    actions:
      action "Cancel" variant "ghost"
      action "Delete" variant "destructive" action "do_delete"
  
  # After deletion → show success toast
  toast "delete_success":
    title: "Order Deleted"
    variant: success
    duration: 3000
    action_label: "Undo"
    action: "undo_delete"
    trigger: "show_delete_success"
```

**Form Validation Pattern**:
```text
page "Settings" at "/settings":
  modal "edit_settings":
    title: "Edit Settings"
    actions:
      action "Validate" variant "primary" action "validate" close false
      action "Save" variant "primary" action "save"
  
  toast "validation_error":
    title: "Validation Failed"
    description: "Please check the highlighted fields"
    variant: error
    duration: 5000
    trigger: "show_validation_error"
  
  toast "save_success":
    title: "Settings Saved"
    variant: success
    duration: 3000
    trigger: "show_save_success"
```

**Multi-Step Process**:
```text
page "Onboarding" at "/onboarding":
  modal "welcome":
    title: "Welcome!"
    description: "Let's get you set up"
    actions:
      action "Get Started" variant "primary" action "start_setup"
  
  toast "step_progress":
    title: "Processing..."
    description: "Step 2 of 5"
    variant: info
    duration: 0  # Persistent until complete
    trigger: "show_step_progress"
  
  modal "setup_complete":
    title: "Setup Complete!"
    description: "You're all set to start using the app"
    actions:
      action "Continue" variant "primary" action "finish_setup"
    trigger: "show_setup_complete"
```

#### Documentation & Resources

- **📖 [Comprehensive Guide](docs/FEEDBACK_COMPONENTS_GUIDE.md)** - 6,500+ lines covering syntax, properties, usage patterns, best practices, accessibility, mobile considerations, complete examples, and troubleshooting
- **⚡ [Quick Reference](FEEDBACK_COMPONENTS_QUICK_REFERENCE.md)** - Handy reference card with common patterns and property tables
- **📊 [Implementation Summary](FEEDBACK_COMPONENTS_SUMMARY.md)** - Technical details, metrics, and integration points
- **🎨 [Demo Application](examples/feedback_demo.ai)** - 3 pages with 6 modals and 9 toasts demonstrating all features

#### Implementation Quality

- **✅ 56 Tests** (100% pass rate): 19 parser + 18 IR builder + 19 codegen
- **✅ Full Stack**: Parser → AST → IR → Codegen → React/TypeScript components
- **✅ Production Ready**: Marked as stable feature with zero breaking changes
- **✅ Accessibility**: ARIA-compliant, keyboard navigation, screen reader support
- **✅ Mobile Responsive**: Optimized for touch devices with appropriate sizing
- **✅ shadcn/ui Integration**: Uses Dialog and Toast components with Tailwind CSS

#### Migration from Browser Alerts

Replace blocking browser alerts with declarative, styled components:

**Before** (Browser APIs):
```javascript
if (confirm("Delete this item?")) {
    deleteItem();
}
alert("Item deleted!");
```

**After** (Namel3ss):
```text
modal "confirm_delete":
  title: "Delete Item?"
  actions:
    action "Cancel" variant "ghost"
    action "Delete" variant "destructive" action "do_delete"

toast "deleted":
  title: "Item Deleted"
  variant: success
  duration: 3000
```

**Benefits**: Declarative syntax, consistent styling, better UX, customizable, event-driven, accessible, mobile-friendly.

## Language Features

### Core Application Structure
- **Apps** with database connections, theming, and configuration
- **Pages** with routes and component composition
- **Datasets** from SQL, REST, CSV with filters, joins, and caching
- **Components**: tables, charts, forms, actions with full styling
- **Logging**: First-class log statements with runtime observability

### AI-Native Constructs

#### Logging & Observability
```text
page "User Dashboard" at "/dashboard":
  log info "Loading dashboard for user {user_id}"
  
  let user_data = dataset users where id == user_id
  
  if user_data.count == 0:
    log warn "User {user_id} not found in database"
    show text "User not found"
  else:
    log debug "User data loaded: {user_data.name}"
    show text "Welcome, {user_data.name}!"

# Configure with CLI: --log-level debug|info|warn|error
# Or environment: NAMEL3SS_LOG_LEVEL=debug
```

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

#### Local Model Deployment
```text
# Define local model with deployment configuration
model "local_llama" using local_engine:
  engine_type: "ollama"  # or "vllm", "localai"
  model_name: "llama3.2:latest"
  deployment:
    port: 11434
    gpu_layers: -1
    context_length: 4096
    max_tokens: 2048
  health_check:
    endpoint: "/api/health"
    timeout: 30

# Deploy and manage local models via CLI
# namel3ss deploy local start local_llama
# namel3ss deploy local stop local_llama
# namel3ss deploy local status

# Use in prompts like any other model
prompt "LocalChat":
  input: message: text
  output: response: text
  using model "local_llama":
    "You are a helpful assistant. {{message}}"
```

## Installation

Namel3ss requires **Python 3.10 or newer**.

### Core Installation (Lightweight)

The core installation includes only the essentials for parsing, code generation, and CLI:

```bash
pip install namel3ss
```

> **⚠️ Important:** Ensure you get the latest version (0.5.1). If you encounter issues:
> ```bash
> pip uninstall namel3ss -y
> pip cache purge
> pip install namel3ss==0.5.1
> ```
> 
> Verify installation: `namel3ss --version` should show `0.5.1`

This minimal installation (~10MB) provides:
- ✅ `.n3` file parsing and AST generation
- ✅ FastAPI backend code generation  
- ✅ Frontend site generation (static HTML, React)
- ✅ CLI commands: `build`, `run`, `test`, `lint`
- ✅ Language Server Protocol (LSP) support
- ✅ Template engine for code generation

**Perfect for**: CI/CD pipelines, code compilation, static analysis, development tools.

### Optional Features (Extras)

Install only the features you need:

#### AI & LLM Providers

```bash
# All AI providers (including local models)
pip install namel3ss[ai]

# Or individual providers
pip install namel3ss[openai]      # OpenAI (GPT-4, etc.)
pip install namel3ss[anthropic]   # Anthropic (Claude)
pip install namel3ss[local-models] # vLLM, Ollama, LocalAI

# Specific local model engines
pip install namel3ss[vllm]        # vLLM (high-performance)
pip install namel3ss[ollama]      # Ollama (easy setup)
pip install namel3ss[localai]     # LocalAI (multi-format)
```

**Enables**: Structured prompts, model adapters, token counting, AI chains, local model deployment

#### Databases

```bash
# All SQL databases
pip install namel3ss[sql]

# Or specific databases
pip install namel3ss[postgres]    # PostgreSQL (asyncpg + psycopg3)
pip install namel3ss[mysql]       # MySQL (aiomysql)
pip install namel3ss[mongo]       # MongoDB (motor + pymongo)
```

**Enables**: SQL datasets, database connectors, ORM models

#### Caching & Queues

```bash
pip install namel3ss[redis]       # Redis caching and pub/sub
```

**Enables**: Redis adapters, queue systems (RQ), caching layers

#### Real-time Features

```bash
pip install namel3ss[realtime]    # WebSockets + Redis
pip install namel3ss[websockets]  # WebSockets only
```

**Enables**: WebSocket endpoints, real-time data streaming, pub/sub

#### Observability

```bash
pip install namel3ss[otel]        # OpenTelemetry instrumentation
```

**Enables**: Distributed tracing, metrics, FastAPI auto-instrumentation

#### Everything

```bash
pip install namel3ss[all]         # All optional features
```

### Installation Examples

**Typical web app with database**:
```bash
pip install namel3ss[postgres]
```

**AI-powered application with external APIs**:
```bash
pip install namel3ss[ai,postgres,redis]
```

**Local AI application with private models**:
```bash
pip install namel3ss[local-models,postgres]
```

**Development with Ollama**:
```bash
pip install namel3ss[ollama,postgres]
```

**Production with vLLM**:
```bash
pip install namel3ss[vllm,postgres,redis,otel]
```

**Full-featured setup**:
```bash
pip install namel3ss[all]
```

### Feature Detection

Namel3ss gracefully handles missing dependencies with helpful error messages:

```python
# If OpenAI not installed
n3 run app.ai
# Error: OpenAI integration requires the 'openai' extra.
# Install with: pip install 'namel3ss[openai]'
# Or for all AI providers: pip install 'namel3ss[ai]'
```

Check available features:
```bash
python -c "from namel3ss.features import print_feature_status; print_feature_status()"
```

Verify installation:

```bash
namel3ss --help
namel3ss --version  # Should show: namel3ss 0.5.0 (language 0.1.0)
```

## 📚 Documentation

**Complete Guides:**
- **[📖 Complete Documentation](NAMEL3SS_DOCUMENTATION.md)** - Comprehensive guide covering all language features, CLI commands, AI components, and deployment
- **[🔧 API Reference](API_REFERENCE.md)** - Detailed Python API documentation for all classes and functions
- **[🛠️ Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions for development with namel3ss
- **[🩺 Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues, syntax errors, JavaScript limitations, and solutions
- **[🛠️ Troubleshooting Guide](TROUBLESHOOTING.md)** - Solutions for common issues, syntax errors, and JavaScript limitations
- **[🛠️ Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions for namel3ss development
- **[🚨 Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions for development problems
- **[🆘 Troubleshooting Guide](TROUBLESHOOTING.md)** - Solutions for common syntax, runtime, and development issues

**Quick References:**
- **[⚡ CLI Quick Reference](docs/archive/status-reports/CLI_QUICK_REFERENCE.md)** - All CLI commands with examples
- **[🧠 AI Features Guide](docs/INDEX.md)** - AI-native constructs, prompts, chains, and agents
- **[💾 Memory System](docs/MEMORY_SYSTEM.md)** - Persistent state and conversation management
- **[🔍 RAG Guide](docs/RAG_GUIDE.md)** - Retrieval-Augmented Generation implementation
- **[📊 Evaluation Framework](docs/EVAL_SUITES.md)** - Testing and metrics for AI applications

**Examples:**
- **[🛠️ Examples Directory](examples/)** - Complete working applications
- **[🏥 Hospital AI](examples/hospital-ai/)** - Multi-agent hospital support system with triage, medication lookup, and patient messaging
- **[📝 Language Syntax](docs/LANGUAGE_REFERENCE.md)** - Complete N3 syntax guide
- **[🏗️ Architecture](docs/ARCHITECTURE.md)** - System design and internals

### Feature → Extra Mapping

Quick reference for what each extra provides:

| Feature | Extra | Packages Installed |
|---------|-------|-------------------|
| **Core** (always included) | _(none)_ | `pydantic`, `jinja2`, `pygls`, `packaging`, `psutil`, `pyyaml`, `rich`, `httpx`, `click` |
| OpenAI (GPT models) | `[openai]` | `openai`, `tiktoken` |
| Anthropic (Claude) | `[anthropic]` | `anthropic` |
| Local models (all) | `[local-models]` | `vllm`, `ollama-python`, `docker` |
| vLLM (high-performance) | `[vllm]` | `vllm` |
| Ollama (easy setup) | `[ollama]` | `ollama-python` |
| LocalAI (multi-format) | `[localai]` | `docker` |
| All AI providers | `[ai]` | `openai`, `anthropic`, `tiktoken`, `vllm`, `ollama-python`, `docker` |
| PostgreSQL | `[postgres]` | `sqlalchemy`, `asyncpg`, `psycopg[binary]` |
| MySQL | `[mysql]` | `sqlalchemy`, `aiomysql` |
| All SQL databases | `[sql]` | `sqlalchemy`, `asyncpg`, `psycopg`, `aiomysql` |
| MongoDB | `[mongo]` | `motor`, `pymongo` |
| Redis caching/queues | `[redis]` | `redis>=5.0` |
| WebSockets | `[websockets]` | `websockets>=12.0` |
| Real-time (WS + Redis) | `[realtime]` | `websockets`, `redis` |
| OpenTelemetry | `[otel]` | `opentelemetry-*` (SDK, instrumentation, exporters) |
| Development tools | `[dev]` | `pytest`, `black`, `mypy`, `ruff`, etc. |
| Everything | `[all]` | All of the above |

### From source

```bash
git clone https://github.com/SsebowaDisan/namel3ss-programming-language.git
cd namel3ss-programming-language
pip install -e .[dev]  # Include dev tools
```

### Legacy extras (deprecated)

The following extras are deprecated and will be removed in future versions:

```bash
# DEPRECATED: Use [ai] instead
pip install namel3ss[ai-connectors]

# sql, redis, mongo extras remain supported
pip install namel3ss[sql]
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

## Local Model Deployment

Namel3ss provides production-grade support for deploying and managing local LLM models using industry-standard engines like vLLM, Ollama, and LocalAI. This enables private, cost-effective, and low-latency AI applications without relying on external APIs.

### Supported Local Model Engines

#### vLLM - High-Performance Production Inference
- **Best for**: Production workloads requiring maximum throughput
- **Features**: GPU optimization, continuous batching, tensor parallelism
- **Models**: Any HuggingFace model with supported architectures
- **Installation**: `pip install namel3ss[vllm]`

#### Ollama - Easy Development & Deployment  
- **Best for**: Development, prototyping, and simple deployment scenarios
- **Features**: Automatic model pulling, easy setup, CPU/GPU support
- **Models**: Pre-optimized models from Ollama library (Llama, Mistral, etc.)
- **Installation**: `pip install namel3ss[ollama]`

#### LocalAI - Multi-Format Model Support
- **Best for**: Diverse model formats and experimental models
- **Features**: OpenAI-compatible API, multiple model formats, Docker deployment
- **Models**: GGML, GGUF, HuggingFace, custom models
- **Installation**: `pip install namel3ss[localai]`

### Quick Local Model Setup

1. **Install local model support**:
   ```bash
   pip install namel3ss[local-models]  # All engines
   # OR
   pip install namel3ss[ollama]        # Just Ollama
   ```

2. **Define a model in your `.ai` file**:
   ```text
   model "chat_model" using local_engine:
     engine_type: "ollama"
     model_name: "llama3.2:latest"
     deployment:
       port: 11434
       gpu_layers: -1
       context_length: 4096
   ```

3. **Deploy the model**:
   ```bash
   namel3ss deploy local start chat_model
   ```

4. **Use in your application**:
   ```text
   prompt "ChatWithLocal":
     input: message: text
     output: response: text  
     using model "chat_model":
       "You are a helpful assistant. {{message}}"
   ```

### Model Configuration Examples

#### vLLM Configuration (Production)
```text
model "production_model" using local_engine:
  engine_type: "vllm"
  model_name: "microsoft/DialoGPT-large"
  deployment:
    port: 8000
    host: "0.0.0.0"
    gpu_memory_utilization: 0.9
    tensor_parallel_size: 2
    max_model_len: 4096
    served_model_name: "gpt-3.5-turbo"  # OpenAI-compatible name
  health_check:
    endpoint: "/health"
    timeout: 30
```

#### Ollama Configuration (Development)
```text
model "dev_model" using local_engine:
  engine_type: "ollama" 
  model_name: "llama3.2:latest"
  deployment:
    port: 11434
    host: "127.0.0.1"
    gpu_layers: -1  # Use all GPU layers
    num_ctx: 4096   # Context length
    temperature: 0.7
  health_check:
    endpoint: "/api/health"
    timeout: 60
```

#### LocalAI Configuration (Multi-format)
```text
model "localai_model" using local_engine:
  engine_type: "localai"
  model_name: "ggml-gpt4all-j.bin"
  deployment:
    port: 8080
    deployment_type: "docker"  # or "binary"
    models_path: "./models"
    threads: 4
    context_size: 4096
  health_check:
    endpoint: "/readiness"
    timeout: 45
```

### Deployment Management

```bash
# Start models
namel3ss deploy local start chat_model
namel3ss deploy local start --all  # Start all defined models

# Monitor status
namel3ss deploy local status       # All models
namel3ss deploy local status chat_model  # Specific model

# Manage running models  
namel3ss deploy local stop chat_model
namel3ss deploy local restart chat_model

# Health monitoring
namel3ss deploy local health chat_model
namel3ss deploy local logs chat_model

# Configuration
namel3ss deploy local list         # List available models
namel3ss deploy local config chat_model  # Show configuration
```

### Production Considerations

- **Resource Requirements**: Each model requires significant RAM/VRAM
- **Port Management**: Ensure ports don't conflict between models  
- **Health Monitoring**: Built-in health checks for reliability
- **Deployment Automation**: Integrates with Docker and orchestration tools
- **OpenAI Compatibility**: All engines support OpenAI-compatible APIs

See `examples/local-model-chat/` for a complete working application demonstrating local model deployment with all three engines.

## Quick Start

### 1. Create your first AI application

```bash
# Create from template
echo 'app "MyApp"' > my_app.ai
echo 'page "home" { show text "Hello World" }' >> my_app.ai

# Build the application
namel3ss build my_app.ai

# For local model examples
cp -r examples/local-model-chat .
cd local-model-chat
namel3ss build app.ai
```

### 2. Install dependencies and run

```bash
cd out/backend
pip install -r requirements.txt

# For local models, also install model engines
pip install namel3ss[local-models]  # All engines
# OR specific engines:
pip install namel3ss[ollama]        # Just Ollama
pip install namel3ss[vllm]          # Just vLLM
pip install namel3ss[localai]       # Just LocalAI

uvicorn main:app --reload
```

### 3. Deploy local models (optional)

```bash
# Start a local model (if using local model example)
namel3ss deploy local start your_model_name

# Check status
namel3ss deploy local status
```

### 4. Open the frontend

Open `out/frontend/index.html` in your browser or serve with:

```bash
cd out/frontend
python -m http.server 8080
```

## CLI Reference

### Core Commands

```bash
# Generate full application
namel3ss generate app.ai output/

# Build with backend
namel3ss build app.ai --build-backend --backend-out backend/

# Run development server
namel3ss run app.ai --dev

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

### Local Model Deployment Commands

```bash
# Deploy local models
namel3ss deploy local start my_model     # Start model deployment
namel3ss deploy local stop my_model      # Stop running model
namel3ss deploy local restart my_model   # Restart model
namel3ss deploy local status             # Show all model statuses
namel3ss deploy local list               # List available models

# Model configuration
namel3ss deploy local config my_model    # Show model configuration
namel3ss deploy local validate my_model  # Validate model config

# Health monitoring
namel3ss deploy local health my_model    # Check model health
namel3ss deploy local logs my_model      # View model logs
```

### Configuration

```bash
# Use environment file
namel3ss run app.ai --env .env.prod

# Check available integrations
namel3ss doctor

# Validate .n3 syntax
namel3ss validate my_app.ai
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

**Local Model Deployment:**
```bash
# Model deployment paths
NAMEL3SS_LOCAL_MODELS_ROOT=/opt/models
NAMEL3SS_DEPLOYMENT_LOGS=/var/log/namel3ss

# vLLM configuration
VLLM_HOST=0.0.0.0
VLLM_PORT=8000
VLLM_GPU_MEMORY_UTILIZATION=0.9

# Ollama configuration 
OLLAMA_HOST=127.0.0.1:11434
OLLAMA_MODELS=/usr/share/ollama/.ollama/models

# LocalAI configuration
LOCALAI_MODELS_PATH=/models
LOCALAI_THREADS=4
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

# Conformance tests (language specification)
namel3ss conformance
pytest tests/conformance_runner/  # Meta-tests for conformance infrastructure
```

Run with coverage:

```bash
pytest --cov=namel3ss --cov-report=html
```

## Conformance & Governance

Namel3ss includes a comprehensive conformance test suite to ensure consistency across implementations and guide language evolution.

### Running Conformance Tests

```bash
# Run all conformance tests
namel3ss conformance

# Run specific category
namel3ss conformance --category parse

# Run specific test
namel3ss conformance --test parse-valid-001

# Verbose output
namel3ss conformance --verbose

# JSON output (for CI)
namel3ss conformance --format json
```

### Conformance Test Results

Current test coverage:
- **Parse Phase**: 30 tests (100% passing)
  - 18 valid syntax tests
  - 12 invalid syntax tests
- **Type System**: Coming soon
- **Runtime**: Coming soon

### For External Implementers

If you're building an alternative Namel3ss implementation:

1. **Read the specification**: See [tests/conformance/SPEC.md](tests/conformance/SPEC.md) for the test format
2. **Run the test suite**: Test descriptors are in `tests/conformance/v1/`
3. **Validate conformance**: Tests are machine-readable YAML files
4. **Report results**: Use the same test IDs for cross-implementation comparison

See [CONFORMANCE.md](CONFORMANCE.md) for a complete guide with examples in Python and JavaScript.

### Language Governance

Namel3ss follows an RFC (Request for Comments) process for language changes:

1. **Submit RFC**: Propose changes with motivation and design
2. **Discussion**: Community feedback (2-4 weeks)
3. **FCP**: Final Comment Period (10 days)
4. **Decision**: Accepted or Rejected by maintainers

**All language-level changes must include**:
- RFC document with rationale
- Conformance tests demonstrating the new behavior
- Implementation in the reference implementation
- Documentation updates

See [GOVERNANCE.md](GOVERNANCE.md) for the complete process and [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Documentation

Comprehensive documentation is available in the `docs/` directory:

### Core Documentation
- **[Documentation Index](docs/INDEX.md)** - Central navigation hub
- **[Testing Guide](docs/TESTING.md)** - Complete testing reference
- **[Memory System](docs/MEMORY_SYSTEM.md)** - Memory scopes and patterns
- **[Control Flow](CONTROL_FLOW_SYNTAX.md)** - Conditionals and loops
- **[CLI Documentation](CLI_DOCUMENTATION.md)** - Full CLI reference

### Conformance & Governance
- **[CONFORMANCE.md](CONFORMANCE.md)** - External implementation guide
- **[GOVERNANCE.md](GOVERNANCE.md)** - Language governance and RFC process
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines with conformance requirements
- **[tests/conformance/SPEC.md](tests/conformance/SPEC.md)** - Conformance test specification

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

### Production Examples

- **[AI Customer Support Console](docs/examples/ai-customer-support-console.md)** - Professional support agent workspace demonstrating chrome navigation (sidebar, navbar, breadcrumbs), dynamic routing, and dataset integration. A complete example showing best practices for production Namel3ss applications. **Status: ✅ Parsing & IR generation working**
  
  **Key Learnings:**
  - Dataset declarations must use `dataset "name" from table table_name` syntax (no `columns:` or `fields:` blocks)
  - Namel3ss requires **4-space indentation** (not 2-space)
  - Chrome components (sidebar, navbar, breadcrumbs) must be declared before page content
  - Dataset names must be quoted strings, not bare identifiers
  - Working examples (like `chrome_demo_clean.ai`) are the authoritative syntax reference

- **[RAG Document Assistant & Citation Explorer](examples/rag-document-assistant-and-citation-explorer.md)** - Production-grade RAG system showcasing complex nested UI patterns: multi-column info grids, deeply nested card sections, tabs with chrome components, and comprehensive citation tracking. Demonstrates tool/agent definitions, vector store configuration, and advanced data display components (file_upload, chat_thread, tool_call_view, log_view, diff_view) for document Q&A workflows. **Status: ✅ All tests passing (13/13) + Enhanced with parser improvements!**
  
  **Key Learnings:**
  - ✅ **Chrome components now work with `show tabs`!** Use simple `tab "Name":` syntax
  - ✅ **Tool definitions fully supported:** Define tools with parameters and descriptions
  - ✅ **Agent definitions fully supported:** Configure agents with LLM, tools, and goals
  - ✅ **Vector store configuration supported:** Define pgvector stores with dimensions
  - ✅ **`condition:` works as alias for `filter_by:`** - Use whichever you prefer
  - Multi-column info_grid layouts (2, 3, 4 columns) work correctly
  - Deeply nested cards (3-4 levels) parse successfully: `card → sections → info_grid → items`
  - Multiple badges and actions per card item are supported
  - `group_by:` property enables data grouping in card displays
  
  **Parser Enhancements (Nov 26, 2025):**
  All three previously-documented limitations are now fixed! 🎉
  1. ✅ Chrome + tabs combination works
  2. ✅ Tool/agent/vector_store definitions parse correctly
  3. ✅ `condition:` property supported as filter_by alias

### AI Examples

- `examples/llm_benchmark_lab/benchmark.ai` - LLM Benchmark & Experiment Lab ([docs](examples/llm_benchmark_lab/README.md))
- `examples/ai_demo.ai` - Complete AI application
- `examples/text_classification.n3` - Classification with prompts
- `examples/rag_qa.n3` - RAG question answering
- `examples/experiment_comparison.n3` - A/B testing models
- `examples/memory_chat_demo.n3` - Stateful conversation

### Traditional Examples

- `examples/app.ai` - Full-stack CRUD application
- `examples/control_flow_demo.n3` - Conditionals and loops

Run any example:

```bash
namel3ss generate examples/ai_demo.ai demo_output
cd demo_output/backend && uvicorn main:app --reload
```

## Repository Structure

```
namel3ss-programming-language/
├── examples/                  # Working example applications
│   ├── minimal/              # Basic N3 usage  
│   ├── content-analyzer/     # Agent-based content analysis
│   └── research-assistant/   # Multi-turn research workflows
├── tests/                    # Comprehensive test suite
│   ├── conformance/          # Language conformance tests
│   │   ├── SPEC.md          # Conformance test specification
│   │   └── v1/              # Version 1.0.0 test suite
│   │       ├── parse/       # Parse phase tests (30 tests)
│   │       └── fixtures/    # Test source files
│   ├── conformance_runner/   # Meta-tests for conformance infrastructure (41 tests)
│   ├── unit/fixtures/       # Unit test fixtures by component
│   └── integration/         # Integration and build validation
├── scripts/                 # Development utilities
│   ├── demos/              # Feature demonstrations
│   ├── tests/              # Standalone test scripts
│   └── utilities/          # Development tools
├── docs/                    # Organized documentation
│   ├── guides/             # User and developer guides
│   ├── specifications/     # Technical specifications
│   └── archive/            # Historical documents
├── namel3ss/               # Core language implementation
│   ├── conformance/        # Conformance test infrastructure
│   │   ├── models.py       # Test descriptors and discovery
│   │   └── runner.py       # Test execution engine
│   ├── cli/                # Command-line interface
│   └── ...                 # Parser, codegen, runtime, etc.
├── rfcs/                   # RFC (Request for Comments) proposals
│   ├── 0000-template.md   # RFC template
│   └── README.md          # RFC process documentation
├── api/                    # REST API server
├── CONFORMANCE.md          # External implementation guide
├── GOVERNANCE.md           # Language governance model
├── CONTRIBUTING.md         # Contribution guidelines
└── [configuration files]   # Build and deployment configs
```

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run an example:**
   ```bash
   namel3ss build examples/minimal/app.ai
   ```

3. **Start development server:**
   ```bash
   namel3ss run examples/minimal/app.ai
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

- **500+ tests** across language, backend, and AI features
- **Conformance suite** with 71 tests (30 conformance + 41 meta-tests)
- **RFC-based governance** for language changes
- **Comprehensive documentation** and guides
- **Type-safe compilation** pipeline
- **Production deployment** patterns

### Contributing Requirements

All contributions must follow these guidelines:

1. **Code changes**: Include tests and documentation
2. **Language changes**: Must include RFC + conformance tests
3. **Bug fixes**: Add regression tests
4. **Features**: Update relevant documentation

See [CONTRIBUTING.md](CONTRIBUTING.md) for complete details on:
- Development setup
- Testing requirements
- RFC process
- Conformance test requirements
- Code style and PR guidelines

## Migration Notice

**⚠️ Breaking Change**: As of the latest version, Namel3ss source files now use `.ai` extensions instead of `.n3`. If you have existing `.n3` files, please rename them to `.ai`:

```bash
# For individual files
mv my_app.n3 my_app.ai

# For all files in a directory
for file in *.n3; do mv "$file" "${file%.n3}.ai"; done
```

All documentation, examples, and CLI commands now use `.ai` extensions. The `.n3` extension is no longer supported.

## License

[MIT License](LICENSE)

## Community

- **GitHub**: [github.com/SsebowaDisan/namel3ss-programming-language](https://github.com/SsebowaDisan/namel3ss-programming-language)
- **Issues**: Report bugs or request features
- **Discussions**: Share your AI applications built with Namel3ss

---

**Namel3ss** — The programming language where AI developers write in prompts, memory, agents, and chains.
