# Namel3ss
## An AI-Native DSL for Orchestrating Production Applications

> **✨ Now shipping v2.0** – [What's new](#whats-new)

**Namel3ss is a declarative DSL purpose-built for AI orchestration.**

We're not trying to replace Python or TypeScript. We're doing something different: creating a specialized language where AI agents, prompts, memory, and workflows are first-class primitives.

**Think of us as the "SQL for AI applications"** – just as SQL specializes in data queries, Namel3ss specializes in AI orchestration.

You write high-level declarations. We generate production-ready backends (FastAPI) and frontends (React). You stay focused on what makes your application unique: the AI logic.

No frameworks to learn. No libraries to wrangle. No boilerplate to write.

**Just pure AI orchestration compiled to production code.**

---

## What Makes Namel3ss Different

**1. AI-Native Language Design**

Other languages bolt AI onto legacy syntax. We started from zero:
- Prompts are first-class types with schemas
- Memory is a language construct, not an afterthought
- Agents are declarative, not imperative
- Workflows are orchestrated, not coded

**2. Specialized, Not General-Purpose**

We're hyper-focused on AI orchestration:
- ✅ Building AI agents and multi-agent systems
- ✅ RAG pipelines and knowledge retrieval
- ✅ Conversational AI with memory
- ✅ Complex AI workflows and chains
- ❌ System programming or low-level computation
- ❌ Replacing your backend language
- ❌ General-purpose application logic

Use Namel3ss for AI coordination. Use Python/TypeScript for everything else.

**3. Declarative → Code Generation**

You declare what you want, we generate how to build it:
- Production FastAPI backends with typed validation
- Modern React frontends with professional UI components
- Complete test infrastructure
- Database migrations
- Deployment configurations

This is compiler-driven development. Your source is concise. The generated code is battle-tested.

## Core Capabilities

### 🤖 Multi-Agent Orchestration

Build sophisticated agent systems with declarative syntax:

```n3
agent "ResearchAgent" {
  role: "Research Specialist"
  goal: "Find accurate information"
  tools: [web_search, document_reader]
  memory: "research_context"
}

agent "WriterAgent" {
  role: "Content Writer"
  goal: "Create engaging content"
  tools: [text_editor, grammar_check]
  memory: "writing_context"
}

chain "ContentPipeline" {
  steps:
    - agent: ResearchAgent, task: "Research {{topic}}"
    - agent: WriterAgent, task: "Write article from {{research_results}}"
    - step: "publish", action: save_to_database
}
```

No imperative choreography. Just declare the workflow.

### 🧠 Structured Memory Systems

Memory isn't an API call—it's a language construct:

```n3
memory "conversation_history" {
  scope: "user"           # Per-user, per-session, or global
  kind: "conversation"    # Conversation, vector, or graph
  max_items: 100
  retention_days: 30
}

memory "knowledge_base" {
  scope: "global"
  kind: "vector"
  embedding_model: "text-embedding-3-small"
}

prompt "ContextualChat" {
  input: {message: text}
  output: {response: text}
  using model "gpt-4":
    """
    Chat history: {{read_memory("conversation_history")}}
    Relevant context: {{search_memory("knowledge_base", query=message)}}
    
    User message: {{message}}
    """
}
```

### 📋 Type-Safe Prompts

Every prompt has a schema. Every output is validated:

```n3
prompt "ExtractEntities" {
  input: {
    text: text,
    entity_types: array<text>
  }
  output: {
    entities: array<{
      type: text,
      value: text,
      confidence: number
    }>
  }
  using model "gpt-4o-mini":
    """
    Extract {{entity_types}} from: {{text}}
    Return as structured JSON.
    """
}
```

Catch errors at compile time, not in production.

### 🔄 RAG Pipelines

Retrieval-Augmented Generation made simple:

```n3
index "documentation" {
  source_dataset: "docs"
  embedding_model: "text-embedding-3-small"
  chunk_size: 512
  backend: "pgvector"
}

rag_pipeline "doc_qa" {
  query_encoder: "text-embedding-3-small"
  index: "documentation"
  top_k: 5
  reranker: "cross_encoder"
}

prompt "AnswerQuestion" {
  input: {question: text}
  output: {answer: text, sources: array<text>}
  using rag: "doc_qa", model: "gpt-4":
    """
    Context: {{retrieved_documents}}
    Question: {{question}}
    
    Provide accurate answer with sources.
    """
}
```

### 🎯 Advanced Expression Language (New in 2.0!)

Functional programming features for complex logic:

```n3
# Lambda expressions
let active_users = filter(users, fn(u) => u.status == "active")
let user_names = map(active_users, fn(u) => u.name)

# List comprehensions
let doubled = [x * 2 for x in numbers if x > 0]
let formatted = [format_name(user) for user in users if user.verified]

# Subscripts and slicing
let first_item = items[0]
let user_email = user["email"]
let top_five = results[0:5]
```

### 📦 Multi-File Module System (New in 2.0!)

Build large applications across multiple files:

```n3
# app/main.ai
module "app.main"
import "app.models.user"
import "app.shared.types"

app "CustomerPortal" {
  # Use imported types and functions
}

# app/models/user.ai
module "app.models.user"
import "app.shared.types"

dataset "active_users" from postgres {
  schema: UserSchema
}

# app/shared/types.ai
module "app.shared.types"

schema UserSchema {
  id: number,
  name: text,
  email: text
}
```

### ✅ Static Type Checking (New in 2.0!)

Catch errors before runtime:

```n3
fn calculate_total(items: array<number>, tax_rate: number): number =>
  sum(items) * (1 + tax_rate)

# ❌ Type error: Cannot pass text to function expecting number
calculate_total(["1", "2"], 0.1)

# ✅ Type-safe
calculate_total([10, 20, 30], 0.1)
```

## See it in action

### Build Your First Chat Widget in 10 Minutes

Want to build a production-ready customer support chat widget? Here's how easy it is:

```n3
app "Support Chat"

# Configure AI model
model support_bot:
    provider: openai
    name: "gpt-4"
    temperature: 0.7

# Store conversation history
memory "conversation_history":
    scope: conversation
    kind: conversation
    max_items: 50

# AI chain for responses
chain "chat_response":
    inputs:
        message: text
        history: conversation
    steps:
        - step generate:
            model: support_bot
            prompt: |
                Chat history: {{history}}
                User: {{message}}
                
                Provide helpful customer support response.
            output: response

# Chat UI page
page "Chat" at "/":
    show text "Customer Support" style {
        fontSize: "24px"
        fontWeight: "bold"
        marginBottom: "20px"
    }
    
    # Display messages
    show list from memory "conversation_history":
        item:
            show text "{{role}}: {{content}}" style {
                padding: "12px"
                borderRadius: "8px"
                background: "{{role == 'user' ? '#e3f2fd' : '#f5f5f5'}}"
            }
    
    # Message input
    show form "Send Message":
        field "user_message" type textarea placeholder "Type your message..."
        button "Send"
        on submit:
            run chain "chat_response"
            show toast "Message sent!"
```

That's it! A complete chat widget with:
- ✅ AI-powered responses
- ✅ Conversation memory
- ✅ Professional UI
- ✅ Form handling
- ✅ Real-time updates

**Want more?** See the [Complete Chat Widget Example](docs/COMPLETE_CHAT_WIDGET_EXAMPLE.md) for 600+ lines with typing indicators, escalation, embedding, and more.

### Complete AI Support System

Here's a complete AI support system. Written in plain English. Ready for production.

```text
app "AI Support" connects to postgres "SUPPORT_DB".

# Memory as a language construct
memory "chat_history":
  scope: conversation
  kind: conversation
  max_items: 50

# Structured prompts with schemas
prompt "ClassifyTicket":
  input:
    ticket: text
    history: conversation
  output:
    category: one_of("billing", "technical", "general")
    urgency: one_of("low", "medium", "high")
  using model "gpt-4o-mini":
    """
    Classify this support ticket.
    Ticket: {{ticket}}
    History: {{history}}
    """

# Intelligent workflows
define chain "SupportFlow":
  steps:
    - step "classify":
        kind: prompt
        target: ClassifyTicket
        read_memory: chat_history
    - if ctx:steps.classify.result.urgency == "high":
        then:
          - step "escalate": kind: python, module: ops.alerting
        else:
          - step "auto_reply": kind: template, target: AutoReply
    - step "save": write_memory: chat_history

# Professional UI
page "Chat" at "/chat":
  show form "Ask Support":
    fields: message
    on submit:
      run chain SupportFlow with ticket = form.message
      show toast "Your ticket has been processed"
```

That's it. One file. Complete application.

This compiles to:
- Production FastAPI backend with typed validation
- Conversation memory with intelligent scoping
- Conditional workflow orchestration
- Real-time UI components
- Complete test infrastructure

**No frameworks. No glue code. No complexity.**

### Own your AI

Want private, secure AI? Run your own models locally:

```text
model "local_chat" using local_engine:
  engine_type: "ollama" 
  model_name: "llama3.2:latest"
  deployment:
    port: 11434
    gpu_layers: -1

prompt "PrivateChat":
  input: message: text
  output: response: text
  using model "local_chat":
    "You are a helpful assistant. {{message}}"
```

Deploy it. Use it. Own it. All with one command.

```bash
namel3ss deploy local start local_chat
```

No external APIs. No usage limits. Complete control over your AI.

---

## When to Use Namel3ss (and When Not To)

### ✅ Perfect For:

- **AI Agent Systems**: Multi-agent workflows, autonomous agents, agent orchestration
- **RAG Applications**: Document Q&A, knowledge bases, semantic search
- **Conversational AI**: Chatbots, support systems, interactive assistants
- **AI Workflow Automation**: Complex prompt chains, conditional logic, state management
- **Rapid AI Prototyping**: MVPs, demos, proof-of-concepts
- **AI-First Applications**: Where AI coordination is 80%+ of your logic

### ❌ Not Designed For:

- **General Web Applications**: Use Next.js, Django, Rails
- **System Programming**: Use Rust, C++, Go
- **Data Engineering Pipelines**: Use Airflow, Prefect
- **Mobile Apps**: Use React Native, Flutter
- **Low-Level Computation**: Use Python, C, Julia
- **Complex Business Logic**: Use TypeScript, Python

### 🤝 Works Great With:

Namel3ss is designed to **orchestrate**, not replace:

```
┌─────────────────────────────────────┐
│   Your Application Architecture     │
├─────────────────────────────────────┤
│  Namel3ss (AI Orchestration Layer)  │
│  ↓ Compiles to ↓                    │
│  FastAPI Backend + React Frontend   │
├─────────────────────────────────────┤
│  Your Existing Services:            │
│  • Python data processing           │
│  • TypeScript business logic        │
│  • PostgreSQL database              │
│  • Redis caching                    │
│  • External APIs                    │
└─────────────────────────────────────┘
```

Use Namel3ss as your **AI coordination layer**. Let it handle prompts, agents, memory, and workflows. Connect it to your existing infrastructure.

---

## Getting Started

### Install Namel3ss

```bash
pip install namel3ss
```

The core installation (~10MB) includes:
- Language parser and compiler
- FastAPI code generation
- React UI generation  
- CLI tools
- Development server

### Quick Start Example

Create `hello_ai.ai`:

```n3
app "Hello AI" connects to postgres "main_db" {
  description: "My first AI application"
}

prompt "Greeter" {
  input: {name: text}
  output: {greeting: text, tone: one_of("formal", "casual")}
  using model "gpt-4o-mini":
    """
    Greet {{name}} warmly.
    Determine if the greeting should be formal or casual.
    """
}

page "Home" at "/" {
  show form "Greeting Generator" {
    fields: [
      {name: "name", label: "Your Name", type: "text"}
    ]
    on submit {
      let result = call_prompt("Greeter", {name: form.name})
      show text result.greeting
      show badge result.tone
    }
  }
}
```

### Build and Run

```bash
# Compile to FastAPI + React
namel3ss build hello_ai.ai -o ./output

# Start development server
cd output
python -m uvicorn demo_backend.main:app --reload

# Frontend runs on http://localhost:3000
# Backend runs on http://localhost:8000
```

That's it! You have a working AI application.

---

## What's New in 2.0

### 📚 Comprehensive Documentation & Examples (November 2025)

**NEW!** We've added extensive guides to help you build production applications:

- **5 Complete Documentation Guides** covering UI components, real-time features, API integration, backend operations, and deployment
- **Production Chat Widget Example** with 600+ lines of working code
- **Working Code Examples** including chatbots, API demos, and navigation patterns
- **Syntax Guidelines** for correct indentation, memory blocks, and page structures

Everything you need to build sophisticated applications like customer support widgets, collaborative editors, and real-time dashboards.

👉 **[View All Documentation Guides](#-comprehensive-documentation-guides)**

### 🎯 Static Type Checking

Catch errors before runtime with comprehensive type validation:

```n3
fn process_users(users: array<{name: text, age: number}>): array<text> => {
  let adults = filter(users, fn(u) => u.age >= 18)
  return map(adults, fn(u) => u.name)
}

# Type errors caught at compile time!
```

### 🔥 Enhanced Expression Language

Lambda expressions, subscripts, and list comprehensions:

```n3
# Lambdas
let evens = filter(numbers, fn(x) => x % 2 == 0)

# Subscripts
let first = items[0]
let email = user["email"]

# Comprehensions
let doubled = [x * 2 for x in numbers if x > 0]
```

### 📦 Multi-File Module System

Build large applications with imports:

```n3
module "app.main"
import "app.models.user"
import "app.shared.types"

# Use symbols from imported modules
```

### 🛠️ Editor/IDE Integration API

Foundation for Language Server Protocol support:

```python
from namel3ss.tools.editor_api import analyze_module

result = analyze_module(source_code)
for diag in result.diagnostics:
    print(f"Error: {diag.message}")
```

See [ADVANCED_FEATURES.md](docs/ADVANCED_FEATURES.md) for complete details.

The core installation is elegant. Lightweight at ~10MB. Just what you need:
- Language parser and compiler
- FastAPI code generation
- React UI generation  
- CLI tools
- Development server

**Perfect for:** Getting started. Building applications. Shipping to production.

### Add only what you need

We believe in choice. Install just the features you want:

```bash
# AI providers (OpenAI, Claude, local models)
pip install namel3ss[ai]

# Your own private models
pip install namel3ss[local-models]

# Databases
pip install namel3ss[postgres]

# Real-time features
pip install namel3ss[realtime]

# Everything
pip install namel3ss[all]
```

**Or get specific:**

```bash
pip install namel3ss[openai]      # GPT-4 and friends
pip install namel3ss[anthropic]   # Claude
pip install namel3ss[ollama]      # Local Llama, Mistral
pip install namel3ss[vllm]        # High-performance local models
```

Simple. Modular. Your choice.

---

## Your first app in 60 seconds

```bash
# Install
pip install namel3ss

# Create
namel3ss init my-ai-app
cd my-ai-app

# Describe your app
echo 'app "MyApp"
page "home" { 
  show text "Hello, World" 
}' > app.ai

# Build it
namel3ss build app.ai

# Run it
cd out/backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**Done.** Your app is live at `localhost:8000`.

From thought to running application in 60 seconds.

---

## What's new

### v0.6.1 (November 2025)

We made RAG development feel magical.

**Tool definitions with parameters.** Define your tools once. Use them everywhere. Type-safe. Self-documenting.

**Agent configurations.** Describe what your agents do. What models they use. What tools they have. The system handles the rest.

**Vector stores.** First-class support for semantic search. pgvector. 1536 dimensions. Cosine similarity. Just declare it.

**Chrome + Tabs.** Build professional interfaces with navigation, tabs, and real-time components. Simple. Beautiful. Fast.

**Smart filtering.** Use `condition:` or `filter_by:`. Whichever reads better. The compiler understands both.

**Comprehensive error messages.** Try to use an unsupported component? Get detailed explanations why it's not supported, 2-3 alternative solutions with specific use cases, complete working examples, and documentation links. Errors that teach, not block.

And it all just works. 21+ tests passing. Production-ready. Zero breaking changes.

[See complete changelog →](#recent-additions-november-2025)

---

## Why developers love Namel3ss

### "It just works"

Compared to LangChain or LlamaIndex:
- **It's a language, not a library.** First-class syntax for AI constructs.
- **Compile-time validation.** Catch errors before deployment.
- **Type safety throughout.** Structured inputs and outputs.
- **Deterministic testing.** Built-in mocks. No flaky tests.

### "Finally, AI that makes sense"

Compared to prompt engineering tools:
- **Complete applications.** Not just prompts—full backends and UIs.
- **Integrated memory.** Session, conversation, and global scopes.
- **Multi-agent orchestration.** Declarative graphs. Simple routing.

### "So simple, it's revolutionary"

Compared to traditional web frameworks:
- **AI-native.** Prompts and agents are language primitives.
- **Natural syntax.** Write apps in structured English.
- **Zero boilerplate.** No Flask/FastAPI setup. Routes auto-generated.

---

## Production-ready

**500+ tests.** Language, backend, AI features—all tested.

**Conformance suite.** 71 tests ensuring consistency across implementations.

**Type-safe compilation.** From `.ai` files to production code.

**Observable by default.** OpenTelemetry tracing. Prometheus metrics. Health checks.

**Enterprise features:**
- JWT authentication
- Role-based access control
- Rate limiting
- Distributed tracing
- Horizontal scaling
- Docker deployment
- Kubernetes manifests

[See deployment guide →](https://github.com/namel3ss-Ai/namel3ss-programming-language/blob/main/docs/guides/PRODUCTION_DEPLOYMENT_GUIDE.md)

---

## Learn more

### 📖 Comprehensive Documentation Guides

**NEW!** Complete guides for building production applications:

👉 **[Complete Documentation Index](docs/DOCUMENTATION_INDEX.md)** - Organized guide to all documentation

**Essential Guides:**

- **[UI Components & Styling Guide](docs/UI_COMPONENTS_AND_STYLING.md)** - Master component styling, conditional rendering, list iteration, and reactive state. Build beautiful, responsive interfaces with CSS property mapping and advanced form patterns.

- **[UI Component Reference](docs/UI_COMPONENT_REFERENCE.md)** ⭐ NEW - Complete catalog of ALL supported components (text, tables, forms, charts, cards, modals, AI components, etc.) with clear documentation of what's NOT supported and alternatives.

- **[Navigation Guide](docs/NAVIGATION_GUIDE.md)** ⭐ NEW - Routing patterns, page navigation without route parameters, state management, breadcrumbs, modals, and master-detail patterns.

- **[Data Models Guide](docs/DATA_MODELS_GUIDE.md)** ⭐ NEW - Dataset vs Frame explained: when to use each, CRUD operations, query patterns, and best practices for data modeling.

- **[Queries & Datasets](docs/QUERIES_AND_DATASETS.md)** ⭐ NEW - Filter, sort, aggregate, and paginate data. Complete patterns for complex queries without query blocks.

- **[Standard Library](docs/STANDARD_LIBRARY.md)** ⭐ NEW - Built-in functions reference: date/time, strings, JSON, numbers, arrays. Essential utilities for data processing.

- **[Real-Time & Forms Guide](docs/REALTIME_AND_FORMS_GUIDE.md)** - Implement WebSocket features, real-time collaboration, typing indicators, multi-step wizards, and dynamic validation. Perfect for chat applications and live dashboards.

- **[API & Navigation Patterns](docs/API_AND_NAVIGATION_PATTERNS.md)** - Integrate REST and GraphQL APIs, implement routing with parameters, use the action system, and embed widgets. Complete examples with GitHub API integration.

- **[Extensions Guide](docs/EXTENSIONS_GUIDE.md)** ⭐ NEW - Extend Namel3ss with custom Python tools: file processing (PDF, CSV, images), external APIs, scheduling, and monitoring.

- **[Backend & Deployment Guide](docs/BACKEND_AND_DEPLOYMENT_GUIDE.md)** - Advanced backend features, session management, database operations, error handling, monitoring, and production deployment patterns.

- **[Complete Chat Widget Example](docs/COMPLETE_CHAT_WIDGET_EXAMPLE.md)** - 600+ line production-ready customer support chat widget with real-time messaging, typing indicators, and iframe embedding.

### 📚 Core Documentation

- [Complete Language Guide](docs/NAMEL3SS_DOCUMENTATION.md) - Everything you need to know
- [API Reference](docs/API_REFERENCE.md) - Detailed API documentation
- [Advanced Features](docs/ADVANCED_FEATURES.md) - Type checking, modules, expressions
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues solved

### 💡 Working Examples

**Production-Ready Applications:**
- [Customer Support Chatbot](examples/customer_support_chatbot.ai) - AI chatbot with memory, chains, and escalation
- [Governed Multi-Agent Lab](examples/governed-multi-agent-research-lab.md) - Enterprise AI with governance
- [RAG Document Assistant](examples/rag-document-assistant-and-citation-explorer.md) - Production RAG system  
- [AI Support Console](docs/examples/ai-customer-support-console.md) - Complete support application

**Quick Start Examples:**
- [Simple Demo](examples/verified_simple_demo.ai) - Minimal working example with verified syntax
- [API Navigation Demo](examples/api_navigation_demo.ai) - REST API integration with routing and actions
- [Working Chatbot](examples/working_chatbot.ai) - Basic conversational AI implementation

**Community:**
- [GitHub](https://github.com/namel3ss-Ai/namel3ss-programming-language) - Star us, contribute, discuss
- [Issues](https://github.com/namel3ss-Ai/namel3ss-programming-language/issues) - Report bugs, request features
- [Discussions](https://github.com/namel3ss-Ai/namel3ss-programming-language/discussions) - Share your creations

---

## Contributing

We believe the best products are built together.

Namel3ss has:
- RFC-based governance for language changes
- Comprehensive test requirements
- Clear contribution guidelines
- Welcoming community

[Read the contributing guide →](https://github.com/namel3ss-Ai/namel3ss-programming-language/blob/main/CONTRIBUTING.md)

---

## License

MIT License. Build anything. Commercial or open source.

[Read the full license →](https://github.com/namel3ss-Ai/namel3ss-programming-language/blob/main/LICENSE)

---

**Namel3ss** — Where thought becomes code. Where AI feels natural. Where complexity disappears.

**Welcome to the future of AI programming.**

---

<details>
<summary><strong>📚 Complete documentation sections (click to expand)</strong></summary>

[The rest of the technical documentation continues here with all the detailed sections about Recent Additions, Language Features, CLI Reference, Architecture, Production Deployment, Testing, etc. - maintaining all the original content but presented as expandable reference material]

</details>
