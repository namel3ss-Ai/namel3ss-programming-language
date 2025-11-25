# Namel3ss (N3) Programming Language - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [CLI Commands Reference](#cli-commands-reference)
4. [Core Language Features](#core-language-features)
5. [AI-Native Components](#ai-native-components)
6. [Provider System](#provider-system)
7. [Memory and State Management](#memory-and-state-management)
8. [RAG and Knowledge Systems](#rag-and-knowledge-systems)
9. [Evaluation and Testing](#evaluation-and-testing)
10. [Code Generation](#code-generation)
11. [Deployment and Production](#deployment-and-production)
12. [API Reference](#api-reference)
13. [Examples](#examples)

---

## Overview

**Namel3ss (N3)** is an AI-native programming language designed for building conversational AI applications, RAG systems, and AI-powered tools using natural English-like syntax. It compiles to production-ready FastAPI backends and frontend applications with full-stack capabilities including:

- **Declarative AI workflows**: Define prompts, chains, and agent systems
- **Built-in LLM integration**: OpenAI, Anthropic, Google, Azure, and local models
- **Memory systems**: Persistent conversation history and context management
- **RAG pipelines**: Document indexing, retrieval, and generation
- **Evaluation framework**: Testing, metrics, and safety guardrails
- **Real-time capabilities**: WebSocket streaming and live updates

### Key Benefits

- **Rapid prototyping**: Build AI applications in minutes, not hours
- **Production-ready**: Generates scalable FastAPI backends and React frontends
- **AI-first design**: Native support for LLMs, vectors, and AI workflows
- **Type safety**: Structured prompts with input/output schemas
- **Observability**: Built-in logging, metrics, and debugging tools

---

## Installation

### Minimal Installation (Core Only)

```bash
pip install namel3ss
```

This installs the core language parser and basic CLI with only 3 dependencies: `jinja2`, `packaging`, and `click`.

### Full Installation with All Features

```bash
pip install namel3ss[all]
```

Includes all optional dependencies for:
- Local model deployment (Ollama, vLLM, LocalAI)
- Enhanced CLI tools
- Development server capabilities

### Optional Extensions

```bash
# CLI and development server
pip install namel3ss[cli]

# Local model deployment
pip install namel3ss[local-deploy]

# Development server with hot reload
pip install uvicorn[standard]
```

---

## CLI Commands Reference

### Core Commands

#### `namel3ss build` - Generate Application Code

Compile `.ai` files into production-ready backend and frontend code.

```bash
# Generate static frontend only
namel3ss build app.ai

# Generate both frontend and backend
namel3ss build app.ai --build-backend

# Backend only
namel3ss build app.ai --backend-only

# Custom output directories
namel3ss build app.ai --out dist --backend-out api

# React frontend target
namel3ss build app.ai --target react-vite

# Enable real-time WebSocket support
namel3ss build app.ai --realtime

# Export API schemas for SDK generation
namel3ss build app.ai --export-schemas --schema-version 2.0.0

# Embed insight results in responses
namel3ss build app.ai --embed-insights

# Set environment variables for generation
namel3ss build app.ai --env DATABASE_URL=postgres://... --env API_KEY=secret
```

#### `namel3ss run` - Development Server

Start development servers with hot reload and debugging capabilities.

```bash
# Basic dev server
namel3ss run app.ai

# Custom host and port
namel3ss run app.ai --host 0.0.0.0 --port 3000

# Disable hot reload
namel3ss run app.ai --no-reload

# Custom backend directory
namel3ss run app.ai --backend-out ./dev_backend

# Enable real-time features
namel3ss run app.ai --realtime

# Run multiple apps simultaneously
namel3ss run --apps app1,app2 --workspace

# Execute a specific chain
namel3ss run chain MyChain --file app.ai --payload '{"input": "Hello"}'

# Execute with JSON output
namel3ss run chain SummarizeText --json --payload '{"text": "..."}'
```

### AI-Specific Commands

#### `namel3ss eval` - Run Experiments

Execute evaluation experiments and metrics.

```bash
# Run experiment
namel3ss eval MyExperiment --file app.ai

# Output format options
namel3ss eval ModelComparison --format json
namel3ss eval ModelComparison --format text

# Limit evaluations for testing
namel3ss eval-suite TestSuite --limit 10

# Continue on errors
namel3ss eval-suite ProductionSuite --continue-on-error
```

#### `namel3ss train` - Machine Learning Training

Invoke training and tuning jobs for custom models.

```bash
# Run training job
namel3ss train MyTrainingJob --file app.ai

# RLHF training
namel3ss train RLHFJob --file app.ai --output-dir ./models

# Compact JSON output
namel3ss train FineTuning --json-compact
```

#### `namel3ss deploy` - Model Deployment

Deploy models to various environments.

```bash
# Deploy to cloud
namel3ss deploy cloud app.ai --model my_model

# Local deployment with vLLM
namel3ss deploy local start --provider vllm --model llama3

# Ollama deployment
namel3ss deploy local start --provider ollama --model mistral

# Check deployment status
namel3ss deploy local status

# Stop local deployment
namel3ss deploy local stop --model llama3

# Scale deployment
namel3ss deploy local scale --model llama3 --replicas 3
```

### Development Tools

#### `namel3ss doctor` - Health Checks

Inspect dependencies and configuration.

```bash
# Check all dependencies
namel3ss doctor

# Check specific feature availability
namel3ss doctor --check-memory
namel3ss doctor --check-rlhf
```

#### `namel3ss test` - Testing Framework

Run application tests with AI-specific assertions.

```bash
# Run all tests
namel3ss test app.ai

# Specific test targets
namel3ss test app.ai --target prompt:SummarizeText
namel3ss test app.ai --target chain:DocumentProcessor
namel3ss test app.ai --target agent:ResearchAssistant

# Mock LLM providers for testing
namel3ss test app.ai --mock-llm
```

#### Development Utilities

```bash
# Lint N3 code
namel3ss lint app.ai

# Type checking
namel3ss typecheck app.ai

# Code formatting
namel3ss format app.ai

# Language server for IDE integration
namel3ss lsp --stdio
```

### Global Options

```bash
# Verbose output with full tracebacks
namel3ss build app.ai --verbose

# Environment variable: NAMEL3SS_VERBOSE=1

# Re-raise exceptions for debugging
# Environment variable: NAMEL3SS_RERAISE=1
```

---

## Core Language Features

### Application Structure

Every Namel3ss program starts with an `app` declaration:

```namel3ss
app "My Application" {
    description: "A sample AI-powered application"
    version: "1.0.0"
    database: "postgresql://localhost/myapp"
}
```

### Pages and Routing

Define web pages with automatic routing:

```namel3ss
page "Home" at "/" {
    title: "Welcome"
    
    show text "Welcome to my app!" {
        size: "large"
        weight: "bold"
        color: "var(--primary)"
    }
    
    show chart "Monthly Sales" from dataset sales {
        chart_type: "line"
        x: "month"
        y: "revenue"
        color: "product_category"
    }
    
    show button "Generate Report" {
        action: run_chain "ReportGeneration"
        style: "primary"
    }
}

page "Dashboard" at "/dashboard" {
    title: "Analytics Dashboard"
    
    show table from dataset analytics {
        columns: ["date", "users", "revenue"]
        sortable: true
        filterable: true
        pagination: 20
    }
    
    show form "Update Settings" {
        fields: [
            {name: "refresh_interval", type: "number", default: 30},
            {name: "email_alerts", type: "boolean", default: true}
        ]
        submit: run_chain "UpdateSettings"
    }
}
```

### Datasets and Data Sources

Connect to various data sources with built-in caching and transformation:

```namel3ss
dataset "sales_data" from sql {
    query: """
        SELECT 
            DATE_TRUNC('month', order_date) as month,
            SUM(amount) as revenue,
            COUNT(*) as order_count,
            product_category
        FROM orders 
        WHERE order_date >= NOW() - INTERVAL '12 months'
        GROUP BY month, product_category
        ORDER BY month DESC
    """
    cache_ttl: 300  # 5 minutes
    refresh_policy: "on_demand"
}

dataset "user_feedback" from csv {
    file: "data/feedback.csv"
    schema: {
        user_id: "string"
        feedback: "text"
        sentiment: "string"
        rating: "number"
    }
    transformations: [
        {filter: "rating >= 3"},
        {add_column: "processed_date", value: "now()"}
    ]
}

dataset "api_metrics" from rest {
    url: "https://api.analytics.com/metrics"
    headers: {
        "Authorization": "Bearer ${env.API_TOKEN}"
        "Content-Type": "application/json"
    }
    params: {
        "timeframe": "last_30_days"
        "granularity": "daily"
    }
    cache_ttl: 600  # 10 minutes
}
```

### Models and Machine Learning

Define and train custom models:

```namel3ss
model "churn_predictor" {
    type: "classification"
    algorithm: "random_forest"
    
    features: {
        tenure: "numeric"
        monthly_charges: "numeric"
        total_charges: "numeric"
        contract_type: "categorical"
        payment_method: "categorical"
    }
    
    target: "churned"
    
    training: {
        dataset: "customer_data"
        test_split: 0.2
        validation_split: 0.1
        hyperparameters: {
            n_estimators: 100
            max_depth: 10
            min_samples_split: 5
        }
    }
    
    evaluation: {
        metrics: ["accuracy", "precision", "recall", "f1_score"]
        cross_validation: 5
    }
    
    deployment: {
        endpoint: "/predict/churn"
        batch_size: 1000
        monitoring: true
    }
}
```

### Themes and Styling

Customize application appearance:

```namel3ss
theme "corporate" {
    colors: {
        primary: "#1f2937"
        secondary: "#f59e0b"
        accent: "#10b981"
        background: "#f9fafb"
        surface: "#ffffff"
        text: "#111827"
        muted: "#6b7280"
    }
    
    typography: {
        font_family: "Inter, system-ui, sans-serif"
        heading_weight: "600"
        body_weight: "400"
        line_height: 1.6
    }
    
    spacing: {
        xs: "0.25rem"
        sm: "0.5rem"
        md: "1rem"
        lg: "1.5rem"
        xl: "2rem"
    }
    
    shadows: {
        sm: "0 1px 2px 0 rgb(0 0 0 / 0.05)"
        md: "0 4px 6px -1px rgb(0 0 0 / 0.1)"
        lg: "0 10px 15px -3px rgb(0 0 0 / 0.1)"
    }
}
```

---

## AI-Native Components

### LLM Definitions

Define and configure Large Language Models:

```namel3ss
llm "gpt4_creative" {
    provider: "openai"
    model: "gpt-4o"
    temperature: 0.8
    max_tokens: 2000
    top_p: 0.9
    frequency_penalty: 0.1
    presence_penalty: 0.1
    system_prompt: "You are a creative writing assistant."
    safety: {
        content_filter: true
        moderation: true
    }
    tools: ["web_search", "calculator"]
    stream: true
}

llm "claude_analytical" {
    provider: "anthropic"
    model: "claude-3-opus-20240229"
    temperature: 0.2
    max_tokens: 4000
    system_prompt: "You are a data analyst who provides clear, evidence-based insights."
}

llm "local_llama" {
    provider: "ollama"
    model: "llama3:8b"
    temperature: 0.7
    host: "localhost:11434"
}
```

### Structured Prompts

Define prompts with typed inputs and outputs:

```namel3ss
prompt "SummarizeDocument" {
    model: "gpt4_creative"
    
    # Input schema
    input: {
        content: text
        max_length: number = 500
        style: enum["brief", "detailed", "bullet_points"] = "brief"
    }
    
    # Output schema
    output: {
        summary: text
        key_points: list<text>
        word_count: number
        confidence: number
    }
    
    # Model parameters
    temperature: 0.3
    max_tokens: 1000
    
    template: """
    Please summarize the following document in {{style}} style, 
    keeping it under {{max_length}} characters:

    {{content}}

    Provide your response in the following JSON format:
    {
        "summary": "...",
        "key_points": ["...", "..."],
        "word_count": number,
        "confidence": 0.9
    }
    """
}

prompt "GenerateContent" {
    model: "claude_analytical"
    
    input: {
        topic: text
        audience: enum["technical", "general", "beginner"]
        format: enum["article", "tutorial", "faq"] = "article"
        keywords: list<text> = []
    }
    
    output: {
        title: text
        content: text
        meta_description: text
        tags: list<text>
        reading_time: number
    }
    
    template: """
    Create {{format}} content about {{topic}} for {{audience}} audience.
    {% if keywords %}Focus on these keywords: {{keywords | join(", ")}}{% endif %}
    
    Requirements:
    - Engaging and informative
    - SEO-optimized
    - Appropriate for {{audience}} level
    - Include practical examples
    """
}
```

### Tools and Functions

Define custom tools for LLM use:

```namel3ss
tool "web_search" {
    description: "Search the web for current information"
    
    parameters: {
        query: {
            type: "string"
            description: "Search query"
            required: true
        }
        num_results: {
            type: "integer"
            description: "Number of results to return"
            default: 5
            maximum: 20
        }
    }
    
    implementation: {
        type: "http"
        url: "https://api.search.com/v1/search"
        method: "GET"
        headers: {
            "Authorization": "Bearer ${env.SEARCH_API_KEY}"
        }
        params: {
            "q": "{{query}}"
            "limit": "{{num_results}}"
        }
    }
}

tool "calculate_metrics" {
    description: "Calculate business metrics from data"
    
    parameters: {
        data: {
            type: "array"
            description: "Array of numeric values"
            required: true
        }
        metric_type: {
            type: "string"
            enum: ["mean", "median", "sum", "growth_rate"]
            required: true
        }
    }
    
    implementation: {
        type: "python"
        code: """
def calculate_metrics(data, metric_type):
    import statistics
    import numpy as np
    
    if metric_type == "mean":
        return statistics.mean(data)
    elif metric_type == "median":
        return statistics.median(data)
    elif metric_type == "sum":
        return sum(data)
    elif metric_type == "growth_rate":
        if len(data) < 2:
            return 0
        return ((data[-1] - data[0]) / data[0]) * 100
    
    return None
        """
    }
}
```

### Chains and Workflows

Create multi-step AI workflows:

```namel3ss
chain "ContentCreationPipeline" {
    description: "End-to-end content creation with review"
    
    input: {
        topic: text
        target_audience: text
        content_type: text
    }
    
    output: {
        final_content: text
        metadata: object
        approval_status: text
    }
    
    steps: [
        step "research" {
            kind: "prompt"
            target: "ResearchTopic"
            options: {
                topic: input.topic
                depth: "comprehensive"
            }
            output_to: "research_data"
        },
        
        step "generate_outline" {
            kind: "prompt"
            target: "CreateOutline"
            options: {
                topic: input.topic
                research: context.research_data
                audience: input.target_audience
            }
            output_to: "content_outline"
        },
        
        step "write_content" {
            kind: "prompt"
            target: "GenerateContent"
            options: {
                outline: context.content_outline
                type: input.content_type
                audience: input.target_audience
            }
            output_to: "draft_content"
        },
        
        step "review_content" {
            kind: "prompt"
            target: "ReviewContent"
            options: {
                content: context.draft_content
                criteria: ["accuracy", "readability", "engagement"]
            }
            output_to: "review_results"
        },
        
        step "finalize" {
            kind: "template"
            template: """
            {
                "final_content": "{{context.draft_content}}",
                "metadata": {
                    "word_count": {{context.draft_content | length}},
                    "review_score": {{context.review_results.score}},
                    "created_at": "{{now()}}"
                },
                "approval_status": "{% if context.review_results.score >= 8 %}approved{% else %}needs_revision{% endif %}"
            }
            """
        }
    ]
    
    # Error handling
    error_handling: {
        retry_failed_steps: 2
        continue_on_error: false
        fallback_chain: "SimpleContentGeneration"
    }
    
    # Memory integration
    memory: {
        read: ["content_preferences", "style_guide"]
        write: ["content_history"]
    }
}

chain "CustomerSupportFlow" {
    description: "Intelligent customer support with escalation"
    
    input: {
        message: text
        customer_id: text
        urgency: enum["low", "medium", "high"] = "medium"
    }
    
    steps: [
        step "classify_intent" {
            kind: "prompt"
            target: "ClassifyIntent"
            options: {
                message: input.message
            }
            output_to: "intent_result"
        },
        
        step "retrieve_context" {
            kind: "rag"
            target: "support_knowledge"
            options: {
                query: input.message
                filters: {
                    category: context.intent_result.category
                }
                top_k: 5
            }
            output_to: "knowledge_context"
        },
        
        step "generate_response" {
            kind: "prompt"
            target: "SupportResponse"
            options: {
                message: input.message
                intent: context.intent_result
                knowledge: context.knowledge_context
                customer_id: input.customer_id
            }
            output_to: "response"
            
            # Conditional execution
            conditions: {
                execute_if: "context.intent_result.confidence > 0.7"
            }
        },
        
        step "escalate" {
            kind: "action"
            target: "CreateTicket"
            options: {
                message: input.message
                customer_id: input.customer_id
                reason: "Low confidence classification"
            }
            conditions: {
                execute_if: "context.intent_result.confidence <= 0.7 or input.urgency == 'high'"
            }
        }
    ]
    
    # Evaluation and monitoring
    evaluation: {
        evaluators: ["response_quality", "helpfulness"]
        guardrails: ["safety_check", "brand_compliance"]
    }
}
```

---

## Provider System

Namel3ss supports multiple LLM providers with unified configuration:

### OpenAI Configuration

```bash
export NAMEL3SS_PROVIDER_OPENAI_API_KEY="sk-..."
export NAMEL3SS_PROVIDER_OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional
```

```namel3ss
llm "gpt4" {
    provider: "openai"
    model: "gpt-4o"
    api_key: env.OPENAI_API_KEY  # Or use environment variable
    temperature: 0.7
    max_tokens: 2000
}
```

### Anthropic Configuration

```bash
export NAMEL3SS_PROVIDER_ANTHROPIC_API_KEY="sk-ant-..."
```

```namel3ss
llm "claude" {
    provider: "anthropic"
    model: "claude-3-opus-20240229"
    temperature: 0.5
    max_tokens: 4000
}
```

### Google/Vertex AI

```bash
export NAMEL3SS_PROVIDER_GOOGLE_PROJECT_ID="my-project"
export NAMEL3SS_PROVIDER_GOOGLE_LOCATION="us-central1"
```

```namel3ss
llm "gemini" {
    provider: "google"
    model: "gemini-pro"
    project_id: env.GOOGLE_PROJECT_ID
    location: "us-central1"
}
```

### Azure OpenAI

```bash
export NAMEL3SS_PROVIDER_AZURE_API_KEY="..."
export NAMEL3SS_PROVIDER_AZURE_ENDPOINT="https://my-resource.openai.azure.com"
```

```namel3ss
llm "azure_gpt" {
    provider: "azure_openai"
    model: "gpt-4"
    api_version: "2024-02-01"
    deployment_name: "my-gpt4-deployment"
}
```

### Local Providers

#### Ollama

```bash
# Install and run Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull llama3
```

```namel3ss
llm "local_llama" {
    provider: "ollama"
    model: "llama3:8b"
    host: "localhost:11434"
    temperature: 0.7
}
```

#### vLLM

```bash
# Deploy with vLLM
namel3ss deploy local start --provider vllm --model microsoft/DialoGPT-medium
```

```namel3ss
llm "vllm_model" {
    provider: "vllm"
    model: "microsoft/DialoGPT-medium"
    host: "localhost:8000"
    tensor_parallel_size: 2
}
```

#### LocalAI

```bash
# Deploy with LocalAI
namel3ss deploy local start --provider localai --model ggml-gpt4all-j
```

```namel3ss
llm "localai_model" {
    provider: "localai"
    model: "ggml-gpt4all-j"
    host: "localhost:8080"
}
```

### HTTP Provider (Custom APIs)

```namel3ss
llm "custom_api" {
    provider: "http"
    endpoint: "https://my-model-api.com/v1/chat/completions"
    headers: {
        "Authorization": "Bearer ${env.CUSTOM_API_KEY}"
        "Content-Type": "application/json"
    }
    request_format: "openai"  # or "custom"
    timeout: 60
}
```

---

## Memory and State Management

Namel3ss provides persistent memory systems for stateful AI applications:

### Memory Types

#### Conversation Memory

```namel3ss
memory "chat_history" {
    scope: "user"  # Per user
    kind: "conversation"
    max_messages: 100
    ttl: 86400  # 24 hours
    metadata: {
        description: "User conversation history"
        compression: "summarize_old"  # Compress old messages
    }
}
```

#### Key-Value Memory

```namel3ss
memory "user_preferences" {
    scope: "user"
    kind: "key_value"
    max_items: 1000
    metadata: {
        description: "User settings and preferences"
        eviction_policy: "lru"
    }
}

memory "session_data" {
    scope: "session"
    kind: "key_value"
    ttl: 3600  # 1 hour
}
```

#### List Memory

```namel3ss
memory "recent_searches" {
    scope: "user"
    kind: "list"
    max_items: 50
    metadata: {
        description: "Track recent search queries"
        eviction_policy: "fifo"
    }
}
```

#### Vector Memory

```namel3ss
memory "semantic_memory" {
    scope: "global"
    kind: "vector"
    dimensions: 1536
    similarity_threshold: 0.8
    max_items: 10000
    metadata: {
        description: "Semantic memory for context retrieval"
        embedding_model: "text-embedding-ada-002"
    }
}
```

### Using Memory in Chains

```namel3ss
chain "PersonalizedChat" {
    input: {
        message: text
        user_id: text
    }
    
    steps: [
        step "load_context" {
            kind: "memory_read"
            target: "chat_history"
            options: {
                scope_key: input.user_id
                limit: 20
            }
            output_to: "conversation_context"
        },
        
        step "load_preferences" {
            kind: "memory_read"
            target: "user_preferences"
            options: {
                scope_key: input.user_id
                keys: ["communication_style", "topics_of_interest"]
            }
            output_to: "user_prefs"
        },
        
        step "generate_response" {
            kind: "prompt"
            target: "ChatResponse"
            options: {
                message: input.message
                history: context.conversation_context
                preferences: context.user_prefs
            }
            output_to: "response"
        },
        
        step "save_interaction" {
            kind: "memory_write"
            target: "chat_history"
            options: {
                scope_key: input.user_id
                messages: [
                    {role: "user", content: input.message},
                    {role: "assistant", content: context.response}
                ]
            }
        }
    ]
}
```

### Memory Configuration

```namel3ss
# Global memory settings in app configuration
app "ChatBot" {
    memory_config: {
        default_provider: "redis"
        redis: {
            host: "localhost"
            port: 6379
            db: 0
            password: env.REDIS_PASSWORD
        }
        backup_provider: "sqlite"
        sqlite: {
            database: "memory.db"
        }
    }
}
```

---

## RAG and Knowledge Systems

### Index Definitions

Create vector indices for document retrieval:

```namel3ss
index "product_docs" {
    description: "Product documentation and FAQs"
    
    # Embedding configuration
    embedding: {
        model: "text-embedding-ada-002"
        provider: "openai"
        dimensions: 1536
    }
    
    # Data sources
    sources: [
        {
            type: "files"
            path: "docs/**/*.md"
            metadata_fields: ["title", "category", "last_updated"]
        },
        {
            type: "url"
            urls: ["https://docs.myapp.com/api-reference"]
            crawl_depth: 2
        },
        {
            type: "database"
            query: "SELECT title, content, category FROM articles WHERE published = true"
            content_field: "content"
            metadata_fields: ["title", "category"]
        }
    ]
    
    # Chunking strategy
    chunking: {
        strategy: "recursive"
        chunk_size: 1000
        chunk_overlap: 200
        separators: ["\n\n", "\n", ".", "?", "!"]
    }
    
    # Vector store configuration
    vector_store: {
        type: "chroma"  # or "pinecone", "weaviate", "qdrant"
        collection_name: "product_docs"
        distance_metric: "cosine"
    }
    
    # Update configuration
    refresh_policy: "daily"
    incremental_updates: true
}

index "customer_feedback" {
    description: "Customer feedback and reviews"
    
    embedding: {
        model: "sentence-transformers/all-mpnet-base-v2"
        provider: "local"
    }
    
    sources: [
        {
            type: "dataset"
            dataset: "feedback_data"
            content_field: "feedback_text"
            metadata_fields: ["rating", "product", "date"]
        }
    ]
    
    chunking: {
        strategy: "sentence"
        max_sentences: 3
        overlap_sentences: 1
    }
    
    vector_store: {
        type: "sqlite_vss"
        database: "feedback_index.db"
    }
}
```

### RAG Pipelines

Define retrieval-augmented generation workflows:

```namel3ss
rag_pipeline "DocumentQA" {
    description: "Answer questions using product documentation"
    
    # Retrieval configuration
    retrieval: {
        index: "product_docs"
        strategy: "hybrid"  # semantic + keyword
        top_k: 5
        similarity_threshold: 0.7
        
        # Reranking
        reranker: {
            model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
            top_k: 3
        }
        
        # Query preprocessing
        query_preprocessing: [
            "expand_acronyms",
            "extract_entities",
            "rephrase_for_search"
        ]
    }
    
    # Generation configuration
    generation: {
        model: "gpt4"
        temperature: 0.1
        max_tokens: 1000
        
        system_prompt: """
        You are a helpful documentation assistant. Answer user questions 
        using only the provided context. If you cannot answer based on 
        the context, say so clearly.
        
        Guidelines:
        - Be accurate and precise
        - Include relevant examples when available
        - Reference specific sections when helpful
        - If context is insufficient, ask for clarification
        """
        
        prompt_template: """
        Context:
        {% for doc in context %}
        ## {{doc.metadata.title}}
        {{doc.content}}
        
        {% endfor %}
        
        Question: {{query}}
        
        Answer based on the context above:
        """
    }
    
    # Post-processing
    post_processing: [
        "add_source_citations",
        "validate_hallucinations",
        "format_markdown"
    ]
    
    # Evaluation
    evaluation: {
        metrics: ["answer_relevance", "context_precision", "faithfulness"]
        ground_truth_dataset: "qa_golden_set"
    }
}

rag_pipeline "ResearchAssistant" {
    description: "Multi-source research and synthesis"
    
    retrieval: {
        # Multiple indices
        indices: [
            {
                index: "academic_papers"
                weight: 0.4
                top_k: 3
            },
            {
                index: "industry_reports"
                weight: 0.3
                top_k: 2
            },
            {
                index: "news_articles"
                weight: 0.3
                top_k: 2
                filters: {
                    date: "last_30_days"
                }
            }
        ]
        
        # Fusion strategy for combining results
        fusion_strategy: "rrf"  # Reciprocal Rank Fusion
    }
    
    generation: {
        model: "claude_analytical"
        temperature: 0.3
        max_tokens: 2000
        
        prompt_template: """
        Research Question: {{query}}
        
        Sources:
        {% for doc in context %}
        ### {{doc.metadata.source}} - {{doc.metadata.title}}
        {{doc.content}}
        
        {% endfor %}
        
        Provide a comprehensive research summary that:
        1. Synthesizes information from all sources
        2. Identifies key themes and patterns
        3. Notes any conflicting information
        4. Suggests areas for further research
        5. Includes proper citations
        """
    }
}
```

### Knowledge Modules

Define logical knowledge bases:

```namel3ss
knowledge "business_rules" {
    description: "Business logic and validation rules"
    
    # Facts
    facts: [
        "customer(X) :- has_account(X).",
        "premium_customer(X) :- customer(X), annual_spend(X, Amount), Amount > 10000.",
        "eligible_for_discount(X) :- premium_customer(X), active_subscription(X).",
        "requires_approval(X) :- refund_amount(X, Amount), Amount > 500."
    ]
    
    # Rules
    rules: [
        {
            name: "discount_calculation"
            condition: "eligible_for_discount(Customer)"
            action: "apply_discount(Customer, 0.1)"
        },
        {
            name: "fraud_detection"
            condition: "transaction_amount(Transaction, Amount), Amount > 1000, unusual_pattern(Transaction)"
            action: "flag_for_review(Transaction)"
        }
    ]
}

knowledge "product_catalog" {
    description: "Product information and compatibility"
    
    facts: [
        "product(laptop_pro, electronics, 1299.99).",
        "product(wireless_mouse, electronics, 29.99).",
        "compatible(laptop_pro, wireless_mouse).",
        "in_stock(laptop_pro, 50).",
        "in_stock(wireless_mouse, 200)."
    ]
    
    queries: [
        {
            name: "compatible_products"
            query: "compatible(Product, ?Compatible)"
            description: "Find products compatible with a given product"
        },
        {
            name: "available_products"
            query: "product(?Name, ?Category, ?Price), in_stock(?Name, Stock), Stock > 0"
            description: "Get all available products with stock"
        }
    ]
}
```

---

## Evaluation and Testing

### Evaluators

Define evaluation functions for AI outputs:

```namel3ss
evaluator "toxicity_checker" {
    kind: "safety"
    provider: "perspective_api"
    config: {
        api_key: env.PERSPECTIVE_API_KEY
        threshold: 0.8
        attributes: ["TOXICITY", "SEVERE_TOXICITY", "HARASSMENT"]
    }
}

evaluator "relevance_scorer" {
    kind: "quality"
    provider: "custom"
    config: {
        model: "gpt4"
        criteria: [
            "Does the response directly address the question?",
            "Is the information accurate and up-to-date?",
            "Is the response helpful and actionable?"
        ]
        scale: {min: 1, max: 5}
    }
}

evaluator "fact_checker" {
    kind: "accuracy"
    provider: "custom"
    config: {
        knowledge_base: "verified_facts"
        llm_judge: "claude"
        confidence_threshold: 0.9
    }
}
```

### Metrics

Aggregate evaluator outputs into metrics:

```namel3ss
metric "safety_score" {
    evaluator: "toxicity_checker"
    aggregation: "mean"
    threshold: 0.95  # 95% of responses should pass safety check
}

metric "average_relevance" {
    evaluator: "relevance_scorer"
    aggregation: "mean"
    params: {
        weight_by: "confidence"
    }
}

metric "fact_accuracy_rate" {
    evaluator: "fact_checker"
    aggregation: "percentage_above_threshold"
    params: {
        threshold: 0.8
    }
}
```

### Guardrails

Define safety policies with automatic actions:

```namel3ss
guardrail "content_safety" {
    evaluators: ["toxicity_checker", "hate_speech_detector"]
    action: "block"
    message: "Content blocked due to safety policy violation"
}

guardrail "quality_gate" {
    evaluators: ["relevance_scorer", "fact_checker"]
    action: "warn"
    message: "Response quality is below threshold"
    conditions: {
        relevance_scorer: "<3"
        fact_checker: "<0.7"
    }
}

guardrail "compliance_check" {
    evaluators: ["pii_detector", "gdpr_compliance"]
    action: "log_only"
    message: "Compliance issue detected"
}
```

### Evaluation Suites

Comprehensive evaluation configurations:

```namel3ss
eval_suite "customer_support_eval" {
    description: "Evaluate customer support chain performance"
    
    # Dataset for evaluation
    dataset: "support_test_cases"
    
    # Target chain to evaluate
    target_chain: "CustomerSupportFlow"
    
    # Built-in metrics
    metrics: [
        {name: "latency", type: "builtin"},
        {name: "cost", type: "builtin"},
        {name: "safety", type: "custom", evaluator: "toxicity_checker"},
        {name: "helpfulness", type: "llm_judge", judge: "helpfulness_judge"}
    ]
    
    # Judge configuration
    judge: {
        model: "gpt4"
        rubric: {
            helpfulness: {
                description: "How helpful is the response to the customer?"
                scale: {min: 1, max: 5}
                criteria: [
                    "1: Not helpful at all, irrelevant or harmful",
                    "2: Minimally helpful, partially addresses the issue",
                    "3: Moderately helpful, addresses main points",
                    "4: Very helpful, comprehensive and accurate",
                    "5: Extremely helpful, goes above and beyond"
                ]
            },
            accuracy: {
                description: "Is the factual information correct?"
                scale: {min: 1, max: 5}
            }
        }
    }
    
    # Execution configuration
    batch_size: 10
    continue_on_error: true
    timeout: 120  # seconds per evaluation
    
    # Reporting
    report_format: "detailed"
    export_results: "eval_results.json"
}
```

### Running Evaluations

```bash
# Run evaluation suite
namel3ss eval-suite customer_support_eval --file app.ai

# Run with specific parameters
namel3ss eval-suite customer_support_eval --limit 50 --batch-size 5

# Compare multiple chains
namel3ss eval experiment ChainComparison --file app.ai

# Export detailed results
namel3ss eval-suite customer_support_eval --output results.json --format detailed
```

---

## Code Generation

Namel3ss compiles to production-ready applications:

### Backend Generation (FastAPI)

Generated backend includes:

```
backend/
├── main.py                 # FastAPI application entry point
├── database.py            # Database configuration and models
├── generated/
│   ├── __init__.py
│   ├── runtime.py         # Core runtime and state
│   ├── schemas.py         # Pydantic models
│   └── routers/
│       ├── __init__.py
│       ├── pages.py       # Page endpoints
│       ├── datasets.py    # Dataset CRUD endpoints
│       ├── models.py      # ML model endpoints
│       ├── chains.py      # AI chain endpoints
│       └── insights.py    # Analytics endpoints
└── custom/                # User customization space
    ├── __init__.py
    ├── middleware.py      # Custom middleware
    └── routes/
        └── custom_api.py  # Additional endpoints
```

### Frontend Generation

#### Static HTML/JavaScript

```bash
namel3ss build app.ai --target static
```

Generates lightweight, dependency-free frontend with:
- Vanilla JavaScript
- Chart.js for visualizations
- Responsive CSS grid layouts
- Progressive enhancement

#### React Application

```bash
namel3ss build app.ai --target react-vite
```

Generates modern React application with:
- TypeScript support
- Vite build system
- React Query for API state
- Tailwind CSS styling
- Component library integration

### Real-time Features

```bash
namel3ss build app.ai --realtime
```

Adds WebSocket support for:
- Live data updates
- Real-time chat interfaces
- Progress notifications
- Collaborative features

### API Schema Export

```bash
namel3ss build app.ai --export-schemas --schema-version 2.0.0
```

Generates:
- OpenAPI 3.0 specifications
- TypeScript client SDK
- Python client SDK
- API documentation

---

## Deployment and Production

### Local Development

```bash
# Start development server with hot reload
namel3ss run app.ai --host 0.0.0.0 --port 8000

# Enable debugging and verbose output
NAMEL3SS_VERBOSE=1 namel3ss run app.ai

# Custom environment variables
namel3ss run app.ai --env DATABASE_URL=postgres://... --env API_KEY=secret
```

### Production Deployment

#### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Build namel3ss application
RUN namel3ss build app.ai --build-backend --backend-out backend

# Run production server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: namel3ss-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: namel3ss-app
  template:
    metadata:
      labels:
        app: namel3ss-app
    spec:
      containers:
      - name: app
        image: my-namel3ss-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

#### Cloud Platforms

##### Vercel (Frontend)

```json
{
  "name": "namel3ss-frontend",
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/node"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ]
}
```

##### Railway/Heroku (Backend)

```yaml
# railway.toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uvicorn backend.main:app --host 0.0.0.0 --port $PORT"

[variables]
PYTHON_VERSION = "3.11"
```

### Environment Configuration

```bash
# Production environment variables
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
export REDIS_URL="redis://localhost:6379"
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Memory configuration
export NAMEL3SS_MEMORY_PROVIDER="redis"
export NAMEL3SS_MEMORY_TTL="86400"

# Vector database
export CHROMA_HOST="localhost"
export CHROMA_PORT="8000"
export PINECONE_API_KEY="..."

# Observability
export NAMEL3SS_LOG_LEVEL="INFO"
export NAMEL3SS_ENABLE_TRACING="true"
export JAEGER_ENDPOINT="http://jaeger:14268/api/traces"
```

### Performance Optimization

#### Caching

```namel3ss
app "OptimizedApp" {
    cache_config: {
        provider: "redis"
        default_ttl: 300
        cache_layers: ["memory", "redis"]
        
        # Cache strategies
        dataset_cache: {
            ttl: 600
            strategy: "write_through"
        }
        
        llm_cache: {
            ttl: 3600
            strategy: "cache_aside"
            key_template: "llm:{{model}}:{{hash}}"
        }
    }
}
```

#### Database Optimization

```namel3ss
app "ScaledApp" {
    database_config: {
        pool_size: 20
        max_overflow: 30
        pool_timeout: 30
        read_replicas: ["replica1:5432", "replica2:5432"]
        
        # Connection optimization
        connection_args: {
            "statement_timeout": "30s"
            "idle_in_transaction_session_timeout": "60s"
        }
    }
}
```

---

## API Reference

### Core Classes

#### Parser

```python
from namel3ss.parser import Parser

# Parse N3 source code
parser = Parser(source_code, path="app.ai")
module = parser.parse()  # Returns Module AST
app = parser.parse_app()  # Returns App AST directly
```

#### Loader

```python
from namel3ss.loader import load_program

# Load entire program with imports
program = load_program("./src")  # Directory with .ai files
```

#### Resolver

```python
from namel3ss.resolver import resolve_program

# Resolve imports and references
resolved_program = resolve_program(program)
```

#### Code Generation

```python
from namel3ss.codegen.backend import generate_backend
from namel3ss.codegen.frontend import generate_site

# Generate backend
generate_backend(
    app,
    output_dir="./backend",
    enable_realtime=True,
    export_schemas=True
)

# Generate frontend
generate_site(
    app,
    output_dir="./frontend",
    target="react-vite",
    enable_realtime=True
)
```

### Provider Integration

```python
from namel3ss.providers import create_provider_from_spec, ProviderMessage
from namel3ss.providers.integration import run_chain_with_provider

# Create provider
provider = create_provider_from_spec(
    provider_type="openai",
    model="gpt-4",
    config={"temperature": 0.7}
)

# Execute chain with provider
result = await run_chain_with_provider(
    chain_steps=chain.steps,
    provider=provider,
    initial_input={"question": "What is AI?"}
)
```

### Evaluation System

```python
from namel3ss.eval import EvalSuiteRunner, create_metric
from namel3ss.eval.judge import LLMJudge

# Create metrics
latency_metric = create_metric("latency", "builtin")
quality_metric = create_metric("quality", "llm_judge", judge_config={...})

# Create judge
judge = LLMJudge(model_provider=provider, rubric=rubric_config)

# Run evaluation
runner = EvalSuiteRunner(
    suite_name="test_suite",
    dataset_rows=test_data,
    chain_executor=chain_executor,
    metrics=[latency_metric, quality_metric],
    judge=judge
)

results = await runner.run_suite()
```

### Memory System

```python
from namel3ss.memory import MemoryManager, MemoryConfig

# Configure memory
config = MemoryConfig(
    provider="redis",
    redis_url="redis://localhost:6379",
    default_ttl=86400
)

memory = MemoryManager(config)

# Use memory
await memory.set("user:123:preferences", {"theme": "dark"})
prefs = await memory.get("user:123:preferences")

# List operations
await memory.lpush("user:123:history", "last_search")
history = await memory.lrange("user:123:history", 0, 10)
```

---

## Examples

### Complete Chat Application

```namel3ss
app "AI Chat Assistant" {
    description: "Intelligent chat with memory and tools"
    database: env.DATABASE_URL
}

# Theme
theme "modern" {
    colors: {
        primary: "#2563eb"
        secondary: "#f59e0b"
        background: "#ffffff"
        surface: "#f8fafc"
    }
}

# LLM Configuration
llm "chat_model" {
    provider: "openai"
    model: "gpt-4o"
    temperature: 0.7
    max_tokens: 2000
    system_prompt: "You are a helpful AI assistant with access to various tools."
    tools: ["web_search", "calculator", "weather"]
}

# Memory for conversations
memory "conversation_history" {
    scope: "user"
    kind: "conversation"
    max_messages: 50
    ttl: 86400  # 24 hours
}

# Tools
tool "web_search" {
    description: "Search the web for current information"
    parameters: {
        query: {type: "string", required: true}
        num_results: {type: "integer", default: 5}
    }
    implementation: {
        type: "http"
        url: "https://api.tavily.com/search"
        headers: {"Authorization": "Bearer ${env.TAVILY_API_KEY}"}
    }
}

# Chat response prompt
prompt "ChatResponse" {
    model: "chat_model"
    
    input: {
        message: text
        history: list<object> = []
        user_context: object = {}
    }
    
    output: {
        response: text
        tool_calls: list<object> = []
        confidence: number
    }
    
    template: """
    {% if history %}
    Previous conversation:
    {% for msg in history[-5:] %}
    {{msg.role}}: {{msg.content}}
    {% endfor %}
    {% endif %}
    
    User: {{message}}
    
    Please respond helpfully and use tools if needed for current information.
    """
}

# Chat chain
chain "ChatFlow" {
    input: {
        message: text
        user_id: text
    }
    
    steps: [
        step "load_history" {
            kind: "memory_read"
            target: "conversation_history"
            options: {
                scope_key: input.user_id
                limit: 10
            }
            output_to: "history"
        },
        
        step "generate_response" {
            kind: "prompt"
            target: "ChatResponse"
            options: {
                message: input.message
                history: context.history
            }
            output_to: "response"
        },
        
        step "save_conversation" {
            kind: "memory_write"
            target: "conversation_history"
            options: {
                scope_key: input.user_id
                messages: [
                    {role: "user", content: input.message},
                    {role: "assistant", content: context.response.response}
                ]
            }
        }
    ]
    
    evaluation: {
        evaluators: ["safety_check", "helpfulness"]
    }
}

# Web interface
page "Chat" at "/" {
    title: "AI Assistant"
    
    show text "AI Chat Assistant" {
        size: "x-large"
        weight: "bold"
        align: "center"
        style: {margin_bottom: "2rem"}
    }
    
    show chat_interface {
        chain: "ChatFlow"
        placeholder: "Ask me anything..."
        height: "500px"
        features: {
            typing_indicator: true
            message_history: true
            export_chat: true
        }
        styling: {
            user_bubble_color: "var(--primary)"
            assistant_bubble_color: "var(--surface)"
            border_radius: "12px"
        }
    }
}
```

### RAG-Powered Documentation Assistant

```namel3ss
app "Documentation Assistant" {
    description: "Smart documentation search and Q&A"
}

# Document index
index "docs" {
    description: "Product documentation and guides"
    
    embedding: {
        model: "text-embedding-ada-002"
        provider: "openai"
    }
    
    sources: [
        {
            type: "files"
            path: "docs/**/*.{md,rst,txt}"
            metadata_fields: ["title", "section", "last_updated"]
        }
    ]
    
    chunking: {
        strategy: "recursive"
        chunk_size: 1000
        chunk_overlap: 200
    }
    
    vector_store: {
        type: "chroma"
        collection_name: "documentation"
    }
}

# RAG pipeline
rag_pipeline "DocQA" {
    retrieval: {
        index: "docs"
        top_k: 5
        similarity_threshold: 0.7
        
        reranker: {
            model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
            top_k: 3
        }
    }
    
    generation: {
        model: "gpt4"
        temperature: 0.1
        
        system_prompt: """
        You are a helpful documentation assistant. Answer questions using 
        only the provided context. Be precise and include relevant examples.
        """
        
        prompt_template: """
        Context:
        {% for doc in context %}
        ## {{doc.metadata.title}}
        {{doc.content}}
        {% endfor %}
        
        Question: {{query}}
        
        Answer:
        """
    }
}

# Search interface
page "Documentation" at "/" {
    title: "Documentation Search"
    
    show search_interface {
        rag_pipeline: "DocQA"
        placeholder: "Search documentation..."
        features: {
            autocomplete: true
            filters: ["section", "last_updated"]
            export_results: true
        }
        result_template: """
        <div class="search-result">
            <h3>{{title}}</h3>
            <p>{{snippet}}</p>
            <div class="metadata">
                <span class="section">{{section}}</span>
                <span class="updated">Updated: {{last_updated}}</span>
            </div>
        </div>
        """
    }
}
```

### E-commerce Analytics Dashboard

```namel3ss
app "E-commerce Analytics" {
    description: "Real-time e-commerce analytics and insights"
    database: "postgresql://localhost/ecommerce"
}

# Sales dataset
dataset "sales_metrics" from sql {
    query: """
        SELECT 
            DATE_TRUNC('day', order_date) as date,
            SUM(total_amount) as revenue,
            COUNT(DISTINCT order_id) as orders,
            COUNT(DISTINCT customer_id) as customers,
            AVG(total_amount) as avg_order_value,
            product_category
        FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
        JOIN products p ON oi.product_id = p.id
        WHERE order_date >= CURRENT_DATE - INTERVAL '90 days'
        GROUP BY DATE_TRUNC('day', order_date), product_category
        ORDER BY date DESC
    """
    cache_ttl: 300
    refresh_policy: "on_access"
}

# Customer insights
dataset "customer_segments" from sql {
    query: """
        WITH customer_metrics AS (
            SELECT 
                customer_id,
                COUNT(*) as total_orders,
                SUM(total_amount) as lifetime_value,
                MAX(order_date) as last_order_date,
                MIN(order_date) as first_order_date
            FROM orders
            GROUP BY customer_id
        )
        SELECT 
            customer_id,
            CASE 
                WHEN lifetime_value > 1000 THEN 'VIP'
                WHEN lifetime_value > 500 THEN 'Regular'
                ELSE 'New'
            END as segment,
            total_orders,
            lifetime_value,
            EXTRACT(days FROM CURRENT_DATE - last_order_date) as days_since_last_order
        FROM customer_metrics
    """
    cache_ttl: 600
}

# Predictive model
model "churn_predictor" {
    type: "classification"
    algorithm: "random_forest"
    
    features: {
        days_since_last_order: "numeric"
        total_orders: "numeric"
        lifetime_value: "numeric"
        avg_order_value: "numeric"
    }
    
    target: "will_churn"
    
    training: {
        dataset: "customer_segments"
        hyperparameters: {
            n_estimators: 100
            max_depth: 10
        }
    }
}

# Analytics insights
insight "revenue_trend" from dataset sales_metrics:
    trend("revenue", period: "daily", lookback: 30)
    
insight "top_categories" from dataset sales_metrics:
    top_n("product_category", metric: "revenue", limit: 5)

# Dashboard page
page "Dashboard" at "/" {
    title: "E-commerce Analytics"
    
    # KPI Cards
    show metric_card "Total Revenue" {
        value: sum(sales_metrics.revenue)
        format: "currency"
        change: percent_change(30_days)
        trend: "up"
    }
    
    show metric_card "Orders Today" {
        value: count(sales_metrics.orders, filter: "date = today()")
        format: "number"
        change: percent_change(yesterday)
    }
    
    # Charts
    show chart "Revenue Trend" from dataset sales_metrics {
        chart_type: "line"
        x: "date"
        y: "revenue"
        color: "product_category"
        title: "Daily Revenue by Category"
        height: "400px"
    }
    
    show chart "Customer Segments" from dataset customer_segments {
        chart_type: "pie"
        category: "segment"
        value: "count(*)"
        title: "Customer Distribution"
    }
    
    # Tables
    show table "Top Products" from dataset sales_metrics {
        columns: ["product_category", "revenue", "orders"]
        sort_by: "revenue"
        sort_order: "desc"
        limit: 10
        pagination: false
    }
    
    # Interactive filters
    show filter_panel {
        filters: [
            {
                field: "date"
                type: "date_range"
                default: "last_30_days"
            },
            {
                field: "product_category"
                type: "multiselect"
                options: "auto"
            }
        ]
    }
}

# Automated insights
chain "GenerateInsights" {
    input: {
        time_period: text = "last_30_days"
    }
    
    steps: [
        step "analyze_trends" {
            kind: "prompt"
            target: "AnalyzeTrends"
            options: {
                data: dataset.sales_metrics
                period: input.time_period
            }
            output_to: "trend_analysis"
        },
        
        step "identify_opportunities" {
            kind: "prompt" 
            target: "FindOpportunities"
            options: {
                trends: context.trend_analysis
                customer_data: dataset.customer_segments
            }
            output_to: "opportunities"
        }
    ]
}
```

---

This comprehensive documentation covers all major aspects of the Namel3ss programming language, from basic CLI usage to advanced AI-powered features. The language is designed to make AI application development accessible while maintaining the power and flexibility needed for production systems.