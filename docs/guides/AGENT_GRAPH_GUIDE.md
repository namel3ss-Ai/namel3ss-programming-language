# Agent and Graph Guide

## Overview

Namel3ss supports **first-class agents and graphs** for building multi-agent AI systems. Agents are autonomous LLM-powered entities that can use tools and maintain memory, while graphs orchestrate multiple agents with conditional routing and state management.

This guide covers both the language constructs and the visual graph editor for building agent workflows.

---

## Table of Contents

1. [Agent Blocks](#agent-blocks)
2. [Graph Blocks](#graph-blocks)
3. [Invoking Graphs from Chains](#invoking-graphs-from-chains)
4. [Memory Management](#memory-management)
5. [Routing and Conditions](#routing-and-conditions)
6. [Best Practices](#best-practices)
7. [Language Examples](#language-examples)
8. [Visual Graph Editor](#visual-graph-editor)
9. [Editor Features](#editor-features)
10. [Editor Quick Start](#editor-quick-start)
11. [Real-time Collaboration](#real-time-collaboration)
12. [Tool Registry](#tool-registry)
13. [Adaptive Policies (RLHF)](#adaptive-policies-rlhf)
14. [Observability](#observability)
15. [Troubleshooting](#troubleshooting)

---

## Agent Blocks

### Basic Syntax

```n3
agent <name> {
  llm: <llm_name>
  goal: "<agent's objective>"
  [tools: [<tool_name>, ...]]
  [memory: { policy: "<policy>", ... }]
  [system_prompt: "<system instruction>"]
  [max_turns: <number>]
  [temperature: <float>]
}
```

### Required Fields

- **`llm`**: Name of the LLM to use (must reference an existing `llm` block)
- **`goal`**: High-level objective describing the agent's purpose

### Optional Fields

- **`tools`**: List of tool names the agent can invoke
- **`memory`**: Memory configuration (see [Memory Management](#memory-management))
- **`system_prompt`**: Custom system instruction for the LLM
- **`max_turns`**: Maximum reasoning turns before stopping (default: 10)
- **`temperature`**: LLM temperature override (0.0-1.0)

### Example

```n3
llm gpt4 {
  provider: "openai"
  model: "gpt-4"
  api_key: $OPENAI_API_KEY
}

tool search {
  type: "http"
  method: "GET"
  url: "https://api.search.com/query"
}

agent researcher {
  llm: gpt4
  goal: "Research topics and gather information"
  tools: [search]
  memory: {
    policy: "conversation_window"
    window_size: 10
  }
  system_prompt: "You are a thorough research assistant."
  max_turns: 15
  temperature: 0.7
}
```

---

## Graph Blocks

### Basic Syntax

```n3
graph <name> {
  start: <agent_name>
  edges: [
    <from_agent> -> <to_agent> [if <condition>]
    ...
  ]
  [termination: [<agent_name>, ...]]
  [termination_condition: <expression>]
  [max_hops: <number>]
  [timeout_ms: <milliseconds>]
}
```

### Required Fields

- **`start`**: Name of the initial agent (entry point)
- **`edges`**: List of directed edges defining agent transitions

### Optional Fields

- **`termination`**: List of terminal agents (execution stops when reached)
- **`termination_condition`**: Expression evaluated after each hop
- **`max_hops`**: Maximum agent transitions (default: 32)
- **`timeout_ms`**: Execution timeout in milliseconds

### Edge Syntax

Edges define the flow between agents:

```n3
# Unconditional edge (always transitions)
agent_a -> agent_b

# Conditional edge (evaluated against agent response)
agent_a -> agent_b if contains("approve")
agent_a -> agent_c if contains("reject")
```

Multiple edges from the same agent create branching logic. The first matching condition determines the route.

### Example

```n3
agent classifier {
  llm: gpt4
  goal: "Classify user intent"
}

agent handler_support {
  llm: gpt4
  goal: "Handle support requests"
}

agent handler_sales {
  llm: gpt4
  goal: "Handle sales inquiries"
}

agent summarizer {
  llm: gpt4
  goal: "Summarize the interaction"
}

graph customer_service {
  start: classifier
  edges: [
    classifier -> handler_support if contains("support")
    classifier -> handler_sales if contains("sales")
    handler_support -> summarizer
    handler_sales -> summarizer
  ]
  termination: [summarizer]
  max_hops: 10
  timeout_ms: 30000
}
```

---

## Invoking Graphs from Chains

Graphs can be invoked as steps in chains using the `graph` step kind:

```n3
chain process_request {
  step graph customer_service {
    input: payload.message
    context: {
      user_id: payload.user_id
      session_id: payload.session_id
    }
  }
}
```

### Graph Step Options

- **`input`**: Input passed to the starting agent
- **`context`**: Additional context variables shared across agents
- **`read_memory`**: Memory names to read from
- **`write_memory`**: Memory names to write results to

### Example Chain

```n3
chain ai_research_pipeline {
  step graph research_workflow {
    input: "Research latest AI trends in 2024"
    context: {
      depth: "comprehensive"
      sources_required: 5
    }
    write_memory: ["research_results"]
  }
  
  step template format_report {
    prompt: "Format the research into a report: {{ steps.research_workflow.output }}"
  }
}
```

---

## Memory Management

Agents support multiple memory policies to manage conversation history:

### Memory Policies

#### 1. None (No Memory)

```n3
agent stateless {
  llm: gpt4
  goal: "Process each request independently"
  memory: { policy: "none" }
}
```

No conversation history is maintained. Each interaction starts fresh.

#### 2. Full History

```n3
agent full_memory {
  llm: gpt4
  goal: "Remember everything"
  memory: {
    policy: "full_history"
    max_items: 100  # Optional: limit total messages
  }
}
```

Maintains complete conversation history. Optional `max_items` truncates from the beginning.

#### 3. Conversation Window

```n3
agent windowed {
  llm: gpt4
  goal: "Remember recent context"
  memory: {
    policy: "conversation_window"
    window_size: 10  # Keep last 10 messages
  }
}
```

Maintains a sliding window of recent messages. Oldest messages are discarded.

#### 4. Summary (Future)

```n3
agent summarized {
  llm: gpt4
  goal: "Maintain summarized context"
  memory: {
    policy: "summary"
    max_items: 50
  }
}
```

Periodically summarizes old messages to reduce token usage while retaining context.

---

## Routing and Conditions

### Conditional Expressions

Conditions are evaluated against the agent's response text:

```n3
# Text matching
agent_a -> agent_b if contains("keyword")
agent_a -> agent_c if starts_with("ERROR")
agent_a -> agent_d if ends_with("approved")

# Pattern matching (regex)
agent_a -> agent_b if matches("pattern.*")
```

### Routing Logic

1. **Sequential Evaluation**: Conditions are checked in order
2. **First Match Wins**: The first edge with a matching condition is taken
3. **Fallback**: If no conditions match, the first unconditional edge is used
4. **No Match**: If no edges match, the graph terminates at the current agent

### Example: Multi-Branch Routing

```n3
graph triage_system {
  start: triager
  edges: [
    # Priority routing
    triager -> urgent_handler if contains("urgent")
    triager -> important_handler if contains("important")
    
    # Category routing
    triager -> tech_support if contains("technical")
    triager -> billing_support if contains("billing")
    triager -> general_support if contains("general")
    
    # Fallback
    triager -> general_support
    
    # All handlers converge
    urgent_handler -> closer
    important_handler -> closer
    tech_support -> closer
    billing_support -> closer
    general_support -> closer
  ]
  termination: [closer]
  max_hops: 15
}
```

---

## Best Practices

### 1. Design Clear Agent Responsibilities

Each agent should have a focused, well-defined goal:

```n3
# ✅ Good: Specific responsibility
agent email_classifier {
  llm: gpt4
  goal: "Classify incoming emails by urgency and category"
}

# ❌ Bad: Too vague
agent email_handler {
  llm: gpt4
  goal: "Handle emails"
}
```

### 2. Use Descriptive Names

Names should clearly indicate agent purpose and graph workflow:

```n3
# ✅ Good names
agent invoice_validator
agent payment_processor
graph accounts_payable_workflow

# ❌ Unclear names
agent agent1
agent processor
graph workflow
```

### 3. Set Appropriate Limits

Configure `max_turns`, `max_hops`, and `timeout_ms` to prevent runaway execution:

```n3
agent researcher {
  llm: gpt4
  goal: "Research topics"
  max_turns: 20  # Prevent infinite reasoning loops
}

graph research_pipeline {
  start: researcher
  max_hops: 10   # Prevent infinite routing loops
  timeout_ms: 60000  # 60 second timeout
  # ...
}
```

### 4. Optimize Memory Configuration

Choose memory policy based on agent purpose:

- **Stateless tasks**: Use `policy: "none"` for independent operations
- **Short conversations**: Use `conversation_window` with small window
- **Long conversations**: Use `full_history` with `max_items` limit
- **Cost-sensitive**: Consider `summary` policy (when available)

### 5. Design Termination Conditions

Always define clear termination to prevent graph execution indefinitely:

```n3
graph workflow {
  start: processor
  edges: [
    processor -> validator
    validator -> processor if contains("retry")
    validator -> finalizer if contains("success")
  ]
  # Explicit termination
  termination: [finalizer]
  # Safety limit
  max_hops: 5
}
```

### 6. Handle Errors Gracefully

Design graphs with error handling paths:

```n3
graph robust_workflow {
  start: processor
  edges: [
    processor -> validator if contains("processed")
    processor -> error_handler if contains("error")
    validator -> finalizer if contains("valid")
    validator -> error_handler if contains("invalid")
    error_handler -> notifier
  ]
  termination: [finalizer, notifier]
}
```

### 7. Use Tools Strategically

Only grant agents the tools they need:

```n3
# ✅ Good: Agent only has needed tools
agent data_fetcher {
  llm: gpt4
  goal: "Fetch customer data"
  tools: [database_query]
}

# ❌ Bad: Unnecessary tool access
agent simple_responder {
  llm: gpt4
  goal: "Respond to greetings"
  tools: [database_query, api_call, file_write]  # Not needed!
}
```

### 8. Test Incrementally

Build and test agents individually before composing graphs:

1. Test agent with mock LLM responses
2. Test agent with real LLM
3. Test graph with 2 agents
4. Expand graph incrementally
5. Test full workflow end-to-end

---

## Language Examples

### Example 1: Simple Linear Workflow

```n3
llm gpt4 {
  provider: "openai"
  model: "gpt-4"
  api_key: $OPENAI_API_KEY
}

agent researcher {
  llm: gpt4
  goal: "Research the given topic thoroughly"
  system_prompt: "You are an expert researcher. Provide detailed, factual information."
}

agent writer {
  llm: gpt4
  goal: "Write a clear, engaging article based on research"
  system_prompt: "You are a skilled writer. Create compelling content."
}

graph content_creation {
  start: researcher
  edges: [
    researcher -> writer
  ]
  termination: [writer]
}

chain create_article {
  step graph content_creation {
    input: "Latest developments in quantum computing"
  }
}
```

### Example 2: Conditional Routing

```n3
agent sentiment_analyzer {
  llm: gpt4
  goal: "Analyze customer feedback sentiment"
}

agent positive_responder {
  llm: gpt4
  goal: "Craft appreciative responses to positive feedback"
}

agent negative_resolver {
  llm: gpt4
  goal: "Address and resolve negative feedback"
  tools: [create_ticket, notify_manager]
}

agent neutral_follower {
  llm: gpt4
  goal: "Follow up on neutral feedback"
}

graph feedback_handler {
  start: sentiment_analyzer
  edges: [
    sentiment_analyzer -> positive_responder if contains("positive")
    sentiment_analyzer -> negative_resolver if contains("negative")
    sentiment_analyzer -> neutral_follower if contains("neutral")
  ]
  termination: [positive_responder, negative_resolver, neutral_follower]
  max_hops: 5
}
```

### Example 3: Multi-Agent Collaboration

```n3
tool web_search {
  type: "http"
  method: "GET"
  url: "https://api.search.com/search"
}

tool code_analyzer {
  type: "python"
  module: "code_tools"
  method: "analyze"
}

agent planner {
  llm: gpt4
  goal: "Create a structured plan for solving the problem"
  memory: { policy: "full_history" }
}

agent researcher {
  llm: gpt4
  goal: "Research relevant information"
  tools: [web_search]
  memory: { policy: "conversation_window", window_size: 5 }
}

agent coder {
  llm: gpt4
  goal: "Write code to implement the solution"
  tools: [code_analyzer]
  memory: { policy: "conversation_window", window_size: 10 }
}

agent reviewer {
  llm: gpt4
  goal: "Review and validate the implementation"
  memory: { policy: "full_history" }
}

graph software_dev {
  start: planner
  edges: [
    planner -> researcher
    researcher -> coder
    coder -> reviewer
    reviewer -> coder if contains("needs revision")
    reviewer -> planner if contains("complete")
  ]
  termination: [planner]
  max_hops: 20
  timeout_ms: 120000
}

chain develop_feature {
  step graph software_dev {
    input: payload.feature_request
    context: {
      project: payload.project_name
      language: payload.language
    }
    write_memory: ["development_log"]
  }
}
```

### Example 4: Error Handling

```n3
agent processor {
  llm: gpt4
  goal: "Process the request"
  max_turns: 5
}

agent validator {
  llm: gpt4
  goal: "Validate processing results"
}

agent error_handler {
  llm: gpt4
  goal: "Handle errors and determine retry strategy"
}

agent success_notifier {
  llm: gpt4
  goal: "Send success notification"
  tools: [send_email]
}

agent failure_notifier {
  llm: gpt4
  goal: "Send failure notification"
  tools: [send_email, create_alert]
}

graph resilient_processing {
  start: processor
  edges: [
    processor -> validator if contains("processed")
    processor -> error_handler if contains("error")
    
    validator -> success_notifier if contains("valid")
    validator -> error_handler if contains("invalid")
    
    error_handler -> processor if contains("retry")
    error_handler -> failure_notifier if contains("fatal")
  ]
  termination: [success_notifier, failure_notifier]
  max_hops: 8
}
```

---

## Visual Graph Editor

The N3 Agent Graph Editor is a production-ready visual interface for building and executing agent workflows with real-time collaboration.

### Architecture

```
┌─────────────────┐
│  React Frontend │  ← Vite + TypeScript + React Flow
│  (port 3000)    │
└────────┬────────┘
         │
         ├─────────────────┬──────────────────┐
         │                 │                  │
         ▼                 ▼                  ▼
┌─────────────┐   ┌─────────────┐   ┌──────────────┐
│ FastAPI     │   │ Yjs WS      │   │ Jaeger UI    │
│ (port 8000) │   │ (port 1234) │   │ (port 16686) │
└──────┬──────┘   └─────────────┘   └──────────────┘
       │
       ▼
┌──────────────┐
│ PostgreSQL   │
│ (port 5432)  │
└──────────────┘
```

### Technology Stack

**Frontend:**
- React 18 + TypeScript 5
- Vite 5 (build tool)
- React Flow 11 (graph visualization)
- Yjs 13 + WebSocket (real-time)
- TanStack Query 5 (server state)
- Tailwind CSS 3 + Radix UI
- Monaco Editor (code editing)

**Backend:**
- FastAPI + Uvicorn
- SQLAlchemy 2.0 (async ORM)
- PostgreSQL 16
- OpenTelemetry SDK
- HuggingFace TRL (RLHF)

**Infrastructure:**
- Docker + docker-compose
- Jaeger (tracing)
- Node.js 18 (Yjs server)

---

## Editor Features

### ✅ Visual Graph Builder
- **React Flow Integration**: Drag-and-drop DAG builder with custom node types
- **Monaco Editor**: In-browser code editing with syntax highlighting
- **Node Types**: Prompts, agents, RAG datasets, memory stores, conditions, Python hooks

### ✅ Real-time Collaboration
- **Yjs CRDT**: Conflict-free collaborative editing
- **User Presence**: Live cursors and activity tracking
- **WebSocket Server**: Persistent connections for instant sync

### ✅ Share Links
- **Token-based Access**: Secure shareable URLs with role permissions
- **Viewer/Editor Roles**: Granular access control
- **Expiration Support**: Time-limited access tokens

### ✅ Backend API
- **FastAPI**: High-performance async Python API
- **PostgreSQL**: Persistent graph storage with SQLAlchemy ORM
- **OpenAPI Docs**: Auto-generated API documentation

### ✅ OpenTelemetry Instrumentation
- **Distributed Tracing**: Full execution observability
- **Jaeger Integration**: Visual trace analysis
- **LLM Span Tracking**: Token counts, costs, model parameters

### ✅ Tool Registry
- **Decorator-based**: Simple `@tool` registration
- **OpenAPI/LangChain Adapters**: Import external tools
- **Execution Tracing**: Per-tool performance metrics

### ✅ Adaptive Agent Policies (RLHF)
- **Feedback Collection**: Score agent responses
- **PPO Training**: Policy optimization with HuggingFace TRL
- **Version Management**: Track policy evolution

---

## Editor Quick Start

### Prerequisites
```bash
# Required
- Docker & docker-compose
- Node.js 18+
- Python 3.11+
```

### Development Setup

1. **Install frontend dependencies**:
```bash
cd src/web/graph-editor
npm install
```

2. **Install backend dependencies**:
```bash
cd n3_server
pip install -r requirements.txt
```

3. **Install Yjs server dependencies**:
```bash
cd yjs-server
npm install
```

4. **Start all services**:
```bash
docker-compose up -d
```

5. **Run database migrations**:
```bash
cd n3_server
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
```

6. **Access services**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Jaeger UI: http://localhost:16686
- Yjs Server: ws://localhost:1234

### API Endpoints

**Graph Management:**
```http
GET  /api/graphs/{projectId}          # Get graph data
PUT  /api/graphs/{projectId}          # Update graph
POST /api/graphs/{projectId}/execute  # Execute with tracing
```

**Share Links:**
```http
POST   /api/projects/{projectId}/shares  # Create share link
GET    /api/projects/{projectId}/shares  # List shares
DELETE /api/projects/{projectId}/shares/{shareId}
GET    /api/projects/open-by-token?token={token}
```

**Tool Registry:**
```http
GET  /api/tools                # List registered tools
POST /api/tools/execute        # Execute tool
POST /api/tools/register       # Register from OpenAPI/LangChain
```

**Adaptive Policies:**
```http
POST /api/feedback/{agentId}        # Submit feedback
POST /api/train_policy/{agentId}    # Train policy with RLHF
GET  /api/policies/{agentId}        # List policy versions
```

---

## Real-time Collaboration

### How Yjs Works
- **CRDT-based**: Conflict-free replicated data types
- **WebSocket**: Persistent connection to yjs-server
- **Awareness**: User presence tracking (cursors, selections)
- **Automatic Sync**: Bi-directional updates

### Integration Example
```typescript
import { useYjsGraph } from '@/hooks/useYjsGraph';

const { users, sync, isConnected } = useYjsGraph({
  projectId,
  nodes,
  edges,
  onRemoteUpdate: (remoteNodes, remoteEdges) => {
    // Apply remote changes
    setNodes(remoteNodes);
    setEdges(remoteEdges);
  },
});

// Sync local changes
useEffect(() => {
  sync(nodes, edges);
}, [nodes, edges]);
```

---

## Tool Registry

### Register Function Tools
```python
from n3_server.api.tools import tool

@tool(description="Calculate sum", tags=["math"])
def add(a: float, b: float) -> float:
    return a + b
```

### Execute Tools
```bash
curl -X POST http://localhost:8000/api/tools/execute \
  -H "Content-Type: application/json" \
  -d '{"name": "add", "args": {"a": 5, "b": 3}}'
```

---

## Adaptive Policies (RLHF)

### Feedback Collection
```bash
curl -X POST http://localhost:8000/api/feedback/agent-123 \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2+2?",
    "response": "4",
    "score": 1.0,
    "runId": "run-abc"
  }'
```

### Train Policy
```bash
curl -X POST http://localhost:8000/api/train_policy/agent-123 \
  -H "Content-Type: application/json" \
  -d '{
    "maxSteps": 1000,
    "learningRate": 0.00001
  }'
```

---

## Observability

### OpenTelemetry Instrumentation

**Automatic Tracing:**
- FastAPI requests (via `FastAPIInstrumentor`)
- OpenAI API calls (via `openinference-instrumentation-openai`)
- Custom tool executions

**Span Attributes:**
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("execute_graph") as span:
    span.set_attribute("project_id", project_id)
    span.set_attribute("entry", entry_point)
    # ... execution logic
```

**View Traces:**
1. Navigate to http://localhost:16686
2. Select service: `n3-server`
3. Browse traces with full context

### Agent Metrics

- **`agent.execution.start`**: Agent begins processing
- **`agent.turn.start`**: Each reasoning turn starts
- **`agent.turn.complete`**: Turn completes (includes tool call count)
- **`agent.tool.success`**: Tool executed successfully
- **`agent.tool.error`**: Tool execution failed
- **`agent.execution.complete`**: Agent finished successfully
- **`agent.execution.max_turns`**: Reached max turn limit
- **`agent.execution.error`**: Execution error occurred

### Graph Metrics

- **`graph.execution.start`**: Graph begins execution
- **`graph.hop.start`**: Each agent transition starts
- **`graph.hop.complete`**: Agent hop completes
- **`graph.hop.error`**: Hop encountered error
- **`graph.routing.decision`**: Routing decision made
- **`graph.routing.terminal`**: Reached terminal agent
- **`graph.execution.complete`**: Graph finished successfully
- **`graph.execution.hops`**: Total hops taken
- **`graph.execution.timeout`**: Execution timed out
- **`graph.execution.max_hops`**: Reached hop limit
- **`graph.execution.error`**: Execution error occurred

### Metric Tags

All metrics include relevant tags for filtering and aggregation:

- **Agent/Graph name**: Identify which agent or graph
- **Status**: success, error, timeout, etc.
- **Error types**: Specific error classifications
- **Turn/Hop numbers**: Track execution progress
- **Tool names**: Which tools were invoked

### Logging

Agents and graphs emit structured logs at different levels:

- **INFO**: Execution start/complete, major milestones
- **DEBUG**: Turn-by-turn details, routing decisions, tool calls
- **WARNING**: Errors, timeouts, limits reached
- **ERROR**: Exceptions with full stack traces

### Example: Monitoring Setup

```python
from namel3ss.observability import register_metric_listener

def metric_handler(name: str, values: dict, labels: dict):
    """Custom metric handler for monitoring"""
    if name == "graph.execution.complete":
        duration = values.get("value", 0)
        graph_name = labels.get("graph", "unknown")
        hops = labels.get("hops", "0")
        print(f"Graph {graph_name} completed in {duration}ms with {hops} hops")
    
    elif name == "agent.execution.error":
        agent = labels.get("agent", "unknown")
        error_type = labels.get("error_type", "unknown")
        print(f"Agent {agent} error: {error_type}")

register_metric_listener(metric_handler)
```

---

## Troubleshooting

### Common Issues

#### 1. Agent Not Terminating

**Problem**: Agent keeps reasoning without finishing.

**Solutions**:
- Reduce `max_turns` to a reasonable limit
- Improve goal and system_prompt clarity
- Check if tool calls are failing silently

#### 2. Graph Infinite Loop

**Problem**: Graph keeps routing between agents.

**Solutions**:
- Set appropriate `max_hops` limit
- Ensure termination agents are reachable
- Add termination_condition as safety
- Review edge conditions for logic errors

#### 3. Routing Not Working

**Problem**: Graph always takes the same path.

**Solutions**:
- Check condition syntax and quotes
- Verify agent response contains expected keywords
- Use DEBUG logging to inspect agent outputs
- Test conditions incrementally

#### 4. Tool Errors

**Problem**: Agent fails when calling tools.

**Solutions**:
- Verify tool is registered in tool_registry
- Check tool function signatures match expected args
- Add error handling in tool implementations
- Review agent logs for specific error messages

#### 5. High Latency

**Problem**: Graph execution takes too long.

**Solutions**:
- Reduce `max_turns` per agent
- Optimize memory policy (use windows instead of full history)
- Set appropriate `timeout_ms`
- Review LLM model choice (faster models for simple tasks)
- Monitor metrics to identify bottleneck agents

#### 6. Collaboration Sync Issues

**Problem**: Changes not syncing between users.

**Solutions**:
- Check WebSocket connection (ws://localhost:1234)
- Verify Yjs server is running (`docker ps`)
- Check browser console for errors
- Ensure users are on the same project ID

#### 7. Database Connection Errors

**Problem**: API fails to connect to PostgreSQL.

**Solutions**:
- Verify PostgreSQL is running (`docker ps`)
- Check DATABASE_URL in environment
- Run migrations (`alembic upgrade head`)
- Inspect logs (`docker logs postgres`)

---

## Advanced Topics

### State Management

Agents can share state through the context dictionary:

```n3
graph stateful_workflow {
  start: collector
  edges: [
    collector -> processor
    processor -> finalizer
  ]
  termination: [finalizer]
}

chain run_workflow {
  step graph stateful_workflow {
    input: "Start"
    context: {
      collected_data: []
      processing_status: "pending"
    }
  }
}
```

Agents can read and modify context during execution, passing information between hops.

### Dynamic Routing

Conditions can be complex expressions:

```n3
# Multiple conditions
agent_a -> agent_b if contains("urgent") and contains("customer")

# Nested logic
agent_a -> agent_b if contains("approve") or (contains("pending") and contains("priority"))

# Pattern matching
agent_a -> agent_b if matches("^TICKET-\\d+")
```

### Memory Persistence

Memory is maintained per agent instance during graph execution:

- Memory persists across hops within the same graph execution
- Memory is reset when the graph completes
- Use `write_memory` in chain steps to persist results beyond graph execution

---

## Testing

### Frontend E2E Tests (Playwright)
```bash
cd src/web/graph-editor
npm run test
```

### Backend Tests (pytest)
```bash
cd n3_server
pytest tests/
```

---

## Deployment

### Production Build
```bash
# Build all containers
docker-compose -f docker-compose.prod.yml build

# Deploy
docker-compose -f docker-compose.prod.yml up -d
```

### Environment Variables
```bash
# Backend (.env)
DATABASE_URL=postgresql+asyncpg://user:pass@host/db
OTLP_ENDPOINT=http://jaeger:4317
JWT_SECRET=your-secret-key
CORS_ORIGINS=["https://yourdomain.com"]

# Frontend (.env)
VITE_API_URL=https://api.yourdomain.com/api
VITE_YJS_URL=wss://collab.yourdomain.com
```

---

## Summary

Agents and graphs provide a powerful abstraction for building multi-agent AI systems in Namel3ss:

- **Agents**: Autonomous LLM-powered entities with tools and memory
- **Graphs**: Orchestrate multiple agents with conditional routing
- **Integration**: Invoke graphs from chains seamlessly
- **Visual Editor**: Production-ready UI with real-time collaboration
- **Observability**: Comprehensive metrics and logging
- **Best Practices**: Design principles for robust workflows

Start simple with linear workflows, then expand to complex multi-agent collaborations as your use cases evolve. Use the visual editor for rapid prototyping or code directly in N3 for version control and programmatic generation.
