# N3 Execution Engine Implementation

## Overview

The N3 Execution Engine provides runtime execution for visual graphs with comprehensive OpenTelemetry instrumentation. It bridges the visual graph editor with N3's native runtime components (AgentRuntime, PromptProgram, RagPipelineRuntime) to enable traced execution of agents, prompts, RAG pipelines, and multi-step chains.

## Architecture

### Components

1. **GraphExecutor** - Main execution orchestrator
   - Converts graph JSON to N3 AST via N3ASTConverter
   - Executes chains with step-by-step instrumentation
   - Collects OpenTelemetry spans with timing and metrics

2. **ExecutionContext** - Execution state container
   - Tracks project ID, entry node, input data
   - Maintains execution variables across steps
   - Accumulates trace spans for analysis

3. **SpanType** - Trace span types
   - `CHAIN` - Overall chain execution
   - `AGENT_TURN` - Agent turn with LLM + tools
   - `PROMPT` - Structured prompt execution
   - `RAG_QUERY` - RAG retrieval + reranking
   - `TOOL_CALL` - Tool invocation
   - `LLM_CALL` - Direct LLM API call

4. **ExecutionSpan** - Traced execution span
   - Timing: start_time, end_time, duration_ms
   - Hierarchy: span_id, parent_span_id
   - Metrics: tokens (prompt/completion), cost, model
   - Data: input_data, output_data for debugging

## Execution Flow

```
Graph JSON → N3ASTConverter → Chain AST → GraphExecutor → Result + Trace

For each ChainStep:
  1. Create span with unique ID
  2. Start OpenTelemetry span
  3. Execute step (prompt/agent/RAG/tool)
  4. Collect metrics (tokens, cost, latency)
  5. Record span with timing and data
  6. Update working data with output
```

## Supported Step Types

### 1. Prompt Step
```python
ChainStep(
    kind="prompt",
    target="greeting_prompt",
    options={"args": {"name": "Alice"}},
    output_key="greeting"
)
```

**Execution:**
- Resolves prompt definition from registry
- Renders template with arguments
- Calls LLM provider with structured output
- Validates output schema
- Records tokens, cost, latency

**Span Attributes:**
- `model`: LLM model name (e.g., "gpt-4")
- `tokens_prompt`: Input token count
- `tokens_completion`: Output token count
- `cost`: Estimated API cost
- `temperature`: Sampling temperature

### 2. Agent Step
```python
ChainStep(
    kind="agent",
    target="research_agent",
    options={
        "goal": "Research AI trends",
        "max_turns": 5
    },
    output_key="research"
)
```

**Execution:**
- Instantiates AgentRuntime with goal
- Executes turns until goal met or max_turns reached
- Each turn: LLM reasoning → tool calls → observation
- Records per-turn spans with hierarchy

**Span Hierarchy:**
```
agent.research_agent (parent)
├── agent.research_agent.turn_1
│   ├── llm.call (reasoning)
│   └── tool.search
├── agent.research_agent.turn_2
│   ├── llm.call (reasoning)
│   └── tool.summarize
└── agent.research_agent.turn_3
    └── llm.call (final answer)
```

**Span Attributes (per turn):**
- `model`: LLM model
- `tokens_prompt`: Turn input tokens
- `tokens_completion`: Turn output tokens
- `cost`: Turn cost

### 3. RAG Query Step
```python
ChainStep(
    kind="knowledge_query",
    target="docs_index",
    options={
        "top_k": 5,
        "reranker": "cohere"
    },
    output_key="documents"
)
```

**Execution:**
- Extracts query from working data
- Embeds query with encoder
- Retrieves top_k documents from vector backend
- Optionally reranks with reranker model
- Returns scored documents

**Span Attributes:**
- `top_k`: Number of documents retrieved
- `reranker`: Reranker model name or None
- `tokens_prompt`: Query embedding tokens
- `model`: Embedding model name

### 4. Tool Call Step
```python
ChainStep(
    kind="tool",
    target="calculator",
    options={"args": {"op": "add", "a": 5, "b": 3}},
    output_key="result"
)
```

**Execution:**
- Resolves tool from ToolRegistry
- Validates input schema
- Executes tool function
- Validates output schema
- Returns result

**Span Attributes:**
- `tool_name`: Tool identifier
- `error`: Error message if failed

## API Integration

### Execute Graph Endpoint

```http
POST /api/graphs/{project_id}/execute
Content-Type: application/json

{
  "entry": "start-node-id",
  "input": {
    "query": "Explain transformers",
    "context": "academic"
  },
  "options": {
    "timeout": 60000,
    "streaming": true
  }
}
```

**Response:**
```json
{
  "result": {
    "status": "completed",
    "output": "Transformers are neural network architectures..."
  },
  "trace": [
    {
      "spanId": "span-abc123",
      "parentSpanId": null,
      "name": "chain.main",
      "type": "chain",
      "startTime": "2024-01-15T10:00:00.000Z",
      "endTime": "2024-01-15T10:00:05.234Z",
      "durationMs": 5234,
      "status": "ok",
      "attributes": {
        "model": null,
        "temperature": null,
        "tokensPrompt": null,
        "tokensCompletion": null,
        "cost": null
      },
      "input": {"query": "Explain transformers"},
      "output": {"status": "completed", "output": "..."}
    },
    {
      "spanId": "span-def456",
      "parentSpanId": "span-abc123",
      "name": "rag.docs_index",
      "type": "rag.query",
      "startTime": "2024-01-15T10:00:00.100Z",
      "endTime": "2024-01-15T10:00:01.200Z",
      "durationMs": 1100,
      "status": "ok",
      "attributes": {
        "model": "text-embedding-ada-002",
        "tokensPrompt": 50,
        "top_k": 5,
        "reranker": "cohere"
      },
      "input": {"query": "Explain transformers"},
      "output": {"documents": [...], "count": 5}
    },
    {
      "spanId": "span-ghi789",
      "parentSpanId": "span-abc123",
      "name": "prompt.summarize",
      "type": "prompt",
      "startTime": "2024-01-15T10:00:01.300Z",
      "endTime": "2024-01-15T10:00:04.500Z",
      "durationMs": 3200,
      "status": "ok",
      "attributes": {
        "model": "gpt-4",
        "temperature": 0.7,
        "tokensPrompt": 1500,
        "tokensCompletion": 800,
        "cost": 0.045
      },
      "input": {"documents": [...]},
      "output": {"summary": "Transformers are..."}
    }
  ]
}
```

## OpenTelemetry Integration

### Span Structure

Every execution operation creates an OpenTelemetry span with:

```python
with tracer.start_as_current_span(f"{type}.{name}") as span:
    span.set_attribute("operation.type", type)
    span.set_attribute("operation.target", name)
    
    # Execute operation
    result = await execute_operation()
    
    span.set_status(Status(StatusCode.OK))
    return result
```

### Span Attributes

Standard attributes across all spans:
- `project_id`: Project identifier
- `entry_node`: Starting node ID
- `chain.name`: Chain name
- `chain.steps`: Number of steps

Step-specific attributes:
- **Prompts**: `model`, `temperature`, `tokens.prompt`, `tokens.completion`, `cost`
- **Agents**: `agent.goal`, `agent.max_turns`, `agent.turns_executed`
- **RAG**: `rag.top_k`, `rag.reranker`, `rag.documents_retrieved`
- **Tools**: `tool.name`, `tool.input_schema`, `tool.output_schema`

### Error Handling

Errors are captured in spans:
```python
try:
    result = await execute_step()
    span.set_status(Status(StatusCode.OK))
except Exception as e:
    span.set_status(Status(StatusCode.ERROR, str(e)))
    span.record_exception(e)
    # Record error span
    context.add_span(ExecutionSpan(
        status="error",
        attributes=SpanAttribute(error=str(e))
    ))
    raise
```

## Usage Examples

### Execute Simple Prompt Chain

```python
from namel3ss.ast import Chain, ChainStep
from n3_server.execution import GraphExecutor, ExecutionContext

# Define chain
chain = Chain(
    name="greeting_chain",
    steps=[
        ChainStep(
            kind="prompt",
            target="greeting_prompt",
            options={"args": {"name": "Alice"}},
            output_key="greeting"
        )
    ],
    input_key="input",
    output_key="greeting"
)

# Create context
context = ExecutionContext(
    project_id="proj-123",
    entry_node="start",
    input_data={"name": "Alice"}
)

# Execute
executor = GraphExecutor()
result = await executor.execute_chain(
    chain,
    context.input_data,
    context
)

# Access trace
for span in context.spans:
    print(f"{span.name}: {span.duration_ms}ms")
    print(f"  Tokens: {span.attributes.tokens_prompt} → {span.attributes.tokens_completion}")
    print(f"  Cost: ${span.attributes.cost}")
```

### Execute Agent with Tools

```python
chain = Chain(
    name="research_chain",
    steps=[
        ChainStep(
            kind="agent",
            target="research_agent",
            options={
                "goal": "Find latest AI research papers",
                "max_turns": 5,
                "tools": ["search", "summarize"]
            },
            output_key="research"
        )
    ],
    input_key="input",
    output_key="research"
)

context = ExecutionContext(
    project_id="proj-456",
    entry_node="start",
    input_data={"topic": "transformers"}
)

executor = GraphExecutor()
result = await executor.execute_chain(chain, context.input_data, context)

# Analyze agent turns
agent_spans = [s for s in context.spans if s.type == SpanType.AGENT_TURN]
print(f"Agent executed {len(agent_spans)} turns")

total_cost = sum(s.attributes.cost or 0 for s in agent_spans)
print(f"Total cost: ${total_cost}")
```

### Execute RAG Pipeline

```python
chain = Chain(
    name="rag_chain",
    steps=[
        ChainStep(
            kind="knowledge_query",
            target="docs_index",
            options={
                "top_k": 10,
                "reranker": "cohere",
                "filters": {"category": "technical"}
            },
            output_key="documents"
        ),
        ChainStep(
            kind="prompt",
            target="answer_prompt",
            options={"args": {"documents": [], "query": ""}},
            output_key="answer"
        )
    ],
    input_key="input",
    output_key="answer"
)

context = ExecutionContext(
    project_id="proj-789",
    entry_node="start",
    input_data={"query": "What is RAG?"}
)

executor = GraphExecutor()
result = await executor.execute_chain(chain, context.input_data, context)

# Check RAG metrics
rag_span = [s for s in context.spans if s.type == SpanType.RAG_QUERY][0]
print(f"Retrieved {rag_span.attributes.top_k} documents")
print(f"Reranker: {rag_span.attributes.reranker}")
```

## Testing

### Run Tests

```bash
# Run all execution engine tests
pytest tests/backend/test_execution_engine.py -v

# Run specific test
pytest tests/backend/test_execution_engine.py::test_execute_simple_chain -v

# Run with coverage
pytest tests/backend/test_execution_engine.py --cov=n3_server.execution --cov-report=html
```

### Test Coverage

The test suite covers:
- ✅ Simple chain execution with prompts
- ✅ Agent execution with multi-turn tracing
- ✅ RAG query with retrieval metrics
- ✅ Tool call execution
- ✅ Multi-step chains with complex flows
- ✅ Context variable persistence
- ✅ Span attribute completeness
- ✅ Error handling and span recording

## Performance Considerations

### Span Collection Overhead

- Each span adds ~100-200 bytes to trace
- Typical chain: 10-50 spans
- Memory overhead: ~5-10 KB per execution
- Consider span sampling for high-throughput scenarios

### Async Execution

All execution methods are async:
```python
# Parallel step execution (future enhancement)
async with asyncio.TaskGroup() as tg:
    task1 = tg.create_task(execute_step(step1))
    task2 = tg.create_task(execute_step(step2))
results = await asyncio.gather(task1, task2)
```

### Token Counting

Token counts are collected from:
1. LLM provider response headers
2. Embedding model responses
3. Estimated from text length (fallback)

Cost calculation uses provider-specific pricing:
```python
cost = (tokens_prompt * price_per_1k_input / 1000) + \
       (tokens_completion * price_per_1k_output / 1000)
```

## Future Enhancements

### Streaming Execution

```python
async def execute_chain_stream(
    chain: Chain,
    input_data: Dict[str, Any],
    context: ExecutionContext
) -> AsyncIterator[ExecutionEvent]:
    """Stream execution events as they happen."""
    for step in chain.steps:
        yield ExecutionEvent(type="step_start", step=step.target)
        result = await execute_step(step)
        yield ExecutionEvent(type="step_complete", result=result)
```

### Parallel Execution

Execute independent steps in parallel:
```python
# Detect independent steps
independent_groups = analyze_dependencies(chain.steps)

# Execute in parallel
for group in independent_groups:
    results = await asyncio.gather(*[
        execute_step(step) for step in group
    ])
```

### Caching

Cache expensive operations:
```python
@lru_cache(maxsize=1000)
async def execute_rag_query(query: str, index: str) -> RagResult:
    # Cache RAG results by query + index
    pass
```

## Related Documentation

- [N3 AST Converter Implementation](./AST_CONVERTER_IMPLEMENTATION.md)
- [Agent Graph Editor Guide](./AGENT_GRAPH_EDITOR_GUIDE.md)
- [OpenTelemetry Integration](./AGENT_GRAPH_EDITOR_GUIDE.md#observability)
- [Tool Registry Documentation](./AGENT_GRAPH_EDITOR_GUIDE.md#tool-registry-api)
