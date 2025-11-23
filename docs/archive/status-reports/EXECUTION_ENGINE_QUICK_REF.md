# N3 Execution Engine - Quick Reference

## Execute Graph

```bash
curl -X POST http://localhost:8000/api/graphs/proj-123/execute \
  -H "Content-Type: application/json" \
  -d '{
    "entry": "start-node-id",
    "input": {"query": "What is AI?"},
    "options": {"timeout": 60000}
  }'
```

## Step Types

| Type | Purpose | Key Attributes |
|------|---------|----------------|
| `prompt` | Execute structured prompt | `model`, `tokens_prompt`, `tokens_completion`, `cost` |
| `agent` | Run multi-turn agent | `goal`, `max_turns`, `turns_executed` |
| `knowledge_query` | RAG retrieval | `top_k`, `reranker`, `documents_retrieved` |
| `tool` | Call registered tool | `tool_name`, `input_schema` |

## Span Types

```python
from n3_server.execution import SpanType

SpanType.CHAIN         # Overall chain execution
SpanType.AGENT_TURN    # Agent turn with LLM + tools
SpanType.PROMPT        # Structured prompt execution
SpanType.RAG_QUERY     # RAG retrieval + reranking
SpanType.TOOL_CALL     # Tool invocation
SpanType.LLM_CALL      # Direct LLM API call
```

## Programmatic Usage

```python
from namel3ss.ast import Chain, ChainStep
from n3_server.execution import GraphExecutor, ExecutionContext

# Define chain
chain = Chain(
    name="my_chain",
    steps=[
        ChainStep(kind="prompt", target="greet", output_key="greeting"),
        ChainStep(kind="agent", target="research", output_key="result"),
    ],
    input_key="input",
    output_key="result"
)

# Execute
context = ExecutionContext(
    project_id="proj-123",
    entry_node="start",
    input_data={"name": "Alice"}
)

executor = GraphExecutor()
result = await executor.execute_chain(chain, context.input_data, context)

# Access trace
for span in context.spans:
    print(f"{span.name}: {span.duration_ms}ms, tokens: {span.attributes.tokens_prompt}")
```

## Trace Analysis

```python
# Filter spans by type
prompt_spans = [s for s in context.spans if s.type == SpanType.PROMPT]
agent_spans = [s for s in context.spans if s.type == SpanType.AGENT_TURN]

# Calculate total cost
total_cost = sum(s.attributes.cost or 0 for s in context.spans)

# Calculate total tokens
total_tokens = sum(
    (s.attributes.tokens_prompt or 0) + (s.attributes.tokens_completion or 0)
    for s in context.spans
)

# Find slowest step
slowest = max(context.spans, key=lambda s: s.duration_ms)
print(f"Slowest: {slowest.name} ({slowest.duration_ms}ms)")
```

## Testing

```bash
# Run execution engine tests
pytest tests/backend/test_execution_engine.py -v

# Test specific scenario
pytest tests/backend/test_execution_engine.py::test_execute_agent_step -v

# With coverage
pytest tests/backend/test_execution_engine.py --cov=n3_server.execution
```

## Common Patterns

### Multi-Step Chain
```python
chain = Chain(
    name="multi_step",
    steps=[
        ChainStep(kind="knowledge_query", target="docs", output_key="docs"),
        ChainStep(kind="prompt", target="summarize", output_key="summary"),
        ChainStep(kind="agent", target="review", output_key="review"),
    ]
)
```

### Agent with Tools
```python
ChainStep(
    kind="agent",
    target="research_agent",
    options={
        "goal": "Find papers",
        "max_turns": 5,
        "tools": ["search", "summarize"]
    }
)
```

### RAG with Reranking
```python
ChainStep(
    kind="knowledge_query",
    target="docs_index",
    options={
        "top_k": 10,
        "reranker": "cohere",
        "filters": {"category": "tech"}
    }
)
```

## Error Handling

```python
try:
    result = await executor.execute_chain(chain, input_data, context)
except Exception as e:
    # Check error spans
    error_spans = [s for s in context.spans if s.status == "error"]
    for span in error_spans:
        print(f"Error in {span.name}: {span.attributes.error}")
```

## Performance Metrics

```python
# Execution summary
chain_span = [s for s in context.spans if s.type == SpanType.CHAIN][0]

metrics = {
    "total_duration_ms": chain_span.duration_ms,
    "total_cost": sum(s.attributes.cost or 0 for s in context.spans),
    "total_tokens": sum(
        (s.attributes.tokens_prompt or 0) + (s.attributes.tokens_completion or 0)
        for s in context.spans
    ),
    "steps_executed": len([s for s in context.spans if s.parent_span_id == chain_span.span_id]),
    "llm_calls": len([s for s in context.spans if s.attributes.model is not None]),
}
```

## Span Attributes Reference

### Prompt Spans
```python
{
    "model": "gpt-4",
    "temperature": 0.7,
    "tokens_prompt": 150,
    "tokens_completion": 80,
    "cost": 0.0024
}
```

### Agent Spans
```python
{
    "model": "gpt-4",
    "tokens_prompt": 300,
    "tokens_completion": 150,
    "cost": 0.0048
}
```

### RAG Spans
```python
{
    "model": "text-embedding-ada-002",
    "tokens_prompt": 50,
    "top_k": 5,
    "reranker": "cohere"
}
```

### Tool Spans
```python
{
    "tool_name": "calculator"
}
```

## OpenTelemetry Export

```python
from opentelemetry import trace
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor

# Configure exporter
exporter = ConsoleSpanExporter()
processor = BatchSpanProcessor(exporter)
trace.get_tracer_provider().add_span_processor(processor)

# Execute (spans automatically exported)
result = await executor.execute_chain(chain, input_data, context)
```

## Related Docs

- [Full Implementation Guide](./EXECUTION_ENGINE_IMPLEMENTATION.md)
- [Agent Graph Editor Guide](./AGENT_GRAPH_EDITOR_GUIDE.md)
- [AST Converter Reference](./AST_CONVERTER_QUICK_REF.md)
