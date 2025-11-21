# Agent Graph E2E Integration - Quick Reference

## What Was Built

### 1. Enhanced Graph Converter
**Location**: `n3_server/converter/`

```python
from n3_server.converter import EnhancedN3ASTConverter, GraphJSON

# Validate and convert graph to AST
converter = EnhancedN3ASTConverter()
graph_json = GraphJSON.model_validate(raw_graph_dict)  # Validates structure
chain_ast, context = await converter.convert_graph_to_chain(graph_json)
```

**Features**:
- Pydantic v2 validation for all graph structures
- Cycle detection prevents infinite loops
- Registry building for runtime components
- Structured error messages with node context

### 2. Runtime Registry
**Location**: `n3_server/execution/registry.py`

```python
from n3_server.execution.registry import RuntimeRegistry
from namel3ss.llm.registry import get_llm

# Build runtime registry from conversion context
llm_registry = {
    "gpt-4": get_llm("openai", model="gpt-4"),
    "gpt-3.5-turbo": get_llm("openai", model="gpt-3.5-turbo"),
}
tool_registry = {
    "search_web": search_web_tool,
    "send_email": send_email_tool,
}

registry = await RuntimeRegistry.from_conversion_context(
    context=context,
    llm_registry=llm_registry,
    tool_registry=tool_registry,
)
```

**Features**:
- Instantiates AgentRuntime with LLMs and tools
- Stores Prompt AST definitions
- Instantiates RagPipelineRuntime with embeddings
- Provides type-safe getters with clear error messages

### 3. Enhanced GraphExecutor
**Location**: `n3_server/execution/executor.py`

```python
from n3_server.execution.executor import GraphExecutor, ExecutionContext

# Execute with runtime registry
executor = GraphExecutor(registry=registry)
context = ExecutionContext(
    project_id="my-project",
    entry_node="start",
    input_data={"input": "Customer inquiry about billing"},
)

result = await executor.execute_chain(
    chain=chain_ast,
    input_data=input_data,
    context=context,
)

# Access results and trace
print(f"Result: {result}")
print(f"Trace spans: {len(context.spans)}")
for span in context.spans:
    print(f"  {span.name}: {span.duration_ms:.1f}ms, {span.attributes.tokens_prompt}+{span.attributes.tokens_completion} tokens")
```

**Features**:
- Real prompt execution via `execute_structured_prompt()`
- Real agent execution via `AgentRuntime.execute()`
- Real RAG queries via `RagPipelineRuntime.execute_query()`
- Token counting and cost estimation
- Full OpenTelemetry instrumentation

## Usage Examples

### Complete Execution Flow

```python
from n3_server.converter import EnhancedN3ASTConverter, GraphJSON
from n3_server.execution.registry import RuntimeRegistry
from n3_server.execution.executor import GraphExecutor, ExecutionContext
from namel3ss.llm.registry import get_llm

# 1. Load and validate graph
graph_dict = load_graph_from_database(project_id, graph_id)
graph_json = GraphJSON.model_validate(graph_dict)

# 2. Convert to AST
converter = EnhancedN3ASTConverter()
chain_ast, conversion_context = await converter.convert_graph_to_chain(graph_json)

# 3. Build runtime registry
llm_registry = {
    "gpt-4": get_llm("openai", model="gpt-4", api_key=openai_api_key),
}
tool_registry = {
    "search": search_tool,
    "calculator": calculator_tool,
}

runtime_registry = await RuntimeRegistry.from_conversion_context(
    context=conversion_context,
    llm_registry=llm_registry,
    tool_registry=tool_registry,
)

# 4. Execute
executor = GraphExecutor(registry=runtime_registry)
execution_context = ExecutionContext(
    project_id=project_id,
    entry_node="start",
    input_data={"input": user_input},
)

result = await executor.execute_chain(
    chain=chain_ast,
    input_data=execution_context.input_data,
    context=execution_context,
)

# 5. Extract metrics
total_cost = sum(
    span.attributes.cost or 0
    for span in execution_context.spans
)
total_tokens = sum(
    (span.attributes.tokens_prompt or 0) + (span.attributes.tokens_completion or 0)
    for span in execution_context.spans
)

print(f"Execution completed!")
print(f"  Result: {result}")
print(f"  Total cost: ${total_cost:.4f}")
print(f"  Total tokens: {total_tokens}")
print(f"  Trace spans: {len(execution_context.spans)}")
```

### Validation Only

```python
from n3_server.converter import EnhancedN3ASTConverter, GraphJSON, ConversionError

converter = EnhancedN3ASTConverter()

try:
    # Validate structure
    graph_json = GraphJSON.model_validate(raw_graph_dict)
    
    # Validate conversion
    validation_errors = converter.validate_graph(graph_json)
    
    if validation_errors:
        print("Graph has issues:")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("Graph is valid!")
        
except ConversionError as e:
    print(f"Conversion error at node '{e.node_id}': {e.details}")
except ValueError as e:
    print(f"Validation error: {e}")
```

## Key Classes Reference

### GraphJSON (models.py)
```python
class GraphJSON(BaseModel):
    projectId: str
    name: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    variables: Dict[str, Any] = {}
```

### GraphNode (models.py)
```python
class GraphNode(BaseModel):
    id: str
    type: NodeType  # START, END, AGENT, PROMPT, RAG_DATASET, etc.
    label: str
    position: Optional[Position] = None
    data: Union[
        StartEndNodeData,
        AgentNodeData,
        PromptNodeData,
        RagNodeData,
        ToolNodeData,
        ConditionNodeData,
    ]
```

### ConversionContext (enhanced_converter.py)
```python
@dataclass
class ConversionContext:
    visited_nodes: Set[str]
    agent_registry: Dict[str, Agent]
    prompt_registry: Dict[str, Prompt]
    rag_registry: Dict[str, RagDataset]
    tool_registry: Dict[str, ToolDefinition]
```

### RuntimeRegistry (registry.py)
```python
@dataclass
class RuntimeRegistry:
    agents: Dict[str, AgentRuntime]
    prompts: Dict[str, Prompt]
    rag_pipelines: Dict[str, RagPipelineRuntime]
    tools: Dict[str, Callable]
    llms: Dict[str, BaseLLM]
```

### ExecutionSpan (executor.py)
```python
@dataclass
class ExecutionSpan:
    span_id: str
    parent_span_id: Optional[str]
    name: str
    type: SpanType  # CHAIN, AGENT_TURN, PROMPT, RAG_QUERY, etc.
    start_time: datetime
    end_time: datetime
    duration_ms: float
    status: str  # "ok" or "error"
    attributes: SpanAttribute
    input_data: Optional[Any]
    output_data: Optional[Any]
```

## Error Handling

### Validation Errors
```python
from n3_server.converter.models import ConversionError

try:
    chain_ast, context = await converter.convert_graph_to_chain(graph_json)
except ConversionError as e:
    # Structured error with node context
    print(f"Error at node '{e.node_id}': {e.details}")
    # e.details is a dict with error context
```

### Registry Errors
```python
from n3_server.execution.registry import RegistryError

try:
    agent_runtime = registry.get_agent("my-agent")
except RegistryError as e:
    # Clear message about missing component
    print(f"Registry error: {e}")
    # e.g., "Agent 'my-agent' not found. Available agents: ['classifier', 'summarizer']"
```

### Execution Errors
All execution errors are captured in ExecutionSpan with status="error" and attributes.error set.

## OpenTelemetry Span Types

- `CHAIN`: Full chain execution (parent of all other spans)
- `AGENT_TURN`: Individual agent turn (may have multiple per agent)
- `AGENT_TOOL`: Tool call within agent turn
- `PROMPT`: Structured prompt execution
- `RAG_QUERY`: RAG pipeline query with retrieval
- `LLM_CALL`: Direct LLM API call
- `TOOL_CALL`: Standalone tool invocation

## Cost Estimation

Built-in pricing for common models (per 1K tokens):

| Model | Prompt | Completion |
|-------|--------|------------|
| gpt-4 | $0.03 | $0.06 |
| gpt-4-turbo | $0.01 | $0.03 |
| gpt-3.5-turbo | $0.0015 | $0.002 |
| claude-3-opus | $0.015 | $0.075 |
| claude-3-sonnet | $0.003 | $0.015 |
| claude-3-haiku | $0.00025 | $0.00125 |

Unknown models default to gpt-4 pricing.

## Next Steps to Complete Integration

1. **Backend API** (Task 4): Implement POST `/api/graphs/{id}/execute` endpoint
2. **Tests** (Task 7): Write unit and integration tests
3. **E2E Examples** (Task 6): Build Customer Support Triage and Research Pipeline
4. **Frontend** (Task 5): Connect React editor to backend API
5. **Documentation** (Task 8): Update guides with execution examples

See `AGENT_GRAPH_E2E_INTEGRATION_PLAN.md` for detailed implementation steps.
