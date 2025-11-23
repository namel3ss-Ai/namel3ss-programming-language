# Agent Graph Editor → N3 Runtime Integration - Implementation Plan

## Status: In Progress

This document outlines the production-grade integration connecting the visual Agent Graph Editor to the N3 runtime execution engine.

## Completed Work

### ✅ 1. Enhanced Graph-to-AST Converter (Task 2)

**Files Created:**
- `n3_server/converter/models.py` - Pydantic v2 validation models
- `n3_server/converter/enhanced_converter.py` - Production converter
- Updated `n3_server/converter/__init__.py` - Package exports

**Features Implemented:**
- Strict Pydantic v2 validation for all graph structures
- `GraphJSON`, `GraphNode`, `GraphEdge` validated models
- Node-specific data models: `AgentNodeData`, `PromptNodeData`, `RagNodeData`, `ToolNodeData`
- `EnhancedN3ASTConverter` with comprehensive error handling
- Cycle detection and graph integrity validation
- Registry building for agents, prompts, RAG pipelines, tools
- Detailed `ConversionError` exceptions with context

**Design Principles:**
- **Deterministic**: Same graph JSON always produces same AST
- **Idempotent**: Multiple conversions produce identical results
- **Strictly Typed**: All data validated with Pydantic v2
- **Fail-Safe**: Clear structured errors, never unhandled exceptions
- **Extensible**: New node types can be added without rewriting core logic

## Remaining Tasks

### 2. Runtime Integration (Task 3) - IN PROGRESS

**Goal:** Connect GraphExecutor to real N3 runtime components.

**Current State:**
- `n3_server/execution/executor.py` has placeholder execution
- Needs connection to:
  - `namel3ss.agents.runtime.AgentRuntime`
  - `namel3ss.prompts.executor.execute_structured_prompt`
  - `namel3ss.rag.pipeline.RagPipelineRuntime`
  - Tool registry from `n3_server.api.tools`

**Implementation Steps:**

#### Step 1: Create Runtime Registry System
```python
# n3_server/execution/registry.py

@dataclass
class RuntimeRegistry:
    """Central registry for all runtime components."""
    agents: Dict[str, AgentRuntime]
    prompts: Dict[str, Prompt]
    rag_pipelines: Dict[str, RagPipelineRuntime]
    tools: Dict[str, Callable]
    llms: Dict[str, BaseLLM]
    
    @classmethod
    async def from_conversion_context(
        cls,
        context: ConversionContext,
        llm_registry: Dict[str, BaseLLM]
    ) -> 'RuntimeRegistry':
        """Build runtime registry from conversion context."""
        # Instantiate AgentRuntime for each agent
        # Create RagPipelineRuntime for each RAG
        # Register tools
        ...
```

#### Step 2: Enhance GraphExecutor with Real Execution
```python
# n3_server/execution/executor.py enhancements

class GraphExecutor:
    def __init__(self, registry: RuntimeRegistry):
        self.registry = registry
        self.tracer = trace.get_tracer(__name__)
    
    async def _execute_prompt_step(self, step, working_data, context, parent_span_id):
        # Get prompt from registry
        prompt = self.registry.prompts.get(step.target)
        if not prompt:
            raise ExecutionError(f"Prompt not found: {step.target}")
        
        # Execute with real prompt executor
        result = await execute_structured_prompt(
            prompt=prompt,
            args=working_data,
            llm=self.registry.llms.get(prompt.model),
        )
        
        # Record span with real metrics
        ...
    
    async def _execute_agent_step(self, step, working_data, context, parent_span_id):
        # Get agent runtime from registry
        agent_runtime = self.registry.agents.get(step.target)
        if not agent_runtime:
            raise ExecutionError(f"Agent not found: {step.target}")
        
        # Execute with real agent runtime
        result = await agent_runtime.execute(
            user_input=working_data.get("input", ""),
            goal=step.options.get("goal"),
            max_turns=step.options.get("max_turns", 10),
        )
        
        # Record turn-level spans
        ...
    
    async def _execute_rag_step(self, step, working_data, context, parent_span_id):
        # Get RAG pipeline from registry
        rag_pipeline = self.registry.rag_pipelines.get(step.target)
        if not rag_pipeline:
            raise ExecutionError(f"RAG pipeline not found: {step.target}")
        
        # Execute query
        result = await rag_pipeline.query(
            query_text=working_data.get("query", ""),
            top_k=step.options.get("top_k", 5),
        )
        
        # Record retrieval metrics
        ...
```

#### Step 3: Token Counting and Cost Tracking
```python
# n3_server/execution/metrics.py

class ExecutionMetrics:
    """Track execution metrics and costs."""
    
    @staticmethod
    def count_tokens(text: str, model: str) -> int:
        """Count tokens for text using tiktoken."""
        ...
    
    @staticmethod
    def calculate_cost(tokens_prompt: int, tokens_completion: int, model: str) -> float:
        """Calculate API cost based on token usage."""
        PRICING = {
            "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
            "gpt-3.5-turbo": {"prompt": 0.0015 / 1000, "completion": 0.002 / 1000},
            ...
        }
        ...
```

### 3. API Endpoints (Task 4)

**Goal:** Production-grade execution API with auth, validation, tracing.

**Files to Create/Update:**
- `n3_server/api/execution.py` - New execution endpoints
- Update `n3_server/api/graphs.py` - Use enhanced converter

**Endpoints:**

#### POST /api/graphs/{project_id}/{graph_id}/execute
```python
class ExecutionRequest(BaseModel):
    entry: str  # Entry node ID or "start"
    input: Dict[str, Any]
    options: Optional[ExecutionOptions] = None

class ExecutionOptions(BaseModel):
    max_steps: int = 100
    timeout_ms: Optional[int] = None
    stream: bool = False  # Future: streaming support
    trace_level: Literal["full", "summary", "none"] = "full"

class ExecutionResponse(BaseModel):
    execution_id: str
    status: Literal["success", "error", "timeout"]
    result: Any
    trace: List[ExecutionSpan]
    metrics: ExecutionMetrics
    error: Optional[str] = None

@router.post("/{project_id}/{graph_id}/execute")
async def execute_graph(
    project_id: str,
    graph_id: str,
    request: ExecutionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),  # Auth
):
    """
    Execute a graph with full instrumentation.
    
    Auth: Requires project access (owner/editor/viewer)
    Tracing: Full OpenTelemetry spans
    Validation: Graph structure + runtime components
    """
    with tracer.start_as_current_span("execute_graph") as span:
        # 1. Load project and validate access
        # 2. Convert graph to AST with validation
        # 3. Build runtime registry
        # 4. Execute with instrumentation
        # 5. Return results + trace
        ...
```

#### GET /api/executions/{execution_id}/trace
```python
@router.get("/executions/{execution_id}/trace")
async def get_execution_trace(
    execution_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve detailed execution trace by ID.
    
    Useful for:
    - Post-execution analysis
    - Debugging failed runs
    - Performance optimization
    """
    ...
```

### 4. Frontend Integration (Task 5)

**Goal:** Connect React editor to real backend APIs.

**Files to Update:**
- `src/web/graph-editor/src/lib/api.ts` - Add execution methods
- `src/web/graph-editor/src/hooks/useExecution.ts` - Execution state hook
- `src/web/graph-editor/src/components/ExecutionPanel.tsx` - Results display

**API Client:**
```typescript
// api.ts additions

export const executionApi = {
  async executeGraph(
    projectId: string,
    graphId: string,
    request: ExecutionRequest
  ): Promise<ExecutionResponse> {
    const response = await apiClient.post(
      `/graphs/${projectId}/${graphId}/execute`,
      request
    );
    return response.data;
  },
  
  async getTrace(executionId: string): Promise<ExecutionTrace> {
    const response = await apiClient.get(`/executions/${executionId}/trace`);
    return response.data;
  },
};
```

**React Hook:**
```typescript
// useExecution.ts

export function useExecution(projectId: string, graphId: string) {
  const [status, setStatus] = useState<ExecutionStatus>("idle");
  const [result, setResult] = useState<any>(null);
  const [trace, setTrace] = useState<ExecutionSpan[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  const execute = useCallback(async (input: Record<string, any>) => {
    setStatus("running");
    try {
      const response = await executionApi.executeGraph(projectId, graphId, {
        entry: "start",
        input,
      });
      setResult(response.result);
      setTrace(response.trace);
      setStatus("success");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }, [projectId, graphId]);
  
  return { status, result, trace, error, execute };
}
```

### 5. E2E Examples (Task 6)

**Goal:** Two complete, working examples.

#### Example 1: Customer Support Triage

**Graph Structure:**
```
[START] 
  → [PROMPT: Extract Info] 
  → [AGENT: Classifier] 
  → [CONDITION: Route by Category]
    ├─ billing → [RAG: Billing Docs] → [AGENT: Billing Specialist]
    ├─ technical → [RAG: Tech Docs] → [AGENT: Tech Support]
    └─ account → [AGENT: Account Manager]
  → [PROMPT: Summary]
  → [END]
```

**Implementation:**
```python
# examples/customer_support_triage.py

async def create_customer_support_graph() -> str:
    """Create and persist customer support triage graph."""
    graph = {
        "projectId": "customer-support",
        "name": "Customer Support Triage",
        "nodes": [
            {
                "id": "start-1",
                "type": "start",
                "label": "START",
                "data": {},
            },
            {
                "id": "prompt-extract",
                "type": "prompt",
                "label": "Extract Ticket Info",
                "data": {
                    "name": "extract_ticket_info",
                    "text": "Extract key information from ticket: {{ticket_text}}",
                    "model": "gpt-4",
                    "arguments": ["ticket_text"],
                    "outputSchema": {
                        "customer_name": "string",
                        "issue_summary": "string",
                        "category_hint": "string"
                    }
                }
            },
            # ... more nodes
        ],
        "edges": [
            {"id": "e1", "source": "start-1", "target": "prompt-extract"},
            # ... more edges
        ],
    }
    
    # Save to database
    async with AsyncSession(engine) as session:
        project = Project(
            id="customer-support",
            name="Customer Support Triage",
            graph_data=graph
        )
        session.add(project)
        await session.commit()
    
    return "customer-support"
```

#### Example 2: Research Pipeline

**Graph Structure:**
```
[START]
  → [PROMPT: Refine Query]
  → [AGENT: Web Search]
  → [RAG: Knowledge Base]
  → [AGENT: Synthesizer]
  → [CONDITION: Quality Check]
    ├─ pass → [AGENT: Reviewer] → [END]
    └─ fail → [AGENT: Web Search] (loop back)
```

### 6. Testing (Task 7)

**Unit Tests:**
- `tests/converter/test_enhanced_converter.py` - Conversion validation
- `tests/execution/test_graph_executor.py` - Execution logic
- `tests/api/test_execution_endpoints.py` - API contracts

**Integration Tests:**
- `tests/integration/test_e2e_execution.py` - Full lifecycle tests
- Mock external APIs (LLMs, vector DBs)
- Test both example workflows

### 7. Documentation (Task 8)

**Files to Update:**
- `AGENT_GRAPH_GUIDE.md` - Add "Visual Graph → Runtime Execution" section
- `EXECUTION_ENGINE_IMPLEMENTATION.md` - Finalize architecture docs
- `docs/AGENT_GRAPH_E2E_EXAMPLES.md` - New file with complete examples

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Visual Graph Editor (React)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Canvas  │  │ Node Lib │  │ Execute  │  │  Trace   │   │
│  │          │  │          │  │  Button  │  │  Viewer  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │ REST API
┌───────────────────────▼─────────────────────────────────────┐
│                   Backend (FastAPI)                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  POST /api/graphs/{id}/execute                       │   │
│  │  ┌──────────────┐  ┌──────────────┐                 │   │
│  │  │   Validate   │→ │   Convert    │                 │   │
│  │  │   GraphJSON  │  │   to AST     │                 │   │
│  │  └──────────────┘  └──────────────┘                 │   │
│  └────────────────────────┬────────────────────────────┘   │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Enhanced N3ASTConverter + Runtime Registry          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │  Agents  │  │ Prompts  │  │   RAG    │          │   │
│  │  └──────────┘  └──────────┘  └──────────┘          │   │
│  └────────────────────────┬────────────────────────────┘   │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  GraphExecutor with N3 Runtime Integration           │   │
│  │  ┌──────────────┐  ┌──────────────┐                 │   │
│  │  │ AgentRuntime │  │PromptExecutor│                 │   │
│  │  └──────────────┘  └──────────────┘                 │   │
│  │  ┌──────────────┐  ┌──────────────┐                 │   │
│  │  │ RAGPipeline  │  │ToolRegistry  │                 │   │
│  │  └──────────────┘  └──────────────┘                 │   │
│  └────────────────────────┬────────────────────────────┘   │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  OpenTelemetry Tracing (Jaeger)                      │   │
│  │  • Chain spans  • Agent turns  • Prompt calls        │   │
│  │  • RAG queries  • Tool calls   • LLM requests        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ▼
                    ┌──────────────┐
                    │  PostgreSQL  │
                    │  (Projects,  │
                    │   Traces)    │
                    └──────────────┘
```

## Next Steps

1. ✅ Complete enhanced converter with validation
2. 🔄 Integrate GraphExecutor with real N3 runtime
3. 📝 Implement execution API endpoints
4. 🎨 Connect frontend execution UI
5. 📚 Build E2E examples
6. ✅ Write tests
7. 📖 Update documentation

## Success Criteria

- [ ] Graph JSON validates with Pydantic v2
- [ ] Conversion to AST is deterministic and idempotent
- [ ] Execution uses real AgentRuntime, PromptExecutor, RAGPipeline
- [ ] All executions have complete OpenTelemetry traces
- [ ] API has auth, validation, error handling
- [ ] Frontend triggers real executions and displays results
- [ ] Two E2E examples work end-to-end
- [ ] Unit + integration tests pass
- [ ] Documentation is complete and accurate
