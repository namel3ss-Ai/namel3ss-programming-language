# Frontend Integration - Agent Graph Execution

## Overview

Complete React + TypeScript frontend integration for executing agent graphs with real-time telemetry visualization.

## Components

### 1. `useExecution` Hook

**File**: `src/hooks/useExecution.ts`

React hook for managing graph execution state and operations.

**Features**:
- Execute graphs with input data
- Track execution progress and status
- Parse and aggregate telemetry data
- Calculate costs and token usage
- Filter spans by type
- Error handling

**API**:
```typescript
interface UseExecutionReturn {
  state: ExecutionState;
  execute: (projectId: string, request: ExecutionRequest) => Promise<void>;
  reset: () => void;
  getSpansByType: (type: string) => ExecutionSpan[];
  getTotalCost: () => number;
  getTotalTokens: () => { prompt: number; completion: number };
  getDuration: () => number;
}
```

**Usage**:
```typescript
const { state, execute, getTotalCost, getTotalTokens } = useExecution();

// Execute graph
await execute(projectId, {
  entry: 'start-1',
  input: { query: 'test query' }
});

// Get metrics
const cost = getTotalCost();
const tokens = getTotalTokens();
const llmSpans = getSpansByType('llm.call');
```

### 2. `ExecutionPanel` Component

**File**: `src/components/ExecutionPanel.tsx`

Full-featured UI for graph execution and trace visualization.

**Features**:
- ✅ Execution controls (input data, entry node)
- ✅ Real-time execution status
- ✅ Summary metrics (duration, tokens, cost)
- ✅ Final result display
- ✅ Hierarchical trace visualization
- ✅ Expandable span details
- ✅ Span attributes, input/output display
- ✅ Icon-based span type identification
- ✅ Status color coding
- ✅ Error handling and display

**Components**:
```tsx
<ExecutionPanel projectId={projectId} />
```

**UI Structure**:
```
┌─────────────────────────────────────┐
│ Execution                           │
├─────────────────────────────────────┤
│ Entry Node: [start-1            ]   │
│ Input Data:                         │
│ ┌─────────────────────────────────┐ │
│ │ { "key": "value" }              │ │
│ └─────────────────────────────────┘ │
│ [▶ Execute] [↻ Reset]               │
├─────────────────────────────────────┤
│ ┌──────┬──────┬──────┬──────┐      │
│ │Status│ Time │Tokens│ Cost │      │
│ └──────┴──────┴──────┴──────┘      │
│                                     │
│ Final Result:                       │
│ { ... }                             │
│                                     │
│ Execution Trace:                    │
│ ▼ 🤖 llm.call        ok   1.2s      │
│   │ Attributes: {...}               │
│   │ Input: {...}                    │
│   │ Output: {...}                   │
│ ▶ 🔧 tool.call       ok   0.5s      │
│ ▶ 📚 rag.retrieve    ok   0.8s      │
└─────────────────────────────────────┘
```

### 3. API Integration

**File**: `src/lib/api.ts`

Updated API client with execution endpoints.

**New Methods**:
```typescript
// Execute graph with instrumentation
graphApi.executeGraph(projectId, {
  entry: 'start-1',
  input: { ... }
}): Promise<ExecutionResult>

// Validate graph structure
graphApi.validateGraph(projectId): Promise<ValidationResult>
```

**Backend Endpoint**: `POST /api/execution/graphs/{project_id}/execute`

**Request**:
```json
{
  "entry": "start-1",
  "input": {
    "ticket_text": "I can't log in",
    "customer_tier": "enterprise"
  },
  "options": {}
}
```

**Response**:
```json
{
  "result": {
    "summary": "Ticket escalated",
    "actions": [...]
  },
  "trace": [
    {
      "spanId": "span-1",
      "parentSpanId": null,
      "name": "classify_ticket",
      "type": "llm.call",
      "startTime": "2025-11-21T10:30:00Z",
      "endTime": "2025-11-21T10:30:01.234Z",
      "durationMs": 1234,
      "status": "ok",
      "attributes": {
        "model": "gpt-4",
        "temperature": 0.7,
        "tokensPrompt": 150,
        "tokensCompletion": 75,
        "cost": 0.0045
      },
      "input": {...},
      "output": {...}
    }
  ]
}
```

## Types

### ExecutionSpan

```typescript
interface ExecutionSpan {
  spanId: string;
  parentSpanId: string | null;
  name: string;
  type: 'llm.call' | 'tool.call' | 'rag.retrieve' | 'agent.step' | 'chain.step';
  startTime: string;
  endTime: string;
  durationMs: number;
  status: 'ok' | 'error';
  attributes: SpanAttribute;
  input?: any;
  output?: any;
}
```

### SpanAttribute

```typescript
interface SpanAttribute {
  model?: string;
  temperature?: number;
  tokensPrompt?: number;
  tokensCompletion?: number;
  cost?: number;
  [key: string]: any;
}
```

## Features

### Execution Controls

- **Entry Node**: Specify which node to start execution from
- **Input Data**: JSON input passed to the graph
- **Execute Button**: Triggers graph execution
- **Reset Button**: Clears results and resets state

### Real-time Status

- Loading indicator during execution
- Error messages with details
- Success confirmation

### Summary Metrics

Displayed as cards with icons:
- ✅ **Status**: Success/Error indicator
- ⏱️ **Duration**: Total execution time
- ⚡ **Tokens**: Input + Output token counts
- 💰 **Cost**: Total LLM cost in USD

### Trace Visualization

**Hierarchical Tree**:
- Parent-child relationships preserved
- Expandable/collapsible spans
- Visual indentation for levels

**Span Details**:
- Icon based on type (🤖 LLM, 🔧 Tool, 📚 RAG, etc.)
- Name and status
- Duration and cost
- Expandable attributes, input, output

**Interactive Features**:
- Click to expand/collapse
- Hover highlighting
- Syntax-highlighted JSON

### Error Handling

- Network errors caught and displayed
- JSON parsing errors handled
- Execution failures shown with details
- Validation errors surfaced

## Integration with Backend

### Execution Flow

```
Frontend                    Backend
────────                    ───────
┌─────────────┐            ┌──────────────────┐
│ ExecutionPanel │         │ /api/execution   │
│               │          │  /graphs/{id}    │
│ useExecution  │──POST───>│  /execute        │
│               │          │                  │
│               │          │ EnhancedConverter│
│               │          │ RuntimeRegistry  │
│               │          │ GraphExecutor    │
│               │<─JSON────│                  │
│               │          │ OpenTelemetry    │
│ Trace Display │          │ Spans           │
└─────────────┘            └──────────────────┘
```

### Pipeline Stages

1. **User Input**: Entry node + JSON input data
2. **API Request**: POST to backend execution endpoint
3. **Backend Processing**:
   - Validate project access
   - Load graph from database
   - Convert to N3 AST (EnhancedN3ASTConverter)
   - Build runtime registry (LLMs, agents, tools, RAG)
   - Execute with instrumentation (GraphExecutor)
   - Collect OpenTelemetry spans
4. **Response**: Result + trace returned
5. **Frontend Display**: Parse and visualize results

## Usage Example

### In GraphEditorPage

```tsx
import ExecutionPanel from './components/ExecutionPanel';

export default function GraphEditorPage() {
  const { projectId } = useParams();
  
  return (
    <div className="flex h-screen">
      <div className="flex-1">
        <GraphCanvas projectId={projectId} />
      </div>
      <div className="w-96 border-l">
        <ExecutionPanel projectId={projectId} />
      </div>
    </div>
  );
}
```

### Programmatic Execution

```typescript
import { useExecution } from './hooks/useExecution';

function MyComponent() {
  const { execute, state } = useExecution();
  
  const handleRun = async () => {
    try {
      await execute('my-project-id', {
        entry: 'start-1',
        input: {
          query: 'What is chain-of-thought?',
          context: 'research'
        }
      });
      
      if (state.result) {
        console.log('Result:', state.result.result);
        console.log('Cost:', getTotalCost());
      }
    } catch (error) {
      console.error('Execution failed:', error);
    }
  };
  
  return <button onClick={handleRun}>Run Graph</button>;
}
```

## Testing

### Manual Testing

1. **Start backend**:
   ```bash
   uvicorn n3_server.main:app --reload
   ```

2. **Start frontend**:
   ```bash
   cd src/web/graph-editor
   npm run dev
   ```

3. **Test execution**:
   - Open graph editor
   - Configure input data
   - Click Execute
   - Verify trace display

### Example Test Data

**Customer Support Triage**:
```json
{
  "ticket_text": "I can't log into my account. Tried password reset but email never arrives. URGENT!",
  "customer_tier": "enterprise"
}
```

**Research Pipeline**:
```json
{
  "research_question": "What are the latest advances in large language model reasoning?"
}
```

## Next Steps

### Enhancements

1. **Real-time Updates**:
   - WebSocket connection for streaming execution
   - Progress indicators for long-running graphs
   - Partial result display

2. **Advanced Visualization**:
   - Timeline view of spans
   - Flamegraph for performance analysis
   - Cost breakdown by component
   - Token usage trends

3. **Debugging Tools**:
   - Span search and filtering
   - Compare multiple executions
   - Export trace data
   - Replay execution

4. **UI Improvements**:
   - Save/load input templates
   - Execution history
   - Quick actions (retry, duplicate)
   - Keyboard shortcuts

5. **Performance**:
   - Virtual scrolling for large traces
   - Lazy loading of span details
   - Cached execution results

## Files Modified

```
src/web/graph-editor/src/
├── hooks/
│   └── useExecution.ts              (NEW - 133 lines)
├── components/
│   └── ExecutionPanel.tsx           (UPDATED - 275 lines)
└── lib/
    └── api.ts                       (UPDATED - added validateGraph)
```

## Status

✅ **Complete**
- useExecution hook with full state management
- ExecutionPanel with comprehensive UI
- API integration with backend
- Telemetry visualization
- Error handling
- Cost/token tracking
- Hierarchical trace display

## Related Documentation

- Backend: `n3_server/api/execution.py`
- Executor: `n3_server/execution/executor.py`
- Converter: `n3_server/converter/enhanced_converter.py`
- Examples: `examples/agent_graphs/`
- Testing: `tests/examples/test_agent_graph_examples.py`
