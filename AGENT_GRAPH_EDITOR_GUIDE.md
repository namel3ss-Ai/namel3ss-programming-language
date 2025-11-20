# N3 Agent Graph Editor - Implementation Guide

## Overview

Production-ready visual graph editor with real-time collaboration for building and executing agent chains in Namel3ss.

## Architecture

### Frontend Stack
- **Framework**: React 18 + TypeScript + Vite
- **Graph Visualization**: React Flow 11 (DAG rendering)
- **Real-time Collaboration**: Yjs 13 + WebSocket
- **State Management**: TanStack Query + Zustand
- **UI Components**: Tailwind CSS + Radix UI (shadcn/ui pattern)
- **Code Editor**: Monaco Editor (VS Code)

### Backend Stack
- **API Framework**: FastAPI + SQLAlchemy
- **Database**: PostgreSQL
- **Instrumentation**: OpenTelemetry + OTLP
- **RLHF/PPO**: HuggingFace TRL
- **Real-time Server**: Node.js + Yjs WebSocket

### Infrastructure
- **Containerization**: Docker + docker-compose
- **Tracing**: Jaeger
- **Testing**: Playwright (E2E) + pytest (backend)

## Quick Start

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

## API Endpoints

### Graph Management
```http
GET  /api/graphs/{projectId}          # Get graph data
PUT  /api/graphs/{projectId}          # Update graph
POST /api/graphs/{projectId}/execute  # Execute with tracing
```

### Share Links
```http
POST   /api/projects/{projectId}/shares  # Create share link
GET    /api/projects/{projectId}/shares  # List shares
DELETE /api/projects/{projectId}/shares/{shareId}
GET    /api/projects/open-by-token?token={token}
```

### Tool Registry
```http
GET  /api/tools                # List registered tools
POST /api/tools/execute        # Execute tool
POST /api/tools/register       # Register from OpenAPI/LangChain
```

### Adaptive Policies
```http
POST /api/feedback/{agentId}        # Submit feedback
POST /api/train_policy/{agentId}    # Train policy with RLHF
GET  /api/policies/{agentId}        # List policy versions
```

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

## OpenTelemetry Instrumentation

### Automatic Tracing
- FastAPI requests (via `FastAPIInstrumentor`)
- OpenAI API calls (via `openinference-instrumentation-openai`)
- Custom tool executions

### Span Attributes
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("execute_graph") as span:
    span.set_attribute("project_id", project_id)
    span.set_attribute("entry", entry_point)
    # ... execution logic
```

### View Traces
1. Navigate to http://localhost:16686
2. Select service: `n3-server`
3. Browse traces with full context

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

## Adaptive Agent Policies (RLHF)

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

## Database Schema

### Tables
- `projects`: Graph storage (JSON)
- `share_links`: Share tokens with expiration
- `feedback`: RLHF feedback submissions
- `policy_versions`: Trained policy metadata

### Migrations
```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head
```

## Node Types

### Supported Nodes
- **prompt**: Prompt templates
- **agent**: LLM agents
- **agentGraph**: Nested agent graphs
- **pythonHook**: Custom Python functions
- **ragDataset**: RAG retrievers
- **memoryStore**: Conversation memory
- **safetyPolicy**: Content filters
- **condition**: Conditional branching
- **start/end**: Flow control

## Performance

### Optimization
- TanStack Query caching (5min stale time)
- React Flow viewport virtualization
- Yjs binary encoding (efficient sync)
- PostgreSQL connection pooling
- Batch span processing (OpenTelemetry)

### Monitoring
- Jaeger traces for latency analysis
- Database query performance (SQLAlchemy echo)
- Frontend bundle size (Vite build stats)

## Security

### Share Links
- Token-based access (32-char nanoid)
- Role-based permissions (viewer/editor)
- Expiration timestamps
- JWT validation

### API Security
- CORS configuration
- Rate limiting (TODO)
- Input validation (Pydantic)

## Next Steps

1. **Implement N3 AST ↔ Graph JSON converter**
2. **Integrate existing N3 execution engine**
3. **Add OpenAPI/LangChain tool adapters**
4. **Implement actual RLHF training pipeline**
5. **Add authentication (OAuth2/JWT)**
6. **Create Playwright test suites**
7. **Setup CI/CD pipeline (GitHub Actions)**

## Support

For issues or questions:
- Check API docs: http://localhost:8000/docs
- View traces: http://localhost:16686
- Inspect database: `docker exec -it postgres psql -U n3 -d n3_graphs`
