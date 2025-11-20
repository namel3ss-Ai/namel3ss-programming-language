# N3 Agent Graph Editor - Quick Reference

## 🚀 Quick Start

### One-line setup:
```bash
./setup-agent-graph.sh
```

### Manual startup:
```bash
# Terminal 1: Backend
cd n3_server && uvicorn api.main:app --reload --port 8000

# Terminal 2: Yjs Server  
cd yjs-server && npm start

# Terminal 3: Frontend
cd src/web/graph-editor && npm run dev
```

### Access Points:
- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Jaeger**: http://localhost:16686

## 📡 API Cheat Sheet

### Graphs
```bash
# Get graph
GET /api/graphs/{projectId}

# Update graph
PUT /api/graphs/{projectId}
  Body: { nodes: [...], edges: [...], metadata: {...} }

# Execute
POST /api/graphs/{projectId}/execute
  Body: { entry: "main", input: {...} }
```

### Shares
```bash
# Create share link
POST /api/projects/{projectId}/shares
  Body: { role: "viewer|editor", expiresInHours: 24 }

# Validate token
GET /api/projects/open-by-token?token={token}
```

### Tools
```bash
# List tools
GET /api/tools

# Execute tool
POST /api/tools/execute
  Body: { name: "add", args: { a: 5, b: 3 } }
```

### RLHF
```bash
# Submit feedback
POST /api/feedback/{agentId}
  Body: { prompt: "...", response: "...", score: 0.9, runId: "..." }

# Train policy
POST /api/train_policy/{agentId}
  Body: { maxSteps: 1000, learningRate: 0.00001 }
```

## 🎨 Node Types

| Type | Color | Purpose |
|------|-------|---------|
| `start` | Gray | Flow entry point |
| `prompt` | Blue | Prompt templates |
| `agent` | Purple | LLM agents |
| `agentGraph` | Purple | Nested graphs |
| `pythonHook` | Purple | Custom functions |
| `ragDataset` | Green | RAG retrievers |
| `memoryStore` | Green | Conversation memory |
| `safetyPolicy` | Green | Content filters |
| `condition` | Yellow | If/else branching |
| `end` | Gray | Flow termination |

## 🔧 Tool Registry

### Register a tool:
```python
from n3_server.api.tools import tool

@tool(description="Your description", tags=["category"])
def my_tool(arg1: str, arg2: int) -> str:
    return f"Result: {arg1} * {arg2}"
```

### Execute from frontend:
```typescript
import { toolApi } from '@/lib/api';

const result = await toolApi.executeTool({
  name: "my_tool",
  args: { arg1: "test", arg2: 42 }
});
```

## 🤝 Collaboration

### Yjs Integration:
```typescript
import { useYjsGraph } from '@/hooks/useYjsGraph';

const { users, sync, isConnected } = useYjsGraph({
  projectId,
  nodes,
  edges,
  onRemoteUpdate: (nodes, edges) => {
    // Update local state
  }
});

// Sync changes
useEffect(() => sync(nodes, edges), [nodes, edges]);
```

## 📊 OpenTelemetry

### View traces:
1. Open http://localhost:16686
2. Select service: `n3-server`
3. Click "Find Traces"

### Custom spans:
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("my_operation") as span:
    span.set_attribute("key", "value")
    # Your code
```

## 🐳 Docker Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop all services
docker-compose down

# Rebuild images
docker-compose build --no-cache

# Database shell
docker exec -it postgres psql -U n3 -d n3_graphs
```

## 🗄️ Database

### Migrations:
```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Tables:
- `projects`: Graph data (JSON)
- `share_links`: Share tokens
- `feedback`: RLHF submissions
- `policy_versions`: Trained policies

## 🧪 Testing

### Frontend:
```bash
cd src/web/graph-editor
npm run test              # Playwright E2E
npm run lint              # ESLint
npm run type-check        # TypeScript
```

### Backend:
```bash
cd n3_server
pytest tests/             # Unit tests
mypy .                    # Type checking
pylint *.py               # Linting
```

## 📦 Project Structure

```
├── src/web/graph-editor/     Frontend (React + TypeScript)
├── n3_server/                Backend (FastAPI + Python)
├── yjs-server/               Real-time collab (Node.js)
├── docker-compose.yml        Orchestration
├── alembic.ini               DB migrations
└── setup-agent-graph.sh      Setup script
```

## 🔑 Environment Variables

### Backend (.env):
```bash
DATABASE_URL=postgresql+asyncpg://n3:n3password@localhost:5432/n3_graphs
OTLP_ENDPOINT=http://localhost:4317
JWT_SECRET=your-secret-key
CORS_ORIGINS=["http://localhost:3000"]
```

### Frontend (.env):
```bash
VITE_API_URL=http://localhost:8000/api
VITE_YJS_URL=ws://localhost:1234
```

## ⚡ Performance Tips

1. **Frontend caching**: 5min stale time (TanStack Query)
2. **Database indexing**: Token, agent_id fields indexed
3. **React Flow**: Virtualized viewport for 1000+ nodes
4. **Yjs compression**: Binary encoding, delta updates
5. **Batch spans**: OpenTelemetry batch processor (default)

## 🐛 Troubleshooting

### "Cannot connect to database"
```bash
docker-compose up -d postgres
sleep 5  # Wait for startup
alembic upgrade head
```

### "Yjs connection failed"
```bash
cd yjs-server && npm start
# Check http://localhost:1234
```

### "Module not found" (Frontend)
```bash
cd src/web/graph-editor
rm -rf node_modules package-lock.json
npm install
```

### "Import error" (Backend)
```bash
cd n3_server
pip install -r requirements.txt --upgrade
```

## 📚 Documentation

- **Implementation Guide**: `AGENT_GRAPH_EDITOR_GUIDE.md`
- **README**: `AGENT_GRAPH_README.md`
- **Summary**: `AGENT_GRAPH_IMPLEMENTATION_SUMMARY.md`
- **API Docs**: http://localhost:8000/docs (interactive)

## 🎯 Status

**Completed** (11/18 tasks):
- ✅ Project structure
- ✅ Database models
- ✅ API endpoints
- ✅ Tool registry
- ✅ React components
- ✅ Graph canvas
- ✅ Yjs collaboration
- ✅ WebSocket server
- ✅ Docker setup
- ✅ Documentation

**Remaining** (7/18 tasks):
- 🔄 N3 AST converter
- 🔄 Execution integration
- 🔄 Tool adapters
- 🔄 RLHF training
- ⏳ Authentication
- ⏳ E2E tests
- ⏳ CI/CD

## 💡 Tips

1. **Use API docs**: http://localhost:8000/docs for interactive testing
2. **Check Jaeger**: Traces show full execution flow
3. **Inspect DB**: `docker exec -it postgres psql -U n3 -d n3_graphs`
4. **Hot reload**: Both frontend and backend support live reload
5. **Share links**: Create via API, test with `/open/{token}` route

## 🚦 Health Checks

```bash
# Backend
curl http://localhost:8000/health

# Yjs Server
curl http://localhost:1234

# Jaeger
curl http://localhost:16686

# Database
docker exec postgres pg_isready -U n3
```

---

**Need help?** Check the full guides or inspect the `/docs` endpoint!
