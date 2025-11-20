# N3 Agent Graph Editor

Production-ready visual graph editor with real-time collaboration for building and executing agent chains in the Namel3ss programming language.

## Features

### ✅ Visual Graph Editor
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

## Architecture

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

## Quick Start

### Prerequisites
- Docker & docker-compose
- Node.js 18+
- Python 3.11+

### Development

1. **Clone and navigate**:
```bash
cd /Users/disanssebowabasalidde/Documents/GitHub/namel3ss-programming-language
```

2. **Install frontend dependencies**:
```bash
cd src/web/graph-editor
npm install
```

3. **Install backend dependencies**:
```bash
cd ../../n3_server
pip install -r requirements.txt
```

4. **Start services**:
```bash
cd ..
docker-compose up -d
```

5. **Run migrations**:
```bash
alembic upgrade head
```

6. **Access**:
- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Jaeger**: http://localhost:16686

### Production

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## API Overview

### Graph Management
```http
GET  /api/graphs/{projectId}
PUT  /api/graphs/{projectId}
POST /api/graphs/{projectId}/execute
```

### Collaboration
```http
POST /api/projects/{projectId}/shares
GET  /api/projects/{projectId}/shares
GET  /api/projects/open-by-token?token={token}
```

### Tools
```http
GET  /api/tools
POST /api/tools/execute
POST /api/tools/register
```

### RLHF
```http
POST /api/feedback/{agentId}
POST /api/train_policy/{agentId}
GET  /api/policies/{agentId}
```

## Technology Stack

### Frontend
- React 18.2 + TypeScript 5.2
- Vite 5 (build tool)
- React Flow 11 (graph visualization)
- Yjs 13 + y-websocket (real-time)
- TanStack Query 5 (server state)
- Tailwind CSS 3 + Radix UI
- Monaco Editor (code editing)

### Backend
- FastAPI + Uvicorn
- SQLAlchemy 2.0 (async ORM)
- PostgreSQL 16
- OpenTelemetry SDK
- HuggingFace TRL (RLHF)

### Infrastructure
- Docker + docker-compose
- Jaeger (tracing)
- Node.js 18 (Yjs server)

## Documentation

- **[Implementation Guide](./AGENT_GRAPH_EDITOR_GUIDE.md)**: Detailed technical docs
- **API Docs**: http://localhost:8000/docs (when running)

## Project Structure

```
├── src/web/graph-editor/         # React frontend
│   ├── src/
│   │   ├── components/           # UI components
│   │   │   ├── nodes/           # Custom React Flow nodes
│   │   │   ├── GraphCanvas.tsx  # Main graph editor
│   │   │   └── ...
│   │   ├── hooks/               # React hooks (Yjs integration)
│   │   ├── lib/                 # API client
│   │   ├── pages/               # Route pages
│   │   └── types/               # TypeScript definitions
│   ├── package.json
│   └── vite.config.ts
│
├── n3_server/                    # FastAPI backend
│   ├── api/
│   │   ├── main.py              # FastAPI app
│   │   ├── graphs.py            # Graph endpoints
│   │   ├── shares.py            # Share links
│   │   ├── tools.py             # Tool registry
│   │   └── policies.py          # RLHF endpoints
│   ├── db/
│   │   ├── models.py            # SQLAlchemy models
│   │   ├── session.py           # Database session
│   │   └── migrations/          # Alembic migrations
│   ├── config.py                # Settings
│   └── requirements.txt
│
├── yjs-server/                   # Real-time collaboration
│   ├── server.js                # WebSocket server
│   └── package.json
│
├── docker-compose.yml            # Development orchestration
├── Dockerfile.backend
├── Dockerfile.frontend
├── Dockerfile.yjs
└── alembic.ini                   # Database migrations
```

## Testing

### Frontend (Playwright)
```bash
cd src/web/graph-editor
npm run test
```

### Backend (pytest)
```bash
cd n3_server
pytest tests/
```

## Contributing

This is part of the Namel3ss language project. The agent graph editor provides a visual interface for building complex LLM agent workflows with:
- Multimodal RAG integration
- Adaptive policies via RLHF
- Real-time collaborative editing
- Full execution observability

## Next Steps

1. ✅ Core infrastructure (complete)
2. 🔄 N3 AST ↔ Graph JSON converter
3. 🔄 Execution engine integration
4. 🔄 OpenAPI/LangChain tool adapters
5. 🔄 RLHF training pipeline
6. ⏳ Authentication (OAuth2/JWT)
7. ⏳ E2E test suites
8. ⏳ CI/CD pipeline

## License

See root LICENSE file.
