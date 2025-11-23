# N3 Graph Server

Backend server for the N3 Graph Editor. Parses `.ai` files using the Python N3 compiler and serves graph data via REST API. Also manages WebSocket connections for real-time collaborative editing.

## Features

- **N3 Parser Bridge**: Interfaces with Python N3 parser to generate AST
- **Graph Transformation**: Converts AST to graph structure (nodes + edges)
- **REST API**: Serves graph data and file listings
- **WebSocket Server**: Manages Yjs collaboration via y-websocket
- **File Watching**: Detects changes to `.ai` files and notifies clients
- **CORS Enabled**: Allows cross-origin requests from frontend

## Technology Stack

- **Runtime**: Node.js with TypeScript
- **Framework**: Express.js
- **WebSocket**: ws + y-websocket
- **Collaboration**: Yjs
- **File Watching**: chokidar
- **Parser**: Python subprocess calling namel3ss parser

## Prerequisites

- Node.js 18+
- Python 3.10+ with namel3ss package installed
- npm or yarn

## Installation

```bash
cd packages/n3-graph-server
npm install
```

## Development

```bash
npm run dev
```

Server runs on `http://localhost:3001` by default.

## Building for Production

```bash
npm run build
npm start
```

## Configuration

Set environment variables:

```bash
PORT=3001                    # HTTP server port
WORKSPACE_PATH=/path/to/n3   # Path to .ai files workspace
```

Or create a `.env` file:

```env
PORT=3001
WORKSPACE_PATH=/Users/username/projects/n3-workspace
```

## API Endpoints

### GET `/api/parse`

Parse a `.ai` file and return graph structure.

**Query Parameters:**
- `file` (required): Relative or absolute path to `.ai` file

**Response:**
```json
{
  "graph": {
    "nodes": [
      {
        "id": "app_1",
        "type": "app",
        "data": {
          "label": "Demo App",
          "description": "Application: Demo App",
          "metadata": { ... }
        },
        "position": { "x": 50, "y": 50 }
      },
      ...
    ],
    "edges": [
      {
        "id": "app_1-llm_2",
        "source": "app_1",
        "target": "llm_2",
        "type": "reference",
        "label": "defines"
      },
      ...
    ]
  },
  "module": {
    "type": "Module",
    "path": "demo_app.ai",
    "body": [ ... ]
  }
}
```

**Example:**
```bash
curl "http://localhost:3001/api/parse?file=demo_app.ai"
```

### POST `/api/parse`

Parse N3 source code without a file.

**Body:**
```json
{
  "source": "app \"My App\"\n\nllm gpt4:\n  provider: openai\n  model: gpt-4",
  "fileName": "source.ai"
}
```

**Response:** Same as GET `/api/parse`

**Example:**
```bash
curl -X POST http://localhost:3001/api/parse \
  -H "Content-Type: application/json" \
  -d '{"source": "app \"Test\"\n", "fileName": "test.ai"}'
```

### GET `/api/files`

List all `.ai` files in the workspace.

**Response:**
```json
{
  "files": [
    "demo_app.ai",
    "examples/agent_example.ai",
    "examples/rag_example.ai"
  ]
}
```

**Example:**
```bash
curl http://localhost:3001/api/files
```

### POST `/api/save`

Save graph changes back to `.ai` file (not yet implemented).

**Body:**
```json
{
  "file": "demo_app.ai",
  "graph": { ... }
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "workspace": "/path/to/workspace"
}
```

## WebSocket Server

The server hosts a Yjs WebSocket server for real-time collaboration.

**Connection:** `ws://localhost:3001/<docName>`

**Example:**
```javascript
import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';

const ydoc = new Y.Doc();
const provider = new WebsocketProvider('ws://localhost:3001', 'my-document', ydoc);
```

### Running Standalone Yjs Server

You can also run a dedicated Yjs server:

```bash
PORT=1234 YPERSISTENCE=./dbDir node node_modules/y-websocket/bin/server.cjs
```

## Parser Bridge

The server uses a Python bridge script (`parser-bridge.py`) to interface with the N3 parser.

### How it works:

1. Node.js spawns a Python subprocess
2. Passes `.ai` file path or source code via stdin
3. Python script imports `namel3ss.lang.grammar.parse_module`
4. Parses the file and serializes AST to JSON
5. Outputs JSON to stdout
6. Node.js receives and processes the JSON

### Python Script Usage

```bash
# Parse a file
python3 parser-bridge.py demo_app.ai

# Parse from stdin
echo 'app "Test"' | python3 parser-bridge.py --stdin test.ai
```

## Graph Transformation

The `graph-transformer.ts` module converts N3 AST to a graph structure:

### Node Types

Each N3 entity becomes a typed node:

- `app`: Application
- `llm`: Language model
- `prompt`: Prompt template
- `chain`: Workflow chain
- `chainStep`: Individual step in a chain
- `agent`: AI agent
- `ragPipeline`: RAG retrieval pipeline
- `index`: Vector index
- `tool`: Agent tool
- `memory`: Memory store
- `dataset`: Data source

### Edge Types

- `reference`: References between entities (e.g., app → llm)
- `step`: Sequential steps in a chain
- `conditional`: Conditional edges (future)

### Layout

Initial layout arranges nodes by type in columns:

```
[App] → [Datasets] → [Indices] → [LLMs] → [Tools] → [Prompts] → [RAG] → [Agents] → [Chains]
```

## File Watching

The server watches for changes to `.ai` files:

- Uses `chokidar` for cross-platform file watching
- Clears cache when file changes
- Notifies connected WebSocket clients
- Clients can reload the file automatically

## Error Handling

- Parse errors return 500 with error message
- Missing files return 404
- Invalid requests return 400
- All errors include descriptive messages

## Architecture

```
n3-graph-server/
├── src/
│   ├── index.ts              # Main Express server
│   ├── parser.ts             # N3 parser bridge
│   └── graph-transformer.ts  # AST → graph conversion
├── parser-bridge.py          # Python parser wrapper
├── package.json
├── tsconfig.json
└── README.md
```

## Testing

```bash
npm test
```

## Linting & Formatting

```bash
npm run lint
npm run format
```

## Deployment

### Docker (recommended)

Create `Dockerfile`:

```dockerfile
FROM node:18-alpine

# Install Python
RUN apk add --no-cache python3 py3-pip

# Install namel3ss
COPY ../../requirements.txt .
RUN pip3 install -r requirements.txt

# Copy server code
WORKDIR /app
COPY package*.json ./
RUN npm install --production

COPY . .
RUN npm run build

EXPOSE 3001
CMD ["npm", "start"]
```

Build and run:

```bash
docker build -t n3-graph-server .
docker run -p 3001:3001 -v /path/to/workspace:/workspace \
  -e WORKSPACE_PATH=/workspace n3-graph-server
```

### Standalone

1. Build the server: `npm run build`
2. Install Python dependencies: `pip install -r ../../requirements.txt`
3. Set environment variables
4. Run: `npm start`

## Troubleshooting

### Python parser not found

Ensure `namel3ss` package is installed:

```bash
cd ../..
pip install -e .
```

### WebSocket connection issues

- Check firewall settings
- Verify port 3001 is accessible
- Check CORS configuration

### File watching not working

- Ensure workspace path is correct
- Check file permissions
- Try restarting the server

## Performance

- **Caching**: Parsed graphs are cached in memory
- **File Watching**: Efficient change detection with chokidar
- **Streaming**: Parser uses stdin/stdout for efficient data transfer
- **Concurrency**: Express handles multiple requests concurrently

## Security

- CORS is enabled for development (configure for production)
- File access is restricted to workspace directory
- No file write operations (except save, when implemented)
- WebSocket connections are isolated by document name

## Future Enhancements

- [ ] Implement graph-to-N3 conversion for bidirectional sync
- [ ] Add authentication for WebSocket connections
- [ ] Implement file upload/create endpoints
- [ ] Add GraphQL API option
- [ ] Cache persistence (Redis/disk)
- [ ] Cluster support for horizontal scaling
- [ ] Metrics and monitoring endpoints

## License

MIT

## Contributing

See [Contributing Guidelines](../../CONTRIBUTING.md)

---

For the frontend editor, see [n3-graph-editor README](../n3-graph-editor/README.md).
