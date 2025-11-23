# N3 Visual Graph Editor

Production-ready visual graph editor companion for the **Namel3ss (N3)** programming language. This editor parses N3 source files and visualizes chains, agents, RAG pipelines, and memories as interactive nodes with real-time collaboration support.

## 🎯 Features

### Core Functionality

- **AST-to-Graph Visualization**: Automatically parses `.ai` files and generates interactive graph representations
- **Rich Node Types**: Specialized components for apps, prompts, chains, agents, RAG pipelines, LLMs, tools, memories, datasets, and indices
- **Real-time Collaboration**: Multiple users can edit the same graph simultaneously via Yjs + WebSockets
- **Auto-layout**: Intelligent graph layout using dagre algorithm
- **Performance Optimized**: Virtualized rendering for large graphs

### User Experience

- **Keyboard Navigation**: Full keyboard support with Tab, Arrow keys, Enter/Space
- **Accessibility**: ARIA labels, roles, and screen reader support
- **Minimap**: Bird's-eye view of the entire graph
- **Search & Filter**: Find nodes and filter by type
- **Property Editing**: Side panel for viewing and editing node properties

### Developer Experience

- **TypeScript**: Fully typed codebase
- **Modern Stack**: React 18, Vite, Tailwind CSS
- **Production Ready**: Linting, formatting, testing configured
- **Comprehensive Docs**: Detailed README files and inline comments

## 📦 Project Structure

```
packages/
├── n3-graph-editor/          # React frontend application
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── nodes/       # Custom node types
│   │   │   ├── GraphEditor.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── Toolbar.tsx
│   │   ├── store/           # Zustand + Yjs state
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── README.md
│
└── n3-graph-server/          # Node.js backend server
    ├── src/
    │   ├── index.ts         # Express server
    │   ├── parser.ts        # N3 parser bridge
    │   └── graph-transformer.ts  # AST → graph
    ├── parser-bridge.py     # Python parser wrapper
    ├── package.json
    └── README.md
```

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- Python 3.10+
- npm or yarn

### Installation

1. **Install the Namel3ss package** (if not already installed):

```bash
cd /path/to/namel3ss-programming-language
pip install -e .
```

2. **Install backend dependencies**:

```bash
cd packages/n3-graph-server
npm install
```

3. **Install frontend dependencies**:

```bash
cd packages/n3-graph-editor
npm install
```

### Running the Editor

1. **Start the backend server**:

```bash
cd packages/n3-graph-server
npm run dev
```

Server runs on `http://localhost:3001`

2. **Start the frontend** (in a new terminal):

```bash
cd packages/n3-graph-editor
npm run dev
```

Editor opens at `http://localhost:3000`

3. **Open the editor** in your browser and load a `.ai` file from the sidebar!

## 📖 Usage

### Loading Files

1. The sidebar displays all `.ai` files in your workspace
2. Click on a file to parse and visualize it
3. The graph will render with automatic layout

### Navigating the Graph

- **Pan**: Click and drag the canvas
- **Zoom**: Mouse wheel or toolbar buttons
- **Select**: Click on nodes
- **Multi-select**: Ctrl/Cmd + Click
- **Fit View**: Click "Fit" or press Ctrl+F

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Tab` | Navigate between nodes |
| `Arrow Keys` | Move selected nodes |
| `Delete` / `Backspace` | Delete selected nodes |
| `Ctrl/Cmd + F` | Fit view |
| `Ctrl/Cmd + L` | Auto-layout |
| `Ctrl/Cmd + S` | Save changes |
| `+` / `-` | Zoom in/out |

### Node Types

The editor visualizes the following N3 entities:

| Node Type | Color | Description |
|-----------|-------|-------------|
| **App** | Purple | Application container |
| **Prompt** | Blue | Prompt templates with input/output schemas |
| **Chain** | Green | Multi-step AI workflows |
| **Agent** | Orange | AI agents with tools and memory |
| **RAG Pipeline** | Pink | Retrieval-augmented generation configs |
| **Index** | Orange | Vector indices for RAG |
| **LLM** | Indigo | Language model configurations |
| **Tool** | Teal | Agent tools and integrations |
| **Memory** | Purple | Memory stores (session/user/global) |
| **Dataset** | Lime | Data sources |

### Properties Panel

Select any node to view its properties:

- **Name**: Node identifier
- **Type**: Entity type
- **Configuration**: Type-specific parameters
- **Relationships**: Connected nodes

## 🏗️ Architecture

### Frontend (n3-graph-editor)

**Stack:**
- React 18 + TypeScript
- React Flow for graph visualization
- Zustand for state management
- Yjs for real-time collaboration
- Tailwind CSS for styling
- Vite for building

**Key Components:**

1. **GraphEditor**: Main React Flow canvas with custom nodes
2. **Sidebar**: File browser and property editor
3. **Toolbar**: Controls for layout, zoom, and save
4. **Custom Nodes**: Type-specific node components with ARIA support
5. **graphStore**: Zustand store integrated with Yjs

### Backend (n3-graph-server)

**Stack:**
- Node.js + TypeScript
- Express.js for REST API
- ws for WebSocket connections
- y-websocket for Yjs collaboration
- chokidar for file watching

**Key Modules:**

1. **index.ts**: Express server with REST endpoints and WebSocket
2. **parser.ts**: Bridge to Python N3 parser via subprocess
3. **graph-transformer.ts**: Converts AST to graph structure
4. **parser-bridge.py**: Python script that calls namel3ss parser

### Data Flow

```
.ai File
   ↓
Python Parser (namel3ss.lang.grammar)
   ↓
AST (JSON)
   ↓
Graph Transformer
   ↓
Graph { nodes, edges }
   ↓
React Flow Visualization
   ↓
Yjs Sync (WebSocket)
   ↓
Other Clients
```

## 🔧 Configuration

### Environment Variables

**Frontend** (`.env` in `n3-graph-editor/`):

```env
VITE_WS_URL=ws://localhost:3001
VITE_API_URL=http://localhost:3001
```

**Backend** (`.env` in `n3-graph-server/`):

```env
PORT=3001
WORKSPACE_PATH=/path/to/your/n3/files
```

## 🧪 Testing

### Frontend Tests

```bash
cd packages/n3-graph-editor
npm test
```

### Backend Tests

```bash
cd packages/n3-graph-server
npm test
```

## 📝 API Reference

See individual README files:

- [Frontend README](./packages/n3-graph-editor/README.md)
- [Backend README](./packages/n3-graph-server/README.md)

### REST Endpoints

- `GET /api/parse?file=<path>` - Parse and visualize a file
- `POST /api/parse` - Parse source code
- `GET /api/files` - List all .ai files
- `POST /api/save` - Save changes (TODO)
- `GET /health` - Health check

### WebSocket

- `ws://localhost:3001/<docName>` - Yjs collaboration

## 🎨 Customization

### Adding New Node Types

1. Define the node component in `src/components/nodes/CustomNodes.tsx`
2. Add the type to `nodeTypes` object
3. Update `graph-transformer.ts` to generate the node
4. Add styling in `src/index.css`

### Custom Layout

Modify `autoLayout()` in `graphStore.ts` to use a different algorithm (e.g., elkjs).

### Theming

Update Tailwind config in `tailwind.config.js` for custom colors.

## 🚀 Deployment

### Docker Deployment

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  backend:
    build: ./packages/n3-graph-server
    ports:
      - "3001:3001"
    environment:
      - WORKSPACE_PATH=/workspace
    volumes:
      - ./:/workspace

  frontend:
    build: ./packages/n3-graph-editor
    ports:
      - "3000:80"
    environment:
      - VITE_WS_URL=ws://localhost:3001
      - VITE_API_URL=http://localhost:3001
```

Run with:

```bash
docker-compose up
```

### Production Build

**Backend:**

```bash
cd packages/n3-graph-server
npm run build
npm start
```

**Frontend:**

```bash
cd packages/n3-graph-editor
npm run build
# Serve dist/ with nginx or any static server
```

## 🐛 Troubleshooting

### Graph not loading

- Ensure backend server is running on port 3001
- Check browser console for errors
- Verify `.ai` file path is correct

### WebSocket connection failed

- Check `VITE_WS_URL` in frontend `.env`
- Ensure no firewall blocking port 3001
- Verify backend server is running

### Python parser errors

- Ensure namel3ss package is installed: `pip install -e .`
- Check Python version (3.10+ required)
- Verify `parser-bridge.py` is executable

### Layout issues

- Click "Layout" button to re-run auto-layout
- Try "Fit" to center the graph
- Manually drag nodes to adjust

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run linting: `npm run lint`
5. Format code: `npm run format`
6. Submit a pull request

## 📄 License

MIT License - see [LICENSE](../../LICENSE)

## 🙏 Credits

Built with:

- [React Flow](https://reactflow.dev/) - Graph visualization
- [Yjs](https://yjs.dev/) - Real-time collaboration
- [Zustand](https://zustand-demo.pmnd.rs/) - State management
- [Tailwind CSS](https://tailwindcss.com/) - Styling
- [Vite](https://vitejs.dev/) - Build tool
- [Express](https://expressjs.com/) - Backend server

## 📞 Support

For issues or questions:

- Open an issue on [GitHub](https://github.com/SsebowaDisan/namel3ss-programming-language/issues)
- Check existing [documentation](./docs/)
- Review example `.ai` files

---

**Happy Graph Editing! 🎉**
