# N3 Graph Editor

A production-ready visual graph editor for the Namel3ss (N3) programming language. This editor parses N3 source files and visualizes chains, agents, RAG pipelines, and memories as interactive nodes with real-time collaboration support.

## Features

- **AST-to-Graph Visualization**: Automatically parses `.ai` files and generates interactive graph representations
- **Custom Node Types**: Specialized components for each N3 entity (prompts, chains, agents, RAG pipelines, LLMs, tools, etc.)
- **Real-time Collaboration**: Powered by Yjs and WebSockets for synchronized multi-user editing
- **Accessibility**: Full keyboard navigation, ARIA labels, and screen reader support
- **Auto-layout**: Automatic graph layout using dagre algorithm
- **Performance Optimized**: Virtualized rendering with React Flow's `onlyRenderVisibleElements`
- **Beautiful UI**: Built with Tailwind CSS and custom styled components

## Technology Stack

- **Frontend**: React 18 + TypeScript + Vite
- **Graph Library**: React Flow (MIT licensed)
- **State Management**: Zustand for local state, Yjs for collaborative state
- **Real-time Sync**: Yjs + y-websocket
- **Styling**: Tailwind CSS
- **Layout**: dagre for automatic graph layout

## Prerequisites

- Node.js 18+ 
- npm or yarn
- Python 3.10+ (for N3 parser backend)

## Installation

```bash
cd packages/n3-graph-editor
npm install
```

## Development

1. Start the backend server (in a separate terminal):

```bash
cd packages/n3-graph-server
npm install
npm run dev
```

2. Start the frontend development server:

```bash
cd packages/n3-graph-editor
npm run dev
```

The editor will be available at `http://localhost:3000`.

## Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Configuration

Create a `.env` file in the root directory:

```env
VITE_WS_URL=ws://localhost:3001
VITE_API_URL=http://localhost:3001
```

## Usage

### Loading a File

1. The sidebar shows all `.ai` files in your workspace
2. Click on a file to load and visualize it
3. The graph will automatically render with proper layout

### Navigation

- **Pan**: Click and drag on the canvas
- **Zoom**: Mouse wheel or use zoom controls
- **Select Node**: Click on any node
- **Multi-select**: Ctrl/Cmd + Click
- **Fit View**: Click "Fit" button or Ctrl/Cmd + F

### Keyboard Shortcuts

- `Tab`: Navigate between nodes
- `Arrow Keys`: Move selected nodes
- `Delete/Backspace`: Delete selected nodes
- `Ctrl/Cmd + F`: Fit view
- `Ctrl/Cmd + L`: Auto-layout
- `Ctrl/Cmd + S`: Save changes
- `+/-`: Zoom in/out

### Node Types

The editor supports the following N3 entity types:

- **App** (purple): Application container
- **Prompt** (blue): Prompt templates
- **Chain** (green): Multi-step workflows
- **Agent** (orange): AI agents with tools
- **RAG Pipeline** (pink): Retrieval-augmented generation pipelines
- **Index** (orange): Vector indices
- **LLM** (indigo): Language models
- **Tool** (teal): Agent tools
- **Memory** (purple): Memory stores
- **Dataset** (lime): Data sources

### Properties Panel

Select any node to view and edit its properties in the sidebar:

- Name and type
- Configuration parameters
- References to other nodes
- Metadata

### Collaboration

Multiple users can edit the same graph simultaneously:

1. Each user connects to the same document (identified by filename)
2. Changes are synchronized in real-time via WebSocket
3. Presence indicators show which users are editing

## Accessibility

The editor is fully accessible:

- All nodes are keyboard navigable
- ARIA labels and roles for screen readers
- Focus indicators and semantic HTML
- Live regions for dynamic updates
- High contrast mode support

## API Integration

The editor communicates with the backend server:

### GET `/api/parse?file=<path>`

Parse a `.ai` file and return graph structure.

**Response:**
```json
{
  "graph": {
    "nodes": [...],
    "edges": [...]
  },
  "module": { ... }
}
```

### POST `/api/parse`

Parse N3 source code directly.

**Body:**
```json
{
  "source": "...",
  "fileName": "source.ai"
}
```

### GET `/api/files`

List all `.ai` files in workspace.

**Response:**
```json
{
  "files": ["demo_app.ai", "example.ai"]
}
```

### POST `/api/save`

Save graph changes back to `.ai` file (not yet implemented).

## Architecture

```
n3-graph-editor/
├── src/
│   ├── components/
│   │   ├── nodes/
│   │   │   └── CustomNodes.tsx    # Node type definitions
│   │   ├── GraphEditor.tsx        # Main editor component
│   │   ├── Sidebar.tsx            # File list & properties
│   │   └── Toolbar.tsx            # Controls
│   ├── store/
│   │   └── graphStore.ts          # Zustand + Yjs integration
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── package.json
├── vite.config.ts
└── tailwind.config.js
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

## Troubleshooting

### Graph not loading

- Ensure the backend server is running
- Check browser console for errors
- Verify the `.ai` file exists in the workspace

### WebSocket connection failed

- Check that the WebSocket URL is correct in `.env`
- Ensure port 3001 is not blocked by firewall
- Verify the backend server is running

### Layout issues

- Click "Layout" to run auto-layout algorithm
- Manually adjust node positions by dragging
- Use "Fit" to center the graph

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT

## Credits

Built with:
- [React Flow](https://reactflow.dev/) - Graph visualization
- [Yjs](https://yjs.dev/) - Real-time collaboration
- [Zustand](https://zustand-demo.pmnd.rs/) - State management
- [Tailwind CSS](https://tailwindcss.com/) - Styling

---

For more information about the Namel3ss language, see the [main README](../../README.md).
