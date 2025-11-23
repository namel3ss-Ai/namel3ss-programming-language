# N3 Graph Editor Integration Guide

This guide explains how to integrate and use the N3 Graph Editor in your Namel3ss projects.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Integration Patterns](#integration-patterns)
5. [Advanced Usage](#advanced-usage)
6. [Customization](#customization)
7. [Troubleshooting](#troubleshooting)

## Installation

### Method 1: Automated Setup (Recommended)

```bash
cd /path/to/namel3ss-programming-language/packages
node setup.js
```

This script will:
- Install all dependencies for frontend and backend
- Create default `.env` files
- Verify the installation

### Method 2: Manual Setup

**Backend:**

```bash
cd packages/n3-graph-server
npm install
cp .env.example .env
# Edit .env to set WORKSPACE_PATH
```

**Frontend:**

```bash
cd packages/n3-graph-editor
npm install
cp .env.example .env
# Edit .env if needed
```

## Quick Start

### Starting the Development Environment

**Option 1: Automated (runs both servers)**

```bash
cd packages
node dev.js
```

**Option 2: Manual (separate terminals)**

Terminal 1 (Backend):
```bash
cd packages/n3-graph-server
npm run dev
```

Terminal 2 (Frontend):
```bash
cd packages/n3-graph-editor
npm run dev
```

### Opening the Editor

1. Navigate to `http://localhost:3000`
2. The sidebar shows all `.ai` files in your workspace
3. Click on a file to visualize it
4. Use the toolbar controls to navigate and edit

## Architecture Overview

### Components

```
┌─────────────────────────────────────────────────────┐
│                  Browser (Frontend)                  │
│  ┌───────────────────────────────────────────────┐  │
│  │         React + React Flow + Zustand          │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │            Yjs Document                  │  │  │
│  │  │  (Collaborative Graph State)            │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└──────────────────┬──────────────────────────────────┘
                   │
                   │ WebSocket (Yjs Sync)
                   │ HTTP (REST API)
                   │
┌──────────────────▼──────────────────────────────────┐
│              Node.js Backend Server                  │
│  ┌───────────────────────────────────────────────┐  │
│  │      Express + WebSocket (y-websocket)        │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │    Python Subprocess (N3 Parser)        │  │  │
│  │  │  ┌───────────────────────────────────┐  │  │  │
│  │  │  │  namel3ss.lang.grammar.parse      │  │  │  │
│  │  │  └───────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Data Flow

1. **Parse Request**
   - User selects `.ai` file in UI
   - Frontend sends `GET /api/parse?file=<path>`
   - Backend spawns Python process
   - Python parses file using `namel3ss.lang.grammar.parse_module`
   - AST returned as JSON

2. **Graph Transformation**
   - Backend converts AST to graph structure
   - Nodes created for each N3 entity (app, llm, prompt, chain, etc.)
   - Edges created for relationships (references, steps, etc.)
   - Initial layout applied

3. **Visualization**
   - Frontend receives graph JSON
   - React Flow renders nodes and edges
   - Custom node components for each type
   - Minimap and controls added

4. **Collaboration**
   - Graph state synchronized via Yjs
   - Changes broadcast to all connected clients
   - Conflict-free replicated data type (CRDT) ensures consistency

## Integration Patterns

### Embedding in Existing Projects

#### As a Standalone Service

Run the editor as a separate service:

```yaml
# docker-compose.yml
services:
  n3-graph-editor:
    build: ./packages
    ports:
      - "3000:3000"
      - "3001:3001"
    volumes:
      - ./your-n3-files:/workspace
    environment:
      - WORKSPACE_PATH=/workspace
```

#### As a Development Tool

Add to your project's `package.json`:

```json
{
  "scripts": {
    "graph-editor": "cd packages && node dev.js"
  }
}
```

#### Integrated in VS Code

Create a VS Code task (`.vscode/tasks.json`):

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Start N3 Graph Editor",
      "type": "shell",
      "command": "cd packages && node dev.js",
      "isBackground": true,
      "problemMatcher": []
    }
  ]
}
```

### Programmatic Usage

#### Parsing N3 Files

```typescript
import axios from 'axios';

async function parseN3File(filePath: string) {
  const response = await axios.get('http://localhost:3001/api/parse', {
    params: { file: filePath }
  });
  
  return response.data.graph;
}

const graph = await parseN3File('my-app.ai');
console.log(graph.nodes.length, 'nodes');
```

#### Parsing N3 Source

```typescript
async function parseN3Source(source: string) {
  const response = await axios.post('http://localhost:3001/api/parse', {
    source,
    fileName: 'dynamic.ai'
  });
  
  return response.data.graph;
}

const source = `
app "Dynamic App"
llm gpt4:
  provider: openai
  model: gpt-4
`;

const graph = await parseN3Source(source);
```

#### Listing Files

```typescript
async function listN3Files() {
  const response = await axios.get('http://localhost:3001/api/files');
  return response.data.files;
}

const files = await listN3Files();
```

## Advanced Usage

### Custom Node Types

Add a new node type to visualize custom N3 entities:

1. **Define the node component** (`src/components/nodes/CustomNodes.tsx`):

```typescript
export const CustomNode: React.FC<NodeProps> = memo((props) => {
  return (
    <div className="n3-node n3-node-custom">
      <Handle type="target" position={Position.Top} />
      <div className="n3-node-header">⭐ {props.data.label}</div>
      <div className="n3-node-description">{props.data.description}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

// Add to nodeTypes
export const nodeTypes = {
  // ... existing types
  custom: CustomNode,
};
```

2. **Update graph transformer** (`src/graph-transformer.ts`):

```typescript
function processCustomEntity(entity: any, nodes: GraphNode[]): string {
  const id = generateNodeId('custom');
  nodes.push({
    id,
    type: 'custom',
    data: {
      label: entity.name,
      description: entity.description,
    },
    position: { x: 0, y: 0 },
  });
  return id;
}
```

3. **Add styling** (`src/index.css`):

```css
.ai-node-custom {
  border-color: #6366f1;
  background: linear-gradient(135deg, #e0e7ff 0%, #ffffff 100%);
}
```

### Custom Layout Algorithms

Replace dagre with elkjs for hierarchical layouts:

```typescript
// In graphStore.ts
import ELK from 'elkjs/lib/elk.bundled.js';

autoLayout: () => {
  const { nodes, edges } = get();
  
  const elk = new ELK();
  const graph = {
    id: 'root',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'DOWN',
    },
    children: nodes.map(node => ({
      id: node.id,
      width: 180,
      height: 80,
    })),
    edges: edges.map(edge => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    })),
  };
  
  elk.layout(graph).then(layouted => {
    // Apply new positions
    const newNodes = nodes.map(node => {
      const elkNode = layouted.children?.find(n => n.id === node.id);
      return {
        ...node,
        position: {
          x: elkNode?.x || 0,
          y: elkNode?.y || 0,
        },
      };
    });
    
    set({ nodes: newNodes });
  });
}
```

### Persistence

Save graph positions to local storage:

```typescript
// In graphStore.ts
const persistedPositions = localStorage.getItem('graph-positions');

// Load on init
if (persistedPositions) {
  const positions = JSON.parse(persistedPositions);
  // Apply positions to nodes
}

// Save on change
onNodesChange: (changes) => {
  // Apply changes...
  
  // Save positions
  const positions = get().nodes.reduce((acc, node) => {
    acc[node.id] = node.position;
    return acc;
  }, {});
  
  localStorage.setItem('graph-positions', JSON.stringify(positions));
}
```

### Multi-Workspace Support

Support multiple workspaces:

```typescript
// Backend: Add workspace parameter
app.get('/api/parse', (req, res) => {
  const { file, workspace } = req.query;
  const workspacePath = workspace || process.env.WORKSPACE_PATH;
  const filePath = path.join(workspacePath, file);
  // ...
});

// Frontend: Select workspace
const [workspace, setWorkspace] = useState('/default/workspace');

const loadFile = async (file: string) => {
  const response = await axios.get('/api/parse', {
    params: { file, workspace }
  });
  // ...
};
```

## Customization

### Theming

Customize colors in `tailwind.config.js`:

```javascript
theme: {
  extend: {
    colors: {
      primary: {
        50: '#eff6ff',
        500: '#3b82f6',
        700: '#1d4ed8',
      },
      // Add custom colors for node types
      nodeApp: '#8b5cf6',
      nodePrompt: '#3b82f6',
      // ...
    }
  }
}
```

### Node Appearance

Modify node styles in `src/index.css`:

```css
.ai-node-prompt {
  border-color: var(--your-custom-color);
  border-width: 3px;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
```

### Keyboard Shortcuts

Add custom shortcuts in `GraphEditor.tsx`:

```typescript
useEffect(() => {
  const handleKeyDown = (event: KeyboardEvent) => {
    // Custom shortcuts
    if (event.ctrlKey && event.key === 'e') {
      event.preventDefault();
      // Export graph
    }
    
    if (event.ctrlKey && event.key === 'r') {
      event.preventDefault();
      // Refresh
    }
  };
  
  document.addEventListener('keydown', handleKeyDown);
  return () => document.removeEventListener('keydown', handleKeyDown);
}, []);
```

## Troubleshooting

### Common Issues

#### 1. Python Parser Not Found

**Error:** `Cannot find module 'namel3ss'`

**Solution:**
```bash
cd /path/to/namel3ss-programming-language
pip install -e .
```

#### 2. WebSocket Connection Failed

**Error:** `WebSocket connection failed`

**Solution:**
- Check backend is running: `curl http://localhost:3001/health`
- Verify `VITE_WS_URL` in frontend `.env`
- Check firewall settings

#### 3. Graph Not Rendering

**Error:** Blank canvas or "No nodes to display"

**Solution:**
- Check browser console for errors
- Verify file exists: `curl "http://localhost:3001/api/files"`
- Check file parsing: `curl "http://localhost:3001/api/parse?file=demo_app.ai"`

#### 4. Hot Reload Not Working

**Solution:**
- Restart Vite dev server
- Clear browser cache
- Check file watcher limits (Linux):
  ```bash
  echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
  sudo sysctl -p
  ```

### Performance Optimization

#### Large Graphs (1000+ nodes)

1. **Enable virtualization:**
   ```typescript
   <ReactFlow
     onlyRenderVisibleElements={true}
     minZoom={0.1}
     maxZoom={2}
   />
   ```

2. **Simplify nodes:**
   - Hide details when zoomed out
   - Use simpler node components
   - Implement collapse/expand for sub-graphs

3. **Lazy loading:**
   ```typescript
   const loadGraph = async () => {
     // Load in chunks
     const chunk1 = await loadNodesChunk(0, 100);
     setNodes(chunk1);
     
     const chunk2 = await loadNodesChunk(100, 200);
     setNodes(prev => [...prev, ...chunk2]);
   };
   ```

#### Memory Management

Monitor memory usage:

```typescript
useEffect(() => {
  const interval = setInterval(() => {
    if (performance.memory) {
      console.log('Memory:', performance.memory.usedJSHeapSize / 1048576, 'MB');
    }
  }, 5000);
  
  return () => clearInterval(interval);
}, []);
```

## Best Practices

### 1. Workspace Organization

```
my-n3-project/
├── apps/
│   ├── web-app.ai
│   └── mobile-app.ai
├── shared/
│   ├── models.ai
│   └── tools.ai
└── tests/
    └── test-app.ai
```

### 2. File Naming

- Use descriptive names: `customer-support-agent.ai`
- Group related components: `rag-*`, `agent-*`
- Avoid spaces: Use hyphens or underscores

### 3. Graph Organization

- Keep chains focused (5-10 steps max)
- Group related entities in same file
- Use meaningful names for all entities

### 4. Version Control

Add to `.gitignore`:

```
node_modules/
dist/
.env
*.log
dbDir/
```

Commit:
- `.ai` source files
- Configuration files
- Documentation

Don't commit:
- Generated code
- Environment variables
- Build artifacts

## Resources

- [React Flow Documentation](https://reactflow.dev/)
- [Yjs Documentation](https://docs.yjs.dev/)
- [Zustand Guide](https://docs.pmnd.rs/zustand/)
- [Namel3ss Language Guide](../../README.md)

## Support

For help:

1. Check this integration guide
2. Review [FAQ](./FAQ.md)
3. Search [GitHub Issues](https://github.com/SsebowaDisan/namel3ss-programming-language/issues)
4. Open a new issue

---

**Happy Building! 🚀**
