/**
 * N3 Graph Server
 * 
 * Express server that:
 * 1. Parses .n3 files and serves graph data
 * 2. Manages WebSocket connections for real-time collaboration via Yjs
 * 3. Watches for file changes and notifies clients
 */

import express from 'express';
import cors from 'cors';
import { WebSocketServer } from 'ws';
import { setupWSConnection } from 'y-websocket/bin/utils';
import path from 'path';
import fs from 'fs';
import chokidar from 'chokidar';
import { parseN3File, parseN3Source } from './parser.js';
import { transformToGraph } from './graph-transformer.js';

const app = express();
const PORT = process.env.PORT || 3001;
const WORKSPACE_PATH = process.env.WORKSPACE_PATH || process.cwd();

// Middleware
app.use(cors());
app.use(express.json());

// In-memory cache for parsed graphs
const graphCache = new Map<string, any>();

/**
 * GET /api/parse
 * Parse a .n3 file and return the graph structure
 */
app.get('/api/parse', async (req, res) => {
  const { file } = req.query;
  
  if (!file || typeof file !== 'string') {
    return res.status(400).json({ error: 'Missing or invalid file parameter' });
  }
  
  const filePath = path.isAbsolute(file) ? file : path.join(WORKSPACE_PATH, file);
  
  // Check if file exists
  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: 'File not found' });
  }
  
  // Check cache
  const cacheKey = filePath;
  if (graphCache.has(cacheKey)) {
    return res.json(graphCache.get(cacheKey));
  }
  
  try {
    // Parse the N3 file
    const parseResult = await parseN3File(filePath);
    
    if (!parseResult.success || !parseResult.module) {
      return res.status(500).json({ error: parseResult.error || 'Parse failed' });
    }
    
    // Transform to graph
    const graph = transformToGraph(parseResult.module);
    
    // Cache result
    graphCache.set(cacheKey, { graph, module: parseResult.module });
    
    res.json({ graph, module: parseResult.module });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/parse
 * Parse N3 source code and return the graph structure
 */
app.post('/api/parse', async (req, res) => {
  const { source, fileName = 'source.n3' } = req.body;
  
  if (!source) {
    return res.status(400).json({ error: 'Missing source parameter' });
  }
  
  try {
    const parseResult = await parseN3Source(source, fileName);
    
    if (!parseResult.success || !parseResult.module) {
      return res.status(500).json({ error: parseResult.error || 'Parse failed' });
    }
    
    const graph = transformToGraph(parseResult.module);
    
    res.json({ graph, module: parseResult.module });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/files
 * List all .n3 files in the workspace
 */
app.get('/api/files', (req, res) => {
  try {
    const files = findN3Files(WORKSPACE_PATH);
    res.json({ files });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/save
 * Save graph changes back to .n3 file
 */
app.post('/api/save', async (req, res) => {
  const { file, graph } = req.body;
  
  if (!file || !graph) {
    return res.status(400).json({ error: 'Missing file or graph parameter' });
  }
  
  const filePath = path.isAbsolute(file) ? file : path.join(WORKSPACE_PATH, file);
  
  try {
    // TODO: Implement graph-to-N3 conversion
    // For now, return success but don't actually write
    res.json({ success: true, message: 'Graph-to-N3 conversion not yet implemented' });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * Health check endpoint
 */
app.get('/health', (req, res) => {
  res.json({ status: 'ok', workspace: WORKSPACE_PATH });
});

// Start HTTP server
const server = app.listen(PORT, () => {
  console.log(`🚀 N3 Graph Server running on http://localhost:${PORT}`);
  console.log(`📁 Workspace: ${WORKSPACE_PATH}`);
});

// Setup WebSocket server for Yjs collaboration
const wss = new WebSocketServer({ server });

wss.on('connection', (ws: any, req: any) => {
  const docName = req.url?.slice(1) || 'default';
  console.log(`🔌 WebSocket connection established for document: ${docName}`);
  
  setupWSConnection(ws, req, { docName });
});

// Watch for file changes
const watcher = chokidar.watch('**/*.n3', {
  cwd: WORKSPACE_PATH,
  ignored: /(^|[\/\\])\../,
  persistent: true,
});

watcher.on('change', (filePath) => {
  console.log(`📝 File changed: ${filePath}`);
  
  // Clear cache for changed file
  const fullPath = path.join(WORKSPACE_PATH, filePath);
  graphCache.delete(fullPath);
  
  // Notify connected WebSocket clients
  wss.clients.forEach((client) => {
    if (client.readyState === 1) { // WebSocket.OPEN
      client.send(JSON.stringify({
        type: 'file-changed',
        file: filePath,
      }));
    }
  });
});

/**
 * Recursively find all .n3 files in a directory
 */
function findN3Files(dir: string, fileList: string[] = []): string[] {
  const files = fs.readdirSync(dir);
  
  for (const file of files) {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    
    if (stat.isDirectory()) {
      // Skip node_modules, .git, etc.
      if (!file.startsWith('.') && file !== 'node_modules') {
        findN3Files(filePath, fileList);
      }
    } else if (file.endsWith('.n3')) {
      fileList.push(path.relative(WORKSPACE_PATH, filePath));
    }
  }
  
  return fileList;
}

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully...');
  server.close(() => {
    watcher.close();
    process.exit(0);
  });
});
