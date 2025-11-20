import http from 'http';
import { WebSocketServer } from 'ws';
import { setupWSConnection } from 'y-websocket/bin/utils.js';

const PORT = process.env.PORT || 1234;

const server = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('N3 Yjs WebSocket Server\n');
});

const wss = new WebSocketServer({ server });

wss.on('connection', (ws, req) => {
  setupWSConnection(ws, req);
  console.log(`Client connected: ${req.url}`);
});

server.listen(PORT, () => {
  console.log(`Yjs WebSocket server running on port ${PORT}`);
});
