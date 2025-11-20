#!/usr/bin/env node

/**
 * Development launcher
 * 
 * Starts both frontend and backend servers concurrently
 */

const { spawn } = require('child_process');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const EDITOR = path.join(ROOT, 'packages', 'n3-graph-editor');
const SERVER = path.join(ROOT, 'packages', 'n3-graph-server');

console.log('🚀 Starting N3 Graph Editor development servers...\n');

// Start backend server
console.log('📡 Starting backend server...');
const backend = spawn('npm', ['run', 'dev'], {
  cwd: SERVER,
  stdio: 'inherit',
  shell: true,
});

// Wait a bit for backend to start
setTimeout(() => {
  // Start frontend server
  console.log('🎨 Starting frontend server...');
  const frontend = spawn('npm', ['run', 'dev'], {
    cwd: EDITOR,
    stdio: 'inherit',
    shell: true,
  });

  frontend.on('error', (error) => {
    console.error('❌ Frontend error:', error);
  });
}, 2000);

backend.on('error', (error) => {
  console.error('❌ Backend error:', error);
});

// Handle shutdown
process.on('SIGINT', () => {
  console.log('\n\n👋 Shutting down servers...');
  backend.kill('SIGINT');
  process.exit(0);
});
