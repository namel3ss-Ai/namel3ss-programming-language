#!/usr/bin/env node

/**
 * Setup script for N3 Graph Editor
 * 
 * Installs dependencies for both frontend and backend packages
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const ROOT = path.join(__dirname, '..');
const EDITOR = path.join(ROOT, 'packages', 'n3-graph-editor');
const SERVER = path.join(ROOT, 'packages', 'n3-graph-server');

console.log('🚀 Setting up N3 Graph Editor...\n');

// Check if directories exist
if (!fs.existsSync(EDITOR)) {
  console.error('❌ Error: n3-graph-editor directory not found');
  process.exit(1);
}

if (!fs.existsSync(SERVER)) {
  console.error('❌ Error: n3-graph-server directory not found');
  process.exit(1);
}

// Install backend dependencies
console.log('📦 Installing backend dependencies...');
try {
  execSync('npm install', { cwd: SERVER, stdio: 'inherit' });
  console.log('✅ Backend dependencies installed\n');
} catch (error) {
  console.error('❌ Failed to install backend dependencies');
  process.exit(1);
}

// Install frontend dependencies
console.log('📦 Installing frontend dependencies...');
try {
  execSync('npm install', { cwd: EDITOR, stdio: 'inherit' });
  console.log('✅ Frontend dependencies installed\n');
} catch (error) {
  console.error('❌ Failed to install frontend dependencies');
  process.exit(1);
}

// Create .env files if they don't exist
const editorEnv = path.join(EDITOR, '.env');
if (!fs.existsSync(editorEnv)) {
  console.log('📝 Creating frontend .env file...');
  fs.writeFileSync(
    editorEnv,
    `VITE_WS_URL=ws://localhost:3001
VITE_API_URL=http://localhost:3001
`
  );
  console.log('✅ Frontend .env created\n');
}

const serverEnv = path.join(SERVER, '.env');
if (!fs.existsSync(serverEnv)) {
  console.log('📝 Creating backend .env file...');
  fs.writeFileSync(
    serverEnv,
    `PORT=3001
WORKSPACE_PATH=${ROOT}
`
  );
  console.log('✅ Backend .env created\n');
}

console.log('✨ Setup complete!\n');
console.log('To start the editor:');
console.log('  1. Terminal 1: cd packages/n3-graph-server && npm run dev');
console.log('  2. Terminal 2: cd packages/n3-graph-editor && npm run dev');
console.log('  3. Open http://localhost:3000 in your browser\n');
