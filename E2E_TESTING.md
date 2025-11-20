# End-to-End Test Suite with Playwright

Complete E2E test suite for the N3 Graph Editor visual programming environment, covering authentication, graph editing, real-time collaboration, graph execution, tool adapters, and RLHF training.

## 📋 Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Setup](#setup)
- [Running Tests](#running-tests)
- [Test Suites](#test-suites)
- [Writing Tests](#writing-tests)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## Overview

This E2E test suite uses **Playwright** to validate the complete user journey through the N3 Graph Editor application stack:

- **Frontend**: React + TypeScript + Vite + React Flow
- **Backend**: FastAPI + SQLAlchemy + PostgreSQL
- **YJS Server**: Real-time collaboration via WebSockets
- **Services**: Authentication, Graph Execution, RLHF Training

### Test Coverage

- ✅ **Authentication** (16 tests): Registration, login, permissions, profile management
- ✅ **Graph Editing** (12 tests): Node creation/deletion, edge connections, properties
- ✅ **Collaboration** (10 tests): Multi-user editing, cursor sync, conflict resolution
- ✅ **Execution** (12 tests): Running graphs, viewing traces, error handling
- ✅ **Tool Adapters** (5 tests): OpenAPI import, LLM providers, RAG nodes
- ✅ **RLHF Training** (12 tests): Feedback submission, policy training, model deployment

**Total**: 67 comprehensive E2E tests

## Test Structure

```
tests/e2e/
├── auth.spec.ts              # Authentication & authorization tests
├── graph-editing.spec.ts     # Graph manipulation tests
├── collaboration.spec.ts     # Real-time collaboration tests
├── execution.spec.ts         # Graph execution & tool adapter tests
├── rlhf.spec.ts             # RLHF training & feedback tests
└── helpers.ts               # Shared test utilities

playwright.config.ts          # Playwright configuration
package.json                 # E2E test dependencies
```

## Setup

### Prerequisites

- **Node.js** 18+ and **npm**
- **Docker** and **Docker Compose**
- **Python** 3.11+ with virtual environment
- **PostgreSQL** (via Docker)

### Installation

```bash
# Install Playwright and dependencies
npm install

# Install Playwright browsers
npx playwright install

# Or use the npm script
npm run install:browsers

# Install browser system dependencies (if needed)
npx playwright install-deps
```

### Environment Setup

Ensure services are running:

```bash
# Start backend, database, and YJS server
docker-compose up -d postgres backend yjs-server

# Or start all services
docker-compose up -d

# Verify services are healthy
docker-compose ps
```

Expected output:
```
NAME                    STATUS
postgres                Up (healthy)
backend                 Up
yjs-server              Up
frontend                Up (if included)
```

## Running Tests

### Basic Commands

```bash
# Run all tests
npm test

# Run tests with UI mode (interactive)
npm run test:ui

# Run tests in headed mode (see browser)
npm run test:headed

# Run specific test file
npx playwright test tests/e2e/auth.spec.ts

# Run tests in debug mode
npm run test:debug
```

### Browser-Specific Tests

```bash
# Chromium only
npm run test:chromium

# Firefox only
npm run test:firefox

# WebKit (Safari) only
npm run test:webkit

# Mobile browsers
npm run test:mobile
```

### Test Suite Commands

```bash
# Run only authentication tests
npm run test:auth

# Run only graph editing tests
npm run test:graph

# Run only collaboration tests
npm run test:collab

# Run only execution tests
npm run test:exec

# Run only RLHF training tests
npm run test:rlhf
```

### Advanced Options

```bash
# Run tests in parallel (faster)
npx playwright test --workers=4

# Run tests matching pattern
npx playwright test --grep "login"

# Run tests excluding pattern
npx playwright test --grep-invert "slow"

# Generate HTML report
npm run test:report

# Update snapshots
npx playwright test --update-snapshots
```

## Test Suites

### 1. Authentication Tests (`auth.spec.ts`)

Tests user authentication and authorization flows.

**Test Cases**:
- Display login page
- Register new user with validation
- Login with username/email
- Wrong password error handling
- Protected route redirection
- Logout functionality
- Update user profile
- Change password
- Create project as owner
- Add/remove collaborators
- Role-based permission checks (Owner/Editor/Viewer)

**Example**:
```typescript
test('should register a new user', async ({ page }) => {
  await page.goto('/');
  await page.click('text=/sign up|register/i');
  
  await page.fill('input[name="email"]', 'test@example.com');
  await page.fill('input[name="username"]', 'testuser');
  await page.fill('input[name="password"]', 'TestPass123!');
  await page.fill('input[name="fullName"]', 'Test User');
  
  await page.click('button[type="submit"]');
  
  await expect(page).toHaveURL(/\/(dashboard|projects)/);
});
```

### 2. Graph Editing Tests (`graph-editing.spec.ts`)

Tests graph manipulation and node operations.

**Test Cases**:
- Create new node via context menu
- Delete node with keyboard shortcut
- Create edge between nodes
- Configure node properties
- Move node by dragging
- Duplicate node (Cmd+D)
- Undo/redo actions (Cmd+Z, Cmd+Shift+Z)
- Create complex multi-node flow
- Save and load graph
- Export graph as JSON
- Import graph from JSON

**Example**:
```typescript
test('should create edge between nodes', async ({ page }) => {
  await createNode(page, 'Input', { x: 100, y: 200 });
  await createNode(page, 'LLM', { x: 400, y: 200 });
  
  const handle1 = page.locator('.react-flow__node').first()
    .locator('.react-flow__handle-right');
  const handle2 = page.locator('.react-flow__node').nth(1)
    .locator('.react-flow__handle-left');
  
  await handle1.hover();
  await page.mouse.down();
  await handle2.hover();
  await page.mouse.up();
  
  await expect(page.locator('.react-flow__edge')).toHaveCount(1);
});
```

### 3. Collaboration Tests (`collaboration.spec.ts`)

Tests real-time multi-user collaboration via Yjs.

**Test Cases**:
- Sync node creation between users
- Sync node deletion between users
- Sync node position updates
- Sync edge creation between users
- Display user cursors
- Show online users list
- Handle concurrent edits gracefully
- Sync node property changes
- Maintain document integrity with undo/redo
- Handle user disconnect and reconnect

**Example**:
```typescript
test('should sync node creation between users', async () => {
  // User 1 creates a node
  await createNode(page1, 'LLM', { x: 200, y: 200 });
  
  // User 2 should see it
  await page2.waitForTimeout(1000); // Wait for sync
  await expect(page2.locator('.react-flow__node')).toHaveCount(1);
});
```

### 4. Execution Tests (`execution.spec.ts`)

Tests graph execution engine and tool adapters.

**Test Cases**:
- Execute simple LLM graph
- Display execution trace
- Show node-level execution results
- Handle execution errors gracefully
- Cancel running execution
- Execute graph with multiple LLM calls
- Import OpenAPI spec
- Configure LLM provider
- Use different LLM providers (OpenAI, Anthropic, Vertex)
- Use RAG node with embeddings
- Chain tool adapters

**Example**:
```typescript
test('should execute simple LLM graph', async ({ page }) => {
  await createNode(page, 'LLM', { x: 200, y: 200 });
  
  await page.click('.react-flow__node');
  await page.selectOption('select[name="provider"]', 'openai');
  await page.fill('input[name="model"]', 'gpt-3.5-turbo');
  await page.fill('textarea[name="prompt"]', 'Say hello!');
  await page.click('button:has-text("Save")');
  
  await page.click('button:has-text("Run")');
  
  await expect(page.locator('text=/completed|success/i'))
    .toBeVisible({ timeout: 30000 });
});
```

### 5. RLHF Training Tests (`rlhf.spec.ts`)

Tests reinforcement learning from human feedback.

**Test Cases**:
- Submit positive/negative feedback
- Provide feedback with comments
- View feedback history
- Train policy from feedback
- Display training progress
- View training metrics
- Compare model versions
- Deploy trained model
- Export feedback dataset
- Configure reward model
- Monitor training in real-time

**Example**:
```typescript
test('should submit positive feedback', async ({ page }) => {
  // Run graph first
  await page.click('button:has-text("Run")');
  await page.waitForSelector('text=/completed/i', { timeout: 30000 });
  
  // Click on node to see output
  await page.click('.react-flow__node');
  
  // Submit positive feedback
  await page.click('button[data-feedback="positive"]');
  
  await expect(page.locator('text=/feedback submitted/i')).toBeVisible();
});
```

## Writing Tests

### Test Helpers

Use shared helpers from `helpers.ts`:

```typescript
import {
  loginUser,
  createNode,
  connectNodes,
  runGraph,
  submitFeedback,
  assertNodeCount,
} from './helpers';

test('my test', async ({ page }) => {
  await loginUser(page, 'testuser', 'password');
  await createNode(page, 'LLM', { x: 200, y: 200 });
  await assertNodeCount(page, 1);
});
```

### Best Practices

1. **Use data attributes**: Target elements with `data-*` attributes for stability
   ```typescript
   await page.click('[data-testid="run-button"]');
   ```

2. **Wait for conditions**: Use `waitFor*` methods instead of arbitrary timeouts
   ```typescript
   await page.waitForSelector('text=/completed/i');
   await page.waitForURL(/dashboard/);
   ```

3. **Handle async operations**: Wait for network requests or state changes
   ```typescript
   await Promise.all([
     page.waitForResponse(resp => resp.url().includes('/api/graphs')),
     page.click('button:has-text("Save")'),
   ]);
   ```

4. **Clean up test data**: Use `beforeEach` and `afterEach` hooks
   ```typescript
   test.afterEach(async ({ page }) => {
     await cleanupTestData(page);
   });
   ```

5. **Use fixtures**: Isolate test setup
   ```typescript
   test.use({ storageState: 'auth.json' });
   ```

### Debugging Tests

```bash
# Run with --debug flag
npx playwright test --debug

# Use Playwright Inspector
PWDEBUG=1 npx playwright test

# Record test
npx playwright codegen http://localhost:3000

# View trace
npx playwright show-trace trace.zip
```

## CI/CD Integration

### GitHub Actions

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: 18
      
      - name: Install dependencies
        run: npm ci
      
      - name: Install Playwright browsers
        run: npx playwright install --with-deps
      
      - name: Start services
        run: docker-compose up -d
      
      - name: Wait for services
        run: npx wait-on http://localhost:8000/health http://localhost:3000
      
      - name: Run E2E tests
        run: npm test
      
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: playwright-report
          path: playwright-report/
```

### Docker Compose Test Environment

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  postgres-test:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: n3_test
      POSTGRES_USER: test
      POSTGRES_PASSWORD: testpass
    
  backend-test:
    build: .
    environment:
      DATABASE_URL: postgresql+asyncpg://test:testpass@postgres-test:5432/n3_test
      TESTING: "true"
    depends_on:
      - postgres-test
```

## Troubleshooting

### Common Issues

**1. Services not starting**
```bash
# Check service health
docker-compose ps

# View logs
docker-compose logs backend
docker-compose logs yjs-server

# Restart services
docker-compose restart
```

**2. Tests timing out**
```bash
# Increase timeout in playwright.config.ts
timeout: 120 * 1000,  // 2 minutes

# Or per test
test('slow test', async ({ page }) => {
  test.setTimeout(120000);
  // ...
});
```

**3. Flaky tests**
```bash
# Run with retries
npx playwright test --retries=3

# Or configure in playwright.config.ts
retries: process.env.CI ? 2 : 0,
```

**4. Browser not found**
```bash
# Reinstall browsers
npx playwright install

# Install system dependencies
npx playwright install-deps
```

**5. Port conflicts**
```bash
# Check ports
lsof -i :3000
lsof -i :8000
lsof -i :1234

# Change ports in docker-compose.yml
ports:
  - "3001:3000"  # Use different host port
```

### Debug Mode

```typescript
// Add console logs
await page.evaluate(() => console.log('Debug point'));

// Take screenshot
await page.screenshot({ path: 'debug.png' });

// Pause execution
await page.pause();

// Slow down actions
await page.click('button', { delay: 1000 });
```

### Test Isolation

Ensure tests are independent:

```typescript
test.beforeEach(async ({ page, context }) => {
  // Clear cookies and storage
  await context.clearCookies();
  await page.evaluate(() => localStorage.clear());
  
  // Reset database state
  await resetTestDatabase();
});
```

## Performance Tips

- **Run tests in parallel**: Use `--workers=4` flag
- **Skip unnecessary tests**: Use `test.skip()` for known issues
- **Use fast assertions**: Prefer `toHaveCount()` over multiple `toBe()` checks
- **Reuse authentication**: Save auth state with `storageState`
- **Mock external APIs**: Use Playwright's route mocking

```typescript
// Mock external API
await page.route('**/api.openai.com/**', route => {
  route.fulfill({
    status: 200,
    body: JSON.stringify({ choices: [{ text: 'Mocked response' }] })
  });
});
```

## Resources

- [Playwright Documentation](https://playwright.dev)
- [Playwright Best Practices](https://playwright.dev/docs/best-practices)
- [React Flow Testing](https://reactflow.dev/learn/advanced-use/testing)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)

---

**Test Suite Status**: ✅ **67 E2E Tests** covering all user workflows

For issues or questions, see the [main README](../README.md) or file an issue.
