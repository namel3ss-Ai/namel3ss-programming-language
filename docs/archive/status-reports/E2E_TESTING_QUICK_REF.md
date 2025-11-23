# E2E Testing Quick Reference

Fast reference for running Playwright E2E tests in the N3 Graph Editor project.

## 🚀 Quick Start

```bash
# Install dependencies
npm install
npx playwright install

# Start services
docker-compose up -d

# Run all tests
npm test

# View results
npm run test:report
```

## 📝 Common Commands

| Command | Description |
|---------|-------------|
| `npm test` | Run all E2E tests |
| `npm run test:ui` | Interactive UI mode |
| `npm run test:headed` | See browser while testing |
| `npm run test:debug` | Debug mode with inspector |
| `npm run test:auth` | Only authentication tests |
| `npm run test:graph` | Only graph editing tests |
| `npm run test:collab` | Only collaboration tests |
| `npm run test:exec` | Only execution tests |
| `npm run test:rlhf` | Only RLHF training tests |
| `npm run test:report` | View HTML report |

## 🧪 Test Suites

### Authentication (16 tests)
```bash
npm run test:auth
```
- Registration, login, logout
- Password change, profile update
- Project permissions (Owner/Editor/Viewer)
- Role-based access control

### Graph Editing (12 tests)
```bash
npm run test:graph
```
- Create/delete nodes and edges
- Configure node properties
- Drag and drop, undo/redo
- Save/load, import/export

### Collaboration (10 tests)
```bash
npm run test:collab
```
- Multi-user editing
- Real-time sync via Yjs
- Cursor tracking
- Conflict resolution

### Execution (12 tests)
```bash
npm run test:exec
```
- Run graphs with LLMs
- View execution traces
- Handle errors and cancellation
- Tool adapters and providers

### RLHF Training (12 tests)
```bash
npm run test:rlhf
```
- Submit feedback
- Train policy models
- View metrics and history
- Deploy trained models

**Total: 67 comprehensive E2E tests**

## 🎯 Specific Test Cases

```bash
# Run single test file
npx playwright test tests/e2e/auth.spec.ts

# Run tests matching pattern
npx playwright test --grep "login"

# Run on specific browser
npx playwright test --project=chromium

# Run with more workers (parallel)
npx playwright test --workers=4
```

## 🔍 Debugging

```bash
# Debug mode (pause and inspect)
npm run test:debug

# Headed mode (see browser)
npm run test:headed

# Record test (generate code)
npm run codegen

# View trace file
npx playwright show-trace trace.zip

# Slow down actions
SLOW_MO=1000 npm test
```

## 📊 Test Results

```bash
# HTML report
npm run test:report

# JSON results
cat test-results/results.json

# JUnit XML (for CI)
npx playwright test --reporter=junit
```

## 🛠️ Test Helpers

```typescript
import {
  loginUser,          // Login existing user
  registerUser,       // Register new user
  createNode,         // Create graph node
  connectNodes,       // Connect two nodes
  runGraph,           // Execute graph
  submitFeedback,     // Submit RLHF feedback
  assertNodeCount,    // Assert node count
  waitForSync,        // Wait for Yjs sync
} from './helpers';
```

### Example Usage

```typescript
test('my test', async ({ page }) => {
  // Login
  await loginUser(page, 'testuser', 'password');
  
  // Create graph
  await createNode(page, 'LLM', { x: 200, y: 200 });
  await createNode(page, 'Output', { x: 400, y: 200 });
  await connectNodes(page, 0, 1);
  
  // Run and verify
  await runGraph(page);
  await assertNodeCount(page, 2);
});
```

## 🐳 Service Management

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs backend
docker-compose logs yjs-server

# Restart service
docker-compose restart backend

# Stop all
docker-compose down
```

## 🔧 Configuration

### Playwright Config (`playwright.config.ts`)

```typescript
{
  testDir: './tests/e2e',
  timeout: 60000,
  retries: 2,
  workers: process.env.CI ? 1 : undefined,
  baseURL: 'http://localhost:3000',
}
```

### Environment Variables

```bash
# Base URL
PLAYWRIGHT_BASE_URL=http://localhost:3000

# Headless mode
HEADED=1 npm test

# Debug mode
PWDEBUG=1 npm test

# Slow motion
SLOW_MO=500 npm test
```

## 📦 Dependencies

```json
{
  "devDependencies": {
    "@playwright/test": "^1.40.1",
    "@types/node": "^20.10.5",
    "typescript": "^5.3.3"
  }
}
```

## 🚨 Troubleshooting

### Tests Failing

```bash
# Reinstall browsers
npx playwright install --with-deps

# Clear cache
rm -rf playwright-report test-results

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

### Port Conflicts

```bash
# Check ports
lsof -i :3000  # Frontend
lsof -i :8000  # Backend
lsof -i :1234  # YJS Server

# Kill process
kill -9 <PID>
```

### Flaky Tests

```bash
# Run with retries
npx playwright test --retries=3

# Run specific test multiple times
npx playwright test --repeat-each=10
```

## 📱 Browser Support

| Browser | Desktop | Mobile |
|---------|---------|--------|
| Chromium | ✅ | ✅ (Pixel 5) |
| Firefox | ✅ | ❌ |
| WebKit | ✅ | ✅ (iPhone 12) |

```bash
# Run on all browsers
npm test

# Chromium only
npm run test:chromium

# Mobile browsers
npm run test:mobile
```

## 🎬 Recording Tests

```bash
# Generate test code
npm run codegen

# Record against localhost
npx playwright codegen http://localhost:3000

# With authentication
npx playwright codegen --load-storage=auth.json http://localhost:3000
```

## 📈 CI/CD Integration

### GitHub Actions

```yaml
- name: Install Playwright
  run: npx playwright install --with-deps

- name: Run tests
  run: npm test

- name: Upload report
  uses: actions/upload-artifact@v3
  with:
    name: playwright-report
    path: playwright-report/
```

### Test Timing

- Authentication: ~30s
- Graph Editing: ~45s
- Collaboration: ~60s (multi-browser)
- Execution: ~90s (LLM calls)
- RLHF Training: ~120s (model training)

**Total Suite**: ~6 minutes (parallel) / ~20 minutes (sequential)

## 🔗 Resources

- [Playwright Docs](https://playwright.dev)
- [Full Test Documentation](./E2E_TESTING.md)
- [Project README](./README.md)

---

**Quick Tip**: Use `npm run test:ui` for interactive debugging and test development!
