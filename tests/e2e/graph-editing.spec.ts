import { test, expect, Page } from '@playwright/test';

/**
 * E2E Tests for Graph Editing
 * Tests node creation, deletion, edge connections, and graph manipulation
 */

test.describe('Graph Editing', () => {
  test.beforeEach(async ({ page }) => {
    // Login and create/open a project
    await page.goto('/');
    await loginUser(page, 'testuser', 'TestPass123!');
    await page.click('text=/new project|create/i');
    await page.fill('input[name="name"]', 'Graph Test Project');
    await page.click('button:has-text("Create")');
    
    // Wait for graph editor to load
    await expect(page.locator('.react-flow')).toBeVisible();
  });

  test('should create a new node', async ({ page }) => {
    // Right-click to open context menu or use add node button
    const canvas = page.locator('.react-flow__pane');
    await canvas.click({ button: 'right', position: { x: 200, y: 200 } });
    
    // Select LLM node type
    await page.click('text=/LLM|AI|Model/i');
    
    // Should see new node
    await expect(page.locator('.react-flow__node')).toHaveCount(1);
  });

  test('should delete a node', async ({ page }) => {
    // Create a node first
    await createNode(page, 'LLM', { x: 200, y: 200 });
    
    // Select and delete
    await page.click('.react-flow__node');
    await page.keyboard.press('Delete');
    
    // Should be gone
    await expect(page.locator('.react-flow__node')).toHaveCount(0);
  });

  test('should create edge between nodes', async ({ page }) => {
    // Create two nodes
    await createNode(page, 'Input', { x: 100, y: 200 });
    await createNode(page, 'LLM', { x: 400, y: 200 });
    
    // Connect them
    const handle1 = page.locator('.react-flow__node').first().locator('.react-flow__handle-right');
    const handle2 = page.locator('.react-flow__node').nth(1).locator('.react-flow__handle-left');
    
    await handle1.hover();
    await page.mouse.down();
    await handle2.hover();
    await page.mouse.up();
    
    // Should see edge
    await expect(page.locator('.react-flow__edge')).toHaveCount(1);
  });

  test('should configure node properties', async ({ page }) => {
    // Create node
    await createNode(page, 'LLM', { x: 200, y: 200 });
    
    // Click to open properties panel
    await page.click('.react-flow__node');
    
    // Should see properties panel
    await expect(page.locator('text=/properties|settings/i')).toBeVisible();
    
    // Configure model
    await page.selectOption('select[name="provider"]', 'openai');
    await page.fill('input[name="model"]', 'gpt-4');
    await page.fill('textarea[name="prompt"]', 'You are a helpful assistant.');
    
    // Save
    await page.click('button:has-text("Save")');
    
    // Should persist
    await page.click('.react-flow__pane');
    await page.click('.react-flow__node');
    await expect(page.locator('input[name="model"]')).toHaveValue('gpt-4');
  });

  test('should move node by dragging', async ({ page }) => {
    // Create node
    await createNode(page, 'LLM', { x: 200, y: 200 });
    
    const node = page.locator('.react-flow__node');
    const box = await node.boundingBox();
    
    // Drag to new position
    await node.hover();
    await page.mouse.down();
    await page.mouse.move(box!.x + 200, box!.y + 100);
    await page.mouse.up();
    
    // Verify position changed
    const newBox = await node.boundingBox();
    expect(newBox!.x).toBeGreaterThan(box!.x + 150);
  });

  test('should duplicate node', async ({ page }) => {
    // Create node
    await createNode(page, 'LLM', { x: 200, y: 200 });
    
    // Select node
    await page.click('.react-flow__node');
    
    // Duplicate (Ctrl+D or Cmd+D)
    await page.keyboard.press('Meta+D');
    
    // Should have 2 nodes
    await expect(page.locator('.react-flow__node')).toHaveCount(2);
  });

  test('should undo and redo actions', async ({ page }) => {
    // Create node
    await createNode(page, 'LLM', { x: 200, y: 200 });
    await expect(page.locator('.react-flow__node')).toHaveCount(1);
    
    // Undo
    await page.keyboard.press('Meta+Z');
    await expect(page.locator('.react-flow__node')).toHaveCount(0);
    
    // Redo
    await page.keyboard.press('Meta+Shift+Z');
    await expect(page.locator('.react-flow__node')).toHaveCount(1);
  });

  test('should create complex graph flow', async ({ page }) => {
    // Create a multi-node pipeline
    await createNode(page, 'Input', { x: 100, y: 200 });
    await createNode(page, 'RAG', { x: 300, y: 200 });
    await createNode(page, 'LLM', { x: 500, y: 200 });
    await createNode(page, 'Output', { x: 700, y: 200 });
    
    // Connect in sequence
    await connectNodes(page, 0, 1);
    await connectNodes(page, 1, 2);
    await connectNodes(page, 2, 3);
    
    // Should have 4 nodes and 3 edges
    await expect(page.locator('.react-flow__node')).toHaveCount(4);
    await expect(page.locator('.react-flow__edge')).toHaveCount(3);
  });

  test('should save and load graph', async ({ page }) => {
    // Create nodes
    await createNode(page, 'LLM', { x: 200, y: 200 });
    await createNode(page, 'Output', { x: 400, y: 200 });
    
    // Save (auto-save or manual)
    await page.click('button:has-text("Save")');
    await expect(page.locator('text=/saved/i')).toBeVisible();
    
    // Navigate away and back
    await page.goto('/dashboard');
    await page.click('text=/Graph Test Project/i');
    
    // Should load saved graph
    await expect(page.locator('.react-flow__node')).toHaveCount(2);
  });

  test('should export graph as JSON', async ({ page }) => {
    // Create simple graph
    await createNode(page, 'LLM', { x: 200, y: 200 });
    
    // Export
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.click('button:has-text("Export")'),
    ]);
    
    // Verify download
    expect(download.suggestedFilename()).toContain('.json');
  });

  test('should import graph from JSON', async ({ page }) => {
    // Prepare test graph JSON
    const graphJSON = {
      nodes: [
        { id: '1', type: 'llm', position: { x: 100, y: 100 }, data: {} },
        { id: '2', type: 'output', position: { x: 300, y: 100 }, data: {} },
      ],
      edges: [
        { id: 'e1-2', source: '1', target: '2' },
      ],
    };
    
    // Upload file
    await page.click('button:has-text("Import")');
    await page.setInputFiles('input[type="file"]', {
      name: 'graph.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(graphJSON)),
    });
    
    // Should load nodes
    await expect(page.locator('.react-flow__node')).toHaveCount(2);
    await expect(page.locator('.react-flow__edge')).toHaveCount(1);
  });
});

// Helper functions
async function loginUser(page: Page, username: string, password: string) {
  await page.fill('input[name="username"]', username);
  await page.fill('input[name="password"]', password);
  await page.click('button[type="submit"]');
  await page.waitForURL(/\/(dashboard|projects)/, { timeout: 5000 });
}

async function createNode(page: Page, type: string, position: { x: number; y: number }) {
  const canvas = page.locator('.react-flow__pane');
  await canvas.click({ button: 'right', position });
  await page.click(`text=/^${type}$/i`);
  await page.waitForTimeout(500); // Wait for node to be created
}

async function connectNodes(page: Page, sourceIndex: number, targetIndex: number) {
  const source = page.locator('.react-flow__node').nth(sourceIndex);
  const target = page.locator('.react-flow__node').nth(targetIndex);
  
  const sourceHandle = source.locator('.react-flow__handle-right');
  const targetHandle = target.locator('.react-flow__handle-left');
  
  await sourceHandle.hover();
  await page.mouse.down();
  await targetHandle.hover();
  await page.mouse.up();
  await page.waitForTimeout(300);
}
