import { test, expect, Page } from '@playwright/test';

/**
 * E2E Tests for Graph Execution and Tool Adapters
 * Tests running graphs, viewing traces, importing OpenAPI, and using LLM providers
 */

test.describe('Graph Execution', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await loginUser(page, 'testuser', 'TestPass123!');
    
    // Create new project
    await page.click('text=/new project|create/i');
    await page.fill('input[name="name"]', 'Execution Test');
    await page.click('button:has-text("Create")');
    
    await expect(page.locator('.react-flow')).toBeVisible();
  });

  test('should execute simple LLM graph', async ({ page }) => {
    // Create LLM node
    await createNode(page, 'LLM', { x: 200, y: 200 });
    
    // Configure it
    await page.click('.react-flow__node');
    await page.selectOption('select[name="provider"]', 'openai');
    await page.fill('input[name="model"]', 'gpt-3.5-turbo');
    await page.fill('textarea[name="prompt"]', 'Say hello!');
    await page.click('button:has-text("Save")');
    
    // Execute
    await page.click('button:has-text("Run")');
    
    // Should show execution status
    await expect(page.locator('text=/running|executing/i')).toBeVisible();
    
    // Wait for completion
    await expect(page.locator('text=/completed|success/i')).toBeVisible({ timeout: 30000 });
  });

  test('should display execution trace', async ({ page }) => {
    // Create and run graph
    await createNode(page, 'LLM', { x: 200, y: 200 });
    await page.click('.react-flow__node');
    await page.selectOption('select[name="provider"]', 'openai');
    await page.fill('textarea[name="prompt"]', 'Test prompt');
    await page.click('button:has-text("Save")');
    await page.click('button:has-text("Run")');
    
    // Wait for completion
    await page.waitForSelector('text=/completed/i', { timeout: 30000 });
    
    // Open trace viewer
    await page.click('button:has-text("View Trace")');
    
    // Should show trace details
    await expect(page.locator('text=/execution|trace|steps/i')).toBeVisible();
    await expect(page.locator('text=/duration|time/i')).toBeVisible();
  });

  test('should show node-level execution results', async ({ page }) => {
    // Create pipeline
    await createNode(page, 'Input', { x: 100, y: 200 });
    await createNode(page, 'LLM', { x: 300, y: 200 });
    await createNode(page, 'Output', { x: 500, y: 200 });
    await connectNodes(page, 0, 1);
    await connectNodes(page, 1, 2);
    
    // Configure
    await page.click('.react-flow__node').first();
    await page.fill('textarea[name="value"]', 'Hello world');
    await page.click('button:has-text("Save")');
    
    // Execute
    await page.click('button:has-text("Run")');
    await page.waitForSelector('text=/completed/i', { timeout: 30000 });
    
    // Click on LLM node to see its output
    const llmNode = page.locator('.react-flow__node').nth(1);
    await llmNode.click();
    
    // Should show execution result
    await expect(page.locator('text=/output|result/i')).toBeVisible();
  });

  test('should handle execution errors gracefully', async ({ page }) => {
    // Create LLM node with invalid config
    await createNode(page, 'LLM', { x: 200, y: 200 });
    await page.click('.react-flow__node');
    await page.selectOption('select[name="provider"]', 'invalid_provider');
    await page.click('button:has-text("Save")');
    
    // Execute
    await page.click('button:has-text("Run")');
    
    // Should show error
    await expect(page.locator('text=/error|failed/i')).toBeVisible({ timeout: 10000 });
  });

  test('should cancel running execution', async ({ page }) => {
    // Create long-running node
    await createNode(page, 'LLM', { x: 200, y: 200 });
    await page.click('.react-flow__node');
    await page.fill('textarea[name="prompt"]', 'Write a long story');
    await page.click('button:has-text("Save")');
    
    // Start execution
    await page.click('button:has-text("Run")');
    await expect(page.locator('text=/running/i')).toBeVisible();
    
    // Cancel
    await page.click('button:has-text("Cancel")');
    
    // Should show cancelled status
    await expect(page.locator('text=/cancelled|stopped/i')).toBeVisible();
  });

  test('should execute graph with multiple LLM calls', async ({ page }) => {
    // Create branching pipeline
    await createNode(page, 'Input', { x: 100, y: 200 });
    await createNode(page, 'LLM', { x: 300, y: 100 });
    await createNode(page, 'LLM', { x: 300, y: 300 });
    await createNode(page, 'Output', { x: 500, y: 200 });
    
    // Connect: Input -> LLM1, Input -> LLM2, LLM1 -> Output, LLM2 -> Output
    await connectNodes(page, 0, 1);
    await connectNodes(page, 0, 2);
    await connectNodes(page, 1, 3);
    await connectNodes(page, 2, 3);
    
    // Execute
    await page.click('button:has-text("Run")');
    
    // Should execute both LLMs
    await expect(page.locator('text=/completed/i')).toBeVisible({ timeout: 45000 });
  });
});

test.describe('Tool Adapters', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await loginUser(page, 'testuser', 'TestPass123!');
    await page.click('text=/new project|create/i');
    await page.fill('input[name="name"]', 'Tool Adapter Test');
    await page.click('button:has-text("Create")');
    await expect(page.locator('.react-flow')).toBeVisible();
  });

  test('should import OpenAPI spec', async ({ page }) => {
    // Click import OpenAPI
    await page.click('button:has-text("Import OpenAPI")');
    
    // Provide OpenAPI spec URL or JSON
    const openAPISpec = {
      openapi: '3.0.0',
      info: { title: 'Test API', version: '1.0.0' },
      paths: {
        '/test': {
          get: {
            summary: 'Test endpoint',
            responses: { '200': { description: 'Success' } }
          }
        }
      }
    };
    
    await page.fill('textarea[name="openapi"]', JSON.stringify(openAPISpec));
    await page.click('button:has-text("Import")');
    
    // Should create API node
    await expect(page.locator('.react-flow__node[data-type="api"]')).toBeVisible();
  });

  test('should configure LLM provider', async ({ page }) => {
    // Open settings
    await page.click('button:has-text("Settings")');
    await page.click('text=/providers|api keys/i');
    
    // Add OpenAI key
    await page.fill('input[name="openai_api_key"]', 'sk-test-key');
    await page.click('button:has-text("Save")');
    
    // Should show success
    await expect(page.locator('text=/saved|success/i')).toBeVisible();
  });

  test('should use different LLM providers', async ({ page }) => {
    // Create LLM nodes with different providers
    await createNode(page, 'LLM', { x: 100, y: 100 });
    await createNode(page, 'LLM', { x: 100, y: 250 });
    await createNode(page, 'LLM', { x: 100, y: 400 });
    
    // Configure OpenAI
    await page.locator('.react-flow__node').first().click();
    await page.selectOption('select[name="provider"]', 'openai');
    await page.fill('input[name="model"]', 'gpt-4');
    await page.click('button:has-text("Save")');
    
    // Configure Anthropic
    await page.locator('.react-flow__node').nth(1).click();
    await page.selectOption('select[name="provider"]', 'anthropic');
    await page.fill('input[name="model"]', 'claude-3-opus');
    await page.click('button:has-text("Save")');
    
    // Configure Vertex AI
    await page.locator('.react-flow__node').nth(2).click();
    await page.selectOption('select[name="provider"]', 'vertex');
    await page.fill('input[name="model"]', 'gemini-pro');
    await page.click('button:has-text("Save")');
    
    // All should be configured
    await expect(page.locator('.react-flow__node[data-provider="openai"]')).toBeVisible();
    await expect(page.locator('.react-flow__node[data-provider="anthropic"]')).toBeVisible();
    await expect(page.locator('.react-flow__node[data-provider="vertex"]')).toBeVisible();
  });

  test('should use RAG node with embeddings', async ({ page }) => {
    // Create RAG node
    await createNode(page, 'RAG', { x: 200, y: 200 });
    
    // Configure
    await page.click('.react-flow__node');
    await page.selectOption('select[name="embeddings"]', 'openai');
    await page.fill('input[name="collection"]', 'test_docs');
    await page.fill('textarea[name="query"]', 'What is AI?');
    await page.click('button:has-text("Save")');
    
    // Execute
    await page.click('button:has-text("Run")');
    
    // Should retrieve documents
    await expect(page.locator('text=/completed/i')).toBeVisible({ timeout: 30000 });
  });

  test('should chain tool adapters', async ({ page }) => {
    // Create tool chain: API -> LLM -> Output
    await createNode(page, 'API', { x: 100, y: 200 });
    await createNode(page, 'LLM', { x: 300, y: 200 });
    await createNode(page, 'Output', { x: 500, y: 200 });
    
    await connectNodes(page, 0, 1);
    await connectNodes(page, 1, 2);
    
    // Execute
    await page.click('button:has-text("Run")');
    
    // Should complete chain
    await expect(page.locator('text=/completed/i')).toBeVisible({ timeout: 30000 });
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
  await page.waitForTimeout(500);
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
