import { test, expect, Page, Browser } from '@playwright/test';

/**
 * E2E Tests for Real-time Collaboration
 * Tests multi-user editing, cursor sync, and Yjs integration
 */

test.describe('Real-time Collaboration', () => {
  let browser1: Browser;
  let browser2: Browser;
  let page1: Page;
  let page2: Page;

  test.beforeAll(async ({ browser }) => {
    // Create two browser contexts for two users
    const context1 = await browser.newContext();
    const context2 = await browser.newContext();
    
    page1 = await context1.newPage();
    page2 = await context2.newPage();
  });

  test.beforeEach(async () => {
    // Login both users
    await page1.goto('/');
    await loginUser(page1, 'testuser', 'TestPass123!');
    
    await page2.goto('/');
    await loginUser(page2, 'collabuser', 'CollabPass123!');
    
    // User 1 creates project and shares with user 2
    await page1.click('text=/new project|create/i');
    await page1.fill('input[name="name"]', 'Collab Project');
    await page1.click('button:has-text("Create")');
    
    // Share with user 2
    await page1.click('text=/share|invite/i');
    await page1.fill('input[name="email"]', 'collab@example.com');
    await page1.selectOption('select[name="role"]', 'EDITOR');
    await page1.click('button:has-text("Invite")');
    
    // User 2 opens the shared project
    await page2.goto('/dashboard');
    await page2.click('text=/Collab Project/i');
    
    // Wait for both to load the editor
    await expect(page1.locator('.react-flow')).toBeVisible();
    await expect(page2.locator('.react-flow')).toBeVisible();
  });

  test('should sync node creation between users', async () => {
    // User 1 creates a node
    await createNode(page1, 'LLM', { x: 200, y: 200 });
    
    // User 2 should see it
    await page2.waitForTimeout(1000); // Wait for sync
    await expect(page2.locator('.react-flow__node')).toHaveCount(1);
  });

  test('should sync node deletion between users', async () => {
    // User 1 creates node
    await createNode(page1, 'LLM', { x: 200, y: 200 });
    await page2.waitForTimeout(1000);
    
    // User 2 deletes it
    await page2.click('.react-flow__node');
    await page2.keyboard.press('Delete');
    
    // User 1 should see it deleted
    await page1.waitForTimeout(1000);
    await expect(page1.locator('.react-flow__node')).toHaveCount(0);
  });

  test('should sync node position updates', async () => {
    // User 1 creates node
    await createNode(page1, 'LLM', { x: 200, y: 200 });
    await page2.waitForTimeout(1000);
    
    // User 1 moves node
    const node1 = page1.locator('.react-flow__node');
    const box = await node1.boundingBox();
    await node1.hover();
    await page1.mouse.down();
    await page1.mouse.move(box!.x + 200, box!.y);
    await page1.mouse.up();
    
    // User 2 should see the new position
    await page2.waitForTimeout(1000);
    const node2 = page2.locator('.react-flow__node');
    const box2 = await node2.boundingBox();
    expect(box2!.x).toBeGreaterThan(box!.x + 150);
  });

  test('should sync edge creation between users', async () => {
    // User 1 creates two nodes
    await createNode(page1, 'Input', { x: 100, y: 200 });
    await createNode(page1, 'LLM', { x: 400, y: 200 });
    await page2.waitForTimeout(1000);
    
    // User 2 connects them
    await connectNodes(page2, 0, 1);
    
    // User 1 should see the edge
    await page1.waitForTimeout(1000);
    await expect(page1.locator('.react-flow__edge')).toHaveCount(1);
  });

  test('should display user cursors', async () => {
    // User 1 moves cursor
    await page1.mouse.move(300, 300);
    
    // User 2 should see user 1 cursor
    await page2.waitForTimeout(500);
    await expect(page2.locator('[data-user-cursor="testuser"]')).toBeVisible();
  });

  test('should show online users', async () => {
    // Both users should see each other in the user list
    await expect(page1.locator('text=/testuser/i')).toBeVisible();
    await expect(page1.locator('text=/collabuser/i')).toBeVisible();
    
    await expect(page2.locator('text=/testuser/i')).toBeVisible();
    await expect(page2.locator('text=/collabuser/i')).toBeVisible();
  });

  test('should handle concurrent edits gracefully', async () => {
    // Both users create nodes simultaneously
    await Promise.all([
      createNode(page1, 'LLM', { x: 100, y: 100 }),
      createNode(page2, 'Output', { x: 400, y: 100 }),
    ]);
    
    // Wait for sync
    await page1.waitForTimeout(1500);
    await page2.waitForTimeout(1500);
    
    // Both should see both nodes
    await expect(page1.locator('.react-flow__node')).toHaveCount(2);
    await expect(page2.locator('.react-flow__node')).toHaveCount(2);
  });

  test('should sync node property changes', async () => {
    // User 1 creates node
    await createNode(page1, 'LLM', { x: 200, y: 200 });
    await page2.waitForTimeout(1000);
    
    // User 1 configures node
    await page1.click('.react-flow__node');
    await page1.fill('input[name="model"]', 'gpt-4');
    await page1.click('button:has-text("Save")');
    
    // User 2 opens node properties and should see the changes
    await page2.waitForTimeout(1000);
    await page2.click('.react-flow__node');
    await expect(page2.locator('input[name="model"]')).toHaveValue('gpt-4');
  });

  test('should maintain document integrity with undo/redo', async () => {
    // User 1 creates nodes
    await createNode(page1, 'LLM', { x: 100, y: 100 });
    await createNode(page1, 'Output', { x: 300, y: 100 });
    await page2.waitForTimeout(1000);
    
    // User 1 undoes last action
    await page1.keyboard.press('Meta+Z');
    await page1.waitForTimeout(500);
    
    // Both should have 1 node
    await expect(page1.locator('.react-flow__node')).toHaveCount(1);
    await page2.waitForTimeout(1000);
    await expect(page2.locator('.react-flow__node')).toHaveCount(1);
  });

  test('should handle user disconnect and reconnect', async () => {
    // User 1 creates node
    await createNode(page1, 'LLM', { x: 200, y: 200 });
    await page2.waitForTimeout(1000);
    
    // User 2 disconnects (close browser tab)
    const context2 = page2.context();
    await page2.close();
    
    // User 1 creates another node
    await createNode(page1, 'Output', { x: 400, y: 200 });
    
    // User 2 reconnects
    page2 = await context2.newPage();
    await page2.goto('/dashboard');
    await page2.click('text=/Collab Project/i');
    await page2.waitForTimeout(1500);
    
    // Should see both nodes
    await expect(page2.locator('.react-flow__node')).toHaveCount(2);
  });

  test.afterAll(async () => {
    await page1.close();
    await page2.close();
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
