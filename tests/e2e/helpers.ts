import { Page, expect as playwrightExpect } from '@playwright/test';

/**
 * E2E Test Utilities
 * Shared helpers and utilities for Playwright E2E tests
 */

// ============================================================================
// Authentication Helpers
// ============================================================================

export async function loginUser(page: Page, username: string, password: string) {
  const currentUrl = page.url();
  if (!currentUrl.includes('/login') && !currentUrl.includes('/signin')) {
    await page.goto('/login');
  }
  
  await page.fill('input[name="username"]', username);
  await page.fill('input[name="password"]', password);
  await page.click('button[type="submit"]');
  
  await page.waitForURL(/\/(dashboard|projects)/, { timeout: 10000 });
}

export async function registerUser(page: Page, user: {
  email: string;
  username: string;
  password: string;
  fullName: string;
}) {
  await page.goto('/');
  await page.click('text=/sign up|register/i');
  
  await page.fill('input[name="email"]', user.email);
  await page.fill('input[name="username"]', user.username);
  await page.fill('input[name="password"]', user.password);
  await page.fill('input[name="fullName"]', user.fullName);
  
  await page.click('button[type="submit"]');
  await page.waitForURL(/\/(dashboard|projects)/, { timeout: 10000 });
}

export async function logoutUser(page: Page) {
  await page.click('text=/logout|sign out/i');
  await page.waitForURL(/\/(login|signin|\/)/);
}

// ============================================================================
// Graph Editing Helpers
// ============================================================================

export async function createNode(
  page: Page,
  type: string,
  position: { x: number; y: number }
) {
  const canvas = page.locator('.react-flow__pane');
  await canvas.click({ button: 'right', position });
  await page.click(`text=/^${type}$/i`);
  await page.waitForTimeout(500);
}

export async function deleteNode(page: Page, nodeIndex: number = 0) {
  const node = page.locator('.react-flow__node').nth(nodeIndex);
  await node.click();
  await page.keyboard.press('Delete');
  await page.waitForTimeout(300);
}

export async function connectNodes(
  page: Page,
  sourceIndex: number,
  targetIndex: number
) {
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

export async function moveNode(
  page: Page,
  nodeIndex: number,
  delta: { x: number; y: number }
) {
  const node = page.locator('.react-flow__node').nth(nodeIndex);
  const box = await node.boundingBox();
  
  if (!box) throw new Error('Node not found');
  
  await node.hover();
  await page.mouse.down();
  await page.mouse.move(box.x + delta.x, box.y + delta.y);
  await page.mouse.up();
  await page.waitForTimeout(300);
}

export async function configureNode(
  page: Page,
  nodeIndex: number,
  config: Record<string, string>
) {
  const node = page.locator('.react-flow__node').nth(nodeIndex);
  await node.click();
  
  for (const [key, value] of Object.entries(config)) {
    const input = page.locator(`input[name="${key}"], textarea[name="${key}"], select[name="${key}"]`);
    
    const tagName = await input.evaluate(el => el.tagName.toLowerCase());
    if (tagName === 'select') {
      await input.selectOption(value);
    } else {
      await input.fill(value);
    }
  }
  
  await page.click('button:has-text("Save")');
  await page.waitForTimeout(300);
}

// ============================================================================
// Project Management Helpers
// ============================================================================

export async function createProject(page: Page, name: string) {
  await page.click('text=/new project|create/i');
  await page.fill('input[name="name"]', name);
  await page.click('button:has-text("Create")');
  await playwrightExpect(page.locator('.react-flow')).toBeVisible();
}

export async function openProject(page: Page, name: string) {
  await page.goto('/dashboard');
  await page.click(`text=/${name}/i`);
  await playwrightExpect(page.locator('.react-flow')).toBeVisible();
}

export async function deleteProject(page: Page, name: string) {
  await page.goto('/dashboard');
  await page.locator(`[data-project="${name}"] button:has-text("Delete")`).click();
  await page.click('button:has-text("Confirm")');
  await page.waitForTimeout(500);
}

export async function shareProject(
  page: Page,
  email: string,
  role: 'OWNER' | 'EDITOR' | 'VIEWER' = 'EDITOR'
) {
  await page.click('text=/share|invite/i');
  await page.fill('input[name="email"]', email);
  await page.selectOption('select[name="role"]', role);
  await page.click('button:has-text("Invite")');
  await page.waitForTimeout(500);
}

// ============================================================================
// Execution Helpers
// ============================================================================

export async function runGraph(page: Page, timeout: number = 30000) {
  await page.click('button:has-text("Run")');
  await playwrightExpect(page.locator('text=/running|executing/i')).toBeVisible();
  await playwrightExpect(page.locator('text=/completed|success/i')).toBeVisible({ timeout });
}

export async function cancelExecution(page: Page) {
  await page.click('button:has-text("Cancel")');
  await playwrightExpect(page.locator('text=/cancelled|stopped/i')).toBeVisible();
}

export async function viewExecutionTrace(page: Page) {
  await page.click('button:has-text("View Trace")');
  await playwrightExpect(page.locator('text=/execution|trace/i')).toBeVisible();
}

export async function getNodeOutput(page: Page, nodeIndex: number): Promise<string> {
  const node = page.locator('.react-flow__node').nth(nodeIndex);
  await node.click();
  
  const output = page.locator('[data-node-output]');
  return await output.textContent() || '';
}

// ============================================================================
// Feedback & Training Helpers
// ============================================================================

export async function submitFeedback(
  page: Page,
  nodeIndex: number,
  rating: 'positive' | 'negative' | number,
  comment?: string
) {
  const node = page.locator('.react-flow__node').nth(nodeIndex);
  await node.click();
  
  if (typeof rating === 'string') {
    await page.click(`button[data-feedback="${rating}"]`);
  } else {
    await page.click('button:has-text("Provide Feedback")');
    await page.click(`button[data-rating="${rating}"]`);
    
    if (comment) {
      await page.fill('textarea[name="comments"]', comment);
    }
    
    await page.click('button:has-text("Submit")');
  }
  
  await playwrightExpect(page.locator('text=/feedback submitted/i')).toBeVisible();
}

export async function startTraining(page: Page) {
  await page.click('text=/training|rlhf/i');
  await page.click('button:has-text("Train Policy")');
  await playwrightExpect(page.locator('text=/training started/i')).toBeVisible();
}

export async function waitForTrainingComplete(page: Page, timeout: number = 120000) {
  await playwrightExpect(page.locator('text=/training complete|finished/i'))
    .toBeVisible({ timeout });
}

// ============================================================================
// Assertion Helpers
// ============================================================================

export async function assertNodeCount(page: Page, count: number) {
  await playwrightExpect(page.locator('.react-flow__node')).toHaveCount(count);
}

export async function assertEdgeCount(page: Page, count: number) {
  await playwrightExpect(page.locator('.react-flow__edge')).toHaveCount(count);
}

export async function assertNodeType(page: Page, nodeIndex: number, type: string) {
  const node = page.locator('.react-flow__node').nth(nodeIndex);
  await playwrightExpect(node).toHaveAttribute('data-type', type);
}

export async function assertToastMessage(page: Page, message: string | RegExp) {
  const toast = page.locator('[data-toast-message]');
  if (typeof message === 'string') {
    await playwrightExpect(toast).toHaveText(message);
  } else {
    await playwrightExpect(toast).toHaveText(message);
  }
}

// ============================================================================
// Wait Helpers
// ============================================================================

export async function waitForSync(page: Page, ms: number = 1000) {
  await page.waitForTimeout(ms);
}

export async function waitForNavigation(page: Page, urlPattern: RegExp) {
  await page.waitForURL(urlPattern, { timeout: 10000 });
}

// ============================================================================
// Test Data Generators
// ============================================================================

export function generateTestUser(suffix: string = '') {
  const timestamp = Date.now();
  return {
    email: `test${suffix}${timestamp}@example.com`,
    username: `testuser${suffix}${timestamp}`,
    password: 'TestPass123!',
    fullName: `Test User ${suffix}`,
  };
}

export function generateProjectName(prefix: string = 'Test') {
  return `${prefix} Project ${Date.now()}`;
}

// ============================================================================
// API Helpers
// ============================================================================

export async function makeAPIRequest(
  page: Page,
  endpoint: string,
  options: {
    method?: string;
    body?: any;
    headers?: Record<string, string>;
  } = {}
) {
  const token = await page.evaluate(() => localStorage.getItem('access_token'));
  
  return await page.request.fetch(`http://localhost:8000${endpoint}`, {
    method: options.method || 'GET',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
      ...options.headers,
    },
    data: options.body,
  });
}

export async function cleanupTestData(page: Page) {
  // Delete all test projects
  await page.goto('/dashboard');
  const deleteButtons = page.locator('button:has-text("Delete")');
  const count = await deleteButtons.count();
  
  for (let i = 0; i < count; i++) {
    await deleteButtons.first().click();
    await page.click('button:has-text("Confirm")');
    await page.waitForTimeout(500);
  }
}
