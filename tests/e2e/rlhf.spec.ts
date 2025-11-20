import { test, expect, Page } from '@playwright/test';

/**
 * E2E Tests for RLHF Training
 * Tests feedback submission, policy training, and model evaluation
 */

test.describe('RLHF Training', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await loginUser(page, 'testuser', 'TestPass123!');
    
    // Create project with LLM graph
    await page.click('text=/new project|create/i');
    await page.fill('input[name="name"]', 'RLHF Test Project');
    await page.click('button:has-text("Create")');
    
    // Create simple LLM graph
    await createNode(page, 'LLM', { x: 200, y: 200 });
    await page.click('.react-flow__node');
    await page.selectOption('select[name="provider"]', 'openai');
    await page.fill('textarea[name="prompt"]', 'Test prompt');
    await page.click('button:has-text("Save")');
    
    // Run to get output
    await page.click('button:has-text("Run")');
    await page.waitForSelector('text=/completed/i', { timeout: 30000 });
  });

  test('should submit positive feedback', async ({ page }) => {
    // Click on node to see output
    await page.click('.react-flow__node');
    
    // Submit positive feedback
    await page.click('button[data-feedback="positive"]');
    
    // Should show success
    await expect(page.locator('text=/feedback submitted|thanks/i')).toBeVisible();
  });

  test('should submit negative feedback', async ({ page }) => {
    // Click on node
    await page.click('.react-flow__node');
    
    // Submit negative feedback
    await page.click('button[data-feedback="negative"]');
    
    // Should show success
    await expect(page.locator('text=/feedback submitted|thanks/i')).toBeVisible();
  });

  test('should provide feedback with comments', async ({ page }) => {
    // Open feedback dialog
    await page.click('.react-flow__node');
    await page.click('button:has-text("Provide Feedback")');
    
    // Fill feedback form
    await page.click('button[data-rating="4"]');
    await page.fill('textarea[name="comments"]', 'Good output but could be better');
    await page.click('button:has-text("Submit")');
    
    // Should show success
    await expect(page.locator('text=/feedback submitted/i')).toBeVisible();
  });

  test('should view feedback history', async ({ page }) => {
    // Navigate to feedback section
    await page.click('text=/feedback|training/i');
    
    // Should see feedback list
    await expect(page.locator('text=/feedback history/i')).toBeVisible();
    await expect(page.locator('[data-feedback-item]')).toHaveCount(1, { timeout: 5000 });
  });

  test('should train policy from feedback', async ({ page }) => {
    // Submit multiple feedbacks first
    for (let i = 0; i < 3; i++) {
      await page.click('button:has-text("Run")');
      await page.waitForSelector('text=/completed/i', { timeout: 30000 });
      await page.click('.react-flow__node');
      await page.click('button[data-feedback="positive"]');
      await page.waitForTimeout(1000);
    }
    
    // Navigate to training
    await page.click('text=/training|rlhf/i');
    
    // Start training
    await page.click('button:has-text("Train Policy")');
    
    // Should show training started
    await expect(page.locator('text=/training started|in progress/i')).toBeVisible();
  });

  test('should display training progress', async ({ page }) => {
    // Navigate to training
    await page.click('text=/training|rlhf/i');
    
    // Start training
    await page.click('button:has-text("Train Policy")');
    
    // Should show progress bar or status
    await expect(page.locator('[data-training-progress]')).toBeVisible();
    await expect(page.locator('text=/epoch|step|loss/i')).toBeVisible();
  });

  test('should view training metrics', async ({ page }) => {
    // Navigate to training history
    await page.click('text=/training|rlhf/i');
    await page.click('text=/history|past runs/i');
    
    // Should see training runs
    await expect(page.locator('[data-training-run]')).toBeVisible();
    
    // Click on a run
    await page.locator('[data-training-run]').first().click();
    
    // Should show metrics
    await expect(page.locator('text=/accuracy|loss|reward/i')).toBeVisible();
  });

  test('should compare model versions', async ({ page }) => {
    // Navigate to models
    await page.click('text=/models|versions/i');
    
    // Should see model versions
    await expect(page.locator('[data-model-version]')).toHaveCount(1, { timeout: 5000 });
    
    // Select two versions to compare
    await page.check('input[type="checkbox"][data-model="v1"]');
    await page.check('input[type="checkbox"][data-model="v2"]');
    await page.click('button:has-text("Compare")');
    
    // Should show comparison
    await expect(page.locator('text=/comparison|metrics/i')).toBeVisible();
  });

  test('should deploy trained model', async ({ page }) => {
    // Navigate to models
    await page.click('text=/models|versions/i');
    
    // Select best model
    await page.locator('[data-model-version]').first().click();
    
    // Deploy
    await page.click('button:has-text("Deploy")');
    await page.click('button:has-text("Confirm")');
    
    // Should show deployed status
    await expect(page.locator('text=/deployed|active/i')).toBeVisible();
  });

  test('should export feedback dataset', async ({ page }) => {
    // Navigate to feedback
    await page.click('text=/feedback|training/i');
    
    // Export
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.click('button:has-text("Export")'),
    ]);
    
    // Should download dataset
    expect(download.suggestedFilename()).toMatch(/feedback.*\.(json|csv)/i);
  });

  test('should configure reward model', async ({ page }) => {
    // Navigate to RLHF settings
    await page.click('text=/settings/i');
    await page.click('text=/rlhf|reward model/i');
    
    // Configure reward model
    await page.selectOption('select[name="reward_model"]', 'custom');
    await page.fill('input[name="model_path"]', 'path/to/model');
    await page.fill('input[name="learning_rate"]', '0.0001');
    await page.click('button:has-text("Save")');
    
    // Should show success
    await expect(page.locator('text=/saved|success/i')).toBeVisible();
  });

  test('should monitor training in real-time', async ({ page }) => {
    // Navigate to training
    await page.click('text=/training|rlhf/i');
    
    // Start training
    await page.click('button:has-text("Train Policy")');
    
    // Should update metrics in real-time
    const initialLoss = await page.locator('[data-metric="loss"]').textContent();
    
    await page.waitForTimeout(5000);
    
    const updatedLoss = await page.locator('[data-metric="loss"]').textContent();
    expect(updatedLoss).not.toBe(initialLoss);
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
