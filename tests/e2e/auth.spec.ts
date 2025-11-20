import { test, expect, Page } from '@playwright/test';

/**
 * E2E Tests for Authentication Flows
 * Tests user registration, login, token refresh, and protected routes
 */

// Test data
const testUser = {
  email: 'test@example.com',
  username: 'testuser',
  password: 'TestPass123!',
  fullName: 'Test User',
};

const testUser2 = {
  email: 'collab@example.com',
  username: 'collabuser',
  password: 'CollabPass123!',
  fullName: 'Collaborator User',
};

test.describe('Authentication', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the app
    await page.goto('/');
  });

  test('should display login page', async ({ page }) => {
    await expect(page.locator('h1')).toContainText(/sign in|login/i);
    await expect(page.locator('input[type="email"], input[name="username"]')).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
  });

  test('should register a new user', async ({ page }) => {
    // Navigate to registration
    await page.click('text=/sign up|register/i');
    
    // Fill registration form
    await page.fill('input[name="email"]', testUser.email);
    await page.fill('input[name="username"]', testUser.username);
    await page.fill('input[name="password"]', testUser.password);
    await page.fill('input[name="fullName"]', testUser.fullName);
    
    // Submit
    await page.click('button[type="submit"]');
    
    // Should redirect to dashboard or show success
    await expect(page).toHaveURL(/\/(dashboard|projects)/);
  });

  test('should login with username', async ({ page }) => {
    // Fill login form
    await page.fill('input[name="username"]', testUser.username);
    await page.fill('input[name="password"]', testUser.password);
    
    // Submit
    await page.click('button[type="submit"]');
    
    // Should be logged in
    await expect(page).toHaveURL(/\/(dashboard|projects)/);
    await expect(page.locator('text=/profile|account|' + testUser.username + '/i')).toBeVisible();
  });

  test('should login with email', async ({ page }) => {
    // Fill login form with email
    await page.fill('input[name="username"]', testUser.email);
    await page.fill('input[name="password"]', testUser.password);
    
    // Submit
    await page.click('button[type="submit"]');
    
    // Should be logged in
    await expect(page).toHaveURL(/\/(dashboard|projects)/);
  });

  test('should show error for wrong password', async ({ page }) => {
    await page.fill('input[name="username"]', testUser.username);
    await page.fill('input[name="password"]', 'WrongPassword123!');
    
    await page.click('button[type="submit"]');
    
    // Should show error message
    await expect(page.locator('text=/invalid|incorrect|wrong/i')).toBeVisible();
  });

  test('should protect routes requiring authentication', async ({ page }) => {
    // Try to access protected route
    await page.goto('/projects/new');
    
    // Should redirect to login
    await expect(page).toHaveURL(/\/(login|signin)/);
  });

  test('should logout successfully', async ({ page }) => {
    // Login first
    await page.fill('input[name="username"]', testUser.username);
    await page.fill('input[name="password"]', testUser.password);
    await page.click('button[type="submit"]');
    
    await expect(page).toHaveURL(/\/(dashboard|projects)/);
    
    // Logout
    await page.click('text=/logout|sign out/i');
    
    // Should redirect to login
    await expect(page).toHaveURL(/\/(login|signin|\/)/);
  });

  test('should update user profile', async ({ page }) => {
    // Login
    await loginUser(page, testUser.username, testUser.password);
    
    // Navigate to profile
    await page.click('text=/profile|account/i');
    
    // Update full name
    const newName = 'Updated Test User';
    await page.fill('input[name="fullName"]', newName);
    await page.click('button:has-text("Save")');
    
    // Should show success
    await expect(page.locator('text=/success|updated/i')).toBeVisible();
  });

  test('should change password', async ({ page }) => {
    // Login
    await loginUser(page, testUser.username, testUser.password);
    
    // Navigate to profile/security
    await page.click('text=/profile|account/i');
    await page.click('text=/security|password/i');
    
    // Change password
    const newPassword = 'NewTestPass123!';
    await page.fill('input[name="currentPassword"]', testUser.password);
    await page.fill('input[name="newPassword"]', newPassword);
    await page.fill('input[name="confirmPassword"]', newPassword);
    await page.click('button:has-text("Change Password")');
    
    // Should show success
    await expect(page.locator('text=/success|updated/i')).toBeVisible();
    
    // Logout and login with new password
    await page.click('text=/logout|sign out/i');
    await loginUser(page, testUser.username, newPassword);
    
    await expect(page).toHaveURL(/\/(dashboard|projects)/);
  });
});

test.describe('Project Permissions', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await loginUser(page, testUser.username, testUser.password);
  });

  test('should create project as owner', async ({ page }) => {
    // Create new project
    await page.click('text=/new project|create/i');
    await page.fill('input[name="name"]', 'Test Project');
    await page.click('button:has-text("Create")');
    
    // Should be owner
    await expect(page.locator('text=/owner/i')).toBeVisible();
  });

  test('should add collaborator to project', async ({ page }) => {
    // Navigate to project settings
    await page.click('text=/settings|manage/i');
    await page.click('text=/members|collaborators/i');
    
    // Add member
    await page.fill('input[name="email"]', testUser2.email);
    await page.selectOption('select[name="role"]', 'EDITOR');
    await page.click('button:has-text("Add Member")');
    
    // Should see member in list
    await expect(page.locator(`text=${testUser2.email}`)).toBeVisible();
    await expect(page.locator('text=/editor/i')).toBeVisible();
  });

  test('should not allow editor to add members', async ({ page, context }) => {
    // Login as collaborator
    await page.click('text=/logout|sign out/i');
    await loginUser(page, testUser2.username, testUser2.password);
    
    // Open project
    await page.click('text=/Test Project/i');
    
    // Try to access members - should be disabled or hidden
    await page.click('text=/settings|manage/i');
    const membersButton = page.locator('text=/members|collaborators/i');
    
    if (await membersButton.isVisible()) {
      await membersButton.click();
      // Should not have "Add Member" button
      await expect(page.locator('button:has-text("Add Member")')).not.toBeVisible();
    }
  });

  test('should remove member from project', async ({ page }) => {
    // Navigate to members
    await page.click('text=/settings|manage/i');
    await page.click('text=/members|collaborators/i');
    
    // Remove member
    await page.locator(`tr:has-text("${testUser2.email}") button:has-text("Remove")`).click();
    await page.click('button:has-text("Confirm")');
    
    // Should not see member anymore
    await expect(page.locator(`text=${testUser2.email}`)).not.toBeVisible();
  });
});

// Helper functions
async function loginUser(page: Page, username: string, password: string) {
  // Check if already on login page
  const currentUrl = page.url();
  if (!currentUrl.includes('/login') && !currentUrl.includes('/signin')) {
    await page.goto('/login');
  }
  
  await page.fill('input[name="username"]', username);
  await page.fill('input[name="password"]', password);
  await page.click('button[type="submit"]');
  
  // Wait for navigation
  await page.waitForURL(/\/(dashboard|projects)/, { timeout: 5000 });
}
