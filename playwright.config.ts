import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright E2E Test Configuration for N3 Graph Editor
 * Tests the complete system: Frontend, Backend, YJS Server, Database
 */
export default defineConfig({
  testDir: './tests/e2e',
  
  // Maximum time one test can run
  timeout: 60 * 1000,
  
  // Test execution settings
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  
  // Reporter configuration
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'test-results/results.json' }],
    ['list']
  ],
  
  // Shared settings for all tests
  use: {
    // Base URL for the frontend
    baseURL: 'http://localhost:3000',
    
    // Capture screenshots on failure
    screenshot: 'only-on-failure',
    
    // Capture video on first retry
    video: 'retain-on-failure',
    
    // Capture trace on first retry
    trace: 'on-first-retry',
    
    // API endpoint
    extraHTTPHeaders: {
      'Accept': 'application/json',
    },
  },

  // Configure projects for different browsers
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    
    // Mobile viewports
    {
      name: 'mobile-chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'mobile-safari',
      use: { ...devices['iPhone 12'] },
    },
  ],

  // Run local dev server before starting tests
  webServer: [
    {
      command: 'docker-compose up -d postgres backend yjs-server',
      port: 8000,
      timeout: 120 * 1000,
      reuseExistingServer: !process.env.CI,
    },
    {
      command: 'cd src/web/graph-editor && npm run dev',
      port: 3000,
      timeout: 120 * 1000,
      reuseExistingServer: !process.env.CI,
    },
  ],
});
