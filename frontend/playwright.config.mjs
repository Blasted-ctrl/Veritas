import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  globalTimeout: 180_000,
  timeout: 30_000,
  expect: { timeout: 10_000 },
  workers: 1,
  retries: 0,
  reporter: [["list"], ["json", { outputFile: "../.turbo/browser-checks/results.json" }]],
  outputDir: "test-results",
  use: {
    baseURL: "http://127.0.0.1:3010",
    viewport: { width: 1280, height: 800 },
    colorScheme: "light",
    reducedMotion: "reduce",
    channel: process.env.PLAYWRIGHT_CHANNEL || undefined,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "node node_modules/next/dist/bin/next start --hostname 127.0.0.1 --port 3010",
    url: "http://127.0.0.1:3010",
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
