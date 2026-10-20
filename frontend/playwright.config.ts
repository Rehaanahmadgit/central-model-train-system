import { defineConfig } from "@playwright/test";

// E2E smoke: sample dataset -> RandomForest -> verify -> train -> LiveRun.
// Served by the backend (which serves the built frontend at /), so only the
// backend needs to be running on 8080 [doc 11 §1 end-to-end].
export default defineConfig({
  testDir: "./e2e",
  timeout: 180_000,
  expect: { timeout: 30_000 },
  use: {
    baseURL: process.env.TRAINHUB_E2E_URL ?? "http://127.0.0.1:8080",
    screenshot: "only-on-failure",
  },
  reporter: [["list"]],
});
