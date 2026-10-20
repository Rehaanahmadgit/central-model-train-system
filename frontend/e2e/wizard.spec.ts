import { expect, test } from "@playwright/test";

// One E2E smoke test [doc 11 §1, DIST-11]: the sample dataset flows through
// the 5-step wizard into a real RandomForest run and reaches `completed`.
test("sample dataset -> RandomForest -> verify -> train -> completed", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("button", { name: "New training" }).click();

  // Step 1: Dataset — use the generated sample (DIST-11).
  await page.getByTestId("try-sample").click();

  // Step 2: Model — pick RandomForest (tabular).
  await page.getByTestId("model-random_forest").click();

  // Step 3: Verify — start and wait for the pass banner.
  await page.getByTestId("start-verification").click();
  await expect(page.getByText("Dataset verified successfully")).toBeVisible({ timeout: 60_000 });
  // Warnings (tiny data, no GPU) need acknowledgement before Next [doc 06 §8].
  const ack = page.getByTestId("ack-warnings");
  if (await ack.isVisible()) await ack.check();
  await page.getByTestId("verify-next").click();

  // Step 4: Configure — defaults are fine; skip straight to Train.
  await page.getByTestId("configure-next").click();

  // Step 5: Train — start and land on the LiveRun page.
  await page.getByTestId("start-training").click();

  await expect(page.getByTestId("run-state")).toHaveText("completed", { timeout: 90_000 });

  // Capture the finished LiveRun page as an artifact for review.
  await page.screenshot({ path: "test-results/liverun-completed.png", fullPage: true });
});
