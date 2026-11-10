import { expect, test } from "@playwright/test";

// "Try it" tab [Slice G1]. REST is mocked (the backend side — known-answer numbers,
// limits, token, torch-free — is covered by backend/tests/test_infer.py); this test
// proves the browser really sends the request and draws the boxes it gets back.
const RUN_ID = "run_20261110_000000_abcd";
const PNG_1X1 = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
  "base64",
);

test("sample images -> boxes render, confidence slider filters them", async ({ page }) => {
  let inferUrl = "";
  await page.route("**/api/v1/runs", (route) =>
    route.fulfill({
      json: { items: [{ id: RUN_ID, name: "det-run", state: "completed", model_family: "rtdetr" }] },
    }),
  );
  await page.route(`**/api/v1/runs/${RUN_ID}`, (route) =>
    route.fulfill({
      json: {
        id: RUN_ID,
        name: "det-run",
        state: "completed",
        model_family: "rtdetr",
        task_type: "object_detection",
        verification_id: "ver_x",
        dataset_hash: "abc",
        checkpoint: {},
        config: { imgsz: 128 },
        seed: 42,
        env: {},
        queued_at: null,
        started_at: null,
        finished_at: null,
        best: { metric: "val/mAP50", value: 0.5, epoch: 1 },
      },
    }),
  );
  await page.route(`**/api/v1/runs/${RUN_ID}/infer-samples*`, (route) => {
    inferUrl = route.request().url();
    return route.fulfill({
      json: {
        run_id: RUN_ID,
        threshold: 0.25,
        class_names: ["apple", "orange"],
        device: { kind: "cpu", name: "CPU" },
        device_note: "No GPU available, so detection ran on CPU (slower, same result)",
        images: [
          {
            name: "img1.png",
            path: "test/images/img1.png",
            split: "test",
            width: 100,
            height: 100,
            gt: [[0, 0.5, 0.5, 0.4, 0.4]],
            pred: [
              [0, 0.9, 0.5, 0.5, 0.4, 0.4],
              [1, 0.3, 0.2, 0.2, 0.1, 0.1],
            ],
          },
        ],
      },
    });
  });
  await page.route(`**/api/v1/runs/${RUN_ID}/images*`, (route) =>
    route.fulfill({ body: PNG_1X1, contentType: "image/png" }),
  );

  await page.goto("/");
  await page.getByRole("button", { name: "Runs", exact: true }).click();
  await page.getByRole("button", { name: /det-run/ }).click();
  await page.getByRole("tab", { name: "Try it" }).click();

  await page.getByTestId("tryit-samples").click();
  expect(inferUrl).toContain("/infer-samples");
  await expect(page.getByTestId("tryit-device")).toContainText("CPU");
  await expect(page.getByTestId("tryit-result")).toHaveCount(1);
  // default confidence 0.25: the 0.9 and 0.3 predictions plus the GT box are drawn
  await expect(page.locator('[data-box="pred"]')).toHaveCount(2);
  await expect(page.locator('[data-box="gt"]')).toHaveCount(1);
  await expect(page.getByTestId("tryit-count")).toContainText("2 detections");

  await page.getByTestId("tryit-confidence").fill("0.5");
  await expect(page.locator('[data-box="pred"]')).toHaveCount(1);
  await expect(page.getByTestId("tryit-count")).toContainText("1 detection");

  const download = page.waitForEvent("download");
  await page.getByTestId("tryit-download").click();
  expect((await download).suggestedFilename()).toBe("img1_detected.png");
});

test("a refused file is explained before any upload", async ({ page }) => {
  await page.route("**/api/v1/runs", (route) =>
    route.fulfill({
      json: { items: [{ id: RUN_ID, name: "det-run", state: "completed", model_family: "rtdetr" }] },
    }),
  );
  await page.route(`**/api/v1/runs/${RUN_ID}`, (route) =>
    route.fulfill({
      json: {
        id: RUN_ID, name: "det-run", state: "completed", model_family: "rtdetr",
        task_type: "object_detection", verification_id: "ver_x", dataset_hash: "abc",
        checkpoint: {}, config: { imgsz: 128 }, seed: 42, env: {}, queued_at: null,
        started_at: null, finished_at: null, best: { metric: "val/mAP50", value: 0.5, epoch: 1 },
      },
    }),
  );
  await page.goto("/");
  await page.getByRole("button", { name: "Runs", exact: true }).click();
  await page.getByRole("button", { name: /det-run/ }).click();
  await page.getByRole("tab", { name: "Try it" }).click();
  await page.getByTestId("tryit-file-input").setInputFiles({
    name: "notes.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("hi"),
  });
  await expect(page.getByTestId("tryit-problems")).toContainText("only PNG, JPG or WEBP");
  await expect(page.getByTestId("tryit-run")).toBeDisabled();
});
