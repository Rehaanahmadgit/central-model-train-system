import { expect, test } from "@playwright/test";
import { crc32 } from "node:zlib";

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

  // Terminal run: "Run again" / "Edit & run" are available, but a RandomForest
  // run has no checkpoints, so no download buttons appear [doc 09 §3].
  await expect(page.getByTestId("run-again")).toBeVisible();
  await expect(page.getByTestId("edit-and-run")).toBeVisible();
  await expect(page.getByTestId("download-best")).toHaveCount(0);
  await expect(page.getByTestId("download-weights")).toHaveCount(0);

  // Capture the finished LiveRun page as an artifact for review.
  await page.screenshot({ path: "test-results/liverun-completed.png", fullPage: true });
});

// --- helpers ---------------------------------------------------------------

const PNG_16PX = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAF0lEQVR4nGP8z0AaYCJR/aiGUQ1DSAMAQC4BH2bjRnMAAAAASUVORK5CYII=",
  "base64",
);

// Minimal stored (uncompressed) ZIP writer; Playwright has no built-in zipper.
function storedZip(entries: Record<string, Buffer>): Buffer {
  const chunks: Buffer[] = [];
  const central: Buffer[] = [];
  let offset = 0;
  for (const [name, data] of Object.entries(entries)) {
    const nameBuf = Buffer.from(name);
    const crc = crc32(data);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0, 8); // method: stored
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(data.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(nameBuf.length, 26);
    chunks.push(local, nameBuf, data);

    const cent = Buffer.alloc(46);
    cent.writeUInt32LE(0x02014b50, 0);
    cent.writeUInt16LE(20, 4);
    cent.writeUInt16LE(20, 6);
    cent.writeUInt32LE(crc, 16);
    cent.writeUInt32LE(data.length, 20);
    cent.writeUInt32LE(data.length, 24);
    cent.writeUInt16LE(nameBuf.length, 28);
    cent.writeUInt32LE(offset, 42);
    central.push(cent, nameBuf);
    offset += local.length + nameBuf.length + data.length;
  }
  const centralSize = central.reduce((n, b) => n + b.length, 0);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(Object.keys(entries).length, 8);
  end.writeUInt16LE(Object.keys(entries).length, 10);
  end.writeUInt32LE(centralSize, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...chunks, ...central, end]);
}

function makeYoloZip(): Buffer {
  const entries: Record<string, Buffer> = {
    "data.yaml": Buffer.from("names:\n  - apple\n  - orange\n"),
    "images/img0.png": PNG_16PX,
    "images/img1.png": PNG_16PX,
    "images/img2.png": PNG_16PX,
    "images/img3.png": PNG_16PX,
    "labels/img0.txt": Buffer.from("0 0.25 0.25 0.5 0.5\n"),
    "labels/img1.txt": Buffer.from("1 0.25 0.25 0.5 0.5\n"),
    "labels/img2.txt": Buffer.from("0 0.25 0.25 0.5 0.5\n"),
    "labels/img3.txt": Buffer.from("1 0.25 0.25 0.5 0.5\n"),
  };
  return storedZip(entries);
}

// Configure must prefill from the verifier's recommended_hparams and
// auto-fill the recommended checkpoint (GPU "full" preset simulated via the
// report route so the test is deterministic on CPU).
test("wizard prefills Configure from recommended_hparams and checkpoint", async ({ page }) => {
  let postedPreset: string | null = null;
  await page.route("**/api/v1/verifications", async (route) => {
    if (route.request().method() === "POST") {
      const body = route.request().postDataJSON() as { preset?: string };
      postedPreset = body.preset ?? null;
    }
    await route.continue();
  });
  await page.route("**/api/v1/verifications/*", async (route) => {
    if (route.request().method() !== "GET") return route.continue();
    await route.fulfill({
      json: {
        id: "ver_mock",
        status: "passed_with_warnings",
        checks: [
          {
            name: "archive_integrity",
            status: "pass",
            severity: "error",
            message: "tree is clean",
            details: {},
            fix_hint: null,
            samples: [],
          },
        ],
        derived: {
          task_type: "object_detection",
          format: "yolo",
          recommended_hparams: {
            preset: "full",
            epochs: 20,
            batch_size: 1,
            imgsz: 640,
            lr: 0.0001,
            freeze_backbone: true,
            device: "cuda",
            model_family: "rtdetr",
          },
          recommended_checkpoints: [
            {
              catalog_id: "rtdetr-r18vd@coco-1.0",
              score: 10,
              reason: "exact task match",
              local: true,
              downloadable: true,
              eligible: true,
            },
          ],
          auto_selected: "rtdetr-r18vd@coco-1.0",
          auto_select_reason:
            "exact task match (object_detection); pretrained on coco; passed: task, family, class head, VRAM, licence, verified sha256",
        },
      },
    });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "New training" }).click();
  await page.locator("#dataset-file-input").setInputFiles({
    name: "apple-orange.zip",
    mimeType: "application/zip",
    buffer: makeYoloZip(),
  });

  await page.getByTestId("model-rtdetr").click();
  await page.getByTestId("start-verification").click();
  await expect(page.getByText("Dataset verified successfully")).toBeVisible({ timeout: 60_000 });
  const ack = page.getByTestId("ack-warnings");
  if (await ack.isVisible()) await ack.check();
  await page.getByTestId("verify-next").click();

  // The wizard must ask the backend to auto-resolve the preset (full on GPU).
  expect(postedPreset).toBe("auto");

  const epochs = page.locator("label").filter({ hasText: "epochs" }).locator("input");
  await expect(epochs).toHaveValue("20");
  const imgsz = page.locator("label").filter({ hasText: "imgsz" }).locator("input");
  await expect(imgsz).toHaveValue("640");
  const devPreset = page.locator("label").filter({ hasText: "dev_preset" }).locator("input");
  await expect(devPreset).not.toBeChecked();
  await expect(page.getByTestId("run-target")).toContainText("GPU (full)");
  const checkpoint = page.locator('input[aria-label="Checkpoint catalog id"]');
  await expect(checkpoint).toHaveValue("rtdetr-r18vd@coco-1.0");
  // "Kyun chuna" is shown, and an override is acknowledged [Slice D].
  await expect(page.getByTestId("auto-reason")).toContainText("exact task match");
  await checkpoint.fill("my-own@1");
  await expect(page.getByTestId("auto-reason")).toHaveCount(0);
  await expect(page.getByTestId("manual-pick")).toContainText("rtdetr-r18vd@coco-1.0");
});

// Scratch (or empty checkpoint) + frozen backbone cannot learn: Next is blocked with
// a red message, and unblocks once freeze_backbone is off [Slice D].
test("scratch with a frozen backbone blocks Configure Next", async ({ page }) => {
  await page.route("**/api/v1/verifications/*", async (route) => {
    if (route.request().method() !== "GET") return route.continue();
    await route.fulfill({
      json: {
        id: "ver_mock_frozen",
        status: "passed",
        can_train: true,
        checks: [],
        derived: {
          task_type: "object_detection",
          format: "yolo",
          recommended_checkpoints: [],
          auto_selected: null,
          auto_select_reason: "no checkpoint passed the hard filters (test)",
        },
      },
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "New training" }).click();
  await page.locator("#dataset-file-input").setInputFiles({
    name: "apple-orange.zip",
    mimeType: "application/zip",
    buffer: makeYoloZip(),
  });
  await page.getByTestId("model-rtdetr").click();
  await page.getByTestId("start-verification").click();
  await expect(page.getByText("Dataset verified successfully")).toBeVisible({ timeout: 60_000 });
  const ack = page.getByTestId("ack-warnings");
  if (await ack.isVisible()) await ack.check();
  await page.getByTestId("verify-next").click();

  // Nothing auto-selected + freeze_backbone default true => blocked.
  const freeze = page.locator("label").filter({ hasText: "freeze_backbone" }).locator("input");
  await expect(freeze).toBeChecked();
  await expect(page.getByTestId("configure-next")).toBeDisabled();
  await expect(page.getByTestId("licence-block")).toContainText("frozen backbone");

  await page.getByLabel("Train from scratch").check();
  await expect(page.getByTestId("configure-next")).toBeDisabled();

  await freeze.uncheck();
  await expect(page.getByTestId("configure-next")).toBeEnabled();
  await freeze.check();
  await expect(page.getByTestId("configure-next")).toBeDisabled();
});


// A failed check must show its one-line message AND the "what to do" hint
// inline, without expanding the row [Slice B]. (Only the REST report is mocked;
// the Next-button gate comes from the live stream and is covered by backend tests.)
test("failed check shows message and fix hint inline", async ({ page }) => {
  await page.route("**/api/v1/verifications/*", async (route) => {
    if (route.request().method() !== "GET") return route.continue();
    await route.fulfill({
      json: {
        id: "ver_mock_fail",
        status: "failed",
        can_train: false,
        checks: [
          {
            name: "empty_labels",
            status: "fail",
            severity: "error",
            message: "all 4 images have no boxes: nothing to train on",
            details: {},
            fix_hint: "Add annotations (YOLO: one 'class x y w h' line per object), then verify again.",
            samples: ["images/img0.png"],
          },
        ],
        derived: { task_type: "object_detection", format: "yolo" },
      },
    });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "New training" }).click();
  await page.locator("#dataset-file-input").setInputFiles({
    name: "apple-orange.zip",
    mimeType: "application/zip",
    buffer: makeYoloZip(),
  });
  await page.getByTestId("model-rtdetr").click();
  await page.getByTestId("start-verification").click();

  const list = page.getByRole("list", { name: "Verification checks" });
  await expect(list.getByText("all 4 images have no boxes: nothing to train on").first()).toBeVisible({
    timeout: 60_000,
  });
  await expect(page.getByTestId("check-hint-empty_labels")).toContainText("Add annotations");
});

// Licence card + accept flow [Slice C]. The catalog comes from the real backend;
// the accept POST is stubbed so the test is repeatable (the server-side record and
// the 409 gate are covered by backend/tests/test_licences.py), but the stub only
// answers when the browser really sends the catalog id.
test("licence card blocks Configure Next until the licence is accepted", async ({ page }) => {
  let acceptedId: string | null = null;
  await page.route("**/api/v1/licences/accept", async (route) => {
    acceptedId = (route.request().postDataJSON() as { catalog_id: string }).catalog_id;
    await route.fulfill({ json: { catalog_id: acceptedId, accepted: true } });
  });
  await page.route("**/api/v1/verifications/*", async (route) => {
    if (route.request().method() !== "GET") return route.continue();
    await route.fulfill({
      json: {
        id: "ver_mock_lic",
        status: "passed",
        can_train: true,
        checks: [],
        derived: { task_type: "object_detection", format: "yolo" },
      },
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "New training" }).click();
  await page.locator("#dataset-file-input").setInputFiles({
    name: "apple-orange.zip",
    mimeType: "application/zip",
    buffer: makeYoloZip(),
  });
  await page.getByTestId("model-rtdetr").click();
  await page.getByTestId("start-verification").click();
  await expect(page.getByText("Dataset verified successfully")).toBeVisible({ timeout: 60_000 });
  const ack = page.getByTestId("ack-warnings");
  if (await ack.isVisible()) await ack.check();
  await page.getByTestId("verify-next").click();

  await page.getByTestId("variant-rtdetr-r18vd@coco-1.0").check();
  const card = page.getByTestId("licence-card");
  await expect(card).toContainText("Apache-2.0");
  await expect(page.getByTestId("verify-chip")).toBeVisible();
  await expect(page.getByTestId("commercial-use")).toContainText("unknown");
  await expect(page.getByTestId("configure-next")).toBeDisabled();
  await expect(page.getByTestId("licence-block")).toContainText("Accept the licence");

  await page.getByTestId("accept-licence").click();
  await expect(page.getByTestId("accept-licence")).toBeChecked();
  expect(acceptedId).toBe("rtdetr-r18vd@coco-1.0");
  await expect(page.getByTestId("configure-next")).toBeEnabled();

  // A variant with no verified download is listed but cannot be used.
  await page.getByTestId("variant-rtdetr-l@1.0").check();
  await expect(page.getByTestId("not-offered-chip")).toBeVisible();
  await expect(page.getByTestId("configure-next")).toBeDisabled();
});
