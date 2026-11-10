import { expect, test } from "@playwright/test";

import { deviceNotice, presetForDevice } from "../src/lib/device";

test("GPU present: auto → GPU, full preset", () => {
  const gpu = { available: true, name: "RTX 2050", memory_gb: 4 };
  const notice = deviceNotice(gpu, true, false);
  expect(notice.kind).toBe("gpu");
  expect(notice.text).toContain("RTX 2050");
  expect(deviceNotice(gpu, true, true).text).toContain("turn it off");
  expect(presetForDevice(gpu)).toBe("full");
});

test("no GPU: CPU + clear warning (needs-GPU model mentions --gpus all)", () => {
  const none = { available: false };
  const heavy = deviceNotice(none, true, true);
  expect(heavy.kind).toBe("cpu-warning");
  expect(heavy.text).toContain("No GPU detected");
  expect(heavy.text).toContain("--gpus all");
  expect(deviceNotice(none, false, true).text).not.toContain("--gpus");
  expect(presetForDevice(none)).toBe("dev");
  expect(presetForDevice(null)).toBe("dev");
});

test("unknown GPU status never claims a GPU", () => {
  expect(deviceNotice(null, true, true).kind).toBe("unknown");
});
