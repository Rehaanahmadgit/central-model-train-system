import { expect, test } from "@playwright/test";

import { coerceConfig, parseFieldValue, type HparamSchema } from "../src/lib/hparams";
import { gpuLabel, resolveBest } from "../src/lib/runSummary";

// Fix 1: a finished run whose run.json still has the placeholder best.
test("best falls back to the evaluation report's val mAP50", () => {
  const resolved = resolveBest(
    { metric: null, value: 0, epoch: 0 },
    { val: { mAP50: 0.878 }, mAP50: 0.878 },
  );
  expect(resolved).toEqual({ metric: "val/mAP50", value: 0.878, source: "evaluation" });
});

test("a real run.json best always wins over the evaluation report", () => {
  const resolved = resolveBest(
    { metric: "val/mAP50", value: 0.42, epoch: 3 },
    { val: { mAP50: 0.9 } },
  );
  expect(resolved).toEqual({ metric: "val/mAP50", value: 0.42, epoch: 3, source: "run" });
});

test("legacy top-level mAP50 and classification accuracy are used when val is absent", () => {
  expect(resolveBest(undefined, { mAP50: 0.31 })).toEqual({
    metric: "val/mAP50",
    value: 0.31,
    source: "evaluation",
  });
  expect(resolveBest({ metric: "" }, { accuracy: 0.77 })?.metric).toBe("val/accuracy");
});

test("nothing to show stays null (no invented numbers)", () => {
  expect(resolveBest({ metric: null }, null)).toBeNull();
  expect(resolveBest({ metric: null }, {})).toBeNull();
  expect(resolveBest({ metric: "val/mAP50", value: null }, null)).toBeNull();
});

// Fix 1b: GPU percent on a finished run.
test("0% GPU is a measurement, not n/a", () => {
  expect(gpuLabel({ gpu_available: true, gpu_util: 0 }, true)).toBe("0%");
  expect(gpuLabel({ gpu_available: true, gpu_util: 64 }, true)).toBe("64%");
});

test("no GPU vs no sample are told apart", () => {
  expect(gpuLabel({ gpu_available: false, gpu_util: null }, true)).toBe("No GPU detected");
  expect(gpuLabel(null, true)).toBe("n/a (run finished)");
  expect(gpuLabel(null, false)).toBe("n/a");
  expect(gpuLabel({ gpu_available: true, gpu_util: null }, true)).toBe("n/a (run finished)");
});

// Fix 3: numeric config fields.
const SCHEMA: HparamSchema = {
  properties: {
    epochs: { type: "integer", default: 1 },
    batch_size: { type: "integer", default: 1 },
    lr: { type: "number", default: 0.0001 },
    amp: { type: "boolean", default: false },
    note: { type: "text" },
  },
};

test("numeric strings from the form are posted as numbers", () => {
  const config = coerceConfig({ epochs: "3", batch_size: "4", lr: "0.002" }, SCHEMA);
  expect(config).toEqual({ epochs: 3, batch_size: 4, lr: 0.002 });
  expect(typeof config.batch_size).toBe("number");
});

test("integers are truncated, booleans and unknown keys survive", () => {
  expect(coerceConfig({ epochs: 2.9 }, SCHEMA).epochs).toBe(2);
  expect(coerceConfig({ amp: "true" }, SCHEMA).amp).toBe(true);
  expect(coerceConfig({ custom: "keep" }, SCHEMA).custom).toBe("keep");
});

test("empty or unusable numeric fields are dropped so the default applies", () => {
  expect(coerceConfig({ batch_size: "", epochs: "abc" }, SCHEMA)).toEqual({});
});

test("input typing follows the schema, not the current value", () => {
  expect(parseFieldValue({ type: "integer" }, "4")).toBe(4);
  expect(parseFieldValue({ type: "number" }, "0.5")).toBe(0.5);
  expect(parseFieldValue({ type: "integer" }, "")).toBe("");
  expect(parseFieldValue({ type: "text" }, "4")).toBe("4");
  expect(parseFieldValue({ type: "boolean" }, "true")).toBe(true);
});
