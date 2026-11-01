import { expect, test } from "@playwright/test";

import { LOSS_AXIS, LR_AXIS, SCORE_AXIS, axisIndexFor, isScoreMetric } from "../src/lib/chartAxes";

// A4: loss (0..15) and mAP/accuracy (0..1) must not share one axis, or the
// score curve is drawn flat. Pure logic, no browser needed.

test("mAP and accuracy use the 0-1 score axis, not the loss axis", () => {
  expect(axisIndexFor("val/mAP50")).toBe(SCORE_AXIS);
  expect(axisIndexFor("val/accuracy")).toBe(SCORE_AXIS);
  expect(axisIndexFor("train/accuracy")).toBe(SCORE_AXIS);
  expect(axisIndexFor("val/f1")).toBe(SCORE_AXIS);
  expect(axisIndexFor("val/recall")).toBe(SCORE_AXIS);
  expect(axisIndexFor("val/precision")).toBe(SCORE_AXIS);
});

test("loss stays on the primary axis", () => {
  expect(axisIndexFor("train/loss")).toBe(LOSS_AXIS);
  expect(axisIndexFor("val/loss")).toBe(LOSS_AXIS);
  expect(axisIndexFor("val/mAP50")).not.toBe(LOSS_AXIS);
});

test("learning rate keeps its own axis (it is ~1e-4)", () => {
  expect(axisIndexFor("lr")).toBe(LR_AXIS);
  expect(axisIndexFor("train/lr")).toBe(LR_AXIS);
});

test("score detection only inspects the metric name, not the split prefix", () => {
  expect(isScoreMetric("val/mAP50")).toBe(true);
  expect(isScoreMetric("train/loss")).toBe(false);
  expect(isScoreMetric("val/iou")).toBe(true);
});
