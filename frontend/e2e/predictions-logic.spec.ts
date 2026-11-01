import { expect, test } from "@playwright/test";

import { boxRect, classIdOf, scoreOf } from "../src/lib/predictions";

// B3/B4: overlay geometry. GT rows are [class, cx, cy, w, h] (5 values) and
// prediction rows are [class, score, cx, cy, w, h] (6 values). Reading the
// box from the front produced NaN in the browser, so both shapes are pinned.

const GT_BOX = [0, 0.5, 0.5, 0.2, 0.4];
const PRED_BOX = [0, 0.87, 0.5, 0.5, 0.2, 0.4];

test("GT and prediction rows map to the same pixel rect", () => {
  expect(boxRect(GT_BOX, 320, 240)).toEqual({ x: 128, y: 72, w: 64, h: 96 });
  expect(boxRect(PRED_BOX, 320, 240)).toEqual({ x: 128, y: 72, w: 64, h: 96 });
});

test("no attribute ever becomes NaN", () => {
  for (const box of [GT_BOX, PRED_BOX]) {
    const rect = boxRect(box, 640, 480);
    for (const value of [rect.x, rect.y, rect.w, rect.h]) {
      expect(Number.isNaN(value)).toBe(false);
    }
  }
});

test("normalized coordinates scale with the real image size", () => {
  const small = boxRect(GT_BOX, 100, 100);
  const large = boxRect(GT_BOX, 400, 400);
  expect(large.x).toBe(small.x * 4);
  expect(large.h).toBe(small.h * 4);
  expect(large.w).toBe(small.w * 4);
});

test("class id and score are read from the right slots", () => {
  expect(classIdOf(GT_BOX)).toBe(0);
  expect(classIdOf([1, 0.2, 0.2, 0.1, 0.1])).toBe(1);
  expect(scoreOf(PRED_BOX)).toBe(0.87);
  expect(scoreOf(GT_BOX)).toBe(1); // GT rows carry no score
});
