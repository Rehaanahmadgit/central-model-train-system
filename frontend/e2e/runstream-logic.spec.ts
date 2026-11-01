import { expect, test } from "@playwright/test";

import { isDuplicateReplay } from "../src/lib/runStreamLogic";

// Pure logic tests for the run WebSocket replay dedupe [doc 05 §8]. No
// browser or backend needed: these assertions run in the Playwright runner.

test("metric/event/state with seq <= snapshot are duplicate replays", () => {
  expect(isDuplicateReplay("metric", 10, 20)).toBe(true);
  expect(isDuplicateReplay("event", 20, 20)).toBe(true);
  expect(isDuplicateReplay("state", 0, 20)).toBe(true);
});

test("metric/event/state with seq > snapshot are fresh", () => {
  expect(isDuplicateReplay("metric", 21, 20)).toBe(false);
  expect(isDuplicateReplay("event", 21, 20)).toBe(false);
  expect(isDuplicateReplay("state", 21, 20)).toBe(false);
});

test("log and system are never duplicates, whatever the seq", () => {
  expect(isDuplicateReplay("log", 5, 20)).toBe(false);
  expect(isDuplicateReplay("system", 50, 20)).toBe(false);
});

test("unknown types and missing seq/snapshot are never duplicates", () => {
  expect(isDuplicateReplay("heartbeat", 10, 20)).toBe(false);
  expect(isDuplicateReplay("result", 10, 20)).toBe(false);
  expect(isDuplicateReplay("metric", null, 20)).toBe(false);
  expect(isDuplicateReplay("metric", undefined, 20)).toBe(false);
  expect(isDuplicateReplay("metric", 10, null)).toBe(false);
  expect(isDuplicateReplay("metric", 10, undefined)).toBe(false);
});
