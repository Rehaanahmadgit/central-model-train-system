import { expect, test } from "@playwright/test";

import {
  autoSelection,
  failureView,
  frozenBackboneBlock,
  selectionSource,
  waitingLabel,
} from "../src/lib/pretrained";

test("autoSelection exposes id + reason, or an honest 'none'", () => {
  expect(autoSelection({ auto_selected: "a@1", auto_select_reason: "because" } as never)).toEqual({
    catalogId: "a@1",
    reason: "because",
  });
  const none = autoSelection({ auto_selected: null, auto_select_reason: "no checkpoint passed" } as never);
  expect(none.catalogId).toBeNull();
  expect(none.reason).toBe("no checkpoint passed");
  expect(autoSelection(null).catalogId).toBeNull();
});

test("selectionSource tells auto from manual override", () => {
  expect(selectionSource("a@1", "a@1", false)).toBe("auto");
  expect(selectionSource("b@1", "a@1", false)).toBe("manual");
  expect(selectionSource(null, "a@1", false)).toBe("none");
  expect(selectionSource("a@1", "a@1", true)).toBe("none");
});

test("frozen backbone is blocked without a checkpoint, scratch or empty", () => {
  const frozen = { freeze_backbone: true };
  expect(frozenBackboneBlock(true, frozen, null, false)).toContain("frozen backbone");
  expect(frozenBackboneBlock(true, frozen, "a@1", true)).toContain("frozen backbone"); // scratch
  expect(frozenBackboneBlock(true, frozen, "", false)).not.toBeNull();
  expect(frozenBackboneBlock(true, frozen, "a@1", false)).toBeNull();
  expect(frozenBackboneBlock(true, { freeze_backbone: false }, null, true)).toBeNull();
  expect(frozenBackboneBlock(false, frozen, null, true)).toBeNull(); // family has no such option
});

test("every checkpoint failure offers Import from file except an unaccepted licence", () => {
  for (const reason of [
    "CHECKPOINT_OFFLINE",
    "CHECKPOINT_SHA_MISMATCH",
    "CHECKPOINT_DISK_FULL",
    "CHECKPOINT_PARTIAL",
    "CHECKPOINT_DOWNLOAD_FAILED",
  ]) {
    const view = failureView({ reason, details: "msg", catalog_id: "a@1", can_import: true });
    expect(view?.canImport, reason).toBe(true);
    expect(view?.hint).toBe("msg");
  }
  expect(failureView({ reason: "LICENCE_NOT_ACCEPTED", details: "x" })?.canImport).toBe(false);
  expect(failureView({ reason: "OOM" })).toBeNull(); // not a checkpoint failure
  expect(failureView(null)).toBeNull();
});

test("waitingLabel covers downloading, lock wait and preparing", () => {
  expect(waitingLabel({ what: "checkpoint", status: "downloading", catalog_id: "a@1", percent: 41.6 }))
    .toBe("waiting: checkpoint a@1 — downloading 42%");
  expect(waitingLabel({ what: "checkpoint", status: "waiting_lock", catalog_id: "a@1" }))
    .toContain("another run is downloading");
  expect(waitingLabel({ what: "checkpoint", status: "resolving", catalog_id: "a@1" }))
    .toContain("preparing");
  expect(waitingLabel(null)).toBeNull();
});
