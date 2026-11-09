import { expect, test } from "@playwright/test";

import {
  commercialLabel,
  fmtSize,
  fmtVram,
  licenceGate,
  needsVerifyChip,
  weightsLicenceLabel,
  type CatalogVariant,
} from "../src/lib/licence";

function variant(overrides: Partial<CatalogVariant> = {}): CatalogVariant {
  return {
    catalog_id: "rtdetr-r18vd@coco-1.0",
    name: "rtdetr-r18vd",
    variant: "r18vd",
    model_family: "rtdetr",
    task_types: ["object_detection"],
    formats: ["yolo"],
    pretrained_on: "coco",
    source_uri: "https://example.invalid/w.pth",
    size_bytes: 81001853,
    min_vram_gb: 0,
    offered: true,
    local: false,
    licence_accepted: false,
    code_license: { spdx: "Apache-2.0", url: null },
    weights_license: { spdx: null, url: null, verified: false },
    commercial_use: "unknown",
    notes: "",
    ...overrides,
  };
}

test("unknown values are shown as unknown, never filled in", () => {
  expect(weightsLicenceLabel({ spdx: null, url: null, verified: false })).toBe("not found");
  expect(commercialLabel("unknown")).toBe("Commercial use: unknown");
  expect(commercialLabel(true)).toBe("Commercial use: allowed");
  expect(commercialLabel(false)).toBe("Commercial use: not allowed");
  expect(fmtVram(0)).toBe("VRAM not measured");
  expect(fmtSize(81001853)).toBe("77.2 MB");
  expect(fmtSize(0)).toBe("size unknown");
});

test("VERIFY chip until the weights licence is verified", () => {
  expect(needsVerifyChip({ spdx: null, url: null, verified: false })).toBe(true);
  expect(needsVerifyChip({ spdx: "MIT", url: null, verified: true })).toBe(false);
});

test("Next/Start stay blocked until the licence is accepted", () => {
  const id = "rtdetr-r18vd@coco-1.0";
  const blocked = licenceGate(id, false, variant(), false);
  expect(blocked.blocked).toBe(true);
  expect(blocked.message).toContain("Accept the licence");
  expect(licenceGate(id, false, variant({ licence_accepted: true }), false).blocked).toBe(false);
  expect(licenceGate(id, false, variant(), true).blocked).toBe(false);
});

test("not-offered and unknown catalog ids are blocked with a clear reason", () => {
  const notOffered = licenceGate("rtdetr-l@1.0", false, variant({ catalog_id: "rtdetr-l@1.0", offered: false }), false);
  expect(notOffered.blocked).toBe(true);
  expect(notOffered.message).toContain("not offered yet");
  const unknown = licenceGate("nope@1", false, null, false);
  expect(unknown.blocked).toBe(true);
  expect(unknown.message).toContain("not in the model catalog");
});

test("training from scratch or without a checkpoint never needs a licence", () => {
  expect(licenceGate("rtdetr-r18vd@coco-1.0", true, variant(), false).blocked).toBe(false);
  expect(licenceGate(null, false, null, false).blocked).toBe(false);
});
