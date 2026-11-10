// Pure logic for checkpoint auto-select, the frozen-backbone guard and run-start
// failures [Slice D]. Unit-tested in e2e/pretrained-logic.spec.ts.

import type { Derived } from "../hooks/useVerificationStream";

export interface AutoSelection {
  catalogId: string | null;
  reason: string;
}

/** What the verifier picked and why; null id = nothing passed the hard filters. */
export function autoSelection(derived: Derived | null | undefined): AutoSelection {
  const id = derived?.auto_selected ?? null;
  const reason =
    derived?.auto_select_reason ??
    (id ? "recommended by the verifier" : "no recommendation available");
  return { catalogId: id, reason };
}

export type SelectionSource = "auto" | "manual" | "none";

export function selectionSource(
  selectedId: string | null,
  autoId: string | null,
  fromScratch: boolean,
): SelectionSource {
  if (fromScratch || !selectedId) return "none";
  return selectedId === autoId ? "auto" : "manual";
}

/**
 * A frozen backbone cannot learn without pretrained weights, so Start is blocked
 * (not just warned). Returns the message, or null when training can proceed.
 */
export function frozenBackboneBlock(
  hasFreezeOption: boolean,
  config: Record<string, unknown>,
  checkpointId: string | null,
  fromScratch: boolean,
): string | null {
  if (!hasFreezeOption || config.freeze_backbone !== true) return null;
  if (!fromScratch && checkpointId) return null;
  return (
    "Pretrained weights ke bina frozen backbone seekh nahi sakta: pick a pretrained " +
    "checkpoint, or turn off freeze_backbone."
  );
}

export interface RunFailure {
  reason?: string | null;
  details?: string | null;
  catalog_id?: string | null;
  can_import?: boolean | null;
}

export interface FailureView {
  title: string;
  hint: string;
  canImport: boolean;
}

const CHECKPOINT_TITLES: Record<string, string> = {
  CHECKPOINT_OFFLINE: "No internet: the checkpoint could not be downloaded",
  CHECKPOINT_SHA_MISMATCH: "Downloaded file failed the sha256 check",
  CHECKPOINT_DISK_FULL: "Not enough free disk for the checkpoint",
  CHECKPOINT_PARTIAL: "Checkpoint download stopped part-way",
  CHECKPOINT_DOWNLOAD_FAILED: "Checkpoint download failed",
  LICENCE_NOT_ACCEPTED: "Licence not accepted",
};

/** Checkpoint failures always offer "Import from file" (works offline). */
export function failureView(failure: RunFailure | null | undefined): FailureView | null {
  const reason = failure?.reason;
  if (!reason) return null;
  const title = CHECKPOINT_TITLES[reason];
  if (!title) return null;
  const canImport = reason !== "LICENCE_NOT_ACCEPTED" && failure?.can_import !== false;
  return {
    title,
    hint: failure?.details ?? "See the run log for details.",
    canImport,
  };
}

export interface Waiting {
  what?: string;
  status?: string;
  catalog_id?: string;
  percent?: number | null;
  downloaded_bytes?: number | null;
  total_bytes?: number | null;
}

/** "waiting: checkpoint" line shown on a preparing run. */
export function waitingLabel(waiting: Waiting | null | undefined): string | null {
  if (!waiting || waiting.what !== "checkpoint") return null;
  const id = waiting.catalog_id ?? "checkpoint";
  if (waiting.status === "waiting_lock") {
    return `waiting: checkpoint ${id} — another run is downloading it, this one continues when it is done`;
  }
  if (waiting.status === "downloading") {
    const pct = typeof waiting.percent === "number" ? ` ${waiting.percent.toFixed(0)}%` : "";
    return `waiting: checkpoint ${id} — downloading${pct}`;
  }
  return `waiting: checkpoint ${id} — preparing`;
}
