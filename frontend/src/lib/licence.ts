// Pure licence/variant logic for the Model Zoo card [Slice C]; unit-tested in
// e2e/licence-logic.spec.ts. Nothing here guesses a licence: unknown stays unknown.

export interface LicenceRef {
  spdx: string | null;
  url: string | null;
}

export interface WeightsLicence extends LicenceRef {
  verified: boolean;
}

export interface CatalogVariant {
  catalog_id: string;
  name: string;
  variant: string;
  model_family: string;
  task_types: string[];
  formats: string[];
  pretrained_on: string;
  source_uri: string;
  size_bytes: number;
  min_vram_gb: number;
  offered: boolean;
  local: boolean;
  licence_accepted: boolean;
  code_license: LicenceRef;
  weights_license: WeightsLicence;
  commercial_use: boolean | "unknown";
  notes: string;
}

export function fmtSize(bytes: number): string {
  if (!bytes || bytes <= 0) return "size unknown";
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

export function fmtVram(gb: number): string {
  return gb > 0 ? `~${gb} GB VRAM` : "VRAM not measured";
}

export function commercialLabel(value: boolean | "unknown"): string {
  if (value === true) return "Commercial use: allowed";
  if (value === false) return "Commercial use: not allowed";
  return "Commercial use: unknown";
}

export function weightsLicenceLabel(weights: WeightsLicence): string {
  return weights.spdx ?? "not found";
}

/** The "VERIFY" chip shows until someone has confirmed the weights licence. */
export function needsVerifyChip(weights: WeightsLicence): boolean {
  return !weights.verified;
}

export interface LicenceGate {
  blocked: boolean;
  message: string | null;
}

/**
 * Should Next / Start be blocked for the chosen checkpoint?
 * `variant` is null when the id is not in the catalog (or the catalog failed to load).
 */
export function licenceGate(
  catalogId: string | null,
  fromScratch: boolean,
  variant: CatalogVariant | null,
  acceptedNow: boolean,
): LicenceGate {
  if (fromScratch || !catalogId) return { blocked: false, message: null };
  if (!variant) {
    return {
      blocked: true,
      message: `${catalogId} is not in the model catalog. Pick a variant from the list or train from scratch.`,
    };
  }
  if (!variant.offered) {
    return {
      blocked: true,
      message: `${variant.catalog_id} is not offered yet (no verified download). Pick another variant, import the file offline, or train from scratch.`,
    };
  }
  if (!variant.licence_accepted && !acceptedNow) {
    return {
      blocked: true,
      message: `Accept the licence for ${variant.catalog_id} below to continue.`,
    };
  }
  return { blocked: false, message: null };
}
