import {
  commercialLabel,
  fmtSize,
  fmtVram,
  needsVerifyChip,
  weightsLicenceLabel,
  type CatalogVariant,
} from "../lib/licence";

interface LicenceCardProps {
  variant: CatalogVariant;
  accepted: boolean;
  busy: boolean;
  error: string | null;
  onAccept: () => void;
}

// Licence card for one pretrained variant: code + weights licence, commercial
// flag, size, VRAM, source link and an explicit "I accept" [Slice C]. Unknowns
// are shown as unknown, never filled in.
export function LicenceCard({ variant, accepted, busy, error, onAccept }: LicenceCardProps) {
  const weights = variant.weights_license;
  return (
    <div
      data-testid="licence-card"
      className="flex flex-col gap-2 rounded-lg border border-border bg-bg p-3 text-sm"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono font-semibold">{variant.catalog_id}</span>
        {!variant.offered && (
          <span data-testid="not-offered-chip" className="rounded-full bg-warning/10 px-2 py-0.5 text-xs text-warning">
            not offered yet
          </span>
        )}
        {variant.local && (
          <span className="rounded-full bg-success/10 px-2 py-0.5 text-xs text-success">cached</span>
        )}
      </div>
      <dl className="grid gap-1 sm:grid-cols-2">
        <div>
          <dt className="text-xs text-text-secondary">Code licence</dt>
          <dd data-testid="code-licence">
            {variant.code_license.spdx ?? "not found"}
            {variant.code_license.url && (
              <>
                {" "}
                <a href={variant.code_license.url} target="_blank" rel="noreferrer" className="text-accent underline">
                  terms
                </a>
              </>
            )}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-text-secondary">Weights licence</dt>
          <dd data-testid="weights-licence" className="flex items-center gap-2">
            {weightsLicenceLabel(weights)}
            {needsVerifyChip(weights) && (
              <span
                data-testid="verify-chip"
                title="The licence of the weights themselves has not been verified. Check the source before commercial use."
                className="rounded-full bg-warning/10 px-2 py-0.5 text-xs font-semibold text-warning"
              >
                VERIFY
              </span>
            )}
            {weights.url && (
              <a href={weights.url} target="_blank" rel="noreferrer" className="text-accent underline">
                source
              </a>
            )}
          </dd>
        </div>
        <div data-testid="commercial-use">{commercialLabel(variant.commercial_use)}</div>
        <div>
          {fmtSize(variant.size_bytes)} · {fmtVram(variant.min_vram_gb)}
        </div>
      </dl>
      {variant.offered && (
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            data-testid="accept-licence"
            checked={accepted}
            disabled={busy || accepted}
            onChange={(event) => event.target.checked && onAccept()}
          />
          I have read the licences above and accept them for {variant.catalog_id}
        </label>
      )}
      {error && (
        <p role="alert" className="text-xs text-danger">
          {error}
        </p>
      )}
    </div>
  );
}
