/**
 * "Recommended checkpoint" card [doc 06 §8, doc 07 §6].
 * Always states WHY, and whether it is cached locally or must be downloaded once.
 */

import type { Derived } from "../hooks/useVerificationStream";

interface Props {
  derived: Derived | null;
}

function formatSize(bytes: number | null | undefined): string {
  if (!bytes) return "unknown size";
  const mb = bytes / (1024 * 1024);
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(0)} MB`;
}

export function RecommendedCheckpoint({ derived }: Props) {
  if (!derived) return null;
  const candidates = derived.recommended_checkpoints ?? [];
  const best = candidates.find((item) => item.catalog_id === derived.auto_selected) ?? candidates[0];
  const autoPicked = Boolean(derived.auto_selected);

  if (!best) {
    return (
      <div className="rounded-lg border border-dashed border-border bg-surface p-4 text-sm text-text-secondary">
        <span className="font-medium text-text">Checkpoint:</span> none recommended yet.
        {derived.checkpoint_catalog_configured === false
          ? " The checkpoint catalog is empty."
          : " Catalog entries still have placeholder sha256 values, so they cannot be downloaded — fill them in catalog.yaml."}{" "}
        Training can also start from scratch.
      </div>
    );
  }

  if (!autoPicked) {
    return (
      <div data-testid="no-auto-checkpoint" className="rounded-lg border border-dashed border-border bg-surface p-4 text-sm text-text-secondary">
        <span className="font-medium text-text">Checkpoint:</span> none auto-selected.{" "}
        {derived.auto_select_reason}. Pick one yourself in Configure, import weights from a file,
        or train from scratch.
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="text-sm font-semibold">Auto-selected checkpoint</span>
        <code className="rounded bg-bg px-1.5 py-0.5 font-mono text-xs">{best.catalog_id}</code>
        {best.local ? (
          <span className="text-xs font-semibold text-success">✓ cached locally</span>
        ) : (
          <span className="text-xs font-semibold text-info">
            will download {formatSize(best.size_bytes)} once
          </span>
        )}
      </div>
      <p className="mt-2 text-xs text-text-secondary">
        <span className="font-semibold text-text">Kyun chuna: </span>
        <span data-testid="auto-select-reason">{derived.auto_select_reason ?? best.reason}</span>
      </p>
      <p className="mt-1 text-xs text-text-secondary">
        {best.downloadable
          ? "sha256 is verified — safe to load."
          : "sha256 is still a placeholder, so the resolver will refuse to download it (by design)."}
        {best.license ? ` License: ${best.license}.` : ""}
      </p>
      {candidates.length > 1 && (
        <details className="mt-2">
          <summary className="cursor-pointer text-xs text-text-secondary">
            {candidates.length - 1} other candidate(s)
          </summary>
          <ul className="mt-1 space-y-1 text-xs text-text-secondary">
            {candidates.slice(1, 5).map((item) => (
              <li key={item.catalog_id}>
                <code className="font-mono">{item.catalog_id}</code> —{" "}
                {item.eligible === false && item.rejected?.length
                  ? `not auto-selected: ${item.rejected.join(", ")}`
                  : item.reason}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
