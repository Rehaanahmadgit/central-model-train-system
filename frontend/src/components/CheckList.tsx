import type { CheckItem } from "../hooks/useVerificationStream";

interface CheckListProps {
  checks: CheckItem[];
}

const STATUS_META: Record<string, { icon: string; label: string; classes: string }> = {
  running: { icon: "◌", label: "Running", classes: "text-info" },
  pass: { icon: "✓", label: "Passed", classes: "text-success" },
  warn: { icon: "⚠", label: "Warning", classes: "text-warning" },
  fail: { icon: "✗", label: "Failed", classes: "text-danger" },
  skipped: { icon: "–", label: "Skipped", classes: "text-text-secondary" },
};

// Streaming checklist: spinner → ✓ / ⚠ / ✗ with expandable details, fix hints
// and offending samples [doc 06 §8, doc 09 §4]. Never colour-only: icon + text.
export function CheckList({ checks }: CheckListProps) {
  return (
    <ul aria-label="Verification checks" className="flex flex-col gap-2">
      {checks.map((check) => {
        const meta = STATUS_META[check.status] ?? STATUS_META.running;
        return (
          <li key={check.name} className="rounded-lg border border-border bg-surface p-3">
            <details>
              <summary className="flex cursor-pointer items-center gap-3 text-sm">
                <span
                  aria-hidden="true"
                  className={`font-mono ${check.status === "running" ? "animate-spin inline-block" : ""}`}
                >
                  {meta.icon}
                </span>
                <span className="font-mono font-medium">{check.name}</span>
                {/* Fail/warn message on one line, no click needed [doc 06 §8]. */}
                {(check.status === "fail" || check.status === "warn") && check.message && (
                  <span
                    className="min-w-0 flex-1 truncate text-xs text-text-secondary"
                    title={check.message}
                  >
                    {check.message}
                  </span>
                )}
                <span className={`ml-auto shrink-0 text-xs font-semibold ${meta.classes}`}>
                  {meta.label}
                </span>
              </summary>
              <div className="mt-2 space-y-2 pl-6 text-xs text-text-secondary">
                {check.message && <p>{check.message}</p>}
                {check.fix_hint && (
                  <p>
                    <span className="font-semibold text-text">Fix: </span>
                    {check.fix_hint}
                  </p>
                )}
                {check.samples.length > 0 && (
                  <ul className="list-disc pl-4 font-mono">
                    {check.samples.slice(0, 10).map((sample) => (
                      <li key={sample}>{sample}</li>
                    ))}
                  </ul>
                )}
              </div>
            </details>
          </li>
        );
      })}
      {checks.length === 0 && (
        <li className="rounded-lg border border-dashed border-border p-4 text-sm text-text-secondary">
          No checks yet — press Start Verification.
        </li>
      )}
    </ul>
  );
}
