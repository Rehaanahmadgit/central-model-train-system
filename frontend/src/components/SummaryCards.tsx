/**
 * Summary cards for a finished verification [doc 05 §3, doc 06 §8, doc 09 §3].
 * Every card is text-first: a label, a value and a unit, never colour alone.
 */

import type { Derived } from "../hooks/useVerificationStream";

interface SummaryCardsProps {
  derived: Derived | null;
}

function Card({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="text-xs font-medium uppercase tracking-wide text-text-secondary">{label}</div>
      <div className="mt-1 text-lg font-semibold">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-text-secondary">{hint}</div>}
    </div>
  );
}

export function SummaryCards({ derived }: SummaryCardsProps) {
  if (!derived) return null;

  const classes = derived.num_classes ?? derived.classes?.length ?? 0;
  const samples = derived.num_images ?? 0;
  const splits = derived.num_samples ?? {};
  const splitText = Object.entries(splits)
    .filter(([, count]) => count > 0)
    .map(([name, count]) => `${name} ${count}`)
    .join(" · ");
  const imageStats = derived.image_stats ?? {};
  const maxWh = imageStats.max_wh;
  const resolution = maxWh ? `≤ ${maxWh[0]}×${maxWh[1]} px` : "n/a";
  const counts = derived.class_counts ?? {};
  const sorted = Object.entries(counts)
    .map(([name, count]) => [name, Number(count)] as const)
    .sort((a, b) => b[1] - a[1]);
  const total = sorted.reduce((sum, [, count]) => sum + count, 0) || 1;
  // Class keys are names for folder/CSV datasets and numeric ids for YOLO/COCO.
  const labelOf = (key: string) => {
    if (!/^\d+$/.test(key)) return key;
    return derived.classes?.[Number(key)] ?? key;
  };
  const hparams = derived.recommended_hparams ?? {};

  return (
    <section aria-label="Dataset summary" className="flex flex-col gap-3">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Card label="Classes" value={String(classes)} hint={derived.classes?.slice(0, 4).join(", ")} />
        <Card label="Samples" value={String(samples)} hint={derived.format ?? undefined} />
        <Card label="Splits" value={splitText || "none"} hint="train/val/test detected from paths" />
        <Card label="Resolution" value={resolution} hint={`median ${imageStats.median_w ?? "?"} px wide`} />
      </div>

      {derived.task_type && (
        <p className="text-xs text-text-secondary">
          <span className="font-semibold text-text">Task:</span> {derived.task_type}
          {derived.task_detection_reason ? ` — ${derived.task_detection_reason}` : ""}
        </p>
      )}

      {sorted.length > 0 && (
        <div className="rounded-lg border border-border bg-surface p-3">
          <div className="text-xs font-medium uppercase tracking-wide text-text-secondary">
            Class distribution
          </div>
          <ul className="mt-2 flex flex-col gap-1">
            {sorted.slice(0, 8).map(([name, count]) => {
              const pct = Math.round((count / total) * 100);
              return (
                <li key={name} className="flex items-center gap-2 text-xs">
                  <span className="w-24 shrink-0 truncate font-mono">{labelOf(name)}</span>
                  <span
                    aria-hidden="true"
                    className="h-2 rounded-sm bg-accent"
                    style={{ width: `${Math.max(pct, 1)}%` }}
                  />
                  <span className="ml-auto whitespace-nowrap text-text-secondary">
                    {count} ({pct}%)
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      <div className="rounded-lg border border-border bg-surface p-3">
        <div className="text-xs font-medium uppercase tracking-wide text-text-secondary">
          Recommended settings
        </div>
        <p className="mt-1 text-sm">
          {Object.entries(hparams)
            .filter(([key]) => !["strategy", "dataset_size_bucket", "note"].includes(key))
            .map(([key, value]) => `${key}=${String(value)}`)
            .join(" · ")}
        </p>
        {typeof hparams.strategy === "string" && (
          <p className="mt-1 text-xs text-text-secondary">Fine-tune strategy: {hparams.strategy}</p>
        )}
      </div>
    </section>
  );
}
