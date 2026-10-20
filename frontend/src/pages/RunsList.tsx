import { useEffect, useState } from "react";
import { apiFetch } from "../lib/api";

interface RunsListProps {
  onOpen: (runId: string) => void;
}

interface RunSummary {
  id: string;
  name: string;
  state: string;
  model_family: string;
  queued_at: string | null;
  started_at: string | null;
}

const STATE_COLORS: Record<string, string> = {
  queued: "bg-text-secondary",
  preparing: "bg-warning",
  running: "bg-success",
  paused: "bg-warning",
  evaluating: "bg-info",
  completed: "bg-success",
  failed: "bg-danger",
  stopped: "bg-text-secondary",
  interrupted: "bg-warning",
};

// Runs list with live-looking state badges; click a run to open the tracker
// [doc 09 §3.5]. Reload restores full state via snapshot replay on the detail page.
export function RunsList({ onOpen }: RunsListProps) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);

  useEffect(() => {
    let disposed = false;
    apiFetch<{ items?: RunSummary[] }>("/api/v1/runs")
      .then((body) => {
        if (!disposed && body?.items) setRuns(body.items);
      })
      .catch(() => {
        if (!disposed) setRuns([]);
      });
    return () => {
      disposed = true;
    };
  }, []);

  if (runs === null) {
    return (
      <div className="space-y-2" aria-label="Loading runs">
        {[0, 1, 2].map((index) => (
          <div key={index} className="h-16 animate-pulse rounded-lg border border-border bg-surface" />
        ))}
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <section className="flex flex-col items-center gap-4 rounded-lg border border-dashed border-border bg-surface px-8 py-16 text-center">
        <h2 className="text-xl font-semibold">No runs yet</h2>
        <p className="max-w-md text-sm text-text-secondary">
          Verify a dataset and start training to see it here with a live status badge.
        </p>
      </section>
    );
  }

  return (
    <ul aria-label="Training runs" className="flex flex-col gap-2">
      {runs.map((run) => (
        <li key={run.id}>
          <button
            type="button"
            onClick={() => onOpen(run.id)}
            className="flex w-full items-center gap-4 rounded-lg border border-border bg-surface p-4 text-left hover:border-accent"
          >
            <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${STATE_COLORS[run.state] ?? "bg-text-secondary"}`} aria-hidden="true" />
            <span className="min-w-0">
              <span className="block truncate text-sm font-semibold">{run.name}</span>
              <span className="block font-mono text-xs text-text-secondary">{run.id}</span>
            </span>
            <span className="ml-auto flex flex-col items-end gap-1">
              <span className="text-xs font-semibold">{run.state}</span>
              <span className="text-xs text-text-secondary">{run.model_family}</span>
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}
