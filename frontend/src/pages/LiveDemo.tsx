import { useEffect, useState } from "react";
import { ConnectionIndicator } from "../components/ConnectionIndicator";
import { MetricChart } from "../components/MetricChart";
import { useRunStream } from "../hooks/useRunStream";

interface LiveDemoProps {
  theme: "light" | "dark";
}

interface SystemInfo {
  running_jobs: { id: string; name?: string; state?: string }[];
  queued_jobs: { id: string; name?: string; state?: string }[];
}

// Live demo page: streams the hello-world fake run [doc 12 Phase 0 M0].
// Finds the run id from GET /api/v1/system [doc 05 §7].
export function LiveDemo({ theme }: LiveDemoProps) {
  const [runId, setRunId] = useState<string | null>(null);
  const stream = useRunStream(runId);

  useEffect(() => {
    let disposed = false;
    fetch("/api/v1/system")
      .then((response) => (response.ok ? (response.json() as Promise<SystemInfo>) : null))
      .then((info) => {
        if (disposed || !info) return;
        const job = info.running_jobs[0] ?? info.queued_jobs[0];
        setRunId(job ? job.id : null);
      })
      .catch(() => {
        if (!disposed) setRunId(null);
      });
    return () => {
      disposed = true;
    };
  }, []);

  if (!runId) {
    return (
      <section className="flex flex-col items-center gap-4 rounded-lg border border-dashed border-border bg-surface px-8 py-16 text-center">
        <h2 className="text-xl font-semibold">No run streaming</h2>
        <p className="max-w-md text-sm text-text-secondary">
          Start the backend with the demo flag: <code className="font-mono">TRAINHUB_DEMO_FAKE_RUN=true</code>.
          The hello-world run streams fake metrics into the chart below [doc 12 Phase 0].
        </p>
        <ConnectionIndicator status={stream.status} />
      </section>
    );
  }

  return (
    <section className="flex flex-col gap-6">
      <header className="flex items-center gap-4">
        <h2 className="text-xl font-semibold">Live run · {runId}</h2>
        <ConnectionIndicator status={stream.status} />
        <span className="text-sm text-text-secondary">
          state: <span className="font-mono">{stream.runState ?? "unknown"}</span>
        </span>
      </header>
      <MetricChart metrics={stream.metrics} theme={theme} />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="rounded-lg border border-border bg-surface p-4">
          <h3 className="mb-2 text-sm font-semibold">Live log</h3>
          <ul
            aria-label="Run log"
            className="max-h-64 space-y-1 overflow-auto font-mono text-xs text-text-secondary"
          >
            {stream.logs.map((line, index) => (
              <li key={index}>
                <span className="text-text">{line.level}</span> {line.message}
              </li>
            ))}
          </ul>
        </div>
        <div className="rounded-lg border border-border bg-surface p-4">
          <h3 className="mb-2 text-sm font-semibold">Events timeline</h3>
          <ul aria-label="Run events" className="max-h-64 space-y-1 overflow-auto text-xs text-text-secondary">
            {stream.events.map((event, index) => (
              <li key={index} className="font-mono">
                {event.ts.slice(11, 19)} · {event.name}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
