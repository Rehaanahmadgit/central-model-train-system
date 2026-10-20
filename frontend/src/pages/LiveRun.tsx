import { useEffect, useMemo, useState } from "react";
import { apiFetch } from "../lib/api";
import { ConnectionIndicator } from "../components/ConnectionIndicator";
import { MetricChart } from "../components/MetricChart";
import { useRunStream } from "../hooks/useRunStream";

interface LiveRunProps {
  runId: string;
  theme: "light" | "dark";
  onBack: () => void;
  onNavigate: (runId: string) => void;
}

interface RunInfo {
  id: string;
  name: string;
  state: string;
  model_family: string;
  task_type: string | null;
  dataset_hash: string;
  checkpoint: { catalog_id?: string | null; sha256?: string | null; source?: string | null };
  config: Record<string, unknown>;
  seed: number;
  env: Record<string, string | null>;
  queued_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  best: { metric?: string | null; value?: number; epoch?: number };
  failure?: { reason?: string | null; details?: string | null } | null;
}

const FAILURE_HINTS: Record<string, string> = {
  WORKER_LOST: "The training process stopped responding or was killed. Check the container's memory/CPU limits, then retry or resume.",
  OOM: "Out of memory. Reduce the batch size (or imgsz) in Configure and retry.",
  NAN_LOSS: "Loss became NaN/Inf. Lower the learning rate, check the data for corrupt values, then retry.",
  DATA_ERROR: "The dataset or configuration is invalid. Read the message, fix it, and re-run verification.",
  UNKNOWN: "An unexpected error occurred. See the run log for the traceback.",
};

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

function fmtElapsed(seconds: number): string {
  const whole = Math.floor(seconds);
  const h = Math.floor(whole / 3600);
  const m = Math.floor((whole % 3600) / 60);
  const s = whole % 60;
  return h > 0 ? `${h}h ${m}m ${s}s` : m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function StatTile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="text-xs font-medium uppercase tracking-wide text-text-secondary">{label}</div>
      <div className="mt-1 truncate text-lg font-semibold" title={value}>{value}</div>
      {hint && <div className="mt-0.5 text-xs text-text-secondary">{hint}</div>}
    </div>
  );
}

// Live Run page: audit-grade tracker [doc 09 §3]. State is restored on reload
// by the snapshot + last_seq resume in useRunStream [doc 05 §8].
export function LiveRun({ runId, theme, onBack, onNavigate }: LiveRunProps) {
  const stream = useRunStream(runId);
  const [info, setInfo] = useState<RunInfo | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now());
  const [logQuery, setLogQuery] = useState("");
  const [logLevel, setLogLevel] = useState("all");

  // Static run info (run.json): params, checkpoint, seed, env, timestamps.
  useEffect(() => {
    let disposed = false;
    apiFetch<RunInfo>(`/api/v1/runs/${runId}`)
      .then((body) => {
        if (!disposed) setInfo(body);
      })
      .catch(() => {
        if (!disposed) setLoadError("Could not load run info.");
      });
    return () => {
      disposed = true;
    };
  }, [runId]);

  // Ticking clock for elapsed/ETA while a run is live.
  useEffect(() => {
    const state = stream.runState ?? info?.state;
    const live = state && !["completed", "failed", "stopped", "interrupted"].includes(state);
    if (!live) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [stream.runState, info?.state]);

  const state = stream.runState ?? info?.state ?? "unknown";
  const startedAt = info?.started_at ? Date.parse(info.started_at) : null;
  const finishedAt = info?.finished_at ? Date.parse(info.finished_at) : null;
  const elapsedSeconds = startedAt
    ? Math.max(0, ((finishedAt ?? now) - startedAt) / 1000)
    : 0;

  const totalEpochs = useMemo(() => {
    const value = info?.config?.epochs;
    return typeof value === "number" && value > 0 ? value : null;
  }, [info?.config]);
  const currentEpoch = useMemo(() => {
    let maxEpoch = 0;
    for (const point of stream.metrics) {
      if (typeof point.epoch === "number" && point.epoch > maxEpoch) maxEpoch = point.epoch;
    }
    return maxEpoch;
  }, [stream.metrics]);
  const progressPct =
    totalEpochs && totalEpochs > 0
      ? Math.min(100, Math.round((currentEpoch / totalEpochs) * 100))
      : null;
  const etaSeconds =
    currentEpoch > 0 && totalEpochs && elapsedSeconds > 0
      ? (elapsedSeconds / currentEpoch) * (totalEpochs - currentEpoch)
      : null;

  const filteredLogs = useMemo(
    () =>
      stream.logs.filter((line) => {
        if (logLevel !== "all" && line.level !== logLevel) return false;
        if (logQuery && !line.message.toLowerCase().includes(logQuery.toLowerCase())) return false;
        return true;
      }),
    [stream.logs, logQuery, logLevel],
  );

  async function control(action: "pause" | "resume" | "stop"): Promise<void> {
    await apiFetch<unknown>(`/api/v1/runs/${runId}/${action}`, { method: "POST" });
  }

  async function resumeInterrupted(): Promise<void> {
    setLoadError(null);
    try {
      const retried = await apiFetch<{ id: string }>(`/api/v1/runs/${runId}/retry`, {
        method: "POST",
        body: JSON.stringify({ resume_from_last: true }),
      });
      onNavigate(retried.id);
    } catch (exc) {
      setLoadError(exc instanceof Error ? exc.message : "Could not resume the run.");
    }
  }

  const live = ["queued", "preparing", "running", "paused", "evaluating"].includes(state);

  return (
    <section className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" onClick={onBack} className="text-sm text-text-secondary hover:text-text">
          ← Runs
        </button>
        <h2 className="text-xl font-semibold">{info?.name ?? "Run"} · <span className="font-mono text-base">{runId}</span></h2>
        <span data-testid="run-state" className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold text-white ${STATE_COLORS[state] ?? "bg-text-secondary"}`}>
          {state}
        </span>
        <ConnectionIndicator status={stream.status} />
        <span className="ml-auto flex gap-2">
          {state === "interrupted" && (
            <button
              type="button"
              onClick={() => void resumeInterrupted()}
              className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-on-accent hover:bg-accent-hover"
            >
              Resume
            </button>
          )}
          {live && (
            <>
              <button
                type="button"
                onClick={() => void control(state === "paused" ? "resume" : "pause")}
                className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-bg"
              >
                {state === "paused" ? "Resume" : "Pause"}
              </button>
              <button
                type="button"
                onClick={() => void control("stop")}
                className="rounded-md border border-danger px-3 py-1.5 text-sm text-danger hover:bg-bg"
              >
                Stop
              </button>
            </>
          )}
        </span>
      </div>

      {loadError && <p role="alert" className="rounded-md border border-danger bg-surface p-3 text-sm text-danger">{loadError}</p>}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatTile label="Elapsed" value={fmtElapsed(elapsedSeconds)} />
        <StatTile label="ETA" value={etaSeconds ? fmtElapsed(etaSeconds) : "—"} hint={totalEpochs ? `epoch ${currentEpoch}/${totalEpochs}` : undefined} />
        <StatTile label="Progress" value={progressPct !== null ? `${progressPct}%` : "—"} />
        <StatTile label="Best" value={info?.best?.metric ? `${info.best.metric} ${info?.best?.value ?? ""}` : "—"} hint={info?.best?.epoch ? `epoch ${info.best.epoch}` : undefined} />
      </div>

      {progressPct !== null && (
        <div role="progressbar" aria-valuenow={progressPct} aria-valuemin={0} aria-valuemax={100} className="h-2 w-full overflow-hidden rounded-full bg-bg">
          <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${progressPct}%` }} />
        </div>
      )}

      <MetricChart metrics={stream.metrics} theme={theme} />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="rounded-lg border border-border bg-surface p-4">
          <h3 className="mb-2 text-sm font-semibold">System</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between"><span className="text-text-secondary">GPU</span><span>{stream.system?.gpu_util != null ? `${stream.system.gpu_util}%` : "No GPU detected"}</span></div>
            <div className="flex justify-between"><span className="text-text-secondary">CPU</span><span>{stream.system?.cpu != null ? `${stream.system.cpu}%` : "—"}</span></div>
            <div className="flex justify-between"><span className="text-text-secondary">RAM</span><span>{stream.system?.ram_mb != null ? `${Math.round(stream.system.ram_mb)} MB` : "—"}</span></div>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-surface p-4">
          <h3 className="mb-2 text-sm font-semibold">Events</h3>
          <ul aria-label="Run events" className="max-h-64 space-y-1 overflow-auto text-xs text-text-secondary">
            {stream.events.length === 0 && <li>No events yet.</li>}
            {stream.events.map((event, index) => (
              <li key={index} className="font-mono">
                {event.ts.slice(11, 19)} · {event.name}
              </li>
            ))}
          </ul>
        </div>

        <div className="rounded-lg border border-border bg-surface p-4">
          <h3 className="mb-2 text-sm font-semibold">Run info</h3>
          {!info ? (
            <div className="animate-pulse space-y-2" aria-label="Loading run info">
              <div className="h-4 w-full rounded bg-bg" />
              <div className="h-4 w-3/4 rounded bg-bg" />
              <div className="h-4 w-1/2 rounded bg-bg" />
            </div>
          ) : (
            <dl className="space-y-1 text-xs">
              {[
                ["Model", info.model_family],
                ["Task", info.task_type ?? "—"],
                ["Seed", String(info.seed)],
                ["Dataset", `${info.dataset_hash.slice(0, 12)}…`],
                ["Checkpoint", info.checkpoint?.catalog_id ?? "from scratch"],
                ["sha256", info.checkpoint?.sha256 ? `${info.checkpoint.sha256.slice(0, 16)}…` : "—"],
                ["Source", info.checkpoint?.source ?? "—"],
                ["Python", info.env?.python ?? "—"],
                ["Torch", info.env?.torch ?? "—"],
                ["CUDA", info.env?.cuda ?? "none"],
                ["Host", info.env?.hostname ?? "—"],
                ["Started", info.started_at ? new Date(info.started_at).toLocaleTimeString() : "—"],
                ["Finished", info.finished_at ? new Date(info.finished_at).toLocaleTimeString() : "—"],
              ].map(([label, value]) => (
                <div key={label} className="flex justify-between gap-2">
                  <dt className="text-text-secondary">{label}</dt>
                  <dd className="truncate text-right font-mono" title={String(value)}>{value}</dd>
                </div>
              ))}
              {info.failure?.reason && (
                <div className="mt-2 rounded-md border border-danger p-2 text-danger">
                  <div className="font-semibold">Failure: {info.failure.reason}</div>
                  <div className="mt-1 text-xs text-text-secondary">
                    {FAILURE_HINTS[info.failure.reason] ?? "See the run log for details."}
                  </div>
                </div>
              )}
            </dl>
          )}
        </div>
      </div>

      <div className="rounded-lg border border-border bg-surface p-4">
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <h3 className="text-sm font-semibold">Log</h3>
          <input
            type="search"
            aria-label="Search logs"
            placeholder="Search logs…"
            value={logQuery}
            onChange={(event) => setLogQuery(event.target.value)}
            className="ml-auto rounded-md border border-border bg-bg px-3 py-1.5 text-xs"
          />
          <select
            aria-label="Filter logs by level"
            value={logLevel}
            onChange={(event) => setLogLevel(event.target.value)}
            className="rounded-md border border-border bg-bg px-2 py-1.5 text-xs"
          >
            <option value="all">All</option>
            <option value="INFO">INFO</option>
            <option value="WARNING">WARNING</option>
            <option value="ERROR">ERROR</option>
          </select>
        </div>
        <ul aria-label="Run log" className="max-h-72 space-y-0.5 overflow-auto rounded bg-bg p-3 font-mono text-xs">
          {filteredLogs.length === 0 && <li className="text-text-secondary">No matching log lines.</li>}
          {filteredLogs.slice(-400).map((line, index) => (
            <li key={index} className={line.level === "ERROR" ? "text-danger" : "text-text"}>
              <span className="text-text-secondary">{line.ts.slice(11, 19)}</span>{" "}
              <span className={line.level === "ERROR" ? "font-semibold" : ""}>{line.level}</span> {line.message}
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
