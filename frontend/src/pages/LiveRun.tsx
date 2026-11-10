import { useEffect, useMemo, useState } from "react";
import { apiFetch, apiUploadFile, getAccessToken } from "../lib/api";
import { failureView, waitingLabel, type RunFailure, type Waiting } from "../lib/pretrained";
import { ConnectionIndicator } from "../components/ConnectionIndicator";
import { MetricChart } from "../components/MetricChart";
import { PredictionGallery } from "../components/PredictionGallery";
import { useRunStream } from "../hooks/useRunStream";
import { isTerminalRunState } from "../lib/runStreamLogic";
import { gpuLabel as formatGpu, resolveBest } from "../lib/runSummary";

interface LiveRunProps {
  runId: string;
  theme: "light" | "dark";
  onBack: () => void;
  onNavigate: (runId: string) => void;
  onEditRun: (edit: {
    verificationId: string;
    config: Record<string, unknown>;
    modelFamily: string;
    checkpointCatalogId: string | null;
    fromScratch: boolean;
  }) => void;
}

interface RunInfo {
  id: string;
  name: string;
  state: string;
  model_family: string;
  task_type: string | null;
  verification_id: string | null;
  dataset_hash: string;
  checkpoint: { catalog_id?: string | null; sha256?: string | null; source?: string | null };
  config: Record<string, unknown>;
  seed: number;
  env: Record<string, string | null>;
  queued_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  best: { metric?: string | null; value?: number; epoch?: number };
  failure?: RunFailure | null;
  waiting_for?: Waiting | null;
}

const FAILURE_HINTS: Record<string, string> = {
  WORKER_LOST: "The training process stopped responding or was killed. Check the container's memory/CPU limits, then retry or resume.",
  OOM: "Out of memory. Reduce the batch size (or imgsz) in Configure and retry.",
  NAN_LOSS: "Loss became NaN/Inf. Lower the learning rate, check the data for corrupt values, then retry.",
  DATA_ERROR: "The dataset or configuration is invalid. Read the message, fix it, and re-run verification.",
  GPU_UNAVAILABLE: "This run needs a GPU and none is visible. Use the dev preset on CPU, or start the container with --gpus all.",
  UNKNOWN: "An unexpected error occurred. See the run log for the traceback.",
};

type RunTab = "live" | "evaluation" | "predictions";

const TABS: { id: RunTab; label: string }[] = [
  { id: "live", label: "Live" },
  { id: "evaluation", label: "Evaluation" },
  { id: "predictions", label: "Predictions" },
];

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
export function LiveRun({ runId, theme, onBack, onNavigate, onEditRun }: LiveRunProps) {
  const stream = useRunStream(runId);
  const [info, setInfo] = useState<RunInfo | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [artifacts, setArtifacts] = useState<string[]>([]);
  const [now, setNow] = useState(Date.now());
  const [logQuery, setLogQuery] = useState("");
  const [logLevel, setLogLevel] = useState("all");
  const [tab, setTab] = useState<RunTab>("live");
  const [evaluation, setEvaluation] = useState<unknown>(null);

  // Run info (run.json): params, checkpoint, seed, env, timestamps. Refetched
  // on every state transition so Started/Finished/Task/best stay current once
  // the worker writes them (the WS only carries state + metrics) [doc 09 §3].
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
  }, [runId, stream.runState]);

  // While the run waits for its checkpoint (download progress) refetch every second.
  useEffect(() => {
    if ((stream.runState ?? info?.state) !== "preparing") return;
    let disposed = false;
    const timer = window.setInterval(() => {
      apiFetch<RunInfo>(`/api/v1/runs/${runId}`)
        .then((body) => !disposed && setInfo(body))
        .catch(() => undefined);
    }, 1000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [runId, stream.runState, info?.state]);

  // "Best" lives in run.json only after the worker's result event. When that
  // never arrived (placeholder {metric: null}) the finished run showed "—"
  // even though evaluation.json had the number: read it as a fallback [D124].
  useEffect(() => {
    if (!info || info.best?.metric || !isTerminalRunState(info.state)) return;
    let disposed = false;
    apiFetch<unknown>(`/api/v1/runs/${runId}/evaluation`)
      .then((body) => {
        if (!disposed) setEvaluation(body);
      })
      .catch(() => undefined);
    return () => {
      disposed = true;
    };
  }, [info, runId]);

  // Artifact list decides which download buttons to show [doc 05 §5].
  useEffect(() => {
    let disposed = false;
    apiFetch<{ artifacts?: { name: string }[] }>(`/api/v1/runs/${runId}/artifacts`)
      .then((body) => {
        if (!disposed) setArtifacts((body.artifacts ?? []).map((item) => item.name));
      })
      .catch(() => {
        if (!disposed) setArtifacts([]);
      });
    return () => {
      disposed = true;
    };
  }, [runId, stream.runState]);

  // Ticking clock for elapsed/ETA while a run is live.
  useEffect(() => {
    const state = stream.runState ?? info?.state;
    const live = state && !isTerminalRunState(state);
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

  const [importMsg, setImportMsg] = useState<string | null>(null);
  const [importBusy, setImportBusy] = useState(false);

  async function importFromFile(file: File, catalogId: string): Promise<void> {
    setImportBusy(true);
    setImportMsg(null);
    try {
      await apiUploadFile(`/api/v1/checkpoints/${encodeURIComponent(catalogId)}/import`, file);
      setImportMsg("Imported and sha256-verified. Press Run again to start with it.");
    } catch (exc) {
      setImportMsg(exc instanceof Error ? exc.message : "Import failed.");
    } finally {
      setImportBusy(false);
    }
  }

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
  const terminal = isTerminalRunState(state);
  const best = resolveBest(info?.best, evaluation);
  const bestText = best ? `${best.metric} ${best.value}` : "—";
  const hasArtifact = (name: string): boolean => artifacts.includes(name);
  // 0% is a real reading; only a missing sample is unknown, and a finished run
  // says so explicitly instead of a bare "n/a" [D124].
  const gpuLabel = formatGpu(stream.system, terminal);

  async function rerun(): Promise<void> {
    if (!info?.verification_id) {
      setLoadError("This run has no verification to reuse.");
      return;
    }
    setLoadError(null);
    try {
      const created = await apiFetch<{ id: string }>("/api/v1/runs", {
        method: "POST",
        body: JSON.stringify({
          verification_id: info.verification_id,
          name: `${info.name ?? "run"}-again`,
          config: info.config,
          checkpoint_catalog_id: info.checkpoint?.catalog_id ?? null,
          from_scratch: false,
        }),
      });
      onNavigate(created.id);
    } catch (exc) {
      setLoadError(exc instanceof Error ? exc.message : "Could not start a new run.");
    }
  }

  function editAndRun(): void {
    if (!info?.verification_id) {
      setLoadError("This run has no verification to reuse.");
      return;
    }
    onEditRun({
      verificationId: info.verification_id,
      config: info.config,
      modelFamily: info.model_family,
      checkpointCatalogId: info.checkpoint?.catalog_id ?? null,
      fromScratch: !info.checkpoint?.catalog_id,
    });
  }

  async function downloadArtifact(name: string): Promise<void> {
    setDownloadError(null);
    try {
      const headers: Record<string, string> = {};
      const token = getAccessToken();
      if (token) headers["X-Access-Token"] = token;
      const response = await fetch(`/api/v1/runs/${runId}/artifacts/${name}`, { headers });
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as { error?: { message?: string } } | null;
        throw new Error(body?.error?.message ?? `Download failed (${response.status})`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = name.split("/").pop() ?? "artifact";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (exc) {
      setDownloadError(exc instanceof Error ? exc.message : "Download failed.");
    }
  }

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
        {/* A finished run has no live stream to indicate: the server closes it
            on purpose, so a "Reconnecting" badge would be misleading. */}
        {!terminal && <ConnectionIndicator status={stream.status} />}
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

      {state === "preparing" && waitingLabel(info?.waiting_for) && (
        <div data-testid="waiting-checkpoint" className="rounded-lg border border-border bg-surface p-4 text-sm">
          <div className="font-medium">{waitingLabel(info?.waiting_for)}</div>
          {typeof info?.waiting_for?.percent === "number" && (
            <div
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(info.waiting_for.percent)}
              className="mt-2 h-2 w-full overflow-hidden rounded-full bg-bg"
            >
              <div className="h-full bg-accent" style={{ width: `${info.waiting_for.percent}%` }} />
            </div>
          )}
        </div>
      )}

      {terminal && (
        <div data-testid="run-finished" className="flex flex-wrap items-center gap-4 rounded-lg border border-border bg-surface p-4">
          <h3 className="text-sm font-semibold">Run finished</h3>
          <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold text-white ${STATE_COLORS[state] ?? "bg-text-secondary"}`}>
            {state}
          </span>
          <span className="text-sm">
            Best:{" "}
            <span className="font-mono">
              {bestText}
              {best?.epoch ? ` (epoch ${best.epoch})` : ""}
            </span>
            {best?.source === "evaluation" && (
              <span data-testid="best-from-evaluation" className="ml-2 text-xs text-text-secondary">
                from evaluation report
              </span>
            )}
          </span>
          <span className="text-sm">
            GPU: <span className="font-mono">{gpuLabel}</span>
          </span>
          <span className="ml-auto flex gap-2">
            <button
              type="button"
              data-testid="run-again"
              onClick={() => void rerun()}
              className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-on-accent hover:bg-accent-hover"
            >
              Run again
            </button>
            <button
              type="button"
              data-testid="edit-and-run"
              onClick={editAndRun}
              className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-bg"
            >
              Edit & run
            </button>
            {hasArtifact("checkpoints/best.ckpt") && (
              <button
                type="button"
                data-testid="download-best"
                onClick={() => void downloadArtifact("checkpoints/best.ckpt")}
                className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-bg"
              >
                Download best.ckpt
              </button>
            )}
            {hasArtifact("checkpoints/weights.pth") && (
              <button
                type="button"
                data-testid="download-weights"
                onClick={() => void downloadArtifact("checkpoints/weights.pth")}
                className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-bg"
              >
                Download weights.pth
              </button>
            )}
          </span>
        </div>
      )}
      {downloadError && <p role="alert" className="rounded-md border border-danger bg-surface p-3 text-sm text-danger">{downloadError}</p>}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatTile label="Elapsed" value={fmtElapsed(elapsedSeconds)} />
        <StatTile label="ETA" value={etaSeconds ? fmtElapsed(etaSeconds) : "—"} hint={totalEpochs ? `epoch ${currentEpoch}/${totalEpochs}` : undefined} />
        <StatTile label="Progress" value={progressPct !== null ? `${progressPct}%` : "—"} />
        <StatTile
          label="Best"
          value={bestText}
          hint={
            best?.epoch
              ? `epoch ${best.epoch}`
              : best?.source === "evaluation"
                ? "from evaluation report"
                : undefined
          }
        />
      </div>

      {progressPct !== null && (
        <div role="progressbar" aria-valuenow={progressPct} aria-valuemin={0} aria-valuemax={100} className="h-2 w-full overflow-hidden rounded-full bg-bg">
          <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${progressPct}%` }} />
        </div>
      )}

      <div role="tablist" aria-label="Run views" className="flex gap-1 border-b border-border">
        {TABS.map((item) => (
          <button
            key={item.id}
            role="tab"
            aria-selected={tab === item.id}
            onClick={() => setTab(item.id)}
            className={`px-4 py-2 text-sm ${tab === item.id ? "border-b-2 border-accent font-semibold text-text" : "text-text-secondary hover:text-text"}`}
          >
            {item.label}
          </button>
        ))}
      </div>

      {tab === "evaluation" && <EvaluationPanel runId={runId} />}

      {tab === "predictions" && <PredictionGallery runId={runId} />}

      {tab === "live" && (
        <>
      <MetricChart metrics={stream.metrics} theme={theme} />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="rounded-lg border border-border bg-surface p-4">
          <h3 className="mb-2 text-sm font-semibold">System</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between"><span className="text-text-secondary">GPU</span><span>{gpuLabel}</span></div>
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
                <div data-testid="run-failure" className="mt-2 rounded-md border border-danger p-2 text-danger">
                  <div className="font-semibold">
                    Failure: {failureView(info.failure)?.title ?? info.failure.reason}
                  </div>
                  <div className="mt-1 text-xs text-text-secondary">
                    {failureView(info.failure)?.hint ??
                      FAILURE_HINTS[info.failure.reason] ??
                      "See the run log for details."}
                  </div>
                  {failureView(info.failure)?.canImport && info.failure.catalog_id && (
                    <label className="mt-2 flex flex-wrap items-center gap-2 text-xs text-text">
                      <span className="rounded-md border border-border px-3 py-1.5 font-medium">
                        {importBusy ? "Importing…" : "Import from file"}
                      </span>
                      <input
                        type="file"
                        data-testid="import-from-file"
                        disabled={importBusy}
                        className="max-w-[16rem] text-xs"
                        onChange={(event) => {
                          const file = event.target.files?.[0];
                          if (file && info.failure?.catalog_id) {
                            void importFromFile(file, info.failure.catalog_id);
                          }
                        }}
                      />
                    </label>
                  )}
                  {importMsg && <div role="status" className="mt-1 text-xs text-text">{importMsg}</div>}
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
        </>
      )}
    </section>
  );
}


interface SplitMetrics {
  mAP50: number;
  per_class_AP?: Record<string, number>;
  samples?: number;
  /** Present when the split is too small for a stable mAP [A2]. */
  warning?: string;
}

interface EvalReport {
  task: string;
  accuracy?: number;
  rmse?: number;
  classes?: number[];
  confusion_matrix?: number[][];
  per_class?: { class_id: number; support: number; precision: number; recall: number; f1: number }[];
  samples?: number;
  // Detection (RT-DETR) [doc 08 §6]: mAP@0.5 + per-class AP keyed by name.
  // The top-level keys are the val numbers (backward compatible); `val`/`test`
  // separate the split that chose the checkpoint from the unseen one.
  mAP50?: number;
  per_class_AP?: Record<string, number>;
  class_names?: string[];
  val?: SplitMetrics;
  test?: SplitMetrics | null;
  test_reason?: string | null;
}

// val drove checkpoint selection, test was never trained on: show both.
function DetectionMetrics({ report }: { report: EvalReport }) {
  const val: SplitMetrics =
    report.val ?? {
      mAP50: report.mAP50 ?? 0,
      per_class_AP: report.per_class_AP,
      samples: report.samples,
    };
  const test = report.test ?? null;
  const classNames = [
    ...new Set([...Object.keys(val.per_class_AP ?? {}), ...Object.keys(test?.per_class_AP ?? {})]),
  ];
  const fmt = (value: number | undefined): string =>
    typeof value === "number" ? value.toFixed(4) : "—";

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatTile
          label="val mAP50"
          value={fmt(val.mAP50)}
          hint="drove checkpoint selection"
        />
        <StatTile
          label="test mAP50"
          value={test ? fmt(test.mAP50) : "n/a"}
          hint={test ? "held out, never trained on" : (report.test_reason ?? "no test split")}
        />
        <StatTile label="val images" value={String(val.samples ?? "—")} />
        <StatTile label="test images" value={test ? String(test.samples ?? "—") : "—"} />
      </div>

      {!test && (
        <p data-testid="no-test-split" className="rounded-md border border-border bg-surface p-3 text-xs text-text-secondary">
          No held-out test split in this dataset, so there is no unseen-number yet. Every image
          shown above was available for training or checkpoint selection — treat the val mAP as an
          optimistic estimate. Add a <span className="font-mono">test/</span> folder and re-upload to
          get an honest number.
        </p>
      )}
      {[...(val.warning ? [val.warning] : []), ...(test?.warning ? [test.warning] : [])].map(
        (warning) => (
          <p
            key={warning}
            data-testid="small-split-warning"
            className="rounded-md border border-warning bg-surface p-3 text-xs text-text-secondary"
          >
            Small split — {warning}.
          </p>
        ),
      )}

      {classNames.length > 0 && (
        <div className="rounded-lg border border-border bg-surface p-4">
          <h3 className="mb-2 text-sm font-semibold">Per-class AP</h3>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-text-secondary">
                <th className="p-1 text-left">Class</th>
                <th className="p-1 text-right">val AP</th>
                <th className="p-1 text-right">test AP</th>
              </tr>
            </thead>
            <tbody>
              {classNames.map((name) => (
                <tr key={name}>
                  <td className="border-b border-border p-1 font-mono">{name}</td>
                  <td className="border-b border-border p-1 text-right font-mono">
                    {fmt(val.per_class_AP?.[name])}
                  </td>
                  <td className="border-b border-border p-1 text-right font-mono">
                    {fmt(test?.per_class_AP?.[name])}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function EvaluationPanel({ runId }: { runId: string }) {
  const [report, setReport] = useState<EvalReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;
    apiFetch<EvalReport>(`/api/v1/runs/${runId}/evaluation`)
      .then((body) => {
        if (!disposed) setReport(body);
      })
      .catch((exc) => {
        if (!disposed) setError(exc instanceof Error ? exc.message : "No evaluation yet.");
      });
    return () => {
      disposed = true;
    };
  }, [runId]);

  if (error && !report) {
    return (
      <div className="rounded-lg border border-dashed border-border bg-surface p-8 text-center text-sm text-text-secondary">
        {error}
      </div>
    );
  }
  if (!report) {
    return (
      <div className="h-32 animate-pulse rounded-lg border border-border bg-surface" aria-label="Loading evaluation" />
    );
  }

  const matrix = report.confusion_matrix;
  const classes = report.classes ?? [];
  const isDetection = report.val !== undefined || report.mAP50 !== undefined;

  if (isDetection) {
    return <DetectionMetrics report={report} />;
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {report.accuracy !== undefined && (
          <StatTile label="Accuracy" value={report.accuracy.toFixed(4)} />
        )}
        {report.rmse !== undefined && (
          <StatTile label="RMSE" value={report.rmse.toFixed(4)} />
        )}
        <StatTile label="Samples" value={String(report.samples ?? "—")} />
      </div>

      {matrix && classes.length > 0 && (
        <div className="rounded-lg border border-border bg-surface p-4">
          <h3 className="mb-2 text-sm font-semibold">Confusion matrix</h3>
          <div className="overflow-auto">
            <table className="w-full text-xs">
              <thead>
                <tr>
                  <th className="border-b border-border p-1 text-left text-text-secondary">class \ predicted</th>
                  {classes.map((c) => (
                    <th key={c} className="border-b border-border p-1 text-right font-mono">{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.map((row, i) => (
                  <tr key={i}>
                    <th className="border-b border-border p-1 text-left font-mono text-text-secondary">{classes[i]}</th>
                    {row.map((value, j) => (
                      <td
                        key={j}
                        className={`border-b border-border p-1 text-right font-mono ${
                          i === j ? "bg-accent/10 font-semibold" : value > 0 ? "text-text" : "text-text-secondary"
                        }`}
                      >
                        {value}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {report.per_class && report.per_class.length > 0 && (
        <div className="rounded-lg border border-border bg-surface p-4">
          <h3 className="mb-2 text-sm font-semibold">Per-class metrics</h3>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-text-secondary">
                <th className="p-1 text-left">Class</th>
                <th className="p-1 text-right">Support</th>
                <th className="p-1 text-right">Precision</th>
                <th className="p-1 text-right">Recall</th>
                <th className="p-1 text-right">F1</th>
              </tr>
            </thead>
            <tbody>
              {report.per_class.map((row) => (
                <tr key={row.class_id}>
                  <td className="border-b border-border p-1 font-mono">{row.class_id}</td>
                  <td className="border-b border-border p-1 text-right font-mono">{row.support}</td>
                  <td className="border-b border-border p-1 text-right font-mono">{row.precision.toFixed(4)}</td>
                  <td className="border-b border-border p-1 text-right font-mono">{row.recall.toFixed(4)}</td>
                  <td className="border-b border-border p-1 text-right font-mono">{row.f1.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
