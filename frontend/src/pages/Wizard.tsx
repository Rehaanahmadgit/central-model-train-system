import { useEffect, useRef, useState } from "react";
import { apiFetch } from "../lib/api";
import { CheckList } from "../components/CheckList";
import { SummaryCards } from "../components/SummaryCards";
import { RecommendedCheckpoint } from "../components/RecommendedCheckpoint";
import {
  useVerificationStream,
  type CheckItem,
  type Derived,
} from "../hooks/useVerificationStream";

interface WizardProps {
  onTrained: (runId: string) => void;
}

interface ModelInfo {
  family: string;
  supported_tasks: string[];
  supported_formats: string[];
  description: string;
  needs_gpu: boolean;
  hparam_schema: { type: string; properties: Record<string, unknown> };
  default_hparams: Record<string, unknown>;
}

interface DatasetUploaded {
  dataset_hash: string;
  deduplicated: boolean;
  dataset: { name: string; file_count: number };
}

const STEPS = ["Dataset", "Model", "Verify", "Configure", "Train"] as const;

function stepState(current: number, index: number): "done" | "current" | "todo" {
  return index < current ? "done" : index === current ? "current" : "todo";
}

export function Wizard({ onTrained }: WizardProps) {
  const [step, setStep] = useState(0);
  const [dataset, setDataset] = useState<DatasetUploaded | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [verificationId, setVerificationId] = useState<string | null>(null);
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [checkpointCatalogId, setCheckpointCatalogId] = useState<string | null>(null);
  const [fromScratch, setFromScratch] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function startTraining(): Promise<void> {
    setStarting(true);
    setError(null);
    try {
      const run = await apiFetch<{ id: string }>("/api/v1/runs", {
        method: "POST",
        body: JSON.stringify({
          verification_id: verificationId,
          name: `${model?.family ?? "run"}-${dataset?.dataset.name ?? "sample"}`,
          config,
          checkpoint_catalog_id: fromScratch ? null : checkpointCatalogId,
          from_scratch: fromScratch,
        }),
      });
      setRunId(run.id);
      onTrained(run.id);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not start training.");
    } finally {
      setStarting(false);
    }
  }

  return (
    <section className="flex flex-col gap-6">
      <ol aria-label="Training wizard steps" className="flex flex-wrap items-center gap-2">
        {STEPS.map((label, index) => {
          const state = stepState(step, index);
          return (
            <li key={label} className="flex items-center gap-2">
              <span
                className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-semibold ${
                  state === "done"
                    ? "bg-success text-white"
                    : state === "current"
                      ? "bg-accent text-on-accent"
                      : "border border-border text-text-secondary"
                }`}
              >
                {state === "done" ? "✓" : index + 1}
              </span>
              <span className={`text-sm ${state === "current" ? "font-semibold" : "text-text-secondary"}`}>
                {label}
              </span>
              {index < STEPS.length - 1 && <span aria-hidden="true" className="text-text-secondary">·</span>}
            </li>
          );
        })}
      </ol>

      {error && <p role="alert" className="rounded-md border border-danger bg-surface p-3 text-sm text-danger">{error}</p>}

      {step === 0 && (
        <DatasetStep
          onDone={(result) => {
            setDataset(result);
            setStep(1);
          }}
        />
      )}
      {step === 1 && (
        <ModelStep
          onSelect={(selected) => {
            setModel(selected);
            setConfig({ ...selected.default_hparams });
            setStep(2);
          }}
        />
      )}
      {step === 2 && dataset && model && (
        <VerifyStep
          datasetHash={dataset.dataset_hash}
          modelFamily={model.family}
          onDone={(id) => {
            setVerificationId(id);
            setStep(3);
          }}
          onBack={() => setStep(1)}
        />
      )}
      {step === 3 && model && (
        <ConfigureStep
          model={model}
          config={config}
          onChange={setConfig}
          checkpointCatalogId={checkpointCatalogId}
          onCheckpoint={setCheckpointCatalogId}
          fromScratch={fromScratch}
          onFromScratch={setFromScratch}
          onBack={() => setStep(2)}
          onNext={() => setStep(4)}
        />
      )}
      {step === 4 && dataset && model && (
        <TrainStep
          dataset={dataset}
          model={model}
          config={config}
          checkpointCatalogId={checkpointCatalogId}
          fromScratch={fromScratch}
          starting={starting}
          onStart={() => void startTraining()}
          onBack={() => setStep(3)}
        />
      )}
      {runId && <p className="text-sm text-text-secondary">Run {runId} starting…</p>}
    </section>
  );
}

// --- Step 1: Dataset -------------------------------------------------------

interface DatasetStepProps {
  onDone: (result: DatasetUploaded) => void;
}

function DatasetStep({ onDone }: DatasetStepProps) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [paused, setPaused] = useState(false);
  const [progress, setProgress] = useState({ bytes: 0, total: 0 });
  const [speed, setSpeed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [sampleLoading, setSampleLoading] = useState(false);
  const pauseRef = useRef(false);
  const speedWindow = useRef<number[]>([]);

  useEffect(() => {
    pauseRef.current = paused;
  }, [paused]);

  async function uploadFile(file: File): Promise<void> {
    setError(null);
    setUploading(true);
    setPaused(false);
    setProgress({ bytes: 0, total: file.size });
    try {
      const init = await apiFetch<{ upload_id: string; chunk_size: number }>("/api/v1/uploads", {
        method: "POST",
        body: JSON.stringify({ filename: file.name, size_bytes: file.size }),
      });
      const chunkSize = init.chunk_size;
      let uploaded = 0;
      let index = 0;
      let lastSample = performance.now();
      let lastBytes = 0;
      while (uploaded < file.size) {
        if (pauseRef.current) {
          await new Promise((resolve) => setTimeout(resolve, 100));
          continue;
        }
        const chunk = file.slice(uploaded, uploaded + chunkSize);
        const digest = await sha256Hex(await chunk.arrayBuffer());
        await apiFetch<unknown>(`/api/v1/uploads/${init.upload_id}/chunks/${index}`, {
          method: "PUT",
          headers: { "Content-Type": "application/octet-stream", "X-Chunk-Sha256": digest },
          body: chunk,
        });
        uploaded += chunk.size;
        index += 1;
        setProgress({ bytes: uploaded, total: file.size });
        const now = performance.now();
        const deltaMs = now - lastSample;
        if (deltaMs >= 500) {
          const bytesDelta = uploaded - lastBytes;
          const currentSpeed = (bytesDelta / deltaMs) * 1000;
          speedWindow.current.push(currentSpeed);
          if (speedWindow.current.length > 10) speedWindow.current.shift();
          setSpeed(Math.round(speedWindow.current.reduce((a, b) => a + b, 0) / speedWindow.current.length));
          lastSample = now;
          lastBytes = uploaded;
        }
      }
      const completed = await apiFetch<DatasetUploaded>(`/api/v1/uploads/${init.upload_id}/complete`, { method: "POST" });
      onDone(completed);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Upload failed.");
    } finally {
      setUploading(false);
      setPaused(false);
    }
  }

  async function trySample(): Promise<void> {
    setError(null);
    setSampleLoading(true);
    try {
      const result = await apiFetch<DatasetUploaded>("/api/v1/datasets/sample", {
        method: "POST",
        body: JSON.stringify({ kind: "tabular" }),
      });
      onDone(result);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not create the sample dataset.");
    } finally {
      setSampleLoading(false);
    }
  }

  const percent = progress.total ? Math.round((progress.bytes / progress.total) * 100) : 0;
  const etaSeconds = speed > 0 && progress.total ? (progress.total - progress.bytes) / speed : null;

  return (
    <div className="flex flex-col gap-4">
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload a dataset (drag and drop or click)"
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          const file = event.dataTransfer.files[0];
          if (file) void uploadFile(file);
        }}
        onClick={() => document.getElementById("dataset-file-input")?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            document.getElementById("dataset-file-input")?.click();
          }
        }}
        className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-12 text-center ${
          dragging ? "border-accent bg-bg" : "border-border bg-surface"
        }`}
      >
        <input
          id="dataset-file-input"
          type="file"
          accept=".zip,.tar,.tar.gz,.tgz,.tar.bz2,.csv,.parquet"
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void uploadFile(file);
          }}
        />
        <p className="font-semibold">Drop a dataset, or click to browse</p>
        <p className="text-sm text-text-secondary">zip · tar · CSV · parquet</p>
      </div>

      {uploading && (
        <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-4">
          <div className="flex items-center justify-between text-sm">
            <span>{paused ? "Paused" : "Uploading…"} {percent}%</span>
            <span className="text-text-secondary">
              {speed > 0 ? `${(speed / 1024).toFixed(0)} KB/s` : "—"}
              {etaSeconds ? ` · ETA ${Math.ceil(etaSeconds)}s` : ""}
            </span>
          </div>
          <div role="progressbar" aria-valuenow={percent} className="h-2 w-full overflow-hidden rounded-full bg-bg">
            <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${percent}%` }} />
          </div>
          <button type="button" onClick={() => setPaused((value) => !value)} className="self-start rounded-md border border-border px-3 py-1.5 text-sm hover:bg-bg">
            {paused ? "Resume" : "Pause"}
          </button>
        </div>
      )}

      <div className="flex items-center gap-3">
        <span className="text-sm text-text-secondary">or</span>
        <button
          type="button"
          data-testid="try-sample"
          onClick={() => void trySample()}
          disabled={sampleLoading}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50"
        >
          {sampleLoading ? "Creating…" : "Try sample dataset"}
        </button>
      </div>

      {error && <p role="alert" className="rounded-md border border-danger bg-surface p-3 text-sm text-danger">{error}</p>}
    </div>
  );
}

// --- Step 2: Model ----------------------------------------------------------

interface ModelStepProps {
  onSelect: (model: ModelInfo) => void;
}

function ModelStep({ onSelect }: ModelStepProps) {
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [query, setQuery] = useState("");

  useEffect(() => {
    let disposed = false;
    apiFetch<{ items: ModelInfo[] }>("/api/v1/models")
      .then((body) => {
        if (!disposed) setModels(body.items);
      })
      .catch(() => {
        if (!disposed) setModels([]);
      });
    return () => {
      disposed = true;
    };
  }, []);

  const filtered = (models ?? []).filter((model) =>
    `${model.family} ${model.description} ${model.supported_tasks.join(" ")}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );

  return (
    <div className="flex flex-col gap-4">
      <input
        type="search"
        aria-label="Search models"
        placeholder="Search models…"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        className="rounded-md border border-border bg-surface px-3 py-2 text-sm"
      />
      {models === null ? (
        <div className="grid gap-3 sm:grid-cols-2" aria-label="Loading models">
          {[0, 1, 2, 3].map((index) => (
            <div key={index} className="h-28 animate-pulse rounded-lg border border-border bg-surface" />
          ))}
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          {filtered.map((model) => (
            <button
              key={model.family}
              type="button"
              data-testid={`model-${model.family}`}
              onClick={() => onSelect(model)}
              className="flex flex-col gap-1 rounded-lg border border-border bg-surface p-4 text-left hover:border-accent"
            >
              <span className="flex items-center gap-2">
                <span className="font-mono text-sm font-semibold">{model.family}</span>
                {model.needs_gpu && (
                  <span className="rounded-full border border-warning px-2 py-0.5 text-xs text-warning">needs GPU</span>
                )}
              </span>
              <span className="text-sm text-text-secondary">{model.description}</span>
              <span className="text-xs text-text-secondary">tasks: {model.supported_tasks.join(", ")}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// --- Step 3: Verify ---------------------------------------------------------

interface VerifyStepProps {
  datasetHash: string;
  modelFamily: string;
  onDone: (verificationId: string) => void;
  onBack: () => void;
}

function VerifyStep({ datasetHash, modelFamily, onDone, onBack }: VerifyStepProps) {
  const [verificationId, setVerificationId] = useState<string | null>(null);
  const [acknowledged, setAcknowledged] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const stream = useVerificationStream(verificationId);
  const [finalDerived, setFinalDerived] = useState<Derived | null>(null);
  const [finalChecks, setFinalChecks] = useState<CheckItem[] | null>(null);

  const finished = stream.verificationStatus !== null;
  const passed = stream.verificationStatus === "passed" || stream.verificationStatus === "passed_with_warnings";
  const failed = stream.verificationStatus === "failed";
  const effectiveChecks = finalChecks ?? stream.checks;
  const hasWarnings = effectiveChecks.some((check) => check.status === "warn");

  useEffect(() => {
    if (!verificationId || !finished) return;
    let disposed = false;
    apiFetch<{ derived?: Derived; checks?: CheckItem[] }>(`/api/v1/verifications/${verificationId}`)
      .then((body) => {
        if (disposed) return;
        if (body.derived) setFinalDerived(body.derived);
        if (body.checks) setFinalChecks(body.checks);
      })
      .catch(() => undefined);
    return () => {
      disposed = true;
    };
  }, [verificationId, finished]);

  async function start(): Promise<void> {
    setError(null);
    setVerificationId(null);
    setAcknowledged(false);
    setFinalDerived(null);
    setFinalChecks(null);
    try {
      const body = await apiFetch<{ id: string }>("/api/v1/verifications", {
        method: "POST",
        body: JSON.stringify({ dataset_hash: datasetHash, model_family: modelFamily, preset: "dev" }),
      });
      setVerificationId(body.id);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Verification could not start.");
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {!verificationId && (
        <button type="button" data-testid="start-verification" onClick={() => void start()} className="self-start rounded-md bg-accent px-4 py-2 text-sm font-medium text-on-accent hover:bg-accent-hover">
          Start Verification
        </button>
      )}
      {error && <p role="alert" className="rounded-md border border-danger bg-surface p-3 text-sm text-danger">{error}</p>}
      {passed && (
        <p role="status" className="rounded-md border border-success bg-surface p-3 text-sm text-success">
          ✓ Dataset verified successfully
        </p>
      )}
      {failed && (
        <p role="alert" className="rounded-md border border-danger bg-surface p-3 text-sm text-danger">
          ✗ Verification failed — fix the checks and re-run.
        </p>
      )}
      {passed && hasWarnings && (
        <label className="flex items-start gap-3 rounded-md border border-warning bg-surface p-3 text-sm">
          <input type="checkbox" data-testid="ack-warnings" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)} />
          <span>I reviewed the {effectiveChecks.filter((check) => check.status === "warn").length} warning(s) and want to continue.</span>
        </label>
      )}
      <SummaryCards derived={finalDerived ?? stream.derived} />
      <RecommendedCheckpoint derived={finalDerived ?? stream.derived} />
      <CheckList checks={effectiveChecks} />
      <div className="flex gap-2">
        <button type="button" onClick={onBack} className="rounded-md border border-border px-4 py-2 text-sm hover:bg-bg">← Back</button>
        <button
          type="button"
          data-testid="verify-next"
          disabled={!finished || !passed || (hasWarnings && !acknowledged)}
          onClick={() => verificationId && onDone(verificationId)}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50"
        >
          Next →
        </button>
      </div>
    </div>
  );
}

// --- Step 4: Configure ------------------------------------------------------

interface ConfigureStepProps {
  model: ModelInfo;
  config: Record<string, unknown>;
  onChange: (config: Record<string, unknown>) => void;
  checkpointCatalogId: string | null;
  onCheckpoint: (value: string | null) => void;
  fromScratch: boolean;
  onFromScratch: (value: boolean) => void;
  onBack: () => void;
  onNext: () => void;
}

function ConfigureStep(props: ConfigureStepProps) {
  const { model, config, onChange } = props;
  const [advanced, setAdvanced] = useState(false);
  const properties = (model.hparam_schema?.properties ?? {}) as Record<string, { type?: string; default?: unknown; minimum?: number; maximum?: number }>;

  const basicKeys = ["epochs", "lr", "batch_size", "imgsz"];
  const visibleKeys = advanced ? Object.keys(properties) : Object.keys(properties).filter((key) => basicKeys.includes(key));

  function setValue(key: string, value: unknown): void {
    onChange({ ...config, [key]: value });
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-3 sm:grid-cols-2">
        {visibleKeys.map((key) => {
          const prop = properties[key] ?? {};
          const value = config[key] ?? prop.default;
          return (
            <label key={key} className="flex flex-col gap-1 text-sm">
              <span className="font-mono text-xs text-text-secondary">{key}</span>
              {typeof value === "boolean" ? (
                <input type="checkbox" checked={Boolean(value)} onChange={(event) => setValue(key, event.target.checked)} />
              ) : (
                <input
                  type={typeof value === "number" ? "number" : "text"}
                  value={value === null || value === undefined ? "" : String(value)}
                  min={prop.minimum}
                  max={prop.maximum}
                  onChange={(event) => {
                    const raw = event.target.value;
                    const parsed = typeof value === "number" && raw !== "" ? Number(raw) : raw;
                    setValue(key, parsed);
                  }}
                  className="rounded-md border border-border bg-bg px-3 py-2"
                />
              )}
            </label>
          );
        })}
      </div>
      <button type="button" onClick={() => setAdvanced((value) => !value)} className="self-start text-sm text-text-secondary hover:text-text">
        {advanced ? "Basic" : "Advanced"} settings
      </button>

      <fieldset className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-4">
        <legend className="px-1 text-sm font-semibold">Checkpoint</legend>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={props.fromScratch} onChange={(event) => props.onFromScratch(event.target.checked)} />
          Train from scratch
        </label>
        {!props.fromScratch && (
          <input
            type="text"
            aria-label="Checkpoint catalog id"
            placeholder="checkpoint catalog id (e.g. resnet50@imagenet-1k)"
            value={props.checkpointCatalogId ?? ""}
            onChange={(event) => props.onCheckpoint(event.target.value || null)}
            className="rounded-md border border-border bg-bg px-3 py-2 text-sm font-mono"
          />
        )}
      </fieldset>

      <div className="flex gap-2">
        <button type="button" onClick={props.onBack} className="rounded-md border border-border px-4 py-2 text-sm hover:bg-bg">← Back</button>
        <button type="button" data-testid="configure-next" onClick={props.onNext} className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-on-accent hover:bg-accent-hover">Next →</button>
      </div>
    </div>
  );
}

// --- Step 5: Train ----------------------------------------------------------

interface TrainStepProps {
  dataset: DatasetUploaded;
  model: ModelInfo;
  config: Record<string, unknown>;
  checkpointCatalogId: string | null;
  fromScratch: boolean;
  starting: boolean;
  onStart: () => void;
  onBack: () => void;
}

function TrainStep(props: TrainStepProps) {
  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-lg border border-border bg-surface p-4 text-sm">
        <dl className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <div><dt className="text-text-secondary">Dataset</dt><dd>{props.dataset.dataset.name} ({props.dataset.dataset.file_count} files)</dd></div>
          <div><dt className="text-text-secondary">Model</dt><dd className="font-mono">{props.model.family}</dd></div>
          <div><dt className="text-text-secondary">Checkpoint</dt><dd>{props.fromScratch ? "from scratch" : props.checkpointCatalogId ?? "none"}</dd></div>
          <div><dt className="text-text-secondary">Hyperparameters</dt><dd className="font-mono text-xs">{JSON.stringify(props.config)}</dd></div>
        </dl>
      </div>
      <div className="flex gap-2">
        <button type="button" onClick={props.onBack} className="rounded-md border border-border px-4 py-2 text-sm hover:bg-bg">← Back</button>
        <button type="button" data-testid="start-training" onClick={props.onStart} disabled={props.starting} className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50">
          {props.starting ? "Starting…" : "Start Training"}
        </button>
      </div>
    </div>
  );
}

async function sha256Hex(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}
