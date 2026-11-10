import { useEffect, useRef, useState } from "react";
import { apiFetch } from "../lib/api";
import { CheckList } from "../components/CheckList";
import { SummaryCards } from "../components/SummaryCards";
import { RecommendedCheckpoint } from "../components/RecommendedCheckpoint";
import { LicenceCard } from "../components/LicenceCard";
import { licenceGate, type CatalogVariant } from "../lib/licence";
import { deviceNotice, type GpuInfo } from "../lib/device";
import { autoSelection, frozenBackboneBlock, selectionSource } from "../lib/pretrained";
import {
  coerceConfig,
  isNumericProperty,
  parseFieldValue,
  propertiesOf,
  type HparamProperty,
} from "../lib/hparams";
import {
  useVerificationStream,
  type CheckItem,
  type Derived,
} from "../hooks/useVerificationStream";

interface WizardEditRun {
  verificationId: string;
  config: Record<string, unknown>;
  modelFamily: string;
  checkpointCatalogId: string | null;
  fromScratch: boolean;
}

interface WizardProps {
  onTrained: (runId: string) => void;
  edit?: WizardEditRun | null;
}

interface ModelInfo {
  family: string;
  supported_tasks: string[];
  supported_formats: string[];
  description: string;
  needs_gpu: boolean;
  hparam_schema: { type: string; properties: Record<string, HparamProperty> };
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

export function Wizard({ onTrained, edit = null }: WizardProps) {
  const [step, setStep] = useState(0);
  const [dataset, setDataset] = useState<DatasetUploaded | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [verificationId, setVerificationId] = useState<string | null>(null);
  const [derived, setDerived] = useState<Derived | null>(null);
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [checkpointCatalogId, setCheckpointCatalogId] = useState<string | null>(null);
  const [fromScratch, setFromScratch] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [licenceBlock, setLicenceBlock] = useState<string | null>(null);
  const [autoId, setAutoId] = useState<string | null>(null);
  const [autoReason, setAutoReason] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  // "Edit & run" from the Live Run page: jump straight to Configure with the
  // previous run's config + checkpoint [doc 09 §3].
  useEffect(() => {
    if (!edit) return;
    let disposed = false;
    apiFetch<{ items: ModelInfo[] }>("/api/v1/models")
      .then((body) => {
        if (disposed) return;
        const found = body.items.find((item) => item.family === edit.modelFamily);
        if (!found) return;
        setModel(found);
        // A previous run may hold string values (old wizard bug): coerce so the
        // numeric inputs render as numbers and post as numbers.
        setConfig(coerceConfig({ ...edit.config }, found.hparam_schema));
        setCheckpointCatalogId(edit.checkpointCatalogId);
        setFromScratch(edit.fromScratch);
        setVerificationId(edit.verificationId);
        setStep(3);
      })
      .catch(() => undefined);
    return () => {
      disposed = true;
    };
  }, [edit]);

  // Prefill Configure with the verifier's recommended_hparams [doc 06 §6]:
  // copy every schema key the recommendation supplies, cap batch_size for a
  // 4 GB GPU, then set the dev_preset boolean from the recommended preset.
  function applyRecommendedHparams(
    recommended: Record<string, unknown> | undefined,
    schema: ModelInfo["hparam_schema"],
  ): void {
    if (!recommended) return;
    const keys = Object.keys(schema.properties ?? {});
    setConfig((current) => {
      const next: Record<string, unknown> = { ...current };
      for (const key of keys) {
        if (key in recommended) next[key] = recommended[key];
      }
      if (typeof next.batch_size === "number" && next.batch_size > 4) {
        next.batch_size = 4;
      }
      if (typeof recommended.preset === "string") {
        next.dev_preset = recommended.preset === "dev";
      }
      return next;
    });
  }

  async function startTraining(): Promise<void> {
    setStarting(true);
    setError(null);
    try {
      const run = await apiFetch<{ id: string }>("/api/v1/runs", {
        method: "POST",
        body: JSON.stringify({
          verification_id: verificationId,
          name: `${model?.family ?? "run"}-${dataset?.dataset.name ?? "sample"}`,
          config: coerceConfig(config, model?.hparam_schema),
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
          onDone={(id, report) => {
            setVerificationId(id);
            setDerived(report);
            applyRecommendedHparams(report?.recommended_hparams, model.hparam_schema);
            // Pre-fill Configure with the verifier's auto-selected checkpoint (it passed
            // every hard filter) and keep its reason; the user can still override it.
            // "Train from scratch" stays OFF [Slice D].
            const auto = autoSelection(report);
            setAutoId(auto.catalogId);
            setAutoReason(auto.reason);
            setCheckpointCatalogId(auto.catalogId);
            setFromScratch(false);
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
          onLicenceBlock={setLicenceBlock}
          autoId={autoId}
          autoReason={autoReason}
          onBack={() => setStep(2)}
          onNext={() => setStep(4)}
        />
      )}
      {step === 4 && model && (dataset || verificationId) && (
        <TrainStep
          dataset={dataset}
          model={model}
          config={config}
          taskType={derived?.task_type ?? null}
          checkpointCatalogId={checkpointCatalogId}
          fromScratch={fromScratch}
          starting={starting}
          licenceBlock={licenceBlock}
          frozenBlock={frozenBackboneBlock(
            "freeze_backbone" in propertiesOf(model.hparam_schema),
            config,
            checkpointCatalogId,
            fromScratch,
          )}
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
  onDone: (verificationId: string, report: Derived | null) => void;
  onBack: () => void;
}

function VerifyStep({ datasetHash, modelFamily, onDone, onBack }: VerifyStepProps) {
  const [verificationId, setVerificationId] = useState<string | null>(null);
  const [acknowledged, setAcknowledged] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const stream = useVerificationStream(verificationId);
  const [finalDerived, setFinalDerived] = useState<Derived | null>(null);
  const [finalChecks, setFinalChecks] = useState<CheckItem[] | null>(null);

  const finished =
    stream.verificationStatus === "passed" ||
    stream.verificationStatus === "passed_with_warnings" ||
    stream.verificationStatus === "failed";
  const passed = stream.verificationStatus === "passed" || stream.verificationStatus === "passed_with_warnings";
  const failed = stream.verificationStatus === "failed";
  const effectiveChecks = finalChecks ?? stream.checks;
  const hasWarnings = effectiveChecks.some((check) => check.status === "warn");
  // Next is enabled only once the full report (derived facts + final checks)
  // has loaded, so Configure prefills with the real recommendations [doc 06 §6].
  const reportReady = finalDerived !== null && finalChecks !== null;

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
        body: JSON.stringify({ dataset_hash: datasetHash, model_family: modelFamily, preset: "auto" }),
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
          disabled={!finished || !passed || !reportReady || (hasWarnings && !acknowledged)}
          onClick={() => verificationId && onDone(verificationId, finalDerived)}
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
  onLicenceBlock: (message: string | null) => void;
  autoId: string | null;
  autoReason: string;
  onBack: () => void;
  onNext: () => void;
}

function ConfigureStep(props: ConfigureStepProps) {
  const { model, config, onChange } = props;
  const [advanced, setAdvanced] = useState(false);
  const [variants, setVariants] = useState<CatalogVariant[] | null>(null);
  const [variantsError, setVariantsError] = useState<string | null>(null);
  const [acceptBusy, setAcceptBusy] = useState(false);
  const [acceptError, setAcceptError] = useState<string | null>(null);
  const [gpu, setGpu] = useState<GpuInfo | null>(null);

  // What "auto" device will really do on this server [Slice F].
  useEffect(() => {
    let disposed = false;
    apiFetch<{ gpu?: GpuInfo }>("/api/v1/system")
      .then((body) => !disposed && setGpu(body.gpu ?? null))
      .catch(() => undefined);
    return () => {
      disposed = true;
    };
  }, []);

  // Variants of this model family, from the catalog (offered ones are selectable).
  useEffect(() => {
    let disposed = false;
    apiFetch<{ items: CatalogVariant[] }>(
      `/api/v1/checkpoints?family=${encodeURIComponent(model.family)}`,
    )
      .then((body) => !disposed && setVariants(body.items))
      .catch((exc) => !disposed && setVariantsError(exc instanceof Error ? exc.message : "Could not load the model catalog."));
    return () => {
      disposed = true;
    };
  }, [model.family]);

  const selected = variants?.find((item) => item.catalog_id === props.checkpointCatalogId) ?? null;
  const gate = licenceGate(props.checkpointCatalogId, props.fromScratch, selected, false);
  const wantsCheckpoint = !props.fromScratch && Boolean(props.checkpointCatalogId);
  let gateMessage: string | null = null;
  if (wantsCheckpoint) {
    if (variantsError) gateMessage = `Could not load the model catalog (${variantsError}), so the licence cannot be checked.`;
    else if (variants === null) gateMessage = "Loading the model catalog…";
    else gateMessage = gate.message;
  }
  const frozenBlock = frozenBackboneBlock(
    "freeze_backbone" in propertiesOf(model.hparam_schema),
    config,
    props.checkpointCatalogId,
    props.fromScratch,
  );
  const blocked = gateMessage !== null || frozenBlock !== null;
  if (gateMessage === null && frozenBlock !== null) gateMessage = frozenBlock;
  const source = selectionSource(props.checkpointCatalogId, props.autoId, props.fromScratch);
  const { onLicenceBlock } = props;
  useEffect(() => {
    onLicenceBlock(blocked ? gateMessage : null);
  }, [blocked, gateMessage, onLicenceBlock]);

  async function acceptLicence(catalogId: string): Promise<void> {
    setAcceptBusy(true);
    setAcceptError(null);
    try {
      await apiFetch("/api/v1/licences/accept", {
        method: "POST",
        body: JSON.stringify({ catalog_id: catalogId }),
      });
      setVariants((current) =>
        current?.map((item) => (item.catalog_id === catalogId ? { ...item, licence_accepted: true } : item)) ?? current,
      );
    } catch (exc) {
      setAcceptError(exc instanceof Error ? exc.message : "Could not record the acceptance.");
    } finally {
      setAcceptBusy(false);
    }
  }
  const properties = propertiesOf(model.hparam_schema);

  const basicKeys = ["epochs", "lr", "batch_size", "imgsz", "dev_preset", "amp", "freeze_backbone"];
  const visibleKeys = advanced ? Object.keys(properties) : Object.keys(properties).filter((key) => basicKeys.includes(key));
  const runTarget = config.dev_preset === false ? "GPU (full)" : "CPU (dev preset)";
  // A frozen backbone with no pretrained checkpoint cannot learn [rtdetr WARNING].
  const fromScratchFrozen = props.fromScratch && config.freeze_backbone === true;

  function setValue(key: string, value: unknown): void {
    onChange({ ...config, [key]: value });
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-3 sm:grid-cols-2">
        {visibleKeys.map((key) => {
          const prop = properties[key] ?? {};
          const value = config[key] ?? prop.default;
          const numeric = isNumericProperty(prop);
          return (
            <label key={key} className="flex flex-col gap-1 text-sm">
              <span className="font-mono text-xs text-text-secondary">{key}</span>
              {typeof value === "boolean" ? (
                <input type="checkbox" checked={Boolean(value)} onChange={(event) => setValue(key, event.target.checked)} />
              ) : (
                <input
                  type={numeric ? "number" : "text"}
                  inputMode={numeric ? "decimal" : undefined}
                  value={value === null || value === undefined ? "" : String(value)}
                  min={prop.minimum}
                  max={prop.maximum}
                  onChange={(event) => setValue(key, parseFieldValue(prop, event.target.value))}
                  className="rounded-md border border-border bg-bg px-3 py-2"
                />
              )}
            </label>
          );
        })}
      </div>
      {(() => {
        const notice = deviceNotice(gpu, model.needs_gpu, config.dev_preset !== false);
        const tone =
          notice.kind === "cpu-warning"
            ? "border-warning bg-warning/10 text-warning"
            : "border-border bg-surface text-text-secondary";
        return (
          <p
            role={notice.kind === "cpu-warning" ? "alert" : undefined}
            data-testid={notice.kind === "cpu-warning" ? "cpu-warning" : "device-notice"}
            className={`rounded-md border p-2 text-sm ${tone}`}
          >
            {notice.text}
          </p>
        );
      })()}
      <p data-testid="run-target" className="text-sm text-text-secondary">
        Run on: <span className="font-semibold text-text">{runTarget}</span>
      </p>
      <button type="button" onClick={() => setAdvanced((value) => !value)} className="self-start text-sm text-text-secondary hover:text-text">
        {advanced ? "Basic" : "Advanced"} settings
      </button>

      <fieldset className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-4">
        <legend className="px-1 text-sm font-semibold">Checkpoint</legend>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={props.fromScratch} onChange={(event) => props.onFromScratch(event.target.checked)} />
          Train from scratch
        </label>
        {fromScratchFrozen && (
          <p role="alert" data-testid="from-scratch-frozen-warning" className="rounded-md border border-warning bg-warning/10 p-2 text-xs text-warning">
            Training from scratch with a frozen backbone cannot learn — load a pretrained
            checkpoint or turn off freeze_backbone.
          </p>
        )}
        {!props.fromScratch && variants && variants.length > 0 && (
          <ul aria-label="Model variants" className="flex flex-col gap-1">
            {variants.map((item) => (
              <li key={item.catalog_id}>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="radio"
                    name="variant"
                    data-testid={`variant-${item.catalog_id}`}
                    checked={item.catalog_id === props.checkpointCatalogId}
                    disabled={!item.offered}
                    title={item.offered ? undefined : "Not offered yet: no verified download (sha256/URL)."}
                    onChange={() => props.onCheckpoint(item.catalog_id)}
                  />
                  <span className={`font-mono ${item.offered ? "" : "text-text-secondary"}`}>{item.catalog_id}</span>
                  <span className="text-xs text-text-secondary">
                    {item.variant}
                    {item.offered ? "" : " — not offered yet (no verified download; cannot be selected)"}
                  </span>
                </label>
              </li>
            ))}
          </ul>
        )}
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
        {source === "auto" && props.autoReason && (
          <p data-testid="auto-reason" className="rounded-md bg-bg p-2 text-xs text-text-secondary">
            <span className="font-semibold text-text">Auto-selected — kyun chuna: </span>
            {props.autoReason}
          </p>
        )}
        {source === "manual" && props.autoId && (
          <p data-testid="manual-pick" className="text-xs text-text-secondary">
            You overrode the auto-selected checkpoint
            {` (${props.autoId})`}.
          </p>
        )}
        {!props.fromScratch && selected && (
          <LicenceCard
            variant={selected}
            accepted={selected.licence_accepted}
            busy={acceptBusy}
            error={acceptError}
            onAccept={() => void acceptLicence(selected.catalog_id)}
          />
        )}
        {blocked && gateMessage && (
          <p role="alert" data-testid="licence-block" className="rounded-md border border-warning bg-warning/10 p-2 text-xs text-warning">
            <span className="font-semibold">Next is disabled: </span>
            {gateMessage}
          </p>
        )}
      </fieldset>

      <div className="flex gap-2">
        <button type="button" onClick={props.onBack} className="rounded-md border border-border px-4 py-2 text-sm hover:bg-bg">← Back</button>
        <button type="button" data-testid="configure-next" onClick={props.onNext} disabled={blocked} title={blocked && gateMessage ? gateMessage : undefined} className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50">Next →</button>
      </div>
    </div>
  );
}

// --- Step 5: Train ----------------------------------------------------------

interface TrainStepProps {
  dataset: DatasetUploaded | null;
  model: ModelInfo;
  config: Record<string, unknown>;
  taskType: string | null;
  checkpointCatalogId: string | null;
  fromScratch: boolean;
  starting: boolean;
  licenceBlock: string | null;
  frozenBlock: string | null;
  onStart: () => void;
  onBack: () => void;
}

function TrainStep(props: TrainStepProps) {
  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-lg border border-border bg-surface p-4 text-sm">
        <dl className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <div>
            <dt className="text-text-secondary">Dataset</dt>
            <dd>
              {props.dataset
                ? `${props.dataset.dataset.name} (${props.dataset.dataset.file_count} files)`
                : "reused verification dataset"}
            </dd>
          </div>
          <div><dt className="text-text-secondary">Model</dt><dd className="font-mono">{props.model.family}</dd></div>
          <div><dt className="text-text-secondary">Task</dt><dd data-testid="train-task">{props.taskType ?? "—"}</dd></div>
          <div><dt className="text-text-secondary">Checkpoint</dt><dd>{props.fromScratch ? "from scratch" : props.checkpointCatalogId ?? "none"}</dd></div>
          <div><dt className="text-text-secondary">Hyperparameters</dt><dd className="font-mono text-xs">{JSON.stringify(props.config)}</dd></div>
        </dl>
      </div>
      <div className="flex gap-2">
        <button type="button" onClick={props.onBack} className="rounded-md border border-border px-4 py-2 text-sm hover:bg-bg">← Back</button>
        <button type="button" data-testid="start-training" onClick={props.onStart} disabled={props.starting || props.licenceBlock !== null || props.frozenBlock !== null} className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50">
          {props.starting ? "Starting…" : "Start Training"}
        </button>
      </div>
      {props.frozenBlock && (
        <p role="alert" data-testid="frozen-backbone-block" className="rounded-md border border-danger bg-danger/10 p-3 text-sm font-medium text-danger">
          {props.frozenBlock}
        </p>
      )}
      {props.licenceBlock && (
        <p role="alert" className="text-xs text-warning">{props.licenceBlock}</p>
      )}
    </div>
  );
}

async function sha256Hex(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}
