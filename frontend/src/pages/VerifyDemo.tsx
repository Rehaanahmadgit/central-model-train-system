import { useEffect, useState } from "react";
import { CheckList } from "../components/CheckList";
import { RecommendedCheckpoint } from "../components/RecommendedCheckpoint";
import { SummaryCards } from "../components/SummaryCards";
import {
  useVerificationStream,
  type CheckItem,
  type Derived,
} from "../hooks/useVerificationStream";

interface DatasetMeta {
  hash: string;
  name: string;
  file_count: number;
}

interface VerificationStart {
  id: string;
  status: string;
  cached?: boolean;
  verifier_version?: string;
}

// Verification demo: pick a dataset, start verification, watch the streaming
// checklist [doc 06 §8]. The 5-step wizard (slice 10) reuses these pieces.
export function VerifyDemo() {
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [selectedHash, setSelectedHash] = useState<string>("");
  const [modelFamily, setModelFamily] = useState("rtdetr");
  const [verificationId, setVerificationId] = useState<string | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const [cachedHit, setCachedHit] = useState(false);
  const [acknowledged, setAcknowledged] = useState(false);
  const stream = useVerificationStream(verificationId);
  const [finalDerived, setFinalDerived] = useState<Derived | null>(null);
  const [finalChecks, setFinalChecks] = useState<CheckItem[] | null>(null);

  useEffect(() => {
    let disposed = false;
    fetch("/api/v1/datasets")
      .then((response) => (response.ok ? response.json() : null))
      .then((body: { items?: DatasetMeta[] } | null) => {
        if (disposed || !body?.items) return;
        setDatasets(body.items);
        if (body.items.length > 0 && !selectedHash) {
          setSelectedHash(body.items[0].hash);
        }
      })
      .catch(() => {
        if (!disposed) setStartError("Could not load datasets.");
      });
    return () => {
      disposed = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // The initial snapshot predates report.json, so refetch once the run ends to
  // pick up the derived facts (task, stats, recommendations) [doc 05 §3].
  useEffect(() => {
    const status = stream.verificationStatus;
    if (!verificationId || !status || status === "running") return;
    let disposed = false;
    fetch(`/api/v1/verifications/${verificationId}`)
      .then((response) => (response.ok ? response.json() : null))
      .then((body: { derived?: Derived; checks?: CheckItem[] } | null) => {
        if (disposed || !body) return;
        if (body.derived) setFinalDerived(body.derived);
        if (Array.isArray(body.checks)) setFinalChecks(body.checks);
      })
      .catch(() => undefined);
    return () => {
      disposed = true;
    };
  }, [stream.verificationStatus, verificationId]);

  async function start(): Promise<void> {
    setStartError(null);
    setVerificationId(null);
    setCachedHit(false);
    setAcknowledged(false);
    setFinalDerived(null);
    setFinalChecks(null);
    const response = await fetch("/api/v1/verifications", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_hash: selectedHash, model_family: modelFamily }),
    });
    if (!response.ok) {
      const body = (await response.json().catch(() => null)) as {
        error?: { message?: string };
      } | null;
      setStartError(body?.error?.message ?? "Verification could not start.");
      return;
    }
    const started = (await response.json()) as VerificationStart;
    setCachedHit(Boolean(started.cached));
    setVerificationId(started.id);
  }

  const finished = stream.verificationStatus !== null;
  const passed =
    stream.verificationStatus === "passed" || stream.verificationStatus === "passed_with_warnings";
  const failed = stream.verificationStatus === "failed";
  const effectiveChecks = finalChecks ?? stream.checks;
  const hasWarnings = effectiveChecks.some((check) => check.status === "warn");
  const needsAck = passed && hasWarnings && !acknowledged;

  return (
    <section className="flex flex-col gap-6">
      <header>
        <h2 className="text-xl font-semibold">Verify dataset</h2>
        <p className="mt-1 text-sm text-text-secondary">
          Run the verification checks against an uploaded dataset [doc 06].
        </p>
      </header>
      {startError && (
        <p role="alert" className="rounded-md border border-danger bg-surface p-4 text-sm text-danger">
          ✗ {startError}
        </p>
      )}
      {passed && (
        <p role="status" className="rounded-md border border-success bg-surface p-4 text-sm text-success">
          ✓ Dataset verified successfully
          {cachedHit || stream.cached ? " (reused the cached report for this dataset)" : ""}
        </p>
      )}
      {passed && hasWarnings && (
        <label className="flex items-start gap-3 rounded-md border border-warning bg-surface p-4 text-sm">
          <input
            type="checkbox"
            checked={acknowledged}
            onChange={(event) => setAcknowledged(event.target.checked)}
            className="mt-0.5"
          />
          <span>
            I reviewed the {effectiveChecks.filter((check) => check.status === "warn").length}{" "}
            warning(s) above and want to continue.
          </span>
        </label>
      )}
      {failed && (
        <p role="alert" className="rounded-md border border-danger bg-surface p-4 text-sm text-danger">
          ✗ Verification failed — fix the failing checks and re-run.
        </p>
      )}
      <div className="flex flex-wrap items-end gap-4">
        <label className="flex flex-col gap-1 text-sm text-text-secondary">
          Dataset
          <select
            value={selectedHash}
            onChange={(event) => setSelectedHash(event.target.value)}
            className="rounded-md border border-border bg-surface px-3 py-2 text-text"
          >
            {datasets.map((dataset) => (
              <option key={dataset.hash} value={dataset.hash}>
                {dataset.name} ({dataset.file_count} files)
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-sm text-text-secondary">
          Model family
          <input
            value={modelFamily}
            onChange={(event) => setModelFamily(event.target.value)}
            className="rounded-md border border-border bg-surface px-3 py-2 text-text"
          />
        </label>
        <button
          type="button"
          onClick={() => void start()}
          disabled={!selectedHash || stream.verificationStatus === null && Boolean(verificationId)}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50"
        >
          {verificationId && stream.verificationStatus === null ? "Verifying…" : "Start Verification"}
        </button>
        {finished && (
          <button
            type="button"
            disabled={!passed || needsAck}
            title={
              !passed
                ? "Fix the failing checks first"
                : needsAck
                  ? "Acknowledge the warnings first"
                  : "Proceeds to configuration (slice 10)"
            }
            className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50"
          >
            Next →
          </button>
        )}
      </div>
      <SummaryCards derived={finalDerived ?? stream.derived} />
      <RecommendedCheckpoint derived={finalDerived ?? stream.derived} />
      <CheckList checks={finalChecks ?? stream.checks} />
    </section>
  );
}
