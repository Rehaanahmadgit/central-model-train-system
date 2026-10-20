import { useEffect, useState } from "react";
import { CheckList } from "../components/CheckList";
import { useVerificationStream } from "../hooks/useVerificationStream";

interface DatasetMeta {
  hash: string;
  name: string;
  file_count: number;
}

interface VerificationStart {
  id: string;
  status: string;
}

// Verification demo: pick a dataset, start verification, watch the streaming
// checklist [doc 06 §8]. The 5-step wizard (slice 10) reuses these pieces.
export function VerifyDemo() {
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [selectedHash, setSelectedHash] = useState<string>("");
  const [modelFamily, setModelFamily] = useState("rtdetr");
  const [verificationId, setVerificationId] = useState<string | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const stream = useVerificationStream(verificationId);

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

  async function start(): Promise<void> {
    setStartError(null);
    setVerificationId(null);
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
    setVerificationId(started.id);
  }

  const finished = stream.verificationStatus !== null;
  const passed =
    stream.verificationStatus === "passed" || stream.verificationStatus === "passed_with_warnings";
  const failed = stream.verificationStatus === "failed";

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
        </p>
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
            disabled={!passed}
            title={passed ? "Proceeds to configuration (slice 10)" : "Fix the failing checks first"}
            className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50"
          >
            Next →
          </button>
        )}
      </div>
      <CheckList checks={stream.checks} />
    </section>
  );
}
