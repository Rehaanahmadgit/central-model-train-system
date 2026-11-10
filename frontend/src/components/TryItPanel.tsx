import { useEffect, useMemo, useRef, useState } from "react";
import { apiFetch, apiPostForm } from "../lib/api";
import {
  annotatedFilename,
  checkFiles,
  DEFAULT_THRESHOLD,
  drawAnnotated,
  MAX_IMAGES,
  visiblePredictions,
  type InferImage,
  type InferResponse,
} from "../lib/tryit";
import { DetectionOverlay, useDatasetImage } from "./PredictionGallery";

/*
 * "Try it" tab [Slice G1]: run the trained detector on your own images (kept only
 * in this browser tab, never stored on the server) or on 6 sample images from the
 * dataset. The slider and class filter re-filter the boxes locally; "Run detection"
 * is only needed to change the model-side threshold floor.
 */

interface Shown {
  image: InferImage;
  url: string | null; // local object URL for uploads; null for samples (fetched by path)
}

function ResultCard({
  runId,
  shown,
  classNames,
  confidence,
  classFilter,
}: {
  runId: string;
  shown: Shown;
  classNames: string[];
  confidence: number;
  classFilter: string;
}) {
  const sample = useDatasetImage(runId, shown.url ? "" : (shown.image.path ?? ""));
  const url = shown.url ?? sample.url;
  const [saveError, setSaveError] = useState<string | null>(null);
  const boxes = visiblePredictions(shown.image, confidence, classFilter, classNames);

  async function download(): Promise<void> {
    if (!url) return;
    setSaveError(null);
    try {
      const blob = await drawAnnotated(url, shown.image, boxes, classNames);
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = annotatedFilename(shown.image.name);
      link.click();
      URL.revokeObjectURL(link.href);
    } catch (exc) {
      setSaveError(exc instanceof Error ? exc.message : "Could not create the PNG.");
    }
  }

  return (
    <figure data-testid="tryit-result" className="overflow-hidden rounded-lg border border-border bg-surface">
      <div className="relative bg-bg">
        {url ? (
          <img src={url} alt={shown.image.name} className="block h-auto w-full" />
        ) : (
          <div className="flex h-40 items-center justify-center text-xs text-text-secondary">
            {sample.failed ? "image unavailable" : "loading…"}
          </div>
        )}
        {url && (
          <DetectionOverlay
            width={shown.image.width || 1}
            height={shown.image.height || 1}
            gt={shown.image.gt ?? []}
            pred={shown.image.pred}
            classNames={classNames}
            confidence={confidence}
            classFilter={classFilter}
            label={shown.image.name}
          />
        )}
      </div>
      <figcaption className="flex flex-wrap items-center justify-between gap-2 p-2 text-xs">
        <span className="truncate font-mono" title={shown.image.name}>
          {shown.image.name}
        </span>
        <span data-testid="tryit-count" className="text-text-secondary">
          {boxes.length} detection{boxes.length === 1 ? "" : "s"}
        </span>
        <button
          type="button"
          data-testid="tryit-download"
          onClick={() => void download()}
          disabled={!url}
          className="rounded-md border border-border px-2 py-1 hover:bg-bg disabled:opacity-50"
        >
          Download annotated PNG
        </button>
        {saveError && <span role="alert" className="w-full text-danger">{saveError}</span>}
      </figcaption>
    </figure>
  );
}

export function TryItPanel({ runId }: { runId: string }) {
  const [files, setFiles] = useState<File[]>([]);
  const [problems, setProblems] = useState<string[]>([]);
  const [result, setResult] = useState<InferResponse | null>(null);
  const [shown, setShown] = useState<Shown[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(DEFAULT_THRESHOLD);
  const [classFilter, setClassFilter] = useState("all");
  const [dragging, setDragging] = useState(false);
  const urls = useRef<string[]>([]);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const created = urls.current;
    return () => created.forEach((url) => URL.revokeObjectURL(url));
  }, []);

  const classNames = useMemo(() => result?.class_names ?? [], [result]);

  function addFiles(list: FileList | File[]): void {
    const checked = checkFiles([...files, ...Array.from(list)]);
    setFiles(checked.ok);
    setProblems(checked.problems);
  }

  async function run(request: () => Promise<InferResponse>, local: File[] | null): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      const body = await request();
      urls.current.forEach((url) => URL.revokeObjectURL(url));
      urls.current = [];
      const next: Shown[] = body.images.map((image, index) => {
        if (!local) return { image, url: null };
        const url = URL.createObjectURL(local[index]);
        urls.current.push(url);
        return { image, url };
      });
      setResult(body);
      setShown(next);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Detection failed.");
    } finally {
      setBusy(false);
    }
  }

  const floor = Math.min(confidence, DEFAULT_THRESHOLD);

  function runUploads(): void {
    const form = new FormData();
    files.forEach((file) => form.append("images", file, file.name));
    void run(
      () => apiPostForm<InferResponse>(`/api/v1/runs/${runId}/infer?threshold=${floor}`, form),
      files,
    );
  }

  function runSamples(): void {
    void run(
      () => apiFetch<InferResponse>(`/api/v1/runs/${runId}/infer-samples?threshold=${floor}`, { method: "POST" }),
      null,
    );
  }

  return (
    <div data-testid="tryit-panel" className="flex flex-col gap-4">
      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          addFiles(event.dataTransfer.files);
        }}
        className={`flex flex-col items-center gap-3 rounded-lg border-2 border-dashed p-6 text-sm ${dragging ? "border-accent bg-accent/5" : "border-border bg-surface"}`}
      >
        <p>
          Drop up to {MAX_IMAGES} images here (PNG, JPG, WEBP, 10 MB each) — they stay in this
          browser tab and are not saved on the server.
        </p>
        <div className="flex flex-wrap justify-center gap-2">
          <input
            ref={input}
            type="file"
            multiple
            accept="image/png,image/jpeg,image/webp"
            data-testid="tryit-file-input"
            className="hidden"
            onChange={(event) => {
              if (event.target.files) addFiles(event.target.files);
              event.target.value = "";
            }}
          />
          <button type="button" onClick={() => input.current?.click()} className="rounded-md border border-border px-3 py-1.5 hover:bg-bg">
            Choose images
          </button>
          <button
            type="button"
            data-testid="tryit-samples"
            onClick={runSamples}
            disabled={busy}
            className="rounded-md border border-border px-3 py-1.5 hover:bg-bg disabled:opacity-50"
          >
            Use sample images
          </button>
          <button
            type="button"
            data-testid="tryit-run"
            onClick={runUploads}
            disabled={busy || files.length === 0}
            className="rounded-md bg-accent px-3 py-1.5 font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50"
          >
            {busy ? "Detecting…" : `Run detection${files.length ? ` (${files.length})` : ""}`}
          </button>
        </div>
        {files.length > 0 && (
          <ul data-testid="tryit-files" className="text-xs text-text-secondary">
            {files.map((file) => (
              <li key={file.name + file.size}>{file.name}</li>
            ))}
          </ul>
        )}
        {problems.length > 0 && (
          <ul role="alert" data-testid="tryit-problems" className="text-xs text-warning">
            {problems.map((problem) => (
              <li key={problem}>{problem}</li>
            ))}
          </ul>
        )}
      </div>

      {error && (
        <p role="alert" data-testid="tryit-error" className="rounded-md border border-danger bg-danger/10 p-2 text-sm text-danger">
          {error}
        </p>
      )}

      {result && (
        <>
          <p data-testid="tryit-device" className="text-xs text-text-secondary">
            {result.device_note}
          </p>
          <div className="flex flex-wrap items-end gap-3 rounded-lg border border-border bg-surface p-3 text-xs">
            <label className="flex min-w-48 flex-col gap-1">
              <span className="text-text-secondary">Confidence ≥ {confidence.toFixed(2)}</span>
              <input
                type="range"
                aria-label="Confidence threshold"
                data-testid="tryit-confidence"
                min={0}
                max={1}
                step={0.05}
                value={confidence}
                onChange={(event) => setConfidence(Number(event.target.value))}
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-text-secondary">Class</span>
              <select
                aria-label="Filter detections by class"
                value={classFilter}
                onChange={(event) => setClassFilter(event.target.value)}
                className="rounded-md border border-border bg-bg px-2 py-1"
              >
                <option value="all">All classes</option>
                {classNames.map((name, index) => (
                  <option key={`${name}-${index}`} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <span className="ml-auto text-text-secondary">
              {result.images.some((item) => item.gt) && (
                <>
                  GT <span className="text-success">green solid</span> ·{" "}
                </>
              )}
              prediction <span className="text-danger">red dashed</span>
            </span>
          </div>
          <div data-testid="tryit-results" className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {shown.map((item) => (
              <ResultCard
                key={item.image.path ?? item.image.name + item.url}
                runId={runId}
                shown={item}
                classNames={classNames}
                confidence={confidence}
                classFilter={classFilter}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
