import { useEffect, useMemo, useState } from "react";
import { apiFetch, getAccessToken } from "../lib/api";
import { boxRect, classIdOf, scoreOf } from "../lib/predictions";

/*
 * Predictions tab [B3]: the twelve worst images for a finished detection run.
 *
 * artifacts/predictions.json holds normalized boxes (cxcywh) for GT and for
 * predictions >= the stored threshold, plus tp/fp/fn/wrong_class per image.
 * Boxes are drawn in the image's own pixel space (viewBox = width x height),
 * so the overlay lines up with whatever size the browser renders.
 */

interface Counts {
  tp: number;
  fp: number;
  fn: number;
  wrong_class: number;
}

export interface PredictionImage {
  split: string;
  path: string;
  width: number;
  height: number;
  gt: number[][];
  pred: number[][];
  counts: Counts;
  error_score: number;
}

interface PredictionsFile {
  class_names?: string[];
  threshold?: number;
  match_iou?: number;
  splits?: Record<string, number>;
  images?: PredictionImage[];
}

const GALLERY_SIZE = 12;
// Default view is deliberately stricter than the stored threshold: 0.25 shows
// a lot of noise boxes, 0.5 is the useful "what did it actually find" view.
const DEFAULT_CONFIDENCE = 0.5;
const GT_COLOR = "#10b981"; // green: ground truth
const PRED_COLOR = "#ef4444"; // red dashed: prediction

function useDatasetImage(runId: string, path: string): { url: string | null; failed: boolean } {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let disposed = false;
    let objectUrl: string | null = null;
    setUrl(null);
    setFailed(false);
    const headers: Record<string, string> = {};
    const token = getAccessToken();
    if (token) headers["X-Access-Token"] = token;
    // The endpoint is token-protected, so <img src> cannot be used directly.
    fetch(`/api/v1/runs/${runId}/images?path=${encodeURIComponent(path)}`, { headers })
      .then((response) => {
        if (!response.ok) throw new Error(`image ${response.status}`);
        return response.blob();
      })
      .then((blob) => {
        if (disposed) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      })
      .catch(() => {
        if (!disposed) setFailed(true);
      });
    return () => {
      disposed = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [runId, path]);

  return { url, failed };
}

function PredictionCard({
  runId,
  image,
  classNames,
  confidence,
  classFilter,
}: {
  runId: string;
  image: PredictionImage;
  classNames: string[];
  confidence: number;
  classFilter: string;
}) {
  const { url, failed } = useDatasetImage(runId, image.path);
  const width = image.width || 1;
  const height = image.height || 1;
  const labelSize = Math.max(10, Math.round(width / 45));
  const strokeWidth = Math.max(1.5, width / 320);
  const matchesClass = (box: number[]): boolean =>
    classFilter === "all" || classNames[classIdOf(box)] === classFilter;
  const gtBoxes = image.gt.filter(matchesClass);
  const predBoxes = image.pred.filter(
    (box) => matchesClass(box) && scoreOf(box) >= confidence,
  );

  return (
    <figure className="overflow-hidden rounded-lg border border-border bg-surface">
      <div className="relative bg-bg">
        {url ? (
          <img src={url} alt={image.path} className="block h-auto w-full" />
        ) : (
          <div className="flex h-40 items-center justify-center text-xs text-text-secondary">
            {failed ? "image unavailable" : "loading…"}
          </div>
        )}
        {url && (
          <svg
            viewBox={`0 0 ${width} ${height}`}
            preserveAspectRatio="none"
            aria-label={`overlay for ${image.path}`}
            className="pointer-events-none absolute inset-0 h-full w-full"
          >
            {gtBoxes.map((box, index) => {
              const rect = boxRect(box, width, height);
              return (
                <g key={`gt-${index}`} data-box="gt">
                  <rect
                    x={rect.x}
                    y={rect.y}
                    width={rect.w}
                    height={rect.h}
                    fill="none"
                    stroke={GT_COLOR}
                    strokeWidth={strokeWidth}
                  />
                  <text
                    x={rect.x + 2}
                    y={Math.max(labelSize, rect.y - 3)}
                    fontSize={labelSize}
                    fill={GT_COLOR}
                    stroke="#000"
                    strokeWidth={labelSize / 6}
                    paintOrder="stroke"
                  >
                    {classNames[classIdOf(box)] ?? classIdOf(box)}
                  </text>
                </g>
              );
            })}
            {predBoxes.map((box, index) => {
              const rect = boxRect(box, width, height);
              return (
                <g key={`pred-${index}`} data-box="pred">
                  <rect
                    x={rect.x}
                    y={rect.y}
                    width={rect.w}
                    height={rect.h}
                    fill="none"
                    stroke={PRED_COLOR}
                    strokeWidth={strokeWidth}
                    strokeDasharray={`${labelSize / 2} ${labelSize / 3}`}
                  />
                  <text
                    x={rect.x + 2}
                    y={rect.y + rect.h + labelSize}
                    fontSize={labelSize}
                    fill={PRED_COLOR}
                    stroke="#000"
                    strokeWidth={labelSize / 6}
                    paintOrder="stroke"
                  >
                    {`${classNames[classIdOf(box)] ?? classIdOf(box)} ${scoreOf(box).toFixed(2)}`}
                  </text>
                </g>
              );
            })}
          </svg>
        )}
      </div>
      <figcaption className="space-y-1 p-2 text-xs">
        <div className="flex items-center justify-between gap-2">
          <span className="truncate font-mono" title={image.path}>
            {image.path}
          </span>
          <span className="rounded-full border border-border px-2 py-0.5 text-text-secondary">
            {image.split}
          </span>
        </div>
        <div className="flex flex-wrap gap-2 text-text-secondary">
          <span data-testid="error-score" className="font-semibold text-text">
            error {image.error_score}
          </span>
          <span>tp {image.counts.tp}</span>
          <span>fp {image.counts.fp}</span>
          <span>fn {image.counts.fn}</span>
          <span>wrong class {image.counts.wrong_class}</span>
        </div>
      </figcaption>
    </figure>
  );
}

export function PredictionGallery({ runId }: { runId: string }) {
  const [data, setData] = useState<PredictionsFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [split, setSplit] = useState("all");
  const [classFilter, setClassFilter] = useState("all");
  const [confidence, setConfidence] = useState(DEFAULT_CONFIDENCE);

  useEffect(() => {
    let disposed = false;
    apiFetch<PredictionsFile>(`/api/v1/runs/${runId}/predictions`)
      .then((body) => {
        if (disposed) return;
        setData(body);
        // Never show fewer boxes than the file stores: if a run kept only
        // >= 0.6 predictions, start there instead of at 0.5.
        setConfidence(Math.max(DEFAULT_CONFIDENCE, body.threshold ?? 0));
      })
      .catch((exc) => {
        if (!disposed) setError(exc instanceof Error ? exc.message : "No predictions yet.");
      });
    return () => {
      disposed = true;
    };
  }, [runId]);

  const classNames = useMemo(() => data?.class_names ?? [], [data]);
  const splits = useMemo(() => Object.keys(data?.splits ?? {}), [data]);
  const images = useMemo(() => data?.images ?? [], [data]);

  const gallery = useMemo(() => {
    const filtered = images.filter((image) => {
      if (split !== "all" && image.split !== split) return false;
      if (classFilter === "all") return true;
      const hasClass = (box: number[]): boolean => classNames[classIdOf(box)] === classFilter;
      return image.gt.some(hasClass) || image.pred.some(hasClass);
    });
    return [...filtered]
      .sort((left, right) => right.error_score - left.error_score)
      .slice(0, GALLERY_SIZE);
  }, [images, split, classFilter, classNames]);

  if (error && !data) {
    return (
      <div className="rounded-lg border border-dashed border-border bg-surface p-8 text-center text-sm text-text-secondary">
        {error}
      </div>
    );
  }
  if (!data) {
    return (
      <div
        className="h-32 animate-pulse rounded-lg border border-border bg-surface"
        aria-label="Loading predictions"
      />
    );
  }
  if (images.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border bg-surface p-8 text-center text-sm text-text-secondary">
        This run produced no predictions to show.
      </div>
    );
  }

  const storedThreshold = data.threshold ?? 0.25;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end gap-3 rounded-lg border border-border bg-surface p-3 text-xs">
        <label className="flex flex-col gap-1">
          <span className="text-text-secondary">Split</span>
          <select
            aria-label="Filter predictions by split"
            value={split}
            onChange={(event) => setSplit(event.target.value)}
            className="rounded-md border border-border bg-bg px-2 py-1"
          >
            <option value="all">All ({images.length})</option>
            {splits.map((name) => (
              <option key={name} value={name}>
                {name} ({data.splits?.[name] ?? 0})
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-text-secondary">Class</span>
          <select
            aria-label="Filter predictions by class"
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
        <label className="flex min-w-48 flex-col gap-1">
          <span className="text-text-secondary">
            Confidence ≥ {confidence.toFixed(2)}
          </span>
          <input
            type="range"
            aria-label="Confidence threshold"
            min={0}
            max={1}
            step={0.05}
            value={confidence}
            onChange={(event) => setConfidence(Number(event.target.value))}
          />
        </label>
        <span className="ml-auto text-text-secondary">
          GT <span className="text-success">green solid</span> · prediction{" "}
          <span className="text-danger">red dashed</span> · saved at score ≥ {storedThreshold.toFixed(2)}
        </span>
      </div>

      {gallery.length === 0 ? (
        <div className="rounded-lg border border-dashed border-border bg-surface p-8 text-center text-sm text-text-secondary">
          No images match this filter.
        </div>
      ) : (
        <div
          data-testid="prediction-gallery"
          className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3"
        >
          {gallery.map((image) => (
            <PredictionCard
              key={image.path}
              runId={runId}
              image={image}
              classNames={classNames}
              confidence={confidence}
              classFilter={classFilter}
            />
          ))}
        </div>
      )}
    </div>
  );
}
