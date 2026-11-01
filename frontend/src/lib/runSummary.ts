/*
 * Header facts for a finished run [LiveRun].
 *
 * `run.json.best` is written by the parent when the worker reports its result.
 * If that event never arrives (worker killed, older record, resume), `best`
 * keeps its placeholder `{metric: null}`, so the header used to show "—" even
 * though evaluation.json already held the number. The fallback below reads the
 * evaluation report instead of inventing anything.
 */

export interface BestInfo {
  metric?: string | null;
  value?: number | null;
  epoch?: number | null;
}

export interface ResolvedBest {
  metric: string;
  value: number;
  epoch?: number;
  source: "run" | "evaluation";
}

function isValidBest(best: BestInfo | null | undefined): best is BestInfo & { metric: string } {
  return (
    typeof best?.metric === "string" &&
    best.metric.length > 0 &&
    typeof best.value === "number" &&
    Number.isFinite(best.value)
  );
}

function numberAt(source: unknown, path: string[]): number | null {
  let current: unknown = source;
  for (const key of path) {
    if (typeof current !== "object" || current === null) return null;
    current = (current as Record<string, unknown>)[key];
  }
  return typeof current === "number" && Number.isFinite(current) ? current : null;
}

export function resolveBest(
  best: BestInfo | null | undefined,
  evaluation: unknown,
): ResolvedBest | null {
  if (isValidBest(best)) {
    return {
      metric: best.metric,
      value: best.value as number,
      epoch: typeof best.epoch === "number" && best.epoch > 0 ? best.epoch : undefined,
      source: "run",
    };
  }
  // Detection: val mAP50 (the number that picked the checkpoint), then the
  // legacy top-level field. Classification falls back to accuracy.
  const detection = numberAt(evaluation, ["val", "mAP50"]) ?? numberAt(evaluation, ["mAP50"]);
  if (detection !== null) {
    return { metric: "val/mAP50", value: detection, source: "evaluation" };
  }
  const accuracy = numberAt(evaluation, ["val", "accuracy"]) ?? numberAt(evaluation, ["accuracy"]);
  if (accuracy !== null) {
    return { metric: "val/accuracy", value: accuracy, source: "evaluation" };
  }
  return null;
}

export interface SystemSample {
  gpu_available: boolean | null;
  gpu_util: number | null;
}

export function gpuLabel(system: SystemSample | null, terminal: boolean): string {
  if (system) {
    if (system.gpu_available === false) return "No GPU detected";
    // 0% is a real measurement (idle or finished), not "unknown".
    if (system.gpu_util != null) return `${system.gpu_util}%`;
  }
  return terminal ? "n/a (run finished)" : "n/a";
}
