/*
 * Axis assignment for the live metric chart [doc 09 §3.3].
 *
 * Loss lives on the primary axis (its own scale, e.g. 0..15). Scores such as
 * mAP50 or accuracy are bounded 0..1 and would look flat on a loss axis, so
 * they get a dedicated right-hand axis with a fixed 0..1 range. The learning
 * rate gets its own axis too (it is ~1e-4 and would be invisible anywhere else).
 */

export const LOSS_AXIS = 0;
export const SCORE_AXIS = 1;
export const LR_AXIS = 2;

// Metric names are `<split>/<name>`; anything that is a 0..1 quality score.
const SCORE_TOKENS = ["map", "accuracy", "acc", "f1", "precision", "recall", "iou"];

export function metricName(name: string): string {
  const slash = name.lastIndexOf("/");
  return (slash >= 0 ? name.slice(slash + 1) : name).toLowerCase();
}

export function isScoreMetric(name: string): boolean {
  const base = metricName(name);
  return SCORE_TOKENS.some((token) => base.includes(token));
}

export function axisIndexFor(name: string): number {
  if (metricName(name) === "lr") return LR_AXIS;
  if (isScoreMetric(name)) return SCORE_AXIS;
  return LOSS_AXIS;
}
