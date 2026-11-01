/*
 * Overlay geometry for the Predictions tab [B3, B4].
 *
 * predictions.json stores normalized cxcywh boxes: GT rows are
 * `[class_id, cx, cy, w, h]` and prediction rows are
 * `[class_id, score, cx, cy, w, h]`. The box is therefore always the LAST four
 * values — reading from the front silently yields NaN for GT rows. The overlay
 * scales those normalized values by the image's real pixel size, so it lines up
 * with whatever size the browser renders.
 */

export interface BoxRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export function boxRect(box: number[], width: number, height: number): BoxRect {
  const [cx, cy, w, h] = box.slice(-4);
  return {
    x: (cx - w / 2) * width,
    y: (cy - h / 2) * height,
    w: w * width,
    h: h * height,
  };
}

export function classIdOf(box: number[]): number {
  return box[0];
}

export function scoreOf(box: number[]): number {
  return box.length > 5 ? box[1] : 1;
}
