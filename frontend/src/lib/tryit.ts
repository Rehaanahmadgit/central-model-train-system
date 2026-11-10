/*
 * "Try it" helpers [Slice G]: pure functions, no DOM except drawAnnotated.
 *
 * Detection rows are `[class_id, score, cx, cy, w, h]` (6 values), ground-truth
 * rows `[class_id, cx, cy, w, h]` (5 values); the box is always the last four.
 */
import { boxRect, classIdOf, scoreOf } from "./predictions";

export const MAX_IMAGES = 8;
export const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
export const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/webp"];
export const DEFAULT_THRESHOLD = 0.25;
export const GT_COLOR = "#10b981";
export const PRED_COLOR = "#ef4444";

export interface InferImage {
  name: string;
  width: number;
  height: number;
  pred: number[][];
  gt?: number[][] | null;
  path?: string;
  split?: string | null;
}

export interface InferResponse {
  run_id: string;
  threshold: number;
  class_names: string[];
  device: { kind?: string; name?: string };
  device_note: string;
  images: InferImage[];
}

/** Client-side pre-check so the user gets the reason before any upload starts. */
export function checkFiles(files: File[]): { ok: File[]; problems: string[] } {
  const ok: File[] = [];
  const problems: string[] = [];
  for (const file of files) {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      problems.push(`${file.name}: only PNG, JPG or WEBP images`);
    } else if (file.size > MAX_IMAGE_BYTES) {
      problems.push(`${file.name}: ${(file.size / 1024 / 1024).toFixed(1)} MB is over the 10 MB limit`);
    } else if (ok.length >= MAX_IMAGES) {
      problems.push(`${file.name}: at most ${MAX_IMAGES} images at a time`);
    } else {
      ok.push(file);
    }
  }
  return { ok, problems };
}

export function visiblePredictions(
  image: InferImage,
  confidence: number,
  classFilter: string,
  classNames: string[],
): number[][] {
  return image.pred.filter(
    (box) =>
      scoreOf(box) >= confidence && (classFilter === "all" || classNames[classIdOf(box)] === classFilter),
  );
}

export function annotatedFilename(name: string): string {
  const stem = name.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9._-]+/g, "_") || "image";
  return `${stem}_detected.png`;
}

/** Draw the image and its visible boxes on a canvas and return PNG bytes. */
export async function drawAnnotated(
  imageUrl: string,
  image: InferImage,
  boxes: number[][],
  classNames: string[],
): Promise<Blob> {
  const element = new Image();
  element.src = imageUrl;
  await element.decode();
  const width = image.width || element.naturalWidth;
  const height = image.height || element.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas is not available in this browser.");
  context.drawImage(element, 0, 0, width, height);
  const label = Math.max(12, Math.round(width / 45));
  context.lineWidth = Math.max(2, width / 320);
  context.font = `${label}px sans-serif`;
  context.textBaseline = "top";
  for (const box of boxes) {
    const rect = boxRect(box, width, height);
    context.strokeStyle = PRED_COLOR;
    context.strokeRect(rect.x, rect.y, rect.w, rect.h);
    const text = `${classNames[classIdOf(box)] ?? classIdOf(box)} ${scoreOf(box).toFixed(2)}`;
    const textWidth = context.measureText(text).width + 6;
    const top = Math.max(0, rect.y - label - 2);
    context.fillStyle = PRED_COLOR;
    context.fillRect(rect.x, top, textWidth, label + 2);
    context.fillStyle = "#fff";
    context.fillText(text, rect.x + 3, top + 1);
  }
  return await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Could not encode PNG"))), "image/png"),
  );
}
