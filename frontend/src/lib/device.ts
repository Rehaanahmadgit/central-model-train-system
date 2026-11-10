// Device banner logic for the Configure step [Slice F]. Pure + unit-tested in
// e2e/device-logic.spec.ts. "auto" means: GPU when one is visible (full preset),
// otherwise CPU with a clear warning.

export interface GpuInfo {
  available: boolean;
  name?: string;
  memory_gb?: number | null;
  message?: string;
}

export type DeviceNotice =
  | { kind: "gpu"; text: string }
  | { kind: "cpu-warning"; text: string }
  | { kind: "unknown"; text: string };

export function deviceNotice(
  gpu: GpuInfo | null | undefined,
  needsGpu: boolean,
  devPreset: boolean,
): DeviceNotice {
  if (!gpu) {
    return { kind: "unknown", text: "Device: could not read the GPU status; the server decides at start." };
  }
  if (gpu.available) {
    const mem = typeof gpu.memory_gb === "number" ? `, ${gpu.memory_gb} GB` : "";
    return {
      kind: "gpu",
      text: `Device: auto → GPU (${gpu.name ?? "GPU"}${mem}). ${devPreset ? "Dev preset is on: turn it off for a full run." : "Full preset."}`,
    };
  }
  return {
    kind: "cpu-warning",
    text:
      "No GPU detected — training will run on the CPU" +
      (needsGpu
        ? " with the tiny dev preset (a real model needs a GPU; start the container with --gpus all)."
        : ", which is fine for this model but slower than a GPU."),
  };
}

/** Auto device => preset: GPU runs the full preset, no GPU the dev preset. */
export function presetForDevice(gpu: GpuInfo | null | undefined): "full" | "dev" {
  return gpu?.available ? "full" : "dev";
}
