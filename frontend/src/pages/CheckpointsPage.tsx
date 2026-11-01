import { useEffect, useState } from "react";
import { apiFetch } from "../lib/api";

interface CheckpointItem {
  catalog_id: string;
  name: string;
  model_family: string;
  local: boolean;
  local_size_bytes: number | null;
  size_bytes: number;
  last_used_at: string | null;
  license: string;
}

interface CheckpointsResponse {
  items: CheckpointItem[];
  total: number;
}

function fmtBytes(value: number | null | undefined): string {
  if (typeof value !== "number" || value <= 0) return "—";
  if (value >= 1024 * 1024 * 1024) return `${(value / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${value} B`;
}

// Checkpoints page: catalog entries + local cache status [doc 05 §4, doc 07 §5].
export function CheckpointsPage() {
  const [items, setItems] = useState<CheckpointItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;
    apiFetch<CheckpointsResponse>("/api/v1/checkpoints")
      .then((body) => {
        if (!disposed) setItems(body.items);
      })
      .catch((exc) => {
        if (!disposed) setError(exc instanceof Error ? exc.message : "Could not load checkpoints.");
      });
    return () => {
      disposed = true;
    };
  }, []);

  if (error) {
    return (
      <p role="alert" className="rounded-md border border-danger bg-surface p-4 text-sm text-danger">
        {error}
      </p>
    );
  }
  if (items === null) {
    return <div className="h-32 animate-pulse rounded-lg border border-border bg-surface" aria-label="Loading checkpoints" />;
  }

  return (
    <section className="flex flex-col gap-4">
      <div className="overflow-auto rounded-lg border border-border bg-surface">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-text-secondary">
              <th className="p-3">Catalog id</th>
              <th className="p-3">Family</th>
              <th className="p-3">Local</th>
              <th className="p-3">Size</th>
              <th className="p-3">Last used</th>
              <th className="p-3">License</th>
            </tr>
          </thead>
          <tbody>
            {items.length === 0 && (
              <tr>
                <td colSpan={6} className="p-6 text-center text-text-secondary">
                  No checkpoints in the catalog yet.
                </td>
              </tr>
            )}
            {items.map((item) => (
              <tr key={item.catalog_id} className="border-b border-border last:border-b-0">
                <td className="p-3 font-mono">{item.catalog_id}</td>
                <td className="p-3 font-mono">{item.model_family}</td>
                <td className="p-3">
                  <span
                    className={`rounded-full px-2 py-0.5 text-xs ${
                      item.local ? "bg-success/10 text-success" : "bg-bg text-text-secondary"
                    }`}
                  >
                    {item.local ? "cached" : "remote"}
                  </span>
                </td>
                <td className="p-3 font-mono">{fmtBytes(item.local ? item.local_size_bytes : item.size_bytes)}</td>
                <td className="p-3 font-mono">
                  {item.last_used_at ? new Date(item.last_used_at).toLocaleString() : "—"}
                </td>
                <td className="p-3 font-mono">{item.license || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
