import type { StreamStatus } from "../hooks/useRunStream";

interface ConnectionIndicatorProps {
  status: StreamStatus;
}

const LABELS: Record<StreamStatus, string> = {
  connecting: "Connecting",
  live: "Live",
  reconnecting: "Reconnecting",
  closed: "Offline",
};

// Status is never colour-only: dot + text [doc 09 §5].
export function ConnectionIndicator({ status }: ConnectionIndicatorProps) {
  const dotClasses: Record<StreamStatus, string> = {
    live: "bg-success animate-pulse",
    connecting: "bg-warning animate-pulse",
    reconnecting: "bg-warning animate-pulse",
    closed: "bg-text-secondary",
  };
  return (
    <span
      role="status"
      className="inline-flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1 text-xs text-text-secondary"
    >
      <span aria-hidden="true" className={`h-2 w-2 rounded-full ${dotClasses[status]}`} />
      {LABELS[status]}
    </span>
  );
}
