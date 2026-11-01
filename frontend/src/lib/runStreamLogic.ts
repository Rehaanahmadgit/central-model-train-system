/*
 * Pure replay-dedupe logic for the run WebSocket stream [doc 05 §8].
 *
 * The server always sends a snapshot (recent metrics + events + state) before
 * replaying missed events from `last_seq`. Replayed `metric`/`event`/`state`
 * records with `seq <= snapshot.last_seq` are already reflected in the
 * snapshot, so the client skips them to avoid duplicate chart points and a
 * state downgrade. `log` and `system` never appear in the snapshot, so they
 * are never skipped.
 */

/*
 * The server replays a finished run's events and then closes the socket
 * [doc 05 §8]. Reconnecting is pointless there: the client would flip to
 * "Reconnecting" forever, which is exactly the misleading badge we removed.
 */
export const TERMINAL_RUN_STATES = ["completed", "failed", "stopped", "interrupted"] as const;

export function isTerminalRunState(state: string | null | undefined): boolean {
  return typeof state === "string" && (TERMINAL_RUN_STATES as readonly string[]).includes(state);
}

export function isDuplicateReplay(
  type: string,
  seq: number | null | undefined,
  snapshotSeen: number | null | undefined,
): boolean {
  if (typeof seq !== "number" || typeof snapshotSeen !== "number") {
    return false;
  }
  if (type === "log" || type === "system") {
    return false;
  }
  if (type === "metric" || type === "event" || type === "state") {
    return seq <= snapshotSeen;
  }
  return false;
}
