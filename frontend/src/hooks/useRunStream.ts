/*
 * useRunStream — WebSocket run stream with auto-reconnect, last_seq resume
 * and ~4 fps state flushes [doc 09 §5]. The last seen seq is kept in
 * localStorage (client-side only; nothing is sent to the server [doc 10 §1]).
 */

import { useEffect, useRef, useState } from "react";
import { wsUrl } from "../lib/api";

export type StreamStatus = "connecting" | "live" | "reconnecting" | "closed";

export interface MetricPoint {
  name: string;
  value: number;
  step: number;
  epoch?: number;
}

export interface LogLine {
  ts: string;
  level: string;
  message: string;
}

export interface TimelineEvent {
  ts: string;
  name: string;
  data: Record<string, unknown>;
}

export interface SystemStat {
  gpu_available: boolean | null;
  gpu_util: number | null;
  gpu_mem_mb: number | null;
  cpu: number | null;
  ram_mb: number | null;
}

export interface StreamState {
  status: StreamStatus;
  lastSeq: number;
  runState: string | null;
  metrics: MetricPoint[];
  logs: LogLine[];
  events: TimelineEvent[];
  system: SystemStat | null;
}

const MAX_METRICS = 500;
const MAX_LOGS = 200;
const MAX_EVENTS = 100;
const FLUSH_MS = 250; // ~4 fps rendering throttle [doc 09 §5]
const MAX_RECONNECT_DELAY_MS = 15_000;

type AnyMessage = {
  type: string;
  ts?: string | null;
  seq?: number | null;
  data?: Record<string, unknown>;
};

function isMetricPoint(value: unknown): value is MetricPoint {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return typeof candidate.name === "string" && typeof candidate.value === "number";
}

export function useRunStream(runId: string | null): StreamState {
  const [state, setState] = useState<StreamState>({
    status: runId ? "connecting" : "closed",
    lastSeq: 0,
    runState: null,
    metrics: [],
    logs: [],
    events: [],
    system: null,
  });

  const buffers = useRef<{
    metrics: MetricPoint[];
    logs: LogLine[];
    events: TimelineEvent[];
    runState: string | null;
    lastSeq: number;
    status: StreamStatus;
    system: SystemStat | null;
  }>({ metrics: [], logs: [], events: [], runState: null, lastSeq: 0, status: "connecting", system: null });
  const dirty = useRef(false);

  useEffect(() => {
    if (!runId) {
      setState((current) => ({ ...current, status: "closed" }));
      return;
    }
    const seqKey = `trainhub-last-seq-${runId}`;
    const storedSeq = Number(window.localStorage.getItem(seqKey) ?? "0") || 0;
    buffers.current = {
      metrics: [],
      logs: [],
      events: [],
      runState: null,
      lastSeq: storedSeq,
      status: "connecting",
      system: null,
    };
    dirty.current = true;

    let socket: WebSocket | null = null;
    let disposed = false;
    let retries = 0;
    let reconnectTimer: number | undefined;
    const flushTimer = window.setInterval(() => {
      if (!dirty.current) return;
      dirty.current = false;
      const buffer = buffers.current;
      setState({
        status: buffer.status,
        lastSeq: buffer.lastSeq,
        runState: buffer.runState,
        metrics: buffer.metrics.slice(-MAX_METRICS),
        logs: buffer.logs.slice(-MAX_LOGS),
        events: buffer.events.slice(-MAX_EVENTS),
        system: buffer.system,
      });
    }, FLUSH_MS);

    const applySnapshot = (data: Record<string, unknown>): void => {
      const buffer = buffers.current;
      const rawMetrics = data.metrics;
      if (Array.isArray(rawMetrics)) {
        buffer.metrics = rawMetrics
          .map((entry) => (entry as Record<string, unknown>).data)
          .filter(isMetricPoint);
      }
      const rawEvents = data.events;
      if (Array.isArray(rawEvents)) {
        buffer.events = rawEvents
          .map((entry) => {
            const record = entry as Record<string, unknown>;
            const eventData = record.data as Record<string, unknown>;
            return {
              ts: String(record.ts ?? ""),
              name: String(eventData?.name ?? record.type ?? "event"),
              data: eventData ?? {},
            };
          })
          .filter((event) => event.name.length > 0);
      }
      buffer.runState = typeof data.state === "string" ? data.state : null;
      dirty.current = true;
    };

    const handleMessage = (raw: string): void => {
      let message: AnyMessage;
      try {
        message = JSON.parse(raw) as AnyMessage;
      } catch {
        return;
      }
      const buffer = buffers.current;
      if (typeof message.seq === "number" && message.seq > buffer.lastSeq) {
        buffer.lastSeq = message.seq;
        window.localStorage.setItem(seqKey, String(message.seq));
      }
      const data = message.data ?? {};
      if (message.type === "snapshot") {
        applySnapshot(data);
        return;
      }
      if (message.type === "metric") {
        const batch = data.batch;
        if (Array.isArray(batch)) {
          buffer.metrics.push(...batch.filter(isMetricPoint));
        } else if (isMetricPoint(data)) {
          buffer.metrics.push(data);
        }
        buffer.metrics = buffer.metrics.slice(-MAX_METRICS);
      } else if (message.type === "log") {
        buffer.logs.push({
          ts: String(message.ts ?? ""),
          level: String(data.level ?? "INFO"),
          message: String(data.message ?? ""),
        });
        buffer.logs = buffer.logs.slice(-MAX_LOGS);
      } else if (message.type === "state") {
        buffer.runState = typeof data.to === "string" ? data.to : buffer.runState;
      } else if (message.type === "event") {
        buffer.events.push({
          ts: String(message.ts ?? ""),
          name: String(data.name ?? "event"),
          data,
        });
        buffer.events = buffer.events.slice(-MAX_EVENTS);
      } else if (message.type === "system") {
        buffer.system = {
          gpu_available: typeof data.gpu_available === "boolean" ? data.gpu_available : null,
          gpu_util: typeof data.gpu_util === "number" ? data.gpu_util : null,
          gpu_mem_mb: typeof data.gpu_mem_mb === "number" ? data.gpu_mem_mb : null,
          cpu: typeof data.cpu === "number" ? data.cpu : null,
          ram_mb: typeof data.ram_mb === "number" ? data.ram_mb : null,
        };
      }
      dirty.current = true;
    };

    const connect = (): void => {
      if (disposed) return;
      buffers.current.status = retries === 0 ? "connecting" : "reconnecting";
      dirty.current = true;
      socket = new WebSocket(wsUrl(`/ws/runs/${runId}`));
      socket.onopen = () => {
        retries = 0;
        buffers.current.status = "live";
        dirty.current = true;
        socket?.send(JSON.stringify({ type: "resume", last_seq: buffers.current.lastSeq }));
      };
      socket.onmessage = (event) => handleMessage(String(event.data));
      socket.onclose = () => {
        socket = null;
        if (disposed) return;
        buffers.current.status = "reconnecting";
        dirty.current = true;
        retries += 1;
        const delay = Math.min(1000 * 2 ** (retries - 1), MAX_RECONNECT_DELAY_MS);
        reconnectTimer = window.setTimeout(connect, delay);
      };
      socket.onerror = () => {
        socket?.close();
      };
    };

    connect();
    return () => {
      disposed = true;
      window.clearInterval(flushTimer);
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [runId]);

  return state;
}
