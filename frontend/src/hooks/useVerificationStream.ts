/*
 * useVerificationStream — WebSocket stream for verification progress
 * [doc 05 §3, doc 09 §5]. Snapshot first, then check events; the server
 * closes the stream when the verification reaches a terminal state.
 */

import { useEffect, useRef, useState } from "react";

export type CheckStatus = "pass" | "warn" | "fail" | "skipped" | "running";

export interface CheckItem {
  name: string;
  severity: string;
  status: CheckStatus;
  message: string;
  details: Record<string, unknown>;
  fix_hint: string | null;
  samples: string[];
}

export interface Derived {
  task_type?: string | null;
  task_detection_reason?: string | null;
  format?: string | null;
  num_classes?: number;
  classes?: string[];
  class_counts?: Record<string, number>;
  num_images?: number;
  num_samples?: Record<string, number>;
  image_stats?: {
    min_wh?: [number, number];
    max_wh?: [number, number];
    median_w?: number | null;
    median_h?: number | null;
    color_modes?: Record<string, number>;
  };
  object_stats?: Record<string, unknown>;
  recommended_hparams?: Record<string, unknown>;
  recommended_checkpoints?: Array<Record<string, unknown>>;
  checkpoint_catalog_configured?: boolean;
}

export interface VerificationStream {
  status: "connecting" | "live" | "closed";
  verificationStatus: string | null;
  checks: CheckItem[];
  derived: Derived | null;
  cached: boolean;
}

const MAX_RECONNECT_DELAY_MS = 15_000;

type AnyMessage = {
  type: string;
  data?: Record<string, unknown>;
};

export function useVerificationStream(verificationId: string | null): VerificationStream {
  const [state, setState] = useState<VerificationStream>({
    status: verificationId ? "connecting" : "closed",
    verificationStatus: null,
    checks: [],
    derived: null,
    cached: false,
  });

  const buffer = useRef<VerificationStream>({
    status: "connecting",
    verificationStatus: null,
    checks: [],
    derived: null,
    cached: false,
  });
  const dirty = useRef(false);

  useEffect(() => {
    if (!verificationId) {
      setState({
        status: "closed",
        verificationStatus: null,
        checks: [],
        derived: null,
        cached: false,
      });
      return;
    }
    buffer.current = {
      status: "connecting",
      verificationStatus: null,
      checks: [],
      derived: null,
      cached: false,
    };
    dirty.current = true;

    let socket: WebSocket | null = null;
    let disposed = false;
    let retries = 0;
    let reconnectTimer: number | undefined;
    let terminal = false;

    const flush = (): void => {
      if (!dirty.current) return;
      dirty.current = false;
      setState({ ...buffer.current, checks: [...buffer.current.checks] });
    };
    const flushTimer = window.setInterval(flush, 250);

    const upsertCheck = (name: string, patch: Partial<CheckItem>): void => {
      const current = buffer.current.checks;
      const existing = current.find((check) => check.name === name);
      if (existing) {
        Object.assign(existing, patch);
      } else {
        current.push({
          name,
          severity: "info",
          status: "running",
          message: "",
          details: {},
          fix_hint: null,
          samples: [],
          ...patch,
        });
      }
    };

    const handleMessage = (raw: string): void => {
      let message: AnyMessage;
      try {
        message = JSON.parse(raw) as AnyMessage;
      } catch {
        return;
      }
      const data = message.data ?? {};
      if (message.type === "snapshot") {
        const verification = data.verification as Record<string, unknown> | null | undefined;
        buffer.current.verificationStatus =
          typeof verification?.status === "string" ? verification.status : null;
        const checks = data.checks;
        if (Array.isArray(checks)) {
          buffer.current.checks = checks as CheckItem[];
        }
        const derivedFacts = verification?.derived;
        if (derivedFacts && typeof derivedFacts === "object") {
          buffer.current.derived = derivedFacts as Derived;
        }
        if (typeof verification?.cached === "boolean") {
          buffer.current.cached = verification.cached;
        }
      } else if (message.type === "check_started") {
        upsertCheck(String(data.name ?? "check"), { status: "running" });
      } else if (message.type === "check") {
        upsertCheck(String(data.name ?? "check"), data as Partial<CheckItem>);
      } else if (message.type === "state") {
        buffer.current.verificationStatus = typeof data.to === "string" ? data.to : null;
      }
      dirty.current = true;
    };

    const connect = (): void => {
      if (disposed || terminal) return;
      buffer.current.status = retries === 0 ? "connecting" : "connecting";
      dirty.current = true;
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(
        `${protocol}//${window.location.host}/ws/verifications/${verificationId}`,
      );
      socket.onopen = () => {
        retries = 0;
        buffer.current.status = "live";
        dirty.current = true;
      };
      socket.onmessage = (event) => handleMessage(String(event.data));
      socket.onclose = () => {
        socket = null;
        if (disposed) return;
        const finished = buffer.current.verificationStatus !== null;
        if (finished) {
          terminal = true;
          buffer.current.status = "closed";
          dirty.current = true;
          flush();
          return;
        }
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
  }, [verificationId]);

  return state;
}
