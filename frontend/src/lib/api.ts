/*
 * Shared API client with access-token support [doc 05 §0, FR-20, Slice E].
 *
 * Auth is ON by default: the server generates a token on first boot (docker logs /
 * /data/access-token). The token identifies nobody and is kept per-viewer in
 * localStorage only — never sent anywhere except this origin [doc 10 §1].
 * A 401 raises AUTH_EVENT; <TokenPrompt /> shows the prompt (or, if a token was
 * already sent, a clear "rejected" banner).
 */

const TOKEN_KEY = "trainhub-access-token";
export const AUTH_EVENT = "trainhub:auth";

export interface AuthDetail {
  /** true = a token was sent and rejected; false = none was sent. */
  rejected: boolean;
  message: string;
}

export function getAccessToken(): string | null {
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setAccessToken(token: string | null): void {
  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage blocked: the prompt will simply reappear */
  }
}

function authHeaders(): Record<string, string> {
  const token = getAccessToken();
  return token ? { "X-Access-Token": token } : {};
}

/** Pure: what the prompt should say for a 401 body. Unit-tested. */
export function authDetailFrom(
  body: { error?: { code?: string; message?: string } } | null,
  sentToken: boolean,
): AuthDetail {
  const rejected = body?.error?.code === "ACCESS_TOKEN_INVALID" || sentToken;
  return {
    rejected,
    message:
      body?.error?.message ??
      (rejected
        ? "That access token was rejected (401)."
        : "This instance requires an access token."),
  };
}

async function failFrom(response: Response, sentToken: boolean, fallback: string): Promise<Error> {
  const body = (await response.json().catch(() => null)) as {
    error?: { code?: string; message?: string };
  } | null;
  if (response.status === 401) {
    const detail = authDetailFrom(body, sentToken);
    window.dispatchEvent(new CustomEvent<AuthDetail>(AUTH_EVENT, { detail }));
    return new Error(detail.message);
  }
  return new Error(body?.error?.message ?? `${fallback} (${response.status})`);
}

/** PUT a file as the raw request body (used by "Import from file"). */
export async function apiUploadFile<T>(path: string, file: File): Promise<T> {
  const response = await fetch(path, {
    method: "PUT",
    headers: { ...authHeaders(), "Content-Type": "application/octet-stream" },
    body: file,
  });
  if (!response.ok) throw await failFrom(response, getAccessToken() !== null, "Upload failed");
  return (await response.json()) as T;
}

export async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...authHeaders() };
  if (options.body && !(options.headers as Record<string, string> | undefined)?.["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  const response = await fetch(path, {
    ...options,
    headers: { ...(options.headers as Record<string, string>), ...headers },
  });
  if (!response.ok) throw await failFrom(response, "X-Access-Token" in headers, "Request failed");
  return (await response.json()) as T;
}

// WebSocket URL with the token as a query parameter [doc 05 §0: WS uses ?token=].
export function wsUrl(path: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const token = getAccessToken();
  const separator = path.includes("?") ? "&" : "?";
  return `${protocol}//${window.location.host}${path}${token ? `${separator}token=${encodeURIComponent(token)}` : ""}`;
}
