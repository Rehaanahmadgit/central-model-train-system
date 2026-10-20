/*
 * Shared API client with optional access-token support [doc 05 §0, FR-20].
 *
 * The token identifies nobody (doc 10 §2) and is kept per-viewer in
 * localStorage only — never sent anywhere except this origin [doc 10 §1].
 * If the server answers 401 (token required/mismatch), the viewer is prompted
 * once and the token is stored for the rest of the session.
 */

const TOKEN_KEY = "trainhub-access-token";
let prompted = false;

export function getAccessToken(): string | null {
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setAccessToken(token: string | null): void {
  if (token) window.localStorage.setItem(TOKEN_KEY, token);
  else window.localStorage.removeItem(TOKEN_KEY);
}

function authHeaders(): Record<string, string> {
  const token = getAccessToken();
  return token ? { "X-Access-Token": token } : {};
}

async function promptForToken(): Promise<string | null> {
  if (prompted) return null;
  prompted = true;
  const value = window.prompt("This instance requires an access token. Enter it:");
  if (value) setAccessToken(value.trim());
  prompted = false;
  return value ? value.trim() : null;
}

export async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...authHeaders() };
  if (options.body && !(options.headers as Record<string, string> | undefined)?.["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  let response = await fetch(path, { ...options, headers: { ...(options.headers as Record<string, string>), ...headers } });

  if (response.status === 401) {
    const token = await promptForToken();
    if (token) {
      headers["X-Access-Token"] = token;
      response = await fetch(path, { ...options, headers: { ...(options.headers as Record<string, string>), ...headers } });
    }
  }
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { error?: { message?: string } } | null;
    throw new Error(body?.error?.message ?? `Request failed (${response.status})`);
  }
  return (await response.json()) as T;
}

// WebSocket URL with the token as a query parameter [doc 05 §0: WS uses ?token=].
export function wsUrl(path: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const token = getAccessToken();
  const separator = path.includes("?") ? "&" : "?";
  return `${protocol}//${window.location.host}${path}${token ? `${separator}token=${encodeURIComponent(token)}` : ""}`;
}
