import { useEffect, useState } from "react";
import { AUTH_EVENT, setAccessToken, type AuthDetail } from "../lib/api";

/**
 * Blocking token prompt [Slice E]. Shown when any API call answers 401. A wrong
 * token gets a clear "rejected (401)" banner instead of a silent failure; the
 * token is stored in this browser only and the page reloads to retry.
 */
export function TokenPrompt() {
  const [detail, setDetail] = useState<AuthDetail | null>(null);
  const [value, setValue] = useState("");

  useEffect(() => {
    function onAuth(event: Event): void {
      setDetail((event as CustomEvent<AuthDetail>).detail);
    }
    window.addEventListener(AUTH_EVENT, onAuth);
    return () => window.removeEventListener(AUTH_EVENT, onAuth);
  }, []);

  if (!detail) return null;

  function submit(event: React.FormEvent): void {
    event.preventDefault();
    const token = value.trim();
    if (!token) return;
    setAccessToken(token);
    window.location.reload();
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="token-prompt-title"
      data-testid="token-prompt"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
    >
      <form onSubmit={submit} className="flex w-full max-w-md flex-col gap-3 rounded-lg border border-border bg-surface p-6 shadow-lg">
        <h2 id="token-prompt-title" className="text-lg font-semibold">Access token required</h2>
        {detail.rejected && (
          <p role="alert" data-testid="token-rejected" className="rounded-md border border-danger bg-danger/10 p-2 text-sm text-danger">
            {detail.message}
          </p>
        )}
        <p className="text-sm text-text-secondary">
          Paste the token printed in the docker logs on first boot, or read it on the
          server with <code className="font-mono">docker exec trainhub cat /data/access-token</code>.
          Lost it? Run <code className="font-mono">docker exec trainhub trainhub --reset-token</code>{" "}
          and restart the container.
        </p>
        <input
          type="password"
          autoFocus
          aria-label="Access token"
          data-testid="token-input"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          className="rounded-md border border-border bg-bg px-3 py-2 font-mono text-sm"
        />
        <button type="submit" data-testid="token-submit" className="self-end rounded-md bg-accent px-4 py-2 text-sm font-medium text-on-accent hover:bg-accent-hover">
          Unlock
        </button>
      </form>
    </div>
  );
}
