import { lazy, Suspense, useEffect, useState } from "react";
import { apiFetch } from "./lib/api";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { RunsList } from "./pages/RunsList";
import { VerifyDemo } from "./pages/VerifyDemo";
import { Wizard } from "./pages/Wizard";
import { CheckpointsPage } from "./pages/CheckpointsPage";
import { applyTheme, initialTheme, type Theme } from "./lib/theme";

// Lazy-loaded so ECharts lands in a separate chunk [code-splitting].
const LiveRun = lazy(() =>
  import("./pages/LiveRun").then((module) => ({ default: module.LiveRun })),
);

type View = "home" | "runs" | "verify" | "wizard" | "checkpoints";

export default function App() {
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [view, setView] = useState<View>("home");
  const [openRunId, setOpenRunId] = useState<string | null>(null);
  const [securityWarning, setSecurityWarning] = useState<string | null>(null);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  // DIST-07: surface the backend's security warning as a banner [doc 16].
  useEffect(() => {
    let disposed = false;
    apiFetch<{ security?: { warning?: string | null } }>("/api/v1/system")
      .then((body) => {
        if (!disposed && typeof body.security?.warning === "string") {
          setSecurityWarning(body.security.warning);
        }
      })
      .catch(() => undefined);
    return () => {
      disposed = true;
    };
  }, []);

  function toggleTheme(): void {
    setTheme((current) => (current === "light" ? "dark" : "light"));
  }

  function navigate(item: string): void {
    setOpenRunId(null);
    if (item === "Runs") setView("runs");
    else if (item === "Datasets") setView("verify");
    else if (item === "Checkpoints") setView("checkpoints");
    else setView("home");
  }

  return (
    <div className="flex min-h-screen bg-bg text-text">
      <Sidebar onSelect={navigate} />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar theme={theme} onToggleTheme={toggleTheme} />
        {securityWarning && (
          <div
            role="alert"
            className="border-b border-warning bg-warning/10 px-6 py-2 text-sm text-warning"
          >
            ⚠ {securityWarning}
          </div>
        )}
        <main className="flex flex-1 flex-col gap-8 overflow-auto p-8">
          {openRunId ? (
            <Suspense fallback={<p className="text-text-secondary">Loading live run…</p>}>
              <LiveRun
                runId={openRunId}
                theme={theme}
                onBack={() => setOpenRunId(null)}
                onNavigate={(runId) => setOpenRunId(runId)}
              />
            </Suspense>
          ) : view === "runs" ? (
            <section className="flex flex-col gap-4">
              <header>
                <h1 className="text-3xl font-semibold">Runs</h1>
                <p className="mt-1 text-text-secondary">
                  Every training run with its live state — click one to watch it.
                </p>
              </header>
              <RunsList onOpen={(runId) => setOpenRunId(runId)} />
            </section>
          ) : view === "verify" ? (
            <VerifyDemo />
          ) : view === "checkpoints" ? (
            <section className="flex flex-col gap-4">
              <header>
                <h1 className="text-3xl font-semibold">Checkpoints</h1>
                <p className="mt-1 text-text-secondary">
                  Pretrained checkpoints in the catalog, with local cache status.
                </p>
              </header>
              <CheckpointsPage />
            </section>
          ) : view === "wizard" ? (
            <Wizard
              onTrained={(runId) => {
                setOpenRunId(runId);
                setView("runs");
              }}
            />
          ) : (
            <>
              <header>
                <h1 className="text-3xl font-semibold">Home</h1>
                <p className="mt-1 text-text-secondary">
                  Projects and recent training runs appear here.
                </p>
              </header>
              <div className="flex flex-col items-center gap-4 rounded-lg border border-dashed border-border bg-surface px-8 py-16 text-center">
                <h2 className="text-xl font-semibold">Train your first model</h2>
                <p className="max-w-md text-sm text-text-secondary">
                  Upload a dataset, pick a model, verify it, configure and train — all in one flow.
                </p>
                <button
                  type="button"
                  onClick={() => setView("wizard")}
                  className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-on-accent hover:bg-accent-hover"
                >
                  New training
                </button>
                <button
                  type="button"
                  onClick={() => navigate("Runs")}
                  className="rounded-md border border-border px-4 py-2 text-sm hover:bg-bg"
                >
                  View runs
                </button>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
