import { lazy, Suspense, useEffect, useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { EmptyState } from "./components/EmptyState";
import { RunsList } from "./pages/RunsList";
import { VerifyDemo } from "./pages/VerifyDemo";
import { applyTheme, initialTheme, type Theme } from "./lib/theme";

// Lazy-loaded so ECharts lands in a separate chunk [code-splitting].
const LiveRun = lazy(() =>
  import("./pages/LiveRun").then((module) => ({ default: module.LiveRun })),
);

type View = "home" | "runs" | "verify";

export default function App() {
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [view, setView] = useState<View>("home");
  const [openRunId, setOpenRunId] = useState<string | null>(null);
  const [wizardNotice, setWizardNotice] = useState(false);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  function toggleTheme(): void {
    setTheme((current) => (current === "light" ? "dark" : "light"));
  }

  function navigate(item: string): void {
    if (item === "Runs") {
      setOpenRunId(null);
      setView("runs");
    } else if (item === "Datasets") {
      setOpenRunId(null);
      setView("verify");
    } else {
      setOpenRunId(null);
      setView("home");
    }
  }

  return (
    <div className="flex min-h-screen bg-bg text-text">
      <Sidebar onSelect={navigate} />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar theme={theme} onToggleTheme={toggleTheme} />
        <main className="flex flex-1 flex-col gap-8 overflow-auto p-8">
          {openRunId ? (
            <Suspense fallback={<p className="text-text-secondary">Loading live run…</p>}>
              <LiveRun runId={openRunId} theme={theme} onBack={() => setOpenRunId(null)} />
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
          ) : (
            <>
              <header>
                <h1 className="text-3xl font-semibold">Home</h1>
                <p className="mt-1 text-text-secondary">
                  Projects and recent training runs appear here.
                </p>
              </header>
              {wizardNotice && (
                <p role="status" className="rounded-md border border-border bg-surface p-4 text-sm">
                  The 5-step wizard (Dataset → Model → Verify → Configure → Train) arrives in Slice 10
                  [doc 12 M2].
                </p>
              )}
              <EmptyState
                title="No runs yet"
                message="Upload a dataset, pick a model and start your first training run."
                actionLabel="New training"
                onAction={() => setWizardNotice(true)}
              />
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={() => navigate("Runs")}
                  className="self-start rounded-md border border-border px-4 py-2 text-sm hover:bg-bg"
                >
                  Open runs
                </button>
                <button
                  type="button"
                  onClick={() => navigate("Datasets")}
                  className="self-start rounded-md border border-border px-4 py-2 text-sm hover:bg-bg"
                >
                  Verify a dataset
                </button>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
