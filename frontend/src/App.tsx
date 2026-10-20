import { lazy, Suspense, useEffect, useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { EmptyState } from "./components/EmptyState";
import { VerifyDemo } from "./pages/VerifyDemo";
import { applyTheme, initialTheme, type Theme } from "./lib/theme";

// Lazy-loaded so ECharts lands in a separate chunk [code-splitting].
const LiveDemo = lazy(() =>
  import("./pages/LiveDemo").then((module) => ({ default: module.LiveDemo })),
);

type View = "home" | "demo" | "verify";

export default function App() {
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [view, setView] = useState<View>("home");
  const [wizardNotice, setWizardNotice] = useState(false);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  function toggleTheme(): void {
    setTheme((current) => (current === "light" ? "dark" : "light"));
  }

  return (
    <div className="flex min-h-screen bg-bg text-text">
      <Sidebar
        onSelect={(item) => {
          if (item === "Runs") {
            setView("demo");
          } else if (item === "Datasets") {
            setView("verify");
          } else {
            setView("home");
          }
        }}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar theme={theme} onToggleTheme={toggleTheme} />
        <main className="flex flex-1 flex-col gap-8 overflow-auto p-8">
          {view === "demo" ? (
            <Suspense fallback={<p className="text-text-secondary">Loading live view…</p>}>
              <LiveDemo theme={theme} />
            </Suspense>
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
                message="Upload a dataset and pick a model to start your first training run."
                actionLabel="New training"
                onAction={() => setWizardNotice(true)}
              />
              <button
                type="button"
                onClick={() => setView("demo")}
                className="self-start rounded-md border border-border px-4 py-2 text-sm hover:bg-bg"
              >
                Open live demo chart
              </button>
              <button
                type="button"
                onClick={() => setView("verify")}
                className="self-start rounded-md border border-border px-4 py-2 text-sm hover:bg-bg"
              >
                Open verification demo
              </button>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
