// Navigation per doc 09 §2. No login, no user menu, no profile [doc 10 §1].
const NAV_ITEMS = [
  "Projects",
  "Datasets",
  "Runs",
  "Compare",
  "Model Registry",
  "Checkpoints",
  "Audit Log",
  "System",
] as const;

// Screens that exist today; the rest are disabled "coming soon" [doc 12].
const ENABLED = new Set(["Datasets", "Runs", "Checkpoints"]);

interface SidebarProps {
  onSelect: (item: string) => void;
}

export function Sidebar({ onSelect }: SidebarProps) {
  return (
    <aside className="flex w-60 shrink-0 flex-col gap-2 border-r border-border bg-surface p-4">
      <div className="flex items-center gap-2 px-2 py-3">
        <span aria-hidden="true" className="h-3 w-3 rounded-sm bg-accent" />
        <span className="text-lg font-semibold">TrainHub</span>
      </div>
      <nav aria-label="Main navigation">
        <ul className="flex flex-col gap-1">
          {NAV_ITEMS.map((item) => {
            const enabled = ENABLED.has(item);
            return (
              <li key={item}>
                <button
                  type="button"
                  disabled={!enabled}
                  onClick={() => onSelect(item)}
                  aria-label={enabled ? item : `${item} (coming soon)`}
                  className={`flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-sm ${
                    enabled
                      ? "text-text-secondary hover:bg-bg hover:text-text"
                      : "cursor-not-allowed text-text-secondary/50"
                  }`}
                >
                  <span>{item}</span>
                  {!enabled && <span className="text-xs text-text-secondary">coming soon</span>}
                </button>
              </li>
            );
          })}
        </ul>
      </nav>
    </aside>
  );
}
