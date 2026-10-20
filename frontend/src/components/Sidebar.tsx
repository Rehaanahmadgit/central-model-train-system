import { useState } from "react";

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

interface SidebarProps {
  onSelect: (item: string) => void;
}

export function Sidebar({ onSelect }: SidebarProps) {
  const [notice, setNotice] = useState<string | null>(null);

  function select(item: string): void {
    if (item !== "Runs" && item !== "Datasets") {
      setNotice(`The ${item} screen arrives in a later slice.`);
    } else {
      setNotice(null);
    }
    onSelect(item);
  }

  return (
    <aside className="flex w-60 shrink-0 flex-col gap-2 border-r border-border bg-surface p-4">
      <div className="flex items-center gap-2 px-2 py-3">
        <span aria-hidden="true" className="h-3 w-3 rounded-sm bg-accent" />
        <span className="text-lg font-semibold">TrainHub</span>
      </div>
      <nav aria-label="Main navigation">
        <ul className="flex flex-col gap-1">
          {NAV_ITEMS.map((item) => (
            <li key={item}>
              <button
                type="button"
                onClick={() => select(item)}
                className="w-full rounded-md px-3 py-2 text-left text-sm text-text-secondary hover:bg-bg hover:text-text"
              >
                {item}
              </button>
            </li>
          ))}
        </ul>
      </nav>
      <div aria-live="polite" className="mt-auto px-3 py-2 text-xs text-text-secondary">
        {notice ?? ""}
      </div>
    </aside>
  );
}
