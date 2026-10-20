import type { Theme } from "../lib/theme";

interface TopBarProps {
  theme: Theme;
  onToggleTheme: () => void;
}

// Top bar: search, notifications, theme toggle [doc 09 §2]. No user menu.
export function TopBar({ theme, onToggleTheme }: TopBarProps) {
  return (
    <div className="flex items-center gap-4 border-b border-border bg-surface px-6 py-3">
      <input
        type="search"
        aria-label="Search runs and datasets"
        placeholder="Search runs, datasets…"
        className="w-full max-w-md rounded-md border border-border bg-bg px-3 py-2 text-sm"
      />
      <div className="ml-auto flex items-center gap-2">
        <button
          type="button"
          aria-label="Notifications (none yet)"
          className="rounded-md px-3 py-2 text-sm text-text-secondary hover:bg-bg hover:text-text"
        >
          Notifications
        </button>
        <button
          type="button"
          onClick={onToggleTheme}
          aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
          className="rounded-md border border-border px-3 py-2 text-sm hover:bg-bg"
        >
          {theme === "light" ? "Dark" : "Light"}
        </button>
      </div>
    </div>
  );
}
