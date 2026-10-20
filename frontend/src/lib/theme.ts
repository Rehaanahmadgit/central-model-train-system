/*
 * Theme handling — client-side only.
 * The choice never leaves the browser and is sent nowhere [doc 10 §1: no cookies, no trackers].
 */
export type Theme = "light" | "dark";

const STORAGE_KEY = "trainhub-theme";

export function initialTheme(): Theme {
  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (stored === "light" || stored === "dark") {
    return stored;
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  window.localStorage.setItem(STORAGE_KEY, theme);
}
