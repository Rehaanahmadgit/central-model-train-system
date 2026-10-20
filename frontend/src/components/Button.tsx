import type { MouseEventHandler } from "react";

interface ButtonProps {
  children: string;
  onClick?: MouseEventHandler<HTMLButtonElement>;
  variant?: "primary" | "secondary";
  disabled?: boolean;
  ariaLabel?: string;
}

// Shared button; semantic colours always carry text, never colour-only [doc 09 §5].
export function Button({ children, onClick, variant = "primary", disabled, ariaLabel }: ButtonProps) {
  const classes =
    variant === "primary"
      ? "bg-accent text-on-accent hover:bg-accent-hover"
      : "border border-border text-text hover:bg-bg";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${classes} disabled:opacity-50`}
    >
      {children}
    </button>
  );
}
