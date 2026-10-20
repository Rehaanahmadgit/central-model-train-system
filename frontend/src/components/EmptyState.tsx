import { Button } from "./Button";

interface EmptyStateProps {
  title: string;
  message: string;
  actionLabel: string;
  onAction: () => void;
}

// Every list has an empty state with a call to action [doc 09 §6].
export function EmptyState({ title, message, actionLabel, onAction }: EmptyStateProps) {
  return (
    <section className="flex flex-col items-center gap-4 rounded-lg border border-dashed border-border bg-surface px-8 py-16 text-center">
      <h2 className="text-xl font-semibold">{title}</h2>
      <p className="max-w-md text-sm text-text-secondary">{message}</p>
      <Button onClick={onAction}>{actionLabel}</Button>
    </section>
  );
}
