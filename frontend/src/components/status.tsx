import { AlertTriangle, Loader2, type LucideIcon } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";

export function Loading({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 py-4 text-sm text-muted-foreground" role="status">
      <Loader2 className="size-4 animate-spin" aria-hidden />
      {label}
    </div>
  );
}

/** Placeholder blocks while a page's data loads. */
export function PageSkeleton({ label, blocks = 3 }: { label: string; blocks?: number }) {
  return (
    <div className="space-y-4" role="status" aria-label={label}>
      <div className="grid gap-4 md:grid-cols-3">
        <Skeleton className="h-28 rounded-2xl" />
        <Skeleton className="h-28 rounded-2xl" />
        <Skeleton className="h-28 rounded-2xl" />
      </div>
      {Array.from({ length: blocks }, (_, i) => (
        <Skeleton key={i} className="h-48 rounded-2xl" />
      ))}
      <p className="text-sm text-muted-foreground">{label}</p>
    </div>
  );
}

/** Shows the API's own message (kept as is) under a friendly title. */
export function ErrorAlert({ error, title = "Something went wrong" }: { error: unknown; title?: string }) {
  if (!error) return null;
  const message = error instanceof Error ? error.message : String(error);
  return (
    <Alert variant="destructive" className="border-warn/30 bg-warn-soft">
      <AlertTriangle aria-hidden />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>{message}</AlertDescription>
    </Alert>
  );
}

export function EmptyState({
  icon: Icon,
  title,
  children,
}: {
  icon: LucideIcon;
  title: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed bg-background px-6 py-12 text-center">
      <span className="flex size-12 items-center justify-center rounded-full bg-brand-soft text-primary">
        <Icon className="size-6" aria-hidden />
      </span>
      <p className="text-base font-semibold">{title}</p>
      {children && <div className="max-w-md text-sm text-muted-foreground">{children}</div>}
    </div>
  );
}
