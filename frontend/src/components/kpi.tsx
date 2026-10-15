import type { LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

/** A big number with a short label and an icon. */
export function Kpi({
  icon: Icon,
  value,
  label,
  hint,
  tone = "brand",
  className,
}: {
  icon: LucideIcon;
  value: React.ReactNode;
  label: React.ReactNode;
  hint?: React.ReactNode;
  tone?: "brand" | "flagged" | "safe" | "neutral";
  className?: string;
}) {
  const toneClass = {
    brand: "bg-brand-soft text-primary",
    flagged: "bg-risk-flagged-soft text-risk-flagged",
    safe: "bg-risk-safe-soft text-risk-safe",
    neutral: "bg-muted text-muted-foreground",
  }[tone];
  return (
    <div className={cn("flex gap-4 rounded-2xl border bg-card p-5 shadow-xs", className)}>
      <span className={cn("flex size-11 shrink-0 items-center justify-center rounded-xl", toneClass)}>
        <Icon className="size-5" aria-hidden />
      </span>
      <div className="min-w-0">
        <div className="text-3xl font-semibold tracking-tight tabular-nums">{value}</div>
        <div className="mt-0.5 text-sm font-medium">{label}</div>
        {hint && <div className="mt-1 text-xs text-muted-foreground">{hint}</div>}
      </div>
    </div>
  );
}
