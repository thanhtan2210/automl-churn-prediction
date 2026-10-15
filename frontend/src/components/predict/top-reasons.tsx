"use client";

import { useState } from "react";

import { pct } from "@/lib/format";
import { commonReasons } from "@/lib/predict-analysis";
import type { PredictResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

/** The ten most common reasons among flagged (or all) customers; clicking one filters the table. */
export function TopReasons({
  result,
  edges,
  selected,
  onSelect,
}: {
  result: PredictResponse;
  edges: Record<string, number[] | null>;
  selected: string | null;
  onSelect: (label: string | null) => void;
}) {
  const [scope, setScope] = useState<"flagged" | "all">(result.n_flagged > 0 ? "flagged" : "all");
  const customers = scope === "flagged" ? result.predictions.filter((p) => p.flagged) : result.predictions;
  const rows = commonReasons(customers, result, edges);
  const max = Math.max(...rows.map((r) => r.share), 0.0001);

  return (
    <section className="space-y-4 rounded-2xl border bg-card p-6 shadow-xs">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Most common reasons</h2>
          <p className="text-sm text-muted-foreground">
            How often each reason is among a customer&apos;s top three. Number columns are grouped into the quarters
            of this file. Click a reason to list those customers.
          </p>
        </div>
        <div className="flex rounded-lg border p-0.5 text-sm" role="group" aria-label="Customers to count">
          {(["flagged", "all"] as const).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setScope(s)}
              aria-pressed={scope === s}
              className={cn("rounded-md px-3 py-1", scope === s ? "bg-primary text-primary-foreground" : "hover:bg-muted")}
            >
              {s === "flagged" ? `Flagged (${result.n_flagged.toLocaleString()})` : `All (${result.n_rows.toLocaleString()})`}
            </button>
          ))}
        </div>
      </div>
      {rows.length === 0 ? (
        <p className="text-sm text-muted-foreground">No reasons to show for these customers.</p>
      ) : (
        <ul className="space-y-1.5">
          {rows.map((r) => {
            const active = selected === r.label;
            return (
              <li key={r.label}>
                <button
                  type="button"
                  onClick={() => onSelect(active ? null : r.label)}
                  aria-pressed={active}
                  className={cn(
                    "grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 rounded-lg px-2 py-1.5 text-left text-sm transition-colors md:grid-cols-[minmax(0,16rem)_1fr_auto]",
                    active ? "bg-brand-soft ring-1 ring-primary/40" : "hover:bg-muted",
                  )}
                >
                  <span className="truncate font-medium">{r.label}</span>
                  <span className="order-last col-span-2 h-2.5 overflow-hidden rounded-full bg-muted md:order-none md:col-span-1" aria-hidden>
                    <span className="block h-full rounded-full bg-primary/70" style={{ width: `${(r.share / max) * 100}%` }} />
                  </span>
                  <span className="text-right tabular-nums text-muted-foreground">
                    {pct(r.share)} <span className="text-xs">({r.count.toLocaleString()})</span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
