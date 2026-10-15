"use client";

import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, Tooltip, XAxis, YAxis } from "recharts";

import { pct } from "@/lib/format";
import { binOf, histogram, type Range } from "@/lib/predict-analysis";
import type { PredictResponse } from "@/lib/types";

const FLAGGED = "var(--risk-flagged)";
const SAFE = "var(--risk-safe)";

/** Customers per 5-point risk band, split into flagged / not flagged; clicking a bar filters the table. */
export function RiskHistogram({
  result,
  selected,
  onSelect,
}: {
  result: PredictResponse;
  selected: Range | null;
  onSelect: (range: Range | null) => void;
}) {
  const bins = histogram(result.predictions);
  const cutoffBin = bins[binOf(result.threshold)];
  const pick = (index: number | undefined) => {
    if (index === undefined) return;
    const b = bins[index];
    onSelect(selected && selected[0] === b.lo ? null : [b.lo, b.hi]);
  };

  return (
    <section className="space-y-3 rounded-2xl border bg-card p-6 shadow-xs">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2 className="text-lg font-semibold">How risk is spread</h2>
          <p className="text-sm text-muted-foreground">
            Customers per risk band. Red = flagged (at or above the cut-off), green = not flagged. Click a bar to list
            those customers.
          </p>
        </div>
        {selected && (
          <button type="button" className="text-sm font-medium text-primary hover:underline" onClick={() => onSelect(null)}>
            Clear band filter ({pct(selected[0])}–{pct(selected[1])})
          </button>
        )}
      </div>
      <BarChart
        responsive
        style={{ width: "100%", height: 280 }}
        data={bins}
        margin={{ top: 24, right: 10, bottom: 20, left: 0 }}
        onClick={(state) => pick(typeof state?.activeTooltipIndex === "number" ? state.activeTooltipIndex : Number(state?.activeTooltipIndex))}
      >
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="label" interval={1} tick={{ fontSize: 11 }} label={{ value: "Churn risk", position: "insideBottom", offset: -12 }} />
        <YAxis allowDecimals={false} tick={{ fontSize: 12 }} width={40} />
        <Tooltip cursor={{ fill: "var(--muted)" }} />
        <Legend verticalAlign="top" align="right" height={24} />
        <ReferenceLine
          x={cutoffBin.label}
          stroke="var(--foreground)"
          strokeDasharray="4 4"
          label={{ value: `cut-off ${pct(result.threshold)}`, position: "top", fontSize: 12 }}
        />
        <Bar dataKey="notFlagged" name="Not flagged" stackId="risk" fill={SAFE} cursor="pointer" isAnimationActive={false}
          fillOpacity={1} />
        <Bar dataKey="flagged" name="Flagged" stackId="risk" fill={FLAGGED} radius={[4, 4, 0, 0]} cursor="pointer"
          isAnimationActive={false} />
      </BarChart>
    </section>
  );
}
