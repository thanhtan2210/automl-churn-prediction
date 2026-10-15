"use client";

import { Layers } from "lucide-react";
import { useState } from "react";
import { Bar, BarChart, LabelList, Tooltip, XAxis, YAxis } from "recharts";

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { pct } from "@/lib/format";
import { segments } from "@/lib/predict-analysis";
import type { ModelMetrics, PredictResponse } from "@/lib/types";

const selectClass =
  "h-9 rounded-lg border border-input bg-background px-3 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50";

/** Flag rate and average risk per value of one category column, computed from this file's scores. */
export function SegmentBreakdown({ result, metrics }: { result: PredictResponse; metrics: ModelMetrics }) {
  const categorical = result.features.filter((f) => !result.numeric_features.includes(f));
  const byImportance = metrics.global_importance_top10.map((g) => g.feature).find((f) => categorical.includes(f));
  const [feature, setFeature] = useState<string | null>(byImportance ?? categorical[0] ?? null);

  if (!feature) {
    return (
      <section className="rounded-2xl border bg-card p-6 text-sm text-muted-foreground shadow-xs">
        <h2 className="mb-1 text-lg font-semibold text-foreground">Compare customer groups</h2>
        This model only uses number columns, so there are no category groups to compare.
      </section>
    );
  }

  const rows = segments(result, feature);
  return (
    <section className="space-y-4 rounded-2xl border bg-card p-6 shadow-xs">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-brand-soft text-primary">
            <Layers className="size-5" aria-hidden />
          </span>
          <div>
            <h2 className="text-lg font-semibold">Compare customer groups</h2>
            <p className="text-sm text-muted-foreground">Share of each group flagged as likely to leave.</p>
          </div>
        </div>
        <label className="flex items-center gap-2 text-sm">
          Group by
          <select className={selectClass} value={feature} onChange={(e) => setFeature(e.target.value)}>
            {categorical.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <BarChart
          responsive
          style={{ width: "100%", height: Math.max(160, rows.length * 44) }}
          data={rows.map((r) => ({ ...r, flaggedPct: r.flaggedShare * 100 }))}
          layout="vertical"
          margin={{ top: 0, right: 48, bottom: 0, left: 10 }}
        >
          <XAxis type="number" domain={[0, 100]} hide />
          <YAxis type="category" dataKey="value" width={140} tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
          <Tooltip formatter={(v) => `${Number(v).toFixed(0)}% flagged`} cursor={{ fill: "var(--muted)" }} />
          <Bar dataKey="flaggedPct" name="Flagged" fill="var(--risk-flagged)" radius={[0, 6, 6, 0]} barSize={22} isAnimationActive={false}>
            <LabelList dataKey="flaggedPct" position="right" formatter={(v) => `${Number(v).toFixed(0)}%`} className="fill-foreground text-xs" />
          </Bar>
        </BarChart>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{feature}</TableHead>
                <TableHead className="text-right">Customers</TableHead>
                <TableHead className="text-right">Flagged</TableHead>
                <TableHead className="text-right">Average risk</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r) => (
                <TableRow key={r.value}>
                  <TableCell className="font-medium whitespace-normal">{r.value}</TableCell>
                  <TableCell className="text-right tabular-nums">{r.count.toLocaleString()}</TableCell>
                  <TableCell className="text-right tabular-nums">{pct(r.flaggedShare)}</TableCell>
                  <TableCell className="text-right tabular-nums">{pct(r.meanProbability)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </section>
  );
}
