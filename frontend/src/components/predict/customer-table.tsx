"use client";

import { ArrowDownUp, ChevronLeft, ChevronRight, Download, Flag, Search, ShieldCheck, X } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { pct } from "@/lib/format";
import type { Range } from "@/lib/predict-analysis";
import type { Prediction, PredictResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

const PAGE_SIZE = 50;

export type Row = Prediction & { row: number; rank: number };
type SortKey = "rank" | "id";

function riskColour(p: number): string {
  // Continuous scale from "not flagged" green to "flagged" red.
  return `color-mix(in oklch, var(--risk-flagged) ${Math.round(p * 100)}%, var(--risk-safe))`;
}

function csvCell(value: string | number | boolean): string {
  const s = String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function downloadCsv(rows: Row[], topN: number, idColumn: string | null, fileName: string | null) {
  const header = [idColumn ?? "row", "rank", "churn_probability", "flagged", "in_top_n", "reason_1", "reason_2", "reason_3"];
  const lines = [...rows]
    .sort((a, b) => a.rank - b.rank)
    .map((p) =>
      [
        p.id,
        p.rank,
        p.probability.toFixed(4),
        p.flagged,
        p.rank <= topN,
        ...[0, 1, 2].map((i) => (p.reasons[i] ? `${p.reasons[i].feature} = ${p.reasons[i].value}` : "")),
      ]
        .map(csvCell)
        .join(","),
    );
  const blob = new Blob([[header.join(","), ...lines].join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `churn_scores_${fileName?.replace(/\.[^.]+$/, "") ?? "file"}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function FilterChip({ label, onClear }: { label: string; onClear: () => void }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-brand-soft px-3 py-1 text-sm text-primary">
      {label}
      <button type="button" onClick={onClear} aria-label={`Clear filter ${label}`} className="rounded-full hover:bg-primary/10">
        <X className="size-3.5" />
      </button>
    </span>
  );
}

export function CustomerTable({
  result,
  rows,
  allRows,
  topN,
  range,
  reason,
  onClearRange,
  onClearReason,
  onOpen,
  fileName,
}: {
  result: PredictResponse;
  rows: Row[];
  allRows: Row[];
  topN: number;
  range: Range | null;
  reason: string | null;
  onClearRange: () => void;
  onClearReason: () => void;
  onOpen: (row: number) => void;
  fileName: string | null;
}) {
  const [search, setSearch] = useState("");
  const [flaggedOnly, setFlaggedOnly] = useState(false);
  const [sort, setSort] = useState<{ key: SortKey; asc: boolean }>({ key: "rank", asc: true });
  const [page, setPage] = useState(0);

  const q = search.trim().toLowerCase();
  const filtered = rows
    .filter((p) => (!flaggedOnly || p.flagged) && (!q || p.id.toLowerCase().includes(q)))
    .sort((a, b) => {
      const d = sort.key === "rank" ? a.rank - b.rank : a.id.localeCompare(b.id, undefined, { numeric: true });
      return sort.asc ? d : -d;
    });
  const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const current = Math.min(page, pages - 1);
  const shown = filtered.slice(current * PAGE_SIZE, (current + 1) * PAGE_SIZE);
  const toggleSort = (key: SortKey) => {
    setSort((s) => ({ key, asc: s.key === key ? !s.asc : true }));
    setPage(0);
  };

  return (
    <section className="space-y-4 rounded-2xl border bg-card p-6 shadow-xs">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Customers</h2>
          <p className="text-sm text-muted-foreground">Click a customer to see what moves their risk.</p>
        </div>
        <Button variant="outline" size="sm" onClick={() => downloadCsv(allRows, topN, result.id_column, fileName)}>
          <Download aria-hidden />
          Download CSV
        </Button>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <label className="relative">
          <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
          <input
            type="search"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(0);
            }}
            placeholder={`Search ${result.id_column ?? "row number"}`}
            aria-label="Search by customer ID"
            className="h-9 w-56 rounded-lg border border-input bg-background pr-3 pl-9 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
          />
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            className="size-4 accent-primary"
            checked={flaggedOnly}
            onChange={(e) => {
              setFlaggedOnly(e.target.checked);
              setPage(0);
            }}
          />
          Flagged only
        </label>
        {range && <FilterChip label={`Risk ${pct(range[0])}–${pct(range[1])}`} onClear={onClearRange} />}
        {reason && <FilterChip label={reason} onClear={onClearReason} />}
        <span className="text-sm text-muted-foreground sm:ml-auto">
          {filtered.length.toLocaleString()} of {result.n_rows.toLocaleString()} customers
        </span>
      </div>

      <div className="overflow-x-auto rounded-xl border">
        <Table>
          <TableHeader>
            <TableRow className="bg-muted/40">
              <TableHead className="w-24">
                <button type="button" onClick={() => toggleSort("rank")} className="inline-flex items-center gap-1">
                  Rank <ArrowDownUp className="size-3.5" aria-hidden />
                </button>
              </TableHead>
              <TableHead>
                <button type="button" onClick={() => toggleSort("id")} className="inline-flex items-center gap-1">
                  {result.id_column ?? "Row"} <ArrowDownUp className="size-3.5" aria-hidden />
                </button>
              </TableHead>
              <TableHead className="min-w-44">Churn risk</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Top reasons</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {shown.map((p) => (
              <TableRow
                key={p.row}
                tabIndex={0}
                onClick={() => onOpen(p.row)}
                onKeyDown={(e) => e.key === "Enter" && onOpen(p.row)}
                className={cn("cursor-pointer", p.rank <= topN && "bg-brand-soft/50")}
              >
                <TableCell className="tabular-nums">
                  {p.rank}
                  {p.rank <= topN && <span className="ml-1.5 text-xs font-medium text-primary">top {topN}</span>}
                </TableCell>
                <TableCell className="font-medium">{p.id}</TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <div className="h-2 w-24 overflow-hidden rounded-full bg-muted" aria-hidden>
                      <div className="h-full rounded-full" style={{ width: `${p.probability * 100}%`, background: riskColour(p.probability) }} />
                    </div>
                    <span className="w-12 text-right tabular-nums">{pct(p.probability, 1)}</span>
                  </div>
                </TableCell>
                <TableCell>
                  {p.flagged ? (
                    <span className="inline-flex items-center gap-1 text-sm font-medium text-risk-flagged">
                      <Flag className="size-3.5" aria-hidden /> Flagged
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-sm text-risk-safe">
                      <ShieldCheck className="size-3.5" aria-hidden /> Not flagged
                    </span>
                  )}
                </TableCell>
                <TableCell className="whitespace-normal">
                  <div className="flex flex-wrap gap-1">
                    {p.reasons.length === 0 && <span className="text-xs text-muted-foreground">none</span>}
                    {p.reasons.map((r) => (
                      <span key={r.feature} className="rounded-md border bg-muted/60 px-2 py-0.5 text-xs">
                        {r.feature} = {r.value}
                      </span>
                    ))}
                  </div>
                </TableCell>
              </TableRow>
            ))}
            {shown.length === 0 && (
              <TableRow>
                <TableCell colSpan={5} className="py-10 text-center text-muted-foreground">
                  No customers match these filters.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      {pages > 1 && (
        <div className="flex items-center justify-end gap-2 text-sm">
          <Button variant="outline" size="sm" disabled={current === 0} onClick={() => setPage(current - 1)} aria-label="Previous page">
            <ChevronLeft aria-hidden />
          </Button>
          <span className="tabular-nums">
            Page {current + 1} of {pages}
          </span>
          <Button variant="outline" size="sm" disabled={current >= pages - 1} onClick={() => setPage(current + 1)} aria-label="Next page">
            <ChevronRight aria-hidden />
          </Button>
        </div>
      )}
    </section>
  );
}
