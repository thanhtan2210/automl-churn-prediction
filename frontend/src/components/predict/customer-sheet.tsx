"use client";

import { ArrowDown, ArrowUp, CircleHelp, Flag, ShieldCheck } from "lucide-react";

import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { pct } from "@/lib/format";
import { binOf, histogram } from "@/lib/predict-analysis";
import type { Prediction, PredictResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

function showValue(v: number | string | null): string {
  if (v === null) return "empty";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return v;
}

/** Where the customer sits among everyone scored: the risk bands, with the customer's band highlighted. */
function RiskPosition({ result, customer }: { result: PredictResponse; customer: Prediction }) {
  const bins = histogram(result.predictions);
  const max = Math.max(...bins.map((b) => b.flagged + b.notFlagged), 1);
  const mine = binOf(customer.probability);
  return (
    <div className="space-y-1.5">
      <div className="flex h-16 items-end gap-0.5" aria-hidden>
        {bins.map((b) => (
          <div
            key={b.index}
            className={cn(
              "flex-1 rounded-t-sm",
              b.index === mine ? "bg-foreground" : b.lo >= result.threshold ? "bg-risk-flagged/35" : "bg-risk-safe/35",
            )}
            style={{ height: `${Math.max(4, ((b.flagged + b.notFlagged) / max) * 100)}%` }}
          />
        ))}
      </div>
      <div className="relative h-4 text-[11px] text-muted-foreground">
        <span className="absolute left-0">0%</span>
        <span className="absolute -translate-x-1/2" style={{ left: `${result.threshold * 100}%` }}>
          cut-off {pct(result.threshold)}
        </span>
        <span className="absolute right-0">100%</span>
      </div>
      <p className="text-xs text-muted-foreground">The dark bar is this customer&apos;s risk band among all customers scored.</p>
    </div>
  );
}

export function CustomerSheet({
  result,
  customer,
  rank,
  onClose,
}: {
  result: PredictResponse;
  customer: Prediction | null;
  rank: number | null;
  onClose: () => void;
}) {
  const items = customer
    ? result.features
        .map((feature, j) => ({ feature, value: customer.values[j], effect: customer.contributions[j] }))
        .sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect))
    : [];
  const maxAbs = Math.max(...items.map((i) => Math.abs(i.effect)), 1e-9);

  return (
    <Sheet open={customer !== null} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="overflow-y-auto data-[side=right]:w-full data-[side=right]:sm:max-w-xl">
        {customer && (
          <>
            <SheetHeader>
              <SheetTitle className="text-xl">
                {result.id_column ? `${result.id_column} ${customer.id}` : `Row ${customer.id}`}
              </SheetTitle>
              <SheetDescription>
                Ranked {rank?.toLocaleString()} of {result.n_rows.toLocaleString()} by churn risk.
              </SheetDescription>
            </SheetHeader>
            <div className="space-y-6 px-4 pb-8">
              <div className="flex items-center gap-4">
                <div className="text-4xl font-semibold tabular-nums">{pct(customer.probability, 1)}</div>
                {customer.flagged ? (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-risk-flagged-soft px-3 py-1 text-sm font-medium text-risk-flagged">
                    <Flag className="size-4" aria-hidden /> Flagged
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-risk-safe-soft px-3 py-1 text-sm font-medium text-risk-safe">
                    <ShieldCheck className="size-4" aria-hidden /> Not flagged
                  </span>
                )}
              </div>
              <RiskPosition result={result} customer={customer} />

              <div className="space-y-2">
                <div className="flex items-center gap-1.5">
                  <h3 className="font-semibold">What moves this customer&apos;s risk</h3>
                  <Tooltip>
                    <TooltipTrigger render={<span className="cursor-help text-muted-foreground" />}>
                      <CircleHelp className="size-4" aria-label="About these bars" />
                    </TooltipTrigger>
                    <TooltipContent>
                      Each bar is the column&apos;s effect on the model&apos;s risk score for this customer (LightGBM
                      contributions, in log-odds). Together with the model&apos;s starting point they add up to the score.
                    </TooltipContent>
                  </Tooltip>
                </div>
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span className="flex items-center gap-1 text-risk-safe">
                    <ArrowDown className="size-3" aria-hidden /> pushes risk down
                  </span>
                  <span className="flex items-center gap-1 text-risk-flagged">
                    pushes risk up <ArrowUp className="size-3" aria-hidden />
                  </span>
                </div>
                <ul className="space-y-1">
                  {items.map((i) => {
                    const width = (Math.abs(i.effect) / maxAbs) * 50;
                    return (
                      <li key={i.feature} className="grid grid-cols-[minmax(0,10rem)_1fr] items-center gap-3 text-sm">
                        <span className="min-w-0">
                          <span className="block truncate font-medium">{i.feature}</span>
                          <span className="block truncate text-xs text-muted-foreground">{showValue(i.value)}</span>
                        </span>
                        <span className="relative h-4" aria-label={`${i.effect > 0 ? "pushes risk up" : "pushes risk down"}`}>
                          <span className="absolute inset-y-0 left-1/2 w-px bg-border" />
                          <span
                            className={cn("absolute inset-y-0.5 rounded-sm", i.effect > 0 ? "bg-risk-flagged" : "bg-risk-safe")}
                            style={i.effect > 0 ? { left: "50%", width: `${width}%` } : { right: "50%", width: `${width}%` }}
                          />
                        </span>
                      </li>
                    );
                  })}
                </ul>
              </div>
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}
