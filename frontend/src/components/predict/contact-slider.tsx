"use client";

import { PhoneCall } from "lucide-react";

import { Slider } from "@/components/ui/slider";
import { pct } from "@/lib/format";
import { contactEstimate } from "@/lib/predict-analysis";
import type { ModelMetrics, PredictResponse } from "@/lib/types";

/** Pick how many top-risk customers the team can contact; estimate their outcome from the test split. */
export function ContactSlider({
  result,
  metrics,
  topN,
  onChange,
}: {
  result: PredictResponse;
  metrics: ModelMetrics;
  topN: number;
  onChange: (n: number) => void;
}) {
  const total = result.n_rows;
  const est = contactEstimate(metrics, topN, total);

  return (
    <section className="space-y-5 rounded-2xl border bg-card p-6 shadow-xs">
      <div className="flex items-start gap-3">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-brand-soft text-primary">
          <PhoneCall className="size-5" aria-hidden />
        </span>
        <div>
          <h2 className="text-lg font-semibold">How many customers can your team contact?</h2>
          <p className="text-sm text-muted-foreground">
            Starts at the number of flagged customers. The top customers by risk are marked in the list below.
          </p>
        </div>
      </div>

      <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <Slider
          className="flex-1"
          min={0}
          max={total}
          step={1}
          value={[topN]}
          onValueChange={(v) => onChange(Array.isArray(v) ? v[0] : v)}
          aria-label="Customers to contact"
        />
        <div className="flex items-baseline gap-2 sm:w-56 sm:justify-end">
          <span className="text-3xl font-semibold tabular-nums">{topN.toLocaleString()}</span>
          <span className="text-sm text-muted-foreground">of {total.toLocaleString()} customers ({pct(total ? topN / total : 0)})</span>
        </div>
      </div>

      {topN === 0 ? (
        <p className="text-sm text-muted-foreground">Move the slider to choose how many customers to contact.</p>
      ) : est ? (
        <p className="rounded-xl bg-brand-soft px-4 py-3 text-base">
          Among the top <strong>{topN.toLocaleString()}</strong>, about <strong>{pct(est.precision)}</strong> would really
          leave, catching about <strong>{pct(est.recall)}</strong> of all leavers.
        </p>
      ) : (
        <p className="text-sm text-muted-foreground">
          This model was trained before the estimate was available. Train it again to see the estimate.
        </p>
      )}
      <p className="text-xs text-muted-foreground">
        Estimated from the held-out test split; assumes new customers look like the training data.
      </p>
    </section>
  );
}
