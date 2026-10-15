import { Flag, Gauge, Users } from "lucide-react";

import { Kpi } from "@/components/kpi";
import { pct } from "@/lib/format";
import { median } from "@/lib/predict-analysis";
import type { PredictResponse } from "@/lib/types";

/** Customers scored, customers flagged, median probability. No "expected leavers" sum: probabilities are not calibrated. */
export function KpiStrip({ result }: { result: PredictResponse }) {
  const med = median(result.predictions.map((p) => p.probability));
  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Kpi icon={Users} value={result.n_rows.toLocaleString()} label="customers scored" tone="neutral" />
      <Kpi
        icon={Flag}
        tone="flagged"
        value={result.n_flagged.toLocaleString()}
        label={`flagged as likely to leave (${pct(result.n_rows ? result.n_flagged / result.n_rows : 0)})`}
        hint={`Risk at or above the model's cut-off of ${pct(result.threshold)}.`}
      />
      <Kpi
        icon={Gauge}
        value={pct(med)}
        label="median churn risk"
        hint="Half of the customers score above this, half below."
      />
    </div>
  );
}
