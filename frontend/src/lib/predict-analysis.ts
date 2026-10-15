// Client-side analysis of a scored file. Everything here is derived from the API's predict response
// (and, for the contact estimate, from the model's test-split metrics); nothing is hard-coded.

import type { ModelMetrics, Prediction, PredictResponse } from "./types";

export const N_BINS = 20;

export type Range = [number, number];

export interface Bin {
  index: number;
  lo: number;
  hi: number;
  label: string;
  flagged: number;
  notFlagged: number;
}

export function binOf(probability: number): number {
  return Math.min(N_BINS - 1, Math.floor(probability * N_BINS));
}

export function histogram(predictions: Prediction[]): Bin[] {
  const bins: Bin[] = Array.from({ length: N_BINS }, (_, i) => ({
    index: i,
    lo: i / N_BINS,
    hi: (i + 1) / N_BINS,
    label: `${Math.round((i * 100) / N_BINS)}–${Math.round(((i + 1) * 100) / N_BINS)}%`,
    flagged: 0,
    notFlagged: 0,
  }));
  for (const p of predictions) {
    const b = bins[binOf(p.probability)];
    if (p.flagged) b.flagged += 1;
    else b.notFlagged += 1;
  }
  return bins;
}

export function median(values: number[]): number {
  if (values.length === 0) return NaN;
  const s = [...values].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

const fmt = (x: number) => (Number.isInteger(x) ? x.toLocaleString() : x.toLocaleString(undefined, { maximumFractionDigits: 2 }));

/** Quartile edges of each numeric feature in this file, used to group numeric reasons into ranges. */
export function quartileEdges(result: PredictResponse): Record<string, number[] | null> {
  const out: Record<string, number[] | null> = {};
  result.features.forEach((f, j) => {
    if (!result.numeric_features.includes(f)) return;
    const vals = result.predictions.map((p) => p.values[j]).filter((v): v is number => typeof v === "number");
    const sorted = [...vals].sort((a, b) => a - b);
    const distinct = new Set(sorted);
    // Few distinct values: show them as they are.
    out[f] = sorted.length === 0 || distinct.size <= 4 ? null : [...new Set([0.25, 0.5, 0.75].map((q) => quantile(sorted, q)))];
  });
  return out;
}

/** "Contract = Month-to-month", or "Tenure Months: 1–9" for numeric features (quartile range in this file). */
export function reasonLabel(
  feature: string,
  value: number | string | null,
  edges: Record<string, number[] | null>,
): string {
  if (value === null) return `${feature} is empty`;
  const e = edges[feature];
  if (typeof value !== "number" || !e) return `${feature} = ${typeof value === "number" ? fmt(value) : value}`;
  if (value <= e[0]) return `${feature} ≤ ${fmt(e[0])}`;
  for (let i = 1; i < e.length; i++) if (value <= e[i]) return `${feature} ${fmt(e[i - 1])}–${fmt(e[i])}`;
  return `${feature} > ${fmt(e[e.length - 1])}`;
}

/** The labels of a customer's (up to three) reasons, using the customer's raw values. */
export function reasonLabels(p: Prediction, result: PredictResponse, edges: Record<string, number[] | null>): string[] {
  return p.reasons.map((r) => reasonLabel(r.feature, p.values[result.features.indexOf(r.feature)], edges));
}

export interface ReasonCount {
  label: string;
  count: number;
  share: number;
}

/** How often each reason label appears among the given customers' top reasons. */
export function commonReasons(
  customers: Prediction[],
  result: PredictResponse,
  edges: Record<string, number[] | null>,
  top = 10,
): ReasonCount[] {
  const counts = new Map<string, number>();
  for (const p of customers) {
    for (const label of new Set(reasonLabels(p, result, edges))) counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([label, count]) => ({ label, count, share: customers.length ? count / customers.length : 0 }))
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label))
    .slice(0, top);
}

export interface Segment {
  value: string;
  count: number;
  flaggedShare: number;
  meanProbability: number;
}

/** Customers grouped by the value of one categorical feature. */
export function segments(result: PredictResponse, feature: string): Segment[] {
  const j = result.features.indexOf(feature);
  const groups = new Map<string, Prediction[]>();
  for (const p of result.predictions) {
    const key = p.values[j] === null ? "(empty)" : String(p.values[j]);
    groups.set(key, [...(groups.get(key) ?? []), p]);
  }
  return [...groups.entries()]
    .map(([value, ps]) => ({
      value,
      count: ps.length,
      flaggedShare: ps.filter((p) => p.flagged).length / ps.length,
      meanProbability: ps.reduce((s, p) => s + p.probability, 0) / ps.length,
    }))
    .sort((a, b) => b.flaggedShare - a.flaggedShare || b.count - a.count);
}

/** Test-split precision/recall at the same share of customers contacted (nearest point of the curve). */
export function contactEstimate(metrics: ModelMetrics | undefined, n: number, total: number) {
  const curve = metrics?.contact_curve;
  if (!curve?.length || n <= 0 || total <= 0) return null;
  const share = n / total;
  let best = curve[0];
  for (const point of curve) if (Math.abs(point.share - share) < Math.abs(best.share - share)) best = point;
  return best;
}

/** Ranks by probability, highest first (1-based); ties keep file order. */
export function ranks(predictions: Prediction[]): number[] {
  const order = predictions.map((p, i) => [p.probability, i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  const out = new Array<number>(predictions.length);
  order.forEach(([, i], r) => (out[i] = r + 1));
  return out;
}
