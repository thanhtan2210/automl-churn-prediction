import type { Interval } from "./types";

export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;

export const num = (x: number | null | undefined, digits = 3) =>
  x === null || x === undefined || Number.isNaN(x) ? "—" : x.toFixed(digits);

export const ci = ([lo, hi]: Interval, digits = 3) => `[${lo.toFixed(digits)}, ${hi.toFixed(digits)}]`;

export const signed = (x: number, digits = 3) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(digits)}`;

export function duration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m}m ${s.toString().padStart(2, "0")}s` : `${s}s`;
}
