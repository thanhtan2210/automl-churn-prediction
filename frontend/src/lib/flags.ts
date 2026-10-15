import { pct } from "./format";
import type { ColumnFlag, ColumnProfile, ColumnRole, Profile } from "./types";

export const FLAG_LABEL: Record<ColumnFlag, string> = {
  target_copy: "gives the answer away",
  likely_leakage: "too good to be true",
  missing_reveals_target: "blanks give it away",
  id: "ID",
  location: "location",
  constant: "same value everywhere",
  datetime: "date",
  check_timing: "check timing",
};

export const ROLE_LABEL: Record<ColumnRole, string> = {
  id: "ID",
  constant: "same value",
  numeric: "number",
  categorical: "category",
  datetime: "date",
};

export const LEAKAGE_FLAGS = new Set<ColumnFlag>(["target_copy", "likely_leakage", "missing_reveals_target"]);

export const isLeak = (c: ColumnProfile) => c.flags.some((f) => LEAKAGE_FLAGS.has(f));

/** One plain sentence on why a leaking column was removed; every number comes from the profile. */
export function leakSentence(c: ColumnProfile): string {
  if (c.flags.includes("missing_reveals_target")) {
    const filledLeft = 1 - c.missing_rate_churned;
    const filledStayed = 1 - c.missing_rate_stayed;
    return `Filled in for ${pct(filledLeft)} of customers who left and ${pct(filledStayed)} of those who stayed — it is written down after the outcome.`;
  }
  const score = c.single_column_auc?.toFixed(2) ?? "—";
  if (c.flags.includes("target_copy")) {
    return `On its own it tells who left almost perfectly (score ${score} of 1.00) — it is the answer written another way.`;
  }
  return `On its own it predicts who left very well (score ${score} of 1.00) — likely a score or status recorded after the fact.`;
}

/** Short reason for columns removed without being a leak. */
export function otherReason(c: ColumnProfile): string {
  if (c.flags.includes("id")) return "looks like an ID — it cannot carry over to new customers";
  if (c.flags.includes("location")) return "a location (zip, city, coordinates) — many values, little signal";
  if (c.flags.includes("constant")) return "has the same value for everyone";
  if (c.flags.includes("datetime")) return "dates are not used by the model yet";
  return "removed";
}

/** Tooltip text for a flag badge; thresholds come from the profile. */
export function flagExplanation(flag: ColumnFlag, t: Profile["thresholds"]): string {
  switch (flag) {
    case "target_copy":
      return `Alone, this column separates leavers from stayers almost perfectly (AUC ≥ ${t.target_copy_auc}).`;
    case "likely_leakage":
      return `Alone, this column predicts churn very well (AUC ≥ ${t.likely_leakage_auc}); check it is known before customers leave.`;
    case "missing_reveals_target":
      return `Whether this column is empty almost tells who left (AUC ≥ ${t.missing_reveals_target_auc}).`;
    case "id":
      return "Mostly unique values, like a customer ID.";
    case "location":
      return "Zip code, coordinates, city or address.";
    case "constant":
      return "Has a single value.";
    case "datetime":
      return "Dates are not used by the model yet.";
    case "check_timing":
      return "The name hints at an outcome or a score. Only keep it if it is known before the prediction date.";
  }
}
