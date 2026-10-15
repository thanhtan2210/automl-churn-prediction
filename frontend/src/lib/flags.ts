import type { ColumnFlag, Profile } from "./types";

export const FLAG_LABEL: Record<ColumnFlag, string> = {
  target_copy: "target copy",
  likely_leakage: "likely leakage",
  missing_reveals_target: "missing reveals target",
  id: "id",
  location: "location",
  constant: "constant",
  datetime: "datetime",
  check_timing: "check timing",
};

export const FLAG_ORDER: ColumnFlag[] = [
  "target_copy",
  "likely_leakage",
  "missing_reveals_target",
  "id",
  "location",
  "constant",
  "datetime",
  "check_timing",
];

export const LEAKAGE_FLAGS = new Set<ColumnFlag>(["target_copy", "likely_leakage", "missing_reveals_target"]);

/** One-line explanation per flag; thresholds come from the profile, not from this file. */
export function flagExplanation(flag: ColumnFlag, t: Profile["thresholds"]): string {
  switch (flag) {
    case "target_copy":
      return `This column alone predicts churn almost perfectly (AUC ≥ ${t.target_copy_auc}) — it is the target in another form or was recorded after the customer left.`;
    case "likely_leakage":
      return `This column alone predicts churn very well (AUC ≥ ${t.likely_leakage_auc}) — check that it is known before the customer leaves.`;
    case "missing_reveals_target":
      return `Whether this column is empty almost tells you who churned (AUC ≥ ${t.missing_reveals_target_auc}) — it is probably filled in only after the outcome.`;
    case "id":
      return "Looks like an identifier (mostly unique values) — it cannot carry over to new customers.";
    case "location":
      return "Geographic identifier (zip code, coordinates, city, address) — many values and little signal.";
    case "constant":
      return "Has a single value — it carries no information.";
    case "datetime":
      return "Dates are not used by the model yet.";
    case "check_timing":
      return "The name hints at an outcome or a score. Is this known before the prediction date?";
  }
}
