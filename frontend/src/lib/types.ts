// Hand-written types matching the JSON of src/api (FastAPI).

export type ColumnFlag =
  | "id"
  | "constant"
  | "datetime"
  | "location"
  | "target_copy"
  | "likely_leakage"
  | "missing_reveals_target"
  | "check_timing";

export type ColumnRole = "id" | "constant" | "numeric" | "categorical" | "datetime";

export interface TargetCandidate {
  name: string;
  values: string[];
  suggested_positive: string | null;
  name_suggests_target: boolean;
}

export interface UploadResponse {
  id: string;
  filename: string;
  source: "upload" | "sample";
  n_rows: number;
  columns: string[];
  target_candidates: TargetCandidate[];
  suggested_target: string | null;
}

export interface ColumnProfile {
  name: string;
  role: ColumnRole;
  dtype: string;
  n_unique: number;
  missing_rate: number;
  single_column_auc: number | null;
  missing_indicator_auc: number | null;
  flags: ColumnFlag[];
  suggested: "use" | "exclude";
}

export interface Profile {
  target: string;
  positive_label: string;
  computed_on: string;
  n_rows_raw: number;
  n_duplicates_removed: number;
  n_train: number;
  n_test: number;
  positive_rate_train: number;
  thresholds: {
    target_copy_auc: number;
    likely_leakage_auc: number;
    missing_reveals_target_auc: number;
    id_unique_ratio: number;
    timing_keywords: string[];
    location_pattern: string;
  };
  columns: ColumnProfile[];
  suggested_features: string[];
  suggested_excluded: Record<string, ColumnFlag[]>;
  timing_questions: string[];
}

export type Interval = [number, number];

export interface ModelScores {
  roc_auc: number;
  roc_auc_ci95: Interval;
  pr_auc: number;
  pr_auc_ci95: Interval;
}

export interface TestMetrics {
  n: number;
  positive_rate: number;
  lightgbm: ModelScores;
  logistic_regression: ModelScores;
  at_threshold_lightgbm: { threshold: number; n_flagged: number; precision: number; recall: number };
  bootstrap: { n_requested: number; n_used: number; paired: boolean; seed: number };
  difference_lightgbm_minus_lr: { roc_auc: number; roc_auc_ci95: Interval; pr_auc: number; pr_auc_ci95: Interval };
}

export interface CurvePoints {
  roc: { fpr: number; tpr: number }[];
  pr: { recall: number; precision: number }[];
}

export interface ModelMetrics {
  target: string;
  positive_label: string;
  features: string[];
  n_rows_raw: number;
  n_duplicates_removed: number;
  n_train: number;
  n_test: number;
  training: {
    n_train: number;
    positive_rate_train: number;
    n_features: number;
    lightgbm: {
      cv_auc_mean: number;
      cv_auc_std: number;
      oof_auc: number;
      threshold_f1_oof: number;
      best_params: Record<string, number>;
    };
    logistic_regression: { cv_auc_mean: number; cv_auc_std: number };
    optuna: {
      sampler: string;
      n_trials_requested: number;
      n_trials_completed: number;
      timeout_seconds: number | null;
      stopped_by_timeout: boolean;
    };
  };
  test: TestMetrics;
  curves: { lightgbm: CurvePoints; logistic_regression: CurvePoints };
  global_importance_top10: { feature: string; mean_abs_contribution: number }[];
  strength_cutoffs: { moderate: number; strong: number };
  notes: string[];
  timing_seconds: Record<string, number>;
}

export type DatasetStatus = "uploaded" | "profiled" | "training" | "trained" | "failed";

export interface Dataset {
  id: string;
  filename: string;
  source: "upload" | "sample";
  uploaded_at: string;
  n_rows: number;
  columns: string[];
  status: DatasetStatus;
  error: string | null;
  training_elapsed_seconds: number | null;
  target: string | null;
  positive_label: string | null;
  profile: Profile | null;
  target_candidates: TargetCandidate[];
  suggested_target: string | null;
  model: {
    created_at: string;
    mlflow_run_id: string | null;
    features: string[];
    metrics: ModelMetrics;
  } | null;
}

export interface TrainRequest {
  target: string;
  positive_label: string | null;
  features: string[];
}

export interface TrainResponse {
  id: string;
  status: "training";
  n_trials: number;
}

export type Strength = "strong" | "moderate" | "slight";

export interface Reason {
  feature: string;
  value: string;
  strength: Strength;
  contribution: number;
}

export interface Prediction {
  id: string;
  probability: number;
  flagged: boolean;
  reasons: Reason[];
}

export interface PredictResponse {
  threshold: number;
  id_column: string | null;
  n_rows: number;
  n_flagged: number;
  predictions: Prediction[];
}
