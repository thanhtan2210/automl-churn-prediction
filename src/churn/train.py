"""Train one LightGBM (tuned with Optuna) and a Logistic Regression baseline on the training split.

Every step that learns from data sits inside one sklearn Pipeline, so the saved model accepts the
user's raw columns. No resampling, no class weights, no calibration: the probabilities are shown in
the app as they come out of the model.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import ROOT, SEED

N_JOBS = 4  # fixed so results do not depend on the machine's core count
DEFAULT_TRIALS = 40
DEFAULT_TIMEOUT = 300


class PrepareColumns(BaseEstimator, TransformerMixin):
    """Selects the model's columns from a raw table and fixes their types.

    Numeric columns are coerced with ``pd.to_numeric`` (blanks become NaN). Categorical columns
    become strings, or a pandas ``category`` with the categories seen in training when
    ``as_category`` is true (unseen values become NaN, which LightGBM handles).
    """

    def __init__(self, numeric: list[str], categorical: list[str], as_category: bool = True):
        self.numeric = numeric
        self.categorical = categorical
        self.as_category = as_category

    @staticmethod
    def _text(s: pd.Series) -> pd.Series:
        def clean(v):
            if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
                return np.nan
            v = str(v).strip()
            return v if v else np.nan

        return s.astype(object).map(clean)

    def fit(self, X: pd.DataFrame, y=None):
        self.categories_ = {c: sorted(self._text(X[c]).dropna().unique().tolist()) for c in self.categorical}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for c in self.numeric:
            out[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        for c in self.categorical:
            text = self._text(X[c])
            out[c] = pd.Categorical(text, categories=self.categories_[c]) if self.as_category else text
        return out[self.numeric + self.categorical]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.numeric + self.categorical, dtype=object)


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def make_lgbm(numeric, categorical, **params) -> Pipeline:
    model = LGBMClassifier(
        objective="binary", random_state=SEED, deterministic=True, force_col_wise=True,
        n_jobs=N_JOBS, verbose=-1, subsample_freq=1, **params,
    )
    return Pipeline([("prepare", PrepareColumns(numeric, categorical, as_category=True)), ("model", model)])


def make_logreg(numeric, categorical) -> Pipeline:
    encode = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("prepare", PrepareColumns(numeric, categorical, as_category=False)),
        ("encode", encode),
        ("model", LogisticRegression(max_iter=5000, random_state=SEED)),
    ])


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def f1_threshold(y, proba) -> float:
    """Threshold that maximises F1 on the given (out-of-fold) predictions."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1[:-1]))])


@dataclass
class ChurnModel:
    """Everything needed to score a new file; saved with joblib."""

    pipeline: Pipeline
    features: list[str]
    numeric: list[str]
    categorical: list[str]
    threshold: float
    target: str
    positive_label: str
    params: dict
    metrics: dict = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]


@dataclass
class TrainResult:
    model: ChurnModel
    baseline: Pipeline
    study: optuna.Study
    metrics: dict
    seconds: dict
    mlflow_run_id: str | None = None


def train(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    target: str,
    positive_label: str,
    n_trials: int = DEFAULT_TRIALS,
    timeout: float | None = DEFAULT_TIMEOUT,
    mlflow_dir: str | Path | None = ROOT / "mlruns",
    run_name: str = "train",
    callbacks: list | None = None,
) -> TrainResult:
    """Tune and fit LightGBM and fit the LR baseline on (X, y), which must be the training split only."""
    numeric, categorical = split_columns(X)
    cv = make_cv()
    t0 = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        pipe = make_lgbm(numeric, categorical, **suggest_params(trial))
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        trial.set_user_attr("cv_auc_std", float(np.std(scores)))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    # callbacks only observe the study (e.g. progress reporting); they do not change the search.
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=callbacks)
    best = dict(study.best_params)
    tune_seconds = time.perf_counter() - t0

    # Decision threshold from out-of-fold predictions on the training split (never the test split).
    lgbm = make_lgbm(numeric, categorical, **best)
    oof = cross_val_predict(lgbm, X, y, cv=cv, method="predict_proba")[:, 1]
    threshold = f1_threshold(y, oof)
    lgbm.fit(X, y)

    baseline = make_logreg(numeric, categorical)
    lr_scores = cross_val_score(baseline, X, y, cv=cv, scoring="roc_auc")
    baseline = clone(baseline).fit(X, y)

    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    metrics = {
        "n_train": int(len(y)),
        "positive_rate_train": float(np.mean(y)),
        "n_features": len(numeric) + len(categorical),
        "lightgbm": {
            "cv_auc_mean": float(study.best_value),
            "cv_auc_std": float(study.best_trial.user_attrs["cv_auc_std"]),
            "oof_auc": float(roc_auc_score(y, oof)),
            "threshold_f1_oof": threshold,
            "best_params": best,
        },
        "logistic_regression": {
            "cv_auc_mean": float(np.mean(lr_scores)),
            "cv_auc_std": float(np.std(lr_scores)),
        },
        "optuna": {
            "sampler": "TPESampler(seed=42)",
            "n_trials_requested": n_trials,
            "n_trials_completed": n_complete,
            "timeout_seconds": timeout,
            "stopped_by_timeout": n_complete < n_trials,
        },
    }
    seconds = {"tune": round(tune_seconds, 1), "train_total": round(time.perf_counter() - t0, 1)}
    model = ChurnModel(lgbm, numeric + categorical, numeric, categorical, threshold,
                       target, positive_label, best, metrics)
    run_id = log_to_mlflow(model, study, metrics, Path(mlflow_dir), run_name) if mlflow_dir is not None else None
    return TrainResult(model, baseline, study, metrics, seconds, run_id)


def log_to_mlflow(model: ChurnModel, study: optuna.Study, metrics: dict, mlflow_dir: Path, run_name: str) -> str:
    """One MLflow run per training call: best params, CV metrics, threshold, model and the trial table."""
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(mlflow_dir.resolve().as_uri())
    mlflow.set_experiment("churn")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(model.params)
        mlflow.log_params({"n_features": metrics["n_features"], "n_train": metrics["n_train"],
                           "n_trials_completed": metrics["optuna"]["n_trials_completed"]})
        mlflow.log_metrics({
            "lgbm_cv_auc": metrics["lightgbm"]["cv_auc_mean"],
            "lgbm_cv_auc_std": metrics["lightgbm"]["cv_auc_std"],
            "lgbm_oof_auc": metrics["lightgbm"]["oof_auc"],
            "threshold": model.threshold,
            "lr_cv_auc": metrics["logistic_regression"]["cv_auc_mean"],
        })
        with tempfile.TemporaryDirectory() as tmp:
            trials_path = os.path.join(tmp, "optuna_trials.csv")
            study.trials_dataframe().to_csv(trials_path, index=False, encoding="utf-8")
            mlflow.log_artifact(trials_path)
            features_path = os.path.join(tmp, "features.json")
            with open(features_path, "w", encoding="utf-8") as f:
                json.dump({"numeric": model.numeric, "categorical": model.categorical}, f, indent=2)
            mlflow.log_artifact(features_path)
        mlflow.sklearn.log_model(model.pipeline, artifact_path="model")
        return run.info.run_id


def save_model(model: ChurnModel, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.joblib"
    joblib.dump(model, path)
    with open(directory / "metrics.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(model.metrics, f, indent=2)
    return path


def load_model(path: str | Path) -> ChurnModel:
    return joblib.load(path)
