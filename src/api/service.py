"""What the endpoints do, kept apart from HTTP details: target suggestions, profiling, training, scoring."""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.churn.data import POSITIVE_TOKENS, PreparedData, _norm_label, prepare, read_table
from src.churn.evaluate import TEST_SET_NOTE, score_test
from src.churn.explain import format_value, global_importance, raw_contributions, top_reasons
from src.churn.predict import check_columns
from src.churn.stats import curve_points, precision_recall_by_share
from src.churn.train import ChurnModel, load_model, save_model, train

from .config import Settings
from .db import Store

TARGET_NAME_HINT = re.compile(r"churn|exit|attrition", re.IGNORECASE)
RETRAIN_NOTE = (
    "Each training run scores the same test split; retraining many times on the same file with different "
    "columns makes the test number optimistic."
)


class InputError(ValueError):
    """Bad user input; shown to the user as a 400 with this message."""


def label_text(value) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def target_candidates(df: pd.DataFrame) -> list[dict]:
    """Columns with exactly two distinct values; the positive class is guessed from POSITIVE_TOKENS."""
    out = []
    for col in df.columns:
        values = df[col].dropna().unique()
        if len(values) != 2:
            continue
        labels = [label_text(v) for v in values]
        guess = [lab for lab, v in zip(labels, values, strict=True) if _norm_label(v) in POSITIVE_TOKENS]
        out.append({
            "name": col,
            "values": sorted(labels),
            "suggested_positive": guess[0] if len(guess) == 1 else None,
            "name_suggests_target": bool(TARGET_NAME_HINT.search(col)),
        })
    return out


def suggested_target(candidates: list[dict]) -> str | None:
    return next((c["name"] for c in candidates if c["name_suggests_target"]), None)


def load_prepared(path: Path, target: str, positive_label: str | None, settings: Settings) -> PreparedData:
    try:
        data = prepare(read_table(path), target, positive_label)
    except ValueError as e:
        raise InputError(str(e)) from e
    if len(data.df) < settings.min_rows:
        raise InputError(f"Only {len(data.df)} rows after removing duplicates; at least {settings.min_rows} needed.")
    minority = int(min(data.y.sum(), len(data.y) - data.y.sum()))
    if minority < settings.min_minority:
        raise InputError(
            f"The smaller class of '{data.target}' has only {minority} rows; at least {settings.min_minority} needed."
        )
    return data


def check_features(data: PreparedData, features: list[str]) -> None:
    if not features:
        raise InputError("Choose at least one column to train on.")
    unknown = [f for f in features if f not in data.df.columns]
    if unknown:
        raise InputError(f"These columns do not exist in the file: {unknown}")
    if data.target in features:
        raise InputError(f"The target column '{data.target}' cannot also be a feature.")


def same_target(row: dict, target: str, positive_label: str | None) -> bool:
    if row.get("target") != target or row.get("profile") is None:
        return False
    return positive_label is None or _norm_label(positive_label) == _norm_label(row.get("positive_label"))


def train_and_score(data: PreparedData, features: list[str], settings: Settings, run_name: str, on_trial=None):
    """Same steps as src.churn.evaluate (minus its class-weight side check), plus curve points for the UI."""
    t0 = time.perf_counter()
    X_tr, y_tr = data.df.loc[data.train_idx, features], data.y.loc[data.train_idx]
    X_te, y_te = data.df.loc[data.test_idx, features], data.y.loc[data.test_idx]
    result = train(
        X_tr, y_tr, target=data.target, positive_label=data.positive_label,
        n_trials=settings.n_trials, timeout=settings.timeout_seconds or None,
        mlflow_dir=settings.mlflow_dir, run_name=run_name,
        callbacks=[lambda study, trial: on_trial(len(study.trials))] if on_trial else None,
    )
    test, p_lgbm, p_lr = score_test(result.model, result.baseline, X_te, y_te)
    y = y_te.to_numpy()
    metrics = {
        "target": data.target,
        "positive_label": data.positive_label,
        "features": features,
        "n_rows_raw": data.n_rows_raw,
        "n_duplicates_removed": data.n_duplicates_removed,
        "n_train": int(len(data.train_idx)),
        "n_test": int(len(data.test_idx)),
        "training": result.metrics,
        "test": test,
        "curves": {"lightgbm": curve_points(y, p_lgbm), "logistic_regression": curve_points(y, p_lr)},
        # For the Predict page's "how many can you contact" estimate (LightGBM, test split).
        "contact_curve": precision_recall_by_share(y, p_lgbm),
        "global_importance_top10": global_importance(result.model, X_tr)[:10],
        "notes": [RETRAIN_NOTE, TEST_SET_NOTE],
        "timing_seconds": {**result.seconds, "total": round(time.perf_counter() - t0, 1)},
    }
    return result, metrics


class Trainer:
    """One background worker thread: the app trains at most one dataset at a time."""

    def __init__(self, store: Store, settings: Settings):
        self.store = store
        self.settings = settings
        self._lock = threading.Lock()
        self._busy: str | None = None
        self._progress: dict[str, int] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="churn-trainer")
        self._models: dict[str, tuple[float, ChurnModel]] = {}

    @property
    def busy(self) -> str | None:
        return self._busy

    def progress(self, dataset_id: str) -> dict | None:
        """Optuna trials finished so far for the dataset being trained (in memory; the app has one worker)."""
        return dict(self._progress) if self._busy == dataset_id and self._progress else None

    def start(self, dataset_id: str, path: Path, target: str, positive_label: str | None,
              features: list[str]) -> bool:
        with self._lock:
            if self._busy is not None:
                return False
            self._busy = dataset_id
        self.store.update_dataset(dataset_id, status="training", error=None)
        self._executor.submit(self._run, dataset_id, path, target, positive_label, features)
        return True

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, dataset_id, path, target, positive_label, features) -> None:
        try:
            data = load_prepared(path, target, positive_label, self.settings)
            self._progress = {"trials_done": 0, "trials_total": self.settings.n_trials}

            def on_trial(done: int) -> None:
                self._progress = {"trials_done": done, "trials_total": self.settings.n_trials}

            result, metrics = train_and_score(data, features, self.settings, run_name=f"api-{dataset_id}",
                                              on_trial=on_trial)
            model_dir = self.settings.models_dir / dataset_id
            model_path = save_model(result.model, model_dir)
            self.store.save_model(dataset_id, str(model_path), result.mlflow_run_id, features, metrics)
            self.store.update_dataset(dataset_id, status="trained", error=None, target=data.target,
                                      positive_label=data.positive_label)
        except Exception as e:  # any failure is shown to the user on the dataset
            self.store.update_dataset(dataset_id, status="failed", error=f"Training failed: {e}")
        finally:
            with self._lock:
                self._busy = None
                self._progress = {}

    def wait(self, timeout: float = 600) -> None:
        """Blocks until the worker is idle (used by tests)."""
        deadline = time.monotonic() + timeout
        while self._busy is not None and time.monotonic() < deadline:
            time.sleep(0.2)

    def model(self, path: str) -> ChurnModel:
        mtime = Path(path).stat().st_mtime
        cached = self._models.get(path)
        if cached is None or cached[0] != mtime:
            cached = (mtime, load_model(path))
            self._models[path] = cached
        return cached[1]


def _feature_values(model: ChurnModel, df: pd.DataFrame) -> list[list]:
    """Each customer's value per model feature: numbers for numeric features, text for the rest, None if empty."""
    columns = []
    for f in model.features:
        if f in model.numeric:
            col = pd.to_numeric(df[f], errors="coerce")
            columns.append([None if pd.isna(v) else float(v) for v in col])
        else:
            columns.append([None if format_value(v) == "missing" else format_value(v) for v in df[f]])
    return [list(row) for row in zip(*columns, strict=True)]


def score_file(model: ChurnModel, df: pd.DataFrame, id_column: str | None) -> dict:
    """Probability, flag, up to 3 reasons (largest positive contributions first), and every feature's
    contribution and value per customer. Contributions are LightGBM pred_contrib (log-odds); with
    base_value they add up to the log-odds of the probability."""
    df = check_columns(model, df)
    proba = model.predict_proba(df)
    raw = raw_contributions(model, df)
    contrib = pd.DataFrame(raw[:, :-1], columns=model.features, index=df.index)
    reasons = top_reasons(model, df, k=3, positive_only=True, contrib=contrib)
    values = _feature_values(model, df)
    if id_column and id_column in df.columns:
        ids = df[id_column].astype(str).tolist()
    else:
        id_column, ids = None, [str(i + 1) for i in range(len(df))]
    predictions = [
        {
            "id": ids[i],
            "probability": float(proba[i]),
            "flagged": bool(proba[i] >= model.threshold),
            "reasons": [{k: r[k] for k in ("feature", "value", "contribution")} for r in reasons[i]],
            "contributions": raw[i, :-1].tolist(),
            "values": values[i],
        }
        for i in range(len(df))
    ]
    return {
        "threshold": model.threshold,
        "id_column": id_column,
        "features": model.features,
        "numeric_features": model.numeric,
        "base_value": float(raw[0, -1]) if len(raw) else 0.0,
        "n_rows": len(predictions),
        "n_flagged": sum(p["flagged"] for p in predictions),
        "predictions": predictions,
    }
