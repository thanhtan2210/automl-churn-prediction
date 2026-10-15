"""What the endpoints do, kept apart from HTTP details: target suggestions, profiling, training, scoring."""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.churn.data import POSITIVE_TOKENS, PreparedData, _norm_label, normalize_column_name, prepare, read_table
from src.churn.evaluate import TEST_SET_NOTE, score_test
from src.churn.explain import global_importance, strength_cutoffs
from src.churn.predict import predict
from src.churn.stats import curve_points
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


def train_and_score(data: PreparedData, features: list[str], settings: Settings, run_name: str):
    """Same steps as src.churn.evaluate (minus its class-weight side check), plus curve points for the UI."""
    t0 = time.perf_counter()
    X_tr, y_tr = data.df.loc[data.train_idx, features], data.y.loc[data.train_idx]
    X_te, y_te = data.df.loc[data.test_idx, features], data.y.loc[data.test_idx]
    result = train(
        X_tr, y_tr, target=data.target, positive_label=data.positive_label,
        n_trials=settings.n_trials, timeout=settings.timeout_seconds or None,
        mlflow_dir=settings.mlflow_dir, run_name=run_name,
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
        "global_importance_top10": global_importance(result.model, X_tr)[:10],
        "strength_cutoffs": strength_cutoffs(result.model, X_tr),
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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="churn-trainer")
        self._models: dict[str, tuple[float, ChurnModel]] = {}

    @property
    def busy(self) -> str | None:
        return self._busy

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
            result, metrics = train_and_score(data, features, self.settings, run_name=f"api-{dataset_id}")
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


def score_file(model: ChurnModel, df: pd.DataFrame, cutoffs: dict, id_column: str | None) -> dict:
    out = predict(model, df, k=3, positive_only=True, cutoffs=cutoffs)
    columns = {normalize_column_name(c): c for c in df.columns}
    if id_column and id_column in columns:
        ids = df[columns[id_column]].astype(str).tolist()
    else:
        id_column, ids = None, [str(i + 1) for i in range(len(df))]
    predictions = [
        {
            "id": ids[i],
            "probability": float(row.churn_probability),
            "flagged": bool(row.predicted_churn),
            "reasons": [{k: r[k] for k in ("feature", "value", "strength", "contribution")} for r in row.reasons],
        }
        for i, row in enumerate(out.itertuples(index=False))
    ]
    return {
        "threshold": model.threshold,
        "id_column": id_column,
        "n_rows": len(predictions),
        "n_flagged": sum(p["flagged"] for p in predictions),
        "predictions": predictions,
    }
