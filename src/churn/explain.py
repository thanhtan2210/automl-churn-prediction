"""Per-customer reasons and global importance from LightGBM ``pred_contrib`` (log-odds units)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .train import ChurnModel


def contributions(model: ChurnModel, X: pd.DataFrame) -> pd.DataFrame:
    """One column per model feature (the user's original column names); the bias term is dropped."""
    prepared = model.pipeline.named_steps["prepare"].transform(X)
    raw = model.pipeline.named_steps["model"].predict(prepared, pred_contrib=True)
    return pd.DataFrame(np.asarray(raw)[:, :-1], columns=model.features, index=X.index)


def format_value(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return "missing"
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if float(value).is_integer() else f"{value:g}"
    return str(value).strip()


def strength_cutoffs(model: ChurnModel, X_train: pd.DataFrame) -> dict:
    """Tertiles of the positive contributions on the training split, used to label reasons."""
    values = contributions(model, X_train).to_numpy().ravel()
    positive = values[values > 0]
    if positive.size == 0:
        return {"moderate": 0.0, "strong": 0.0}
    return {"moderate": float(np.percentile(positive, 100 / 3)), "strong": float(np.percentile(positive, 200 / 3))}


def strength(contribution: float, cutoffs: dict) -> str:
    if contribution >= cutoffs["strong"]:
        return "strong"
    if contribution >= cutoffs["moderate"]:
        return "moderate"
    return "slight"


def top_reasons(
    model: ChurnModel, X: pd.DataFrame, k: int = 3, positive_only: bool = False, cutoffs: dict | None = None
) -> list[list[dict]]:
    """For each row, the k features that push the churn score up the most (largest contributions first).

    With ``positive_only`` only contributions above zero are returned (so a row can get fewer than k).
    With ``cutoffs`` (from ``strength_cutoffs``) each reason also gets a ``strength`` label.
    """
    contrib = contributions(model, X)
    out = []
    for idx, row in contrib.iterrows():
        top = row.sort_values(ascending=False, kind="mergesort")
        if positive_only:
            top = top[top > 0]
        reasons = []
        for feature, value in top.head(k).items():
            shown = format_value(X.at[idx, feature])
            reason = {
                "feature": feature,
                "value": shown,
                "contribution": float(value),
                "text": f"{feature} = {shown} ({value:+.2f})",
            }
            if cutoffs is not None:
                reason["strength"] = strength(float(value), cutoffs)
            reasons.append(reason)
        out.append(reasons)
    return out


def global_importance(model: ChurnModel, X: pd.DataFrame) -> list[dict]:
    """Mean absolute contribution per feature over X (the training split), largest first."""
    mean_abs = contributions(model, X).abs().mean().sort_values(ascending=False, kind="mergesort")
    return [{"feature": f, "mean_abs_contribution": float(v)} for f, v in mean_abs.items()]
