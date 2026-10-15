"""Per-customer reasons and global importance from LightGBM ``pred_contrib`` (log-odds units)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .train import ChurnModel


def raw_contributions(model: ChurnModel, X: pd.DataFrame) -> np.ndarray:
    """LightGBM pred_contrib: one column per model feature, then the bias (expected value) column."""
    prepared = model.pipeline.named_steps["prepare"].transform(X)
    return np.asarray(model.pipeline.named_steps["model"].predict(prepared, pred_contrib=True))


def contributions(model: ChurnModel, X: pd.DataFrame) -> pd.DataFrame:
    """One column per model feature (the user's original column names); the bias term is dropped."""
    return pd.DataFrame(raw_contributions(model, X)[:, :-1], columns=model.features, index=X.index)


def format_value(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return "missing"
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if float(value).is_integer() else f"{value:g}"
    return str(value).strip()


def top_reasons(
    model: ChurnModel, X: pd.DataFrame, k: int = 3, positive_only: bool = False, contrib: pd.DataFrame | None = None
) -> list[list[dict]]:
    """For each row, the k features that push the churn score up the most (largest contributions first).

    With ``positive_only`` only contributions above zero are returned (so a row can get fewer than k).
    ``contrib`` lets a caller pass contributions it already computed.
    """
    contrib = contributions(model, X) if contrib is None else contrib
    values = contrib.to_numpy()
    # Stable sort on the negated values = largest first, ties in feature order (same as a stable pandas sort).
    order = np.argsort(-values, axis=1, kind="stable")
    features = list(contrib.columns)
    out = []
    for i, idx in enumerate(X.index):
        reasons = []
        for j in order[i]:
            value = values[i, j]
            if len(reasons) == k or (positive_only and value <= 0):
                break
            feature = features[j]
            shown = format_value(X.at[idx, feature])
            reasons.append({
                "feature": feature,
                "value": shown,
                "contribution": float(value),
                "text": f"{feature} = {shown} ({value:+.2f})",
            })
        out.append(reasons)
    return out


def global_importance(model: ChurnModel, X: pd.DataFrame) -> list[dict]:
    """Mean absolute contribution per feature over X (the training split), largest first."""
    mean_abs = contributions(model, X).abs().mean().sort_values(ascending=False, kind="mergesort")
    return [{"feature": f, "mean_abs_contribution": float(v)} for f, v in mean_abs.items()]
