"""Metric helpers shared by the evaluation script, the API and the leakage audit."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def ci95(values) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def paired_bootstrap(y, scores_a, scores_b, n_boot: int = 1000, seed: int = 42) -> dict:
    """Paired bootstrap of ROC-AUC and PR-AUC for two models scored on the same rows.

    Each resample draws the same row indices for both models, so the CI of the difference
    accounts for their correlation. Resamples with a single class are skipped.
    Returns the per-resample values under ``roc_a``, ``roc_b``, ``pr_a``, ``pr_b``,
    ``roc_diff`` and ``pr_diff`` (a minus b).
    """
    y = np.asarray(y)
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    rng = np.random.default_rng(seed)
    boot: dict[str, list[float]] = {k: [] for k in ("roc_a", "roc_b", "pr_a", "pr_b", "roc_diff", "pr_diff")}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        ra, rb = roc_auc_score(y[idx], a[idx]), roc_auc_score(y[idx], b[idx])
        pa, pb = average_precision_score(y[idx], a[idx]), average_precision_score(y[idx], b[idx])
        boot["roc_a"].append(ra)
        boot["roc_b"].append(rb)
        boot["pr_a"].append(pa)
        boot["pr_b"].append(pb)
        boot["roc_diff"].append(ra - rb)
        boot["pr_diff"].append(pa - pb)
    return boot


def _downsample(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def curve_points(y, proba, max_points: int = 200) -> dict:
    """ROC and precision-recall curve points, each reduced to at most ``max_points`` (ends kept)."""
    fpr, tpr, _ = roc_curve(y, proba)
    precision, recall, _ = precision_recall_curve(y, proba)
    roc_idx = _downsample(len(fpr), max_points)
    pr_idx = _downsample(len(recall), max_points)
    return {
        "roc": [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in roc_idx],
        "pr": [{"recall": float(recall[i]), "precision": float(precision[i])} for i in pr_idx],
    }


def precision_recall_by_share(y, proba, n_points: int = 200) -> list[dict]:
    """Precision and recall when contacting the top share of customers, ranked by probability.

    For share = i / n_points (i = 1..n_points) the top ceil(share * n) customers are "contacted";
    precision = leavers among them / contacted, recall = leavers among them / all leavers.
    """
    y = np.asarray(y)
    order = np.argsort(-np.asarray(proba), kind="mergesort")
    hits = np.cumsum(y[order])
    n, total = len(y), max(int(y.sum()), 1)
    out = []
    for i in range(1, n_points + 1):
        share = i / n_points
        k = max(1, int(np.ceil(share * n)))
        out.append({"share": share, "precision": float(hits[k - 1] / k), "recall": float(hits[k - 1] / total)})
    return out
