"""Profile → drop suggested columns → train LightGBM + LR on train → score the test split once.

Usage (from the repo root):
    python -m src.churn.evaluate --dataset telco
    python -m src.churn.evaluate --dataset iranian --n-trials 40

Writes reports/metrics_<dataset>.json and models/<dataset>/{model.joblib,metrics.json};
logs one MLflow run to ./mlruns. Every number in the README comes from these files.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score

from .data import DATASETS, ROOT, SEED, load_dataset
from .explain import global_importance
from .profile import profile_dataset
from .runinfo import environment, sha256_file
from .stats import ci95, paired_bootstrap
from .train import DEFAULT_TRIALS, make_cv, make_lgbm, save_model, split_columns, train

N_BOOTSTRAP = 1000
LIBRARIES = ["pandas", "numpy", "scikit-learn", "lightgbm", "optuna", "mlflow", "openpyxl", "scipy", "joblib"]
TEST_SET_NOTE = (
    "The train/test split (stratified 80/20, random_state=42, after removing exact duplicates) is identical "
    "to the Phase 1 leakage audit, so this test set was already looked at in Phase 1. No choice in Phase 2 "
    "(features, thresholds, search space, model) was based on test-set numbers."
)


def _audit_reference(name: str) -> dict | None:
    path = ROOT / "reports" / f"leakage_audit_{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        audit = json.load(f)
    final = audit["final"]["lightgbm"]
    return {
        "source": f"reports/leakage_audit_{name}.json (final LightGBM: fixed params, class_weight='balanced', "
                  "generic ratio features, audit feature set)",
        "cv_auc_mean": final["cv_auc_mean"],
        "test_roc_auc": final["test_roc_auc"],
    }


def score_test(model, baseline, X_te, y_te) -> tuple[dict, np.ndarray, np.ndarray]:
    """Scores LightGBM and the LR baseline once on the test split, with paired bootstrap CIs.

    Returns the test metrics and both models' test probabilities.
    """
    p_lgbm = model.predict_proba(X_te)
    p_lr = baseline.predict_proba(X_te)[:, 1]
    y = y_te.to_numpy()
    boot = paired_bootstrap(y, p_lgbm, p_lr, n_boot=N_BOOTSTRAP, seed=SEED)
    flagged = p_lgbm >= model.threshold
    test = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "lightgbm": {
            "roc_auc": float(roc_auc_score(y, p_lgbm)), "roc_auc_ci95": ci95(boot["roc_a"]),
            "pr_auc": float(average_precision_score(y, p_lgbm)), "pr_auc_ci95": ci95(boot["pr_a"]),
        },
        "logistic_regression": {
            "roc_auc": float(roc_auc_score(y, p_lr)), "roc_auc_ci95": ci95(boot["roc_b"]),
            "pr_auc": float(average_precision_score(y, p_lr)), "pr_auc_ci95": ci95(boot["pr_b"]),
        },
        "at_threshold_lightgbm": {
            "threshold": model.threshold,
            "n_flagged": int(flagged.sum()),
            "precision": float(precision_score(y, flagged, zero_division=0)),
            "recall": float(recall_score(y, flagged, zero_division=0)),
        },
        "bootstrap": {"n_requested": N_BOOTSTRAP, "n_used": len(boot["roc_a"]), "paired": True, "seed": SEED},
    }
    test["difference_lightgbm_minus_lr"] = {
        "roc_auc": test["lightgbm"]["roc_auc"] - test["logistic_regression"]["roc_auc"],
        "roc_auc_ci95": ci95(boot["roc_diff"]),
        "pr_auc": test["lightgbm"]["pr_auc"] - test["logistic_regression"]["pr_auc"],
        "pr_auc_ci95": ci95(boot["pr_diff"]),
    }

    return test, p_lgbm, p_lr


def evaluate(name: str, n_trials: int = DEFAULT_TRIALS, timeout: float | None = None) -> dict:
    t0 = time.perf_counter()
    data = load_dataset(name)
    prof = profile_dataset(data)
    t_profile = time.perf_counter() - t0
    features = prof["suggested_features"]

    X_tr, y_tr = data.df.loc[data.train_idx, features], data.y.loc[data.train_idx]
    X_te, y_te = data.df.loc[data.test_idx, features], data.y.loc[data.test_idx]

    result = train(X_tr, y_tr, target=data.target, positive_label=data.positive_label,
                   n_trials=n_trials, timeout=timeout, run_name=f"evaluate-{name}")
    model = result.model

    # Effect of dropping class weights, with the audit's fixed LightGBM settings on this pipeline (CV on train).
    numeric, categorical = split_columns(X_tr)
    fixed = {"n_estimators": 300, "learning_rate": 0.05}
    class_weight_check = {}
    for label, extra in (("no_class_weight", {}), ("class_weight_balanced", {"class_weight": "balanced"})):
        scores = cross_val_score(make_lgbm(numeric, categorical, **fixed, **extra), X_tr, y_tr,
                                 cv=make_cv(), scoring="roc_auc")
        class_weight_check[label] = {"cv_auc_mean": float(np.mean(scores)), "cv_auc_std": float(np.std(scores))}

    # The only use of the test split.
    test, _, _ = score_test(model, result.baseline, X_te, y_te)

    model.metrics["test"] = {k: test[k] for k in ("lightgbm", "logistic_regression", "at_threshold_lightgbm")}
    save_model(model, ROOT / "models" / name)

    spec = DATASETS[name]
    return {
        "dataset": name,
        "generated_by": f"python -m src.churn.evaluate --dataset {name} --n-trials {n_trials}",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "note": TEST_SET_NOTE,
        "data": {
            "file": str(spec.path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(spec.path),
            "target": data.target,
            "positive_label": data.positive_label,
            "n_rows_raw": data.n_rows_raw,
            "n_duplicates_removed": data.n_duplicates_removed,
            "n_rows": int(len(data.df)),
            "n_train": int(len(data.train_idx)),
            "n_test": int(len(data.test_idx)),
            "positive_rate_train": float(y_tr.mean()),
            "positive_rate_test": float(y_te.mean()),
        },
        "features": {
            "used": features,
            "excluded": prof["suggested_excluded"],
            "timing_questions": prof["timing_questions"],
            "profile": [{k: c[k] for k in ("name", "role", "single_column_auc", "missing_indicator_auc", "flags")}
                        for c in prof["columns"]],
        },
        "training": result.metrics,
        "class_weight_check": {
            "settings": "LGBM n_estimators=300, learning_rate=0.05, this pipeline and feature set, 5-fold CV on train",
            **class_weight_check,
            "audit_s6": _audit_reference(name),
        },
        "test": test,
        "global_importance_top10": global_importance(model, X_tr)[:10],
        "timing_seconds": {
            "profile": round(t_profile, 1), **result.seconds, "total": round(time.perf_counter() - t0, 1),
        },
        "environment": environment(LIBRARIES),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the churn model on a bundled dataset.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--n-trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--timeout", type=float, default=0,
                        help="Optuna timeout in seconds; 0 (default) = no timeout, so reruns give the same result")
    args = parser.parse_args(argv)

    report = evaluate(args.dataset, n_trials=args.n_trials, timeout=args.timeout or None)
    out = ROOT / "reports" / f"metrics_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    t = report["test"]
    print(f"[{args.dataset}] LightGBM test ROC-AUC {t['lightgbm']['roc_auc']:.4f}, "
          f"LR {t['logistic_regression']['roc_auc']:.4f}; wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
