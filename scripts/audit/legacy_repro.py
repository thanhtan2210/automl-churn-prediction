"""Reproduce the number the OLD AutoML pipeline reports on IBM Telco (cleanup Phase 1.3).

Only runs on a checkout of branch ``archive/before-cleanup`` (the old ``backend/`` package
was removed from ``cv-refactor``). The old code is not modified. Example:

    git worktree add ../cdp-legacy archive/before-cleanup
    cp scripts/audit/legacy_repro.py ../cdp-legacy/legacy_repro.py
    cd ../cdp-legacy
    python -m venv .venv-legacy
    .venv-legacy/Scripts/python -m pip install pandas==2.3.3 numpy==2.2.6 scipy==1.15.3 \
        scikit-learn==1.7.2 xgboost==3.2.0 joblib==1.5.3 imbalanced-learn==0.14.2 lightgbm==4.7.0 \
        mlflow==2.22.4 optuna==4.9.0 pandera==0.29.0 pydantic==2.12.5 boto3==1.42.61 \
        openpyxl==3.1.5 pyarrow==19.0.1
    .venv-legacy/Scripts/python legacy_repro.py --out reports/legacy_repro_telco.json

CatBoost, PySpark and torch are not needed (the old model router skips CatBoost when it is
missing). Environment variables set by this script before importing the old code:

    MLFLOW_TRACKING_URI=file:./mlruns_audit   OPTUNA_N_TRIALS=10   ENABLE_STACKING=false
    STORAGE_MODE=local   LOCAL_DATA_PATH=./legacy_data   COMPOSITE_SYNTHESIS_ENABLED=false
    ENABLE_LLM_LAYER=false   MLFLOW_EXPERIMENT_NAME=leakage-audit-legacy

Protocol: the same 80/20 stratified split (random_state=42) as scripts/audit/leakage_audit.py.
The old ``run_automl`` sees only the train part; the model it returns is scored once on the test
part. Profiles are confirmed as a user would: ``Churn Value`` is the target, ``Churn Label`` (its
copy) is ignored, every other role is left as the old profiler suggested.
"""

import argparse
import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

os.environ.update({
    "MLFLOW_TRACKING_URI": "file:./mlruns_audit",
    "MLFLOW_EXPERIMENT_NAME": "leakage-audit-legacy",
    "OPTUNA_N_TRIALS": "10",
    "ENABLE_STACKING": "false",
    "STORAGE_MODE": "local",
    "LOCAL_DATA_PATH": str(ROOT / "legacy_data"),
    "COMPOSITE_SYNTHESIS_ENABLED": "false",
    "ENABLE_LLM_LAYER": "false",
    "DISABLE_PANDERA_IMPORT_WARNING": "True",
})

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from backend.app.core.ingestion.parsers import parse_file  # noqa: E402
from backend.app.core.profiler.column_profile import DataRole  # noqa: E402
from backend.app.core.profiler.orchestrator import run_profiling  # noqa: E402
from backend.app.core.training import automl  # noqa: E402
from backend.app.core.training.mlflow_utils import setup_mlflow  # noqa: E402

SEED = 42
TARGET = "Churn Value"
# Same columns the audit drops before de-duplication.
AUDIT_DROPPED = ["CustomerID", "Count", "Country", "State", "City", "Zip Code",
                 "Lat Long", "Latitude", "Longitude", "Churn Label"]


def add_generic_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Same formulas as the old _generic_feature_engineer, with the column choice made on train."""
    df = df.copy()
    values = {c: pd.to_numeric(df[c], errors="coerce").fillna(0.0) for c in columns}
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            df[f"gen_ratio_{a}__{b}"] = values[a] / (values[b].abs() + 1e-6)
    for a in columns:
        df[f"gen_log1p_{a}"] = np.log1p(values[a].abs())
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/legacy_repro_telco.json")
    args = parser.parse_args()
    t0 = time.perf_counter()

    path = ROOT / "data" / "raw" / "Telco_customer_churn.xlsx"
    df = parse_file(content=path.read_bytes(), filename=path.name).df
    audit_view = df.drop(columns=AUDIT_DROPPED)
    if audit_view.duplicated().any():
        raise RuntimeError("Telco has duplicates in the audit view; the split would not match the audit.")
    y = df[TARGET].astype(int)
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=y, random_state=SEED)
    train_df = df.loc[sorted(train_idx)].reset_index(drop=True)
    test_df = df.loc[sorted(test_idx)].reset_index(drop=True)

    profiles, target_analysis = run_profiling(train_df.copy())
    out = {
        "generated_by": "python legacy_repro.py (on a checkout of archive/before-cleanup)",
        "settings": {k: os.environ[k] for k in (
            "OPTUNA_N_TRIALS", "ENABLE_STACKING", "MLFLOW_TRACKING_URI", "COMPOSITE_SYNTHESIS_ENABLED")},
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "profiler_recommended_target": target_analysis.recommended_target,
        "profiler_leakage_flags": {p.name: p.leakage_score for p in profiles if p.potential_leakage},
    }
    for p in profiles:
        if p.name == TARGET:
            p.inferred_role = DataRole.TARGET
        elif p.name == "Churn Label" or p.inferred_role == DataRole.TARGET:
            p.inferred_role = DataRole.IGNORE
    out["confirmed_roles"] = {p.name: p.inferred_role.value for p in profiles}

    setup_mlflow()
    t_train = time.perf_counter()
    model_uri, _ = automl.run_automl(train_df.copy(), profiles, TARGET, dataset_id="audit-telco")
    out["train_seconds"] = round(time.perf_counter() - t_train, 1)
    run = mlflow.tracking.MlflowClient().get_run(model_uri.split("/")[1])
    out["model_class"] = run.data.tags.get("model_class")
    out["optuna_n_trials"] = int(run.data.params.get("optuna_n_trials"))
    out["mlflow_metrics"] = run.data.metrics
    out["reported_best_roc_auc"] = run.data.metrics.get("best_roc_auc")

    model = mlflow.sklearn.load_model(model_uri)
    y_test = test_df[TARGET].astype(int)

    # Defect 4: score the raw uploaded columns, as the old /predict does.
    try:
        model.predict_proba(test_df)
        out["predict_on_raw_columns"] = "ok"
    except Exception as e:
        out["predict_on_raw_columns"] = f"{type(e).__name__}: {str(e)[:200]}"

    fe_train, _ = automl._generic_feature_engineer(
        *automl._auto_feature_engineer(train_df.copy(), profiles, TARGET), TARGET
    )
    chosen = [c[len("gen_log1p_"):] for c in fe_train.columns if c.startswith("gen_log1p_")]
    out["generic_fe_columns_chosen_on_train"] = chosen
    test_fe, _ = automl._auto_feature_engineer(test_df.copy(), profiles, TARGET)
    test_fe = add_generic_features(test_fe, chosen)
    out["test_roc_auc"] = float(roc_auc_score(y_test, model.predict_proba(test_fe)[:, 1]))

    libs = ["pandas", "numpy", "scikit-learn", "xgboost", "lightgbm", "imbalanced-learn",
            "mlflow", "optuna", "pandera", "pydantic"]
    out["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "libraries": {name: metadata.version(name) for name in libs},
    }
    out["runtime_seconds"] = round(time.perf_counter() - t0, 1)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({k: out[k] for k in ("reported_best_roc_auc", "test_roc_auc", "model_class",
                                          "optuna_n_trials", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
