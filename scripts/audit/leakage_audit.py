"""Leakage audit for the churn datasets (cleanup Phase 1).

Measures how much of the old AutoML score came from leakage, using a fixed LightGBM so that
only the evaluation protocol changes between rows. Independent of ``backend/``.

Usage (from the repo root):
    python -m scripts.audit.leakage_audit --dataset telco
    python -m scripts.audit.leakage_audit --dataset iranian
    python -m scripts.audit.leakage_audit --render-only   # rebuild the markdown from the JSON files

Outputs:
    reports/leakage_audit_<dataset>.json
    reports/single_feature_auc_<dataset>.csv
    docs/cleanup/leakage_audit.md (rendered from whichever JSON files exist)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.churn.runinfo import environment, sha256_file
from src.churn.stats import ci95, paired_bootstrap

SEED = 42
N_BOOTSTRAP = 1000
ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
DOC_PATH = ROOT / "docs" / "cleanup" / "leakage_audit.md"

# Identifier / location columns, never used as features.
ID_COLUMNS = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude",
]
TELCO_TARGET = "Churn Value"
TELCO_TARGET_COPY = "Churn Label"
TELCO_REASON = ("Churn Reason",)
TELCO_SCORES = ("Churn Score", "CLTV")

# Thresholds from the Phase 1 brief: above these, stop and ask before calling a column leakage.
IRANIAN_SUSPECT_COLUMNS = ("Status", "Customer Value")
SINGLE_AUC_ALERT = 0.90
CLEAN_TEST_AUC_ALERT = 0.97


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

@dataclass
class Dataset:
    name: str
    file: Path
    df: pd.DataFrame          # de-duplicated, feature candidates + target
    target: str
    features: list[str]       # widest feature set (S0)
    train_idx: pd.Index
    test_idx: pd.Index
    info: dict

    @property
    def y(self) -> pd.Series:
        return self.df[self.target]


def load_raw(name: str) -> tuple[pd.DataFrame, str, Path]:
    if name == "telco":
        path = ROOT / "data" / "raw" / "Telco_customer_churn.xlsx"
        df = pd.read_excel(path)
        # 11 rows have a blank string (new customers, tenure 0); a type fix, nothing learned.
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        return df, TELCO_TARGET, path
    if name == "iranian":
        path = ROOT / "data" / "raw" / "iranian_churn" / "Customer Churn.csv"
        df = pd.read_csv(path, encoding="utf-8")
        return df, "Churn", path
    raise ValueError(f"Unknown dataset: {name}")


def prepare(name: str) -> tuple[Dataset, pd.DataFrame]:
    """Returns the de-duplicated dataset with its split, plus the raw frame (for single-feature AUC)."""
    raw, target, path = load_raw(name)
    dropped = [c for c in ID_COLUMNS + [TELCO_TARGET_COPY] if c in raw.columns]
    df = raw.drop(columns=dropped)
    features = [c for c in df.columns if c != target]

    # De-duplicate on all feature candidates + target, before any split.
    n_raw = len(df)
    keep = ~df.duplicated(keep="first")
    df = df.loc[keep].reset_index(drop=True)
    raw = raw.loc[keep.values].reset_index(drop=True)
    y = df[target].astype(int)
    df[target] = y

    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, stratify=y, random_state=SEED
    )
    train_idx, test_idx = pd.Index(sorted(train_idx)), pd.Index(sorted(test_idx))

    info = {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "n_rows_raw": int(n_raw),
        "n_duplicates_removed": int(n_raw - len(df)),
        "n_rows": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "positive_rate_all": float(y.mean()),
        "positive_rate_train": float(y.loc[train_idx].mean()),
        "positive_rate_test": float(y.loc[test_idx].mean()),
        "target": target,
        "dropped_identifier_columns": dropped,
        "feature_candidates": features,
    }
    return Dataset(name, path, df, target, features, train_idx, test_idx, info), raw


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------

def rank_by_target_corr(X: pd.DataFrame, y, k: int) -> list[str]:
    """Top-k numeric columns by |Pearson corr| with the target (as the old generic FE did)."""
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if len(numeric) < 2:
        return []
    y = pd.Series(np.asarray(y, dtype=float), index=X.index)
    corr = {}
    for c in numeric:
        value = X[c].corr(y)
        corr[c] = 0.0 if pd.isna(value) else abs(float(value))
    return sorted(corr, key=corr.get, reverse=True)[:k]


class GenericRatioFE(BaseEstimator, TransformerMixin):
    """Re-implementation of the old ``_generic_feature_engineer``.

    Picks the top ``max_pairs`` numeric columns by |corr| with the target, then adds pairwise
    ratios ``a / (|b| + 1e-6)`` and ``log1p(|a|)``. With ``fixed_columns`` the selection is
    given from outside (used to reproduce selection on all rows before the split).
    """

    def __init__(self, max_pairs: int = 5, fixed_columns: list[str] | None = None):
        self.max_pairs = max_pairs
        self.fixed_columns = fixed_columns

    def fit(self, X, y=None):
        if self.fixed_columns is not None:
            self.selected_ = [c for c in self.fixed_columns if c in X.columns]
        else:
            self.selected_ = rank_by_target_corr(X, y, self.max_pairs)
        return self

    def transform(self, X):
        X = X.copy()
        cols = self.selected_
        values = {c: pd.to_numeric(X[c], errors="coerce").fillna(0.0) for c in cols}
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                X[f"gen_ratio_{a}__{b}"] = values[a] / (values[b].abs() + 1e-6)
        for a in cols:
            X[f"gen_log1p_{a}"] = np.log1p(values[a].abs())
        return X


def make_tree_prep() -> ColumnTransformer:
    """Categoricals first (ordinal codes), then numerics (median-imputed)."""
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("cat", cat, make_column_selector(dtype_exclude=np.number)),
        ("num", SimpleImputer(strategy="median"), make_column_selector(dtype_include=np.number)),
    ])


def make_sampler(n_categorical: int):
    # Categorical columns come first in make_tree_prep's output.
    if n_categorical > 0:
        return SMOTENC(categorical_features=list(range(n_categorical)), k_neighbors=5, random_state=SEED)
    return SMOTE(k_neighbors=5, random_state=SEED)


def make_lgbm(class_weight=None, **extra) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1,
        class_weight=class_weight, **extra,
    )


def make_logreg() -> Pipeline:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("num", num, make_column_selector(dtype_include=np.number)),
        ("cat", cat, make_column_selector(dtype_exclude=np.number)),
    ])
    model = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)
    return Pipeline([("prep", prep), ("model", model)])


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


# --------------------------------------------------------------------------------------
# (a) Leakage ladder
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage:
    key: str
    description: str
    report: str = "max_val_calibrated"   # or "cv"
    smote: str = "before_cv"             # or "in_pipeline", "none" (class_weight="balanced")
    fe_select: str = "all_rows"          # or "train_fold"
    drop: tuple[str, ...] = ()


S0 = Stage("S0", "As the old code: Churn Score/CLTV/Churn Reason kept; FE columns chosen on all rows; "
                 "SMOTENC on the whole train part before CV; score = max(CV AUC, AUC on a val split "
                 "after isotonic calibration on that same split)")

LADDER = [
    S0,
    replace(S0, key="S1", description="Report CV AUC only (no max with calibrated val AUC)", report="cv"),
    replace(S0, key="S2", description="SMOTENC inside the imblearn Pipeline (fit per fold)",
            report="cv", smote="in_pipeline"),
    replace(S0, key="S3", description="Generic FE column choice fitted on the training fold (in Pipeline)",
            report="cv", smote="in_pipeline", fe_select="train_fold"),
    replace(S0, key="S4", description="Drop Churn Reason", report="cv", smote="in_pipeline",
            fe_select="train_fold", drop=TELCO_REASON),
    replace(S0, key="S5", description="Drop Churn Score and CLTV", report="cv", smote="in_pipeline",
            fe_select="train_fold", drop=TELCO_REASON + TELCO_SCORES),
    replace(S0, key="S6", description="No SMOTE; LightGBM class_weight='balanced'", report="cv",
            smote="none", fe_select="train_fold", drop=TELCO_REASON + TELCO_SCORES),
]

ONE_FIX = [
    replace(S0, key="fix: report CV only", description="S0 but report CV AUC only", report="cv"),
    replace(S0, key="fix: SMOTENC in Pipeline", description="S0 but SMOTENC fitted per fold",
            smote="in_pipeline"),
    replace(S0, key="fix: FE choice on train", description="S0 but generic FE columns chosen on the train part",
            fe_select="train_fold"),
    replace(S0, key="fix: drop Churn Reason", description="S0 without Churn Reason", drop=TELCO_REASON),
    replace(S0, key="fix: drop Churn Score, CLTV", description="S0 without Churn Score and CLTV",
            drop=TELCO_SCORES),
    replace(S0, key="fix: class_weight, no SMOTE", description="S0 with class_weight='balanced' instead of SMOTENC",
            smote="none"),
]

# Settings matching the rows of the pre-audit quick test (for the expected-vs-actual table).
REFERENCE = [
    replace(S0, key="clean features, SMOTENC before CV", description="", report="cv",
            fe_select="train_fold", drop=TELCO_REASON + TELCO_SCORES),
    replace(S0, key="clean features, SMOTENC in Pipeline", description="", report="cv",
            smote="in_pipeline", fe_select="train_fold", drop=TELCO_REASON + TELCO_SCORES),
    replace(S0, key="clean features + Churn Score, SMOTENC before CV", description="", report="cv",
            fe_select="train_fold", drop=TELCO_REASON + ("CLTV",)),
]


def run_stage(stage: Stage, ds: Dataset) -> dict:
    t0 = time.perf_counter()
    feats = [c for c in ds.features if c not in stage.drop]
    X_all, y_all = ds.df[feats], ds.y
    X_tr, y_tr = X_all.loc[ds.train_idx], y_all.loc[ds.train_idx]
    X_te, y_te = X_all.loc[ds.test_idx], y_all.loc[ds.test_idx]

    fixed = rank_by_target_corr(X_all, y_all, 5) if stage.fe_select == "all_rows" else None
    n_cat = sum(not pd.api.types.is_numeric_dtype(X_all[c]) for c in feats)
    class_weight = "balanced" if stage.smote == "none" else None

    if stage.report == "max_val_calibrated":
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED
        )
    else:
        X_fit, y_fit = X_tr, y_tr

    if stage.smote == "before_cv":
        # Old behaviour: oversample the whole fitting part once, then cross-validate on it.
        fe = GenericRatioFE(fixed_columns=fixed).fit(X_fit, y_fit)
        prep = make_tree_prep().fit(fe.transform(X_fit))
        Z = prep.transform(fe.transform(X_fit))
        Z_res, y_res = make_sampler(n_cat).fit_resample(Z, y_fit)
        cv_scores = cross_val_score(make_lgbm(class_weight), Z_res, y_res, cv=make_cv(), scoring="roc_auc")
        model = make_lgbm(class_weight).fit(Z_res, y_res)
        predictor = Pipeline([("fe", fe), ("prep", prep), ("model", model)])  # already fitted
        selected = fe.selected_
        n_fit_rows = int(len(y_res))
    else:
        steps = [("fe", GenericRatioFE(fixed_columns=fixed)), ("prep", make_tree_prep())]
        if stage.smote == "in_pipeline":
            steps.append(("smote", make_sampler(n_cat)))
        steps.append(("model", make_lgbm(class_weight)))
        pipe = ImbPipeline(steps)
        cv_scores = cross_val_score(pipe, X_fit, y_fit, cv=make_cv(), scoring="roc_auc")
        predictor = clone(pipe).fit(X_fit, y_fit)
        selected = predictor.named_steps["fe"].selected_
        n_fit_rows = int(len(y_fit))

    cv_mean, cv_std = float(np.mean(cv_scores)), float(np.std(cv_scores))
    reported = cv_mean
    final = predictor
    val = None
    if stage.report == "max_val_calibrated":
        pre = float(roc_auc_score(y_val, predictor.predict_proba(X_val)[:, 1]))
        method = "isotonic" if len(X_val) >= 1000 else "sigmoid"  # same rule as the old code
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # cv="prefit" is deprecated in sklearn>=1.6
            cal = CalibratedClassifierCV(predictor, method=method, cv="prefit").fit(X_val, y_val)
        post = float(roc_auc_score(y_val, cal.predict_proba(X_val)[:, 1]))
        accepted = post >= pre - 0.005
        if accepted:
            final = cal
            reported = max(cv_mean, post)
        val = {
            "n_val": int(len(y_val)), "calibration": method,
            "val_auc_before_calibration": pre, "val_auc_after_calibration_on_val": post,
            "calibration_accepted": bool(accepted),
        }

    test_auc = float(roc_auc_score(y_te, final.predict_proba(X_te)[:, 1]))
    return {
        "key": stage.key,
        "description": stage.description,
        "settings": {k: v for k, v in asdict(stage).items() if k not in ("key", "description")},
        "n_features_in": len(feats),
        "fe_selected_columns": list(selected),
        "n_rows_cv": n_fit_rows,
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        "cv_auc_folds": [float(s) for s in cv_scores],
        "val": val,
        "reported_score": float(reported),
        "test_auc": test_auc,
        "seconds": round(time.perf_counter() - t0, 2),
    }


# --------------------------------------------------------------------------------------
# (b) Final clean model vs baseline, test set used once
# --------------------------------------------------------------------------------------

def final_comparison(ds: Dataset, drop: tuple[str, ...]) -> dict:
    t0 = time.perf_counter()
    feats = [c for c in ds.features if c not in drop]
    X_tr, y_tr = ds.df.loc[ds.train_idx, feats], ds.y.loc[ds.train_idx]
    X_te, y_te = ds.df.loc[ds.test_idx, feats], ds.y.loc[ds.test_idx]

    models = {
        "lightgbm": ImbPipeline([
            ("fe", GenericRatioFE()), ("prep", make_tree_prep()), ("model", make_lgbm("balanced")),
        ]),
        "logistic_regression": make_logreg(),
    }
    out: dict = {"features": feats, "n_features": len(feats)}
    proba = {}
    for name, pipe in models.items():
        cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=make_cv(), scoring="roc_auc")
        fitted = clone(pipe).fit(X_tr, y_tr)
        proba[name] = fitted.predict_proba(X_te)[:, 1]
        out[name] = {
            "cv_auc_mean": float(np.mean(cv_scores)),
            "cv_auc_std": float(np.std(cv_scores)),
            "test_roc_auc": float(roc_auc_score(y_te, proba[name])),
            "test_pr_auc": float(average_precision_score(y_te, proba[name])),
        }

    # Paired bootstrap on the test set: same resampled rows for both models.
    y = y_te.to_numpy()
    a, b = proba["lightgbm"], proba["logistic_regression"]
    boot = paired_bootstrap(y, a, b, n_boot=N_BOOTSTRAP, seed=SEED)

    out["lightgbm"]["test_roc_auc_ci95"] = ci95(boot["roc_a"])
    out["lightgbm"]["test_pr_auc_ci95"] = ci95(boot["pr_a"])
    out["logistic_regression"]["test_roc_auc_ci95"] = ci95(boot["roc_b"])
    out["logistic_regression"]["test_pr_auc_ci95"] = ci95(boot["pr_b"])
    out["difference_lightgbm_minus_lr"] = {
        "roc_auc": out["lightgbm"]["test_roc_auc"] - out["logistic_regression"]["test_roc_auc"],
        "roc_auc_ci95": ci95(boot["roc_diff"]),
        "pr_auc": out["lightgbm"]["test_pr_auc"] - out["logistic_regression"]["test_pr_auc"],
        "pr_auc_ci95": ci95(boot["pr_diff"]),
    }
    out["bootstrap"] = {"n_requested": N_BOOTSTRAP, "n_used": len(boot["roc_a"]), "paired": True, "seed": SEED}
    out["test_n"] = int(len(y))
    out["test_positive_rate"] = float(y.mean())
    out["seconds"] = round(time.perf_counter() - t0, 2)
    return out


# --------------------------------------------------------------------------------------
# (c) Single-feature AUC
# --------------------------------------------------------------------------------------

def single_feature_auc(raw: pd.DataFrame, ds: Dataset) -> pd.DataFrame:
    """CV AUC of a shallow LightGBM on one column at a time, on the training split only."""
    rows = []
    y = ds.y.loc[ds.train_idx]
    for col in raw.columns:
        if col == ds.target:
            continue
        s = raw.loc[ds.train_idx, col]
        if pd.api.types.is_numeric_dtype(s):
            x = s.to_frame()
        else:
            # Sorted-category codes (no label information, independent of row order).
            enc = OrdinalEncoder()
            x = pd.DataFrame(enc.fit_transform(s.astype("object").fillna("__missing__").astype(str).to_frame()),
                             columns=[col], index=s.index)
        scores = cross_val_score(make_lgbm(max_depth=3), x, y, cv=make_cv(), scoring="roc_auc")
        missing = s.isna()
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "missing_rate": float(missing.mean()),
            "missing_rate_positive": float(missing[y == 1].mean()),
            "missing_rate_negative": float(missing[y == 0].mean()),
            "missing_indicator_auc": float(roc_auc_score(y, missing.astype(int))),
            "cv_auc_mean": float(np.mean(scores)),
            "cv_auc_std": float(np.std(scores)),
            "used_as_feature": col in ds.features,
        })
    return pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False, kind="mergesort").reset_index(drop=True)


def high_auc_diagnostics(ds: Dataset, suspects: tuple[str, ...], derived: str) -> dict:
    """Where does a very high clean AUC come from? Training split only, the test set is not touched."""
    t0 = time.perf_counter()
    X, y = ds.df.loc[ds.train_idx, ds.features], ds.y.loc[ds.train_idx]

    def lgbm_pipe():
        return ImbPipeline([("fe", GenericRatioFE()), ("prep", make_tree_prep()), ("model", make_lgbm("balanced"))])

    def cv_auc(frame: pd.DataFrame) -> float:
        return float(np.mean(cross_val_score(lgbm_pipe(), frame, y, cv=make_cv(), scoring="roc_auc")))

    base = cv_auc(X)
    drop_one = [{"dropped": c, "cv_auc": (s := cv_auc(X.drop(columns=[c]))), "delta": s - base} for c in ds.features]
    drop_suspects = cv_auc(X.drop(columns=list(suspects)))

    pairs = []
    for i, a in enumerate(ds.features):
        for b in ds.features[i + 1:]:
            s = cross_val_score(make_lgbm(max_depth=3), X[[a, b]], y, cv=make_cv(), scoring="roc_auc")
            pairs.append({"columns": [a, b], "cv_auc": float(np.mean(s))})
    pairs.sort(key=lambda r: r["cv_auc"], reverse=True)

    # Is the derived column a function of the others?
    others = [c for c in ds.features if c != derived]
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2 = {}
    for label, reg in (("linear", LinearRegression()),
                       ("lightgbm", LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1))):
        r2[label] = float(np.mean(cross_val_score(reg, X[others], X[derived], cv=kfold, scoring="r2")))

    rates = {}
    for c in suspects + ("Complains",):
        if c in X.columns and X[c].nunique() <= 10:
            g = y.groupby(X[c])
            rates[c] = {str(k): {"n": int(g.size()[k]), "churn_rate": float(g.mean()[k])} for k in g.size().index}

    return {
        "computed_on": "training split only (5-fold CV); S6 LightGBM pipeline unless stated",
        "cv_auc_all_features": base,
        "drop_one": drop_one,
        "drop_suspects": {"dropped": list(suspects), "cv_auc": drop_suspects},
        "top_pairs_shallow_lgbm": pairs[:10],
        "derived_column": derived,
        "derived_column_r2_from_others": r2,
        "churn_rate_by_value": rates,
        "seconds": round(time.perf_counter() - t0, 2),
    }


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def audit(name: str) -> dict:
    t0 = time.perf_counter()
    ds, raw = prepare(name)
    print(f"[{name}] rows={ds.info['n_rows']} (removed {ds.info['n_duplicates_removed']} duplicates), "
          f"train={ds.info['n_train']}, test={ds.info['n_test']}", flush=True)

    report: dict = {
        "dataset": name,
        "generated_by": "python -m scripts.audit.leakage_audit --dataset " + name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "data": ds.info,
        "protocol": {
            "dedup": "exact duplicates over all feature candidates + target, before the split",
            "split": "80/20 stratified train_test_split, random_state=42",
            "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on the train part",
            "model": "LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)",
            "tree_preprocessing": "categoricals: impute 'missing' + OrdinalEncoder; numerics: median impute",
            "baseline": "LogisticRegression(class_weight='balanced'), one-hot + StandardScaler in a Pipeline",
            "cv_auc_std": "population std (ddof=0) over the 5 folds",
        },
    }

    if name == "telco":
        report["ladder"] = []
        for stage in LADDER:
            r = run_stage(stage, ds)
            print(f"  {r['key']}: CV {r['cv_auc_mean']:.4f} ± {r['cv_auc_std']:.4f}, "
                  f"reported {r['reported_score']:.4f}, test {r['test_auc']:.4f} ({r['seconds']} s)", flush=True)
            report["ladder"].append(r)
        report["one_fix_from_s0"] = []
        for stage in ONE_FIX:
            r = run_stage(stage, ds)
            print(f"  {r['key']}: reported {r['reported_score']:.4f}, test {r['test_auc']:.4f}", flush=True)
            report["one_fix_from_s0"].append(r)
        report["reference_settings"] = []
        for stage in REFERENCE:
            r = run_stage(stage, ds)
            print(f"  ref {r['key']}: CV {r['cv_auc_mean']:.4f}, test {r['test_auc']:.4f}", flush=True)
            report["reference_settings"].append(r)
        clean_drop = TELCO_REASON + TELCO_SCORES
    else:
        clean_drop = ()

    # Duplicates that only appear once the dropped columns are gone (informational).
    clean_cols = [c for c in ds.features if c not in clean_drop] + [ds.target]
    report["data"]["n_duplicates_in_clean_feature_set"] = int(ds.df[clean_cols].duplicated().sum())
    report["data"]["n_conflicting_label_duplicates_in_clean_feature_set"] = int(
        ds.df.groupby([c for c in clean_cols if c != ds.target], dropna=False)[ds.target]
        .nunique().gt(1).sum()
    )

    report["final"] = final_comparison(ds, clean_drop)
    f = report["final"]
    print(f"  final LightGBM test ROC-AUC {f['lightgbm']['test_roc_auc']:.4f}, "
          f"LR {f['logistic_regression']['test_roc_auc']:.4f}", flush=True)

    sf = single_feature_auc(raw, ds)
    csv_path = REPORTS_DIR / f"single_feature_auc_{name}.csv"
    sf.to_csv(csv_path, index=False, encoding="utf-8", float_format="%.6f")
    report["single_feature"] = {
        "file": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
        "computed_on": "training split only, 5-fold CV, LGBMClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)",
        "rows": sf.to_dict(orient="records"),
    }

    if name == "iranian":
        single = {r["column"]: r["cv_auc_mean"] for r in report["single_feature"]["rows"]}
        checks = {f"single_auc_{c}": single.get(c) for c in IRANIAN_SUSPECT_COLUMNS}
        checks["clean_test_auc_lightgbm"] = f["lightgbm"]["test_roc_auc"]
        checks["triggered"] = bool(
            any(single.get(c, 0) > SINGLE_AUC_ALERT for c in IRANIAN_SUSPECT_COLUMNS)
            or f["lightgbm"]["test_roc_auc"] > CLEAN_TEST_AUC_ALERT
        )
        checks["rule"] = (f"stop if single-feature AUC of {', '.join(IRANIAN_SUSPECT_COLUMNS)} > {SINGLE_AUC_ALERT} "
                          f"or clean LightGBM test AUC > {CLEAN_TEST_AUC_ALERT}")
        report["stop_check"] = checks
        if checks["triggered"]:
            print("  STOP CHECK TRIGGERED: review before calling any column leakage.", flush=True)
        report["diagnostics"] = high_auc_diagnostics(ds, IRANIAN_SUSPECT_COLUMNS, "Customer Value")

    report["environment"] = environment(
        ["pandas", "numpy", "scikit-learn", "imbalanced-learn", "lightgbm", "openpyxl", "scipy"]
    )
    report["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    return report


# --------------------------------------------------------------------------------------
# Markdown rendering (numbers only from the JSON files)
# --------------------------------------------------------------------------------------

def _f(x: float | None, d: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def _ci_str(ci: list[float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _final_table(f: dict) -> list[str]:
    lines = [
        "| Model | CV ROC-AUC (train) | Test ROC-AUC [95% CI] | Test PR-AUC [95% CI] |",
        "|---|---|---|---|",
    ]
    for key, label in (("lightgbm", "LightGBM (S6 pipeline)"), ("logistic_regression", "Logistic Regression baseline")):
        m = f[key]
        lines.append(f"| {label} | {_f(m['cv_auc_mean'])} ± {_f(m['cv_auc_std'])} | "
                     f"{_f(m['test_roc_auc'])} {_ci_str(m['test_roc_auc_ci95'])} | "
                     f"{_f(m['test_pr_auc'])} {_ci_str(m['test_pr_auc_ci95'])} |")
    d = f["difference_lightgbm_minus_lr"]
    lines.append(f"| Difference (LightGBM − LR), paired | | {d['roc_auc']:+.3f} {_ci_str(d['roc_auc_ci95'])} | "
                 f"{d['pr_auc']:+.3f} {_ci_str(d['pr_auc_ci95'])} |")
    lines.append("")
    lines.append(f"Test set: n = {f['test_n']}, positive rate = {_f(f['test_positive_rate'])} "
                 f"(the PR-AUC of a random ranking). Bootstrap: {f['bootstrap']['n_used']} paired resamples.")
    return lines


def _single_table(rows: list[dict], top: int) -> list[str]:
    lines = [
        "| Column | Used as feature | CV AUC alone | Missing-indicator AUC | Missing rate (churn / no churn) |",
        "|---|---|---|---|---|",
    ]
    for r in rows[:top]:
        lines.append(f"| `{r['column']}` | {'yes' if r['used_as_feature'] else 'no'} | "
                     f"{_f(r['cv_auc_mean'])} ± {_f(r['cv_auc_std'])} | {_f(r['missing_indicator_auc'])} | "
                     f"{r['missing_rate_positive']:.1%} / {r['missing_rate_negative']:.1%} |")
    return lines


def _stage_row(r: dict) -> str:
    val = r["val"]
    val_txt = "—" if val is None else f"{_f(val['val_auc_after_calibration_on_val'])}"
    return (f"| {r['key']} | {r['description']} | {_f(r['cv_auc_mean'])} ± {_f(r['cv_auc_std'])} | "
            f"{val_txt} | **{_f(r['reported_score'])}** | {_f(r['test_auc'])} |")


def render_markdown() -> None:
    reports = {}
    for name in ("telco", "iranian"):
        p = REPORTS_DIR / f"leakage_audit_{name}.json"
        if p.exists():
            with open(p, encoding="utf-8") as fh:
                reports[name] = json.load(fh)

    L: list[str] = [
        "# Leakage audit",
        "",
        "<!-- Generated by `python -m scripts.audit.leakage_audit`. Do not edit by hand. -->",
        "",
        "Every number on this page is read from `reports/leakage_audit_*.json` and "
        "`reports/single_feature_auc_*.csv`, written by `scripts/audit/leakage_audit.py`.",
        "",
        "Protocol: exact duplicate rows removed before the split; 80/20 stratified split "
        "(`random_state=42`); 5-fold stratified CV on the train part; one fixed "
        "`LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)` for every row, no tuning. "
        "The test part is only scored, never used to choose anything.",
        "",
    ]

    if "telco" in reports:
        t = reports["telco"]
        d = t["data"]
        L += [
            "## IBM Telco",
            "",
            f"`{d['file']}`: {d['n_rows_raw']} rows, {d['n_duplicates_removed']} duplicates removed, "
            f"train {d['n_train']} / test {d['n_test']}, churn rate {d['positive_rate_all']:.1%}. "
            f"Always dropped (identifiers, and `Churn Label`, a copy of the target): "
            f"{', '.join('`' + c + '`' for c in d['dropped_identifier_columns'])}.",
            "",
            "### (a) Leakage ladder",
            "",
            "Each step fixes one more problem than the step above. *Reported* is the number the "
            "old system would show: for S0 it is max(CV AUC, AUC on the validation split after "
            "calibrating on that same split); from S1 on it is the CV AUC.",
            "",
            "| Step | Change | CV AUC (mean ± std) | Calibrated val AUC | Reported | Test AUC |",
            "|---|---|---|---|---|---|",
        ]
        L += [_stage_row(r) for r in t["ladder"]]
        L += [
            "",
            "### One fix at a time from S0",
            "",
            "The ladder depends on the order of the fixes, so each row below changes exactly one thing "
            "relative to S0.",
            "",
            "| Fix | CV AUC (mean ± std) | Calibrated val AUC | Reported | Test AUC |",
            "|---|---|---|---|---|",
        ]
        for r in t["one_fix_from_s0"]:
            val = r["val"]
            val_txt = "—" if val is None else _f(val["val_auc_after_calibration_on_val"])
            L.append(f"| {r['key']} | {_f(r['cv_auc_mean'])} ± {_f(r['cv_auc_std'])} | {val_txt} | "
                     f"**{_f(r['reported_score'])}** | {_f(r['test_auc'])} |")

        s0, s6 = t["ladder"][0], t["ladder"][-1]
        gaps = [(r["key"], r["cv_auc_mean"] - r["test_auc"]) for r in t["ladder"]]
        sf_rows = {r["column"]: r for r in t["single_feature"]["rows"]}
        reason = sf_rows.get("Churn Reason")
        score = sf_rows.get("Churn Score")
        L += [
            "",
            "### What the numbers say",
            "",
            f"- The old protocol reports **{_f(s0['reported_score'])}** on this file; the clean setup (S6) "
            f"scores **{_f(s6['test_auc'])}** on the untouched test set.",
            "- Gap between CV AUC and test AUC per step (positive = CV is optimistic): "
            + ", ".join(f"{k} {g:+.3f}" for k, g in gaps) + ".",
        ]
        if reason:
            L.append(
                f"- `Churn Reason` is missing for {reason['missing_rate_negative']:.1%} of non-churners and "
                f"{reason['missing_rate_positive']:.1%} of churners in the training split: its missingness alone "
                f"separates the classes (missing-indicator AUC {_f(reason['missing_indicator_auc'])}, "
                f"i.e. {_f(1 - reason['missing_indicator_auc'])} inverted). It only exists after a customer has left."
            )
        if score:
            L.append(
                f"- `Churn Score` alone reaches CV AUC {_f(score['cv_auc_mean'])}. Per IBM's column description, "
                "`Churn Score` and `CLTV` are outputs of IBM's own models, not raw customer attributes."
            )
        refs = {r["key"]: r for r in t.get("reference_settings", [])}
        before = refs.get("clean features, SMOTENC before CV")
        inside = refs.get("clean features, SMOTENC in Pipeline")
        if before and inside:
            L.append(
                "- `Churn Reason` makes every model perfect, which hides the SMOTE effect in the ladder. "
                f"With clean features, SMOTENC before CV gives CV AUC {_f(before['cv_auc_mean'])} vs "
                f"{_f(inside['cv_auc_mean'])} when it is fitted per fold, while test AUC is "
                f"{_f(before['test_auc'])} vs {_f(inside['test_auc'])}: the oversampling leak inflates the "
                "CV estimate, not the model."
            )
        diff = t["final"]["difference_lightgbm_minus_lr"]
        lo, hi = diff["roc_auc_ci95"]
        L.append(
            f"- Clean LightGBM minus Logistic Regression on test ROC-AUC: {diff['roc_auc']:+.3f} "
            f"(95% CI [{lo:.3f}, {hi:.3f}]). "
            + ("The interval contains 0: no evidence that LightGBM beats the linear baseline on this file."
               if lo <= 0 <= hi else
               ("LightGBM is better than the baseline." if lo > 0 else "The baseline is better than LightGBM."))
        )
        fe_all_rows = t["ladder"][2]["fe_selected_columns"]
        fe_train = t["ladder"][3]["fe_selected_columns"]
        same = sorted(fe_all_rows) == sorted(fe_train)
        L.append(
            f"- Generic FE columns chosen on all rows (S2): {', '.join('`' + c + '`' for c in fe_all_rows)}; "
            f"chosen on the train part (S3): {', '.join('`' + c + '`' for c in fe_train)}. "
            + ("The selection is identical, so S2 → S3 isolates no leakage on this file."
               if same else "The selections differ; S2 → S3 shows the effect.")
        )
        L += [
            "",
            "### (b) Clean result on the test set (S6 features)",
            "",
        ]
        L += _final_table(t["final"])
        L += [
            "",
            "### (c) Single-feature AUC (training split, top 15)",
            "",
            f"Full table: `{t['single_feature']['file']}`. Shallow LightGBM (`max_depth=3`) on one column at a "
            "time; text columns are sorted-category codes; missing values kept.",
            "",
        ]
        L += _single_table(t["single_feature"]["rows"], 15)
        if "reference_settings" in t:
            L += [
                "",
                "### Reference settings (for comparison with the pre-audit quick test)",
                "",
                "| Setting | CV AUC (mean ± std) | Test AUC |",
                "|---|---|---|",
            ]
            for r in t["reference_settings"]:
                L.append(f"| {r['key']} | {_f(r['cv_auc_mean'])} ± {_f(r['cv_auc_std'])} | {_f(r['test_auc'])} |")
        L.append("")

    if "iranian" in reports:
        ir = reports["iranian"]
        d = ir["data"]
        L += [
            "## UCI Iranian Churn",
            "",
            f"`{d['file']}`: {d['n_rows_raw']} rows, **{d['n_duplicates_removed']} exact duplicates removed**, "
            f"train {d['n_train']} / test {d['n_test']}, churn rate {d['positive_rate_all']:.1%}. "
            f"After de-duplication, {d['n_conflicting_label_duplicates_in_clean_feature_set']} groups of rows "
            "share identical features but have different labels. "
            "All 13 columns are used as features (no column is excluded).",
            "",
            "### (b) Result on the test set (all features)",
            "",
        ]
        L += _final_table(ir["final"])
        L += [
            "",
            "### (c) Single-feature AUC (training split)",
            "",
            f"Full table: `{ir['single_feature']['file']}`.",
            "",
        ]
        L += _single_table(ir["single_feature"]["rows"], len(ir["single_feature"]["rows"]))
        sc = ir.get("stop_check")
        if sc:
            L += [
                "",
                "### Stop check",
                "",
                f"Rule: {sc['rule']}.",
                "",
            ]
            for c in IRANIAN_SUSPECT_COLUMNS:
                L.append(f"- `{c}` alone: {_f(sc.get('single_auc_' + c))}")
            L.append(f"- Clean LightGBM test AUC: {_f(sc['clean_test_auc_lightgbm'])}")
            L.append(f"- Triggered: **{'yes — needs a decision before any column is called leakage' if sc['triggered'] else 'no'}**")
        dg = ir.get("diagnostics")
        if dg:
            worst = min(dg["drop_one"], key=lambda r: r["delta"])
            max_abs = max(abs(r["delta"]) for r in dg["drop_one"])
            best_pair = dg["top_pairs_shallow_lgbm"][0]
            best_single = ir["single_feature"]["rows"][0]
            L += [
                "",
                "### Where the high AUC comes from (training split only)",
                "",
                f"All 13 columns: CV AUC {_f(dg['cv_auc_all_features'])}.",
                "",
                "| Dropped column | CV AUC | Change |",
                "|---|---|---|",
            ]
            for r in sorted(dg["drop_one"], key=lambda r: r["delta"]):
                L.append(f"| `{r['dropped']}` | {_f(r['cv_auc'])} | {r['delta']:+.3f} |")
            ds_ = dg["drop_suspects"]
            L.append(f"| {' + '.join('`' + c + '`' for c in ds_['dropped'])} | {_f(ds_['cv_auc'])} | "
                     f"{ds_['cv_auc'] - dg['cv_auc_all_features']:+.3f} |")
            L += [
                "",
                "Best two-column models (shallow LightGBM): "
                + "; ".join(f"{' + '.join('`' + c + '`' for c in p['columns'])} {_f(p['cv_auc'])}"
                            for p in dg["top_pairs_shallow_lgbm"][:5]) + ".",
                "",
                f"`{dg['derived_column']}` predicted from the other 12 columns: CV R² "
                f"{_f(dg['derived_column_r2_from_others']['linear'])} (linear), "
                f"{_f(dg['derived_column_r2_from_others']['lightgbm'])} (LightGBM).",
                "",
            ]
            for col, vals in dg["churn_rate_by_value"].items():
                L.append(f"- Churn rate by `{col}`: " + ", ".join(
                    f"{k} → {v['churn_rate']:.1%} (n={v['n']})" for k, v in vals.items()) + ".")
            L += [
                "",
                "**Reading.** The best single column reaches "
                f"{_f(best_single['cv_auc_mean'])} (`{best_single['column']}`) and the best pair "
                f"{_f(best_pair['cv_auc'])}; removing any one column moves CV AUC by at most {max_abs:.3f} "
                f"(largest drop: `{worst['dropped']}`). The score comes from many usage columns together, "
                f"not from one column that encodes the label. `{dg['derived_column']}` is close to a function "
                "of the other columns, so it is redundant rather than a separate leak. Per the UCI description "
                "all columns are aggregated over months 1–9 and the label is taken at month 12. "
                "Decision (2026-10-14): no column is treated as leakage; all 13 are kept, and the AUC is "
                "reported as unusually high for a churn problem. The source does not document why this file "
                "is so separable.",
            ]
        L.append("")

    if reports:
        L += ["## Run environment", ""]
        for name, r in reports.items():
            e = r["environment"]
            libs = ", ".join(f"{k} {v}" for k, v in e["libraries"].items())
            L.append(f"- `{name}`: {r['generated_at']}, {r['runtime_seconds']} s; Python {e['python']}, "
                     f"{e['platform']}, {e['cpu_count']} logical CPUs, {e['ram_gb']} GB RAM; {libs}.")
        L.append("")

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DOC_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))
    print(f"wrote {DOC_PATH.relative_to(ROOT)}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=["telco", "iranian"])
    parser.add_argument("--render-only", action="store_true", help="only rebuild the markdown")
    args = parser.parse_args(argv)
    if not args.render_only and not args.dataset:
        parser.error("--dataset is required unless --render-only is given")

    if args.dataset:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report = audit(args.dataset)
        out = REPORTS_DIR / f"leakage_audit_{args.dataset}.json"
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"wrote {out.relative_to(ROOT)}", flush=True)
    render_markdown()


if __name__ == "__main__":
    sys.exit(main())
