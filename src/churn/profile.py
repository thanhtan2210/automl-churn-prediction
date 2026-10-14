"""Column roles and leakage warnings, measured on the training split only.

Thresholds come from reports/single_feature_auc_*.csv (Phase 1 audit): the strongest legitimate
columns reach a single-column AUC of 0.74 (IBM Telco) and 0.79 (UCI Iranian), while the known leaks
reach 1.00 (Churn Label, Churn Reason) and 0.94 (Churn Score).
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder

from .data import SEED, PreparedData

TARGET_COPY_AUC = 0.99
LIKELY_LEAKAGE_AUC = 0.90
MISSING_REVEALS_AUC = 0.90
ID_UNIQUE_RATIO = 0.5
TIMING_KEYWORDS = ("churn", "reason", "score", "label", "cancel", "exit", "attrition", "retention")
# Geographic identifiers (zip codes, coordinates, cities): weak and high-cardinality, excluded by default.
LOCATION_PATTERN = re.compile(r"\b(zip|postal|postcode|latitude|longitude|lat|lon|lng|city|address)\b", re.IGNORECASE)

# Flags that exclude a column by default; check_timing is only a question to the user.
EXCLUDING = ("id", "constant", "datetime", "location", "target_copy", "likely_leakage", "missing_reveals_target")


def infer_role(s: pd.Series) -> str:
    non_null = s.dropna()
    if non_null.nunique() <= 1:
        return "constant"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        is_int = bool(np.all(np.mod(non_null, 1) == 0))
        if is_int and non_null.is_unique and non_null.is_monotonic_increasing:
            return "id"
        return "numeric"
    text = non_null.astype(str)
    if text.nunique() / len(text) > ID_UNIQUE_RATIO:
        return "id"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(text, errors="coerce", format="mixed")
    if parsed.notna().mean() >= 0.9:
        return "datetime"
    return "categorical"


def single_column_auc(s: pd.Series, y: pd.Series) -> float:
    """5-fold CV ROC-AUC of a shallow LightGBM on one column (missing values kept).

    Same method as the Phase 1 audit, so the thresholds above stay comparable.
    """
    if pd.api.types.is_numeric_dtype(s):
        x = s.astype(float).to_frame()
    else:
        codes = OrdinalEncoder().fit_transform(s.astype(object).fillna("__missing__").astype(str).to_frame())
        x = pd.DataFrame(codes, columns=[s.name], index=s.index)
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED, verbose=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return float(np.mean(cross_val_score(model, x, y, cv=cv, scoring="roc_auc")))


def symmetric_auc(y, score) -> float:
    auc = roc_auc_score(y, score)
    return float(max(auc, 1.0 - auc))


def profile_column(s: pd.Series, y: pd.Series) -> dict:
    role = infer_role(s)
    missing = s.isna()
    info = {
        "name": str(s.name),
        "role": role,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "missing_rate": float(missing.mean()),
        "single_column_auc": None,
        "missing_indicator_auc": None,
        "flags": [],
    }
    if role in ("id", "constant", "datetime"):
        info["flags"].append(role)
    if LOCATION_PATTERN.search(info["name"]):
        info["flags"].append("location")
    if missing.any():
        info["missing_indicator_auc"] = symmetric_auc(y, missing.astype(int))
        if info["missing_indicator_auc"] >= MISSING_REVEALS_AUC:
            info["flags"].append("missing_reveals_target")
    if role in ("numeric", "categorical"):
        auc = single_column_auc(s, y)
        info["single_column_auc"] = auc
        if auc >= TARGET_COPY_AUC:
            info["flags"].append("target_copy")
        elif auc >= LIKELY_LEAKAGE_AUC:
            info["flags"].append("likely_leakage")
    if any(k in info["name"].lower() for k in TIMING_KEYWORDS):
        info["flags"].append("check_timing")
    info["suggested"] = "exclude" if any(f in EXCLUDING for f in info["flags"]) else "use"
    return info


def profile_dataset(data: PreparedData) -> dict:
    """Profile every non-target column on the training split. Returns plain JSON-able data."""
    train = data.df.loc[data.train_idx]
    y = data.y.loc[data.train_idx]
    columns = [profile_column(train[c], y) for c in data.features]
    return {
        "target": data.target,
        "positive_label": data.positive_label,
        "computed_on": "training split only",
        "n_rows_raw": data.n_rows_raw,
        "n_duplicates_removed": data.n_duplicates_removed,
        "n_train": int(len(data.train_idx)),
        "n_test": int(len(data.test_idx)),
        "positive_rate_train": float(y.mean()),
        "thresholds": {
            "target_copy_auc": TARGET_COPY_AUC,
            "likely_leakage_auc": LIKELY_LEAKAGE_AUC,
            "missing_reveals_target_auc": MISSING_REVEALS_AUC,
            "id_unique_ratio": ID_UNIQUE_RATIO,
            "timing_keywords": list(TIMING_KEYWORDS),
            "location_pattern": LOCATION_PATTERN.pattern,
        },
        "columns": columns,
        "suggested_features": [c["name"] for c in columns if c["suggested"] == "use"],
        "suggested_excluded": {
            c["name"]: [f for f in c["flags"] if f in EXCLUDING] for c in columns if c["suggested"] == "exclude"
        },
        "timing_questions": [c["name"] for c in columns if "check_timing" in c["flags"] and c["suggested"] == "use"],
    }
