"""Loading, cleaning, target encoding and the one train/test split used everywhere."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

# Values recognised as the positive class when no positive_label is given (case-insensitive).
POSITIVE_TOKENS = ("1", "yes", "true", "churn", "churned", "attrited customer")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    target: str
    positive_label: str | None = None


DATASETS = {
    "telco": DatasetSpec("telco", RAW_DIR / "Telco_customer_churn.xlsx", "Churn Value"),
    "iranian": DatasetSpec("iranian", RAW_DIR / "iranian_churn" / "Customer Churn.csv", "Churn"),
}


@dataclass
class PreparedData:
    df: pd.DataFrame          # cleaned, de-duplicated; includes the target column as uploaded
    y: pd.Series              # 0/1 target aligned with df
    target: str
    positive_label: str
    train_idx: pd.Index
    test_idx: pd.Index
    n_rows_raw: int
    n_duplicates_removed: int

    @property
    def features(self) -> list[str]:
        return [c for c in self.df.columns if c != self.target]


def normalize_column_name(name) -> str:
    return re.sub(r"\s+", " ", str(name)).strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    names = [normalize_column_name(c) for c in df.columns]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"Column names collide after normalising whitespace: {dupes}")
    out = df.copy()
    out.columns = names
    return out


def read_table(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif suffix == ".csv":
        df = pd.read_csv(path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type '{suffix}'; use .csv or .xlsx")
    return clean_types(normalize_columns(df))


def clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Blank strings become NaN; text columns whose non-blank values are all numbers become numeric.

    Purely a type fix (no statistics are learned), so it is safe before the split.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if s.dtype != object and not pd.api.types.is_string_dtype(s):
            continue
        stripped = s.astype("string").str.strip()
        stripped = stripped.mask(stripped == "")
        as_num = pd.to_numeric(stripped, errors="coerce")
        if stripped.notna().any() and as_num[stripped.notna()].notna().all():
            out[col] = as_num.astype(float)
        else:
            out[col] = stripped.astype(object).where(stripped.notna(), np.nan)
    return out


def drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drops exact duplicate rows (all columns); keeps the first. Must run before the split."""
    keep = ~df.duplicated(keep="first")
    return df.loc[keep].reset_index(drop=True), int((~keep).sum())


def _norm_label(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip().lower()


def encode_target(series: pd.Series, positive_label=None) -> tuple[pd.Series, str]:
    """Returns (0/1 target, positive label as found in the data).

    With ``positive_label`` the given value is the positive class. Without it, exactly one of the
    two classes must be one of POSITIVE_TOKENS; anything else raises ValueError.
    """
    if series.isna().any():
        raise ValueError(f"Target '{series.name}' has {int(series.isna().sum())} missing values")
    labels = series.map(_norm_label)
    classes = sorted(labels.unique())
    if len(classes) != 2:
        raise ValueError(f"Target '{series.name}' must have exactly 2 classes, found {len(classes)}: {classes[:10]}")

    if positive_label is not None:
        pos = _norm_label(positive_label)
        if pos not in classes:
            raise ValueError(f"positive_label {positive_label!r} not found in target values {classes}")
    else:
        matches = [c for c in classes if c in POSITIVE_TOKENS]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot tell which class of '{series.name}' is churn (values: {classes}). "
                f"Pass positive_label explicitly."
            )
        pos = matches[0]

    original = series[labels == pos].iloc[0]
    return (labels == pos).astype(int).rename(series.name), str(original)


def split_indices(y: pd.Series) -> tuple[pd.Index, pd.Index]:
    """80/20 stratified split with random_state=42, shared by profile, train and evaluate."""
    train_idx, test_idx = train_test_split(y.index, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    return pd.Index(sorted(train_idx)), pd.Index(sorted(test_idx))


def prepare(df: pd.DataFrame, target: str, positive_label=None) -> PreparedData:
    df = normalize_columns(df)
    target = normalize_column_name(target)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")
    n_raw = len(df)
    df, n_dupes = drop_duplicate_rows(df)
    y, pos = encode_target(df[target], positive_label)
    train_idx, test_idx = split_indices(y)
    return PreparedData(df, y, target, pos, train_idx, test_idx, n_raw, n_dupes)


def load_dataset(name: str) -> PreparedData:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Known: {sorted(DATASETS)}")
    spec = DATASETS[name]
    return prepare(read_table(spec.path), spec.target, spec.positive_label)
