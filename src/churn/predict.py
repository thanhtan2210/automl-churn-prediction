"""Score a new customer file with a trained model."""

from __future__ import annotations

import pandas as pd

from .data import normalize_columns
from .explain import top_reasons
from .train import ChurnModel


class MissingColumnsError(ValueError):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"The file is missing {len(missing)} column(s) the model needs: {missing}")


def check_columns(model: ChurnModel, df: pd.DataFrame) -> pd.DataFrame:
    """Normalises column names and checks that every model feature is present. Extra columns are ignored."""
    df = normalize_columns(df)
    missing = [c for c in model.features if c not in df.columns]
    if missing:
        raise MissingColumnsError(missing)
    return df


def predict(model: ChurnModel, df: pd.DataFrame, k: int = 3, positive_only: bool = False) -> pd.DataFrame:
    """Returns one row per customer: churn probability, label at the model's threshold, top-k reasons.

    ``positive_only`` is passed to ``explain.top_reasons``.
    """
    df = check_columns(model, df)
    proba = model.predict_proba(df)
    return pd.DataFrame({
        "churn_probability": proba,
        "predicted_churn": proba >= model.threshold,
        "reasons": top_reasons(model, df, k=k, positive_only=positive_only),
    }, index=df.index)
