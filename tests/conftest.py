from pathlib import Path

import pandas as pd
import pytest

from src.churn.data import prepare, read_table
from src.churn.profile import profile_dataset
from src.churn.train import train

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TELCO = FIXTURES / "telco_sample.csv"
IRANIAN = FIXTURES / "iranian_sample.csv"


@pytest.fixture(scope="session")
def telco_data():
    return prepare(read_table(TELCO), "Churn Value")


@pytest.fixture(scope="session")
def iranian_data():
    return prepare(read_table(IRANIAN), "Churn")


@pytest.fixture(scope="session")
def telco_profile(telco_data):
    return profile_dataset(telco_data)


@pytest.fixture(scope="session")
def iranian_profile(iranian_data):
    return profile_dataset(iranian_data)


@pytest.fixture(scope="session")
def telco_result(telco_data, telco_profile):
    """A small model trained on the Telco sample with the profiler's suggested columns."""
    features = telco_profile["suggested_features"]
    X = telco_data.df.loc[telco_data.train_idx, features]
    y = telco_data.y.loc[telco_data.train_idx]
    return train(X, y, target=telco_data.target, positive_label=telco_data.positive_label,
                 n_trials=2, timeout=None, mlflow_dir=None)


@pytest.fixture
def telco_raw():
    """The fixture file exactly as a user would upload it (raw names, blanks, extra columns)."""
    return pd.read_csv(TELCO, encoding="utf-8")
