"""Re-create the small test fixtures from the raw datasets.

    python -m tests.fixtures.make_fixtures

Rows are a stratified sample (random_state=42) of the raw files, kept in their raw form
(original column names, blank strings, duplicates), so tests exercise the same cleaning as the app.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

SAMPLES = {
    "telco_sample.csv": (ROOT / "data" / "raw" / "Telco_customer_churn.xlsx", "Churn Value", 800),
    "iranian_sample.csv": (ROOT / "data" / "raw" / "iranian_churn" / "Customer Churn.csv", "Churn", 500),
}


def main() -> None:
    for name, (path, target, n) in SAMPLES.items():
        df = pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path, encoding="utf-8")
        sample, _ = train_test_split(df, train_size=n, stratify=df[target], random_state=42)
        sample.sort_index().to_csv(HERE / name, index=False, encoding="utf-8", lineterminator="\n")
        print(f"wrote {name}: {len(sample)} rows")


if __name__ == "__main__":
    main()
