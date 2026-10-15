"""Write example "new customer" files for the Predict page.

    python -m src.churn.make_samples

Rows are a random sample (seed 42) of the TEST split (never used for training) of each bundled
dataset, at most 300 per file. A random sample, not the first rows: the IBM Telco file lists its
churners first. Columns: the id column (if the profile found one) plus the features the
evaluation model uses (reports/metrics_<dataset>.json); the target and every excluded column
(leaks, locations, constants) are dropped.
"""

from __future__ import annotations

import json

import numpy as np

from .data import ROOT, SEED, load_dataset

MAX_ROWS = 300
OUT_DIR = ROOT / "data" / "samples"


def make_sample(name: str) -> int:
    with open(ROOT / "reports" / f"metrics_{name}.json", encoding="utf-8") as f:
        report = json.load(f)
    used = report["features"]["used"]
    ids = [c["name"] for c in report["features"]["profile"] if c["role"] == "id"]
    data = load_dataset(name)
    rng = np.random.default_rng(SEED)
    picked = np.sort(rng.choice(data.test_idx, size=min(MAX_ROWS, len(data.test_idx)), replace=False))
    rows = data.df.loc[picked, ids + used]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT_DIR / f"{name}_new_customers.csv", index=False, encoding="utf-8", lineterminator="\n")
    return len(rows)


def main() -> None:
    for name in ("telco", "iranian"):
        n = make_sample(name)
        print(f"wrote data/samples/{name}_new_customers.csv ({n} test-split rows)")


if __name__ == "__main__":
    main()
