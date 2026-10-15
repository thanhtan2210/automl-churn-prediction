import numpy as np

from src.churn.stats import precision_recall_by_share


def test_precision_recall_by_share_counts_the_top_customers():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    proba = np.linspace(1, 0, 10)  # already ranked
    rows = precision_recall_by_share(y, proba, n_points=10)
    assert [r["share"] for r in rows][:3] == [0.1, 0.2, 0.3]
    assert rows[0] == {"share": 0.1, "precision": 1.0, "recall": 1 / 3}
    assert rows[2]["precision"] == 2 / 3 and rows[2]["recall"] == 2 / 3
    assert rows[-1] == {"share": 1.0, "precision": 0.3, "recall": 1.0}
