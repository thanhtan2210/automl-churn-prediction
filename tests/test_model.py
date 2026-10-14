import numpy as np
import pytest

from src.churn.explain import global_importance, top_reasons
from src.churn.predict import MissingColumnsError, predict
from src.churn.train import load_model, save_model, train


def test_predict_works_on_the_raw_uploaded_columns(telco_result, telco_raw):
    # Regression for defect 4: no engineered columns are needed at prediction time,
    # and extra columns (IDs, the target, leaks) are ignored.
    out = predict(telco_result.model, telco_raw.head(25))
    assert len(out) == 25
    assert out["churn_probability"].between(0, 1).all()
    assert out["predicted_churn"].dtype == bool


def test_predict_reports_missing_columns(telco_result, telco_raw):
    with pytest.raises(MissingColumnsError) as err:
        predict(telco_result.model, telco_raw.drop(columns=["Contract", "Tenure Months"]))
    assert set(err.value.missing) == {"Contract", "Tenure Months"}


def test_predict_accepts_unseen_categories_and_blanks(telco_result, telco_raw):
    rows = telco_raw.head(3).copy()
    rows["Contract"] = ["Ten-year", "", None]
    rows["Total Charges"] = [" ", "12.5", None]
    out = predict(telco_result.model, rows)
    assert np.isfinite(out["churn_probability"]).all()


def test_explain_returns_three_reasons_with_original_column_names(telco_result, telco_raw):
    model = telco_result.model
    reasons = top_reasons(model, telco_raw.head(10), k=3)
    assert len(reasons) == 10
    for row in reasons:
        assert len(row) == 3
        for r in row:
            assert r["feature"] in model.features
            assert r["text"].startswith(f"{r['feature']} = ")
        contributions = [r["contribution"] for r in row]
        assert contributions == sorted(contributions, reverse=True)
    # Original names with spaces survive (no one-hot or renamed columns).
    assert "Tenure Months" in model.features


def test_global_importance_covers_every_feature(telco_result, telco_data):
    model = telco_result.model
    X = telco_data.df.loc[telco_data.train_idx]
    imp = global_importance(model, X)
    assert [i["feature"] for i in imp] != []
    assert set(i["feature"] for i in imp) == set(model.features)
    assert all(i["mean_abs_contribution"] >= 0 for i in imp)


def test_training_twice_gives_the_same_metrics(telco_data, telco_profile):
    X = telco_data.df.loc[telco_data.train_idx, telco_profile["suggested_features"]]
    y = telco_data.y.loc[telco_data.train_idx]
    kwargs = dict(target=telco_data.target, positive_label=telco_data.positive_label,
                  n_trials=3, timeout=None, mlflow_dir=None)
    first, second = train(X, y, **kwargs), train(X, y, **kwargs)
    assert first.metrics == second.metrics


def test_threshold_comes_from_training_data(telco_result):
    m = telco_result.metrics["lightgbm"]
    assert 0 < m["threshold_f1_oof"] < 1
    assert m["threshold_f1_oof"] == telco_result.model.threshold


def test_save_and_load_roundtrip(telco_result, telco_raw, tmp_path):
    path = save_model(telco_result.model, tmp_path / "telco")
    loaded = load_model(path)
    assert (tmp_path / "telco" / "metrics.json").exists()
    np.testing.assert_allclose(loaded.predict_proba(telco_raw.head(5)),
                               telco_result.model.predict_proba(telco_raw.head(5)))


def test_one_mlflow_run_per_training_call(iranian_data, iranian_profile, tmp_path):
    import mlflow

    X = iranian_data.df.loc[iranian_data.train_idx, iranian_profile["suggested_features"]]
    y = iranian_data.y.loc[iranian_data.train_idx]
    train(X, y, target=iranian_data.target, positive_label=iranian_data.positive_label,
          n_trials=3, timeout=None, mlflow_dir=tmp_path / "mlruns")
    mlflow.set_tracking_uri((tmp_path / "mlruns").resolve().as_uri())
    runs = mlflow.search_runs(experiment_names=["churn"])
    assert len(runs) == 1
    assert "metrics.lgbm_cv_auc" in runs.columns
