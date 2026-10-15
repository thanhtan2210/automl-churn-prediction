import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.config import Settings
from src.api.db import INTERRUPTED, Store
from src.api.main import create_app

from .conftest import IRANIAN, TELCO

LEAKS = {"Churn Label", "Churn Reason", "Churn Score"}


def _settings(tmp_path) -> Settings:
    return Settings(app_dir=tmp_path / "app", n_trials=3, timeout_seconds=0, cors_origins=["http://localhost:3000"],
                    mlflow_dir=None)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(_settings(tmp_path))) as c:
        yield c


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _upload(client, path=TELCO, name="customers.csv") -> dict:
    res = client.post("/api/datasets", files={"file": (name, path.read_bytes(), "text/csv")})
    assert res.status_code == 201, res.text
    return res.json()


def test_full_flow_upload_profile_train_predict(client):
    up = _upload(client)
    assert up["n_rows"] == 800
    assert any(c["name"] == "Churn Value" for c in up["target_candidates"])
    assert up["suggested_target"] in ("Churn Label", "Churn Value")

    res = client.get(f"/api/datasets/{up['id']}", params={"target": "Churn Value"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "profiled"
    excluded = body["profile"]["suggested_excluded"]
    assert LEAKS <= set(excluded)
    features = body["profile"]["suggested_features"]

    res = client.post(f"/api/datasets/{up['id']}/train", json={"target": "Churn Value", "features": features})
    assert res.status_code == 202, res.text
    client.app.state.trainer.wait()

    body = client.get(f"/api/datasets/{up['id']}").json()
    assert body["status"] == "trained", body["error"]
    metrics = body["model"]["metrics"]
    assert set(metrics["test"]) >= {"lightgbm", "logistic_regression", "difference_lightgbm_minus_lr"}
    for model in ("lightgbm", "logistic_regression"):
        assert 2 <= len(metrics["curves"][model]["roc"]) <= 200
        assert 2 <= len(metrics["curves"][model]["pr"]) <= 200
    assert len(metrics["global_importance_top10"]) <= 10
    assert any("optimistic" in n for n in metrics["notes"])

    new = pd.read_csv(TELCO).drop(columns=["Churn Value", "Churn Label"]).head(40)
    res = client.post(f"/api/datasets/{up['id']}/predict", files={"file": ("new.csv", _csv(new), "text/csv")})
    assert res.status_code == 200, res.text
    out = res.json()
    assert out["n_rows"] == 40
    assert out["id_column"] == "CustomerID"
    assert [p["id"] for p in out["predictions"]] == new["CustomerID"].tolist()
    for p in out["predictions"]:
        assert 0 <= p["probability"] <= 1
        assert p["flagged"] == (p["probability"] >= out["threshold"])
        assert len(p["reasons"]) <= 3
        for r in p["reasons"]:
            assert r["contribution"] > 0
            assert r["strength"] in ("strong", "moderate", "slight")
            assert r["feature"] in features


def test_predict_with_a_missing_feature_column_returns_422(client):
    up = _upload(client)
    profile = client.get(f"/api/datasets/{up['id']}", params={"target": "Churn Value"}).json()["profile"]
    client.post(f"/api/datasets/{up['id']}/train",
                json={"target": "Churn Value", "features": profile["suggested_features"]})
    client.app.state.trainer.wait()

    new = pd.read_csv(TELCO).drop(columns=["Contract"]).head(5)
    res = client.post(f"/api/datasets/{up['id']}/predict", files={"file": ("new.csv", _csv(new), "text/csv")})
    assert res.status_code == 422
    assert res.json()["detail"]["missing_columns"] == ["Contract"]
    assert "Contract" in res.json()["detail"]["message"]


def test_train_while_training_returns_409(client):
    up = _upload(client)
    features = ["Tenure Months", "Contract", "Monthly Charges"]
    first = client.post(f"/api/datasets/{up['id']}/train", json={"target": "Churn Value", "features": features})
    assert first.status_code == 202
    second = client.post(f"/api/datasets/{up['id']}/train", json={"target": "Churn Value", "features": features})
    assert second.status_code == 409
    client.app.state.trainer.wait()
    assert client.get(f"/api/datasets/{up['id']}").json()["status"] == "trained"


@pytest.mark.parametrize(
    "target, message",
    [
        ("Tenure Months", "exactly 2 classes"),     # not binary
        ("Gender", "positive_label"),               # binary, but no value means churn
        ("No such column", "not found"),
    ],
)
def test_invalid_target_returns_400(client, target, message):
    up = _upload(client)
    res = client.get(f"/api/datasets/{up['id']}", params={"target": target})
    assert res.status_code == 400
    assert message in res.json()["detail"]
    res = client.post(f"/api/datasets/{up['id']}/train", json={"target": target, "features": ["Contract"]})
    assert res.status_code == 400


def test_invalid_features_return_400(client):
    up = _upload(client)
    url = f"/api/datasets/{up['id']}/train"
    assert client.post(url, json={"target": "Churn Value", "features": []}).status_code == 400
    res = client.post(url, json={"target": "Churn Value", "features": ["Contract", "Nope"]})
    assert res.status_code == 400 and "Nope" in res.json()["detail"]
    res = client.post(url, json={"target": "Churn Value", "features": ["Churn Value"]})
    assert res.status_code == 400


def test_too_few_rows_and_bad_files_return_400(client):
    small = pd.read_csv(TELCO).head(150)
    res = client.post("/api/datasets", files={"file": ("small.csv", _csv(small), "text/csv")})
    assert res.status_code == 400 and "200" in res.json()["detail"]
    res = client.post("/api/datasets", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400
    assert client.post("/api/datasets").status_code == 400


def test_minority_class_too_small_returns_400(client):
    df = pd.read_csv(IRANIAN)
    df = pd.concat([df[df["Churn"] == 0], df[df["Churn"] == 1].head(20)])
    up = client.post("/api/datasets", files={"file": ("few.csv", _csv(df), "text/csv")}).json()
    res = client.get(f"/api/datasets/{up['id']}", params={"target": "Churn"})
    assert res.status_code == 400 and "smaller class" in res.json()["detail"]


def test_sample_iranian_upload(client):
    res = client.post("/api/datasets", data={"sample": "iranian"})
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["n_rows"] == 3150
    assert body["suggested_target"] == "Churn"
    assert "Call Failure" in body["columns"]  # whitespace normalised
    assert client.post("/api/datasets", data={"sample": "nope"}).status_code == 400


def test_unknown_dataset_returns_404(client):
    assert client.get("/api/datasets/missing").status_code == 404
    res = client.post("/api/datasets/missing/predict", files={"file": ("a.csv", io.BytesIO(b"a\n1"), "text/csv")})
    assert res.status_code == 404


def test_training_cut_off_by_a_restart_is_marked_failed(tmp_path):
    settings = _settings(tmp_path)
    store = Store(settings.db_path)
    store.add_dataset("abc", "x.csv", "upload", 300, ["a"])
    store.update_dataset("abc", status="training")
    store.close()
    with TestClient(create_app(settings)):
        pass
    row = Store(settings.db_path).get_dataset("abc")
    assert row["status"] == "failed"
    assert row["error"] == INTERRUPTED
