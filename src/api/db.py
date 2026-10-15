"""SQLite storage (standard-library sqlite3): one row per uploaded dataset, one latest model per dataset."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    source TEXT NOT NULL,              -- upload | sample
    uploaded_at TEXT NOT NULL,
    n_rows INTEGER NOT NULL,
    columns_json TEXT NOT NULL,
    target TEXT,
    positive_label TEXT,
    profile_json TEXT,
    status TEXT NOT NULL,              -- uploaded | profiled | training | trained | failed
    error TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    dataset_id TEXT PRIMARY KEY REFERENCES datasets(id),
    created_at TEXT NOT NULL,
    path TEXT NOT NULL,
    mlflow_run_id TEXT,
    features_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL
);
"""

INTERRUPTED = "Training was interrupted because the API stopped. Start training again."


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock, self._conn:
            return self._conn.execute(sql, params)

    def add_dataset(self, id: str, filename: str, source: str, n_rows: int, columns: list[str]) -> None:
        ts = now()
        self._execute(
            "INSERT INTO datasets (id, filename, source, uploaded_at, n_rows, columns_json, status, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'uploaded', ?)",
            (id, filename, source, ts, n_rows, json.dumps(columns), ts),
        )

    def get_dataset(self, id: str) -> dict | None:
        row = self._execute("SELECT * FROM datasets WHERE id = ?", (id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["columns"] = json.loads(out.pop("columns_json"))
        raw_profile = out.pop("profile_json")
        out["profile"] = json.loads(raw_profile) if raw_profile else None
        return out

    def update_dataset(self, id: str, **fields) -> None:
        if "profile" in fields:
            fields["profile_json"] = json.dumps(fields.pop("profile"))
        fields["updated_at"] = now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._execute(f"UPDATE datasets SET {cols} WHERE id = ?", (*fields.values(), id))

    def save_model(self, dataset_id: str, path: str, mlflow_run_id: str | None, features: list[str],
                   metrics: dict) -> None:
        self._execute(
            "INSERT OR REPLACE INTO models (dataset_id, created_at, path, mlflow_run_id, features_json, metrics_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (dataset_id, now(), path, mlflow_run_id, json.dumps(features), json.dumps(metrics)),
        )

    def get_model(self, dataset_id: str) -> dict | None:
        row = self._execute("SELECT * FROM models WHERE dataset_id = ?", (dataset_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["features"] = json.loads(out.pop("features_json"))
        out["metrics"] = json.loads(out.pop("metrics_json"))
        return out

    def fail_interrupted_training(self) -> int:
        """Called at startup: a dataset still marked 'training' was cut off by a restart."""
        cur = self._execute(
            "UPDATE datasets SET status = 'failed', error = ?, updated_at = ? WHERE status = 'training'",
            (INTERRUPTED, now()),
        )
        return cur.rowcount
