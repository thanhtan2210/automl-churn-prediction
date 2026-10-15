"""API settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.churn.data import ROOT


def _origins() -> list[str]:
    return [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]


def _mlflow_dir() -> Path | None:
    value = os.getenv("CHURN_MLFLOW_DIR", str(ROOT / "mlruns"))
    return Path(value) if value else None


@dataclass
class Settings:
    app_dir: Path = field(default_factory=lambda: Path(os.getenv("CHURN_APP_DIR", str(ROOT / "data" / "app"))))
    n_trials: int = field(default_factory=lambda: int(os.getenv("CHURN_N_TRIALS", "40")))
    timeout_seconds: float = field(default_factory=lambda: float(os.getenv("CHURN_TIMEOUT_SECONDS", "300")))
    cors_origins: list[str] = field(default_factory=_origins)
    mlflow_dir: Path | None = field(default_factory=_mlflow_dir)
    max_upload_mb: int = 50
    min_rows: int = 200
    min_minority: int = 50
    max_predict_rows: int = 20_000

    @property
    def db_path(self) -> Path:
        return self.app_dir / "app.db"

    @property
    def uploads_dir(self) -> Path:
        return self.app_dir / "uploads"

    @property
    def models_dir(self) -> Path:
        return self.app_dir / "models"
