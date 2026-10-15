"""FastAPI app: upload a dataset, profile it, train a churn model, score new customers.

Run from the repo root:
    python -m uvicorn src.api.main:app --port 8000
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.churn.data import DATASETS, read_table
from src.churn.predict import MissingColumnsError
from src.churn.profile import profile_dataset

from .config import Settings
from .db import Store
from .service import (
    InputError,
    Trainer,
    check_features,
    load_prepared,
    same_target,
    score_file,
    suggested_target,
    target_candidates,
)

ALLOWED_SUFFIXES = (".csv", ".xlsx")
SAMPLES = {
    "telco": {"name": "IBM Telco Customer Churn", "source": "IBM Cognos Analytics sample (no published license)"},
    "iranian": {"name": "UCI Iranian Churn", "source": "UCI ML Repository #563 (CC BY 4.0)"},
}


class TrainRequest(BaseModel):
    target: str
    positive_label: str | None = None
    features: list[str]


def _safe_name(filename: str) -> str:
    name = Path(filename or "upload").name
    return "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in name).strip() or "upload"


def _read_upload(upload: UploadFile, dest_dir: Path, max_mb: int) -> Path:
    """Streams the upload to dest_dir, refusing files above max_mb or with an unknown extension."""
    name = _safe_name(upload.filename)
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type. Upload a {' or '.join(ALLOWED_SUFFIXES)} file.")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    limit, size = max_mb * 1024 * 1024, 0
    with open(dest, "wb") as out:
        while chunk := upload.file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                out.close()
                shutil.rmtree(dest_dir, ignore_errors=True)
                raise HTTPException(413, f"The file is larger than {max_mb} MB.")
            out.write(chunk)
    return dest


def _dataset_path(settings: Settings, row: dict) -> Path:
    return settings.uploads_dir / row["id"] / row["filename"]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    store = Store(settings.db_path)
    trainer = Trainer(store, settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.fail_interrupted_training()
        yield
        trainer.shutdown()

    app = FastAPI(title="Churn prediction API", lifespan=lifespan)
    app.state.settings, app.state.store, app.state.trainer = settings, store, trainer
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    def get_row(dataset_id: str) -> dict:
        row = store.get_dataset(dataset_id)
        if row is None:
            raise HTTPException(404, "Dataset not found.")
        return row

    @app.post("/api/datasets", status_code=201)
    def upload_dataset(file: Annotated[UploadFile | None, File()] = None, sample: Annotated[str | None, Form()] = None):
        """Store a CSV/XLSX upload (or copy a bundled sample) and suggest target columns. No profiling yet."""
        if (file is None) == (sample is None):
            raise HTTPException(400, "Send either a file or sample=telco|iranian.")
        dataset_id = uuid.uuid4().hex[:12]
        dest_dir = settings.uploads_dir / dataset_id
        if sample is not None:
            if sample not in SAMPLES:
                raise HTTPException(400, f"Unknown sample '{sample}'. Choose one of: {', '.join(SAMPLES)}.")
            src = DATASETS[sample].path
            dest_dir.mkdir(parents=True, exist_ok=True)
            path = Path(shutil.copy(src, dest_dir / src.name))
            source = "sample"
        else:
            path = _read_upload(file, dest_dir, settings.max_upload_mb)
            source = "upload"
        try:
            df = read_table(path)
        except Exception as e:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise HTTPException(400, f"Could not read the file: {e}") from e
        if len(df) < settings.min_rows:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise HTTPException(400, f"The file has {len(df)} rows; at least {settings.min_rows} are needed.")
        store.add_dataset(dataset_id, path.name, source, len(df), list(df.columns))
        candidates = target_candidates(df)
        return {
            "id": dataset_id,
            "filename": path.name,
            "source": source,
            "n_rows": len(df),
            "columns": list(df.columns),
            "target_candidates": candidates,
            "suggested_target": suggested_target(candidates),
        }

    @app.get("/api/datasets/{dataset_id}")
    def get_dataset(
        dataset_id: str, target: Annotated[str | None, Query()] = None,
        positive_label: Annotated[str | None, Query()] = None,
    ):
        """Dataset status; with ?target= the profile (training split only, cached per target); model metrics."""
        row = get_row(dataset_id)
        path = _dataset_path(settings, row)
        if target is not None and not same_target(row, target, positive_label):
            try:
                data = load_prepared(path, target, positive_label, settings)
            except InputError as e:
                raise HTTPException(400, str(e)) from e
            prof = profile_dataset(data)
            status = row["status"] if row["status"] in ("training", "trained") else "profiled"
            store.update_dataset(dataset_id, target=data.target, positive_label=data.positive_label,
                                 profile=prof, status=status, error=None if status == "profiled" else row["error"])
            row = get_row(dataset_id)

        candidates = target_candidates(read_table(path)) if row["profile"] is None else None
        model = store.get_model(dataset_id)
        elapsed = None
        if row["status"] == "training":
            started = datetime.fromisoformat(row["updated_at"])
            elapsed = round((datetime.now(timezone.utc) - started).total_seconds())
        return {
            "id": row["id"],
            "filename": row["filename"],
            "source": row["source"],
            "uploaded_at": row["uploaded_at"],
            "n_rows": row["n_rows"],
            "columns": row["columns"],
            "status": row["status"],
            "error": row["error"],
            "training_elapsed_seconds": elapsed,
            "target": row["target"],
            "positive_label": row["positive_label"],
            "profile": row["profile"],
            "target_candidates": candidates,
            "suggested_target": suggested_target(candidates) if candidates is not None else row["target"],
            "model": None if model is None else {
                "created_at": model["created_at"],
                "mlflow_run_id": model["mlflow_run_id"],
                "features": model["features"],
                "metrics": model["metrics"],
            },
        }

    @app.post("/api/datasets/{dataset_id}/train", status_code=202)
    def train_dataset(dataset_id: str, req: TrainRequest):
        """Start training in the background worker; poll GET /api/datasets/{id} for the result."""
        row = get_row(dataset_id)
        if trainer.busy is not None:
            raise HTTPException(409, "A model is already training. Wait for it to finish.")
        path = _dataset_path(settings, row)
        try:
            data = load_prepared(path, req.target, req.positive_label, settings)
            check_features(data, req.features)
        except InputError as e:
            raise HTTPException(400, str(e)) from e
        if not trainer.start(dataset_id, path, req.target, req.positive_label, req.features):
            raise HTTPException(409, "A model is already training. Wait for it to finish.")
        return {"id": dataset_id, "status": "training", "n_trials": settings.n_trials}

    @app.post("/api/datasets/{dataset_id}/predict")
    def predict_file(dataset_id: str, file: Annotated[UploadFile, File()]):
        """Score a file of customers: probability, flag at the model's threshold, up to 3 reasons each."""
        row = get_row(dataset_id)
        model_row = store.get_model(dataset_id)
        if model_row is None:
            raise HTTPException(400, "This dataset has no trained model yet.")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = _read_upload(file, Path(tmp), settings.max_upload_mb)
            try:
                df = read_table(path)
            except Exception as e:
                raise HTTPException(400, f"Could not read the file: {e}") from e
        if df.empty:
            raise HTTPException(400, "The file has no rows.")
        model = trainer.model(model_row["path"])
        id_columns = [c["name"] for c in (row["profile"] or {}).get("columns", []) if c["role"] == "id"]
        try:
            return score_file(model, df, model_row["metrics"]["strength_cutoffs"], next(iter(id_columns), None))
        except MissingColumnsError as e:
            raise HTTPException(422, {"message": str(e), "missing_columns": e.missing}) from e

    return app


app = create_app()
