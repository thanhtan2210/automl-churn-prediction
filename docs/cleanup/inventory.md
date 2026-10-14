# Cleanup inventory (Phase 0)

Snapshot of `main` at `6607bcf`, taken on 2026-10-14 before any code was removed.
The full old code is preserved on branch `archive/before-cleanup`; the multi-task /
continual-learning / PCGrad research code will live on `archive/research`.

Target scope (from `CLAUDE.md`, row `automl-churn-prediction`):

> Upload a telecom subscriber list, find out who is about to cancel and why.
> 3 telecom datasets (UCI Iranian Churn #563, IBM Telco, Cell2Cell) → upload → profile with
> leakage warnings → **one LightGBM model** (Optuna, every step inside a `Pipeline`) vs a
> Logistic Regression baseline, per-customer reasons via `pred_contrib` → FastAPI with 4 endpoints
> → Next.js with 4 pages (Upload, Profile, Model, Predict), SQLite + local MLflow (`./mlruns`), 1 CI.

Decision legend: **keep** (stays mostly as is), **rewrite** (idea stays, code is rewritten in
Phase 2), **cut** (removed from `cv-refactor`, still on `archive/before-cleanup`),
**research** (removed from `cv-refactor`, kept on `archive/research`).

## 1. Modules

Line counts are `wc -l` over tracked files (binary images and lock files excluded).

### Backend API (`backend/app/api`)

| Path | Lines | Decision | Reason |
|---|---:|---|---|
| `api/v1/datasets.py` | 597 | rewrite | Becomes `POST /datasets` + `GET /datasets/{id}`; drop composite-target, sheet selection, re-evaluate-leakage routes |
| `api/v1/jobs.py` | 299 | rewrite | Becomes `POST /datasets/{id}/train`; job dedup and timeout are broken (defects 8, 10) |
| `api/v1/predict.py` | 696 | rewrite | Becomes `POST /datasets/{id}/predict` with per-row reasons; drop drift and duplicate feature-importance routes |
| `api/v1/models.py` | 347 | cut | Promote/compare/versioning is out of scope (one model per dataset) |
| `api/v1/monitoring.py` | 114 | cut | Monitoring is out of scope |
| `api/v1/auth.py` | 154 | cut | JWT/auth is listed under "Cut" |
| `api/schemas.py` | 227 | rewrite | Only the schemas the 4 endpoints need |

### Backend core (`backend/app/core`)

| Path | Lines | Decision | Reason |
|---|---:|---|---|
| `core/training/automl.py` | 930 | rewrite | Becomes one LightGBM + LR baseline, all steps in a `Pipeline`, untouched test split (defects 1–5, 14) |
| `core/training/model_router.py` | 162 | cut | Routes XGBoost/CatBoost/RF; scope is one LightGBM |
| `core/training/mlflow_utils.py` | 41 | rewrite | Keep a tiny local `./mlruns` setup; drop run deletion |
| `core/training/mtl_trainer.py` | 172 | research | Multi-task learning |
| `core/training/continual_trainer.py` | 581 | research | Continual learning |
| `core/training/pcgrad_optimizer.py` | 279 | research | PCGrad |
| `core/training/coreset_sampler.py` | 222 | research | Coreset replay for continual learning |
| `core/training/representation_analyzer.py` | 137 | research | Research diagnostics |
| `core/profiler/*` (9 files) | 895 | rewrite | Keep column typing (`layer1_stats`, `column_profile`); replace 0.95-correlation leakage rule with single-feature AUC + missingness checks (defect 7); drop `layer3_llm` and `target_synthesizer` (composite target) |
| `core/pipeline/*` (9 files) | 322 | rewrite | `builder.py` + `registry.py` are the seed of the new `Pipeline`; 5 transform files are empty |
| `core/ingestion/parsers.py` | 117 | keep | CSV/Excel parsing; drop ODS support |
| `core/etl/*` (7 files) | 1,592 | cut | Telco-specific batch ETL with lineage/partitioning/observability; not on the upload → train → predict path |
| `core/serving/model_loader.py` | 127 | rewrite | Simple joblib/MLflow load with a cache that returns one type, not a tuple (defect 11) |
| `core/serving/drift_detector.py` | 200 | cut | Drift is listed under "Cut" |
| `core/serving/retrain_loop.py` | 139 | cut | Retrain loop is listed under "Cut" |
| `core/serving/ab_service.py` | 73 | cut | A/B is listed under "Cut" |
| `core/services/exposure_store.py` | 59 | cut | A/B exposure logging |
| `core/storage.py` | 167 | rewrite | Local filesystem only; drop boto3/R2/MinIO |
| `core/config.py` | 116 | rewrite | Only local paths, SQLite URL, MLflow `file:` URI |
| `core/dependencies.py`, `core/security.py` | 158 | cut | Auth |
| `core/limiter.py` | 16 | cut | `slowapi` rate limit is listed under "Cut" (and not in `requirements.txt`, defect 12) |
| `core/logging_config.py`, `core/exceptions.py` | 56 | keep | Small, generic |
| `core/note.md` | 7 | cut | Personal note |
| `app/db/*` (13 files) | 539 | rewrite | SQLite + SQLAlchemy `create_all`; Alembic migrations are listed under "Cut" |
| `app/middleware/logging_middleware.py` | 43 | keep | Request logging, small |
| `app/main.py` | 140 | rewrite | 4 routers, no auth/slowapi/drift loop/migrations |
| `app/reports/benchmark_results.json` | 9 | cut | Latency numbers with no script/config attached in repo |

### Other top-level folders

| Path | Lines | Decision | Reason |
|---|---:|---|---|
| `backend/jobs/jobs/*` (6 files) | 588 | cut | Spark/MinIO ingest + `evaluation_job.py` (uses `Churn Score`, defect 6) |
| `frontend/` (src: 30 files) | 1,829 | rewrite | Next.js + Tailwind + shadcn/ui shell is reusable; all 5 pages are "Coming soon" (defect 13); routes become Upload, Profile, Model, Predict |
| `analytics/` (Streamlit) | 75 | cut | Streamlit is listed under "Cut"; shows mock numbers (defect 13) |
| `infra/` (Dockerfiles, entrypoint) | 318 | cut | Docker is listed under "Cut" |
| `docker-compose.yml` | 151 | cut | Docker |
| `alembic.ini` | 38 | cut | Alembic |
| `launcher.py` | 152 | cut | Starts the old multi-service stack |
| `Makefile` | 33 | rewrite | Optional; at most install/test/run |
| `notebooks/` | 3,212 | keep (untouched) | Original notebooks are not edited (CLAUDE.md §3.5) |
| `package.json`, `package-lock.json` (root) | 7,480 | cut | Only installs `@google/gemini-cli`; unrelated to the app |
| `.env.example` | 48 | rewrite | Only the few variables the new app reads |
| `.gitignore` | 84 | rewrite | Contains unresolved merge-conflict markers (lines 25–31, 60–81) |
| `data/dataset/catalog.yaml` | — | rewrite | Lists 8+ multi-industry datasets whose files are not in the repo; becomes the 3 telecom datasets |
| `data/parquet/*`, `data/processed/*`, `data/raw/cleaned_telco.csv` | — | cut | Outputs of the old ETL; raw `Telco_customer_churn.xlsx` is the source |

### Scripts (`scripts/*`)

| Path | Lines | Decision | Reason |
|---|---:|---|---|
| `scripts/ab_testing/*` (9 files) | 411 | cut | A/B; several scripts generate synthetic outcomes |
| `scripts/benchmark/benchmark_predict.py` | 184 | cut | Latency benchmark against the old stack |
| `scripts/ci/check_ci.ps1` | 96 | cut | Helper for the old CI |
| `scripts/experiments/test_entropy_sensitivity.py` | 44 | research | Paper experiment |
| `scripts/run_ablation_study.py` | 300 | research | MTL/continual ablation (imports torch-side code) |
| `scripts/test_datasets_e2e.py` | 406 | cut | Multi-industry benchmark against the running API; datasets not in repo, writes no result file |
| `scripts/run_e2e.ps1` | 28 | cut | Wrapper for the above |
| `scripts/validation/*` (7 files) | 363 | cut | Checks Postgres/MinIO/MLflow server |
| `scripts/audit/` (new) | — | keep | Phase 1 leakage audit |

### Tests (`tests/*`)

| Path | Lines | Decision | Reason |
|---|---:|---|---|
| `tests/phase3/*` (23 files) | 3,252 | research (most) / cut | `test_h1..h4_*`, `test_causal_*`, `test_spectral_*`, `test_dynamic_*`, `test_coreset_*`, `test_continual_learning`, `test_paper_empirical_defense`, `test_appendix_visualizer`, `test_extra_diversity_experiments`, `test_final_polish_artifacts` → research; auth, A/B, drift, monitoring, rate limit, versioning, storage → cut |
| `tests/phase2/test_mtl_trainer.py` | 239 | research | MTL |
| `tests/phase2/test_pipeline.py` | 280 | rewrite | Pipeline tests are needed, but for the new pipeline |
| `tests/phase1/*` (3 files) | 766 | cut | Composite-target synthesizer |
| `tests/integration/*` (9 files) | 1,740 | rewrite | Replace with tests for the 4 endpoints using `TestClient`, SQLite in tmp, no network |
| `tests/unit/profiler/*`, `tests/profiler/*` | 306 | rewrite | Keep the idea (target detection, column typing); add leakage-check tests |
| `tests/unit/pipeline/*` | 320 | cut | Old ETL |
| `tests/unit/training/test_automl_profiler.py` | 71 | rewrite | New training module |
| `tests/foundation/*`, `tests/e2e/*` | 417 | cut | Need Postgres/MinIO/MLflow server or a running API |
| `tests/fixtures/*.csv` | 45 | keep | Small fixtures, `target_leakage.csv` is useful for the new profiler |
| `tests/conftest.py` | 42 | rewrite | New fixtures |

### Docs (`docs/*`)

| Path | Lines | Decision | Reason |
|---|---:|---|---|
| `docs/architecture/*` (11 files) | 1,360 | cut | Describes the old multi-service platform |
| `docs/diagrams/*`, `docs/img/*` | 1,116 | cut | Old architecture diagrams; new README gets one one-line flow + a screenshot |
| `docs/guides/*` (10 files) | 897 | cut | Airflow, Streamlit, deployment, data cleaning for the old stack |
| `docs/deployment/*` | 81 | cut | No deployment in scope |
| `docs/phase2/*`, `docs/phase3/*`, `docs/ablation_study_analysis.md` | 1,411 | research | Continual learning / ablation write-ups |
| `docs/notes/*`, `docs/roadmap/*`, `docs/PROJECT_CONTEXT.md` | 1,102 | cut | Planning notes |
| `docs/reports/*` (8 files) | 7,639 | cut | A/B outputs (partly simulated) and `offline_evaluation.*` (ROI from hard-coded assumptions, defect 6) |
| `docs/ETL_PIPELINE.md`, `docs/getting_started.md`, `docs/README.md` | 150 | cut | Old stack |
| `docs/cleanup/` (new) | — | keep | This inventory + leakage audit |

### CI (`.github/workflows/*`)

| Path | Lines | Decision | Reason |
|---|---:|---|---|
| `ci.yml` | 62 | rewrite | Becomes the single CI: Python 3.11, `pytest -q`, frontend lint/build; drops Java 11 setup |
| `ab_notebook_ci.yml` | 96 | cut | A/B notebook; uses Python 3.10 |
| `api-image.yml` | 51 | cut | Docker image build |
| `deploy.yml` | 62 | cut | No deployment in scope |

## 2. Endpoints

All routers are mounted under `/api/v1` (`backend/app/main.py:107-130`). 30 route registrations today:

| # | Method + path | File | Fate |
|---:|---|---|---|
| 1 | `POST /auth/register` | `auth.py:51` | cut |
| 2 | `POST /auth/login` | `auth.py:75` | cut |
| 3 | `POST /auth/refresh` | `auth.py:97` | cut |
| 4 | `GET /auth/me` | `auth.py:127` | cut |
| 5 | `POST /auth/change-password` | `auth.py:141` | cut |
| 6 | `POST /datasets/upload` | `datasets.py:52` | → `POST /datasets` |
| 7 | `POST /datasets/{id}/profile` | `datasets.py:157` | merged into upload (profile runs on upload) |
| 8 | `GET /datasets/{id}/profile` | `datasets.py:211` | → `GET /datasets/{id}` |
| 9 | `POST /datasets/{id}/confirm-composite` | `datasets.py:261` | cut |
| 10 | `POST /datasets/{id}/select-sheet` | `datasets.py:423` | cut (first sheet, or sheet as upload parameter) |
| 11 | `POST /datasets/{id}/re-evaluate-leakage` | `datasets.py:489` | cut (user excludes columns in the train request) |
| 12 | `POST /jobs/datasets/{id}/train` | `jobs.py:200` | → `POST /datasets/{id}/train` |
| 13 | `GET /jobs/{job_id}/status` | `jobs.py:285` | merged into `GET /datasets/{id}` (training status + metrics) |
| 14 | `GET /models/{id}` | `models.py:21` | cut |
| 15 | `POST /models/{id}/promote` | `models.py:55` | cut |
| 16 | `GET /models/{id}/compare` | `models.py:104` | cut |
| 17 | `GET /models/{id}/feature-importance` | `models.py:159` | cut (global importance returned by train) |
| 18 | `GET /monitoring/health` | `monitoring.py:35` | cut |
| 19 | `GET /monitoring/metrics` | `monitoring.py:57` | cut |
| 20 | `GET /monitoring/jobs/summary` | `monitoring.py:89` | cut |
| 21 | `POST /predict` | `predict.py:29` | → `POST /datasets/{id}/predict` |
| 22 | `POST /predict/batch` | `predict.py:230` | merged into `POST /datasets/{id}/predict` (file upload) |
| 23 | `GET /predict/datasets/{id}/feature-importance` | `predict.py:454` | cut (duplicate of #17) |
| 24 | `POST /predict/datasets/{id}/drift` | `predict.py:521` | cut |
| 25 | `POST /predict/{id}/drift` | `predict.py:522` | cut (same handler registered twice) |
| 26 | `POST /ab/assign` | `ab_service.py:39` | cut |
| 27 | `POST /ab/log_exposure` | `ab_service.py:49` | cut |
| 28 | `GET /ab/health` | `ab_service.py:71` | cut |
| 29 | `GET /` | `main.py:133` | cut |
| 30 | `GET /health` | `main.py:138` | keep as a plain health check (not counted in the 4) |

Target endpoints:

| Method + path | Purpose |
|---|---|
| `POST /datasets` | Upload CSV/XLSX, store it, run the profiler (types, target guess, leakage warnings) |
| `GET /datasets/{id}` | Profile + latest training status and metrics |
| `POST /datasets/{id}/train` | Train LightGBM (Optuna, CV on train) + LR baseline, evaluate once on held-out test |
| `POST /datasets/{id}/predict` | Score a file of customers; return probability + top reasons per row (`pred_contrib`) |

## 3. Dependencies

### `requirements.txt`

| Package | Decision | Reason |
|---|---|---|
| pandas, numpy, scikit-learn, joblib | keep | Core |
| scipy | keep (transitive) | Pulled in by scikit-learn; not imported directly |
| lightgbm | keep | The one model |
| optuna | keep | Tuning (with a sane budget, defect 14) |
| mlflow | keep | Local `./mlruns` tracking |
| fastapi, uvicorn, pydantic, python-multipart | keep | API + file upload |
| sqlalchemy | keep | SQLite |
| openpyxl | keep | XLSX upload (IBM Telco ships as `.xlsx`) |
| httpx | keep (dev) | FastAPI `TestClient` |
| imbalanced-learn | cut | No SMOTE; only in `requirements-audit.txt` for the Phase 1 audit |
| xgboost, catboost | cut | One model |
| shap | cut | Reasons come from LightGBM `pred_contrib` |
| statsmodels, pandera, pyarrow | cut | Not needed on the main path |
| pyspark | cut | Spark is listed under "Cut" |
| gunicorn | cut | No deployment |
| requests | cut | `httpx` covers tests |
| email-validator, python-jose, passlib, bcrypt | cut | Auth |
| alembic, psycopg2-binary | cut | Alembic/Postgres listed under "Cut" |
| boto3, s3fs, fsspec | cut | R2/MinIO listed under "Cut" |
| streamlit, matplotlib, seaborn | cut | Streamlit listed under "Cut" |
| odfpy | cut | ODS upload not needed |
| *(missing)* slowapi, plotly | — | Imported but never declared (defect 12); both go away with their modules |

`requirements-local.txt` (pytest, pytest-cov, hypothesis, ruff, flake8, mypy, black, isort): rewrite to
`pytest`, `httpx`, `ruff`.

### `frontend/package.json`

| Package | Decision | Reason |
|---|---|---|
| next, react, react-dom | keep | Framework |
| @tanstack/react-query | keep | In scope |
| recharts | keep | In scope |
| tailwindcss, @tailwindcss/postcss, tw-animate-css | keep | Tailwind |
| class-variance-authority, clsx, tailwind-merge, lucide-react | keep | shadcn/ui runtime |
| @base-ui/react, @radix-ui/react-slot | keep | Primitives used by the generated shadcn components (`style: base-nova`) |
| sonner | keep (optional) | Toasts for upload/train errors; drop if unused after the rewrite |
| shadcn | move to dev / remove | CLI only; can be run with `npx` |
| @tanstack/react-query-devtools | cut | Dev nicety |
| next-intl | cut | Listed under "Cut" (also `messages/`, `middleware.ts`, `[locale]` routes, `i18n.ts`) |
| zustand | cut | Listed under "Cut" (`store/useAppStore.ts`) |
| axios | cut | Listed under "Cut"; use `fetch` |
| react-hook-form, @hookform/resolvers, zod | cut | Listed under "Cut" (also `components/ui/form.tsx`) |
| next-themes | cut | Dark-mode toggle not needed |
| eslint, eslint-config-next, typescript, @types/* | keep (dev) | Lint/build in CI |
| prettier, prettier-plugin-tailwindcss | keep (dev, optional) | Formatting |

## 4. Known defects

Each item was re-checked against the code at `6607bcf`. "Confirmed" means the code does what
the item says; items checked by reading only (not executed) are marked as such.

1. **SMOTE/SMOTENC before cross-validation — confirmed.**
   `backend/app/core/training/automl.py:643` oversamples the whole training part
   (`X, y = _try_smart_oversample(X, y, churn_rate)`), and `automl.py:677` runs
   `cross_val_score` on the oversampled `X, y`, so synthetic neighbours of a validation-fold row
   sit in the training folds. Oversampling fires whenever the positive rate is < 0.30 or > 0.70
   (`automl.py:262`), which includes IBM Telco (26.5%). The README (line 19) claims the opposite
   ("`fit` strictly on training folds"). Measured effect: see `docs/cleanup/leakage_audit.md`.
2. **Reported score is optimistically biased; no untouched test set — confirmed.**
   `automl.py:810-813` sets `best_overall_score = max(best_overall_score, post_calibration_auc)`,
   where `post_calibration_auc` is the AUC on `X_val` after calibrating on the same `X_val`
   (`automl.py:803-808`). The decision threshold is also picked on `X_val`
   (`automl.py:833-857`). `X_val` is the only hold-out (`automl.py:610`); there is no test set
   that was not used for a decision. Side note: `CalibratedClassifierCV(..., cv="prefit")`
   (`automl.py:803-805`) is deprecated in scikit-learn ≥ 1.6.
3. **Generic feature selection uses the target on all rows before the split — confirmed.**
   `_generic_feature_engineer` ranks numeric columns by `|corr(column, target)|` over the whole
   dataframe (`automl.py:217-224`), called at `automl.py:367`, before `train_test_split` at
   `automl.py:610`. It only runs when the target is already numeric (`automl.py:213-215`).
   On IBM Telco the effect is nil in practice because exactly 5 numeric columns exist and
   `max_pairs=5`, so all are selected regardless (see audit).
4. **Feature engineering outside the `Pipeline` breaks prediction on raw files — confirmed by
   reading, not executed.** `_auto_feature_engineer` and `_generic_feature_engineer`
   (`automl.py:364-367`) add `charge_per_tenure`, `gen_ratio_*`, `gen_log1p_*` to the dataframe and
   to `confirmed_profiles`; `build_pipeline` then selects those columns by name. `predict`
   (`backend/app/api/v1/predict.py:141`) and batch predict call `model.predict_proba(input_df)`
   on the uploaded columns without re-running either function (no other caller exists:
   `grep feature_engineer backend/` only matches `automl.py`), so the `ColumnTransformer` will
   raise on missing `gen_*` columns. Also, `_auto_feature_engineer` matches Kaggle column names
   (`MonthlyCharges`, `tenure`), so it never fires on the IBM `.xlsx` (`Monthly Charges`,
   `Tenure Months`).
5. **Positive class = `sorted(unique)[1]` — confirmed.** `automl.py:604-606` (and the MTL path at
   `automl.py:514-518`). Works for `No/Yes` and `0/1` by accident; for
   `"Attrited Customer"/"Existing Customer"` it picks `"Existing Customer"` as the positive class.
6. **`evaluation_job.py` uses `Churn Score` as a feature; ROI from hard-coded assumptions —
   confirmed.** `backend/jobs/jobs/evaluation_job.py:29-36` puts `Churn Score` and `CLTV` in the
   feature list. `evaluation_job.py:105-113` hard-codes a 30% outreach success rate, $5 per
   contact and the mean `CLTV` as the value of a saved customer. The committed output
   `docs/reports/offline_evaluation.json` shows AUC 0.9717, precision@k 1.0 and
   `estimated_roi` 263.09 — the "263x".
7. **Profiler flags leakage only at correlation > 0.95 — confirmed.**
   `backend/app/core/profiler/orchestrator.py:289` (`check_leakage`), with the same 0.95 rule at
   `orchestrator.py:195` and a 0.98 rule at `orchestrator.py:221` and `:253`. A correlation rule
   cannot see leakage through missingness (IBM `Churn Reason` is empty exactly for non-churners)
   or through a non-linear score; see the single-feature AUC tables in the audit.
8. **Job dedup returns an old job regardless of the new request — confirmed.**
   `backend/app/api/v1/jobs.py:238-256`: the "idempotency check" matches only
   `dataset_id`, `target_column`, `status == "completed"`, `roc_auc > 0.65` and
   `prior_model_uri`. A new request with different confirmed columns (e.g. after removing a
   leaking column) gets the old model back.
9. **Predict picks the job with the highest `roc_auc` — confirmed.**
   `backend/app/api/v1/predict.py:46-56` falls back to `order_by(TrainingJob.roc_auc.desc())` when
   no job is active (same at `predict.py:255`, `predict.py:460`, `models.py:185-191`). Combined
   with defects 2 and 8, the model with the most leakage wins.
10. **Timeout does not stop training — confirmed.** `jobs.py:150-161` wraps
    `asyncio.to_thread(run_training_sync, ...)` in `asyncio.wait_for`. On timeout the coroutine is
    cancelled but the worker thread keeps running; when it finishes, `run_training_sync`
    (`jobs.py:93-97`) writes `status = "completed"` over the `"failed"` written by the timeout
    handler.
11. **Feature importance is always empty — confirmed; "duplicate route in `models.py`" partly
    confirmed.** `predict.py:471` assigns `model = model_cache.get_model(...)`, which returns a
    tuple `(model, model_type)` or `(None, None)` (`backend/app/core/serving/model_loader.py:79-89`);
    a tuple has neither `.steps` nor `feature_importances_`, so the route returns
    `{"feature_importances": []}` (`predict.py:507-508`). Even with the tuple unpacked, the saved
    model is usually a `CalibratedClassifierCV` (defect 2), which has no `.steps`. The same feature
    is exposed twice: `GET /models/{id}/feature-importance` (`models.py:159`) and
    `GET /predict/datasets/{id}/feature-importance` (`predict.py:454`); `models.py` itself
    declares it once. The drift handler is registered on two paths (`predict.py:521-522`).
12. **Undeclared dependencies; MLflow defaults to a server — confirmed.** `slowapi` is imported in
    `backend/app/main.py:90-92` and `backend/app/core/limiter.py:2-3`; `plotly` in
    `analytics/streamlit_app.py:4`. Neither is in `requirements.txt`,
    `infra/dockerfiles/api/requirements.txt` or `infra/dockerfiles/streamlit/requirements.txt`.
    `MLFLOW_TRACKING_URI` defaults to `http://localhost:5000`
    (`backend/app/core/training/mlflow_utils.py:6-7`, `backend/app/core/config.py:38`).
13. **Frontend pages are placeholders; Streamlit shows fake numbers — confirmed.** All 5 Next.js
    pages render only "… — Coming soon": `frontend/src/app/[locale]/{upload,dashboard,monitoring}/page.tsx:2`,
    `jobs/[job_id]/page.tsx:2`, `review/[dataset_id]/page.tsx:2`. `analytics/streamlit_app.py:30`
    hard-codes `"10,240"` rows (`# Mock for now`); `streamlit_app.py:40-42` plots a fixed mock
    histogram for every column.
14. **Oversized Optuna budget; per-trial MLflow runs; pruner does nothing — confirmed.**
    `automl.py:615-623`: for > 10,000 training rows `n_trials = max(OPTUNA_N_TRIALS, 150)`; the
    router adds 4 models for > 2,000 rows (`model_router.py:73-149`: XGBoost, LightGBM, CatBoost,
    RandomForest) and every trial is a 5-fold CV (`automl.py:652`, `:677`), i.e. ≥ 3,000 model fits
    before stacking. Each trial opens an MLflow run (`automl.py:689-692`), then
    `_cleanup_mlflow` keeps only the 5 most recent runs of the experiment
    (`automl.py:37-46`, `mlflow_utils.py:17-41`). `MedianPruner` (`automl.py:697`) never prunes:
    the objective never calls `trial.report`/`trial.should_prune`.
15. **README numbers have no source; old name and badges; "enterprise-grade" — confirmed.**
    `README.md:30-43` reports a "8 real-world datasets" benchmark (e.g. Iranian 99.16%,
    Bank Marketing 98.90%) — no committed script writes these numbers and the datasets are not in
    the repo (`scripts/test_datasets_e2e.py` only prints). Title and badges use
    `Scalable-Customer-Data-Platform-CDP` (`README.md:1`, `:3`, `:133`, `:166-167`);
    "enterprise-grade" at `README.md:11`; "state-of-the-art" at `README.md:43`.

Additional defects found during the inventory:

16. `.gitignore` contains unresolved merge-conflict markers (lines 25–31 and 60–81).
17. When the positive rate is < 0.25, both `class_weight="balanced"` (`automl.py:631-639`) and
    SMOTE (`automl.py:643`) are applied, double-correcting the imbalance.
18. `.github/workflows/ab_notebook_ci.yml:30` uses Python 3.10 while the other workflows use 3.11.
19. `data/dataset/catalog.yaml` points to files under `data/dataset/...` that are not tracked
    (`data/` is git-ignored); only `data/raw/Telco_customer_churn.xlsx` and old ETL outputs are
    committed.

## 5. Datasets

| Dataset | Status after Phase 1 | Notes |
|---|---|---|
| IBM Telco (Cognos Analytics 11.1.3 sample) | in repo: `data/raw/Telco_customer_churn.xlsx` | 7,043 rows, 33 columns; contains post-outcome columns `Churn Score`, `CLTV`, `Churn Reason` |
| UCI Iranian Churn (#563) | downloaded in Phase 1: `data/raw/iranian_churn/` | See `data/raw/iranian_churn/SOURCE.md` |
| Cell2Cell (Duke / Teradata CRM tournament) | **not downloaded** | See below |

### Cell2Cell: candidate sources and what to verify

Not downloaded in Phase 1. Candidate sources (none verified yet):

- Kaggle dataset `jpacse/datasets-for-churn-telecom` (`cell2celltrain.csv`, `cell2cellholdout.csv`);
  needs a Kaggle account. Commonly cited as 51,047 + 20,000 = 71,047 rows.
- The original Teradata Center for CRM at Duke University tournament files (Fuqua School of
  Business), commonly described as a calibration sample of about 40,000 customers with churn
  over-sampled to about 50%, plus validation samples; total figures of about 100,000 are also
  quoted.

To verify before using it:

- Row counts per file (40k / 71k / 100k all appear in secondary sources) and which file is which.
- Whether the holdout file has labels at all (if not, only the training file is usable).
- The churn rate per file (the calibration sample is reported to be balanced by design, which
  changes PR-AUC and any threshold).
- License / terms of use (Kaggle mirror vs. Duke original); if no license is stated, treat like
  IBM Telco: educational use only, do not redistribute large copies.
- Which columns are post-outcome (e.g. retention-call columns) — run the same single-feature AUC
  audit before training.
