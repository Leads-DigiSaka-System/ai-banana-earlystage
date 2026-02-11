# Project Gaps & Documentation Checklist (MLOps / ML Engineering)

**Role:** Senior MLOps Dev & Senior ML Engineer  
**Purpose:** Identify what is missing or under-documented so the team can prioritize and document.

---

## 1. Documentation Gaps

### 1.1 Root-level docs (missing or placeholder)

| Item | Status | Recommendation |
|------|--------|----------------|
| **CHANGELOG.md** | Missing | Add: version, date, added/fixed/changed (per release). Helps audits and release notes. |
| **CONTRIBUTING.md** | Missing | Add: how to run tests, branch naming, PR checklist, code style (e.g. ruff/black). |
| **LICENSE** | Placeholder in README | Add actual license file (e.g. MIT, Apache-2.0) and year/copyright. |
| **.env.example** | Present | Keep; ensure README and this doc reference it for all required vars. |

### 1.2 README accuracy

| Item | Status | Action |
|------|--------|--------|
| Project structure: `export_feedback_for_training.py` | Says "TODO: export for retraining" | Update: script is implemented; describe purpose and point to Phase 3 doc. |
| Phase 3 training | README describes notebook only | Add 1–2 sentences: programmatic retrain via `scripts/run_retrain.py` and `training/train_with_mlflow.py`; link to `docs/enhancement/implementation/PHASE3_INTEGRATION.md`. |
| Docker Compose path | README says `docker-compose up` | Note: MLOps stack uses `docker/docker-mlops-pipeline.yml`; app may use `docker/docker-compose.yml` — clarify which file for which use case. |

### 1.3 Technical documentation

| Topic | Status | Recommendation |
|-------|--------|----------------|
| **Database schema** | Tables in `database/README.md` and `init_db.sql` | Add short **data dictionary** (column meanings, constraints, relationships) in `database/README.md` or `docs/DATA_SCHEMA.md`. |
| **Model versioning** | `MODEL_VERSION` in `services/feedback_service.py` (hardcoded `v1.0`) | Document: where it comes from, how to change it per release, and how it ties to MLflow runs (e.g. tag run with same version). |
| **MLflow workflow** | PHASE1 + Phase 3 doc describe usage | Add: how to promote model (Staging → Production) using `mlflow_integration.py` (e.g. `get_latest_version`, `transition_model_version_stage`), and how to download `best.pt` from MLflow for deployment. |
| **Environment matrix** | Scattered in README, .env.example, Phase 3 | Add **docs/ENVIRONMENT.md**: table of every env var, required/optional, which component uses it (app, export, retrain, MLflow, MinIO). |
| **Migrations** | `migrations/README.md` says "TODO: alembic init" | Either: (a) add Alembic and document migration workflow, or (b) document that schema changes are done via manual SQL and `init_db.sql` for new installs. |

---

## 2. Testing Gaps

| Area | Status | Recommendation |
|------|--------|----------------|
| **Integration tests** | `tests/integration/` exists but empty (only `__init__.py`) | Add at least: (1) API test vs running app (e.g. health, predict with sample image), (2) DB test with real Postgres (or testcontainers). |
| **Test coverage** | No coverage config in repo | Add `pytest-cov` and a command (e.g. `uv run pytest tests/ -v --cov=. --cov-report=term-missing`); document in CONTRIBUTING or README. |
| **E2E / retrain flow** | No test that runs export + train (even with 1 sample) | Optional: script or pytest that runs export + `train_model_with_tracking` in a temp dir and checks artifacts; can be slow, mark as optional/smoke. |
| **Contract / API schema** | Not automated | Optional: add OpenAPI export and a test that checks response schema for key endpoints (e.g. `/predict/classify`, `/feedback/stats`). |

---

## 3. MLOps / ML Engineering Gaps

### 3.1 Reproducibility & versioning

| Item | Status | Recommendation |
|------|--------|----------------|
| **Training code version** | Not explicitly logged in MLflow | Log git commit SHA (and optionally branch) as MLflow run tag in `train_with_mlflow.py` so every run is tied to code version. |
| **Data versioning** | Export output path is ad-hoc | Document convention (e.g. `exported_feedback_YYYYMMDD` or include run id). Optional: log export path or dataset hash in MLflow when running retrain. |
| **Base dataset version** | `--base-dataset` in run_retrain | Document: what “base dataset” means (e.g. combined_yolo_dataset snapshot), where it lives, and how to version it (e.g. copy with date or tag). |
| **Python/package versions** | requirements.txt / pyproject.toml | Consider logging in MLflow: `pip freeze` or `uv pip freeze` as artifact for full reproducibility. |

### 3.2 Model lifecycle

| Item | Status | Recommendation |
|------|--------|----------------|
| **Promotion to production** | Code in `mlflow_integration.py` (get_latest_version, transition_model_version_stage) | Document in README or `docs/MLFLOW_OPERATIONS.md`: how to promote a run’s model to Production and how deployment (e.g. copying `best.pt` to `models/weights/`) should use that. |
| **Rollback** | Not documented | Add short runbook: how to revert to previous model (e.g. re-download from MLflow, replace `models/weights/best.pt`, restart API). |
| **A/B or shadow mode** | Not implemented | Leave as future; document as “Enhancement / backlog” if relevant. |

### 3.3 Observability & operations

| Item | Status | Recommendation |
|------|--------|----------------|
| **Monitoring** | ENHANCEMENT_5 describes plan; not implemented | Keep as roadmap; document “Current: no APM; planned: see ENHANCEMENT_5.” in README or ops doc. |
| **Logging** | Standard Python/FastAPI logging | Document: log format, where logs go (stdout / file), and any log level (e.g. INFO for production). |
| **Health checks** | `/health` exists | Document: expected response; optional: add dependency checks (DB, MinIO if used) to health or a separate `/ready` endpoint. |
| **Runbooks** | None | Add **docs/RUNBOOKS.md** (or one file per incident type): e.g. “API down”, “Retrain failed”, “DB connection failed”, “MinIO unreachable” — steps and who to contact. |

### 3.4 Security & compliance

| Item | Status | Recommendation |
|------|--------|----------------|
| **Secrets** | .env, not committed | Document: never commit `.env`; use env vars or secret manager in production. |
| **API auth** | Not implemented | Document as “Not implemented; recommend adding auth for production” and link to Enhancement or backlog. |
| **Data retention** | Not documented | Document: how long predictions/feedback/training_data are kept, and any deletion/archival policy (if any). |

---

## 4. Deployment & DevOps

| Item | Status | Recommendation |
|------|--------|----------------|
| **Docker Compose files** | Multiple: docker-compose.yml, docker-mlops-pipeline.yml, docker-airflow.yml, docker-minio.yml | Add **docs/DOCKER_COMPOSE_GUIDE.md**: which file to use for app-only, app+DB+MinIO, MLOps (MLflow+Postgres+MinIO), and optional Airflow. |
| **Port matrix** | Ports in .env and compose files | Document in ENVIRONMENT or DOCKER_COMPOSE_GUIDE: 8000 (API), 5433/55432 (Postgres app/MLflow), 5000 (MLflow), 9000/9011 (MinIO S3/console). |
| **CI/CD** | ENHANCEMENT_3 describes plan; not implemented | Keep as roadmap; in README or CONTRIBUTING, state “CI/CD: planned (see ENHANCEMENT_3).” |
| **Retrain scheduling** | Documented in PHASE3_INTEGRATION (cron, Task Scheduler) | ENHANCEMENT_3 is CI/CD only; add 1–2 sentence cross-reference to PHASE3 for “when to retrain” and scheduling. |

---

## 5. Data & Model Specs (for handover / audits)

| Item | Status | Recommendation |
|------|--------|----------------|
| **Class set** | In config and README (7 classes) | Single source of truth: keep in `config.CLASS_NAMES`; document “Model and API use these 7 classes only” and reference config. |
| **Input spec** | Image size, format in README | Add short **Model Card** or **docs/MODEL_SPEC.md**: input size (736), tile size (256), confidence threshold, supported formats, and expected latency (e.g. CPU/GPU). |
| **Feedback → training_data** | PHASE_CONNECTION_CHECKLIST + code | Already clear; ensure PHASE3_INTEGRATION mentions that only `source='feedback'` rows are exported (and optionally document other sources if you add them). |

---

## 6. Quick Wins (can do immediately)

1. **README:** Remove “TODO” from `export_feedback_for_training.py` and add one line: “Export feedback to YOLO dataset for retraining; see Phase 3 integration doc.”
2. **README:** In Model Training Workflow, add: “For programmatic retrain (export + MLflow training), see `scripts/run_retrain.py` and [PHASE3_INTEGRATION.md](docs/enhancement/implementation/PHASE3_INTEGRATION.md).”
3. **docs/ENVIRONMENT.md:** Create table of all env vars (from .env.example + MLflow + any other) with “Required/Optional” and “Used by.”
4. **migrations/README.md:** Replace “TODO: alembic” with either “Alembic not in use; schema changes via database/init_db.sql and manual migration” or add Alembic and one migration.
5. **database/README.md:** Add a short “Data dictionary” (table + column purpose) or link to a new docs/DATA_SCHEMA.md.

---

## 7. Priority Summary

| Priority | Focus | Docs / actions |
|----------|--------|----------------|
| **P0** | Correctness & onboarding | Fix README (export script, Phase 3 link); ENVIRONMENT.md; DATA_SCHEMA or database README update. |
| **P1** | Operations & reproducibility | RUNBOOKS.md; MLflow promotion/rollback doc; log git SHA (and optionally pip freeze) in MLflow. |
| **P2** | Quality & lifecycle | Integration tests; test coverage; CONTRIBUTING.md; CHANGELOG.md. |
| **P3** | Roadmap & polish | DOCKER_COMPOSE_GUIDE; MODEL_SPEC/Model Card; migrations decision; ENHANCEMENT_3 ↔ PHASE3 cross-link. |

---

*Document generated from a Senior MLOps / ML Engineer review. Update this file as gaps are closed.*
