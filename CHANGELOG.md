# Changelog

All notable changes to this project will be documented in this file. Format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] — 2025-02-11 (example)

### Added

- Detection API (FastAPI): `/api/v1/predict`, `/api/v1/predict/classify`, health and model info.
- Feedback: POST `/api/v1/feedback/submit`, GET `/api/v1/feedback/stats`; PostgreSQL and optional MinIO storage.
- Database CRUD: predictions, feedback, training_data, model_performance.
- MLOps: MLflow integration (`mlops/mlflow_integration.py`), `training/train_with_mlflow.py` with git SHA and pip freeze logging.
- Phase 3: `scripts/export_feedback_for_training.py`, `scripts/run_retrain.py` (export + train with optional `--base-dataset`, `--tile`, `--resume`/`--from-scratch`).
- Docker: `docker/docker-compose.yml` (API), `docker/docker-mlops-pipeline.yml` (MLflow + Postgres + MinIO).
- Docs: README, ENVIRONMENT.md, RUNBOOKS.md, MLFLOW_OPERATIONS.md, PROJECT_GAPS_AND_DOCUMENTATION.md, TODO_LIST.md, database data dictionary, migrations README, CONTRIBUTING.md, EXPLANATION_AND_RECOMMENDATION_ROADMAP.md.
- Unit tests: feedback service, storage service, database CRUD, feedback schemas, export/retrain, MLflow training.
- Integration tests: API health and classify with sample image (TestClient).

### Changed

- README: Docker Compose clarified (which file for API vs MLOps); link to ENVIRONMENT.md and Phase 3 programmatic retrain.

### Fixed

- (Add fixes here as needed.)

---

## [Unreleased]

- (Add upcoming changes here.)

---

[0.1.0]: https://github.com/your-org/ai-banana-earlystage/releases/tag/v0.1.0
