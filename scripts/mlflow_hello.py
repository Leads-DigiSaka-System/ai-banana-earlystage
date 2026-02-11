#!/usr/bin/env python3
"""
Phase 1 - Step 1: Hello MLflow
Run this FIRST to see how MLflow works (no training, no GPU).

Prereqs:
  1. docker compose -f docker/docker-mlops-pipeline.yml up -d
  2. pip install mlflow boto3
  3. Create bucket "mlflow-artifacts" in MinIO (http://localhost:9011)

Run from project root:
  python scripts/mlflow_hello.py

Then open http://localhost:5000 and you should see 1 experiment with 1 run
containing params, metrics, and a small artifact.
"""
import os
import sys
from pathlib import Path

# Project root on path so "mlops" is found when run as scripts/mlflow_hello.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Optional: use .env for tracking URI
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from mlops.mlflow_integration import MLflowManager


def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    print(f"Using MLflow at: {tracking_uri}")

    manager = MLflowManager(
        tracking_uri=tracking_uri,
        experiment_name="banana_sigatoka_detection",
    )
    manager.start_run(
        run_name="hello_mlflow",
        tags={"phase": "1", "purpose": "test"},
    )

    try:
        # 1. Log some parameters (e.g. config)
        manager.log_parameters({
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 5,
        })

        # 2. Log some metrics (e.g. fake training steps)
        for step in range(1, 4):
            manager.log_metrics({
                "loss": 1.0 - step * 0.2,
                "accuracy": 0.5 + step * 0.15,
            }, step=step)

        # 3. Log a tiny artifact (so you see artifacts in MinIO)
        import tempfile
        tmpdir = Path(tempfile.mkdtemp())
        (tmpdir / "hello_mlflow.txt").write_text("Hello from Phase 1 MLflow test.\n")
        try:
            manager.log_artifacts(tmpdir)
        finally:
            (tmpdir / "hello_mlflow.txt").unlink(missing_ok=True)
            tmpdir.rmdir()

        print(f"Run ID: {manager.run.info.run_id}")
        print("Done. Open http://localhost:5000 -> Experiments -> banana_sigatoka_detection")
    finally:
        manager.end_run()


if __name__ == "__main__":
    main()
