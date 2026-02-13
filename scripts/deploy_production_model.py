#!/usr/bin/env python3
# Phase 4: Download current Production model from MLflow to a path (e.g. for API to serve).
# Usage: python -m scripts.deploy_production_model [--dest models/weights/best.pt]

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from mlops.mlflow_integration import MLflowManager


def main():
    p = argparse.ArgumentParser(description="Deploy Production model from MLflow to local path")
    p.add_argument("--dest", type=Path, default=Path("models/weights/best.pt"), help="Destination path for best.pt")
    p.add_argument("--model-name", default="banana_sigatoka_detector", help="Registered model name")
    p.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    args = p.parse_args()

    uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mgr = MLflowManager(tracking_uri=uri)
    weights_path = mgr.download_model_weights(
        registered_model_name=args.model_name,
        stage="Production",
        artifact_path="model",
        dest_dir=Path("/tmp/deploy_production"),
    )
    if not weights_path or not weights_path.exists():
        print("No Production model found in MLflow.", file=sys.stderr)
        sys.exit(1)
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights_path, dest)
    print(f"Deployed Production model to {dest}")


if __name__ == "__main__":
    main()
