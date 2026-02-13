#!/usr/bin/env python3
# Phase 4: Rollback Production model to a previous version (MLflow).
# Usage: python -m scripts.rollback_model --version 2
# Or: uv run python scripts/rollback_model.py --version 2

import argparse
import os
import sys

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlops.mlflow_integration import MLflowManager


def main():
    p = argparse.ArgumentParser(description="Rollback Production model to a given version")
    p.add_argument("--version", type=int, required=True, help="Model version to promote to Production")
    p.add_argument("--model-name", default="banana_sigatoka_detector", help="Registered model name")
    p.add_argument("--tracking-uri", default=None, help="MLflow tracking URI (default: env MLFLOW_TRACKING_URI or http://localhost:5000)")
    args = p.parse_args()

    uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mgr = MLflowManager(tracking_uri=uri)
    mgr.rollback_to_version(
        registered_model_name=args.model_name,
        version=args.version,
    )
    print(f"Rollback done: version {args.version} is now Production. Redeploy API to use new model (run deploy script or copy artifact).")


if __name__ == "__main__":
    main()
