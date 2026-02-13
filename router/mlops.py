# Phase 4: MLOps API – rollback, status (monitoring).

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/mlops", tags=["mlops"])


class RollbackRequest(BaseModel):
    version: int
    model_name: str = "banana_sigatoka_detector"


class RollbackResponse(BaseModel):
    ok: bool
    message: str
    version: int


@router.post("/rollback", response_model=RollbackResponse)
async def rollback_model(body: RollbackRequest):
    """
    Promote a given model version to Production (rollback).
    Current Production is moved to Archived.
    In production, protect this endpoint (e.g. API key or admin-only).
    """
    try:
        from mlops.mlflow_integration import MLflowManager
        import os
        uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mgr = MLflowManager(tracking_uri=uri)
        mgr.rollback_to_version(
            registered_model_name=body.model_name,
            version=body.version,
        )
        return RollbackResponse(
            ok=True,
            message=f"Version {body.version} is now Production. Redeploy API to serve it.",
            version=body.version,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def mlops_status(
    model_name: str = Query("banana_sigatoka_detector", description="Registered model name"),
):
    """
    Phase 4: Monitoring – current Production and Staging versions, last run info.
    """
    try:
        from mlops.mlflow_integration import MLflowManager
        import os
        uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mgr = MLflowManager(tracking_uri=uri)
        prod = mgr.get_best_model(registered_model_name=model_name, stage="Production")
        staging = mgr.get_best_model(registered_model_name=model_name, stage="Staging")
        return {
            "model_name": model_name,
            "production": {
                "version": prod.version if prod else None,
                "run_id": prod.run_id if prod else None,
            } if prod else None,
            "staging": {
                "version": staging.version if staging else None,
                "run_id": staging.run_id if staging else None,
            } if staging else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def prometheus_metrics(
    model_name: str = Query("banana_sigatoka_detector", description="Registered model name"),
):
    """
    Phase 4: Prometheus-style metrics (text format).
    Scrape with: metrics_path: /api/v1/mlops/metrics
    """
    try:
        from fastapi.responses import PlainTextResponse
        from mlops.mlflow_integration import MLflowManager
        import os
        uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mgr = MLflowManager(tracking_uri=uri)
        prod = mgr.get_best_model(registered_model_name=model_name, stage="Production")
        staging = mgr.get_best_model(registered_model_name=model_name, stage="Staging")
        lines = [
            "# HELP mlops_model_version Model version in stage",
            "# TYPE mlops_model_version gauge",
        ]
        if prod:
            lines.append(f'mlops_model_version{{model="{model_name}",stage="Production"}} {prod.version}')
        else:
            lines.append(f'mlops_model_version{{model="{model_name}",stage="Production"}} 0')
        if staging:
            lines.append(f'mlops_model_version{{model="{model_name}",stage="Staging"}} {staging.version}')
        else:
            lines.append(f'mlops_model_version{{model="{model_name}",stage="Staging"}} 0')
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
