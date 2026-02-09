"""
CRUD / search API for banana database tables (predictions, training_data, model_performance, feedback).
Feedback submit & stats: /api/v1/feedback (router/feedback.py). List/Get/Patch/Delete dito sa /api/v1/db/feedback.
"""
import uuid
from datetime import date, datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.connection import get_db
from database.model_performance import ModelPerformance
from database.models import Feedback, Prediction, TrainingData

router = APIRouter(prefix="/api/v1/db", tags=["database"])


def _row_to_dict(row: Any) -> dict:
    """Serialize SQLAlchemy row to JSON-safe dict (column name = key)."""
    if row is None:
        return {}
    d = {}
    for c in row.__table__.columns:
        attr_key = c.key if c.key != "metadata" else "metadata_"
        v = getattr(row, attr_key, None)
        if isinstance(v, uuid.UUID):
            v = str(v)
        elif isinstance(v, (datetime, date)):
            v = v.isoformat() if v else None
        d[c.key] = v
    return d


# --- Predictions ---

@router.get("/predictions")
async def list_predictions(
    db: Session = Depends(get_db),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user_id: Optional[str] = Query(None),
    predicted_class_name: Optional[str] = Query(None),
    model_version: Optional[str] = Query(None),
):
    """List predictions with optional filters. Search by user_id, class, model_version."""
    q = db.query(Prediction)
    if user_id:
        q = q.filter(Prediction.user_id.ilike(f"%{user_id}%"))
    if predicted_class_name:
        q = q.filter(Prediction.predicted_class_name.ilike(f"%{predicted_class_name}%"))
    if model_version:
        q = q.filter(Prediction.model_version == model_version)
    total = q.count()
    rows = q.order_by(Prediction.timestamp.desc()).offset(offset).limit(limit).all()
    return JSONResponse(content={
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [_row_to_dict(r) for r in rows],
    })


@router.get("/predictions/{prediction_id}")
async def get_prediction(
    prediction_id: str,
    db: Session = Depends(get_db),
):
    """Get one prediction by ID."""
    try:
        pid = uuid.UUID(prediction_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid prediction_id")
    row = db.query(Prediction).filter(Prediction.id == pid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return JSONResponse(content=_row_to_dict(row))


class PredictionUpdate(BaseModel):
    user_location: Optional[str] = None
    predicted_class_name: Optional[str] = None
    predicted_class_id: Optional[int] = None
    confidence: Optional[float] = None


@router.patch("/predictions/{prediction_id}")
async def update_prediction(
    prediction_id: str,
    body: PredictionUpdate,
    db: Session = Depends(get_db),
):
    """Update a prediction (partial)."""
    try:
        pid = uuid.UUID(prediction_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid prediction_id")
    row = db.query(Prediction).filter(Prediction.id == pid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")
    if body.user_location is not None:
        row.user_location = body.user_location
    if body.predicted_class_name is not None:
        row.predicted_class_name = body.predicted_class_name
    if body.predicted_class_id is not None:
        row.predicted_class_id = body.predicted_class_id
    if body.confidence is not None:
        row.confidence = body.confidence
    db.commit()
    return JSONResponse(content={"ok": True, "prediction_id": prediction_id})


@router.delete("/predictions/{prediction_id}")
async def delete_prediction(
    prediction_id: str,
    db: Session = Depends(get_db),
):
    """Delete a prediction (cascade deletes related feedback)."""
    try:
        pid = uuid.UUID(prediction_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid prediction_id")
    row = db.query(Prediction).filter(Prediction.id == pid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")
    db.delete(row)
    db.commit()
    return JSONResponse(content={"ok": True, "deleted": prediction_id})


# --- Training data ---

@router.get("/training-data")
async def list_training_data(
    db: Session = Depends(get_db),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    class_name: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    dataset_split: Optional[str] = Query(None),
):
    """List training_data with optional filters."""
    q = db.query(TrainingData)
    if class_name:
        q = q.filter(TrainingData.class_name.ilike(f"%{class_name}%"))
    if source:
        q = q.filter(TrainingData.source == source)
    if dataset_split:
        q = q.filter(TrainingData.dataset_split == dataset_split)
    total = q.count()
    rows = q.order_by(TrainingData.added_date.desc()).offset(offset).limit(limit).all()
    return JSONResponse(content={
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [_row_to_dict(r) for r in rows],
    })


@router.get("/training-data/{item_id}")
async def get_training_data(
    item_id: str,
    db: Session = Depends(get_db),
):
    """Get one training_data row by ID."""
    try:
        iid = uuid.UUID(item_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid id")
    row = db.query(TrainingData).filter(TrainingData.id == iid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Training data not found")
    return JSONResponse(content=_row_to_dict(row))


class TrainingDataUpdate(BaseModel):
    class_name: Optional[str] = None
    class_id: Optional[int] = None
    source: Optional[str] = None
    is_validated: Optional[bool] = None
    dataset_split: Optional[str] = None


@router.patch("/training-data/{item_id}")
async def update_training_data(
    item_id: str,
    body: TrainingDataUpdate,
    db: Session = Depends(get_db),
):
    """Update training_data (partial)."""
    try:
        iid = uuid.UUID(item_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid id")
    row = db.query(TrainingData).filter(TrainingData.id == iid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Training data not found")
    if body.class_name is not None:
        row.class_name = body.class_name
    if body.class_id is not None:
        row.class_id = body.class_id
    if body.source is not None:
        row.source = body.source
    if body.is_validated is not None:
        row.is_validated = body.is_validated
    if body.dataset_split is not None:
        row.dataset_split = body.dataset_split
    db.commit()
    return JSONResponse(content={"ok": True, "id": item_id})


@router.delete("/training-data/{item_id}")
async def delete_training_data(
    item_id: str,
    db: Session = Depends(get_db),
):
    """Delete a training_data row."""
    try:
        iid = uuid.UUID(item_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid id")
    row = db.query(TrainingData).filter(TrainingData.id == iid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Training data not found")
    db.delete(row)
    db.commit()
    return JSONResponse(content={"ok": True, "deleted": item_id})


# --- Model performance ---

@router.get("/model-performance")
async def list_model_performance(
    db: Session = Depends(get_db),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    model_version: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None, description="ISO date YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="ISO date YYYY-MM-DD"),
):
    """List model_performance with optional filters."""
    q = db.query(ModelPerformance)
    if model_version:
        q = q.filter(ModelPerformance.model_version == model_version)
    if date_from:
        try:
            q = q.filter(ModelPerformance.date >= date.fromisoformat(date_from))
        except ValueError:
            pass
    if date_to:
        try:
            q = q.filter(ModelPerformance.date <= date.fromisoformat(date_to))
        except ValueError:
            pass
    total = q.count()
    rows = q.order_by(ModelPerformance.date.desc()).offset(offset).limit(limit).all()
    return JSONResponse(content={
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [_row_to_dict(r) for r in rows],
    })


@router.get("/model-performance/{item_id}")
async def get_model_performance(
    item_id: str,
    db: Session = Depends(get_db),
):
    """Get one model_performance row by ID."""
    try:
        iid = uuid.UUID(item_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid id")
    row = db.query(ModelPerformance).filter(ModelPerformance.id == iid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Model performance not found")
    return JSONResponse(content=_row_to_dict(row))


class ModelPerformanceUpdate(BaseModel):
    accuracy: Optional[float] = None
    class_metrics: Optional[dict] = None
    avg_confidence: Optional[float] = None


@router.patch("/model-performance/{item_id}")
async def update_model_performance(
    item_id: str,
    body: ModelPerformanceUpdate,
    db: Session = Depends(get_db),
):
    """Update model_performance (partial)."""
    try:
        iid = uuid.UUID(item_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid id")
    row = db.query(ModelPerformance).filter(ModelPerformance.id == iid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Model performance not found")
    if body.accuracy is not None:
        row.accuracy = body.accuracy
    if body.class_metrics is not None:
        row.class_metrics = body.class_metrics
    if body.avg_confidence is not None:
        row.avg_confidence = body.avg_confidence
    db.commit()
    return JSONResponse(content={"ok": True, "id": item_id})


@router.delete("/model-performance/{item_id}")
async def delete_model_performance(
    item_id: str,
    db: Session = Depends(get_db),
):
    """Delete a model_performance row."""
    try:
        iid = uuid.UUID(item_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid id")
    row = db.query(ModelPerformance).filter(ModelPerformance.id == iid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Model performance not found")
    db.delete(row)
    db.commit()
    return JSONResponse(content={"ok": True, "deleted": item_id})


# --- Feedback (list, get, update, delete; submit & stats sa /api/v1/feedback) ---

@router.get("/feedback")
async def list_feedback(
    db: Session = Depends(get_db),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    prediction_id: Optional[str] = Query(None),
    is_correct: Optional[bool] = Query(None),
):
    """List feedback with optional filters (prediction_id, is_correct)."""
    q = db.query(Feedback)
    if prediction_id:
        try:
            pid = uuid.UUID(prediction_id)
            q = q.filter(Feedback.prediction_id == pid)
        except ValueError:
            pass
    if is_correct is not None:
        q = q.filter(Feedback.is_correct == is_correct)
    total = q.count()
    rows = q.order_by(Feedback.timestamp.desc()).offset(offset).limit(limit).all()
    return JSONResponse(content={
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [_row_to_dict(r) for r in rows],
    })


@router.get("/feedback/{feedback_id}")
async def get_feedback(
    feedback_id: str,
    db: Session = Depends(get_db),
):
    """Get one feedback by ID."""
    try:
        fid = uuid.UUID(feedback_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid feedback_id")
    row = db.query(Feedback).filter(Feedback.id == fid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return JSONResponse(content=_row_to_dict(row))


class FeedbackUpdate(BaseModel):
    is_correct: Optional[bool] = None
    correct_class_name: Optional[str] = None
    correct_class_id: Optional[int] = None
    user_comment: Optional[str] = None
    confidence_rating: Optional[int] = None
    processed_for_training: Optional[bool] = None


@router.patch("/feedback/{feedback_id}")
async def update_feedback(
    feedback_id: str,
    body: FeedbackUpdate,
    db: Session = Depends(get_db),
):
    """Update feedback (partial). confidence_rating must be 1-5 if set."""
    try:
        fid = uuid.UUID(feedback_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid feedback_id")
    row = db.query(Feedback).filter(Feedback.id == fid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Feedback not found")
    if body.is_correct is not None:
        row.is_correct = body.is_correct
    if body.correct_class_name is not None:
        row.correct_class_name = body.correct_class_name
    if body.correct_class_id is not None:
        row.correct_class_id = body.correct_class_id
    if body.user_comment is not None:
        row.user_comment = body.user_comment
    if body.confidence_rating is not None:
        if body.confidence_rating < 1 or body.confidence_rating > 5:
            raise HTTPException(status_code=400, detail="confidence_rating must be 1-5")
        row.confidence_rating = body.confidence_rating
    if body.processed_for_training is not None:
        row.processed_for_training = body.processed_for_training
    db.commit()
    return JSONResponse(content={"ok": True, "feedback_id": feedback_id})


@router.delete("/feedback/{feedback_id}")
async def delete_feedback(
    feedback_id: str,
    db: Session = Depends(get_db),
):
    """Delete one feedback by ID."""
    try:
        fid = uuid.UUID(feedback_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid feedback_id")
    row = db.query(Feedback).filter(Feedback.id == fid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Feedback not found")
    db.delete(row)
    db.commit()
    return JSONResponse(content={"ok": True, "deleted": feedback_id})
