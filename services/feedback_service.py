"""
Feedback enhancement: save predictions, save feedback, stats.
Uses PostgreSQL; prediction images: MinIO (Phase 2) if configured, else local disk.
"""
import hashlib
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from database.models import Feedback, Prediction, TrainingData

# Local storage fallback when MinIO is not configured
UPLOADS_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads" / "predictions"
MODEL_VERSION = "v1.0"
MODEL_NAME = "yolo12n"


def _ensure_uploads_dir() -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOADS_DIR


def save_prediction(
    db: Session,
    image_bytes: bytes,
    filename: str,
    user_id: str,
    prediction_result: dict,
    model_version: str = MODEL_VERSION,
    model_name: str = MODEL_NAME,
    inference_time_ms: Optional[float] = None,
    user_location: Optional[str] = None,
    prediction_id: Optional[str] = None,
) -> str:
    """
    Save image to MinIO (if configured) or local disk, then insert prediction row.
    Returns prediction_id (UUID string). Pass prediction_id to use a specific UUID (e.g. from router).
    prediction_result: at least class_id, class_name, confidence; optional bbox.
    """
    prediction_id = uuid.UUID(prediction_id) if prediction_id else uuid.uuid4()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    ext = Path(filename).suffix.lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    ext_clean = ext.lstrip(".")

    # Phase 2: MinIO when configured
    try:
        from services.storage_service import (
            get_client,
            ensure_buckets,
            upload_prediction_image,
            is_configured,
        )
        client = get_client() if is_configured() else None
    except Exception:
        client = None

    image_path = None
    if client is not None:
        try:
            ensure_buckets(client)
            image_path = upload_prediction_image(
                client,
                image_bytes,
                user_id=user_id,
                prediction_id=str(prediction_id),
                image_hash_prefix=image_hash,
                file_extension=ext_clean,
            )
        except Exception as e:
            logging.getLogger(__name__).warning(
                "MinIO upload failed, saving image locally: %s", e
            )
            image_path = None
    if image_path is None:
        _ensure_uploads_dir()
        path = UPLOADS_DIR / f"{prediction_id}{ext}"
        path.write_bytes(image_bytes)
        image_path = f"data/uploads/predictions/{prediction_id}{ext}"
    image_size_kb = len(image_bytes) // 1024
    width = height = None  # optional: decode and get size if needed
    class_id = prediction_result.get("class_id", 0)
    class_name = prediction_result.get("class_name", "Unknown")
    confidence = float(prediction_result.get("confidence", 0.0))
    bbox = prediction_result.get("bbox")

    row = Prediction(
        id=prediction_id,
        user_id=user_id,
        user_location=user_location,
        image_path=image_path,
        image_size_kb=image_size_kb,
        image_width=width,
        image_height=height,
        image_hash=image_hash,
        predicted_class_id=class_id,
        predicted_class_name=class_name,
        confidence=confidence,
        bbox_data=bbox,
        model_version=model_version,
        model_name=model_name,
        inference_time_ms=inference_time_ms,
    )
    db.add(row)
    db.commit()
    return str(prediction_id)


def save_feedback(
    db: Session,
    prediction_id: str,
    is_correct: bool,
    correct_class_name: Optional[str] = None,
    correct_class_id: Optional[int] = None,
    user_comment: Optional[str] = None,
    confidence_rating: Optional[int] = None,
    feedback_source: str = "api",
) -> str:
    """Save feedback row. Optionally add to training_data if wrong. Returns feedback_id."""
    raw = (prediction_id or "").strip()
    if not raw:
        raise ValueError("prediction_id is required")
    try:
        pid = uuid.UUID(raw)
    except ValueError:
        raise ValueError("Invalid prediction_id (must be a valid UUID)")

    prediction = db.query(Prediction).filter(Prediction.id == pid).first()
    if not prediction:
        log = logging.getLogger(__name__)
        log.warning("Prediction not found: id=%s (UUID=%s). Check same DB and that predict saved.", raw, pid)
        raise ValueError(
            f"Prediction not found for id {raw}. "
            "Use the prediction_id from the predict response (same request that returned it). "
            "If you just predicted, ensure the API saved it (response must include prediction_id)."
        )

    if confidence_rating is not None and (confidence_rating < 1 or confidence_rating > 5):
        raise ValueError("confidence_rating must be 1-5")

    feedback = Feedback(
        prediction_id=pid,
        is_correct=is_correct,
        correct_class_id=correct_class_id,
        correct_class_name=correct_class_name,
        user_comment=user_comment,
        confidence_rating=confidence_rating,
        feedback_source=feedback_source,
    )
    db.add(feedback)
    db.commit()

    if not is_correct and correct_class_name is not None:
        _queue_for_training(db, prediction, correct_class_name, correct_class_id, feedback.id)

    return str(feedback.id)


def _queue_for_training(
    db: Session,
    prediction: Prediction,
    correct_class_name: str,
    correct_class_id: Optional[int],
    source_id: uuid.UUID,
) -> None:
    """Add corrected sample to training_data for later retraining."""
    class_id = correct_class_id if correct_class_id is not None else 0
    existing = db.query(TrainingData).filter(
        TrainingData.image_hash == prediction.image_hash
    ).first()
    if existing:
        existing.class_name = correct_class_name
        existing.class_id = class_id
        existing.source = "feedback"
        existing.source_id = source_id
    else:
        row = TrainingData(
            image_path=prediction.image_path,
            image_hash=prediction.image_hash,
            class_id=class_id,
            class_name=correct_class_name,
            bbox_data=prediction.bbox_data,
            source="feedback",
            source_id=source_id,
            dataset_split="pending",
        )
        db.add(row)
    db.commit()


def get_feedback_stats(
    db: Session,
    model_version: Optional[str] = None,
    days: int = 7,
) -> dict[str, Any]:
    """Return stats: total_predictions, total_feedback, correct, incorrect, accuracy, class_statistics."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    q = db.query(Prediction).filter(Prediction.timestamp >= since)
    if model_version:
        q = q.filter(Prediction.model_version == model_version)
    predictions = q.all()
    pred_ids = [p.id for p in predictions]
    feedbacks = db.query(Feedback).filter(Feedback.prediction_id.in_(pred_ids)).all() if pred_ids else []

    total_predictions = len(predictions)
    total_feedback = len(feedbacks)
    correct = sum(1 for f in feedbacks if f.is_correct)
    incorrect = total_feedback - correct
    accuracy = (correct / total_feedback) if total_feedback else 0.0

    class_stats: dict[str, dict] = {}
    for p in predictions:
        c = p.predicted_class_name
        if c not in class_stats:
            class_stats[c] = {"total_predictions": 0, "total_feedback": 0, "correct": 0, "incorrect": 0, "accuracy": 0.0}
        class_stats[c]["total_predictions"] += 1
    for f in feedbacks:
        pred = next((p for p in predictions if p.id == f.prediction_id), None)
        if pred:
            c = pred.predicted_class_name
            class_stats[c]["total_feedback"] += 1
            if f.is_correct:
                class_stats[c]["correct"] += 1
            else:
                class_stats[c]["incorrect"] += 1
    for c in class_stats:
        t = class_stats[c]["total_feedback"]
        class_stats[c]["accuracy"] = (class_stats[c]["correct"] / t) if t else 0.0

    return {
        "period_days": days,
        "total_predictions": total_predictions,
        "total_feedback": total_feedback,
        "correct_predictions": correct,
        "incorrect_predictions": incorrect,
        "overall_accuracy": accuracy,
        "feedback_rate": (total_feedback / total_predictions) if total_predictions else 0.0,
        "class_statistics": class_stats,
    }
