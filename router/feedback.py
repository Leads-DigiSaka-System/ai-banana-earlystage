"""
Feedback API — submit at stats lang. List/Get/Patch/Delete ng feedback nasa database CRUD (/api/v1/db/feedback).
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from database.connection import get_db
from schemas.feedback import FeedbackSubmitRequest
from services.feedback_service import get_feedback_stats, save_feedback

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


@router.post(
    "/submit",
    summary="Submit feedback for a prediction",
    description="""
**Para saan:** I-submit kung tama o mali ang prediction.

**Kailangan:** `prediction_id` (from /predict or /predict/classify response when you sent `user_id`), at `is_correct` (true/false).

**Kung mali ang prediction:** Pwede mong isama ang `correct_class_name` (e.g. "Stage3") at optional `correct_class_id` (3). 
Ire-record ito at pwedeng gamitin para sa retraining.

**Optional:** `user_comment`, `confidence_rating` (1–5).
    """,
)
async def feedback_submit(
    body: FeedbackSubmitRequest,
    db: Session = Depends(get_db),
):
    try:
        feedback_id = save_feedback(
            db,
            prediction_id=body.prediction_id,
            is_correct=body.is_correct,
            correct_class_name=body.correct_class_name,
            correct_class_id=body.correct_class_id,
            user_comment=body.user_comment,
            confidence_rating=body.confidence_rating,
            feedback_source="api",
        )
        return JSONResponse(content={
            "success": True,
            "feedback_id": feedback_id,
            "message": "Thank you for your feedback!",
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/stats",
    summary="Feedback statistics (accuracy from users)",
    description="""
**Para saan:** Makita ang accuracy ng model batay sa feedback (ilang tama vs mali), 
per class stats, at feedback rate.

**Query params:** `days` (default 7) = last N days; `model_version` = filter by version.
    """,
)
async def feedback_stats(
    days: int = Query(7, ge=1, le=365, description="Last N days to include"),
    model_version: Optional[str] = Query(None, description="Filter by model version"),
    db: Session = Depends(get_db),
):
    stats = get_feedback_stats(db, model_version=model_version, days=days)
    return JSONResponse(content=stats)
