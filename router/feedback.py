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
**How to use:** Send a **JSON body** (Content-Type: application/json). No form-data; no `user_id` here.

**Step 1 — Get a prediction_id:**  
Call **POST /api/v1/predict** or **POST /api/v1/predict/classify** and include **user_id** in the form. The response will contain **prediction_id**.

**Step 2 — Submit feedback:**  
Call this endpoint with that `prediction_id` and `is_correct` (true/false). Optionally add correct class or comment.

**Required in body:**
- **prediction_id** (string): UUID from the predict response.
- **is_correct** (boolean): `true` if the prediction was right, `false` if wrong.

**Optional in body (when wrong):**
- **correct_class_name** (string): e.g. `"Stage3"` — used for retraining.
- **correct_class_id** (integer): 0–6.

**Optional in body (any time):**
- **user_comment** (string): Free text.
- **confidence_rating** (integer): 1–5 (how sure the user is).

**Example — correct prediction:**
```json
{"prediction_id": "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11", "is_correct": true}
```

**Example — wrong prediction (with correct class):**
```json
{
  "prediction_id": "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
  "is_correct": false,
  "correct_class_name": "Stage3",
  "correct_class_id": 3
}
```
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
**How to use:** GET with optional query parameters. No request body.

**Query parameters:**
- **days** (default: 7): Last N days of data. Min 1, max 365.
- **model_version** (optional): Filter by model version string.

**Response:** Overall accuracy, total predictions, total feedback, correct/incorrect counts, feedback rate, and per-class statistics (accuracy per class).
    """,
)
async def feedback_stats(
    days: int = Query(7, ge=1, le=365, description="Last N days to include"),
    model_version: Optional[str] = Query(None, description="Filter by model version"),
    db: Session = Depends(get_db),
):
    stats = get_feedback_stats(db, model_version=model_version, days=days)
    return JSONResponse(content=stats)
