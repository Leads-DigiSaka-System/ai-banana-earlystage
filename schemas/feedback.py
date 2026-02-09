# Request/response schemas for feedback API.

from typing import Optional
from pydantic import BaseModel, Field


class FeedbackSubmitRequest(BaseModel):
    """Request body for POST /api/v1/feedback/submit. Send as JSON."""

    prediction_id: str = Field(
        ...,
        description="UUID of the prediction (from /predict or /predict/classify response when you sent user_id).",
    )
    is_correct: bool = Field(
        ...,
        description="True if the prediction was correct, false if wrong.",
    )
    correct_class_name: Optional[str] = Field(
        None,
        description="When is_correct is false: the actual class name (e.g. 'Stage3'). Used for retraining.",
    )
    correct_class_id: Optional[int] = Field(
        None,
        description="When is_correct is false: the actual class ID (0–6). Optional.",
    )
    user_comment: Optional[str] = Field(
        None,
        description="Optional free-text comment from the user.",
    )
    confidence_rating: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="How confident the user is in this feedback, 1–5 (1=not sure, 5=very sure).",
    )
