# Request/response schemas for feedback API.
# TODO: Add FeedbackSubmit, FeedbackStats, etc. when implementing endpoints.

from typing import Optional
from pydantic import BaseModel


class FeedbackSubmitRequest(BaseModel):
    """Body for POST /api/v1/feedback/submit."""
    prediction_id: str
    is_correct: bool
    correct_class_name: Optional[str] = None
    correct_class_id: Optional[int] = None
    user_comment: Optional[str] = None
    confidence_rating: Optional[int] = None
