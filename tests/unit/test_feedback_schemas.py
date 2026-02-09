"""Unit tests for feedback API schemas."""

import pytest
from pydantic import ValidationError

from schemas.feedback import FeedbackSubmitRequest


def test_feedback_submit_request_valid_minimal():
    """Minimal valid body: prediction_id and is_correct only."""
    body = FeedbackSubmitRequest(
        prediction_id="a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
        is_correct=True,
    )
    assert body.prediction_id == "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"
    assert body.is_correct is True
    assert body.correct_class_name is None
    assert body.correct_class_id is None
    assert body.user_comment is None
    assert body.confidence_rating is None


def test_feedback_submit_request_valid_wrong_prediction():
    """Valid body for wrong prediction with correct class."""
    body = FeedbackSubmitRequest(
        prediction_id="b1ffcd00-ad1c-5fg9-cc7e-7cc0ce491b22",
        is_correct=False,
        correct_class_name="Stage3",
        correct_class_id=3,
        user_comment="Actually Stage3",
        confidence_rating=2,
    )
    assert body.is_correct is False
    assert body.correct_class_name == "Stage3"
    assert body.correct_class_id == 3
    assert body.user_comment == "Actually Stage3"
    assert body.confidence_rating == 2


def test_feedback_submit_request_prediction_id_required():
    """prediction_id is required."""
    with pytest.raises(ValidationError) as exc_info:
        FeedbackSubmitRequest(is_correct=True)
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("prediction_id",) for e in errors)


def test_feedback_submit_request_is_correct_required():
    """is_correct is required."""
    with pytest.raises(ValidationError) as exc_info:
        FeedbackSubmitRequest(
            prediction_id="a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
        )
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("is_correct",) for e in errors)


def test_feedback_submit_request_confidence_rating_optional():
    """confidence_rating can be 1-5 or None (Pydantic allows any int; API validates 1-5)."""
    body = FeedbackSubmitRequest(
        prediction_id="a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
        is_correct=True,
        confidence_rating=5,
    )
    assert body.confidence_rating == 5
