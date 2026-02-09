"""Unit tests for feedback service (save_prediction, save_feedback, get_feedback_stats)."""

import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.feedback_service import (
    get_feedback_stats,
    save_feedback,
    save_prediction,
)


class TestSavePrediction:
    """Tests for save_prediction."""

    @pytest.fixture(autouse=True)
    def temp_uploads_dir(self, tmp_path):
        """Patch UPLOADS_DIR to tmp_path and force MinIO off so tests use local path (unless overridden)."""
        with patch("services.feedback_service.UPLOADS_DIR", tmp_path / "predictions"):
            with patch("services.storage_service.is_configured", return_value=False):
                yield

    def test_save_prediction_calls_db_add_and_commit(
        self, mock_db, sample_image_bytes, sample_prediction_result
    ):
        save_prediction(
            mock_db,
            sample_image_bytes,
            "test.jpg",
            "user_1",
            sample_prediction_result,
        )
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        added = mock_db.add.call_args[0][0]
        assert added.user_id == "user_1"
        assert added.predicted_class_name == "Stage2"
        assert added.predicted_class_id == 2
        assert added.confidence == 0.85
        assert added.model_version == "v1.0"
        assert added.image_path.startswith("data/uploads/predictions/")
        assert added.image_path.endswith(".jpg")

    def test_save_prediction_returns_uuid_string(
        self, mock_db, sample_image_bytes, sample_prediction_result
    ):
        out = save_prediction(
            mock_db,
            sample_image_bytes,
            "leaf.png",
            "user_2",
            {"class_id": 1, "class_name": "Stage1", "confidence": 0.9},
        )
        uuid.UUID(out)
        assert isinstance(out, str)

    def test_save_prediction_normalizes_extension(
        self, mock_db, sample_image_bytes, sample_prediction_result
    ):
        save_prediction(
            mock_db,
            sample_image_bytes,
            "x.JPEG",
            "u",
            sample_prediction_result,
        )
        added = mock_db.add.call_args[0][0]
        assert added.image_path.endswith(".jpeg") or ".jpeg" in added.image_path

    def test_save_prediction_with_optional_args(
        self, mock_db, sample_image_bytes, sample_prediction_result
    ):
        save_prediction(
            mock_db,
            sample_image_bytes,
            "test.jpg",
            "user_1",
            sample_prediction_result,
            inference_time_ms=50.0,
            user_location="Manila",
        )
        added = mock_db.add.call_args[0][0]
        assert added.inference_time_ms == 50.0
        assert added.user_location == "Manila"

    def test_save_prediction_uses_minio_path_when_configured(
        self, mock_db, sample_image_bytes, sample_prediction_result
    ):
        """When MinIO is configured, image_path should be bucket/predictions/... (no local write)."""
        mock_client = MagicMock()
        minio_path = "ai-banana-early-stage/predictions/user_1/2025/02/abc123_xyz.jpg"
        with patch("services.storage_service.get_client", return_value=mock_client):
            with patch("services.storage_service.is_configured", return_value=True):
                with patch("services.storage_service.ensure_buckets") as ensure_buckets:
                    with patch(
                        "services.storage_service.upload_prediction_image",
                        return_value=minio_path,
                    ) as upload_mock:
                        out = save_prediction(
                            mock_db,
                            sample_image_bytes,
                            "test.jpg",
                            "user_1",
                            sample_prediction_result,
                        )
        ensure_buckets.assert_called_once_with(mock_client)
        upload_mock.assert_called_once()
        # upload_prediction_image(client, image_data, user_id=..., ...)
        args, kwargs = upload_mock.call_args
        assert args[1] == sample_image_bytes
        assert kwargs["user_id"] == "user_1"
        added = mock_db.add.call_args[0][0]
        assert added.image_path == minio_path
        assert added.image_path.startswith("ai-banana-early-stage/predictions/")
        uuid.UUID(out)


class TestSaveFeedback:
    """Tests for save_feedback."""

    def test_save_feedback_invalid_prediction_id_raises(self, mock_db):
        with pytest.raises(ValueError, match="Invalid prediction_id"):
            save_feedback(mock_db, "not-a-uuid", is_correct=True)

    def test_save_feedback_prediction_not_found_raises(
        self, mock_db, prediction_id_str
    ):
        mock_db.query.return_value.filter.return_value.first.return_value = None
        with pytest.raises(ValueError, match="Prediction not found"):
            save_feedback(mock_db, prediction_id_str, is_correct=True)

    def test_save_feedback_confidence_rating_out_of_range_raises(
        self, mock_db, prediction_id_str, mock_prediction
    ):
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_prediction
        )
        with pytest.raises(ValueError, match="confidence_rating must be 1-5"):
            save_feedback(
                mock_db,
                prediction_id_str,
                is_correct=True,
                confidence_rating=0,
            )
        with pytest.raises(ValueError, match="confidence_rating must be 1-5"):
            save_feedback(
                mock_db,
                prediction_id_str,
                is_correct=True,
                confidence_rating=6,
            )

    def test_save_feedback_success_correct(
        self, mock_db, prediction_id_str, mock_prediction
    ):
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_prediction
        )
        feedback_id = save_feedback(
            mock_db,
            prediction_id_str,
            is_correct=True,
            confidence_rating=5,
        )
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called()
        added = mock_db.add.call_args[0][0]
        assert added.is_correct is True
        assert added.prediction_id == mock_prediction.id
        assert added.confidence_rating == 5
        assert isinstance(feedback_id, str) and len(feedback_id) > 0

    def test_save_feedback_success_incorrect_queues_training(
        self, mock_db, prediction_id_str, mock_prediction
    ):
        mock_prediction.image_hash = "hash123"
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            mock_prediction,
            None,
        ]
        save_feedback(
            mock_db,
            prediction_id_str,
            is_correct=False,
            correct_class_name="Stage3",
            correct_class_id=3,
        )
        assert mock_db.add.call_count >= 2
        assert mock_db.commit.call_count >= 2


class TestGetFeedbackStats:
    """Tests for get_feedback_stats."""

    def test_get_feedback_stats_empty(self, mock_db):
        mock_db.query.return_value.filter.return_value.all.return_value = []
        out = get_feedback_stats(mock_db, days=7)
        assert out["period_days"] == 7
        assert out["total_predictions"] == 0
        assert out["total_feedback"] == 0
        assert out["correct_predictions"] == 0
        assert out["incorrect_predictions"] == 0
        assert out["overall_accuracy"] == 0.0
        assert out["feedback_rate"] == 0.0
        assert out["class_statistics"] == {}

    def test_get_feedback_stats_with_data(self, mock_db, mock_prediction):
        mock_prediction.predicted_class_name = "Stage2"
        mock_prediction.id = uuid.uuid4()
        mock_feedback = MagicMock()
        mock_feedback.prediction_id = mock_prediction.id
        mock_feedback.is_correct = True
        mock_db.query.return_value.filter.return_value.all.side_effect = [
            [mock_prediction],
            [mock_feedback],
        ]
        out = get_feedback_stats(mock_db, days=14)
        assert out["total_predictions"] == 1
        assert out["total_feedback"] == 1
        assert out["correct_predictions"] == 1
        assert out["incorrect_predictions"] == 0
        assert out["overall_accuracy"] == 1.0
        assert "Stage2" in out["class_statistics"]
        assert out["class_statistics"]["Stage2"]["total_feedback"] == 1
        assert out["class_statistics"]["Stage2"]["correct"] == 1
        assert out["class_statistics"]["Stage2"]["accuracy"] == 1.0

    def test_get_feedback_stats_filters_by_model_version(self, mock_db):
        get_feedback_stats(mock_db, model_version="v1.0", days=7)
        filter_calls = mock_db.query.return_value.filter.call_count
        assert filter_calls >= 1
