"""Unit tests for database CRUD router (/api/v1/db)."""

import uuid
from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from tests.conftest import make_query_chain, mock_row


# --- Predictions ---


class TestPredictions:
    def test_list_predictions_empty(self, db_crud_client, mock_db):
        mock_db.query.return_value = make_query_chain(count=0, all_items=[])
        r = db_crud_client.get("/api/v1/db/predictions")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["items"] == []
        assert data["limit"] == 100
        assert data["offset"] == 0

    def test_list_predictions_with_items(self, db_crud_client, mock_db):
        pid = uuid.uuid4()
        ts = datetime.now().isoformat()
        row = mock_row(
            id=pid,
            user_id="user1",
            predicted_class_name="Stage2",
            timestamp=ts,
            metadata_=None,
        )
        mock_db.query.return_value = make_query_chain(count=1, all_items=[row])
        r = db_crud_client.get("/api/v1/db/predictions?limit=10&offset=0")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["id"] == str(pid)
        assert data["items"][0]["user_id"] == "user1"

    def test_get_prediction_invalid_uuid(self, db_crud_client):
        r = db_crud_client.get("/api/v1/db/predictions/not-a-uuid")
        assert r.status_code == 400
        assert "Invalid prediction_id" in r.json()["detail"]

    def test_get_prediction_not_found(self, db_crud_client, mock_db):
        mock_db.query.return_value = make_query_chain(first_result=None)
        r = db_crud_client.get(f"/api/v1/db/predictions/{uuid.uuid4()}")
        assert r.status_code == 404
        assert "Prediction not found" in r.json()["detail"]

    def test_get_prediction_ok(self, db_crud_client, mock_db):
        pid = uuid.uuid4()
        row = mock_row(id=pid, user_id="u1", predicted_class_name="Stage1", metadata_=None)
        mock_db.query.return_value = make_query_chain(first_result=row)
        r = db_crud_client.get(f"/api/v1/db/predictions/{pid}")
        assert r.status_code == 200
        assert r.json()["id"] == str(pid)
        assert r.json()["user_id"] == "u1"

    def test_update_prediction_not_found(self, db_crud_client, mock_db):
        mock_db.query.return_value = make_query_chain(first_result=None)
        r = db_crud_client.patch(
            f"/api/v1/db/predictions/{uuid.uuid4()}",
            json={"predicted_class_name": "Stage3"},
        )
        assert r.status_code == 404

    def test_update_prediction_ok(self, db_crud_client, mock_db):
        pid = uuid.uuid4()
        row = MagicMock()
        row.id = pid
        row.user_id = "u1"
        row.predicted_class_name = "Stage1"
        row.user_location = None
        row.predicted_class_id = 1
        row.confidence = 0.9
        mock_db.query.return_value = make_query_chain(first_result=row)
        r = db_crud_client.patch(
            f"/api/v1/db/predictions/{pid}",
            json={"predicted_class_name": "Stage3"},
        )
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert row.predicted_class_name == "Stage3"
        mock_db.commit.assert_called_once()

    def test_delete_prediction_invalid_uuid(self, db_crud_client):
        r = db_crud_client.delete("/api/v1/db/predictions/bad-uuid")
        assert r.status_code == 400

    def test_delete_prediction_ok(self, db_crud_client, mock_db):
        pid = uuid.uuid4()
        row = MagicMock()
        mock_db.query.return_value = make_query_chain(first_result=row)
        r = db_crud_client.delete(f"/api/v1/db/predictions/{pid}")
        assert r.status_code == 200
        assert r.json()["deleted"] == str(pid)
        mock_db.delete.assert_called_once_with(row)
        mock_db.commit.assert_called_once()


# --- Training data ---


class TestTrainingData:
    def test_list_training_data_empty(self, db_crud_client, mock_db):
        mock_db.query.return_value = make_query_chain(count=0, all_items=[])
        r = db_crud_client.get("/api/v1/db/training-data")
        assert r.status_code == 200
        assert r.json()["total"] == 0
        assert r.json()["items"] == []

    def test_get_training_data_invalid_uuid(self, db_crud_client):
        r = db_crud_client.get("/api/v1/db/training-data/not-a-uuid")
        assert r.status_code == 400

    def test_get_training_data_not_found(self, db_crud_client, mock_db):
        mock_db.query.return_value = make_query_chain(first_result=None)
        r = db_crud_client.get(f"/api/v1/db/training-data/{uuid.uuid4()}")
        assert r.status_code == 404
        assert "Training data not found" in r.json()["detail"]

    def test_get_training_data_ok(self, db_crud_client, mock_db):
        iid = uuid.uuid4()
        row = mock_row(id=iid, class_name="Stage2", source="manual", metadata_=None)
        mock_db.query.return_value = make_query_chain(first_result=row)
        r = db_crud_client.get(f"/api/v1/db/training-data/{iid}")
        assert r.status_code == 200
        assert r.json()["id"] == str(iid)

    def test_update_training_data_ok(self, db_crud_client, mock_db):
        iid = uuid.uuid4()
        row = MagicMock()
        row.id = iid
        row.class_name = "Stage1"
        row.dataset_split = None
        mock_db.query.return_value = make_query_chain(first_result=row)
        r = db_crud_client.patch(
            f"/api/v1/db/training-data/{iid}",
            json={"dataset_split": "train"},
        )
        assert r.status_code == 200
        assert row.dataset_split == "train"
        mock_db.commit.assert_called_once()

    def test_delete_training_data_ok(self, db_crud_client, mock_db):
        iid = uuid.uuid4()
        row = MagicMock()
        mock_db.query.return_value = make_query_chain(first_result=row)
        r = db_crud_client.delete(f"/api/v1/db/training-data/{iid}")
        assert r.status_code == 200
        mock_db.delete.assert_called_once_with(row)


# --- Model performance ---


class TestModelPerformance:
    def test_list_model_performance_empty(self, db_crud_client, mock_db):
        mock_db.query.return_value = make_query_chain(count=0, all_items=[])
        r = db_crud_client.get("/api/v1/db/model-performance")
        assert r.status_code == 200
        assert r.json()["total"] == 0

    def test_get_model_performance_not_found(self, db_crud_client, mock_db):
        mock_db.query.return_value = make_query_chain(first_result=None)
        r = db_crud_client.get(f"/api/v1/db/model-performance/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_get_model_performance_ok(self, db_crud_client, mock_db):
        iid = uuid.uuid4()
        row = mock_row(
            id=iid,
            model_version="v1",
            date=date.today().isoformat(),
            accuracy=0.85,
            metadata_=None,
        )
        mock_db.query.return_value = make_query_chain(first_result=row)
        r = db_crud_client.get(f"/api/v1/db/model-performance/{iid}")
        assert r.status_code == 200
        assert r.json()["id"] == str(iid)

    def test_update_model_performance_ok(self, db_crud_client, mock_db):
        iid = uuid.uuid4()
        row = MagicMock()
        row.id = iid
        row.accuracy = 0.8
        row.class_metrics = None
        row.avg_confidence = None
        mock_db.query.return_value = make_query_chain(first_result=row)
        r = db_crud_client.patch(
            f"/api/v1/db/model-performance/{iid}",
            json={"accuracy": 0.92},
        )
        assert r.status_code == 200
        assert row.accuracy == 0.92
        mock_db.commit.assert_called_once()

    def test_delete_model_performance_ok(self, db_crud_client, mock_db):
        iid = uuid.uuid4()
        row = MagicMock()
        mock_db.query.return_value = make_query_chain(first_result=row)
        r = db_crud_client.delete(f"/api/v1/db/model-performance/{iid}")
        assert r.status_code == 200
        mock_db.delete.assert_called_once_with(row)


# --- Feedback ---


class TestFeedback:
    def test_list_feedback_empty(self, db_crud_client, mock_db):
        mock_db.query.return_value = make_query_chain(count=0, all_items=[])
        r = db_crud_client.get("/api/v1/db/feedback")
        assert r.status_code == 200
        assert r.json()["total"] == 0
        assert r.json()["items"] == []

    def test_get_feedback_invalid_uuid(self, db_crud_client):
        r = db_crud_client.get("/api/v1/db/feedback/not-a-uuid")
        assert r.status_code == 400
        assert "Invalid feedback_id" in r.json()["detail"]

    def test_get_feedback_not_found(self, db_crud_client, mock_db):
        mock_db.query.return_value = make_query_chain(first_result=None)
        r = db_crud_client.get(f"/api/v1/db/feedback/{uuid.uuid4()}")
        assert r.status_code == 404
        assert "Feedback not found" in r.json()["detail"]

    def test_get_feedback_ok(self, db_crud_client, mock_db):
        fid = uuid.uuid4()
        row = mock_row(
            id=fid,
            prediction_id=uuid.uuid4(),
            is_correct=True,
            timestamp=datetime.now().isoformat(),
            metadata_=None,
        )
        mock_db.query.return_value = make_query_chain(first_result=row)
        r = db_crud_client.get(f"/api/v1/db/feedback/{fid}")
        assert r.status_code == 200
        assert r.json()["id"] == str(fid)

    def test_update_feedback_ok(self, db_crud_client, mock_db):
        fid = uuid.uuid4()
        row = MagicMock()
        row.id = fid
        row.is_correct = True
        row.correct_class_name = None
        row.correct_class_id = None
        row.user_comment = None
        row.confidence_rating = None
        row.processed_for_training = False
        mock_db.query.return_value = make_query_chain(first_result=row)
        r = db_crud_client.patch(
            f"/api/v1/db/feedback/{fid}",
            json={"processed_for_training": True},
        )
        assert r.status_code == 200
        assert row.processed_for_training is True
        mock_db.commit.assert_called_once()

    def test_update_feedback_confidence_rating_invalid(self, db_crud_client, mock_db):
        fid = uuid.uuid4()
        row = MagicMock()
        row.id = fid
        row.confidence_rating = 3
        mock_db.query.return_value = make_query_chain(first_result=row)
        r = db_crud_client.patch(
            f"/api/v1/db/feedback/{fid}",
            json={"confidence_rating": 10},
        )
        assert r.status_code == 400
        assert "1-5" in r.json()["detail"]

    def test_delete_feedback_ok(self, db_crud_client, mock_db):
        fid = uuid.uuid4()
        row = MagicMock()
        mock_db.query.return_value = make_query_chain(first_result=row)
        r = db_crud_client.delete(f"/api/v1/db/feedback/{fid}")
        assert r.status_code == 200
        assert r.json()["deleted"] == str(fid)
        mock_db.delete.assert_called_once_with(row)
        mock_db.commit.assert_called_once()
