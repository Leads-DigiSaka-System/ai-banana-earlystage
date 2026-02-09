"""Unit tests for MinIO storage_service (Phase 2)."""

import io
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from services import storage_service


class TestIsConfigured:
    def test_false_when_storage_endpoint_unset(self):
        with patch.dict("os.environ", {"STORAGE_ENDPOINT": ""}, clear=False):
            with patch.object(storage_service, "_MINIO_AVAILABLE", True):
                assert storage_service.is_configured() is False

    def test_true_when_endpoint_set_and_minio_available(self):
        with patch.dict("os.environ", {"STORAGE_ENDPOINT": "localhost:9000"}, clear=False):
            with patch.object(storage_service, "_MINIO_AVAILABLE", True):
                assert storage_service.is_configured() is True

    def test_false_when_minio_not_installed(self):
        with patch.dict("os.environ", {"STORAGE_ENDPOINT": "localhost:9000"}, clear=False):
            with patch.object(storage_service, "_MINIO_AVAILABLE", False):
                assert storage_service.is_configured() is False


class TestGetBucket:
    def test_default_bucket(self):
        with patch.dict("os.environ", {}, clear=False):
            assert storage_service.get_bucket() == "ai-banana-early-stage"

    def test_custom_bucket_from_env(self):
        with patch.dict("os.environ", {"STORAGE_BUCKET": "my-bucket"}, clear=False):
            assert storage_service.get_bucket() == "my-bucket"


class TestGetClient:
    def test_returns_none_when_endpoint_unset(self):
        with patch.dict("os.environ", {"STORAGE_ENDPOINT": ""}, clear=False):
            with patch.object(storage_service, "_MINIO_AVAILABLE", True):
                assert storage_service.get_client() is None

    def test_returns_minio_client_when_configured(self):
        with patch.dict(
            "os.environ",
            {
                "STORAGE_ENDPOINT": "localhost:9000",
                "STORAGE_ACCESS_KEY": "key",
                "STORAGE_SECRET_KEY": "secret",
            },
            clear=False,
        ):
            with patch.object(storage_service, "_MINIO_AVAILABLE", True):
                with patch("services.storage_service.Minio") as MinioMock:
                    client = storage_service.get_client()
                    assert client is not None
                    MinioMock.assert_called_once_with(
                        "localhost:9000",
                        access_key="key",
                        secret_key="secret",
                        secure=False,
                    )


class TestEnsureBuckets:
    def test_calls_bucket_exists_and_make_bucket_when_missing(self):
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = False
        with patch("services.storage_service.get_bucket", return_value="ai-banana-early-stage"):
            storage_service.ensure_buckets(mock_client)
        mock_client.bucket_exists.assert_called_once_with("ai-banana-early-stage")
        mock_client.make_bucket.assert_called_once_with("ai-banana-early-stage")

    def test_does_not_make_bucket_when_exists(self):
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        with patch("services.storage_service.get_bucket", return_value="ai-banana-early-stage"):
            storage_service.ensure_buckets(mock_client)
        mock_client.make_bucket.assert_not_called()


class TestUploadPredictionImage:
    def test_puts_object_and_returns_path_with_bucket_and_predictions_prefix(self):
        mock_client = MagicMock()
        with patch("services.storage_service.get_bucket", return_value="ai-banana-early-stage"):
            path = storage_service.upload_prediction_image(
                mock_client,
                image_data=b"\xff\xd8\xff",
                user_id="user1",
                prediction_id="uuid-123",
                image_hash_prefix="abc123def456",
                file_extension="jpg",
            )
        assert path.startswith("ai-banana-early-stage/predictions/")
        assert "user1" in path
        assert "uuid-123" in path
        assert path.endswith(".jpg")
        mock_client.put_object.assert_called_once()
        call_kw = mock_client.put_object.call_args[1]
        assert call_kw["bucket_name"] == "ai-banana-early-stage"
        assert call_kw["object_name"].startswith("predictions/user1/")
        assert call_kw["content_type"] == "image/jpeg"
        assert call_kw["length"] == 3

    def test_content_type_jpeg_for_jpg(self):
        mock_client = MagicMock()
        with patch("services.storage_service.get_bucket", return_value="bucket"):
            storage_service.upload_prediction_image(
                mock_client,
                image_data=b"x",
                user_id="u",
                prediction_id="p",
                image_hash_prefix="h",
                file_extension="jpg",
            )
        assert mock_client.put_object.call_args[1]["content_type"] == "image/jpeg"


class TestGetImage:
    def test_returns_bytes_from_mock_response(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.read.return_value = b"image bytes"
        mock_client.get_object.return_value = mock_response
        data = storage_service.get_image(
            mock_client,
            "ai-banana-early-stage/predictions/user1/2025/02/file.jpg",
        )
        assert data == b"image bytes"
        mock_client.get_object.assert_called_once_with(
            "ai-banana-early-stage",
            "predictions/user1/2025/02/file.jpg",
        )
        mock_response.close.assert_called_once()
        mock_response.release_conn.assert_called_once()

    def test_invalid_path_raises(self):
        mock_client = MagicMock()
        with pytest.raises(ValueError, match="Invalid image_path"):
            storage_service.get_image(mock_client, "nopath")


class TestGetPresignedUrl:
    def test_calls_presigned_get_object_and_returns_url(self):
        mock_client = MagicMock()
        mock_client.presigned_get_object.return_value = "http://minio/url"
        url = storage_service.get_presigned_url(
            mock_client,
            "ai-banana-early-stage/predictions/user1/file.jpg",
            expires=timedelta(hours=2),
        )
        assert url == "http://minio/url"
        mock_client.presigned_get_object.assert_called_once_with(
            "ai-banana-early-stage",
            "predictions/user1/file.jpg",
            expires=timedelta(hours=2),
        )


class TestReadImageBytes:
    """Ensures whole phase connects: same code path can read local or MinIO image_path."""

    def test_read_local_path(self, tmp_path):
        (tmp_path / "data" / "uploads" / "predictions").mkdir(parents=True)
        img_path = tmp_path / "data" / "uploads" / "predictions" / "test.jpg"
        img_path.write_bytes(b"local image bytes")
        out = storage_service.read_image_bytes(
            "data/uploads/predictions/test.jpg",
            project_root=tmp_path,
        )
        assert out == b"local image bytes"

    def test_read_minio_path_uses_client(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.read.return_value = b"minio image bytes"
        mock_client.get_object.return_value = mock_response
        with patch("services.storage_service.get_client", return_value=mock_client):
            out = storage_service.read_image_bytes(
                "ai-banana-early-stage/predictions/user1/2025/02/file.jpg"
            )
        assert out == b"minio image bytes"
        mock_client.get_object.assert_called_once_with(
            "ai-banana-early-stage",
            "predictions/user1/2025/02/file.jpg",
        )

    def test_read_minio_path_no_client_raises(self):
        with patch("services.storage_service.get_client", return_value=None):
            with pytest.raises(ValueError, match="STORAGE_ENDPOINT is not set"):
                storage_service.read_image_bytes(
                    "ai-banana-early-stage/predictions/user1/file.jpg"
                )
