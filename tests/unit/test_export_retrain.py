"""Unit tests for export_feedback_for_training and run_retrain."""

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# --- Export script: pure helpers (no DB) ---

class TestGetImageSize:
    """Tests for _get_image_size in export_feedback_for_training."""

    def test_returns_fallback_for_empty_bytes(self):
        from scripts.export_feedback_for_training import _get_image_size
        assert _get_image_size(b"") == (736, 736)

    def test_returns_fallback_for_invalid_bytes(self):
        from scripts.export_feedback_for_training import _get_image_size
        assert _get_image_size(b"not an image") == (736, 736)

    def test_returns_size_from_jpeg(self, sample_image_bytes):
        from scripts.export_feedback_for_training import _get_image_size
        w, h = _get_image_size(sample_image_bytes)
        assert w == 1 and h == 1


class TestBboxXyxyToYoloNorm:
    """Tests for _bbox_xyxy_to_yolo_norm."""

    def test_center_and_size_normalized(self):
        from scripts.export_feedback_for_training import _bbox_xyxy_to_yolo_norm
        # 100x100 image, box 10-90 x 20-80
        xc, yc, w, h = _bbox_xyxy_to_yolo_norm(10.0, 20.0, 90.0, 80.0, 100, 100)
        assert abs(xc - 0.5) < 1e-6
        assert abs(yc - 0.5) < 1e-6
        assert abs(w - 0.8) < 1e-6
        assert abs(h - 0.6) < 1e-6

    def test_zero_dimensions_return_center_full(self):
        from scripts.export_feedback_for_training import _bbox_xyxy_to_yolo_norm
        xc, yc, w, h = _bbox_xyxy_to_yolo_norm(0, 0, 10, 10, 0, 0)
        assert (xc, yc, w, h) == (0.5, 0.5, 1.0, 1.0)

    def test_negative_dimensions_return_center_full(self):
        from scripts.export_feedback_for_training import _bbox_xyxy_to_yolo_norm
        xc, yc, w, h = _bbox_xyxy_to_yolo_norm(0, 0, 10, 10, -1, 100)
        assert (xc, yc, w, h) == (0.5, 0.5, 1.0, 1.0)

    def test_values_clamped_to_zero_one(self):
        from scripts.export_feedback_for_training import _bbox_xyxy_to_yolo_norm
        # Box outside image
        xc, yc, w, h = _bbox_xyxy_to_yolo_norm(-10, -10, 200, 200, 100, 100)
        assert 0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1


# --- Export script: export_feedback_for_training (mocked DB + storage) ---

def _make_mock_training_row(
    row_id: uuid.UUID | None = None,
    image_path: str = "data/uploads/predictions/test.jpg",
    image_hash: str = "abc12345",
    class_name: str = "Stage2",
    bbox_data: dict | None = None,
):
    row = MagicMock()
    row.id = row_id or uuid.uuid4()
    row.image_path = image_path
    row.image_hash = image_hash
    row.class_name = class_name
    row.bbox_data = bbox_data or {"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0}
    return row


class TestExportFeedbackForTraining:
    """Tests for export_feedback_for_training with mocked DB and read_image_bytes."""

    def test_empty_rows_raises_system_exit(self, tmp_path):
        from scripts.export_feedback_for_training import export_feedback_for_training

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
        mock_session.close = MagicMock()

        with patch("scripts.export_feedback_for_training._get_session_factory") as p:
            p.return_value.return_value = mock_session
            with pytest.raises(SystemExit, match="No training_data rows"):
                export_feedback_for_training(output_dir=tmp_path / "out", project_root=tmp_path)

    def test_export_creates_dirs_and_data_yaml(self, tmp_path, sample_image_bytes):
        from scripts.export_feedback_for_training import export_feedback_for_training

        row = _make_mock_training_row(
            image_path="data/uploads/predictions/foo.jpg",
            class_name="Stage1",
            bbox_data={"x1": 10.0, "y1": 20.0, "x2": 90.0, "y2": 80.0},
        )
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [row]
        mock_session.close = MagicMock()

        (tmp_path / "data" / "uploads" / "predictions").mkdir(parents=True)
        (tmp_path / "data" / "uploads" / "predictions" / "foo.jpg").write_bytes(sample_image_bytes)

        with patch("scripts.export_feedback_for_training._get_session_factory") as p_sess:
            p_sess.return_value.return_value = mock_session
            with patch("scripts.export_feedback_for_training.read_image_bytes", return_value=sample_image_bytes):
                data_yaml_path, n_train, n_val = export_feedback_for_training(
                    output_dir=tmp_path / "export",
                    val_ratio=0.5,
                    project_root=tmp_path,
                )

        assert data_yaml_path == tmp_path / "export" / "data.yaml"
        assert (n_train + n_val) == 1
        assert (tmp_path / "export" / "images" / "train").exists() or (tmp_path / "export" / "images" / "val").exists()
        assert (tmp_path / "export" / "labels" / "train").exists() or (tmp_path / "export" / "labels" / "val").exists()

        import yaml
        with open(data_yaml_path) as f:
            data = yaml.safe_load(f)
        assert data["train"] == "images/train"
        assert data["val"] == "images/val"
        assert "Stage1" in data["names"]
        assert data["nc"] >= 1

    def test_export_label_content_with_bbox(self, tmp_path, sample_image_bytes):
        from scripts.export_feedback_for_training import export_feedback_for_training

        row = _make_mock_training_row(
            class_name="Healthy",
            bbox_data={"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0},
        )
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [row]
        mock_session.close = MagicMock()

        with patch("scripts.export_feedback_for_training._get_session_factory") as p_sess:
            p_sess.return_value.return_value = mock_session
            with patch("scripts.export_feedback_for_training.read_image_bytes", return_value=sample_image_bytes):
                export_feedback_for_training(
                    output_dir=tmp_path / "export",
                    val_ratio=0.0,
                    project_root=tmp_path,
                )

        labels_dir = tmp_path / "export" / "labels" / "train"
        label_files = list(labels_dir.glob("*.txt"))
        assert len(label_files) == 1
        content = label_files[0].read_text()
        parts = content.strip().split()
        assert len(parts) == 5
        assert parts[0] == "0"  # Healthy is first when sorted
        assert 0 <= float(parts[1]) <= 1 and 0 <= float(parts[2]) <= 1

    def test_export_skips_missing_image_and_continues(self, tmp_path, sample_image_bytes):
        from scripts.export_feedback_for_training import export_feedback_for_training

        row_ok = _make_mock_training_row(image_path="data/uploads/predictions/ok.jpg", image_hash="aaa")
        row_bad = _make_mock_training_row(image_path="data/uploads/predictions/missing.jpg", image_hash="bbb")
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [row_ok, row_bad]
        mock_session.close = MagicMock()

        def read_image(path, project_root=None):
            if "missing" in path:
                raise FileNotFoundError(f"not found: {path}")
            return sample_image_bytes

        with patch("scripts.export_feedback_for_training._get_session_factory") as p_sess:
            p_sess.return_value.return_value = mock_session
            with patch("scripts.export_feedback_for_training.read_image_bytes", side_effect=read_image):
                data_yaml_path, n_train, n_val = export_feedback_for_training(
                    output_dir=tmp_path / "export",
                    val_ratio=0.0,
                    project_root=tmp_path,
                )

        assert (n_train + n_val) == 1
        assert (tmp_path / "export" / "images" / "train").exists()


# --- run_retrain: count and main ---

class TestCountFeedbackTrainingData:
    """Tests for _count_feedback_training_data in run_retrain."""

    def test_returns_count_from_db(self):
        from scripts.run_retrain import _count_feedback_training_data

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.count.return_value = 7
        mock_session.close = MagicMock()

        with patch("database.connection._get_session_factory") as p:
            p.return_value.return_value = mock_session
            count = _count_feedback_training_data()
        assert count == 7


class TestRunRetrainMain:
    """Tests for run_retrain.main()."""

    def test_exits_zero_when_below_min_samples(self):
        from scripts.run_retrain import main

        with patch("sys.argv", ["run_retrain", "--min-samples", "10"]):
            with patch("scripts.run_retrain._count_feedback_training_data", return_value=3):
                with patch("scripts.export_feedback_for_training.export_feedback_for_training") as p_export:
                    with patch("training.train_with_mlflow.train_model_with_tracking") as p_train:
                        exit_code = main()
                        assert exit_code == 0
                        p_export.assert_not_called()
                        p_train.assert_not_called()

    def test_dry_run_does_not_export_or_train(self):
        from scripts.run_retrain import main

        with patch("sys.argv", ["run_retrain", "--dry-run"]):
            with patch("scripts.run_retrain._count_feedback_training_data", return_value=20):
                with patch("scripts.export_feedback_for_training.export_feedback_for_training") as p_export:
                    with patch("training.train_with_mlflow.train_model_with_tracking") as p_train:
                        exit_code = main()
                        assert exit_code == 0
                        p_export.assert_not_called()
                        p_train.assert_not_called()

    def test_calls_export_and_train_when_above_threshold(self, tmp_path):
        from scripts.run_retrain import main

        with patch("sys.argv", ["run_retrain", "--min-samples", "10"]):
            with patch("scripts.run_retrain._count_feedback_training_data", return_value=15):
                with patch("scripts.export_feedback_for_training.export_feedback_for_training") as p_export:
                    p_export.return_value = (tmp_path / "data.yaml", 12, 3)
                    with patch("training.train_with_mlflow.train_model_with_tracking") as p_train:
                        exit_code = main()
                        assert exit_code == 0
                        p_export.assert_called_once()
                        p_train.assert_called_once()
                        call_kw = p_train.call_args[1]
                        assert str(call_kw["data_yaml"]).endswith("data.yaml")
                        assert call_kw["epochs"] == 50
                        assert call_kw["batch_size"] == 32
