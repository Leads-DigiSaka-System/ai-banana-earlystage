"""Unit tests for MLflow training: train_with_mlflow and _get_metric."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGetMetric:
    """Tests for _get_metric in train_with_mlflow."""

    def test_returns_value_from_results_dict(self):
        from training.train_with_mlflow import _get_metric
        results = MagicMock()
        results.results_dict = {"metrics/mAP50(B)": 0.85}
        assert _get_metric(results, "metrics/mAP50(B)") == 0.85

    def test_returns_default_when_key_missing(self):
        from training.train_with_mlflow import _get_metric
        results = MagicMock()
        results.results_dict = {}
        assert _get_metric(results, "metrics/mAP50(B)", 0.0) == 0.0

    def test_returns_default_when_no_results_dict(self):
        from training.train_with_mlflow import _get_metric
        results = MagicMock(spec=[])
        del results.results_dict
        results.results = None
        assert _get_metric(results, "x", 1.0) == 1.0


class TestTrainModelWithTracking:
    """Tests for train_model_with_tracking with mocked YOLO and MLflowManager."""

    def test_calls_mlflow_start_run_and_end_run(self, tmp_path):
        from training.train_with_mlflow import train_model_with_tracking

        (tmp_path / "data.yaml").write_text(
            f"path: {tmp_path!s}\ntrain: images/train\nval: images/val\nnames: [A,B]\nnc: 2\n"
        )
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "images" / "val").mkdir(parents=True)

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.train.return_value = MagicMock(
            results_dict={"metrics/mAP50(B)": 0.8},
            save_dir=tmp_path / "runs" / "detect" / "test",
        )
        (tmp_path / "runs" / "detect" / "test" / "weights").mkdir(parents=True)
        (tmp_path / "runs" / "detect" / "test" / "weights" / "best.pt").write_text("")

        with patch("training.train_with_mlflow.MLflowManager") as MockMLflow:
            with patch("training.train_with_mlflow.YOLO", return_value=mock_yolo_instance):
                mock_manager = MagicMock()
                MockMLflow.return_value = mock_manager
                train_model_with_tracking(
                    data_yaml=str(tmp_path / "data.yaml"),
                    base_model="yolo12n.pt",
                    epochs=2,
                    batch_size=4,
                    run_name="test_run",
                )
                mock_manager.start_run.assert_called_once()
                mock_manager.end_run.assert_called_once()

    def test_passes_data_and_params_to_model_train(self, tmp_path):
        from training.train_with_mlflow import train_model_with_tracking

        (tmp_path / "data.yaml").write_text(
            f"path: {tmp_path!s}\ntrain: images/train\nval: images/val\nnames: [A]\nnc: 1\n"
        )
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "images" / "val").mkdir(parents=True)

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.train.return_value = MagicMock(
            results_dict={},
            save_dir=tmp_path / "runs" / "detect" / "test",
        )
        (tmp_path / "runs" / "detect" / "test" / "weights").mkdir(parents=True)
        (tmp_path / "runs" / "detect" / "test" / "weights" / "best.pt").write_text("")

        with patch("training.train_with_mlflow.MLflowManager"):
            with patch("training.train_with_mlflow.YOLO", return_value=mock_yolo_instance):
                train_model_with_tracking(
                    data_yaml=str(tmp_path / "data.yaml"),
                    base_model="yolo12n.pt",
                    epochs=3,
                    batch_size=8,
                    run_name="test_run",
                )
                call_kw = mock_yolo_instance.train.call_args[1]
                assert call_kw["epochs"] == 3
                assert call_kw["batch"] == 8
                assert "data" in call_kw
                assert str(call_kw["data"]).endswith("data.yaml")
