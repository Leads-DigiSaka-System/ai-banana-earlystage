# training/train_with_mlflow.py
# Train YOLO with MLflow tracking (Enhancement 2).

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from mlops.mlflow_integration import MLflowManager
from ultralytics import YOLO


def _get_metric(results: Any, key: str, default: float = 0.0) -> float:
    """Get metric from Ultralytics results (works across versions)."""
    rd = getattr(results, "results_dict", None) or getattr(results, "results", None)
    if isinstance(rd, dict):
        val = rd.get(key)
        if val is not None:
            return float(val)
    return default


def train_model_with_tracking(
    data_yaml: str,
    base_model: str = "yolo12n.pt",
    epochs: int = 50,
    batch_size: int = 32,
    img_size: Optional[int] = None,
    lr0: float = 0.001,
    optimizer: str = "AdamW",
    experiment_name: str = "banana_sigatoka_detection",
    run_name: Optional[str] = None,
    tracking_uri: str = "http://localhost:5000",
    registered_model_name: str = "banana_sigatoka_detector",
):
    """
    Train YOLO model with MLflow tracking.

    Args:
        data_yaml: Path to dataset YAML (e.g. combined_yolo_dataset/data.yaml)
        base_model: Pretrained weights (e.g. yolo12n.pt) or path to checkpoint to continue training (e.g. models/weights/best.pt, runs/detect/exp/weights/last.pt)
        epochs: Number of epochs
        batch_size: Batch size
        img_size: Image size
        lr0: Initial learning rate
        optimizer: Optimizer name
        experiment_name: MLflow experiment name
        run_name: MLflow run name (default: training_YYYYMMDD_HHMMSS)
        tracking_uri: MLflow server URL
        registered_model_name: Model name in MLflow registry
    """
    mlflow_manager = MLflowManager(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    if run_name is None:
        run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_yaml_path = Path(data_yaml).resolve()
    data_yaml = str(data_yaml_path)
    if img_size is None:
        try:
            from config import MODEL_IMAGE_SIZE
            img_size = int(MODEL_IMAGE_SIZE)
        except Exception:
            img_size = 736

    mlflow_manager.start_run(
        run_name=run_name,
        tags={
            "model_type": base_model.replace(".pt", ""),
            "task": "object_detection",
            "dataset": Path(data_yaml).stem,
        },
    )

    try:
        params = {
            "model": base_model,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "lr0": lr0,
            "optimizer": optimizer,
            "patience": 20,
            "warmup_epochs": 5,
        }
        mlflow_manager.log_parameters(params)

        # Dataset info: resolve paths relative to data.yaml dir or "path" key
        try:
            with open(data_yaml) as f:
                dataset_config = yaml.safe_load(f)
            yaml_dir = data_yaml_path.parent
            base = Path(dataset_config.get("path", str(yaml_dir))).resolve()
            if not base.is_absolute():
                base = yaml_dir / base
            train_p = base / dataset_config.get("train", "images/train")
            val_p = base / dataset_config.get("val", "images/val")
            test_p = base / dataset_config.get("test", dataset_config.get("val", "images/val"))
            dataset_info: Dict[str, Any] = {
                "dataset_path": str(base),
                "num_classes": dataset_config.get("nc", 0),
                "class_names": dataset_config.get("names", []),
                "train_images": len(list(train_p.glob("*.jpg"))) + len(list(train_p.glob("*.png"))) if train_p.exists() else 0,
                "val_images": len(list(val_p.glob("*.jpg"))) + len(list(val_p.glob("*.png"))) if val_p.exists() else 0,
                "test_images": len(list(test_p.glob("*.jpg"))) + len(list(test_p.glob("*.png"))) if test_p.exists() else 0,
            }
            mlflow_manager.log_dataset_info(dataset_info)
        except Exception as e:
            print(f"Note: could not log dataset info: {e}")

        # Training config aligned with notebook (bsed-training): augmentation + optimizer
        train_kw: Dict[str, Any] = {
            "data": data_yaml,
            "epochs": epochs,
            "imgsz": img_size,
            "batch": batch_size,
            "lr0": lr0,
            "optimizer": optimizer,
            "patience": 20,
            "save": True,
            "plots": True,
            "verbose": True,
            "project": "runs/detect",
            "name": run_name,
            # Optimizer / LR (notebook)
            "lrf": 0.01,
            "warmup_epochs": 5,
            "cos_lr": True,
            # Augmentation - CRITICAL for pest/disease (notebook Cell 8)
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 10.0,
            "translate": 0.1,
            "fliplr": 0.5,
            "scale": 0.7,
            "shear": 2.0,
            "perspective": 0.0001,
            "flipud": 0.0,
            "mosaic": 1.0,
            "mixup": 0.1,
            "copy_paste": 0.1,
            "erasing": 0.4,
            "close_mosaic": 15,
        }
        # Optional params (may not exist in all Ultralytics versions)
        for key, val in [("auto_augment", "randaugment"), ("box", 7.5), ("cls", 0.5), ("dfl", 1.5)]:
            train_kw[key] = val
        train_kw["workers"] = 8

        model = YOLO(base_model)
        print(f"Starting training with run name: {run_name}")
        try:
            results = model.train(**train_kw)
        except TypeError as e:
            if "unexpected keyword" in str(e).lower() or "keyword" in str(e).lower():
                print(f"Note: falling back to minimal train kwargs (some augmentation params not supported): {e}")
                minimal_kw = {
                    "data": data_yaml,
                    "epochs": epochs,
                    "imgsz": img_size,
                    "batch": batch_size,
                    "lr0": lr0,
                    "optimizer": optimizer,
                    "patience": 20,
                    "save": True,
                    "plots": True,
                    "verbose": True,
                    "project": "runs/detect",
                    "name": run_name,
                }
                results = model.train(**minimal_kw)
            else:
                raise

        final_metrics = {
            "final_map50": _get_metric(results, "metrics/mAP50(B)"),
            "final_map50_95": _get_metric(results, "metrics/mAP50-95(B)"),
            "final_precision": _get_metric(results, "metrics/precision(B)"),
            "final_recall": _get_metric(results, "metrics/recall(B)"),
            "final_train_loss": _get_metric(results, "train/box_loss"),
            "final_val_loss": _get_metric(results, "val/box_loss"),
        }
        mlflow_manager.log_metrics(final_metrics)

        train_dir = Path("runs/detect") / run_name
        best_model_path = train_dir / "weights" / "best.pt"
        if best_model_path.exists():
            mlflow_manager.log_model(
                best_model_path,
                model_name="model",
                registered_model_name=registered_model_name,
            )

        for plot in ("results.png", "confusion_matrix.png", "F1_curve.png"):
            if (train_dir / plot).exists():
                mlflow_manager.log_artifacts(train_dir)
                break

        print(f"Training completed. MLflow run: {mlflow_manager.run.info.run_id}")
        print(f"Results: {final_metrics}")
        return results, mlflow_manager.run.info.run_id

    finally:
        mlflow_manager.end_run()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train YOLO with MLflow")
    p.add_argument("data_yaml", nargs="?", default="combined_yolo_dataset/data.yaml", help="Path to data.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--name", default=None, help="MLflow run name")
    p.add_argument("--tracking-uri", default="http://localhost:5000", help="MLflow tracking URI")
    p.add_argument("--resume", default=None, metavar="PATH", help="Continue from checkpoint (e.g. models/weights/best.pt). Default: yolo12n.pt")
    args = p.parse_args()

    base_model = args.resume if args.resume else "yolo12n.pt"
    train_model_with_tracking(
        data_yaml=args.data_yaml,
        base_model=base_model,
        epochs=args.epochs,
        batch_size=args.batch,
        run_name=args.name,
        tracking_uri=args.tracking_uri,
    )
