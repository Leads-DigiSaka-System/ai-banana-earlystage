# Run retraining from feedback: export dataset then train with MLflow.
# Run: python -m scripts.run_retrain [--min-samples 10] [--output dir] [--epochs 50] [--dry-run]
# Optional: --base-dataset path/to/combined_yolo_dataset to merge feedback into base and train on combined.
# Schedule via Task Scheduler (Windows) or cron (Linux) once feedback is flowing.

import argparse
import os
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _merge_export_into_base(export_dir: Path, base_dir: Path) -> Path:
    """Copy exported feedback images/labels into base dataset train and val. Returns base data.yaml path."""
    base_dir = Path(base_dir)
    export_dir = Path(export_dir)
    # Base may use 'valid' (notebook) or 'val'; map our 'val' to whichever exists
    val_dst = "valid" if (base_dir / "valid" / "images").exists() else "val"
    for split_export, split_base in [("train", "train"), ("val", val_dst)]:
        src_im = export_dir / "images" / split_export
        src_lb = export_dir / "labels" / split_export
        if not src_im.exists():
            continue
        dst_im = base_dir / split_base / "images"
        dst_lb = base_dir / split_base / "labels"
        dst_im.mkdir(parents=True, exist_ok=True)
        dst_lb.mkdir(parents=True, exist_ok=True)
        for f in src_im.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                shutil.copy2(f, dst_im / f.name)
        for f in src_lb.iterdir():
            if f.suffix == ".txt":
                shutil.copy2(f, dst_lb / f.name)
    data_yaml = base_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Base dataset must have data.yaml: {data_yaml}")
    print(f"Merged feedback into {base_dir}; using {data_yaml}")
    return data_yaml


def _count_feedback_training_data() -> int:
    from database.connection import _get_session_factory
    from database.models import TrainingData

    SessionLocal = _get_session_factory()
    db = SessionLocal()
    try:
        return db.query(TrainingData).filter(TrainingData.source == "feedback").count()
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export feedback dataset and run YOLO training with MLflow."
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum training_data (feedback) count to run (default: 10)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="exported_feedback_dataset",
        help="Export output directory (default: exported_feedback_dataset)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio for export (default: 0.2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI or http://localhost:5000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check count and exit without exporting or training",
    )
    parser.add_argument(
        "--base-dataset",
        default=None,
        help="Path to base YOLO dataset (e.g. combined_yolo_dataset). If set, feedback is merged into it and training uses merged data.",
    )
    parser.add_argument(
        "--tile",
        action="store_true",
        help="Tile feedback images (4x4, 256x256) before export, matching notebook data prep.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help="Continue training from this checkpoint (e.g. models/weights/best.pt or runs/detect/exp/weights/last.pt). Default: use models/weights/best.pt if it exists, else yolo12n.pt.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Ignore existing checkpoint; train from pretrained yolo12n.pt (first-time training).",
    )
    args = parser.parse_args()

    count = _count_feedback_training_data()
    print(f"Training_data (source=feedback) count: {count}")

    if count < args.min_samples:
        print(
            f"Below threshold ({args.min_samples}). Skipping retrain. "
            "Submit more wrong-prediction feedback or use --min-samples."
        )
        return 0

    if args.dry_run:
        print("Dry run: would export and train. Exiting.")
        return 0

    from scripts.export_feedback_for_training import export_feedback_for_training

    data_yaml_path, n_train, n_val = export_feedback_for_training(
        output_dir=args.output,
        val_ratio=args.val_ratio,
        project_root=_ROOT,
        use_tile=args.tile,
    )

    if args.base_dataset:
        base_path = Path(args.base_dataset)
        if not base_path.is_absolute():
            base_path = _ROOT / base_path
        data_yaml_path = _merge_export_into_base(Path(args.output), base_path)

    # Continue from current model (best.pt) if it exists, unless --from-scratch or --resume
    base_model = "yolo12n.pt"
    if args.from_scratch:
        base_model = "yolo12n.pt"
    elif args.resume:
        base_model = str(Path(args.resume).resolve())
    else:
        try:
            from config import MODEL_PATH
            checkpoint = Path(MODEL_PATH)
            if checkpoint.exists():
                base_model = str(checkpoint.resolve())
                print(f"Continuing from checkpoint: {base_model}")
        except Exception:
            pass

    from training.train_with_mlflow import train_model_with_tracking

    data_yaml_resolved = Path(data_yaml_path).resolve()
    train_model_with_tracking(
        data_yaml=str(data_yaml_resolved),
        base_model=base_model,
        epochs=args.epochs,
        batch_size=args.batch,
        tracking_uri=args.tracking_uri,
    )
    print("Retrain pipeline finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
