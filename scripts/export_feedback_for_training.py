# Export feedback (training_data from wrong predictions) to YOLO dataset.
# Run: python -m scripts.export_feedback_for_training [--output dir] [--val-ratio 0.2]
# Reads app Postgres + MinIO/local images; writes images/ and labels/ + data.yaml.

import argparse
import hashlib
import sys
from pathlib import Path

# Project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from database.connection import _get_session_factory
from database.models import TrainingData
from services.storage_service import read_image_bytes

# Fixed 7 classes — must match config.CLASS_NAMES and app inference (0=Healthy, 1=Stage1, ... 6=Stage6)
def _get_fixed_class_config():
    from config import CLASS_NAMES
    names = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
    name_to_id = {n: i for i, n in enumerate(names)}
    return names, name_to_id, len(names)


def _get_image_size(image_bytes: bytes) -> tuple[int, int]:
    """Return (width, height). Prefer PIL, fallback to minimal header read."""
    try:
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(image_bytes))
        return (im.width, im.height)
    except Exception:
        pass
    try:
        import struct
        # JPEG: SOF0 marker then width (2 bytes) height (2 bytes)
        b = image_bytes[:2]
        i = 0
        while i < len(image_bytes) - 1:
            if image_bytes[i : i + 2] == b"\xff\xc0":
                if i + 9 <= len(image_bytes):
                    h, w = struct.unpack(">HH", image_bytes[i + 5 : i + 9])
                    return (w, h)
                break
            i += 1
    except Exception:
        pass
    return (736, 736)  # fallback


def _tile_image(
    image_bytes: bytes,
    tile_size: int = 256,
    grid_cols: int = 4,
    grid_rows: int = 4,
) -> list[tuple[bytes, str]]:
    """Split image into grid tiles and resize each to tile_size. Returns list of (tile_bytes, suffix) e.g. ('_tile_0_0', ...)."""
    import io
    from PIL import Image

    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = im.size
    if w < grid_cols or h < grid_rows:
        return [(image_bytes, "")]
    tw, th = w / grid_cols, h / grid_rows
    out = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x1 = int(col * tw)
            y1 = int(row * th)
            x2 = int((col + 1) * tw)
            y2 = int((row + 1) * th)
            tile = im.crop((x1, y1, x2, y2))
            tile = tile.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            tile.save(buf, format="JPEG", quality=95)
            out.append((buf.getvalue(), f"_tile_{row}_{col}"))
    return out


def _bbox_xyxy_to_yolo_norm(
    x1: float, y1: float, x2: float, y2: float,
    img_width: int, img_height: int,
) -> tuple[float, float, float, float]:
    """Convert xyxy pixel coords to YOLO normalized x_center y_center width height."""
    if img_width <= 0 or img_height <= 0:
        return (0.5, 0.5, 1.0, 1.0)
    xc = (x1 + x2) / 2.0 / img_width
    yc = (y1 + y2) / 2.0 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    # Clamp to [0, 1]
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return (xc, yc, w, h)


def export_feedback_for_training(
    output_dir: str | Path,
    val_ratio: float = 0.2,
    source_filter: str = "feedback",
    project_root: Path | None = None,
    use_tile: bool = False,
    tile_size: int = 256,
    grid_cols: int = 4,
    grid_rows: int = 4,
) -> tuple[Path, int, int]:
    """
    Export training_data rows to a YOLO dataset under output_dir.
    Creates images/train, images/val, labels/train, labels/val and data.yaml.
    If use_tile=True, each image is split into grid_cols x grid_rows tiles (tile_size x tile_size), matching notebook data prep.
    Returns (path_to_data_yaml, num_train, num_val).
    """
    output_dir = Path(output_dir)
    project_root = project_root or _ROOT

    SessionLocal = _get_session_factory()
    db = SessionLocal()
    try:
        rows = (
            db.query(TrainingData)
            .filter(TrainingData.source == source_filter)
            .order_by(TrainingData.added_date)
            .all()
        )
    finally:
        db.close()

    if not rows:
        raise SystemExit("No training_data rows with source='feedback'. Submit wrong-prediction feedback first.")

    # Use fixed 7 classes (Healthy, Stage1..Stage6) so trained model matches app inference
    class_names, name_to_id, nc = _get_fixed_class_config()

    # Output layout
    train_images = output_dir / "images" / "train"
    train_labels = output_dir / "labels" / "train"
    val_images = output_dir / "images" / "val"
    val_labels = output_dir / "labels" / "val"
    for d in (train_images, train_labels, val_images, val_labels):
        d.mkdir(parents=True, exist_ok=True)

    n_train, n_val = 0, 0
    seen_hashes = set()

    for i, r in enumerate(rows):
        if r.image_hash and r.image_hash in seen_hashes:
            continue
        if r.image_hash:
            seen_hashes.add(r.image_hash)

        try:
            image_bytes = read_image_bytes(r.image_path, project_root=project_root)
        except FileNotFoundError as e:
            print(f"Warning: image not found for {r.image_path}: {e}", file=sys.stderr)
            continue
        except ValueError as e:
            print(f"Warning: cannot read image {r.image_path}: {e}", file=sys.stderr)
            continue

        img_w, img_h = _get_image_size(image_bytes)
        class_id = name_to_id.get(r.class_name, 0)

        # Train/val split (deterministic by row id)
        bucket = int.from_bytes(r.id.bytes[:4], "big") % 100
        use_val = bucket < int(100 * val_ratio) if val_ratio > 0 else False
        if use_val:
            im_dir, lb_dir = val_images, val_labels
        else:
            im_dir, lb_dir = train_images, train_labels

        base_stem = f"{r.id}_{(r.image_hash or hashlib.sha256(image_bytes).hexdigest())[:8]}"
        ext = ".jpg"

        if use_tile:
            # Tile image (notebook-style): 4x4 grid, each tile tile_size x tile_size
            tiles = _tile_image(image_bytes, tile_size=tile_size, grid_cols=grid_cols, grid_rows=grid_rows)
            for tile_bytes, suffix in tiles:
                stem = f"{base_stem}{suffix}"
                image_path = im_dir / f"{stem}{ext}"
                label_path = lb_dir / f"{stem}.txt"
                image_path.write_bytes(tile_bytes)
                label_path.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n")
                if use_val:
                    n_val += 1
                else:
                    n_train += 1
        else:
            if use_val:
                n_val += 1
            else:
                n_train += 1
            image_path = im_dir / f"{base_stem}{ext}"
            label_path = lb_dir / f"{base_stem}.txt"
            image_path.write_bytes(image_bytes)
            bbox = r.bbox_data if isinstance(r.bbox_data, dict) else None
            if bbox and "x1" in bbox and "y1" in bbox and "x2" in bbox and "y2" in bbox:
                x1 = float(bbox["x1"])
                y1 = float(bbox["y1"])
                x2 = float(bbox["x2"])
                y2 = float(bbox["y2"])
                xc, yc, w, h = _bbox_xyxy_to_yolo_norm(x1, y1, x2, y2, img_w, img_h)
                line = f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
            else:
                line = f"{class_id} 0.5 0.5 1.0 1.0\n"
            label_path.write_text(line)

    if n_train == 0 and n_val == 0:
        raise SystemExit("No images could be written. Check image_path and MinIO/local access.")

    # data.yaml (paths relative to output_dir so it's portable)
    import yaml
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
        "nc": nc,
    }
    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"Exported {n_train} train, {n_val} val images to {output_dir}")
    print(f"data.yaml: {data_yaml_path}")
    return data_yaml_path, n_train, n_val


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export feedback (training_data) to YOLO dataset for retraining."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="exported_feedback_dataset",
        help="Output directory (default: exported_feedback_dataset)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples for validation (default: 0.2)",
    )
    parser.add_argument(
        "--source",
        default="feedback",
        help="Filter training_data by source (default: feedback)",
    )
    parser.add_argument(
        "--tile",
        action="store_true",
        help="Tile each image into 4x4 grid (256x256 per tile), matching notebook data prep and inference tiling",
    )
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size when --tile (default: 256)")
    parser.add_argument("--grid-cols", type=int, default=4, help="Grid columns when --tile (default: 4)")
    parser.add_argument("--grid-rows", type=int, default=4, help="Grid rows when --tile (default: 4)")
    args = parser.parse_args()

    export_feedback_for_training(
        output_dir=args.output,
        val_ratio=args.val_ratio,
        source_filter=args.source,
        use_tile=args.tile,
        tile_size=args.tile_size,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
    )


if __name__ == "__main__":
    main()
