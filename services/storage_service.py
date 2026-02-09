"""
Phase 2: MinIO (S3-compatible) object storage for prediction and training images.
If STORAGE_ENDPOINT is not set, get_client() returns None and callers fall back to local disk.

Flow: Predict (with user_id) → save_prediction → image_path in DB is either
  - local: "data/uploads/predictions/{uuid}.jpg"
  - MinIO: "{bucket}/predictions/{user_id}/{year}/{month}/{uuid}_{hash}.jpg"
Use read_image_bytes(image_path) to load image bytes from either backend.
"""
import io
import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

# Optional MinIO; import only when used so app runs without minio installed if not using storage
try:
    from minio import Minio
    from minio.error import S3Error
    _MINIO_AVAILABLE = True
except ImportError:
    _MINIO_AVAILABLE = False
    Minio = None  # type: ignore
    S3Error = Exception  # type: ignore


# Single bucket for the project (object paths: predictions/..., training-data/...)
def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def get_bucket() -> str:
    """Bucket name (e.g. ai-banana-early-stage). Override with STORAGE_BUCKET env."""
    return _env("STORAGE_BUCKET", "ai-banana-early-stage") or "ai-banana-early-stage"


# Object name prefixes inside the bucket
OBJECT_PREFIX_PREDICTIONS = "predictions"
OBJECT_PREFIX_TRAINING = "training-data"


def is_configured() -> bool:
    """True if MinIO is available and STORAGE_ENDPOINT is set."""
    return bool(_MINIO_AVAILABLE and _env("STORAGE_ENDPOINT"))


def get_client() -> Optional["Minio"]:
    """
    Return MinIO client if STORAGE_ENDPOINT is set and minio is installed; else None.
    Caller should fall back to local storage when None.
    """
    if not _MINIO_AVAILABLE:
        return None
    endpoint = _env("STORAGE_ENDPOINT")
    if not endpoint:
        return None
    access = _env("STORAGE_ACCESS_KEY", "minioadmin")
    secret = _env("STORAGE_SECRET_KEY", "minioadmin123")
    secure = _env("STORAGE_SECURE", "false").lower() in ("1", "true", "yes")
    client = Minio(endpoint, access_key=access, secret_key=secret, secure=secure)
    return client


def ensure_buckets(client: "Minio") -> None:
    """Create the project bucket if it does not exist."""
    bucket = get_bucket()
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
    except S3Error:
        pass


def upload_prediction_image(
    client: "Minio",
    image_data: bytes,
    user_id: str,
    prediction_id: str,
    image_hash_prefix: str,
    file_extension: str = "jpg",
) -> str:
    """
    Upload prediction image to MinIO. Returns path string to store in DB.
    Stored path: {bucket}/predictions/{user_id}/{year}/{month}/{prediction_id}_{hash}.ext
    """
    from datetime import datetime, timezone
    bucket = get_bucket()
    now = datetime.now(timezone.utc)
    object_name = (
        f"{OBJECT_PREFIX_PREDICTIONS}/{user_id}/{now.year}/{now.month:02d}/"
        f"{prediction_id}_{image_hash_prefix[:8]}.{file_extension.lstrip('.')}"
    )
    content_type = f"image/{file_extension.lstrip('.')}"
    if content_type == "image/jpg":
        content_type = "image/jpeg"
    client.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=io.BytesIO(image_data),
        length=len(image_data),
        content_type=content_type,
    )
    return f"{bucket}/{object_name}"


def get_image(client: "Minio", image_path: str) -> bytes:
    """
    Retrieve image bytes from MinIO. image_path format: {bucket}/predictions/... or {bucket}/training-data/...
    """
    parts = image_path.split("/", 1)
    bucket = parts[0]
    object_name = parts[1] if len(parts) > 1 else ""
    if not object_name:
        raise ValueError("Invalid image_path for MinIO")
    response = client.get_object(bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def get_presigned_url(
    client: "Minio",
    image_path: str,
    expires: timedelta = timedelta(hours=1),
) -> str:
    """Get a temporary URL to view/download the image."""
    parts = image_path.split("/", 1)
    bucket = parts[0]
    object_name = parts[1] if len(parts) > 1 else ""
    if not object_name:
        raise ValueError("Invalid image_path for MinIO")
    return client.presigned_get_object(bucket, object_name, expires=expires)


def read_image_bytes(image_path: str, project_root: Optional[Path] = None) -> bytes:
    """
    Load image bytes from either local disk or MinIO (so whole phase is connected).
    - If image_path starts with "data/", read from project_root / image_path (local).
    - Otherwise treat as MinIO path (bucket/object_name) and fetch via get_client/get_image.
    project_root defaults to repo root (parent of services/).
    """
    if image_path.startswith("data/"):
        root = project_root or Path(__file__).resolve().parent.parent
        path = root / image_path
        if not path.exists():
            raise FileNotFoundError(f"Local image not found: {path}")
        return path.read_bytes()
    client = get_client()
    if client is None:
        raise ValueError(
            "image_path looks like MinIO but STORAGE_ENDPOINT is not set; cannot read image"
        )
    return get_image(client, image_path)
