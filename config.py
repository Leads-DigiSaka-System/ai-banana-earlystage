"""
Configuration file for Banana Disease Detection API
"""
from pathlib import Path

# Model Configuration
MODEL_PATH = Path("models/weights/best.pt")  # Update this path to your trained model
MODEL_CONFIDENCE = 0.25  # Confidence threshold (0.0 - 1.0) - Lowered to catch more disease detections
MODEL_IOU = 0.7  # IoU threshold for NMS
MODEL_IMAGE_SIZE = 736  # Image size for inference - MUST match training size (imgsz=736) for best accuracy
# Note: YOLO automatically resizes any uploaded image to this size during inference
# Training: 256x256 tiles were resized to 736x736 during training
# Inference: Any size image (mobile, camera, etc.) will be resized to 736x736

# Tiling Configuration (CRITICAL for accuracy!)
USE_TILING = True  # Tile images before inference to match training format (256x256 tiles)
TILE_SIZE = 256  # Match training tile size
TILE_OVERLAP = 0.1  # 10% overlap between tiles (reduces edge artifacts)

# Classification Configuration (Simplified - no re-classification)
# Removed: Re-classification logic was diluting confidence scores

# Class Names Mapping (from training)
CLASS_NAMES = {
    0: "Healthy",
    1: "Stage1",
    2: "Stage2",
    3: "Stage3",
    4: "Stage4",
    5: "Stage5",
    6: "Stage6"
}

# API Configuration
# Note: YOLO will automatically resize images during inference
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_DIMENSION = 10000  # Maximum width or height in pixels (safety limit)
MAX_IMAGE_MEMORY_MB = 50  # Maximum image size in MB (safety limit)

# Server Configuration
HOST = "0.0.0.0"
PORT = 8000

# Phase 2 (MinIO): configured via env (STORAGE_ENDPOINT, STORAGE_ACCESS_KEY, STORAGE_SECRET_KEY).
# See services/storage_service.py and .env.example.

