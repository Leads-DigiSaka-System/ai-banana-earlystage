"""
Model inference utilities
"""
from typing import List, Dict, Optional
from ultralytics import YOLO
import numpy as np

from config import (
    MODEL_PATH,
    MODEL_CONFIDENCE,
    MODEL_IOU,
    MODEL_IMAGE_SIZE,
    CLASS_NAMES,
    USE_TILING
)
from services.image_processing import tile_image

# Global model variable (loaded once)
_model: Optional[YOLO] = None


def load_model() -> YOLO:
    """
    Load YOLO model (cached - only loads once)
    
    Returns:
        YOLO: Loaded model instance
    """
    global _model
    
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Please copy your trained best.pt to the models/ folder."
            )
        
        print(f"Loading model from {MODEL_PATH}...")
        _model = YOLO(str(MODEL_PATH))
        print("Model loaded successfully!")
    
    return _model


def run_inference(image: np.ndarray, confidence_threshold: float = None, use_tiling: bool = None) -> List[Dict]:
    """
    Run YOLO inference on image - Simple classification only
    
    Args:
        image: Preprocessed image (RGB numpy array)
        confidence_threshold: Optional confidence threshold (overrides MODEL_CONFIDENCE)
        use_tiling: If True, tiles image before inference (matches training format)
        
    Returns:
        List[Dict]: List of detection results
    """
    # Load model
    model = load_model()
    
    # Use provided threshold or default
    conf_threshold = confidence_threshold if confidence_threshold is not None else MODEL_CONFIDENCE
    
    # Determine if we should use tiling
    should_tile = use_tiling if use_tiling is not None else USE_TILING
    
    # If tiling is enabled and image is large, tile it
    if should_tile and (image.shape[0] > 256 or image.shape[1] > 256):
        tiles = tile_image(image)
        all_detections = []
        
        # Run inference on each tile
        for tile, (x_offset, y_offset) in tiles:
            results = model.predict(
                tile,
                conf=conf_threshold,
                iou=MODEL_IOU,
                imgsz=MODEL_IMAGE_SIZE,
                verbose=False
            )
            
            if len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    # Adjust bounding boxes to original image coordinates
                    for box, cls_id, conf in zip(boxes, classes, confidences):
                        detection = {
                            "class_id": int(cls_id),
                            "class_name": CLASS_NAMES.get(int(cls_id), f"Unknown_{cls_id}"),
                            "confidence": float(conf),
                            "bbox": {
                                "x1": float(box[0] + x_offset),
                                "y1": float(box[1] + y_offset),
                                "x2": float(box[2] + x_offset),
                                "y2": float(box[3] + y_offset)
                            }
                        }
                        all_detections.append(detection)
        
        return all_detections
    else:
        # Run inference on full image (original method)
        results = model.predict(
            image,
            conf=conf_threshold,
            iou=MODEL_IOU,
            imgsz=MODEL_IMAGE_SIZE,
            verbose=False
        )
        
        # Process results
        detections = []
        
        if len(results) > 0:
            result = results[0]  # First (and only) result
            
            # Get boxes, classes, and confidences
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                classes = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                
                # Format detections
                for box, cls_id, conf in zip(boxes, classes, confidences):
                    detection = {
                        "class_id": int(cls_id),
                        "class_name": CLASS_NAMES.get(int(cls_id), f"Unknown_{cls_id}"),
                        "confidence": float(conf),
                        "bbox": {
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3])
                        }
                    }
                    detections.append(detection)
        
        return detections

