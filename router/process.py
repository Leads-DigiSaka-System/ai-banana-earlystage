"""
API Router for Banana Disease Detection endpoints
"""
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from database.connection import get_db
from services.detection_service import process_image_from_bytes
from services.feedback_service import save_prediction
from services.inference import load_model

# Create router
router = APIRouter(prefix="/api/v1", tags=["detection"])


@router.post(
    "/predict",
    summary="Full detection with bounding boxes",
    description="""
**How to use:** Send the image as **form-data** (multipart).

- **file** (required): Image file — JPG, PNG, or JPEG.
- **user_id** (optional): If you send this, the prediction is saved and the response includes **prediction_id**. Use that ID later to submit feedback (POST /api/v1/feedback/submit).
- **user_location** (optional): e.g. city or region.

**Response:** Detections with class_name, confidence, and bbox. If you sent `user_id`, you also get `prediction_id` — copy it for the feedback API.

**Example (curl):**
```
curl -X POST "http://localhost:8000/api/v1/predict" \\
  -F "file=@leaf.jpg" \\
  -F "user_id=my_app_user_123"
```
    """,
)
async def predict(
    file: UploadFile = File(..., description="Image file (JPG, PNG, JPEG)"),
    user_id: Optional[str] = Form(None, description="Optional. If set, prediction is saved and response includes prediction_id for feedback."),
    user_location: Optional[str] = Form(None, description="Optional location (e.g. city)."),
    db: Session = Depends(get_db),
):
    image_bytes = await file.read()
    result = await process_image_from_bytes(
        image_bytes, filename=file.filename or "image.jpg", user_id=user_id
    )
    if not result.get("success", False):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Unknown error occurred"),
        )

    if user_id and result.get("detections"):
        prediction_id = str(uuid.uuid4())
        try:
            best = max(result["detections"], key=lambda d: d["confidence"])
            save_prediction(
                db,
                image_bytes,
                file.filename or "image.jpg",
                user_id,
                best,
                inference_time_ms=None,
                user_location=user_location,
                prediction_id=prediction_id,
            )
            result["prediction_id"] = prediction_id
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to save prediction for feedback: %s", e)

    return JSONResponse(content=result)


@router.post(
    "/predict/classify",
    summary="Classification only (no bounding boxes)",
    description="""
**How to use:** Send the image as **form-data** (multipart).

- **file** (required): Image file — JPG, PNG, or JPEG.
- **user_id** (optional): If you send this, the prediction is saved and the response includes **prediction_id** for feedback.
- **user_location** (optional): e.g. city or region.

**Response:** Single class (class_name, confidence). If you sent `user_id`, you also get `prediction_id` — use it when calling POST /api/v1/feedback/submit.

**Example (curl):**
```
curl -X POST "http://localhost:8000/api/v1/predict/classify" \\
  -F "file=@leaf.jpg" \\
  -F "user_id=my_app_user_123"
```
    """,
)
async def predict_classify(
    file: UploadFile = File(..., description="Image file (JPG, PNG, JPEG)"),
    user_id: Optional[str] = Form(None, description="Optional. If set, response includes prediction_id for feedback."),
    user_location: Optional[str] = Form(None, description="Optional location."),
    db: Session = Depends(get_db),
):
    image_bytes = await file.read()
    result = await process_image_from_bytes(
        image_bytes, filename=file.filename or "image.jpg", user_id=user_id
    )
    if not result.get("success", False):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Unknown error occurred"),
        )

    detections = result.get("detections", [])
    user_id_result = result.get("user_id", user_id)

    if not detections:
        out = {
            "user_id": user_id_result,
            "class_name": "No detection",
            "confidence": 0.0,
            "reliable": False,
        }
        return JSONResponse(content=out)

    # Use the single detection with highest confidence as the classification result.
    # This avoids many weak detections (e.g. Stage5) outweighing one strong detection (e.g. Stage1).
    best_det = max(detections, key=lambda d: d["confidence"])
    best_conf = float(best_det["confidence"])
    # Hint for accuracy: treat as reliable when confidence >= 0.5 (configurable threshold)
    CONFIDENCE_RELIABLE_THRESHOLD = 0.5
    classification = {
        "user_id": user_id_result,
        "class_name": best_det["class_name"],
        "confidence": best_conf,
        "reliable": best_conf >= CONFIDENCE_RELIABLE_THRESHOLD,
    }

    if user_id and detections:
        prediction_id = str(uuid.uuid4())
        try:
            save_prediction(
                db,
                image_bytes,
                file.filename or "image.jpg",
                user_id,
                best_det,
                inference_time_ms=None,
                user_location=user_location,
                prediction_id=prediction_id,
            )
            classification["prediction_id"] = prediction_id
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to save prediction for feedback: %s", e)

    return JSONResponse(content=classification)


@router.post(
    "/predict/classify/debug",
    summary="Debug: all detections (low threshold)",
    description="""
**How to use:** Same as /predict/classify — form-data with **file** and optional **user_id**.

Returns **all** detections with a very low confidence threshold (0.01) so you can see weak detections. Use for debugging only, not for production.
    """,
)
async def predict_classify_debug(
    file: UploadFile = File(..., description="Image file (JPG, PNG, JPEG)"),
    user_id: Optional[str] = Form(None, description="Optional user identifier."),
):
    from services.image_processing import preprocess_image
    from services.inference import run_inference
    
    try:
        # Preprocess image
        image = await preprocess_image(file)
        
        # Run inference with very low threshold to see ALL detections
        detections = run_inference(image, confidence_threshold=0.01, use_tiling=True)
        
        # Aggregate by class
        class_confidence_map = {}
        for det in detections:
            class_name = det["class_name"]
            confidence = det["confidence"]
            if class_name not in class_confidence_map or confidence > class_confidence_map[class_name]:
                class_confidence_map[class_name] = confidence
        
        # Sort by confidence
        all_classes = sorted(
            class_confidence_map.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return JSONResponse(content={
            "user_id": user_id,
            "all_detections": detections,
            "classes_detected": [
                {"class_name": name, "confidence": float(conf)}
                for name, conf in all_classes
            ],
            "total_detections": len(detections),
            "top_class": all_classes[0][0] if all_classes else "None",
            "top_confidence": float(all_classes[0][1]) if all_classes else 0.0
        })
    except Exception as e:
        return JSONResponse(content={
            "user_id": user_id,
            "error": str(e)
        }, status_code=400)
