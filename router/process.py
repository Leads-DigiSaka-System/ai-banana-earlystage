"""
API Router for Banana Disease Detection endpoints
"""
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
        try:
            best = max(result["detections"], key=lambda d: d["confidence"])
            prediction_id = save_prediction(
                db,
                image_bytes,
                file.filename or "image.jpg",
                user_id,
                best,
                inference_time_ms=None,
                user_location=user_location,
            )
            result["prediction_id"] = prediction_id
        except Exception:
            pass  # DB save optional; response still has detections

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
        out = {"user_id": user_id_result, "class_name": "No detection", "confidence": 0.0}
        return JSONResponse(content=out)

    class_detections = {}
    for det in detections:
        c, conf = det["class_name"], det["confidence"]
        class_detections.setdefault(c, []).append(conf)
    class_max_confidences = {}
    for class_name, confidences in class_detections.items():
        max_c = max(confidences)
        n = len(confidences)
        class_max_confidences[class_name] = {
            "confidence": max_c,
            "count": n,
            "weighted_score": max_c * (1 + 0.05 * min(n, 5)),
        }
    best_class = max(class_max_confidences.items(), key=lambda x: x[1]["weighted_score"])
    classification = {
        "user_id": user_id_result,
        "class_name": best_class[0],
        "confidence": float(best_class[1]["confidence"]),
    }

    if user_id and detections:
        try:
            best_det = max(detections, key=lambda d: d["confidence"])
            prediction_id = save_prediction(
                db,
                image_bytes,
                file.filename or "image.jpg",
                user_id,
                best_det,
                inference_time_ms=None,
                user_location=user_location,
            )
            classification["prediction_id"] = prediction_id
        except Exception:
            pass

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
