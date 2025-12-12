"""
API Router for Banana Disease Detection endpoints
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional

from services.detection_service import process_image
from services.inference import load_model

# Create router
router = APIRouter(prefix="/api/v1", tags=["detection"])


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None)
):
    """
    Main prediction endpoint - Full detection with bounding boxes
    
    Args:
        file: Image file (JPG, PNG, JPEG)
        user_id: Optional user identifier for tracking
        
    Returns:
        JSON response with detections and user_id
    """
    # Process image
    result = await process_image(file, user_id=user_id)
    
    # Check if processing was successful
    if not result.get("success", False):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Unknown error occurred")
        )
    
    return JSONResponse(content=result)


@router.post("/predict/classify")
async def predict_classify(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None)
):
    """
    Classification-only endpoint - Returns class name and confidence only
    No bounding boxes needed - just classifies the image
    
    Args:
        file: Image file (JPG, PNG, JPEG)
        user_id: Optional user identifier for tracking
        
    Returns:
        JSON response with classification and user_id (class_name, confidence, user_id)
    """
    # Process image
    result = await process_image(file, user_id=user_id)
    
    # Check if processing was successful
    if not result.get("success", False):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Unknown error occurred")
        )
    
    # Extract only classification info (no bounding boxes)
    detections = result.get("detections", [])
    user_id_result = result.get("user_id", user_id)
    
    if not detections:
        # No detections found
        return JSONResponse(content={
            "user_id": user_id_result,
            "class_name": "No detection",
            "confidence": 0.0
        })
    
    # Aggregate detections by class for better classification accuracy
    if detections:
        # Group by class: collect all confidences per class
        class_detections = {}
        for det in detections:
            class_name = det["class_name"]
            confidence = det["confidence"]
            
            if class_name not in class_detections:
                class_detections[class_name] = []
            class_detections[class_name].append(confidence)
        
        # For classification: Use MAX confidence per class (not average)
        # Reason: When tiling, each tile might have lower confidence
        # Using MAX captures the best detection per class
        class_max_confidences = {}
        for class_name, confidences in class_detections.items():
            max_confidence = max(confidences)
            count = len(confidences)
            # Weighted score: max confidence with small boost for multiple detections
            # This favors classes detected in multiple tiles
            weighted_score = max_confidence * (1 + 0.05 * min(count, 5))  # Cap boost at 5 detections
            class_max_confidences[class_name] = {
                "confidence": max_confidence,
                "count": count,
                "weighted_score": weighted_score
            }
        
        # Get class with highest weighted score
        best_class = max(class_max_confidences.items(), key=lambda x: x[1]["weighted_score"])
        
        classification = {
            "user_id": user_id_result,
            "class_name": best_class[0],
            "confidence": float(best_class[1]["confidence"])
        }
    else:
        classification = {
            "user_id": user_id_result,
            "class_name": "No detection",
            "confidence": 0.0
        }
    
    # Return clean, simple JSON response with user_id
    return JSONResponse(content=classification)


@router.post("/predict/classify/debug")
async def predict_classify_debug(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None)
):
    """
    Debug endpoint - Shows all detections with very low threshold to diagnose issues
    
    Args:
        file: Image file (JPG, PNG, JPEG)
        user_id: Optional user identifier for tracking
        
    Returns:
        JSON response with all detections (even low confidence) and user_id
    """
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
