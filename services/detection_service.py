"""
Main detection service - Simple classification only
"""
from typing import Dict, Optional

from services.validation import validate_image
from services.image_processing import preprocess_image
from services.inference import run_inference


class _BytesUpload:
    """File-like so we can run process_image from in-memory bytes (e.g. after read for feedback save)."""
    def __init__(self, data: bytes, filename: str = "image.jpg"):
        self._data = data
        self.filename = filename

    async def read(self) -> bytes:
        return self._data


async def process_image(file, user_id: Optional[str] = None) -> Dict:
    """
    Simple classification - Run inference once and return detections
    
    Args:
        file: FastAPI UploadFile object
        user_id: Optional user identifier for tracking
        
    Returns:
        Dict: Simple response with detections and user_id
    """
    # Validate image
    is_valid, error_msg = await validate_image(file)
    if not is_valid:
        return {
            "success": False,
            "error": error_msg,
            "user_id": user_id
        }
    
    try:
        # Preprocess image
        image = await preprocess_image(file)
        image_shape = image.shape[:2]  # (height, width)
        
        # Run inference ONCE with good threshold
        detections = run_inference(image)
        
        # Simple response with user_id
        return {
            "success": True,
            "user_id": user_id,
            "image_size": {
                "width": int(image_shape[1]),
                "height": int(image_shape[0])
            },
            "detections": detections
        }
        
    except ValueError as e:
        return {
            "success": False,
            "error": str(e),
            "user_id": user_id
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Processing error: {str(e)}",
            "user_id": user_id
        }


async def process_image_from_bytes(
    image_bytes: bytes, filename: str = "image.jpg", user_id: Optional[str] = None
) -> Dict:
    """Same as process_image but takes bytes (so caller can reuse bytes e.g. for saving)."""
    return await process_image(_BytesUpload(image_bytes, filename), user_id=user_id)

