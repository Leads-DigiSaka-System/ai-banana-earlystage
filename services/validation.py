"""
Image validation utilities
"""
from pathlib import Path
from typing import Tuple
from config import ALLOWED_EXTENSIONS


async def validate_image(file) -> Tuple[bool, str]:
    """
    Validate uploaded image file
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    return True, ""

