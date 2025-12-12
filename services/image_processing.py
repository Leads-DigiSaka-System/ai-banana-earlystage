"""
Image preprocessing utilities
"""
import cv2
import numpy as np
from typing import List, Tuple
from config import MAX_IMAGE_DIMENSION, MAX_IMAGE_MEMORY_MB, USE_TILING, TILE_SIZE, TILE_OVERLAP


async def preprocess_image(file) -> np.ndarray:
    """
    Preprocess uploaded image for inference
    
    Note: YOLO will automatically resize images during inference
    based on MODEL_IMAGE_SIZE in config.py
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        np.ndarray: Preprocessed image (RGB format for YOLO)
        
    Raises:
        ValueError: If image cannot be decoded or exceeds size limits
    """
    # Read image bytes
    image_bytes = await file.read()
    
    # Check file size (safety limit)
    file_size_mb = len(image_bytes) / (1024 * 1024)
    if file_size_mb > MAX_IMAGE_MEMORY_MB:
        raise ValueError(
            f"Image file too large: {file_size_mb:.2f}MB. "
            f"Maximum allowed: {MAX_IMAGE_MEMORY_MB}MB"
        )
    
    # Convert to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode image. Please check file format.")
    
    # Validate image dimensions (safety limit)
    height, width = image.shape[:2]
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Image dimensions too large: {width}x{height}. "
            f"Maximum dimension: {MAX_IMAGE_DIMENSION}px"
        )
    
    # Check if image is empty
    if width == 0 or height == 0:
        raise ValueError("Image has invalid dimensions (width or height is 0)")
    
    # Convert BGR to RGB (YOLO expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image_rgb


def tile_image(image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Tile image into smaller patches to match training format
    
    Args:
        image: Input image (RGB numpy array)
        
    Returns:
        List of (tile, (x, y)) tuples where (x, y) is the top-left position
    """
    h, w = image.shape[:2]
    tiles = []
    step = int(TILE_SIZE * (1 - TILE_OVERLAP))  # Step size with overlap
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Extract tile
            y_end = min(y + TILE_SIZE, h)
            x_end = min(x + TILE_SIZE, w)
            
            tile = image[y:y_end, x:x_end]
            
            # Only add complete tiles (256x256)
            if tile.shape[0] == TILE_SIZE and tile.shape[1] == TILE_SIZE:
                tiles.append((tile, (x, y)))
            elif tile.shape[0] >= TILE_SIZE // 2 and tile.shape[1] >= TILE_SIZE // 2:
                # Pad incomplete tiles at edges
                padded_tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=image.dtype)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tiles.append((padded_tile, (x, y)))
    
    return tiles

