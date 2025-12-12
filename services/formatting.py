"""
Response formatting utilities
"""
from typing import List, Dict, Tuple


def format_response(image_shape: Tuple[int, int], detections: List[Dict]) -> Dict:
    """
    Format inference results into API response
    
    Args:
        image_shape: (height, width) of original image
        detections: List of detection dictionaries
        
    Returns:
        Dict: Formatted response
    """
    # Get unique classes found
    classes_found = list(set([det["class_name"] for det in detections]))
    
    # Count detections per class
    class_counts = {}
    for det in detections:
        class_name = det["class_name"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    response = {
        "success": True,
        "image_size": {
            "width": int(image_shape[1]),
            "height": int(image_shape[0])
        },
        "detections": detections,
        "summary": {
            "total_detections": len(detections),
            "classes_found": classes_found,
            "class_counts": class_counts
        }
    }
    
    return response

