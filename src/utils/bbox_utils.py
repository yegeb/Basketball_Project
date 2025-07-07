from typing import List, Tuple

def get_center_of_bbox(bbox: List[int]) -> Tuple[int, int]:
    """
    Calculates the center point (x, y) of a bounding box.

    Args:
        bbox (List[int]): Bounding box in the format [x1, y1, x2, y2].

    Returns:
        tuple: Coordinates of the center point (center_x, center_y).
    """
    x1, y1, x2, y2 = bbox  # Extract coordinates from the bounding box

    # Compute and return the center of the box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox: List[int]) -> int:
    """
    Calculates the width of a bounding box.

    Args:
        bbox (List[int]): Bounding box in the format [x1, y1, x2, y2].

    Returns:
        int: Width of the bounding box.
    """
    # Width = right x - left x
    return bbox[2] - bbox[0]


def get_foot_pos(bbox: List[int]) -> Tuple[int]:
    x1, y1, x2, y2 = bbox

    return (int((x1 + x2)/2), y2)