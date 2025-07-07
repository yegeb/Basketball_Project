import cv2 
import numpy as np
from typing import List, Tuple
from .bbox_utils import get_center_of_bbox, get_bbox_width

def draw_ellipse(frame: np.ndarray, bbox: List[int], color: Tuple[int, int, int], track_id: int = None) -> None:
    """
    Draws an ellipse and an optional ID tag (track_id) on a given frame based on a bounding box.

    Args:
        frame (np.ndarray): The video frame to draw on.
        bbox (List[int]): Bounding box in the format [x1, y1, x2, y2].
        color (tuple): Color of the ellipse and ID tag (in BGR).
        track_id (int, optional): Unique ID to label the object (e.g., a player). Default is None.

    Returns:
        None: This function modifies the frame in-place.
    """
    # Get the bottom y-coordinate of the box
    y2 = int(bbox[3])

    # Get the center x-coordinate of the box (horizontal midpoint)
    x_center, _ = get_center_of_bbox(bbox)

    # Calculate the width of the bounding box
    width = get_bbox_width(bbox)

    # Draw an elliptical base under the player
    cv2.ellipse(img=frame, 
                center=(x_center, y2),  # Centered horizontally at bottom of bbox
                axes=(int(width), int(0.35 * width)),  # Ellipse shape based on box width
                angle=0,
                startAngle=-45,  # Start and end angles create a partial ellipse
                endAngle=235,
                color=color,
                thickness=2,
                lineType=cv2.LINE_4)

    # Dimensions for the rectangle tag showing the track ID
    rectangle_width  = 40
    rectangle_height = 20

    # Coordinates for the rectangle centered on x_center, slightly below the bbox
    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = (y2 - rectangle_height // 2) + 15
    y2_rect = (y2 + rectangle_height // 2) + 15

    # If a track ID is provided, draw the filled rectangle and text
    if track_id is not None:
        # Draw the filled rectangle (used as a label background)
        cv2.rectangle(img=frame,
                      pt1=(int(x1_rect), int(y1_rect)), 
                      pt2=(int(x2_rect), int(y2_rect)),
                      color=color,
                      thickness=cv2.FILLED)

        # Adjust text x-offset to center longer numbers (3-digit IDs)
        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10

        # Draw the track ID above the rectangle
        cv2.putText(img=frame,
                    text=str(track_id),
                    org=(int(x1_text), int(y1_rect + 15)),  # Position slightly above rectangle
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 0, 0))  # Black text

def draw_triangle(frame: np.ndarray, bbox: List[int], color: Tuple[int, int, int]) -> None:
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)

    triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]], dtype=np.int32).reshape((-1, 1, 2))

    cv2.drawContours(image=frame, 
                     contours=[triangle_points],
                     contourIdx=0, 
                     color=color, 
                     thickness=cv2.FILLED)
    
    cv2.drawContours(image=frame, 
                     contours=[triangle_points],
                     contourIdx=0, 
                     color=(0, 0, 0), 
                     thickness=2)