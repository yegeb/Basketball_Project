import supervision as sv
import numpy as np
from typing import List, Any


class CourtKeypointAnnotator:
    """
    Annotates a list of video frames with court keypoints using supervision library.

    Attributes:
        keypoint_color (str): Hex code for the color used to annotate keypoints and their labels.
    """

    def __init__(self):
        # Set the color for keypoints and labels (red)
        self.keypoint_color = "#ff2c2c"

    def draw_keypoints(self, frames: List[np.ndarray], court_keypoints: List[Any]) -> List[np.ndarray]:
        """
        Annotates each frame in the list with keypoints and their labels.

        Args:
            frames (List[np.ndarray]): List of video frames as NumPy arrays (BGR format).
            court_keypoints (List[Any]): List of keypoint data for each frame. Each item should be a tensor or array-like of shape (N, 2).

        Returns:
            List[np.ndarray]: List of annotated frames.
        """

        # Initialize vertex (dot) annotator with specified color and size
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            radius=6
        )

        # Initialize label annotator for showing keypoint indices or coordinates
        vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1
        )

        out_frames = []  # List to store annotated frames

        for frame_num, frame in enumerate(frames):
            # Create a copy to avoid modifying the original frame
            annotate_frame = frame.copy()

            # Retrieve keypoints for the current frame
            keypoints = court_keypoints[frame_num]

            # Draw the keypoints on the frame
            annotate_frame = vertex_annotator.annotate(
                scene=annotate_frame,
                key_points=keypoints
            )

            # Convert PyTorch tensor to NumPy array if needed
            keypoints_np = keypoints.cpu().numpy()

            # Draw labels for the keypoints (e.g., index or coordinates)
            annotate_frame = vertex_label_annotator.annotate(
                scene=annotate_frame,
                key_points=keypoints_np
            )

            # Append the annotated frame to the output list
            out_frames.append(annotate_frame)

        return out_frames