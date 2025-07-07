from typing import List, Dict, Any
import numpy as np
import sys
sys.path.append("../")
from utils import draw_triangle  

class BallPointer:
    def __init__(self):
        # Set the color of the ball pointer (green in BGR)
        self.ball_pointer_color = (0, 255, 0)

    def draw_tracks(self, video_frames: List[np.ndarray], 
                    tracks: List[Dict[int, Dict[str, Any]]]) -> List[np.ndarray]:
        """
        Draws tracked ball pointer (a green arrow) on each video frame.

        Args:
            video_frames (List[np.ndarray]): List of frames from the video.
            tracks (List[Dict[int, Dict[str, Any]]]): 
                A list of dictionaries, one per frame.
                Each dictionary maps a track ID to a ball's data (with a "bbox" key).

        Returns:
            List[np.ndarray]: List of frames with annotations drawn.
        """   
        out_video_frames = []  # Store annotated frames

        for frame_num, frame in enumerate(video_frames):
            out_frame = frame.copy()  # Avoid modifying the original frame

            ball_dict = tracks[frame_num]  # Get all ball detections for this frame

            for _, track in ball_dict.items():
                bbox = track["bbox"]  # Extract the bounding box for the tracked object

                if bbox is None:
                    continue  # Skip if there's no bounding box

                # Draw a triangle pointer for the ball on the frame
                draw_triangle(out_frame, bbox, self.ball_pointer_color)

            out_video_frames.append(out_frame)  # Store the annotated frame

        return out_video_frames