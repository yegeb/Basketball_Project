import cv2
import numpy as np
from typing import List, Dict, Any

black = (0, 0, 0)

class SpeedAndDistAnnotator:
    """
    Annotates video frames with per-player speed and total distance covered.

    Methods:
        draw(video_frames, player_tracks, player_dist_per_frame, player_speed_per_frame):
            Overlays speed and distance info on each frame for visible players.
    """

    def __init__(self):
        # No initialization needed at this point
        pass

    def draw(self, video_frames: List[np.ndarray],
             player_tracks: List[Dict[int, Dict[str, Any]]], 
             player_dist_per_frame: List[Dict[int, float]], 
             player_speed_per_frame: List[Dict[int, float]]) -> List[np.ndarray]:
        """
        Draws speed and cumulative distance annotations on each frame.

        Args:
            video_frames (List[np.ndarray]): List of frames from the video.
            player_tracks (List[Dict[int, Dict[str, Any]]]): Bounding boxes per player per frame.
            player_dist_per_frame (List[Dict[int, float]]): Frame-wise distances traveled by players.
            player_speed_per_frame (List[Dict[int, float]]): Frame-wise speed of players.

        Returns:
            List[np.ndarray]: Annotated video frames with text overlays.
        """

        out_video_frames = []
        total_dist = {}  # Store cumulative distance for each player

        # Loop over each frame with its corresponding tracking/speed/distance data
        for frame, player_tracks_per_frame, player_distance, player_speed in zip(video_frames, player_tracks, player_dist_per_frame, player_speed_per_frame):
            out_frame = frame.copy()  # Avoid modifying the original frame

            # Accumulate total distance traveled per player
            for player_id, distance in player_distance.items():
                if player_id not in total_dist:
                    total_dist[player_id] = 0
                total_dist[player_id] += distance

            # Draw annotations for each visible player
            for player_id, bbox in player_tracks_per_frame.items():
                x1, y1, x2, y2 = bbox["bbox"]

                # Calculate display position slightly below the bounding box
                position = [int((x1 + x2) / 2), int(y2)]
                position[1] += 40 # 40 px buffer to down

                # Fetch current speed and total distance
                distance = total_dist.get(player_id, None)
                speed = player_speed.get(player_id, None)

                # Draw speed in m/s
                if speed is not None:
                    cv2.putText(out_frame, f"{speed: .2f} m/s", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)

                # Draw cumulative distance in meters
                if distance is not None:
                    cv2.putText(out_frame, f"{distance: .2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)

            out_video_frames.append(out_frame)

        return out_video_frames