from typing import List, Dict, Any, Tuple
import numpy as np
import sys
sys.path.append("../")
from utils import draw_ellipse, draw_triangle

# COLORS
white = (255, 245, 238)
blue  = (128,   0,   0)
red   = (  0,   0, 255)

class PlayerAnnotator:
    def __init__(self, team_1_color: Tuple[int, int, int] = white, team_2_color: Tuple[int, int, int] = blue):
        """
        Initializes the PlayerAnnotator class.
        You can later add options for styling (colors, fonts, etc.).
        """
        self.default_team_id = 1

        self.team_1_color = team_1_color
        self.team_2_color = team_2_color
        

    def draw_tracks(self, video_frames: List[np.ndarray], 
                    tracks: List[Dict[int, Dict[str, Any]]],
                    player_teams: List[Dict[str, int]],
                    ball_acquisition) -> List[np.ndarray]:
        """
        Draws tracked player annotations (ellipses) on each video frame.

        Args:
            video_frames (List[np.ndarray]): List of frames from the video.
            tracks (List[Dict[int, Dict[str, Any]]]): 
                A list of dictionaries, one per frame.
                Each dictionary maps a track ID to a player's data (with a "bbox" key).

        Returns:
            List[np.ndarray]: List of frames with annotations drawn.
        """
        out_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            out_frame = frame.copy()  # Avoid modifying the original frame

            player_dict = tracks[frame_num]  # Get all tracked players for this frame

            team_asgn_for_frame = player_teams[frame_num]

            player_wball_id = ball_acquisition[frame_num]

            

            # Draw each player's bounding ellipse
            for track_id, player in player_dict.items():
                team_id = team_asgn_for_frame.get(track_id, self.default_team_id)

                color = self.team_1_color if (team_id == 1) else self.team_2_color 

                if (track_id == player_wball_id):
                    draw_triangle(out_frame, player["bbox"], red)
                      
                draw_ellipse(out_frame, player["bbox"], color, track_id)

            out_video_frames.append(out_frame)  # Store the annotated frame

        return out_video_frames