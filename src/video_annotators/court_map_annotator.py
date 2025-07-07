import cv2
from typing import Tuple, List, Dict, Any
import numpy as np

bgr_red =   (  0,   0, 255)
bgr_green = (  0, 255,   0)
t1_color =  (255, 245, 238)
t2_color =  (128,   0,   0)

class CourtMapAnnotator:
    """
    Annotator class to overlay a scaled court image and draw court keypoints,
    player positions, and the ball owner on each frame of a video sequence.
    
    Attributes:
        lc_x (int): X offset for placing the court image on the frame.
        lc_y (int): Y offset for placing the court image on the frame.
        team_1_color (tuple): BGR color for team 1 players.
        team_2_color (tuple): BGR color for team 2 players.
        ALPHA (float): Transparency factor for blending the court image onto the frame.
    """ 
    ALPHA = 0.6

    def __init__(self, team_1_color: Tuple[int, int, int] = t1_color, team_2_color: Tuple[int, int, int] = t2_color):
        self.lc_x = 20
        self.lc_y = 40
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color 

    def draw_map(self, video_frames: List[np.ndarray], 
                 court_image_path: str, 
                 width: int, height: int, 
                 court_keypoints: List[List[Tuple[int, int]]],
                 player_positions: List[Dict[int, List[float]]], 
                 player_teams: List[Dict[int, int]] = None, 
                 ball_acqs: List[int] = None) -> List[np.ndarray]:
        """
        Draws a court map with keypoints, player positions, and ball possession on each frame.
        
        Args:
            video_frames (List[np.ndarray]): List of video frames.
            court_image_path (str): Path to the court image to be overlaid.
            width (int): Width to which the court image should be resized.
            height (int): Height to which the court image should be resized.
            court_keypoints (List[List[Tuple[int, int]]]): List of keypoint coordinates for each frame.
            player_positions (List[Dict[int, List[float]]]): Player positions per frame.
            player_teams (List[Dict[int, int]], optional): Mapping from player ID to team ID per frame.
            ball_acqs (List[int], optional): Player ID possessing the ball per frame.

        Returns:
            List[np.ndarray]: List of frames with the court and annotations drawn.
        """

        # Load and resize the court image to the specified dimensions
        court_image = cv2.imread(court_image_path)
        court_image = cv2.resize(court_image, (width, height))

        out_video_frames = []

        # Iterate over all frames
        for frame_num, frame in enumerate(video_frames):
            frame_copy = frame.copy()

            # Define overlay region based on top-left corner (lc_x, lc_y)
            x1 = self.lc_x
            y1 = self.lc_y
            x2 = x1 + width
            y2 = y1 + height

            # Create overlay by blending the resized court image with the current frame section
            overlay = frame_copy[y1:y2, x1:x2].copy()
            cv2.addWeighted(court_image, self.ALPHA, overlay, 1 - self.ALPHA, 0, frame_copy[y1:y2, x1:x2])

            # Draw court keypoints and their indices
            for keypoint_index, keypoint in enumerate(court_keypoints):
                x, y = keypoint
                x  += self.lc_x
                y  += self.lc_y

                point = (x, y)

                # Draw red circle at each keypoint location
                cv2.circle(img=frame_copy, center=point, radius=5, color=bgr_red, thickness=-1)

                # Label each keypoint with its index in green
                cv2.putText(img=frame_copy, text=str(keypoint_index), org=point, 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=bgr_green, thickness=2)

                # Get player data for current frame
                frame_positions = player_positions[frame_num]
                frame_teams = player_teams[frame_num]
                player_wball = ball_acqs[frame_num]

                # Iterate over players and draw their positions
                for player_id, position in frame_positions.items():
                    team_id = frame_teams.get(player_id, 1)  # Default to team 1 if missing

                    # Set player color based on team
                    color = self.team_1_color if (team_id == 1) else self.team_2_color

                    # Shift player position by court placement offset
                    x,y = int(position[0] + self.lc_x), int(position[1] + self.lc_y)

                    player_radius = 6

                    # Draw filled circle for player
                    cv2.circle(img=frame_copy, center=(x, y), radius=player_radius, color=color, thickness=-1)

                    # Highlight the player with the ball using a red border
                    if(player_id == player_wball):
                        cv2.circle(img=frame_copy, center=(x, y), radius=player_radius+2, color=bgr_red, thickness=2)

            # Store annotated frame
            out_video_frames.append(frame_copy)

        return out_video_frames