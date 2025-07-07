import numpy as np
import cv2
from typing import List, Dict
from copy import deepcopy

white = (255, 255, 255)
black = (  0,   0,   0)

class TeamStatsAnnotator:
    """
    Annotates video frames with team-level ball possession statistics over time.
    """

    def __init__(self):
        # No parameters to initialize
        pass

    def get_team_ball_control(self, player_teams: List[Dict[int, int]], ball_acquisition: List[int]) -> np.ndarray:
        """
        Determines which team has ball control for each frame.

        Args:
            player_teams (List[Dict[int, int]]): List of player-to-team mappings per frame.
            ball_acquisition (List[int]): List of player IDs who possess the ball each frame (-1 if no one).

        Returns:
            np.ndarray: Array of control values per frame (1 for Team 1, 2 for Team 2, -1 for unknown).
        """

        team_ball_control = []

        for team_asgn_frame, ball_acqs_frame in zip(player_teams, ball_acquisition):
            if (ball_acqs_frame == -1):
                # No ball possession
                team_ball_control.append(-1)
                continue

            if (ball_acqs_frame not in team_asgn_frame):
                # Possessor not in team map
                team_ball_control.append(-1)
                continue

            # Assign control based on possessor's team
            if (team_asgn_frame[ball_acqs_frame] == 1):
                team_ball_control.append(1)
            else:
                team_ball_control.append(2)

        return np.array(team_ball_control)

    def draw(self, video_frames: List[np.ndarray], player_teams: List[Dict[int, int]], ball_acquisition: List[int]) -> List[np.ndarray]:
        """
        Draws possession statistics on each video frame.

        Args:
            video_frames (List[np.ndarray]): List of video frames.
            player_teams (List[Dict[int, int]]): Player team assignments per frame.
            ball_acquisition (List[int]): Ball possession data per frame.

        Returns:
            List[np.ndarray]: Frames with possession statistics annotated.
        """

        # Calculate which team controls the ball per frame
        team_ball_control = self.get_team_ball_control(player_teams, ball_acquisition)
        out_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame_drawn = self.draw_frame(frame, frame_num, team_ball_control)
            out_video_frames.append(frame_drawn)

        return out_video_frames

    def draw_frame(self, frame, frame_num, team_ball_control):
        """
        Draws ball control percentages on a single frame.

        Args:
            frame (np.ndarray): Original frame to annotate.
            frame_num (int): Current frame index.
            team_ball_control (np.ndarray): Array of team control per frame.

        Returns:
            np.ndarray: Annotated frame.
        """

        overlay = frame.copy() 
        font_scale = 0.7
        font_thickness = 2

        # Get frame dimensions
        frame_height, frame_width = overlay.shape[:2]

        # Coordinates for white background rectangle
        rect_x1 = int(frame_width * 0.60)
        rect_y1 = int(frame_height * 0.75)
        rect_x2 = int(frame_width * 0.99)
        rect_y2 = int(frame_height * 0.90)

        # Text positions
        text_x  = int(frame_width * 0.63)
        text_y1 = int(frame_height * 0.80)
        text_y2 = int(frame_height * 0.88)

        # Draw white background for text
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), white, -1)

        # Blend overlay with frame
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Get current control data up to this frame
        current_team_ball_ctrl = team_ball_control[:frame_num + 1]
        current_total_frame = current_team_ball_ctrl.shape[0]

        # Count how many frames each team had the ball
        team_1_num_frames = current_team_ball_ctrl[current_team_ball_ctrl == 1].shape[0]
        team_2_num_frames = current_team_ball_ctrl[current_team_ball_ctrl == 2].shape[0]

        # Compute percentage possession
        team_1_perc = (team_1_num_frames / current_total_frame) * 100
        team_2_perc = (team_2_num_frames / current_total_frame) * 100

        # Draw text annotations
        cv2.putText(frame, f"Team 1 Ball Control:  {team_1_perc: .2f}%", 
                    (text_x, text_y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, black, font_thickness)

        cv2.putText(frame, f"Team 2 Ball Control:  {team_2_perc: .2f}%", 
                    (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, black, font_thickness)

        return frame