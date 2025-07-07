import cv2
import numpy as np
from typing import List

# Define color constants in BGR format
white = (255, 255, 255)
black = (  0,   0,   0)

class PassInterceptionAnnotator:
    """
    Annotates video frames with pass and interception statistics for two teams.
    
    Attributes:
        ALPHA (float): Transparency factor used when overlaying the stats box on frames.
        FONT_SCALE (float): Scale factor that determines the size of the font used in annotations.
        FONT_THICKNESS (int): Thickness of the font used for rendering text on the frames.
    """


    ALPHA = 0.8
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2 

    def __init__(self):
        # No special initialization needed
        pass

    def get_stats(self, passes, interceptions):
        """
        Computes cumulative pass and interception statistics per team.
        
        Args:
            passes (List[int]): A list where each entry indicates which team completed a pass (1 or 2), or 0 for none.
            interceptions (List[int]): A list where each entry indicates which team made an interception (1 or 2), or 0 for none.
        
        Returns:
            Tuple[int, int, int, int]: Count of team 1 passes, team 2 passes, team 1 interceptions, team 2 interceptions.
        """

        # Frame indices where each event type occurred
        team_1_passes        = []
        team_2_passes        = []
        team_1_interceptions = []
        team_2_interceptions = []

        for frame_num in range(len(passes)):
            # Check which team made a pass
            if (passes[frame_num] == 1):
                team_1_passes.append(frame_num)
            elif (passes[frame_num] == 2):
                team_2_passes.append(frame_num)

            # Check which team made an interception
            if (interceptions[frame_num] == 1):
                team_1_interceptions.append(frame_num)
            elif (interceptions[frame_num] == 2):
                team_2_interceptions.append(frame_num)    

        # Return counts of each type
        return len(team_1_passes), len(team_2_passes), len(team_1_interceptions), len(team_2_interceptions)               

    def draw(self, video_frames: List[np.ndarray], passes: List[int], interceptions: List[int]) -> List[np.ndarray]:
        """
        Annotates a sequence of video frames with ongoing pass/interception stats.
        
        Args:
            video_frames (List[np.ndarray]): List of video frames (as images).
            passes (List[int]): Pass data per frame.
            interceptions (List[int]): Interception data per frame.
        
        Returns:
            List[np.ndarray]: List of annotated video frames.
        """

        out_video_frames = []

        # Process each frame individually
        for frame_num, frame in enumerate(video_frames):
            frame_drawn = self.draw_frame(frame, frame_num, passes, interceptions)
            out_video_frames.append(frame_drawn)

        return out_video_frames

    def draw_frame(self, frame: np.ndarray, frame_num: int, passes: List[int], interceptions: List[int]):
        """
        Draws a transparent rectangle and writes stats text for a single frame.
        
        Args:
            frame (np.ndarray): The original frame to annotate.
            frame_num (int): The index of the current frame.
            passes (List[int]): List of passes up to current frame.
            interceptions (List[int]): List of interceptions up to current frame.
        
        Returns:
            np.ndarray: The frame with stats box and text drawn.
        """

        # Copy frame to prepare an overlay for blending
        overlay = frame.copy()

        # Determine rectangle position based on frame dimensions
        frame_height, frame_width = overlay.shape[:2]
        rect_x1 = int(frame_width * 0.16)
        rect_y1 = int(frame_height * 0.75)
        rect_x2 = int(frame_width * 0.55)
        rect_y2 = int(frame_height * 0.90)

        # Text positions inside the rectangle
        text_x  = int(frame_width * 0.19)
        text_y1 = int(frame_height * 0.80)
        text_y2 = int(frame_height * 0.88)

        # Draw solid white rectangle on overlay
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), white, -1)
        
        # Blend overlay with original frame using transparency
        cv2.addWeighted(overlay, self.ALPHA, frame, 1 - self.ALPHA, 0, overlay)

        # Get data up to current frame
        cur_pass_frames = passes[:frame_num + 1]
        cur_intcp_frames = interceptions[:frame_num + 1]

        # Get event counts per team
        team_1_passes, team_2_passes, team_1_interceptions, team_2_interceptions = self.get_stats(cur_pass_frames, cur_intcp_frames)

        # Draw team 1 stats
        cv2.putText(overlay, f"Team 1 - Passes: {team_1_passes}, Interceptions: {team_1_interceptions}", 
                    (text_x, text_y1), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, black, self.FONT_THICKNESS)

        # Draw team 2 stats
        cv2.putText(overlay, f"Team 2 - Passes: {team_2_passes}, Interceptions: {team_2_interceptions}", 
                    (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, black, self.FONT_THICKNESS)  

        # Return the annotated frame
        return overlay