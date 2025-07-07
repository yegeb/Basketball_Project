import sys
sys.path.append("../")
import math
from typing import Dict, List, Tuple

SCALING_FACTOR = 0.4

class SpeedAndDistCalc:
    """
    A class to calculate distance (in meters) covered by players
    across frames based on their positions in a court map.

    Attributes:
        width_in_px (int): Width of the court in pixels.
        height_in_px (int): Height of the court in pixels.
        width_in_m (float): Width of the court in meters.
        height_in_m (float): Height of the court in meters.
        m_per_px_x (float): Meters per pixel in X direction.
        m_per_px_y (float): Meters per pixel in Y direction.
    """

    def __init__(self,
                 width_in_px,
                 height_in_px,
                 width_in_m,
                 height_in_m):
        """
        Initializes the conversion factors from pixels to meters.

        Args:
            width_in_px (int): Court width in pixels.
            height_in_px (int): Court height in pixels.
            width_in_m (float): Court width in meters.
            height_in_m (float): Court height in meters.
        """
        self.width_in_px  = width_in_px
        self.height_in_px = height_in_px
        self.width_in_m   = width_in_m
        self.height_in_m  = height_in_m

        # Conversion factors: meters per pixel
        self.m_per_px_x = width_in_m  / width_in_px
        self.m_per_px_y = height_in_m / height_in_px    

    def calc_dist(self, map_player_positions: List[Dict[int, Tuple[int, int]]]) -> List[Dict[int, float]]:
        """
        Calculates distance traveled (in meters) for each player across frames.

        Args:
            map_player_positions (List[Dict[int, Tuple[int, int]]]):
                List of dictionaries mapping player IDs to (x, y) pixel positions per frame.

        Returns:
            List[Dict[int, float]]: Distance traveled by each player in each frame.
        """
        out_distances = []
        prev_player_positions = {}

        for frame_num, map_player_pos_frame in enumerate(map_player_positions):
            out_distances.append({})  # Initialize dict for current frame

            for player_id, cur_player_pos in map_player_pos_frame.items():
                if (player_id in prev_player_positions):
                    prev_player_pos = prev_player_positions[player_id]
                    # Calculate distance from previous position
                    meter_dist = self.calc_m_dist(prev_player_pos, cur_player_pos)
                    out_distances[frame_num][player_id] = meter_dist

                # Update current position for next frame's comparison
                prev_player_positions[player_id] = cur_player_pos

        return out_distances            

    def calc_m_dist(self, prev_pixel_pos: Tuple[int, int], cur_pixel_pos: Tuple[int, int]) -> float:
        """
        Calculates Euclidean distance in meters between two pixel positions.

        Args:
            prev_pixel_pos (Tuple[int, int]): Previous (x, y) position in pixels.
            cur_pixel_pos (Tuple[int, int]): Current (x, y) position in pixels.

        Returns:
            float: Distance in meters between the two positions.
        """
        prev_px_x, prev_px_y = prev_pixel_pos
        cur_px_x, cur_px_y = cur_pixel_pos  

        # Convert pixel coordinates to meters
        prev_m_x = prev_px_x * self.m_per_px_x
        prev_m_y = prev_px_y * self.m_per_px_y

        cur_m_x = cur_px_x * self.m_per_px_x
        cur_m_y = cur_px_y * self.m_per_px_y  

        # Euclidean distance in meters
        meter_dist = math.dist((prev_m_x, prev_m_y), (cur_m_x, cur_m_y))

        # Optional scaling 
        meter_dist = meter_dist * SCALING_FACTOR

        return meter_dist
    
    def calc_speed(self, distances: List[Dict[int, float]], fps: int = 30) -> List[Dict[int, float]]:
        """
        Calculates player speeds (in meters per second) for each frame using a sliding window approach.

        Args:
            distances (List[Dict[int, float]]): 
                A list where each element corresponds to a frame and contains a dictionary 
                mapping player_id to the distance (in meters) covered **in that frame**.
            fps (int, optional): 
                Frames per second of the video. Default is 30.

        Returns:
            List[Dict[int, float]]: 
                A list of the same length as `distances`, where each element is a dictionary 
                mapping player_id to their speed (in m/s) at that frame.
        """
        speeds = []
        WINDOW_SIZE = 5  # Minimum number of valid distance entries required to compute speed

        for frame_num in range(len(distances)):
            speeds.append({}) 

            for player_id in distances[frame_num].keys():
                start_frame    = max(0, frame_num - (WINDOW_SIZE * 3) + 1)  # Define sliding window range
                total_dist     = 0.0   # Sum of distances over the window
                present_frames = 0     # Count of valid frames where the player had movement data
                cur_frame      = None  # Used to prevent including the very first data point as a step

                # Accumulate total distance for the player within the sliding window
                for i in range(start_frame, frame_num + 1):
                    if (player_id in distances[i]):
                        if cur_frame is not None:
                            
                            total_dist += distances[i][player_id]
                            present_frames += 1
                        cur_frame = i  

                # Only compute speed if we have enough samples (to smooth noise)
                if (present_frames > WINDOW_SIZE):
                    time_in_s = present_frames / fps  

                    if (time_in_s > 0):
                        speed_ms = total_dist / time_in_s  
                        speeds[frame_num][player_id] = speed_ms
                    else:
                        speeds[frame_num][player_id] = 0.0  
                else:
                    speeds[frame_num][player_id] = 0.0  # Not enough data to compute speed

        return speeds