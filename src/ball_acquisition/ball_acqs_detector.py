import math
from typing import List, Dict, Tuple, Any
import sys
sys.path.append("../")
from utils import get_center_of_bbox

class BallAcqsDetector:
    """
    Detects which player possesses the ball based on bounding box intersection and distance.
    """

    def __init__(self):
        """
        Initializes thresholds for determining possession.
        """
        self.possession_threshold = 50    # Max allowed distance in pixels for loose possession
        self.min_frames = 11              # Number of consecutive frames to confirm possession
        self.containment_threshold = 0.8  # % of ball bbox inside player bbox for close possession

    def get_key_player_bbox_points(self, player_bbox: List[int], ball_center: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Generates key edge/center points on the player's bounding box, influenced by ball position.

        Args:
            player_bbox (List[int]): [x1, y1, x2, y2] coordinates of the player bounding box.
            ball_center (Tuple[int, int]): (x, y) coordinates of the ball center.

        Returns:
            List[Tuple[int, int]]: A list of important reference points on the player's bbox.
        """
        ball_x, ball_y = ball_center
        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        height = y2 - y1 

        out_points = []

        # Add horizontal intersection points at ball_y
        if y1 < ball_y < y2:
            out_points.append((x1, ball_y))
            out_points.append((x2, ball_y)) 

        # Add vertical intersection points at ball_x
        if x1 < ball_x < x2:
            out_points.append((ball_x, y1))
            out_points.append((ball_x, y2)) 

        # Add corners and edge centers
        out_points.extend([
            (x1, y1),               # Top left
            (x2, y1),               # Top right
            (x1, y2),               # Bottom left
            (x2, y2),               # Bottom right
            (x1 + width // 2, y1),  # Top center
            (x1 + width // 2, y2),  # Bottom center
            (x1, y1 + height // 2), # Left center
            (x2, y1 + height // 2)  # Right center
        ])  

        return out_points

    def min_dist_to_ball(self, player_bbox: List[int], ball_center: Tuple[int, int]) -> float:
        """
        Calculates the shortest distance from any key point of a player's bbox to the ball center.

        Args:
            player_bbox (List[int]): Player bounding box [x1, y1, x2, y2].
            ball_center (Tuple[int, int]): Ball center (x, y).

        Returns:
            float: Minimum distance in pixels from player to ball.
        """
        key_points = self.get_key_player_bbox_points(player_bbox, ball_center)
        min_distance = math.inf

        for key_point in key_points:
            dist = math.dist(ball_center, key_point)  
            min_distance = min(min_distance, dist)

        return min_distance

    def calc_ball_cont_ratio(self, player_bbox: List[int], ball_bbox: List[int]) -> float:
        """
        Calculates how much of the ball's bbox is contained within the player's bbox.

        Args:
            player_bbox (List[int]): [x1, y1, x2, y2]
            ball_bbox (List[int]): [x1, y1, x2, y2]

        Returns:
            float: Ratio of ball bbox area that overlaps with player bbox.
        """
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox

        # Compute areas
        ball_area = (bx2 - bx1) * (by2 - by1)

        # Compute intersection coordinates
        intsc_x1 = max(px1, bx1)
        intsc_y1 = max(py1, by1)
        intsc_x2 = min(px2, bx2)
        intsc_y2 = min(py2, by2)

        # Compute intersection area
        intersection_area = max(0, intsc_x2 - intsc_x1) * max(0, intsc_y2 - intsc_y1)

        containment_ratio = intersection_area / ball_area if ball_area != 0 else 0

        return containment_ratio

    def find_best_cndt_for_poss(self, ball_bbox: List[int], ball_center: Tuple[int, int], player_tracks_frame: Dict[int, Dict[str, Any]]) -> int:
        """
        Identifies the player most likely in possession of the ball.

        Args:
            ball_bbox (List[int]): Ball bounding box [x1, y1, x2, y2].
            ball_center (Tuple[int, int]): Ball center (x, y).
            player_tracks_frame (Dict[int, Dict[str, Any]]): Tracking data for all players in the frame.

        Returns:
            int: player_id of the best candidate, or -1 if none found.
        """
        close_players = []
        reg_dist_players = []

        for player_id, player_info in player_tracks_frame.items():
            player_bbox = player_info.get("bbox", [])
            if not player_bbox:
                continue

            # Check bbox overlap ratio
            containment = self.calc_ball_cont_ratio(player_bbox, ball_bbox)

            # Check spatial distance to ball center
            min_dist = self.min_dist_to_ball(player_bbox, ball_center)

            # Priority 1: containment (tight possession)
            if containment > self.containment_threshold:
                close_players.append((player_id, containment))
            else:
                # Priority 2: loose possession by distance
                reg_dist_players.append((player_id, min_dist))

        if close_players:
            # Highest containment wins
            best_candidate = max(close_players, key=lambda x: x[1])
            return best_candidate[0]
        
        if reg_dist_players:
            # Closest distance wins, if within threshold
            best_candidate = min(reg_dist_players, key=lambda x: x[1])
            if best_candidate[1] < self.possession_threshold:
                return best_candidate[0]

        return -1

    def detect_ball_possession(self, player_tracks: List[Dict[int, Dict[str, Any]]], ball_tracks: List[Dict[int, Dict[str, Any]]]) -> List[int]: 
        """
        Detects possession for each frame based on tracking data.

        Args:
            player_tracks (List[Dict[int, Dict[str, Any]]]):
                List of dicts with player_id → {"bbox": [...]}, one per frame.
            ball_tracks (List[Dict[int, Dict[str, Any]]]):
                List of dicts with ball_id → {"bbox": [...]}, one per frame.

        Returns:
            List[int]: List of player_ids (or -1) representing possession per frame.
        """
        num_frames = len(ball_tracks)
        possession_list = [-1] * num_frames  # Default: no possession
        consecutive_poss_cnt = {}  # Tracks consecutive frame counts per player

        for frame_num in range(num_frames):
            ball_info = ball_tracks[frame_num].get(1, {})  # Ball ID assumed to be 1
            if not ball_info:
                continue

            ball_bbox = ball_info.get("bbox", [])
            if not ball_bbox:
                continue

            ball_center = get_center_of_bbox(ball_bbox)

            best_player_id = self.find_best_cndt_for_poss(ball_bbox, ball_center, player_tracks[frame_num])

            if best_player_id != -1:
                # Increase consecutive count or reset for new player
                num_of_consecutive_frames = consecutive_poss_cnt.get(best_player_id, 0) + 1
                consecutive_poss_cnt = {best_player_id: num_of_consecutive_frames}

                if consecutive_poss_cnt[best_player_id] >= self.min_frames:
                    possession_list[frame_num] = best_player_id
            else:
                consecutive_poss_cnt = {}  # Reset streak if no possession

        return possession_list