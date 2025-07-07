from copy import deepcopy
from typing import List, Dict, Any
import math  
import numpy as np
import cv2
from .homography import Homography
import sys
sys.path.append("../")
from utils import get_foot_pos

class CourtMap:
    """
    Handles mapping players' positions from image space to court coordinates using keypoints
    and homography transformations.
    """

    def __init__(self, court_img_path: str):
        """
        Initializes the CourtMap with image dimensions and real-world court dimensions.

        Args:
            court_img_path (str): Path to the static court image used for mapping.
        """
        self.court_img_path = court_img_path
        self.width  = 300  # pixel width of court image
        self.height = 161  # pixel height of court image

        # Real-world dimensions in meters (standard basketball court size)
        self.actual_width_in_meters  = 28.65
        self.actual_height_in_meters = 15.24

        # Predefined reference keypoints on the court map (pixel space)
        self.keypoints = [
            # Left Edge
            (0, 0),
            (0, int((0.91 / self.actual_height_in_meters) * self.height)),
            (0, int((5.18 / self.actual_height_in_meters) * self.height)),
            (0, int((10.0 / self.actual_height_in_meters) * self.height)),
            (0, int((14.1 / self.actual_height_in_meters) * self.height)),
            (0, int(self.height)),

            # Left FT Line
            (int((5.79 / self.actual_width_in_meters) * self.width), int((5.18 / self.actual_height_in_meters) * self.height)),
            (int((5.79 / self.actual_width_in_meters) * self.width), int((10.0 / self.actual_height_in_meters) * self.height)),

            # Middle Line
            (int(self.width / 2), self.height),
            (int(self.width / 2), 0),

            # Right FT Line
            (int(((self.actual_width_in_meters - 5.79) / self.actual_width_in_meters) * self.width), int((5.18 / self.actual_height_in_meters) * self.height)),
            (int(((self.actual_width_in_meters - 5.79) / self.actual_width_in_meters) * self.width), int((10.0 / self.actual_height_in_meters) * self.height)),

            # Right Edge
            (self.width, int(self.height)),
            (self.width, int((14.1 / self.actual_height_in_meters) * self.height)),
            (self.width, int((10.0 / self.actual_height_in_meters) * self.height)),
            (self.width, int((5.18 / self.actual_height_in_meters) * self.height)),
            (self.width, int((0.91 / self.actual_height_in_meters) * self.height)),
            (self.width, 0)
        ]

    def validate_map(self, keypoints_list: List[Any]) -> List[Any]:
        """
        Validates and filters out invalid court keypoints by comparing geometric ratios.

        Args:
            keypoints_list (List[Any]): List of keypoint detection results for each frame.

        Returns:
            List[Any]: Modified list with unreliable keypoints zeroed out.
        """
        keypoints_list = deepcopy(keypoints_list)

        for frame_num, frame_kps in enumerate(keypoints_list):
            
            frame_kps = frame_kps.xy.tolist()[0]

            # Find keypoints that are detected (non-zero)
            detected_indices = [i for i, kp in enumerate(frame_kps) if kp[0] > 0 and kp[1] > 0]

            if len(detected_indices) < 3:
                continue  # Need at least 3 to perform ratio check

            invalid_keypoints = []

            for i1 in detected_indices:
                if frame_kps[i1][0] == 0 and frame_kps[i1][1] == 0:
                    continue

                # Exclude current and already invalid keypoints
                other_indices = [i for i in detected_indices if i != i1 and i not in invalid_keypoints]

                if len(other_indices) < 2:
                    continue  # Need two more for comparison

                # Use the first two remaining as reference
                i2, i3 = other_indices[0], other_indices[1]

                # Compute distances in frame
                frame_dist_p1p2 = math.dist(frame_kps[i1], frame_kps[i2])
                frame_dist_p1p3 = math.dist(frame_kps[i1], frame_kps[i3])

                # Compute distances on the map
                map_dist_p1p2 = math.dist(self.keypoints[i1], self.keypoints[i2])
                map_dist_p1p3 = math.dist(self.keypoints[i1], self.keypoints[i3])

                if map_dist_p1p2 > 0 and map_dist_p1p3 > 0:
                    ratio_frame = frame_dist_p1p2 / frame_dist_p1p3 if frame_dist_p1p3 > 0 else math.inf
                    ratio_map = map_dist_p1p2 / map_dist_p1p3 if map_dist_p1p3 > 0 else math.inf

                    err = abs((ratio_frame - ratio_map) / ratio_map)

                    # If geometric distortion is too large, invalidate keypoint
                    if err > 0.8:
                        keypoints_list[frame_num].xy[0][i1] = 0
                        keypoints_list[frame_num].xyn[0][i1] = 0
                        invalid_keypoints.append(i1)

        return keypoints_list

    def integrate_players_into_map(self, keypoints_list: List[Any], player_tracks: List[Dict[int, Dict[str, Any]]]) -> List[Dict[int, List[float]]]:
        """
        Transforms all player positions from frame pixel space to the court map using homography.

        Args:
            keypoints_list (List[Any]): List of detected court keypoints per frame.
            player_tracks (List[Dict[int, Dict[str, Any]]]): Tracked player bounding boxes per frame.

        Returns:
            List[Dict[int, List[float]]]: For each frame, a dictionary mapping player_id to court map coordinates [x, y].
        """
        map_player_positions = []

        for frame_num, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
            
            frame_map_positions = {}
            frame_keypoints = frame_keypoints.xy.tolist()[0]

            if frame_keypoints is None:
                map_player_positions.append(frame_map_positions)
                continue

            # Filter only valid keypoints
            valid_indices = [idx for idx, kp in enumerate(frame_keypoints) if kp[0] > 0 and kp[1] > 0]

            # Require at least 4 keypoints for homography
            if len(valid_indices) < 4:
                map_player_positions.append(frame_map_positions)
                continue

            source_points = np.array([frame_keypoints[i] for i in valid_indices], dtype=np.float32)
            target_points = np.array([self.keypoints[i]     for i in valid_indices], dtype=np.float32)

            try:
                homography = Homography(source_points, target_points)
            except (ValueError, cv2.error) as e:
                map_player_positions.append(frame_map_positions)
                continue  # Skip this frame on homography failure

            for player_id, player_data in frame_tracks.items():
                bbox = player_data["bbox"]
                player_pos = np.array([get_foot_pos(bbox)])  # Get bottom-center foot position

                try:
                    map_player_pos = homography.transform_points(player_pos)
                    frame_map_positions[player_id] = map_player_pos[0].tolist()
                except (ValueError, cv2.error) as e:
                    continue  # Skip this player if transformation fails

            map_player_positions.append(frame_map_positions)

        return map_player_positions