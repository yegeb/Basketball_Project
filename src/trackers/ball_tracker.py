from ultralytics import YOLO
import numpy as np
import supervision as sv
import pandas as pd
from typing import List, Dict, Any
import sys
sys.path.append("../")
from utils import read_stub, save_stub

class BallTracker:
    def __init__(self, model_path: str):
        """
        Initializes the PlayerTracker with a YOLO detection model and a ByteTrack tracker.

        Args:
            model_path (str): Path to the YOLO model file.
        """
        self.model = YOLO(model_path)
        

    def detect_frames(self, frames: List[np.ndarray]) -> List[Any]:
        """
        Runs object detection on a list of frames in batches.

        Args:
            frames (List[np.ndarray]): List of video frames as NumPy arrays.

        Returns:
            List[Any]: List of detection results from the YOLO model.
        """
        batch_size = 32  # Process frames in batches to improve efficiency
        detections = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i: i + batch_size]  # Get a batch of frames
            batch_detections = self.model.predict(batch_frames, conf=0.5)  # Run detection
            detections += batch_detections  # Accumulate detections

        return detections
    
    def get_object_tracks(self, frames: List[np.ndarray], read_from_stub: bool = False, stub_path: str = None) -> List[Dict[int, Dict[str, Any]]]:
        """
        Detects players in each frame and returns tracked bounding boxes per frame.

        Args:
            frames (List[np.ndarray]): List of video frames as NumPy arrays.
            read_from_stub (bool): If True, try to load tracking data from a saved checkpoint file.
            stub_path (str): Path to the checkpoint file for reading or saving tracking results.

        Returns:
            List[Dict[int, Dict[str, Any]]]: 
                A list where each element corresponds to a frame and contains a dictionary mapping
                `track_id` to the bounding box coordinates.
        """

        tracks = read_stub(read_from_stub, stub_path)  # Attempt to load cached tracking results if available

        if (tracks is not None):
            if (len(tracks) == len(frames)):
                return tracks  # Return cached results if they match the current number of frames
            
        detections = self.detect_frames(frames)
        tracks     = []

        for frame_num, detection in enumerate(detections):
            # Attempt to retrieve class name mappings
            cls_dict = detection.names  
            cls_names   = {v: k for k, v in cls_dict.items()}  # Reverse mapping: ID → Name

            # Convert Ultralytics detection to Supervision-compatible detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)  
            tracks.append({})
            chosen_bbox = None
            max_conf    = 0 

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist() # Extract bounding box as list 
                cls_id = frame_detection[3]        # Class ID
                conf = frame_detection[2]          # Confidence score

                if (cls_id == cls_names["Ball"]):
                    if(max_conf < conf):
                        chosen_bbox = bbox 

            if (chosen_bbox is not None):
                tracks[frame_num][1] = {"bbox": chosen_bbox}             
            
        save_stub(stub_path, tracks)  # Save computed tracks to a checkpoint file for future reuse

        return tracks 

 
    def trim_false_detections(self, ball_positions: List[Dict[int, Dict[str, Any]]]) -> List[Dict[int, Dict[str, Any]]]:
        """
        Removes false ball detections based on sudden, unrealistic jumps in position.

        Args:
            ball_positions (List[Dict[int, Dict[str, Any]]]): 
                A list where each element corresponds to a frame, and each frame contains
                a dictionary of tracked balls, keyed by track ID (e.g., 1),
                with bounding boxes under the key "bbox".

        Returns:
            List[Dict[int, Dict[str, Any]]]: 
                The filtered list where false detections (too far from last valid position)
                are removed by setting the frame entry to an empty dictionary.
        """
        maximum_allowed_distance = 25  # Maximum acceptable movement between two valid frames
        last_valid_frame_index   = -1  # Index of the last frame with a valid detection

        for i in range(len(ball_positions)):
            # Get the current bounding box for track ID 1
            current_bbox = ball_positions[i].get(1, {}).get("bbox", [])

            # Skip frames with no detection
            if len(current_bbox) < 2:
                continue

            # First valid detection, initialize baseline
            if last_valid_frame_index == -1:
                last_valid_frame_index = i
                continue

            # Get the last valid bounding box
            last_valid_bbox = ball_positions[last_valid_frame_index].get(1, {}).get("bbox", [])

            # Skip if last valid bbox is also invalid
            if len(last_valid_bbox) < 2:
                continue

            # Calculate how many frames have passed
            frame_gap = i - last_valid_frame_index

            # Allow more movement for frames that are farther apart
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            # Compute Euclidean distance between the centers of the two boxes
            dist = np.linalg.norm(np.array(current_bbox[:2]) - np.array(last_valid_bbox[:2]))

            if dist > adjusted_max_distance:
                # Detected jump is too large — likely a false detection
                ball_positions[i] = {}
            else:
                # Detection is valid — update last known good position
                last_valid_frame_index = i

        return ball_positions
  
    def interpolate_ball_pos(self, ball_positions: List[Dict[int, Dict[str, Any]]]) -> List[Dict[int, Dict[str, Any]]]:
        """
        Fills in missing ball positions by interpolating bounding box coordinates across frames.

        Args:
            ball_positions (List[Dict[int, Dict[str, Any]]]): 
                A list of dictionaries, one per frame.
                Each frame may contain ball tracking info under track ID 1 and key "bbox".

        Returns:
            List[Dict[int, Dict[str, Any]]]: 
                A new list with interpolated bounding boxes where values were missing.
        """
        
        # Extract bounding box for track ID 1 from each frame
        # If not present, fallback to an empty dictionary (causing NaNs in the DataFrame)
        ball_positions = [x.get(1, {}).get("bbox", {}) for x in ball_positions]

        # Convert list of bounding boxes into a DataFrame with columns [x1, y1, x2, y2]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        # Interpolate missing (NaN) values linearly along each column
        df_ball_positions = df_ball_positions.interpolate()

        
        # Fill any remaining NaNs from earlier or later values
        df_ball_positions = df_ball_positions.bfill().ffill()

        # Finally, replace any remaining NaNs with 0 as a safe fallback
        df_ball_positions = df_ball_positions.fillna(0)

        # Convert DataFrame back to the original ball_positions format:
        # [{1: {"bbox": [...]}}] — one per frame
        ball_positions = [{1: {"bbox": [int(val) for val in bbox]}} for bbox in df_ball_positions.to_numpy().tolist()]

        return ball_positions