from ultralytics import YOLO
import supervision as sv
import numpy as np
from typing import List, Dict, Any
import sys
sys.path.append("../")
from utils import read_stub, save_stub


class PlayerTracker:
    def __init__(self, model_path: str):
        """
        Initializes the PlayerTracker with a YOLO detection model and a ByteTrack tracker.

        Args:
            model_path (str): Path to the YOLO model file.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

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
        tracks = []

        for frame_num, detection in enumerate(detections):
            # Attempt to retrieve class name mappings
            cls_dict  = detection.names  
            cls_names = {v: k for k, v in cls_dict.items()}  # Reverse mapping: ID → Name

            # Convert Ultralytics detection to Supervision-compatible detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Run ByteTrack tracking on the current frame's detections
            detection_wtracks = self.tracker.update_with_detections(detection_supervision)

            # Initialize empty track dictionary for current frame
            tracks.append({})

            # Iterate over each detected object with a track ID
            for frame_detection in detection_wtracks:
                bbox = frame_detection[0].tolist()  # Extract bounding box as list
                cls_id = frame_detection[3]         # Class ID
                track_id = frame_detection[4]       # Unique tracking ID

                # Filter only for players based on class name
                if cls_id == cls_names["Player"]:
                    tracks[frame_num][track_id] = {"bbox": bbox}

        save_stub(stub_path, tracks)  # Save computed tracks to a checkpoint file for future reuse

        return tracks