from ultralytics import YOLO
import sys
sys.path.append("../")  # Add parent directory to import utils module
from utils import read_stub, save_stub  # Helper functions for caching/stubbing results
from typing import List, Any
import numpy as np

class CourtKeypointDetector:
    """
    A class for detecting court keypoints from video frames using a YOLO model.

    Attributes:
        model (YOLO): A YOLO model instance for inference.
    """

    def __init__(self, model_path: str):
        """
        Initializes the keypoint detector with a YOLO model.

        Args:
            model_path (str): Path to the YOLO model file.
        """
        self.model = YOLO(model_path)  # Load the YOLO model

    def get_court_keypoints(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str = None
    ) -> List[Any]:
        """
        Detects court keypoints for each frame in the list. Optionally loads from or saves to a stub cache.

        Args:
            frames (List[np.ndarray]): List of video frames (as numpy arrays).
            read_from_stub (bool): Whether to read from cached results.
            stub_path (str): Path to the stub file for saving/loading keypoints.

        Returns:
            List[Any]: A list of keypoints (per frame).
        """
        
        # Try reading cached keypoints from a stub file
        court_keypoints = read_stub(read_from_stub, stub_path)

        
        if court_keypoints is not None:
            if len(court_keypoints) == len(frames):  # Use cached results only if they match frame count
                return court_keypoints

        # If cache is missing or mismatched, run detection
        batch_size = 32  # Run inference in batches to optimize memory/speed
        court_keypoints = []

        for i in range(0, len(frames), batch_size):
            # Predict keypoints on the current batch of frames
            detections_batch = self.model.predict(frames[i : i + batch_size], conf=0.8)
            
            # Extract keypoints from each detection result in the batch
            for detection in detections_batch:
                court_keypoints.append(detection.keypoints)
                
                
        # Save results to a stub file for later reuse
        save_stub(stub_path, court_keypoints)

        return court_keypoints