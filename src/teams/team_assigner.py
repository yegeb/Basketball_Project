from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
from typing import Any, Dict, List
import sys
sys.path.append("../")
from utils import read_stub, save_stub

class TeamAssigner:
    def __init__(self, 
                 team_1_class_name: str = "white shirt",
                 team_2_class_name: str = "dark blue shirt"):
        """
        Initializes a TeamAssigner with two team class labels.

        Args:
            team_1_class_name (str): Text description for Team 1.
            team_2_class_name (str): Text description for Team 2.
        """
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name

        self.player_team_dict = {}


    def load_model(self) -> None:
        """
        Loads the pretrained CLIP model and processor for image-text classification.
        """
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")


    def get_player_color(self, frame: np.ndarray, bbox: List[int]) -> str:
        """
        Uses a pretrained CLIP model to classify the jersey color of a player
        based on a cropped image of their bounding box.

        Args:
            frame (np.ndarray): The full video frame in BGR format (as from OpenCV).
            bbox (List[int]): Bounding box in format [x1, y1, x2, y2].

        Returns:
            str: The predicted team class name (either team_1 or team_2).
        """
        # Crop the player image from the full frame using the bounding box
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]  # frame[y1:y2, x1:x2]

        # Convert cropped image from BGR (OpenCV) to RGB (PIL/CLIP requirement)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Text labels for classification
        classes = [self.team_1_class_name, self.team_2_class_name]

        # Preprocess image and text for CLIP
        inputs = self.processor(text=classes, images=pil_img, return_tensors="pt", padding=True)

        # Forward pass through CLIP model
        outputs = self.model(**inputs)

        # Extract similarity scores between image and each class
        logits_per_img = outputs.logits_per_image

        # Apply softmax to get probabilities
        pred_probs = logits_per_img.softmax(dim=1)

        # Return the class with the highest probability
        class_name = classes[pred_probs.argmax(dim=1)[0]] 
        return class_name

    def get_player_team(self, frame: np.ndarray, player_bbox: List[int], player_id: int) -> int:
        """
        Determines the team ID of a player by analyzing their jersey color.
        Caches the result using the player ID to avoid recomputation.

        Args:
            frame (np.ndarray): The video frame containing the player.
            player_bbox (List[int]): Bounding box of the player [x1, y1, x2, y2].
            player_id (int): Unique identifier for the player (e.g., tracking ID).

        Returns:
            int: Team ID — 1 for team_1_class_name, 2 for team_2_class_name.
        """

        # Use CLIP to classify player's jersey color
        player_color = self.get_player_color(frame, player_bbox)

        # If team already assigned to this player, return it
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Default team assignment is 2
        team_id = 2

        # If color matches team 1, assign team ID 1
        if player_color == self.team_1_class_name:
            team_id = 1

        # Cache the result for this player ID
        self.player_team_dict[player_id] = team_id

        return team_id

    
    def get_player_teams_across_frames(self, 
                                    video_frames: List[np.ndarray], 
                                    player_tracks: List[Dict[int, Dict[str, Any]]],
                                    read_from_stub: bool = False, 
                                    stub_path: str = None) -> List[Dict[int, int]]:
        """
        Assigns team IDs to all tracked players across video frames using jersey color classification.

        Args:
            video_frames (List[np.ndarray]): List of video frames.
            player_tracks (List[Dict[int, Dict[str, Any]]]): 
                List of dictionaries for each frame, mapping player IDs to tracking data (must include "bbox").
            read_from_stub (bool): If True, try to load cached assignments from file.
            stub_path (str): File path for reading/writing cached player-team assignments.

        Returns:
            List[Dict[int, int]]: A list of dictionaries per frame mapping player IDs to their team IDs.
        """

        # Attempt to load cached team assignments from stub file
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if (len(player_assignment) == len(video_frames)):  
                return player_assignment

        # Load the CLIP model for jersey classification
        self.load_model()
        player_assignment = []

        # Process each frame and assign team IDs
        for frame_num, player_track in enumerate(player_tracks):
            player_assignment.append({})  # Initialize empty dict for this frame

            # Every 50 frames, clear the memory of past assignments to allow reassignment
            if frame_num % 50 == 0:
                self.player_team_dict = {}

            # Loop over each player in the current frame
            for player_id, track in player_track.items():  
                team = self.get_player_team(video_frames[frame_num], track["bbox"], player_id)
                player_assignment[frame_num][player_id] = team

        # Save computed assignments to stub file
        save_stub(stub_path, player_assignment)

        return player_assignment