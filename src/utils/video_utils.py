import cv2
import os
import numpy as np
from typing import List

def read_video(video_path: str) -> List[np.ndarray]:
    """
    Reads a video file and returns its frames as a list of NumPy arrays.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        List[np.ndarray]: List of video frames, each represented as a NumPy array (image).
    """
    cap = cv2.VideoCapture(video_path)  # Create a video capture object
    frames = []

    while True:
        ret, frame = cap.read()  # Read a single frame
        if not ret:
            break  # Stop if no more frames are available

        frames.append(frame)  # Store the frame in the list

    cap.release()  # Release the video capture object
    return frames


def save_video(output_video_frames: List[np.ndarray], output_video_path: str) -> None:
    """
    Saves a list of video frames to a video file at the specified output path.

    Args:
        output_video_frames (List[np.ndarray]): List of frames to write into the video.
        output_video_path (str): Path where the output video will be saved.

    Returns:
        None
    """
    # Ensure the output directory exists; if not, create it
    output_dir = os.path.dirname(output_video_path)
    if (output_dir and not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    # Define the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for AVI files
    height, width = output_video_frames[0].shape[:2]  # Get frame dimensions
    width, height = int(width), int(height)  
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))  # 30 FPS

    # Write each frame to the output video
    for frame in output_video_frames:

        out.write(frame)

    out.release()  # Finalize and close the video file