# Basketball Video Analysis System

This project is a modular pipeline designed to analyze basketball games from video footage. 
It combines state-of-the-art object detection, tracking, homography mapping, event detection, and statistical overlays to generate rich visual and data insights for each game.

## Overview

- **Player & Ball Detection/Tracking** using YOLO and ByteTrack
- **Court Keypoint Detection** and **Homography Mapping** to real court coordinates
- **Ball Possession Detection**
- **Pass & Interception Recognition**
- **Team Assignment** using CLIP-based jersey color classification
- **Player Speed & Distance Metrics**
- **Annotated Video Output** with overlays including:
  - Player identity and team
  - Ball pointer
  - Possession status
  - Court map projection
  - Speed and distance stats
  - Passes and interceptions
 
## Repository Structure
### Source Code (src/)
Contains all the Python files implementing the core functionality:
main.py: Entry point to the full pipeline
player_tracker.py, ball_tracker.py: Tracking modules using YOLO + ByteTrack
court_kp_detector.py, court_map.py, homography.py: For court keypoint detection and coordinate mapping
team_assigner.py: Classifies players into teams based on jersey color
ball_acqs_detector.py: Detects ball possession
pass_and_interception_detector.py: Detects pass and steal events
speed_and_dist_calc.py: Computes speed and distance covered

### Models (models/)
Stores trained YOLO and CLIP models used in detection and classification tasks.

### Input Videos (input_videos/)
Raw basketball match videos to be analyzed.

### Output Videos (output_videos/)
Contains the generated video with overlays and annotations for player tracking, court map projection, ball possession, stats, etc.

### Configuration & Utilities (config.py, utils.py)
Shared configuration (e.g., model paths, court dimensions) and helper functions (e.g., bbox math, I/O).

### Stub Cache (stubs/) (optional)
Intermediate computation results (e.g., detection, keypoints, team assignments) stored to reduce re-computation time during development.




