import os
import argparse
from ultralytics import YOLO

from utils import read_video, save_video 
from trackers import PlayerTracker, BallTracker
from video_annotators import (
    PlayerAnnotator, BallPointer, TeamStatsAnnotator, 
    PassInterceptionAnnotator, CourtKeypointAnnotator, 
    CourtMapAnnotator, SpeedAndDistAnnotator
)
from teams import TeamAssigner
from ball_acquisition import BallAcqsDetector
from pass_and_interception import PassAndInterceptionDetector
from court_keypoint import CourtKeypointDetector
from court import CourtMap
from speed_and_distance import SpeedAndDistCalc
from config import (
    COURT_IMG_PATH, PLAYER_DETECTOR_PATH, BALL_DETECTOR_PATH, COURT_KEYPOINT_DETECTOR_PATH, 
    OUTPUT_VIDEO_PATH, COURT_IMG_PATH, PTRACKER_STUB_DEFAULT_PATH, BTRACKER_STUB_DEFAULT_PATH, 
    KPTRACKER_STUB_DEFAULT_PATH, PTEAMS_STUB_DEFAULT_PATH
)


def parse_args():
    parser = argparse.ArgumentParser(description="Basketball Video Analysis")
    parser.add_argument("--input_video", type=str, help="Path to input video file")
    parser.add_argument("--output_video_path", type=str, default=OUTPUT_VIDEO_PATH, help="Path to output video")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Input video path:", args.input_video)


    # ------------------- Load Video ------------------- #
    video_frames = read_video(args.input_video)
    
    # ------------------- Initialize Modules ------------------- #
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    ball_tracker = BallTracker(BALL_DETECTOR_PATH)
    court_kp_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)
    team_assigner = TeamAssigner()
    ball_acqs_detector = BallAcqsDetector()
    pass_detector = PassAndInterceptionDetector()
    court_map = CourtMap(court_img_path=COURT_IMG_PATH)
    
    # ------------------- Tracking ------------------- #
    
    
    player_tracks = player_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=PTRACKER_STUB_DEFAULT_PATH)
    ball_tracks = ball_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=BTRACKER_STUB_DEFAULT_PATH)
    ball_tracks = ball_tracker.trim_false_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_pos(ball_tracks)

    # ------------------- Court Keypoints ------------------- #
    court_keypoints = court_kp_detector.get_court_keypoints(video_frames, read_from_stub=False, stub_path=KPTRACKER_STUB_DEFAULT_PATH)
    court_keypoints = court_map.validate_map(court_keypoints)
    


    # ------------------- Team Assignment ------------------- #
    player_teams = team_assigner.get_player_teams_across_frames(video_frames, player_tracks, read_from_stub=True, stub_path=PTEAMS_STUB_DEFAULT_PATH)

    # ------------------- Ball Possession, Passes, Interceptions ------------------- #
    ball_possession = ball_acqs_detector.detect_ball_possession(player_tracks, ball_tracks)
    passes = pass_detector.detect_passes(ball_possession, player_teams)
    interceptions = pass_detector.detect_interception(ball_possession, player_teams)

    # ------------------- Map Player Positions to Court ------------------- #
    map_player_positions = court_map.integrate_players_into_map(court_keypoints, player_tracks)

    # ------------------- Speed and Distance Calculation ------------------- #
    speed_calc = SpeedAndDistCalc(court_map.width, court_map.height, court_map.actual_width_in_meters, court_map.actual_height_in_meters)
    player_dist_per_frame = speed_calc.calc_dist(map_player_positions)
    player_speed_per_frame = speed_calc.calc_speed(player_dist_per_frame, fps=30)

    # ------------------- Initialize Annotators ------------------- #
    player_annotator = PlayerAnnotator()
    ball_pointer = BallPointer()
    team_stats_annotator = TeamStatsAnnotator()
    pass_interception_annotator = PassInterceptionAnnotator()
    court_kp_annotator = CourtKeypointAnnotator()
    court_map_annotator = CourtMapAnnotator()
    speed_annotator = SpeedAndDistAnnotator()

    # ------------------- Annotate Frames ------------------- #
    
    out_frames = player_annotator.draw_tracks(video_frames, player_tracks, player_teams, ball_possession)
    

    out_frames = ball_pointer.draw_tracks(out_frames, ball_tracks)
    
    out_frames = team_stats_annotator.draw(out_frames, player_teams, ball_possession)
    
    out_frames = pass_interception_annotator.draw(out_frames, passes, interceptions)
    
    out_frames = court_kp_annotator.draw_keypoints(out_frames, court_keypoints)
        
    out_frames = court_map_annotator.draw_map(
        out_frames,
        court_map.court_img_path,
        court_map.width,
        court_map.height,
        court_map.keypoints,
        map_player_positions,  
        player_teams,
        ball_possession
    )

    out_frames = speed_annotator.draw(out_frames, player_tracks, player_dist_per_frame, player_speed_per_frame)

    # ------------------- Save Output ------------------- #
    save_video(out_frames, args.output_video_path)


if __name__ == "__main__":
    main()