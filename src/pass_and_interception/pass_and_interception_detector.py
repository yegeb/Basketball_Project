from typing import List, Dict, Any

class PassAndInterceptionDetector:
    """
    Detects pass and interception events based on ball possession changes
    and player team assignments over time.
    """

    def __init__(self):
        """
        Initializes the detector. Currently no parameters needed.
        """
        pass

    def detect_passes(self, ball_acquisition: List[int], player_teams: List[Dict[int, int]]) -> List[int]:
        """
        Detects passes based on changes in ball possession between players of the same team.

        Args:
            ball_acquisition (List[int]): A list where each index is a frame number and each value
                                          is the player_id holding the ball in that frame (-1 if none).
            player_teams (List[Dict[int, int]]): A list where each index is a frame number and each
                                                 value is a dictionary mapping player_id to team_id.

        Returns:
            List[int]: A list of same length as `ball_acquisition`, where each index contains the team_id
                       if a pass occurred at that frame, otherwise -1.
        """
        passes = [-1] * len(ball_acquisition)  # Initialize all frames as no-pass (-1)
        prev_holder    = -1  # Last known player holding the ball
        prev_frame_num = -1  # Frame number of previous holder

        for frame_num in range(1, len(ball_acquisition)):
            # Update previous holder if there was valid possession in the previous frame
            if ball_acquisition[frame_num - 1] != -1:
                prev_holder = ball_acquisition[frame_num - 1]
                prev_frame_num = frame_num - 1

            cur_holder = ball_acquisition[frame_num]

            # Check if valid transition occurred: different player now holds the ball
            if prev_frame_num != -1 and cur_holder != -1 and prev_holder != cur_holder:
                # Get team IDs of previous and current ball holders
                prev_team = player_teams[prev_frame_num].get(prev_holder, -1)
                cur_team = player_teams[frame_num].get(cur_holder, -1)

                # If same team and valid team IDs, it's a pass
                if prev_team == cur_team and prev_team != -1:
                    passes[frame_num] = prev_team  # Mark pass at this frame

        return passes

    def detect_interception(self, ball_acquisition: List[int], player_teams: List[Dict[int, int]]) -> List[int]:
        """
        Detects interceptions based on changes in ball possession between players of different teams.

        Args:
            ball_acquisition (List[int]): A list where each index is a frame number and each value
                                          is the player_id holding the ball in that frame (-1 if none).
            player_teams (List[Dict[int, int]]): A list where each index is a frame number and each
                                                 value is a dictionary mapping player_id to team_id.

        Returns:
            List[int]: A list of same length as `ball_acquisition`, where each index contains the team_id
                       of the player who intercepted the ball, otherwise -1.
        """
        interceptions = [-1] * len(ball_acquisition)  # Initialize all frames as no-interception (-1)
        prev_holder = -1  # Last known player holding the ball
        prev_frame_num = -1  # Frame number of previous holder

        for frame_num in range(1, len(ball_acquisition)):
            # Update previous holder if valid
            if ball_acquisition[frame_num - 1] != -1:
                prev_holder = ball_acquisition[frame_num - 1]
                prev_frame_num = frame_num - 1

            cur_holder = ball_acquisition[frame_num]

            # If the ball holder changed to someone else
            if prev_frame_num != -1 and cur_holder != -1 and prev_holder != cur_holder:
                prev_team = player_teams[prev_frame_num].get(prev_holder, -1)
                cur_team = player_teams[frame_num].get(cur_holder, -1)

                # If teams differ and both teams are valid, it's an interception
                if prev_team != cur_team and prev_team != -1 and cur_team != -1:
                    interceptions[frame_num] = cur_team  # Team that intercepted the ball

        return interceptions