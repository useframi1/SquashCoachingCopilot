"""Data validation utilities."""

from typing import Optional, Tuple
import numpy as np
from .data_models import PlayerData, BallData, FrameData


class DataValidator:
    """Validates and cleans pipeline data."""

    def __init__(
        self,
        min_confidence: float = 0.3,
        max_position_change: float = 200.0,
        court_bounds: Optional[Tuple[float, float, float, float]] = None,
    ):
        """
        Initialize validator.

        Args:
            min_confidence: Minimum confidence threshold for detections
            max_position_change: Maximum allowed position change between frames (pixels)
            court_bounds: Court boundaries (x_min, y_min, x_max, y_max) in real coordinates
        """
        self.min_confidence = min_confidence
        self.max_position_change = max_position_change
        self.court_bounds = court_bounds

    def validate_player_data(
        self, player_data: PlayerData, prev_player_data: Optional[PlayerData] = None
    ) -> bool:
        """
        Validate player data.

        Args:
            player_data: Current player data
            prev_player_data: Previous frame player data for temporal validation

        Returns:
            True if data is valid, False otherwise
        """
        # Check if player has basic data
        if not player_data.is_valid():
            return False

        # Check confidence threshold
        if player_data.confidence is not None:
            if player_data.confidence < self.min_confidence:
                return False

        # Check for unrealistic position changes
        if prev_player_data is not None and prev_player_data.is_valid():
            prev_pos = prev_player_data.position
            curr_pos = player_data.position

            if prev_pos and curr_pos:
                distance = np.sqrt(
                    (curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2
                )
                if distance > self.max_position_change:
                    return False

        # Check court bounds if available
        if self.court_bounds and player_data.real_position:
            x, y = player_data.real_position
            x_min, y_min, x_max, y_max = self.court_bounds
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                return False

        return True

    def validate_ball_data(
        self, ball_data: BallData, prev_ball_data: Optional[BallData] = None
    ) -> bool:
        """
        Validate ball data.

        Args:
            ball_data: Current ball data
            prev_ball_data: Previous frame ball data for temporal validation

        Returns:
            True if data is valid, False otherwise
        """
        if not ball_data.is_valid():
            return False

        # Check confidence if available
        if ball_data.confidence is not None:
            if ball_data.confidence < self.min_confidence:
                return False

        # Check for unrealistic position changes
        if prev_ball_data is not None and prev_ball_data.is_valid():
            prev_pos = prev_ball_data.position
            curr_pos = ball_data.position

            if prev_pos and curr_pos:
                distance = np.sqrt(
                    (curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2
                )
                # Ball can move faster than players
                if distance > self.max_position_change * 2:
                    return False

        return True

    def validate_frame_data(
        self, frame_data: FrameData, prev_frame_data: Optional[FrameData] = None
    ) -> dict:
        """
        Validate entire frame data and return validation results.

        Args:
            frame_data: Current frame data
            prev_frame_data: Previous frame data

        Returns:
            Dictionary with validation results for each component
        """
        results = {
            "player1_valid": False,
            "player2_valid": False,
            "ball_valid": False,
            "court_valid": frame_data.court.is_calibrated,
        }

        # Validate players
        prev_p1 = prev_frame_data.player1 if prev_frame_data else None
        prev_p2 = prev_frame_data.player2 if prev_frame_data else None

        results["player1_valid"] = self.validate_player_data(
            frame_data.player1, prev_p1
        )
        results["player2_valid"] = self.validate_player_data(
            frame_data.player2, prev_p2
        )

        # Validate ball
        prev_ball = prev_frame_data.ball if prev_frame_data else None
        results["ball_valid"] = self.validate_ball_data(frame_data.ball, prev_ball)

        return results
