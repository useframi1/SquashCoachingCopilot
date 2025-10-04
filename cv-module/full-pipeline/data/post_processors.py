"""Post-processing utilities for smoothing and filtering data."""

from collections import deque
from typing import Optional, Tuple, List
import numpy as np
from .data_models import PlayerData, BallData


class TemporalSmoother:
    """Applies temporal smoothing to tracking data (supports both streaming and batch modes)."""

    def __init__(self, window_size: int = 5):
        """
        Initialize temporal smoother.

        Args:
            window_size: Number of frames to use for smoothing
        """
        self.window_size = window_size
        self.player_position_history = {
            1: deque(maxlen=window_size),
            2: deque(maxlen=window_size),
        }
        self.ball_position_history = deque(maxlen=window_size)

    def smooth_player_positions(
        self, player_data_list: List[PlayerData], player_id: int
    ) -> List[PlayerData]:
        """
        Smooth player positions across all frames using centered windows (batch mode).

        Args:
            player_data_list: List of PlayerData for all frames
            player_id: Player ID

        Returns:
            List of PlayerData with smoothed positions
        """
        smoothed = []
        half_window = self.window_size // 2

        for i, player_data in enumerate(player_data_list):
            # Define centered window
            start = max(0, i - half_window)
            end = min(len(player_data_list), i + half_window + 1)
            window = player_data_list[start:end]

            # Collect valid positions in window
            positions = []
            real_positions = []
            for p in window:
                if p and p.is_valid():
                    if p.position:
                        positions.append(p.position)
                    if p.real_position:
                        real_positions.append(p.real_position)

            # Calculate smoothed positions
            if positions:
                avg_pos = tuple(np.mean(positions, axis=0))
            else:
                avg_pos = player_data.position if player_data else None

            if real_positions:
                avg_real_pos = tuple(np.mean(real_positions, axis=0))
            else:
                avg_real_pos = player_data.real_position if player_data else None

            # Create smoothed player data
            if player_data:
                smoothed_player = PlayerData(
                    player_id=player_id,
                    position=avg_pos,
                    real_position=avg_real_pos,
                    bbox=player_data.bbox,
                    confidence=player_data.confidence,
                    keypoints=player_data.keypoints,
                )
            else:
                smoothed_player = PlayerData(player_id=player_id)

            smoothed.append(smoothed_player)

        return smoothed

    def smooth_ball_positions(self, ball_data_list: List[BallData]) -> List[BallData]:
        """
        Smooth ball positions across all frames using centered windows (batch mode).

        Args:
            ball_data_list: List of BallData for all frames

        Returns:
            List of BallData with smoothed positions
        """
        smoothed = []
        half_window = self.window_size // 2

        for i, ball_data in enumerate(ball_data_list):
            # Define centered window
            start = max(0, i - half_window)
            end = min(len(ball_data_list), i + half_window + 1)
            window = ball_data_list[start:end]

            # Collect valid positions in window
            positions = []
            for b in window:
                if b and b.is_valid():
                    positions.append(b.position)

            # Calculate smoothed position
            if positions:
                avg_pos = tuple(np.mean(positions, axis=0).astype(int))
                smoothed_ball = BallData(
                    position=avg_pos,
                    confidence=ball_data.confidence if ball_data else None,
                )
            else:
                smoothed_ball = ball_data if ball_data else BallData()

            smoothed.append(smoothed_ball)

        return smoothed

    def reset(self):
        """Reset smoothing history."""
        self.player_position_history = {
            1: deque(maxlen=self.window_size),
            2: deque(maxlen=self.window_size),
        }
        self.ball_position_history = deque(maxlen=self.window_size)


class MissingDataHandler:
    """Handles missing or invalid data through interpolation (supports both streaming and batch modes)."""

    def __init__(self, max_interpolation_frames: int = 10):
        """
        Initialize missing data handler.

        Args:
            max_interpolation_frames: Maximum number of frames to interpolate
        """
        self.max_interpolation_frames = max_interpolation_frames
        self.last_valid_player_data = {1: None, 2: None}
        self.last_valid_ball_data = None
        self.missing_frame_count = {1: 0, 2: 0, "ball": 0}

    def handle_missing_player(
        self,
        player_data_list: List[PlayerData],
        player_id: int,
        validation_results: List[bool],
    ) -> List[PlayerData]:
        """
        Interpolate missing player positions using bidirectional search (batch mode).

        Args:
            player_data_list: List of PlayerData for all frames
            player_id: Player ID
            validation_results: List of boolean validation results

        Returns:
            List of PlayerData with interpolated positions
        """
        interpolated = []

        for i, (player_data, is_valid) in enumerate(
            zip(player_data_list, validation_results)
        ):
            if is_valid:
                interpolated.append(player_data)
            else:
                # Find nearest valid frames (bidirectional)
                before_idx = self._find_valid_player_before(player_data_list, i)
                after_idx = self._find_valid_player_after(player_data_list, i)

                interp_player = self._interpolate_player(
                    player_data_list, i, before_idx, after_idx, player_data
                )
                interpolated.append(interp_player)

        return interpolated

    def handle_missing_ball(
        self, ball_data_list: List[BallData], validation_results: List[bool]
    ) -> List[BallData]:
        """
        Interpolate missing ball positions using bidirectional search (batch mode).

        Args:
            ball_data_list: List of BallData for all frames
            validation_results: List of boolean validation results

        Returns:
            List of BallData with interpolated positions
        """
        interpolated = []

        for i, (ball_data, is_valid) in enumerate(
            zip(ball_data_list, validation_results)
        ):
            if is_valid:
                interpolated.append(ball_data)
            else:
                # Find nearest valid frames (bidirectional)
                before_idx = self._find_valid_ball_before(ball_data_list, i)
                after_idx = self._find_valid_ball_after(ball_data_list, i)

                interp_ball = self._interpolate_ball(
                    ball_data_list, i, before_idx, after_idx, ball_data
                )
                interpolated.append(interp_ball)

        return interpolated

    def _find_valid_player_before(
        self, player_data_list: List[PlayerData], current_idx: int
    ) -> Optional[int]:
        """Find nearest valid player frame before current index."""
        for i in range(
            current_idx - 1,
            max(-1, current_idx - self.max_interpolation_frames - 1),
            -1,
        ):
            if player_data_list[i] and player_data_list[i].is_valid():
                return i
        return None

    def _find_valid_player_after(
        self, player_data_list: List[PlayerData], current_idx: int
    ) -> Optional[int]:
        """Find nearest valid player frame after current index."""
        for i in range(
            current_idx + 1,
            min(len(player_data_list), current_idx + self.max_interpolation_frames + 1),
        ):
            if player_data_list[i] and player_data_list[i].is_valid():
                return i
        return None

    def _find_valid_ball_before(
        self, ball_data_list: List[BallData], current_idx: int
    ) -> Optional[int]:
        """Find nearest valid ball frame before current index."""
        for i in range(
            current_idx - 1,
            max(-1, current_idx - self.max_interpolation_frames - 1),
            -1,
        ):
            if ball_data_list[i] and ball_data_list[i].is_valid():
                return i
        return None

    def _find_valid_ball_after(
        self, ball_data_list: List[BallData], current_idx: int
    ) -> Optional[int]:
        """Find nearest valid ball frame after current index."""
        for i in range(
            current_idx + 1,
            min(len(ball_data_list), current_idx + self.max_interpolation_frames + 1),
        ):
            if ball_data_list[i] and ball_data_list[i].is_valid():
                return i
        return None

    def _interpolate_player(
        self,
        player_data_list: List[PlayerData],
        current_idx: int,
        before_idx: Optional[int],
        after_idx: Optional[int],
        current_data: PlayerData,
    ) -> PlayerData:
        """Interpolate player data between valid frames."""
        if before_idx is not None and after_idx is not None:
            # Linear interpolation
            before_player = player_data_list[before_idx]
            after_player = player_data_list[after_idx]

            t = (current_idx - before_idx) / (after_idx - before_idx)
            interp_pos = (
                before_player.position[0]
                + t * (after_player.position[0] - before_player.position[0]),
                before_player.position[1]
                + t * (after_player.position[1] - before_player.position[1]),
            )

            return PlayerData(
                player_id=current_data.player_id,
                position=interp_pos,
                real_position=current_data.real_position,
                bbox=current_data.bbox,
                confidence=current_data.confidence,
                keypoints=current_data.keypoints,
            )
        elif before_idx is not None:
            return player_data_list[before_idx]
        elif after_idx is not None:
            return player_data_list[after_idx]
        else:
            return current_data

    def _interpolate_ball(
        self,
        ball_data_list: List[BallData],
        current_idx: int,
        before_idx: Optional[int],
        after_idx: Optional[int],
        current_data: BallData,
    ) -> BallData:
        """Interpolate ball data between valid frames."""
        if before_idx is not None and after_idx is not None:
            # Linear interpolation
            before_ball = ball_data_list[before_idx]
            after_ball = ball_data_list[after_idx]

            t = (current_idx - before_idx) / (after_idx - before_idx)
            interp_pos = (
                int(
                    before_ball.position[0]
                    + t * (after_ball.position[0] - before_ball.position[0])
                ),
                int(
                    before_ball.position[1]
                    + t * (after_ball.position[1] - before_ball.position[1])
                ),
            )

            return BallData(position=interp_pos, confidence=None)
        elif before_idx is not None:
            return ball_data_list[before_idx]
        elif after_idx is not None:
            return ball_data_list[after_idx]
        else:
            return current_data

    def reset(self):
        """Reset handler state."""
        self.last_valid_player_data = {1: None, 2: None}
        self.last_valid_ball_data = None
        self.missing_frame_count = {1: 0, 2: 0, "ball": 0}
