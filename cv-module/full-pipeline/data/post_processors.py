"""Post-processing utilities for smoothing and filtering data."""

from collections import deque
from typing import Optional, Tuple, List
import numpy as np
from scipy.signal import medfilt, savgol_filter
from .data_models import PlayerData, BallData


class TemporalSmoother:
    """Applies temporal smoothing to tracking data (supports both streaming and batch modes)."""

    def __init__(
        self,
        window_size: int = 5,
        median_window: int = 5,
        savgol_window: int = 11,
        savgol_poly: int = 3,
    ):
        """
        Initialize temporal smoother.

        Args:
            window_size: Number of frames to use for smoothing
            median_window: Window size for median filter (must be odd)
            savgol_window: Window length for Savitzky-Golay filter (must be odd)
            savgol_poly: Polynomial order for Savitzky-Golay filter
        """
        self.window_size = window_size
        self.median_window = (
            median_window if median_window % 2 == 1 else median_window + 1
        )
        self.savgol_window = (
            savgol_window if savgol_window % 2 == 1 else savgol_window + 1
        )
        self.savgol_poly = savgol_poly
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
                    stroke_type=player_data.stroke_type,
                )
            else:
                smoothed_player = PlayerData(player_id=player_id)

            smoothed.append(smoothed_player)

        return smoothed

    def smooth_ball_positions(self, ball_data_list: List[BallData]) -> List[BallData]:
        """
        Smooth ball positions using median filter followed by Savitzky-Golay filter (batch mode).

        Args:
            ball_data_list: List of BallData for all frames

        Returns:
            List of BallData with smoothed positions
        """
        # Extract positions
        ball_positions = []
        for ball_data in ball_data_list:
            if ball_data and ball_data.is_valid() and ball_data.position:
                ball_positions.append(ball_data.position)
            else:
                ball_positions.append((None, None))

        # Separate x and y coordinates
        x_coords = np.array(
            [pos[0] if pos[0] is not None else np.nan for pos in ball_positions]
        )
        y_coords = np.array(
            [pos[1] if pos[1] is not None else np.nan for pos in ball_positions]
        )

        # Check if we have enough valid data to smooth
        valid_count = np.sum(~np.isnan(x_coords))
        if valid_count < max(self.median_window, self.savgol_window):
            # Not enough data to smooth, return original
            return ball_data_list

        # Stage 1: Apply median filter to remove outlier spikes
        x_median = medfilt(x_coords, kernel_size=self.median_window)
        y_median = medfilt(y_coords, kernel_size=self.median_window)

        # Stage 2: Apply Savitzky-Golay filter for smooth curves
        x_smoothed = savgol_filter(
            x_median, window_length=self.savgol_window, polyorder=self.savgol_poly
        )
        y_smoothed = savgol_filter(
            y_median, window_length=self.savgol_window, polyorder=self.savgol_poly
        )

        # Create BallData list with smoothed positions
        smoothed = []
        for i, ball_data in enumerate(ball_data_list):
            if not np.isnan(x_smoothed[i]) and not np.isnan(y_smoothed[i]):
                smoothed_pos = (int(x_smoothed[i]), int(y_smoothed[i]))
                smoothed.append(
                    BallData(
                        position=smoothed_pos,
                        confidence=ball_data.confidence if ball_data else None,
                        is_wall_hit=ball_data.is_wall_hit if ball_data else False,
                    )
                )
            else:
                # Keep original if smoothing failed
                smoothed.append(ball_data if ball_data else BallData())

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
        Interpolate missing ball positions using numpy interpolation (batch mode).

        Args:
            ball_data_list: List of BallData for all frames
            validation_results: List of boolean validation results

        Returns:
            List of BallData with interpolated positions
        """
        # Extract positions, using None for invalid entries
        ball_positions = []
        for ball_data, is_valid in zip(ball_data_list, validation_results):
            if is_valid and ball_data and ball_data.position:
                ball_positions.append(ball_data.position)
            else:
                ball_positions.append((None, None))

        # Separate x and y coordinates
        x_coords = np.array(
            [pos[0] if pos[0] is not None else np.nan for pos in ball_positions]
        )
        y_coords = np.array(
            [pos[1] if pos[1] is not None else np.nan for pos in ball_positions]
        )

        # Interpolate missing values (NaNs)
        valid_x_indices = ~np.isnan(x_coords)
        valid_y_indices = ~np.isnan(y_coords)

        if np.sum(valid_x_indices) > 1:
            x_coords_interp = np.interp(
                np.arange(len(x_coords)),
                np.where(valid_x_indices)[0],
                x_coords[valid_x_indices],
            )
        else:
            x_coords_interp = x_coords

        if np.sum(valid_y_indices) > 1:
            y_coords_interp = np.interp(
                np.arange(len(y_coords)),
                np.where(valid_y_indices)[0],
                y_coords[valid_y_indices],
            )
        else:
            y_coords_interp = y_coords

        # Create BallData list with interpolated positions
        interpolated = []
        for i, ball_data in enumerate(ball_data_list):
            if not np.isnan(x_coords_interp[i]) and not np.isnan(y_coords_interp[i]):
                interp_pos = (int(x_coords_interp[i]), int(y_coords_interp[i]))
                interpolated.append(
                    BallData(
                        position=interp_pos,
                        confidence=ball_data.confidence if ball_data else None,
                        is_wall_hit=ball_data.is_wall_hit if ball_data else False,
                    )
                )
            else:
                # Keep original if interpolation failed
                interpolated.append(ball_data if ball_data else BallData())

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
                stroke_type=current_data.stroke_type,
            )
        elif before_idx is not None:
            return player_data_list[before_idx]
        elif after_idx is not None:
            return player_data_list[after_idx]
        else:
            return current_data

    def reset(self):
        """Reset handler state."""
        self.last_valid_player_data = {1: None, 2: None}
        self.last_valid_ball_data = None
        self.missing_frame_count = {1: 0, 2: 0, "ball": 0}
