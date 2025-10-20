"""Data collector for aggregating and processing pipeline outputs."""

from typing import Optional, List
import numpy as np
import cv2
from scipy.signal import find_peaks
from copy import deepcopy
from .data_models import FrameData, PlayerData, BallData, CourtData, RallyData
from .validators import DataValidator
from .post_processors import TemporalSmoother, MissingDataHandler


class DataCollector:
    """
    Collects, validates, and post-processes data from all sub-pipelines.

    Responsibilities:
    - Aggregate raw outputs from sub-pipelines
    - Validate data quality (during collection)
    - Store raw data during collection
    - Apply temporal smoothing (after collection is complete)
    - Handle missing data (after collection is complete)
    - Provide clean, structured data for analysis
    """

    def __init__(
        self,
        enable_smoothing: bool = True,
        smoothing_window: int = 5,
        median_window: int = 5,
        savgol_window: int = 11,
        savgol_poly: int = 3,
        enable_validation: bool = True,
        min_confidence: float = 0.3,
        max_position_change: float = 200.0,
        handle_missing_data: bool = True,
        max_interpolation_frames: int = 10,
        prominence: float = 50.0,
        width: int = 3,
        min_distance: int = 20,
        min_states_duration: dict = {},
    ):
        """
        Initialize data collector.

        Args:
            enable_smoothing: Whether to apply temporal smoothing (post-processing)
            smoothing_window: Window size for smoothing
            median_window: Window size for median filter (if used)
            savgol_window: Window size for Savitzky-Golay filter (if used)
            savgol_poly: Polynomial order for Savitzky-Golay filter (if used)
            enable_validation: Whether to validate data during collection
            min_confidence: Minimum confidence threshold
            max_position_change: Maximum position change between frames
            handle_missing_data: Whether to handle missing data (post-processing)
            max_interpolation_frames: Maximum frames to interpolate
            prominence: Minimum prominence for peak detection
            width: Minimum width for peak detection
            min_distance: Minimum distance between peaks
            min_states_duration: Minimum duration for each rally state
        """
        self.enable_smoothing = enable_smoothing
        self.smoothing_window = smoothing_window
        self.median_window = median_window
        self.savgol_window = savgol_window
        self.savgol_poly = savgol_poly
        self.enable_validation = enable_validation
        self.handle_missing_data = handle_missing_data
        self.max_interpolation_frames = max_interpolation_frames
        self.min_confidence = min_confidence
        self.max_position_change = max_position_change
        self.prominence = prominence
        self.width = width
        self.min_distance = min_distance
        self.min_states_duration = min_states_duration

        # Initialize validator (used during collection)
        self.validator = (
            DataValidator(
                min_confidence=min_confidence,
                max_position_change=max_position_change,
            )
            if enable_validation
            else None
        )

        self.smoother = TemporalSmoother(
            window_size=self.smoothing_window,
            median_window=self.median_window,
            savgol_window=self.savgol_window,
            savgol_poly=self.savgol_poly,
            min_states_duration=self.min_states_duration,
        )

        self.interpolator = MissingDataHandler(
            max_interpolation_frames=self.max_interpolation_frames
        )

        # Storage
        self.raw_frame_history: List[FrameData] = []  # Raw data

    def collect_frame_data(
        self,
        frame_number: int,
        timestamp: float,
        court_data: dict,
        player_results: dict,
        ball_position: tuple,
        rally_state: str,
        stroke_data: dict,
    ) -> FrameData:
        """
        Collect raw data from all pipelines for a single frame.

        Note: Smoothing and missing data handling are NOT applied during collection.
        Call post_process() after all frames are collected to apply these operations.

        Args:
            frame_number: Current frame number
            timestamp: Frame timestamp in seconds
            court_data: Court detection data (homographies, keypoints, calibrated)
            player_results: Player tracking results from PlayerTracker
            ball_position: Ball position (x, y) or (None, None)
            rally_state: Rally state from RallyStateDetector
            stroke_data: Stroke detection data from StrokeDetector

        Returns:
            Raw FrameData object (not smoothed or interpolated)
        """
        # Aggregate raw data into structured models
        frame_data = self._aggregate_data(
            frame_number,
            timestamp,
            court_data,
            player_results,
            ball_position,
            rally_state,
            stroke_data,
        )

        # Store raw data
        self.raw_frame_history.append(frame_data)

        return frame_data

    def _aggregate_data(
        self,
        frame_number: int,
        timestamp: float,
        court_data: dict,
        player_results: dict,
        ball_position: tuple,
        rally_state: str,
        stroke_data: dict,
    ) -> FrameData:
        """Aggregate raw pipeline outputs into structured data models."""
        # Court data
        court = CourtData(
            homographies=court_data.get("homographies"),
            keypoints=court_data.get("keypoints"),
            is_calibrated=court_data.get("is_calibrated", False),
        )

        # Player 1 data
        p1_raw = player_results.get(1, {})
        player1 = PlayerData(
            player_id=1,
            position=p1_raw.get("position"),
            real_position=(
                tuple(p1_raw["real_position"])
                if p1_raw.get("real_position") is not None
                else None
            ),
            bbox=tuple(p1_raw["bbox"]) if p1_raw.get("bbox") else None,
            confidence=p1_raw.get("confidence"),
            keypoints=p1_raw.get("keypoints"),
            stroke_type=stroke_data[1]["stroke"],
        )

        # Player 2 data
        p2_raw = player_results.get(2, {})
        player2 = PlayerData(
            player_id=2,
            position=p2_raw.get("position"),
            real_position=(
                tuple(p2_raw["real_position"])
                if p2_raw.get("real_position") is not None
                else None
            ),
            bbox=tuple(p2_raw["bbox"]) if p2_raw.get("bbox") else None,
            confidence=p2_raw.get("confidence"),
            keypoints=p2_raw.get("keypoints"),
            stroke_type=stroke_data[2]["stroke"],
        )

        # Ball data
        ball = BallData(
            position=ball_position if ball_position[0] is not None else None,
            confidence=None,  # Ball tracker doesn't provide confidence yet
        )

        return FrameData(
            frame_number=frame_number,
            timestamp=timestamp,
            court=court,
            player1=player1,
            player2=player2,
            ball=ball,
            rally_state=rally_state,
        )

    def post_process(self) -> List[RallyData]:
        """
        Apply post-processing to all collected frames.

        This method should be called after all frames have been collected.
        It applies smoothing and missing data handling to the entire dataset.

        Returns:
            List of processed RallyData objects
        """
        rally_states = [f.rally_state for f in self.raw_frame_history]
        rally_states_smoothed = self.smoother.smooth_rally_states(rally_states)

        for i, frame in enumerate(self.raw_frame_history):
            frame.rally_state = rally_states_smoothed[i]

        rallies = self._segment_rallies(self.raw_frame_history)
        for rally in rallies:
            self._post_process_rally(rally.rally_frames)

        return rallies

    def _post_process_rally(self, rally_frames: List[FrameData]) -> List[FrameData]:
        """Post-process a single rally (subset of frames).

        Modifies frames in-place.

        Args:
            rally_frames: List of frames to post-process (modified in-place)

        Returns:
            The same list (for convenience)
        """
        # 1. Validate all frames
        if self.enable_validation and self.validator:
            validation_results = self._validate_all_frames(rally_frames)
        else:
            validation_results = None

        # 2. Handle missing data through interpolation
        if self.handle_missing_data and validation_results:
            self._handle_missing_values(rally_frames, validation_results)

        # 3. Apply temporal smoothing
        if self.enable_smoothing:
            self._smooth_all_frames(rally_frames)

        # 4. Detect front wall hits
        self._detect_front_wall_hits(rally_frames)

        return rally_frames

    def _segment_rallies(self, frames: List[FrameData]) -> List[RallyData]:
        """Segment frames into rallies based on rally state."""
        rallies = []
        current_rally = []
        in_rally = False

        for frame in frames:
            if frame.rally_state == "start" and not in_rally:
                in_rally = True
                current_rally = [frame]
            elif frame.rally_state == "active" and in_rally:
                current_rally.append(frame)
            elif frame.rally_state == "end" and in_rally:
                current_rally.append(frame)
                rally = RallyData(
                    rally_frames=current_rally,
                    start_frame=current_rally[0].frame_number,
                    end_frame=current_rally[-1].frame_number,
                )
                rallies.append(rally)
                in_rally = False
                current_rally = []
            else:
                if in_rally:
                    current_rally.append(frame)

        if in_rally and current_rally:
            rally = RallyData(
                rally_frames=current_rally,
                start_frame=current_rally[0].frame_number,
                end_frame=current_rally[-1].frame_number,
            )
            rallies.append(rally)

        return rallies

    def _validate_all_frames(self, frames: List[FrameData]) -> List[dict]:
        """Validate all frames and return validation results."""
        results = []
        for i, frame in enumerate(frames):
            prev_frame = frames[i - 1] if i > 0 else None
            validation = self.validator.validate_frame_data(frame, prev_frame)
            results.append(validation)
        return results

    def _handle_missing_values(
        self, frames: List[FrameData], validation_results: List[dict]
    ) -> None:
        """Interpolate missing data using MissingDataHandler.

        Modifies frames in-place.

        Args:
            frames: List of frames to interpolate (modified in-place)
            validation_results: Validation results for each frame
        """
        # Extract player and ball data
        p1_data = [f.player1 for f in frames]
        p2_data = [f.player2 for f in frames]
        ball_data = [f.ball for f in frames]

        p1_valid = [v["player1_valid"] for v in validation_results]
        p2_valid = [v["player2_valid"] for v in validation_results]
        ball_valid = [v["ball_valid"] for v in validation_results]

        # Interpolate each dataset
        p1_interpolated = self.interpolator.handle_missing_player(
            p1_data, player_id=1, validation_results=p1_valid
        )
        p2_interpolated = self.interpolator.handle_missing_player(
            p2_data, player_id=2, validation_results=p2_valid
        )
        ball_interpolated = self.interpolator.handle_missing_ball(
            ball_data, validation_results=ball_valid
        )

        # Update frames in-place
        for i, frame in enumerate(frames):
            frame.player1 = p1_interpolated[i]
            frame.player2 = p2_interpolated[i]
            frame.ball = ball_interpolated[i]

    def _smooth_all_frames(self, frames: List[FrameData]) -> None:
        """Apply temporal smoothing using TemporalSmoother.

        Modifies frames in-place.

        Args:
            frames: List of frames to smooth (modified in-place)
        """
        # Extract player and ball data
        p1_data = [f.player1 for f in frames]
        p2_data = [f.player2 for f in frames]
        ball_data = [f.ball for f in frames]

        # Smooth each dataset
        p1_smoothed = self.smoother.smooth_player_positions(p1_data, player_id=1)
        p2_smoothed = self.smoother.smooth_player_positions(p2_data, player_id=2)
        ball_smoothed = self.smoother.smooth_ball_positions(ball_data)

        # Update frames in-place
        for i, frame in enumerate(frames):
            frame.player1 = p1_smoothed[i]
            frame.player2 = p2_smoothed[i]
            frame.ball = ball_smoothed[i]

    def _detect_front_wall_hits(self, frames: List[FrameData]):
        """Detect front wall hits using local minima in Y-coordinate.

        Front wall hits appear as valleys (local minima) in the Y-coordinate curve.
        The algorithm finds these minima and validates them based on prominence and width.

        This function modifies the frames in-place by setting ball.is_wall_hit = True
        for frames where a wall hit is detected.

        Args:
            frames: List of frame data (modified in-place)
            prominence: Minimum depth of valley in pixels (higher = only significant hits)
            width: Minimum width of valley in frames (filters noise)
            min_distance: Minimum frames between consecutive hits (prevents duplicates)
        """
        if len(frames) < self.width:
            return

        ball_data = [f.ball for f in frames]
        positions = [f.position for f in ball_data]

        # Extract Y coordinates
        y_coords = np.array([p[1] for p in positions])

        # Invert Y to find minima as peaks
        # (find_peaks finds maxima, so we negate to find minima)
        inverted_y = -y_coords

        # Find peaks in inverted signal (= minima in original signal)
        peaks, properties = find_peaks(
            inverted_y,
            prominence=self.prominence,  # Minimum valley depth
            width=self.width,  # Minimum valley width
            distance=self.min_distance,  # Minimum spacing between hits
        )

        # Update frame data
        for peak_idx in peaks:
            # Set is_wall_hit flag on the corresponding frame
            frames[peak_idx].ball.is_wall_hit = True
            frames[peak_idx].ball.ball_hit_real_position = cv2.perspectiveTransform(
                np.array([[positions[peak_idx]]], dtype=np.float32),
                frames[peak_idx].court.homographies["wall"],
            )[0][0]

    def reset(self):
        """Reset collector state."""
        self.raw_frame_history = []
