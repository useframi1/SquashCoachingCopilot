"""Data collector for aggregating and processing pipeline outputs."""

from typing import Optional, List
import numpy as np
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
        enable_validation: bool = True,
        min_confidence: float = 0.3,
        max_position_change: float = 200.0,
        handle_missing_data: bool = True,
        max_interpolation_frames: int = 10,
    ):
        """
        Initialize data collector.

        Args:
            enable_smoothing: Whether to apply temporal smoothing (post-processing)
            smoothing_window: Window size for smoothing
            enable_validation: Whether to validate data during collection
            min_confidence: Minimum confidence threshold
            max_position_change: Maximum position change between frames
            handle_missing_data: Whether to handle missing data (post-processing)
            max_interpolation_frames: Maximum frames to interpolate
        """
        self.enable_smoothing = enable_smoothing
        self.smoothing_window = smoothing_window
        self.enable_validation = enable_validation
        self.handle_missing_data = handle_missing_data
        self.max_interpolation_frames = max_interpolation_frames
        self.min_confidence = min_confidence
        self.max_position_change = max_position_change

        # Initialize validator (used during collection)
        self.validator = (
            DataValidator(
                min_confidence=min_confidence,
                max_position_change=max_position_change,
            )
            if enable_validation
            else None
        )

        # Storage
        self.raw_frame_history: List[FrameData] = []  # Raw data
        self.processed_frame_history: Optional[List[FrameData]] = None  # Processed data
        self.last_frame_data: Optional[FrameData] = None
        self.is_post_processed: bool = False

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
        self.last_frame_data = frame_data

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

    def post_process(self) -> List[FrameData]:
        """
        Apply post-processing to all collected frames.

        This method should be called after all frames have been collected.
        It applies smoothing and missing data handling to the entire dataset.

        Returns:
            List of processed RallyData objects
        """
        rallies = self._segment_rallies()
        processed_rallies = []
        for rally in rallies:
            processed = self._post_process_rally(rally.rally_frames)
            processed_rallies.append(processed)

        # Flatten processed rallies into a single list of frames
        self.processed_frame_history = [
            frame for rally in processed_rallies for frame in rally
        ]
        self.is_post_processed = True
        return self.processed_frame_history

    def _post_process_rally(self, rally_frames: List[FrameData]) -> List[FrameData]:
        """Post-process a single rally (subset of frames)."""
        processed_frames = [frame for frame in rally_frames]

        # 1. Validate all frames
        if self.enable_validation and self.validator:
            validation_results = self._validate_all_frames(processed_frames)
        else:
            validation_results = None

        # 2. Handle missing data through interpolation
        if self.handle_missing_data and validation_results:
            processed_frames = self._handle_missing_values(
                processed_frames, validation_results
            )

        # 3. Apply temporal smoothing
        if self.enable_smoothing:
            processed_frames = self._smooth_all_frames(processed_frames)

        return processed_frames

    def _segment_rallies(self) -> List[RallyData]:
        """Segment frames into rallies based on rally state."""
        rallies = []
        current_rally = []
        in_rally = False

        for frame in self.raw_frame_history:
            if frame.rally_state == "start" and not in_rally:
                in_rally = True
                current_rally = [frame]
            elif frame.rally_state == "active" and in_rally:
                current_rally.append(frame)
            elif frame.rally_state == "end" and in_rally:
                current_rally.append(frame)
                rally = RallyData(rally_frames=current_rally)
                rallies.append(rally)
                in_rally = False
                current_rally = []
            else:
                if in_rally:
                    current_rally.append(frame)

        if in_rally and current_rally:
            rally = RallyData(rally_frames=current_rally)
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
    ) -> List[FrameData]:
        """Interpolate missing data using MissingDataHandler."""
        # Create interpolator
        interpolator = MissingDataHandler(
            max_interpolation_frames=self.max_interpolation_frames
        )

        # Extract player and ball data
        p1_data = [f.player1 for f in frames]
        p2_data = [f.player2 for f in frames]
        ball_data = [f.ball for f in frames]

        p1_valid = [v["player1_valid"] for v in validation_results]
        p2_valid = [v["player2_valid"] for v in validation_results]
        ball_valid = [v["ball_valid"] for v in validation_results]

        # Interpolate each dataset
        p1_interpolated = interpolator.handle_missing_player(
            p1_data, player_id=1, validation_results=p1_valid
        )
        p2_interpolated = interpolator.handle_missing_player(
            p2_data, player_id=2, validation_results=p2_valid
        )
        ball_interpolated = interpolator.handle_missing_ball(
            ball_data, validation_results=ball_valid
        )

        # Reconstruct frames
        interpolated_frames = []
        for i, frame in enumerate(frames):
            new_frame = FrameData(
                frame_number=frame.frame_number,
                timestamp=frame.timestamp,
                court=frame.court,
                player1=p1_interpolated[i],
                player2=p2_interpolated[i],
                ball=ball_interpolated[i],
                rally_state=frame.rally_state,
            )
            interpolated_frames.append(new_frame)

        return interpolated_frames

    def _smooth_all_frames(self, frames: List[FrameData]) -> List[FrameData]:
        """Apply temporal smoothing using TemporalSmoother."""
        # Create smoother
        smoother = TemporalSmoother(window_size=self.smoothing_window)

        # Extract player and ball data
        p1_data = [f.player1 for f in frames]
        p2_data = [f.player2 for f in frames]
        ball_data = [f.ball for f in frames]

        # Smooth each dataset
        p1_smoothed = smoother.smooth_player_positions(p1_data, player_id=1)
        p2_smoothed = smoother.smooth_player_positions(p2_data, player_id=2)
        ball_smoothed = smoother.smooth_ball_positions(ball_data)

        # Reconstruct frames
        smoothed_frames = []
        for i, frame in enumerate(frames):
            new_frame = FrameData(
                frame_number=frame.frame_number,
                timestamp=frame.timestamp,
                court=frame.court,
                player1=p1_smoothed[i],
                player2=p2_smoothed[i],
                ball=ball_smoothed[i],
                rally_state=frame.rally_state,
            )
            smoothed_frames.append(new_frame)

        return smoothed_frames

    def get_frame_history(
        self, num_frames: Optional[int] = None, raw: bool = False
    ) -> List[FrameData]:
        """
        Get frame history.

        Args:
            num_frames: Number of recent frames to return (None for all)
            raw: If True, return raw (unprocessed) data. If False, return processed data if available.

        Returns:
            List of FrameData objects
        """
        # Use processed frames if available and not requesting raw
        if (
            not raw
            and self.is_post_processed
            and self.processed_frame_history is not None
        ):
            frames = self.processed_frame_history
        else:
            frames = self.raw_frame_history

        if num_frames is None:
            return frames
        return frames[-num_frames:]

    def reset(self):
        """Reset collector state."""
        self.raw_frame_history = []
        self.processed_frame_history = None
        self.last_frame_data = None
        self.is_post_processed = False
