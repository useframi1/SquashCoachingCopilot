"""
Data models for the squash coaching copilot pipeline.

This module defines all input/output models for the pipeline stages,
designed for a DataFrame-based data flow architecture.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from squashcopilot.common.types.base import Frame
from squashcopilot.common.types.geometry import Point2D, BoundingBox, Homography


# =============================================================================
# VIDEO METADATA
# =============================================================================


@dataclass
class VideoMetadata:
    """Metadata about the video being analyzed."""

    filepath: Path
    fps: float
    total_frames: int
    width: int
    height: int
    duration_seconds: float


# =============================================================================
# STAGE 1: COURT CALIBRATION
# =============================================================================


@dataclass
class CourtCalibrationInput:
    """Input for court calibration module (Stage 1)."""

    frame: Frame


@dataclass
class CourtCalibrationOutput:
    """Output from court calibration module (Stage 1)."""

    frame_number: int
    calibration_success: bool

    # Homography matrices (3x3)
    floor_homography: np.ndarray
    wall_homography: np.ndarray

    # Court keypoints in pixels
    court_keypoints: Dict[str, Point2D]

    # Ball detection configuration
    is_black_ball: bool

    def pixel_to_floor(self, point: Point2D) -> Point2D:
        """Transform pixel coordinates to floor meters."""
        if self.floor_homography is None:
            return point
        pt_homo = np.array([point.x, point.y, 1.0])
        transformed = self.floor_homography @ pt_homo
        if transformed[2] != 0:
            x = transformed[0] / transformed[2]
            y = transformed[1] / transformed[2]
        else:
            x, y = transformed[0], transformed[1]
        return Point2D(x=float(x), y=float(y))

    def pixel_to_wall(self, point: Point2D) -> Point2D:
        """Transform pixel coordinates to wall meters."""
        if self.wall_homography is None:
            return point
        pt_homo = np.array([point.x, point.y, 1.0])
        transformed = self.wall_homography @ pt_homo
        if transformed[2] != 0:
            x = transformed[0] / transformed[2]
            y = transformed[1] / transformed[2]
        else:
            x, y = transformed[0], transformed[1]
        return Point2D(x=float(x), y=float(y))


# =============================================================================
# STAGE 2a: PLAYER TRACKING
# =============================================================================


@dataclass
class PlayerTrackingInput:
    """Input for player tracking module (Stage 2a) - per frame."""

    frame: Frame
    calibration: CourtCalibrationOutput


@dataclass
class PlayerTrackingOutput:
    """Output from player tracking module (Stage 2a) - per frame."""

    frame_number: int
    timestamp: float

    # Player 1 detection
    player_1_detected: bool
    player_1_x_pixel: Optional[float]  # pixels
    player_1_y_pixel: Optional[float]  # pixels
    player_1_x_meter: Optional[float]  # meters (court coordinates)
    player_1_y_meter: Optional[float]  # meters (court coordinates)
    player_1_confidence: float
    player_1_bbox: Optional[BoundingBox]
    player_1_keypoints: Optional[np.ndarray]  # Shape: (12, 2) - body keypoints only

    # Player 2 detection
    player_2_detected: bool
    player_2_x_pixel: Optional[float]  # pixels
    player_2_y_pixel: Optional[float]  # pixels
    player_2_x_meter: Optional[float]  # meters (court coordinates)
    player_2_y_meter: Optional[float]  # meters (court coordinates)
    player_2_confidence: float
    player_2_bbox: Optional[BoundingBox]
    player_2_keypoints: Optional[np.ndarray]  # Shape: (12, 2) - body keypoints only


def player_tracking_output_to_dict(output: PlayerTrackingOutput) -> Dict[str, Any]:
    """Convert PlayerTrackingOutput to dictionary for DataFrame row."""
    return {
        "frame_number": output.frame_number,
        "timestamp": output.timestamp,
        "player_1_detected": output.player_1_detected,
        "player_1_x_pixel": output.player_1_x_pixel,
        "player_1_y_pixel": output.player_1_y_pixel,
        "player_1_x_meter": output.player_1_x_meter,
        "player_1_y_meter": output.player_1_y_meter,
        "player_1_confidence": output.player_1_confidence,
        "player_2_detected": output.player_2_detected,
        "player_2_x_pixel": output.player_2_x_pixel,
        "player_2_y_pixel": output.player_2_y_pixel,
        "player_2_x_meter": output.player_2_x_meter,
        "player_2_y_meter": output.player_2_y_meter,
        "player_2_confidence": output.player_2_confidence,
    }


def player_tracking_outputs_to_dataframe(
    outputs: List[PlayerTrackingOutput],
) -> tuple[pd.DataFrame, Dict[str, List]]:
    """Convert list of PlayerTrackingOutput to DataFrame."""
    rows = [player_tracking_output_to_dict(out) for out in outputs]
    df = pd.DataFrame(rows).set_index("frame_number")

    # Extract complex data
    complex_data = {
        "player_1_keypoints": [out.player_1_keypoints for out in outputs],
        "player_2_keypoints": [out.player_2_keypoints for out in outputs],
        "player_1_bboxes": [out.player_1_bbox for out in outputs],
        "player_2_bboxes": [out.player_2_bbox for out in outputs],
    }

    return df, complex_data


@dataclass
class PlayerPostprocessingInput:
    """Input for player postprocessing (interpolation + smoothing)."""

    df: pd.DataFrame
    player_keypoints: Dict[int, List[Optional[np.ndarray]]]  # {1: [...], 2: [...]}
    player_bboxes: Dict[int, List[Optional[BoundingBox]]]  # {1: [...], 2: [...]}

    def __post_init__(self):
        """Validate required columns."""
        required_cols = [
            "player_1_x_pixel",
            "player_1_y_pixel",
            "player_1_detected",
            "player_2_x_pixel",
            "player_2_y_pixel",
            "player_2_detected",
        ]
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


@dataclass
class PlayerPostprocessingOutput:
    """Output from player postprocessing."""

    df: pd.DataFrame
    player_keypoints: Dict[int, List[np.ndarray]]  # Interpolated, no None
    player_bboxes: Dict[int, List[BoundingBox]]  # Interpolated, no None

    # Metadata
    num_player_1_gaps_filled: int
    num_player_2_gaps_filled: int


# =============================================================================
# STAGE 2b: BALL TRACKING
# =============================================================================


@dataclass
class BallTrackingInput:
    """Input for ball tracking module (Stage 2b) - per frame."""

    frame: Frame


@dataclass
class BallTrackingOutput:
    """Output from ball tracking module (Stage 2b) - per frame."""

    frame_number: int
    timestamp: float

    # Ball detection
    ball_detected: bool
    ball_x: Optional[float]  # pixels
    ball_y: Optional[float]  # pixels
    ball_confidence: float


def ball_tracking_output_to_dict(output: BallTrackingOutput) -> Dict[str, Any]:
    """Convert BallTrackingOutput to dictionary for DataFrame row."""
    return {
        "frame_number": output.frame_number,
        "timestamp": output.timestamp,
        "ball_detected": output.ball_detected,
        "ball_x": output.ball_x,
        "ball_y": output.ball_y,
        "ball_confidence": output.ball_confidence,
    }


def ball_tracking_outputs_to_dataframe(
    outputs: List[BallTrackingOutput],
) -> pd.DataFrame:
    """Convert list of BallTrackingOutput to DataFrame."""
    rows = [ball_tracking_output_to_dict(out) for out in outputs]
    df = pd.DataFrame(rows).set_index("frame_number")
    return df


@dataclass
class BallPostprocessingInput:
    """Input for ball postprocessing (interpolation + smoothing)."""

    df: pd.DataFrame

    def __post_init__(self):
        """Validate required columns."""
        required_cols = [
            "ball_x",
            "ball_y",
            "ball_detected",
        ]
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


@dataclass
class BallPostprocessingOutput:
    """Output from ball postprocessing."""

    df: pd.DataFrame

    # Metadata
    num_ball_outliers: int
    num_ball_gaps_filled: int


# =============================================================================
# STAGE 4: RALLY SEGMENTATION
# =============================================================================


@dataclass
class RallySegment:
    """Single rally segment."""

    rally_id: int
    start_frame: int
    end_frame: int

    @property
    def num_frames(self) -> int:
        """Number of frames in this rally."""
        return self.end_frame - self.start_frame + 1

    def contains(self, frame_number: int) -> bool:
        """Check if frame is within this rally."""
        return self.start_frame <= frame_number <= self.end_frame


@dataclass
class RallySegmentationInput:
    """Input for rally segmentation module (Stage 4)."""

    df: pd.DataFrame
    calibration: CourtCalibrationOutput

    def __post_init__(self):
        """Validate required columns."""
        required_cols = [
            "ball_y",
            "player_1_x_meter",
            "player_1_y_meter",
            "player_2_x_meter",
            "player_2_y_meter",
        ]
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


@dataclass
class RallySegmentationOutput:
    """Output from rally segmentation module (Stage 4)."""

    df: pd.DataFrame
    segments: List[RallySegment]

    # Metadata
    total_frames: int
    num_rallies: int
    rally_frame_count: int


# =============================================================================
# STAGE 5: HIT DETECTION
# =============================================================================


@dataclass
class WallHitDetectionInput:
    """Input for wall hit detection module (Stage 5a)."""

    df: pd.DataFrame
    segments: List[RallySegment]
    calibration: CourtCalibrationOutput

    def __post_init__(self):
        """Validate required columns."""
        required_cols = [
            "ball_x",
            "ball_y",
        ]
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


@dataclass
class WallHitDetectionOutput:
    """Output from wall hit detection module (Stage 5a)."""

    df: pd.DataFrame

    # Metadata
    num_wall_hits: int
    wall_hits_per_rally: Dict[int, int]


@dataclass
class RacketHitDetectionInput:
    """Input for racket hit detection module (Stage 5b)."""

    df: pd.DataFrame
    segments: List[RallySegment]

    def __post_init__(self):
        """Validate required columns."""
        required_cols = [
            "ball_x",
            "ball_y",
            "player_1_x_pixel",
            "player_1_y_pixel",
            "player_2_x_pixel",
            "player_2_y_pixel",
            "is_wall_hit",
        ]
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


@dataclass
class RacketHitDetectionOutput:
    """Output from racket hit detection module (Stage 5b)."""

    df: pd.DataFrame

    # Metadata
    num_racket_hits: int
    racket_hits_per_rally: Dict[int, int]


# =============================================================================
# STAGE 6: CLASSIFICATION
# =============================================================================


@dataclass
class StrokeClassificationInput:
    """Input for stroke classification module (Stage 6a)."""

    df: pd.DataFrame
    player_keypoints: Dict[int, List[np.ndarray]]  # {1: [...], 2: [...]}

    def __post_init__(self):
        """Validate required columns."""
        required_cols = [
            "is_racket_hit",
            "racket_hit_player_id",
        ]
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


@dataclass
class StrokeClassificationOutput:
    """Output from stroke classification module (Stage 6a)."""

    df: pd.DataFrame

    # Metadata
    num_strokes: int
    stroke_counts: Dict[str, int]


@dataclass
class ShotClassificationInput:
    """Input for shot classification module (Stage 6b)."""

    df: pd.DataFrame
    segments: List[RallySegment]

    def __post_init__(self):
        """Validate required columns."""
        required_cols = [
            "player_1_x_meter",
            "player_1_y_meter",
            "player_2_x_meter",
            "player_2_y_meter",
            "is_racket_hit",
            "racket_hit_player_id",
            "is_wall_hit",
            "wall_hit_x_meter",
        ]
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


@dataclass
class ShotClassificationOutput:
    """Output from shot classification module (Stage 6b)."""

    df: pd.DataFrame

    # Metadata
    num_shots: int
    shot_counts: Dict[str, int]


# =============================================================================
# PIPELINE SESSION
# =============================================================================


@dataclass
class PipelineSession:
    """
    Complete pipeline session container.

    Holds all data from the pipeline execution with DataFrame-based storage.
    """

    video_metadata: VideoMetadata
    calibration: Optional[CourtCalibrationOutput]

    # Current DataFrame (grows columns each stage)
    current_df: Optional[pd.DataFrame]

    # Complex data (keypoints, bboxes)
    complex_data: Dict[str, List]

    # Rally segments
    rally_segments: List[RallySegment]

    # Processing statistics
    processing_stats: Dict[str, Any] = field(default_factory=dict)

    def get_rally_frames(self, rally_id: int) -> pd.DataFrame:
        """Get DataFrame slice for a specific rally."""
        if self.current_df is None:
            raise ValueError("No DataFrame available")
        segment = self.rally_segments[rally_id]
        return self.current_df.loc[segment.start_frame : segment.end_frame]

    def export_csv(self, output_path: Path, rally_frames_only: bool = True) -> None:
        """Export DataFrame to CSV."""
        if self.current_df is None:
            raise ValueError("No DataFrame available")

        df_to_export = self.current_df
        if rally_frames_only and "is_rally_frame" in df_to_export.columns:
            df_to_export = df_to_export[df_to_export["is_rally_frame"]]

        df_to_export.to_csv(output_path)
