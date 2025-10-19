"""Data models for representing pipeline outputs."""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object to convert

    Returns:
        Object with numpy types converted to Python types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy_types(item) for item in obj)
    else:
        return obj


@dataclass
class CourtData:
    """Court detection and calibration data."""

    homographies: Optional[Dict[str, np.ndarray]] = None
    keypoints: Optional[Dict[str, np.ndarray]] = None
    is_calibrated: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "homographies": {
                k: v.tolist() if v is not None else None
                for k, v in (self.homographies or {}).items()
            },
            "keypoints": {
                k: v.tolist() if v is not None else None
                for k, v in (self.keypoints or {}).items()
            },
            "is_calibrated": self.is_calibrated,
        }


@dataclass
class PlayerData:
    """Player tracking data for a single player."""

    player_id: int
    position: Optional[Tuple[float, float]] = None
    real_position: Optional[Tuple[float, float]] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: Optional[float] = None
    keypoints: Optional[Dict[str, List]] = None
    stroke_type: str = field(default="neither")

    def is_valid(self) -> bool:
        """Check if player data is valid (has position and bbox)."""
        return self.position is not None and self.bbox is not None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "player_id": convert_numpy_types(self.player_id),
            "position": convert_numpy_types(self.position),
            "real_position": convert_numpy_types(self.real_position),
            "bbox": convert_numpy_types(self.bbox),
            "confidence": convert_numpy_types(self.confidence),
            "keypoints": convert_numpy_types(self.keypoints),
            "stroke_type": self.stroke_type,
        }


@dataclass
class BallData:
    """Ball detection data."""

    position: Optional[Tuple[int, int]] = None
    confidence: Optional[float] = None
    is_wall_hit: bool = False

    def is_valid(self) -> bool:
        """Check if ball data is valid."""
        return self.position is not None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "position": convert_numpy_types(self.position),
            "confidence": convert_numpy_types(self.confidence),
            "is_wall_hit": self.is_wall_hit,
        }


@dataclass
class FrameData:
    """Complete data for a single frame."""

    frame_number: int
    timestamp: float
    court: CourtData
    player1: PlayerData
    player2: PlayerData
    ball: BallData
    rally_state: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "frame_number": convert_numpy_types(self.frame_number),
            "timestamp": convert_numpy_types(self.timestamp),
            "court": self.court.to_dict(),
            "player1": self.player1.to_dict(),
            "player2": self.player2.to_dict(),
            "ball": self.ball.to_dict(),
            "rally_state": self.rally_state,
        }

    def get_player(self, player_id: int) -> Optional[PlayerData]:
        """Get player data by ID."""
        if player_id == 1:
            return self.player1
        elif player_id == 2:
            return self.player2
        return None

    def has_valid_players(self) -> bool:
        """Check if both players are valid."""
        return self.player1.is_valid() and self.player2.is_valid()

    def has_valid_ball(self) -> bool:
        """Check if ball is valid."""
        return self.ball.is_valid()


@dataclass
class RallyData:
    """Rally data."""

    rally_frames: List[FrameData]
    start_frame: int
    end_frame: int

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "rally_frames": [frame.to_dict() for frame in self.rally_frames],
            "start_frame": convert_numpy_types(self.start_frame),
            "end_frame": convert_numpy_types(self.end_frame),
        }
