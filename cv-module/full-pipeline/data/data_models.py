"""Data models for representing pipeline outputs."""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np


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

    def is_valid(self) -> bool:
        """Check if player data is valid (has position and bbox)."""
        return self.position is not None and self.bbox is not None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "player_id": self.player_id,
            "position": self.position,
            "real_position": self.real_position,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "keypoints": self.keypoints,
        }


@dataclass
class BallData:
    """Ball detection data."""

    position: Optional[Tuple[int, int]] = None
    confidence: Optional[float] = None

    def is_valid(self) -> bool:
        """Check if ball data is valid."""
        return self.position is not None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "position": self.position,
            "confidence": self.confidence,
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
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
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
