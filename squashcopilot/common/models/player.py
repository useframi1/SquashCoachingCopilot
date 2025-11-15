"""
Data models for player detection and tracking.

This module defines input and output models for the player-detection module.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from squashcopilot.common.types import Frame, Config, Point2D, BoundingBox, Homography


# ============================================================================
# Player Keypoints Models
# ============================================================================

@dataclass
class PlayerKeypointsData:
    """
    Player keypoints in COCO format.

    Attributes:
        xy: List of 17 COCO keypoints [x0, y0, x1, y1, ..., x16, y16] or None
        conf: List of 17 confidence values [c0, c1, ..., c16] or None
    """
    xy: Optional[List[float]]
    conf: Optional[List[float]]

    def get_keypoint(self, index: int) -> Optional[Tuple[float, float, float]]:
        """
        Get a specific keypoint by index.

        Args:
            index: COCO keypoint index (0-16)

        Returns:
            Tuple (x, y, confidence) or None if not available
        """
        if self.xy is None or index < 0 or index >= len(self.xy) // 2:
            return None

        x = self.xy[index * 2]
        y = self.xy[index * 2 + 1]
        conf = self.conf[index] if self.conf and index < len(self.conf) else 0.0

        return (x, y, conf)

    def get_keypoint_as_point(self, index: int) -> Optional[Point2D]:
        """
        Get a specific keypoint as Point2D.

        Args:
            index: COCO keypoint index (0-16)

        Returns:
            Point2D or None if not available
        """
        kpt = self.get_keypoint(index)
        if kpt is None:
            return None
        return Point2D(x=kpt[0], y=kpt[1])

    @property
    def num_keypoints(self) -> int:
        """Get number of keypoints."""
        if self.xy is None:
            return 0
        return len(self.xy) // 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'xy': self.xy,
            'conf': self.conf
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PlayerKeypointsData':
        """
        Create PlayerKeypointsData from dictionary.

        Args:
            d: Dictionary with xy and conf

        Returns:
            PlayerKeypointsData instance
        """
        return cls(
            xy=d.get('xy'),
            conf=d.get('conf')
        )


# ============================================================================
# Player Detection Models
# ============================================================================

@dataclass
class PlayerTrackingInput:
    """
    Input for player tracking on a single frame.

    Attributes:
        frame: Video frame to process
        homography: Optional homography for coordinate transformation
    """
    frame: Frame
    homography: Optional[Homography] = None


@dataclass
class PlayerDetectionResult:
    """
    Result of player detection for a single player.

    Attributes:
        player_id: Player identifier (1 or 2)
        position: Bottom-center position in pixels
        real_position: Position in real-world coordinates (meters)
        bbox: Bounding box around player
        confidence: Detection confidence (0.0-1.0)
        keypoints: Player pose keypoints (COCO format)
        frame_number: Frame index
    """
    player_id: int
    position: Point2D
    real_position: Optional[Point2D]
    bbox: BoundingBox
    confidence: float
    keypoints: Optional[PlayerKeypointsData]
    frame_number: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'player_id': self.player_id,
            'position': self.position.to_dict(),
            'real_position': self.real_position.to_dict() if self.real_position else None,
            'bbox': self.bbox.to_dict(),
            'confidence': self.confidence,
            'keypoints': self.keypoints.to_dict() if self.keypoints else None,
            'frame_number': self.frame_number
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PlayerDetectionResult':
        """
        Create PlayerDetectionResult from dictionary.

        Args:
            d: Dictionary with all fields

        Returns:
            PlayerDetectionResult instance
        """
        return cls(
            player_id=d['player_id'],
            position=Point2D(**d['position']),
            real_position=Point2D(**d['real_position']) if d.get('real_position') else None,
            bbox=BoundingBox(**d['bbox']) if 'x1' in d['bbox'] else BoundingBox.from_list(d['bbox']),
            confidence=d['confidence'],
            keypoints=PlayerKeypointsData.from_dict(d['keypoints']) if d.get('keypoints') else None,
            frame_number=d['frame_number']
        )


@dataclass
class PlayerTrackingResult:
    """
    Result of player tracking for all players in a frame.

    Attributes:
        players: Dictionary mapping player_id to detection result
        frame_number: Frame index
        num_players_detected: Number of players detected
    """
    players: Dict[int, PlayerDetectionResult]
    frame_number: int
    num_players_detected: int = field(init=False)

    def __post_init__(self):
        """Calculate num_players_detected."""
        self.num_players_detected = len(self.players)

    def get_player(self, player_id: int) -> Optional[PlayerDetectionResult]:
        """
        Get detection result for a specific player.

        Args:
            player_id: Player identifier

        Returns:
            PlayerDetectionResult or None if not found
        """
        return self.players.get(player_id)

    def has_player(self, player_id: int) -> bool:
        """
        Check if a player was detected.

        Args:
            player_id: Player identifier

        Returns:
            True if player was detected
        """
        return player_id in self.players

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'players': {
                player_id: result.to_dict()
                for player_id, result in self.players.items()
            },
            'frame_number': self.frame_number,
            'num_players_detected': self.num_players_detected
        }

    @classmethod
    def no_players_detected(cls, frame_number: int) -> 'PlayerTrackingResult':
        """
        Create a result for when no players are detected.

        Args:
            frame_number: Frame index

        Returns:
            PlayerTrackingResult with empty players dict
        """
        return cls(
            players={},
            frame_number=frame_number
        )


# ============================================================================
# Player Postprocessing Models
# ============================================================================

@dataclass
class PlayerPostprocessingInput:
    """
    Input for player position postprocessing.

    Attributes:
        positions_history: Dictionary mapping player_id to list of positions
                          (can contain None for missing detections)
        config: Optional configuration for postprocessing
    """
    positions_history: Dict[int, List[Optional[Point2D]]]
    config: Optional[Config] = None


@dataclass
class PlayerTrajectory:
    """
    Processed player trajectory with smoothed positions.

    Attributes:
        player_id: Player identifier
        positions: List of smoothed positions (gaps filled)
        original_positions: Original positions before postprocessing
        gaps_filled: Number of gaps interpolated
    """
    player_id: int
    positions: List[Point2D]
    original_positions: List[Optional[Point2D]]
    gaps_filled: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'player_id': self.player_id,
            'positions': [p.to_dict() for p in self.positions],
            'original_positions': [
                p.to_dict() if p else None
                for p in self.original_positions
            ],
            'gaps_filled': self.gaps_filled
        }


@dataclass
class PlayerPostprocessingResult:
    """
    Result of player trajectory postprocessing.

    Attributes:
        trajectories: Dictionary mapping player_id to trajectory
    """
    trajectories: Dict[int, PlayerTrajectory]

    def get_trajectory(self, player_id: int) -> Optional[PlayerTrajectory]:
        """
        Get trajectory for a specific player.

        Args:
            player_id: Player identifier

        Returns:
            PlayerTrajectory or None if not found
        """
        return self.trajectories.get(player_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trajectories': {
                player_id: traj.to_dict()
                for player_id, traj in self.trajectories.items()
            }
        }
