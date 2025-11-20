"""
Data models for court detection and calibration.

This module defines input and output models for the court-detection module.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from squashcopilot.common.types import Frame, Homography, Keypoints


# ============================================================================
# Wall Color Detection Models
# ============================================================================


@dataclass
class WallColorDetectionInput:
    """
    Input for wall color detection.

    Attributes:
        frame: Video frame to analyze
        keypoints_per_class: Optional pre-detected court keypoints
    """

    frame: Frame
    keypoints_per_class: Optional[Dict[str, Keypoints]] = None


@dataclass
class WallColorResult:
    """
    Result of wall color detection.

    Attributes:
        is_white: Whether the wall is white (vs colored)
        mean_brightness: Average brightness of wall region (0-255)
        wall_color_rgb: Dominant wall color in RGB format
        wall_color_bgr: Dominant wall color in BGR format
        recommended_ball_color: Recommended ball color ('black' or 'white')
    """

    is_white: bool
    mean_brightness: float
    wall_color_rgb: Tuple[int, int, int]
    wall_color_bgr: Tuple[int, int, int]
    recommended_ball_color: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_white": self.is_white,
            "mean_brightness": self.mean_brightness,
            "wall_color_rgb": self.wall_color_rgb,
            "wall_color_bgr": self.wall_color_bgr,
            "recommended_ball_color": self.recommended_ball_color,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WallColorResult":
        """
        Create WallColorResult from dictionary.

        Args:
            d: Dictionary with all fields

        Returns:
            WallColorResult instance
        """
        return cls(
            is_white=d["is_white"],
            mean_brightness=d["mean_brightness"],
            wall_color_rgb=tuple(d["wall_color_rgb"]),
            wall_color_bgr=tuple(d["wall_color_bgr"]),
            recommended_ball_color=d["recommended_ball_color"],
        )


# ============================================================================
# Court Calibration Models
# ============================================================================


@dataclass
class CourtCalibrationInput:
    """
    Input for court calibration.

    Attributes:
        frame: Video frame to process (typically first frame)
    """

    frame: Frame


@dataclass
class CourtCalibrationResult:
    """
    Result of court calibration.

    Attributes:
        homographies: Homography transformations for different planes
        keypoints_per_class: Detected keypoints for each court element
        frame_number: Frame index that was calibrated
        calibrated: Whether calibration was successful
    """

    homographies: Dict[str, Homography]
    keypoints_per_class: Dict[str, Keypoints]
    frame_number: int
    calibrated: bool
    wall_color: WallColorResult

    def get_homography(self, plane: str) -> Optional[Homography]:
        """
        Get homography for a specific plane.

        Args:
            plane: Plane name ('floor', 'wall', etc.)

        Returns:
            Homography or None if not found
        """
        return self.homographies.get(plane)

    def get_keypoints(self, class_name: str) -> Optional[Keypoints]:
        """
        Get keypoints for a specific court element.

        Args:
            class_name: Element name ('tin', 'left-square', etc.)

        Returns:
            Keypoints or None if not found
        """
        return self.keypoints_per_class.get(class_name)

    def is_wall_white(self) -> bool:
        """
        Check if the wall is white.

        Returns:
            True if wall is white, False otherwise
        """
        return self.wall_color.is_white

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "homographies": {
                name: homo.to_dict() for name, homo in self.homographies.items()
            },
            "keypoints_per_class": {
                name: kpts.to_dict() for name, kpts in self.keypoints_per_class.items()
            },
            "frame_number": self.frame_number,
            "calibrated": self.calibrated,
        }

    @classmethod
    def not_calibrated(cls, frame_number: int) -> "CourtCalibrationResult":
        """
        Create a result for when calibration fails.

        Args:
            frame_number: Frame index

        Returns:
            CourtCalibrationResult with calibrated=False
        """
        return cls(
            homographies={},
            keypoints_per_class={},
            frame_number=frame_number,
            calibrated=False,
        )
