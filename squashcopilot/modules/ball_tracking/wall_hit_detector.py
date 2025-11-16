"""
Wall hit detection for squash ball tracking.

Detects front wall hits by finding local minima in the Y-coordinate curve.
"""

import numpy as np
from scipy.signal import find_peaks
from squashcopilot.common.utils import load_config

from squashcopilot.common import Point2D, WallHitInput, WallHit, WallHitDetectionResult


class WallHitDetector:
    """Detects front wall hits using local minima in Y-coordinate.

    Front wall hits appear as valleys (local minima) in the Y-coordinate curve.
    The algorithm finds these minima and validates them based on prominence and width.
    """

    def __init__(self, config: dict = None):
        """Initialize the wall hit detector.

        Args:
            config: Configuration dictionary. If None, loads from config.json
        """
        if config is None:
            config = load_config(config_name='ball_tracking')

        wall_config = config.get("wall_hit_detection", {})
        self.prominence = wall_config.get("prominence", 50.0)
        self.width = wall_config.get("width", 3)
        self.min_distance = wall_config.get("min_distance", 20)

    def detect(self, input_data: WallHitInput) -> WallHitDetectionResult:
        """Detect front wall hits using local minima in Y-coordinate.

        Args:
            input_data: WallHitInput with positions and config

        Returns:
            WallHitDetectionResult with detected wall hits
        """
        # Convert Point2D to tuples for processing
        positions_tuples = [(p.x, p.y) for p in input_data.positions]

        # Override config if provided
        if input_data.config:
            prominence = input_data.config.get(
                "wall_hit_detection.prominence", self.prominence
            )
            width = input_data.config.get("wall_hit_detection.width", self.width)
            min_distance = input_data.config.get(
                "wall_hit_detection.min_distance", self.min_distance
            )
        else:
            prominence = self.prominence
            width = self.width
            min_distance = self.min_distance

        if len(positions_tuples) < width:
            return WallHitDetectionResult(wall_hits=[])

        # Extract Y coordinates
        y_coords = np.array([p[1] for p in positions_tuples])
        x_coords = np.array([p[0] for p in positions_tuples])

        # Invert Y to find minima as peaks
        inverted_y = -y_coords

        # Find peaks in inverted signal (= minima in original signal)
        peaks, properties = find_peaks(
            inverted_y,
            prominence=prominence,
            width=width,
            distance=min_distance,
        )

        # Build WallHit objects
        wall_hits = []
        for i, peak_idx in enumerate(peaks):
            # Get pixel position
            position = Point2D(
                x=float(x_coords[peak_idx]), y=float(y_coords[peak_idx])
            )

            # Transform to wall meter coordinates if homography is provided
            position_meter = None
            if input_data.wall_homography is not None:
                position_meter = input_data.wall_homography.transform_point(position)

            wall_hit = WallHit(
                frame=int(peak_idx),
                position=position,
                prominence=float(properties["prominences"][i]),
                position_meter=position_meter
            )
            wall_hits.append(wall_hit)

        return WallHitDetectionResult(wall_hits=wall_hits)
