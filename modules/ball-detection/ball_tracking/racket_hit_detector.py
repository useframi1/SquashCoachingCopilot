"""
Racket hit detection for squash ball tracking.

Detects racket hits by finding steep negative slopes (downward) before wall hits.
"""

import numpy as np
from typing import List, Tuple, Dict
from ball_tracking.utils import load_config


class RacketHitDetector:
    """Detects racket hits using steep negative slopes before wall hits.

    Racket hits create steep negative slopes in Y-coordinate (ball accelerating
    toward wall). This algorithm:
    1. Takes detected wall hits as input
    2. Looks backward from each wall hit for steep negative slopes
    3. Identifies the point with the steepest downward acceleration (most negative slope)
    4. Selects the one closest to the wall hit
    """

    def __init__(self, config: dict = None):
        """Initialize the racket hit detector.

        Args:
            config: Configuration dictionary. If None, loads from config.json
        """
        if config is None:
            config = load_config()

        racket_config = config.get("racket_hit_detection", {})
        self.slope_window = racket_config.get("slope_window", 5)
        self.slope_threshold = racket_config.get("slope_threshold", 15.0)
        self.min_distance = racket_config.get("min_distance", 15)
        self.lookback_frames = racket_config.get("lookback_frames", 20)

    def detect(
        self,
        positions: List[Tuple[float, float]],
        wall_hits: List[Dict],
    ) -> List[Dict]:
        """Detect racket hits using steep negative slopes (downward) before wall hits.

        Args:
            positions: List of (x, y) tuples (smoothed ball positions)
            wall_hits: List of wall hit dictionaries from WallHitDetector

        Returns:
            List of dictionaries with racket hit information:
            [
                {
                    'frame': int,        # Frame index where hit occurred
                    'x': float,          # X position at impact (lateral position)
                    'y': float,          # Y position at impact (height)
                    'slope': float       # Steepness of slope (negative value, impact strength indicator)
                },
                ...
            ]
        """
        if len(positions) < self.slope_window + self.lookback_frames or not wall_hits:
            return []

        # Extract Y coordinates
        y_coords = np.array([p[1] for p in positions])
        x_coords = np.array([p[0] for p in positions])

        # For each wall hit, look backward for steep negative slope (racket hit)
        racket_hits = []

        for wall_hit in wall_hits:
            wall_hit_frame = wall_hit["frame"]

            # Define search window: look back from wall hit
            search_start = max(0, wall_hit_frame - self.lookback_frames)
            search_end = wall_hit_frame

            if search_end - search_start < self.slope_window:
                continue

            # Calculate slopes in the search window
            # We want the most negative slope (steepest downward)
            min_slope = np.inf  # Most negative will be smallest
            min_slope_frame = None

            for i in range(search_start, search_end - self.slope_window):
                # Calculate slope over slope_window frames
                y_change = y_coords[i + self.slope_window] - y_coords[i]
                slope = y_change / self.slope_window

                # Track minimum slope (most negative = steepest downward acceleration)
                # Only consider negative slopes that exceed threshold
                if slope < min_slope and slope < -self.slope_threshold:
                    min_slope = slope
                    min_slope_frame = i

            # If we found a steep enough negative slope, record it as a racket hit
            if min_slope_frame is not None:
                # Check if this hit is far enough from previous hits
                if (
                    racket_hits
                    and (min_slope_frame - racket_hits[-1]["frame"]) < self.min_distance
                ):
                    continue

                racket_hit = {
                    "frame": int(min_slope_frame),
                    "x": float(x_coords[min_slope_frame]),
                    "y": float(y_coords[min_slope_frame]),
                    "slope": float(min_slope),  # Will be negative
                }
                racket_hits.append(racket_hit)

        return racket_hits

    @staticmethod
    def calculate_statistics(racket_hits: List[Dict], fps: int) -> Dict:
        """Calculate statistics about detected racket hits.

        Args:
            racket_hits: List of racket hit dictionaries
            fps: Frames per second

        Returns:
            Dictionary with statistics:
            {
                'total_hits': int,
                'avg_hit_interval_sec': float,
                'avg_impact_height': float,
                'min_impact_height': float,
                'max_impact_height': float
            }
        """
        if not racket_hits:
            return {
                "total_hits": 0,
                "avg_hit_interval_sec": 0,
                "avg_impact_height": 0,
                "min_impact_height": 0,
                "max_impact_height": 0,
            }

        # Calculate intervals between hits
        frames = [hit["frame"] for hit in racket_hits]
        intervals = [frames[i + 1] - frames[i] for i in range(len(frames) - 1)]
        avg_interval_frames = np.mean(intervals) if intervals else 0
        avg_interval_sec = avg_interval_frames / fps if fps > 0 else 0

        # Calculate height statistics
        heights = [hit["y"] for hit in racket_hits]

        stats = {
            "total_hits": len(racket_hits),
            "avg_hit_interval_sec": round(avg_interval_sec, 2),
            "avg_impact_height": round(np.mean(heights), 1),
            "min_impact_height": round(np.min(heights), 1),
            "max_impact_height": round(np.max(heights), 1),
        }

        return stats
