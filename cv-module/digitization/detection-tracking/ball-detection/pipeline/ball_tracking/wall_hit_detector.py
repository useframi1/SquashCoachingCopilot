"""
Wall hit detection for squash ball tracking.

Detects front wall hits by finding local minima in the Y-coordinate curve.
"""

import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple, Dict
from ball_tracking.utils import load_config


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
            config = load_config()

        wall_config = config.get("wall_hit_detection", {})
        self.prominence = wall_config.get("prominence", 50.0)
        self.width = wall_config.get("width", 3)
        self.min_distance = wall_config.get("min_distance", 20)

    def detect(
        self,
        positions: List[Tuple[float, float]],
    ) -> List[Dict]:
        """Detect front wall hits using local minima in Y-coordinate.

        Args:
            positions: List of (x, y) tuples (smoothed ball positions)

        Returns:
            List of dictionaries with wall hit information:
            [
                {
                    'frame': int,        # Frame index where hit occurred
                    'x': float,          # X position at impact (lateral position on wall)
                    'y': float,          # Y position at impact (height)
                    'prominence': float  # Depth of the valley (impact strength indicator)
                },
                ...
            ]
        """
        if len(positions) < self.width:
            return []

        # Extract Y coordinates
        y_coords = np.array([p[1] for p in positions])
        x_coords = np.array([p[0] for p in positions])

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

        # Build wall hit results
        wall_hits = []
        for i, peak_idx in enumerate(peaks):
            wall_hit = {
                "frame": int(peak_idx),
                "x": float(x_coords[peak_idx]),
                "y": float(y_coords[peak_idx]),
                "prominence": float(properties["prominences"][i]),
            }
            wall_hits.append(wall_hit)

        return wall_hits

    @staticmethod
    def calculate_statistics(wall_hits: List[Dict], fps: int) -> Dict:
        """Calculate statistics about detected wall hits.

        Args:
            wall_hits: List of wall hit dictionaries
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
        if not wall_hits:
            return {
                "total_hits": 0,
                "avg_hit_interval_sec": 0,
                "avg_impact_height": 0,
                "min_impact_height": 0,
                "max_impact_height": 0,
            }

        # Calculate intervals between hits
        frames = [hit["frame"] for hit in wall_hits]
        intervals = [frames[i + 1] - frames[i] for i in range(len(frames) - 1)]
        avg_interval_frames = np.mean(intervals) if intervals else 0
        avg_interval_sec = avg_interval_frames / fps if fps > 0 else 0

        # Calculate height statistics
        heights = [hit["y"] for hit in wall_hits]

        stats = {
            "total_hits": len(wall_hits),
            "avg_hit_interval_sec": round(avg_interval_sec, 2),
            "avg_impact_height": round(np.mean(heights), 1),
            "min_impact_height": round(np.min(heights), 1),
            "max_impact_height": round(np.max(heights), 1),
        }

        return stats
