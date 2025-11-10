"""
Wall hit detection for squash ball tracking.

Detects front wall hits by finding local minima in the Y-coordinate curve.
"""

import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple, Dict


def detect_racket_hits(
    positions: List[Tuple[float, float]],
    wall_hits: List[Dict],
    slope_window: int = 5,
    slope_threshold: float = 15.0,
    min_distance: int = 15,
    lookback_frames: int = 20,
) -> List[Dict]:
    """Detect racket hits using steep negative slopes (downward) before wall hits.

    Racket hits create steep negative slopes in Y-coordinate (ball accelerating
    toward wall). This algorithm:
    1. Takes detected wall hits as input
    2. Looks backward from each wall hit for steep negative slopes
    3. Identifies the point with the steepest downward acceleration (most negative slope)
    4. Selects the one closest to the wall hit

    Args:
        positions: List of (x, y) tuples (smoothed ball positions)
        wall_hits: List of wall hit dictionaries from detect_front_wall_hits()
        slope_window: Number of frames to calculate slope over
        slope_threshold: Minimum absolute slope (pixels/frame) to consider as racket hit
        min_distance: Minimum frames between consecutive hits (prevents duplicates)
        lookback_frames: How many frames to look back from wall hit to find racket hit
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
    if len(positions) < slope_window + lookback_frames or not wall_hits:
        return []

    # Extract Y coordinates
    y_coords = np.array([p[1] for p in positions])
    x_coords = np.array([p[0] for p in positions])

    # For each wall hit, look backward for steep negative slope (racket hit)
    racket_hits = []

    for wall_hit in wall_hits:
        wall_hit_frame = wall_hit["frame"]

        # Define search window: look back from wall hit
        search_start = max(0, wall_hit_frame - lookback_frames)
        search_end = wall_hit_frame

        if search_end - search_start < slope_window:
            continue

        # Calculate slopes in the search window
        # We want the most negative slope (steepest downward)
        min_slope = np.inf  # Most negative will be smallest
        min_slope_frame = None

        for i in range(search_start, search_end - slope_window):
            # Calculate slope over slope_window frames
            y_change = y_coords[i + slope_window] - y_coords[i]
            slope = y_change / slope_window

            # Track minimum slope (most negative = steepest downward acceleration)
            # Only consider negative slopes that exceed threshold
            if slope < min_slope and slope < -slope_threshold:
                min_slope = slope
                min_slope_frame = i

        # If we found a steep enough negative slope, record it as a racket hit
        if min_slope_frame is not None:
            # Check if this hit is far enough from previous hits
            if (
                racket_hits
                and (min_slope_frame - racket_hits[-1]["frame"]) < min_distance
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


def detect_front_wall_hits(
    positions: List[Tuple[float, float]],
    prominence: float = 50.0,
    width: int = 3,
    min_distance: int = 20,
) -> List[Dict]:
    """Detect front wall hits using local minima in Y-coordinate.

    Front wall hits appear as valleys (local minima) in the Y-coordinate curve.
    The algorithm finds these minima and validates them based on prominence and width.

    Args:
        positions: List of (x, y) tuples (smoothed ball positions)
        prominence: Minimum depth of valley in pixels (higher = only significant hits)
        width: Minimum width of valley in frames (filters noise)
        min_distance: Minimum frames between consecutive hits (prevents duplicates)

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
    if len(positions) < width:
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
        prominence=prominence,  # Minimum valley depth
        width=width,  # Minimum valley width
        distance=min_distance,  # Minimum spacing between hits
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


def calculate_hit_statistics(wall_hits: List[Dict], fps: int) -> Dict:
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
