"""
Wall hit detection for squash ball tracking.

Detects front wall hits by finding local minima in the Y-coordinate curve.
"""

import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple, Dict


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
        width=width,           # Minimum valley width
        distance=min_distance   # Minimum spacing between hits
    )

    # Build wall hit results
    wall_hits = []
    for i, peak_idx in enumerate(peaks):
        wall_hit = {
            'frame': int(peak_idx),
            'x': float(x_coords[peak_idx]),
            'y': float(y_coords[peak_idx]),
            'prominence': float(properties['prominences'][i])
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
            'total_hits': 0,
            'avg_hit_interval_sec': 0,
            'avg_impact_height': 0,
            'min_impact_height': 0,
            'max_impact_height': 0
        }

    # Calculate intervals between hits
    frames = [hit['frame'] for hit in wall_hits]
    intervals = [frames[i+1] - frames[i] for i in range(len(frames)-1)]
    avg_interval_frames = np.mean(intervals) if intervals else 0
    avg_interval_sec = avg_interval_frames / fps if fps > 0 else 0

    # Calculate height statistics
    heights = [hit['y'] for hit in wall_hits]

    stats = {
        'total_hits': len(wall_hits),
        'avg_hit_interval_sec': round(avg_interval_sec, 2),
        'avg_impact_height': round(np.mean(heights), 1),
        'min_impact_height': round(np.min(heights), 1),
        'max_impact_height': round(np.max(heights), 1)
    }

    return stats


def save_wall_hits_csv(wall_hits: List[Dict], output_path: str):
    """Save wall hit data to CSV file.

    Args:
        wall_hits: List of wall hit dictionaries
        output_path: Path to output CSV file
    """
    with open(output_path, 'w') as f:
        f.write("frame,x,y,prominence\n")
        for hit in wall_hits:
            f.write(f"{hit['frame']},{hit['x']},{hit['y']},{hit['prominence']}\n")
