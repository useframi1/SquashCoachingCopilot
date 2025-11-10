"""
Simple postprocessing for ball tracking data.

This module provides:
1. Outlier removal using rolling window distance check
2. Linear interpolation for missing values
"""

import numpy as np
from scipy.signal import savgol_filter, medfilt
from typing import List, Tuple, Optional


def remove_outliers(
    positions: List[Tuple[Optional[float], Optional[float]]],
    window: int = 10,
    threshold: float = 100,
) -> List[Tuple[Optional[float], Optional[float]]]:
    """Remove outlier positions using rolling window distance check.

    Detects positions that are too far from the average position of neighbors.

    Args:
        positions: List of (x, y) tuples
        window: Size of rolling window (default: 10 frames)
        threshold: Maximum distance in pixels from average of neighbors (default: 100 pixels)

    Returns:
        Positions with outliers marked as (None, None)
    """
    n = len(positions)
    result = list(positions)

    for i in range(n):
        if positions[i][0] is None or positions[i][1] is None:
            continue

        current_x, current_y = positions[i]

        # Get neighboring positions in the window
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)

        neighbors = []
        for j in range(start, end):
            if j != i and positions[j][0] is not None and positions[j][1] is not None:
                neighbors.append(positions[j])

        if len(neighbors) < 2:
            # Need at least 2 neighbors to calculate average
            continue

        # Calculate average position of neighbors
        avg_x = np.mean([nx for nx, ny in neighbors])
        avg_y = np.mean([ny for nx, ny in neighbors])

        # Calculate distance from current position to average of neighbors
        distance = np.sqrt((current_x - avg_x) ** 2 + (current_y - avg_y) ** 2)

        # If distance from average is >= threshold, mark as outlier
        if distance >= threshold:
            result[i] = (None, None)

    return result


def impute_missing(
    positions: List[Tuple[Optional[float], Optional[float]]],
) -> List[Tuple[float, float]]:
    """Fill missing values using linear interpolation.

    Args:
        positions: List of (x, y) tuples with possible None values

    Returns:
        List of (x, y) tuples with all values filled
    """
    n = len(positions)
    x_coords = np.array([p[0] if p[0] is not None else np.nan for p in positions])
    y_coords = np.array([p[1] if p[1] is not None else np.nan for p in positions])

    # Interpolate each axis
    for coords in [x_coords, y_coords]:
        valid_mask = ~np.isnan(coords)
        if np.sum(valid_mask) < 2:
            # Not enough valid points
            coords[:] = np.nanmean(coords) if np.sum(valid_mask) > 0 else 0
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_values = coords[valid_mask]

        # Linear interpolation
        coords[:] = np.interp(
            np.arange(n),
            valid_indices,
            valid_values,
            left=valid_values[0],
            right=valid_values[-1],
        )

    return [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]


def postprocess_positions(
    positions: List[Tuple[Optional[float], Optional[float]]], config: dict
) -> List[Tuple[float, float]]:
    """Simple 2-step postprocessing pipeline.

    Steps:
    1. Remove outliers using rolling window distance check
    2. Fill missing values using linear interpolation

    Args:
        positions: Raw position data
        fps: Frames per second (not used in simple version)
        config: Configuration with postprocessing settings

    Returns:
        Clean position data with outliers removed and gaps interpolated
    """
    if not config.get("enabled", True):
        # If postprocessing disabled, just fill missing values
        return impute_missing(positions)

    print("Postprocessing ball positions...")

    # Step 1: Remove outliers
    outlier_config = config.get("outlier_detection", {})
    cleaned = remove_outliers(
        positions,
        window=outlier_config.get("window", 10),
        threshold=outlier_config.get("threshold", 100),
    )
    n_outliers = sum(
        1 for i, p in enumerate(cleaned) if p[0] is None and positions[i][0] is not None
    )
    if n_outliers > 0:
        print(f"  Removed {n_outliers} outliers")

    # Step 2: Fill missing values with interpolation
    imputed = impute_missing(cleaned)
    print(f"  Filled missing values with interpolation")

    print("Postprocessing complete!")
    return imputed
