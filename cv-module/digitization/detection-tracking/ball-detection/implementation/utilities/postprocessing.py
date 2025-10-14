"""
Simple postprocessing for ball tracking data focused on wall hit detection.

For detecting wall hits, we need clean Y-coordinate curves to see direction changes.
This module provides:
1. Outlier removal (removes position jumps from bad detections)
2. Imputation (fills missing values)
3. Smoothing (makes the position curve clean and clear)
"""

import numpy as np
from scipy.signal import savgol_filter, medfilt
from typing import List, Tuple, Optional


def remove_outliers(
    positions: List[Tuple[Optional[float], Optional[float]]],
    window: int = 7,
    threshold: float = 2.5,
) -> List[Tuple[Optional[float], Optional[float]]]:
    """Remove outlier positions using median filtering.

    Detects positions that deviate significantly from their local neighborhood.

    Args:
        positions: List of (x, y) tuples
        window: Size of local neighborhood to compare against
        threshold: Number of standard deviations to be considered an outlier

    Returns:
        Positions with outliers marked as (None, None)
    """
    n = len(positions)
    result = list(positions)

    # Convert to arrays
    x_coords = np.array([p[0] if p[0] is not None else np.nan for p in positions])
    y_coords = np.array([p[1] if p[1] is not None else np.nan for p in positions])

    # Check each coordinate separately
    for coords, axis in [(x_coords, 'x'), (y_coords, 'y')]:
        for i in range(n):
            if np.isnan(coords[i]):
                continue

            # Get local window
            start = max(0, i - window // 2)
            end = min(n, i + window // 2 + 1)
            local = coords[start:end]
            local = local[~np.isnan(local)]

            if len(local) < 3:
                continue

            # Check if point is an outlier compared to local median
            median = np.median(local)
            std = np.std(local)

            if std > 0 and abs(coords[i] - median) > threshold * std:
                result[i] = (None, None)

    return result


def impute_missing(
    positions: List[Tuple[Optional[float], Optional[float]]]
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
            right=valid_values[-1]
        )

    return [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]


def smooth_positions(
    positions: List[Tuple[float, float]],
    method: str = "savgol",
    window: int = 7,
    poly: int = 2,
) -> List[Tuple[float, float]]:
    """Smooth the position curve to make direction changes clear.

    Args:
        positions: List of (x, y) tuples
        method: "savgol" (default) or "median"
        window: Window size (must be odd)
        poly: Polynomial order for savgol (2 or 3 recommended)

    Returns:
        Smoothed positions
    """
    if len(positions) < window:
        return positions

    # Ensure window is odd
    if window % 2 == 0:
        window += 1

    x_coords = np.array([p[0] for p in positions], dtype=float)
    y_coords = np.array([p[1] for p in positions], dtype=float)

    if method == "savgol":
        # Savitzky-Golay filter - good for preserving peaks and valleys
        x_smooth = savgol_filter(x_coords, window, poly)
        y_smooth = savgol_filter(y_coords, window, poly)
    elif method == "median":
        # Median filter - good for removing spikes
        x_smooth = medfilt(x_coords, kernel_size=window)
        y_smooth = medfilt(y_coords, kernel_size=window)
    else:
        return positions

    return [(int(x), int(y)) for x, y in zip(x_smooth, y_smooth)]


def postprocess_positions(
    positions: List[Tuple[Optional[float], Optional[float]]],
    fps: int,
    config: dict,
) -> List[Tuple[float, float]]:
    """Simple 3-step postprocessing pipeline.

    Steps:
    1. Remove outliers (bad detections)
    2. Fill missing values (interpolation)
    3. Smooth the curve (make Y direction changes clear)

    Args:
        positions: Raw position data
        fps: Frames per second (not used in simple version)
        config: Configuration with postprocessing settings

    Returns:
        Clean, smooth position data ready for wall hit detection
    """
    if not config.get("enabled", True):
        # If postprocessing disabled, just fill missing values
        return impute_missing(positions)

    print("Postprocessing ball positions...")

    # Step 1: Remove outliers
    outlier_config = config.get("outlier_detection", {})
    cleaned = remove_outliers(
        positions,
        window=outlier_config.get("window", 7),
        threshold=outlier_config.get("threshold", 2.5)
    )
    n_outliers = sum(1 for p in cleaned if p[0] is None)
    if n_outliers > 0:
        print(f"  Removed {n_outliers} outliers")

    # Step 2: Fill missing values
    imputed = impute_missing(cleaned)
    print(f"  Filled missing values")

    # Step 3: Smooth the curve
    smoothing_config = config.get("smoothing", {})
    if smoothing_config.get("enabled", True):
        smoothed = smooth_positions(
            imputed,
            method=smoothing_config.get("method", "savgol"),
            window=smoothing_config.get("window", 7),
            poly=smoothing_config.get("poly", 2)
        )
        print(f"  Smoothed curve (window={smoothing_config.get('window', 7)})")
    else:
        smoothed = imputed

    print("Postprocessing complete!")
    return smoothed
