"""Common metric computation utilities."""

import numpy as np
from typing import List, Tuple, Optional


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        point1: (x, y) coordinates
        point2: (x, y) coordinates

    Returns:
        Distance between points
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_speed(
    trajectory: List[Tuple[float, float]], fps: float
) -> List[float]:
    """
    Calculate instantaneous speed along a trajectory.

    Args:
        trajectory: List of (x, y) positions
        fps: Frames per second

    Returns:
        List of speeds in units/second
    """
    if len(trajectory) < 2:
        return []

    speeds = []
    for i in range(1, len(trajectory)):
        distance = calculate_distance(trajectory[i - 1], trajectory[i])
        speed = distance * fps  # Convert to units per second
        speeds.append(speed)

    return speeds


def calculate_acceleration(speeds: List[float], fps: float) -> List[float]:
    """
    Calculate acceleration from speed data.

    Args:
        speeds: List of speeds
        fps: Frames per second

    Returns:
        List of accelerations in units/second²
    """
    if len(speeds) < 2:
        return []

    accelerations = []
    for i in range(1, len(speeds)):
        accel = (speeds[i] - speeds[i - 1]) * fps
        accelerations.append(accel)

    return accelerations


def calculate_total_distance(trajectory: List[Tuple[float, float]]) -> float:
    """
    Calculate total distance traveled along a trajectory.

    Args:
        trajectory: List of (x, y) positions

    Returns:
        Total distance
    """
    if len(trajectory) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(trajectory)):
        total += calculate_distance(trajectory[i - 1], trajectory[i])

    return total


def calculate_average_position(
    trajectory: List[Tuple[float, float]]
) -> Optional[Tuple[float, float]]:
    """
    Calculate average (centroid) position of a trajectory.

    Args:
        trajectory: List of (x, y) positions

    Returns:
        Average (x, y) position or None if trajectory is empty
    """
    if not trajectory:
        return None

    positions = np.array(trajectory)
    return tuple(np.mean(positions, axis=0))


def calculate_court_coverage(
    trajectory: List[Tuple[float, float]], grid_size: int = 10
) -> float:
    """
    Calculate court coverage as percentage of grid cells visited.

    Args:
        trajectory: List of (x, y) positions in real coordinates
        grid_size: Number of grid cells (grid_size x grid_size)

    Returns:
        Coverage percentage (0-100)
    """
    if not trajectory:
        return 0.0

    # Define squash court bounds (in meters)
    # Standard squash court: 9.75m x 6.4m
    court_length = 9.75
    court_width = 6.4

    # Create grid
    cell_length = court_length / grid_size
    cell_width = court_width / grid_size

    visited_cells = set()

    for x, y in trajectory:
        # Calculate grid cell indices
        cell_x = int(x / cell_length)
        cell_y = int(y / cell_width)

        # Ensure within bounds
        cell_x = max(0, min(grid_size - 1, cell_x))
        cell_y = max(0, min(grid_size - 1, cell_y))

        visited_cells.add((cell_x, cell_y))

    coverage = (len(visited_cells) / (grid_size * grid_size)) * 100
    return coverage


def smooth_trajectory(
    trajectory: List[Tuple[float, float]], window_size: int = 5
) -> List[Tuple[float, float]]:
    """
    Smooth trajectory using moving average.

    Args:
        trajectory: List of (x, y) positions
        window_size: Window size for smoothing

    Returns:
        Smoothed trajectory
    """
    if len(trajectory) < window_size:
        return trajectory

    smoothed = []
    positions = np.array(trajectory)

    for i in range(len(positions)):
        start = max(0, i - window_size // 2)
        end = min(len(positions), i + window_size // 2 + 1)
        window = positions[start:end]
        smoothed.append(tuple(np.mean(window, axis=0)))

    return smoothed


def detect_direction_changes(
    trajectory: List[Tuple[float, float]], threshold: float = 45.0
) -> int:
    """
    Detect number of significant direction changes in trajectory.

    Args:
        trajectory: List of (x, y) positions
        threshold: Minimum angle change (degrees) to count as direction change

    Returns:
        Number of direction changes
    """
    if len(trajectory) < 3:
        return 0

    positions = np.array(trajectory)
    direction_changes = 0

    for i in range(1, len(positions) - 1):
        # Calculate vectors
        v1 = positions[i] - positions[i - 1]
        v2 = positions[i + 1] - positions[i]

        # Calculate angle between vectors
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))

            if angle_deg > threshold:
                direction_changes += 1

    return direction_changes
