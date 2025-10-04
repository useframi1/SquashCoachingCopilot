"""Movement analysis for player performance metrics."""

from typing import List, Dict, Optional
import numpy as np
from data.data_models import FrameData
from .metrics import (
    calculate_speed,
    calculate_acceleration,
    calculate_total_distance,
    calculate_average_position,
    calculate_court_coverage,
    detect_direction_changes,
)


class MovementAnalyzer:
    """
    Analyzes player movement patterns and metrics.

    Responsibilities:
    - Calculate speed, acceleration, and distance metrics
    - Analyze court coverage and positioning
    - Detect movement patterns (direction changes, sprints, etc.)
    - Provide movement-based insights for coaching
    """

    def __init__(self, fps: float = 30.0):
        """
        Initialize movement analyzer.

        Args:
            fps: Video frames per second
        """
        self.fps = fps

    def analyze_player_movement(
        self, frames: List[FrameData], player_id: int
    ) -> Dict:
        """
        Analyze movement metrics for a single player.

        Args:
            frames: List of frame data
            player_id: Player ID (1 or 2)

        Returns:
            Dictionary containing movement metrics
        """
        # Extract trajectory
        trajectory = self._extract_trajectory(frames, player_id)

        if len(trajectory) < 2:
            return self._empty_movement_metrics()

        # Calculate metrics
        speeds = calculate_speed(trajectory, self.fps)
        accelerations = calculate_acceleration(speeds, self.fps)

        metrics = {
            "total_distance": calculate_total_distance(trajectory),
            "average_position": calculate_average_position(trajectory),
            "court_coverage": calculate_court_coverage(trajectory),
            "max_speed": max(speeds) if speeds else 0.0,
            "average_speed": np.mean(speeds) if speeds else 0.0,
            "max_acceleration": max(accelerations) if accelerations else 0.0,
            "direction_changes": detect_direction_changes(trajectory),
            "trajectory_length": len(trajectory),
        }

        return metrics

    def analyze_both_players(self, frames: List[FrameData]) -> Dict:
        """
        Analyze movement for both players.

        Args:
            frames: List of frame data

        Returns:
            Dictionary with metrics for both players
        """
        return {
            "player1": self.analyze_player_movement(frames, player_id=1),
            "player2": self.analyze_player_movement(frames, player_id=2),
        }

    def get_speed_profile(
        self, frames: List[FrameData], player_id: int
    ) -> List[float]:
        """
        Get instantaneous speed profile for a player.

        Args:
            frames: List of frame data
            player_id: Player ID

        Returns:
            List of speed values
        """
        trajectory = self._extract_trajectory(frames, player_id)
        return calculate_speed(trajectory, self.fps)

    def get_acceleration_profile(
        self, frames: List[FrameData], player_id: int
    ) -> List[float]:
        """
        Get acceleration profile for a player.

        Args:
            frames: List of frame data
            player_id: Player ID

        Returns:
            List of acceleration values
        """
        trajectory = self._extract_trajectory(frames, player_id)
        speeds = calculate_speed(trajectory, self.fps)
        return calculate_acceleration(speeds, self.fps)

    def detect_sprints(
        self, frames: List[FrameData], player_id: int, speed_threshold: float = 3.0
    ) -> List[Dict]:
        """
        Detect sprint events (high-speed movements).

        Args:
            frames: List of frame data
            player_id: Player ID
            speed_threshold: Minimum speed to count as sprint (m/s)

        Returns:
            List of sprint events with start/end frames and metrics
        """
        speeds = self.get_speed_profile(frames, player_id)
        sprints = []
        in_sprint = False
        sprint_start = 0

        for i, speed in enumerate(speeds):
            if speed >= speed_threshold and not in_sprint:
                in_sprint = True
                sprint_start = i
            elif speed < speed_threshold and in_sprint:
                in_sprint = False
                sprint = {
                    "start_frame": sprint_start,
                    "end_frame": i,
                    "duration": (i - sprint_start) / self.fps,
                    "peak_speed": max(speeds[sprint_start:i]),
                }
                sprints.append(sprint)

        return sprints

    def analyze_positioning(
        self, frames: List[FrameData], player_id: int
    ) -> Dict:
        """
        Analyze player positioning on court.

        Args:
            frames: List of frame data
            player_id: Player ID

        Returns:
            Dictionary with positioning metrics
        """
        trajectory = self._extract_trajectory(frames, player_id)

        if not trajectory:
            return {"average_position": None, "court_coverage": 0.0, "positioning_variability": 0.0}

        # Calculate metrics
        avg_position = calculate_average_position(trajectory)
        coverage = calculate_court_coverage(trajectory)

        # Calculate positioning variability (standard deviation)
        positions = np.array(trajectory)
        variability = np.std(positions, axis=0)

        return {
            "average_position": avg_position,
            "court_coverage": coverage,
            "positioning_variability": float(np.mean(variability)),
        }

    def _extract_trajectory(
        self, frames: List[FrameData], player_id: int
    ) -> List[tuple]:
        """Extract player trajectory from frame data."""
        trajectory = []

        for frame in frames:
            player = frame.get_player(player_id)
            if player and player.is_valid() and player.real_position:
                trajectory.append(player.real_position)

        return trajectory

    def _empty_movement_metrics(self) -> Dict:
        """Return empty movement metrics structure."""
        return {
            "total_distance": 0.0,
            "average_position": None,
            "court_coverage": 0.0,
            "max_speed": 0.0,
            "average_speed": 0.0,
            "max_acceleration": 0.0,
            "direction_changes": 0,
            "trajectory_length": 0,
        }
