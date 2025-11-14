"""
Feature extraction utilities for shot classification.

All features are computed in pixel coordinates for consistency
and to avoid 3D projection errors when the ball is airborne.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class ShotFeatureExtractor:
    """
    Extracts pixel-based features from ball trajectories for shot classification.

    All spatial features are in pixels, temporal features in frames.
    """

    def __init__(self, config: dict):
        """
        Initialize feature extractor with configuration.

        Args:
            config: Configuration dictionary with feature_windows section
        """
        feature_config = config.get("shot_classification", {}).get(
            "feature_windows", {}
        )
        self.velocity_window = feature_config.get("velocity_frames", 5)
        self.acceleration_window = feature_config.get("acceleration_frames", 10)

    def calculate_velocity(
        self, positions: List[Tuple[float, float]], frame_idx: int, window: int = None
    ) -> Tuple[float, float, float]:
        """
        Calculate ball velocity using rolling window average.

        Args:
            positions: List of (x, y) ball positions in pixels
            frame_idx: Frame index to calculate velocity at
            window: Number of frames to average over (default: from config)

        Returns:
            Tuple of (vx, vy, speed) where:
                vx: Horizontal velocity in pixels/frame
                vy: Vertical velocity in pixels/frame
                speed: Magnitude of velocity in pixels/frame
        """
        if window is None:
            window = self.velocity_window

        # Get window of positions around frame_idx
        start = max(0, frame_idx - window // 2)
        end = min(len(positions), frame_idx + window // 2 + 1)

        # Filter out None positions
        valid_positions = [
            (i, pos) for i, pos in enumerate(positions[start:end], start=start) if pos and pos[0] is not None
        ]

        if len(valid_positions) < 2:
            return 0.0, 0.0, 0.0

        # Calculate velocities between consecutive positions
        velocities = []
        for i in range(len(valid_positions) - 1):
            idx1, (x1, y1) = valid_positions[i]
            idx2, (x2, y2) = valid_positions[i + 1]

            frame_diff = idx2 - idx1
            if frame_diff > 0:
                vx = (x2 - x1) / frame_diff
                vy = (y2 - y1) / frame_diff
                velocities.append((vx, vy))

        if not velocities:
            return 0.0, 0.0, 0.0

        # Average velocities
        vx_avg = np.mean([v[0] for v in velocities])
        vy_avg = np.mean([v[1] for v in velocities])
        speed = np.sqrt(vx_avg**2 + vy_avg**2)

        return float(vx_avg), float(vy_avg), float(speed)

    def get_lateral_displacement(
        self, racket_pos: Tuple[float, float], wall_pos: Tuple[float, float]
    ) -> float:
        """
        Calculate horizontal (lateral) displacement of shot.

        Args:
            racket_pos: (x, y) position at racket hit in pixels
            wall_pos: (x, y) position at wall hit in pixels

        Returns:
            Lateral displacement (Δx) in pixels
            Positive = moved right, Negative = moved left
        """
        return wall_pos[0] - racket_pos[0]

    def get_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        """
        Calculate Euclidean distance between two pixel positions.

        Args:
            pos1: First position (x, y) in pixels
            pos2: Second position (x, y) in pixels

        Returns:
            Distance in pixels
        """
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return float(np.sqrt(dx**2 + dy**2))

    def get_player_at_hit(
        self,
        ball_pos: Tuple[float, float],
        player_positions: Dict[int, List[Tuple[float, float]]],
        frame_idx: int,
        max_distance_px: float = 120,
    ) -> Optional[int]:
        """
        Determine which player hit the ball based on proximity.

        Args:
            ball_pos: Ball position (x, y) at hit frame in pixels
            player_positions: Dict mapping player_id to list of (x, y) positions
            frame_idx: Frame index of the hit
            max_distance_px: Maximum distance to attribute hit to player

        Returns:
            Player ID (1 or 2) if within threshold, None otherwise
        """
        if not player_positions or ball_pos is None or ball_pos[0] is None:
            return None

        min_distance = float("inf")
        closest_player = None

        for player_id, positions in player_positions.items():
            if frame_idx >= len(positions):
                continue

            player_pos = positions[frame_idx]

            # Skip if player position is invalid
            if player_pos is None or player_pos[0] is None:
                continue

            # Calculate distance
            distance = self.get_distance(ball_pos, player_pos)

            if distance < min_distance:
                min_distance = distance
                closest_player = player_id

        # Only attribute if within threshold
        if min_distance <= max_distance_px:
            return closest_player

        return None

    def extract_trajectory_segment(
        self, positions: List[Tuple[float, float]], start_frame: int, end_frame: int
    ) -> List[Tuple[float, float]]:
        """
        Extract a segment of the trajectory between two frames.

        Args:
            positions: Full trajectory positions
            start_frame: Starting frame index
            end_frame: Ending frame index

        Returns:
            List of positions in the segment
        """
        return positions[start_frame : end_frame + 1]

    def calculate_average_velocity_segment(
        self, positions: List[Tuple[float, float]], start_frame: int, end_frame: int
    ) -> float:
        """
        Calculate average velocity over a trajectory segment.

        Args:
            positions: Full trajectory positions
            start_frame: Starting frame index
            end_frame: Ending frame index

        Returns:
            Average velocity in pixels/frame
        """
        segment = self.extract_trajectory_segment(positions, start_frame, end_frame)

        # Filter out None positions
        valid_positions = [(i, pos) for i, pos in enumerate(segment) if pos and pos[0] is not None]

        if len(valid_positions) < 2:
            return 0.0

        # Calculate total distance
        total_distance = 0.0
        for i in range(len(valid_positions) - 1):
            _, pos1 = valid_positions[i]
            _, pos2 = valid_positions[i + 1]
            total_distance += self.get_distance(pos1, pos2)

        # Calculate total time (in frames)
        total_frames = end_frame - start_frame

        if total_frames <= 0:
            return 0.0

        return total_distance / total_frames
