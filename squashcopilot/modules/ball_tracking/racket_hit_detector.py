"""
Racket hit detection for squash ball tracking.

Detects racket hits by finding steep negative slopes (downward) before wall hits.
"""

import numpy as np
import math
from squashcopilot.common.utils import load_config

from squashcopilot.common import (
    Point2D,
    RacketHitInput,
    RacketHit,
    RacketHitDetectionResult,
)


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
            config = load_config(config_name="ball_tracking")

        racket_config = config.get("racket_hit_detection", {})
        self.slope_window = racket_config.get("slope_window", 5)
        self.slope_threshold = racket_config.get("slope_threshold", 15.0)
        self.min_distance = racket_config.get("min_distance", 15)
        self.lookback_frames = racket_config.get("lookback_frames", 20)

    def detect(self, input_data: RacketHitInput) -> RacketHitDetectionResult:
        """Detect racket hits using steep negative slopes (downward) before wall hits.

        Args:
            input_data: RacketHitInput with positions, wall hits, player positions, and config

        Returns:
            RacketHitDetectionResult with detected racket hits
        """
        # Convert Point2D to tuples for processing
        positions_tuples = [(p.x, p.y) for p in input_data.positions]

        # Override config if provided
        if input_data.config:
            slope_window = input_data.config.get(
                "racket_hit_detection.slope_window", self.slope_window
            )
            slope_threshold = input_data.config.get(
                "racket_hit_detection.slope_threshold", self.slope_threshold
            )
            min_distance = input_data.config.get(
                "racket_hit_detection.min_distance", self.min_distance
            )
            lookback_frames = input_data.config.get(
                "racket_hit_detection.lookback_frames", self.lookback_frames
            )
        else:
            slope_window = self.slope_window
            slope_threshold = self.slope_threshold
            min_distance = self.min_distance
            lookback_frames = self.lookback_frames

        if (
            len(positions_tuples) < slope_window + lookback_frames
            or not input_data.wall_hits
        ):
            return RacketHitDetectionResult(racket_hits=[])

        # Extract Y coordinates
        y_coords = np.array([p[1] for p in positions_tuples])
        x_coords = np.array([p[0] for p in positions_tuples])

        # For each wall hit, look backward for steep negative slope (racket hit)
        racket_hits = []

        for wall_hit in input_data.wall_hits:
            wall_hit_frame = wall_hit.frame

            # Define search window: look back from wall hit
            search_start = max(0, wall_hit_frame - lookback_frames)
            search_end = wall_hit_frame

            if search_end - search_start < slope_window:
                continue

            # Calculate slopes in the search window
            min_slope = np.inf
            min_slope_frame = None

            for i in range(search_start, search_end - slope_window):
                y_change = y_coords[i + slope_window] - y_coords[i]
                slope = y_change / slope_window

                if slope < min_slope and slope < -slope_threshold:
                    min_slope = slope
                    min_slope_frame = i

            # If we found a steep enough negative slope, record it as a racket hit
            if min_slope_frame is not None:
                # Check if this hit is far enough from previous hits
                if (
                    racket_hits
                    and (min_slope_frame - racket_hits[-1].frame) < min_distance
                ):
                    continue

                # Placeholder player_id (will be assigned later)
                racket_hit = RacketHit(
                    frame=int(min_slope_frame),
                    position=Point2D(
                        x=float(x_coords[min_slope_frame]),
                        y=float(y_coords[min_slope_frame]),
                    ),
                    slope=float(min_slope),
                    player_id=0,  # Will be assigned in attribution step
                )
                racket_hits.append(racket_hit)

        # Attribute racket hits to players
        racket_hits = self._attribute_hits_to_players(
            racket_hits, input_data.player_positions
        )

        return RacketHitDetectionResult(racket_hits=racket_hits)

    def _attribute_hits_to_players(self, racket_hits, player_positions):
        """Attribute racket hits to players based on distance comparison.

        For the first hit, determine player by comparing distances to both players.
        The player closest to the hit gets attributed.
        For subsequent hits, alternate between players.

        Args:
            racket_hits: List of RacketHit objects
            player_positions: Dict mapping player_id to list of positions

        Returns:
            List of RacketHit objects with player_id assigned
        """
        if not racket_hits or not player_positions:
            return racket_hits

        # Determine first player based on distance comparison
        first_hit = racket_hits[0]
        first_hit_pos = first_hit.position
        first_hit_frame = first_hit.frame

        # Calculate distance to each player at the first hit frame
        min_distance = float("inf")
        first_player = 1  # Default to player 1

        for player_id, positions in player_positions.items():
            if first_hit_frame < len(positions):
                player_pos = positions[first_hit_frame]
                # Calculate Euclidean distance
                dx = first_hit_pos.x - player_pos.x
                dy = first_hit_pos.y - player_pos.y
                distance = np.sqrt(dx**2 + dy**2)

                if distance < min_distance:
                    min_distance = distance
                    first_player = player_id

        # Assign to the closest player (no threshold check)
        current_player = first_player

        # Assign player IDs with alternation
        updated_hits = []
        for hit in racket_hits:
            # Create new RacketHit with updated player_id
            updated_hit = RacketHit(
                frame=hit.frame,
                position=hit.position,
                slope=hit.slope,
                player_id=current_player,
            )
            updated_hits.append(updated_hit)

            # Alternate for next hit
            current_player = 2 if current_player == 1 else 1

        return updated_hits
