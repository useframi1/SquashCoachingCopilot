"""
Racket hit detection for squash ball tracking.

Detects racket hits by finding steep negative slopes (downward) before wall hits.
Uses DataFrame-based pipeline architecture.
"""

import numpy as np
import pandas as pd
from typing import List, Dict

from squashcopilot.common.utils import load_config
from squashcopilot.common.models import (
    RallySegment,
    RacketHitDetectionInput,
    RacketHitDetectionOutput,
)


class RacketHitDetector:
    """Detects racket hits using steep negative slopes before wall hits.

    Racket hits create steep negative slopes in Y-coordinate (ball accelerating
    toward wall). This algorithm:
    1. Takes DataFrame with wall hit markers
    2. Looks backward from each wall hit for steep negative slopes
    3. Identifies the point with the steepest downward acceleration
    4. Attributes hits to players based on proximity
    """

    def __init__(self, config: dict = None):
        """Initialize the racket hit detector.

        Args:
            config: Configuration dictionary. If None, loads from config.json
        """
        if config is None:
            config = load_config(config_name="hit_detection")

        racket_config = config.get("racket_hit_detection", {})
        self.slope_window = racket_config.get("slope_window", 5)
        self.slope_threshold = racket_config.get("slope_threshold", 15.0)
        self.min_distance = racket_config.get("min_distance", 15)
        self.lookback_frames = racket_config.get("lookback_frames", 20)
        self.confidence_ratio = racket_config.get("confidence_ratio", 0.7)

    def detect_racket_hits(
        self,
        input_data: RacketHitDetectionInput,
    ) -> RacketHitDetectionOutput:
        """Detect racket hits and add columns to DataFrame.

        Args:
            input_data: RacketHitDetectionInput with df and segments

        Returns:
            RacketHitDetectionOutput with df containing added columns:
            is_racket_hit, racket_hit_player_id
        """
        df = input_data.df
        segments = input_data.segments

        df = df.copy()
        df['is_racket_hit'] = False
        df['racket_hit_player_id'] = -1

        # Process each rally segment
        for segment in segments:
            rally_df = df.loc[segment.start_frame:segment.end_frame]
            frame_numbers = rally_df.index.tolist()

            # Get wall hit frames in this rally
            wall_hit_frames = rally_df[rally_df['is_wall_hit']].index.tolist()

            if not wall_hit_frames:
                continue

            y_coords = rally_df['ball_y'].values

            # Track racket hits for player attribution
            racket_hits_in_rally = []

            for wall_hit_frame in wall_hit_frames:
                # Get relative position in rally
                wall_hit_idx = frame_numbers.index(wall_hit_frame)

                # Define search window: look back from wall hit
                search_start_idx = max(0, wall_hit_idx - self.lookback_frames)
                search_end_idx = wall_hit_idx

                if search_end_idx - search_start_idx < self.slope_window:
                    continue

                # Calculate slopes in the search window
                min_slope = np.inf
                min_slope_idx = None

                for i in range(search_start_idx, search_end_idx - self.slope_window):
                    y_start = y_coords[i]
                    y_end = y_coords[i + self.slope_window]

                    # Skip if NaN
                    if np.isnan(y_start) or np.isnan(y_end):
                        continue

                    y_change = y_end - y_start
                    slope = y_change / self.slope_window

                    if slope < min_slope and slope < -self.slope_threshold:
                        min_slope = slope
                        min_slope_idx = i

                # If we found a steep enough negative slope, record it
                if min_slope_idx is not None:
                    racket_hit_frame = frame_numbers[min_slope_idx]

                    # Check if this hit is far enough from previous hits
                    if racket_hits_in_rally:
                        last_hit_frame = racket_hits_in_rally[-1]
                        if (racket_hit_frame - last_hit_frame) < self.min_distance:
                            continue

                    racket_hits_in_rally.append(racket_hit_frame)

            # Attribute hits to players
            player_assignments = self._attribute_hits_to_players(
                racket_hits_in_rally, df
            )

            # Mark racket hits in DataFrame
            for frame_num, player_id in player_assignments.items():
                df.loc[frame_num, 'is_racket_hit'] = True
                df.loc[frame_num, 'racket_hit_player_id'] = player_id

        # Compute stats
        stats = self.get_racket_hit_stats(df, segments)

        return RacketHitDetectionOutput(
            df=df,
            num_racket_hits=stats["num_racket_hits"],
            racket_hits_per_rally=stats["racket_hits_per_rally"],
        )

    def _attribute_hits_to_players(
        self,
        racket_hit_frames: List[int],
        df: pd.DataFrame,
    ) -> Dict[int, int]:
        """Attribute racket hits to players based on distance comparison.

        Args:
            racket_hit_frames: List of frame numbers with racket hits
            df: DataFrame with player positions

        Returns:
            Dictionary mapping frame_number -> player_id
        """
        if not racket_hit_frames:
            return {}

        assignments = {}

        # Determine first player based on distance comparison
        first_frame = racket_hit_frames[0]
        first_hit_x = df.loc[first_frame, 'ball_x']
        first_hit_y = df.loc[first_frame, 'ball_y']

        # Calculate distance to each player at the first hit frame
        player_1_x = df.loc[first_frame, 'player_1_x_pixel']
        player_1_y = df.loc[first_frame, 'player_1_y_pixel']
        player_2_x = df.loc[first_frame, 'player_2_x_pixel']
        player_2_y = df.loc[first_frame, 'player_2_y_pixel']

        dist_1 = np.sqrt((first_hit_x - player_1_x)**2 + (first_hit_y - player_1_y)**2)
        dist_2 = np.sqrt((first_hit_x - player_2_x)**2 + (first_hit_y - player_2_y)**2)

        # Handle NaN distances
        if np.isnan(dist_1):
            dist_1 = float('inf')
        if np.isnan(dist_2):
            dist_2 = float('inf')

        current_player = 1 if dist_1 <= dist_2 else 2

        # Assign player IDs with alternation and validation
        for frame_num in racket_hit_frames:
            hit_x = df.loc[frame_num, 'ball_x']
            hit_y = df.loc[frame_num, 'ball_y']

            # Calculate distances to both players
            p1_x = df.loc[frame_num, 'player_1_x_pixel']
            p1_y = df.loc[frame_num, 'player_1_y_pixel']
            p2_x = df.loc[frame_num, 'player_2_x_pixel']
            p2_y = df.loc[frame_num, 'player_2_y_pixel']

            d1 = np.sqrt((hit_x - p1_x)**2 + (hit_y - p1_y)**2)
            d2 = np.sqrt((hit_x - p2_x)**2 + (hit_y - p2_y)**2)

            # Handle NaN
            if np.isnan(d1):
                d1 = float('inf')
            if np.isnan(d2):
                d2 = float('inf')

            # Validate assignment if we have both player positions
            if d1 != float('inf') and d2 != float('inf'):
                min_dist = min(d1, d2)
                max_dist = max(d1, d2)
                closer_player = 1 if d1 < d2 else 2

                # Check if there's a clear winner
                if max_dist > 0 and (min_dist / max_dist) < self.confidence_ratio:
                    if closer_player != current_player:
                        current_player = closer_player

            assignments[frame_num] = current_player

            # Alternate for next hit
            current_player = 2 if current_player == 1 else 1

        return assignments

    def get_racket_hit_stats(
        self, df: pd.DataFrame, segments: List[RallySegment]
    ) -> Dict:
        """Get racket hit statistics.

        Args:
            df: DataFrame with is_racket_hit column
            segments: List of rally segments

        Returns:
            Dictionary with stats: num_racket_hits, racket_hits_per_rally
        """
        num_racket_hits = int(df['is_racket_hit'].sum())

        racket_hits_per_rally = {}
        for segment in segments:
            rally_df = df.loc[segment.start_frame:segment.end_frame]
            racket_hits_per_rally[segment.rally_id] = int(rally_df['is_racket_hit'].sum())

        return {
            'num_racket_hits': num_racket_hits,
            'racket_hits_per_rally': racket_hits_per_rally,
        }
