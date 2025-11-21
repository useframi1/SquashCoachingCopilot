"""
Shot classification module for squash ball tracking.

Classifies shots based on player positions and wall hits in meter coordinates,
combining direction, depth, and spatial analysis to determine shot types.
Uses DataFrame-based pipeline architecture.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

from squashcopilot.common.utils import load_config
from squashcopilot.common.types.enums import ShotDirection, ShotDepth, ShotType
from squashcopilot.common.models import ShotClassificationInput, ShotClassificationOutput


class ShotClassifier:
    """
    Classify shots based on meter-based player position and wall hit analysis.

    Uses rule-based classification to determine shot direction and depth,
    then combines them into specific shot types.
    """

    def __init__(self, config: dict = None, fps: float = 30):
        """
        Initialize shot classifier.

        Args:
            config: Configuration dictionary (loaded from config.json if None)
            fps: Video frame rate
        """
        self.config = (
            config if config else load_config(config_name="shot_type_classification")
        )
        self.fps = fps

        # Load thresholds from config
        shot_config = self.config.get("shot_classification", {})
        self.court_center_x = shot_config.get("court_center_x_m", 4.57)
        self.dir_thresholds = shot_config.get("direction_thresholds", {})
        self.depth_thresholds = shot_config.get("depth_thresholds", {})

    def classify_shots(
        self,
        input_data: ShotClassificationInput,
    ) -> ShotClassificationOutput:
        """
        Classify shots and add columns to DataFrame.

        Args:
            input_data: ShotClassificationInput with df and segments

        Returns:
            ShotClassificationOutput with df containing added columns:
            shot_direction, shot_depth, shot_type
        """
        df = input_data.df
        segments = input_data.segments

        df = df.copy()
        df["shot_direction"] = ""
        df["shot_depth"] = ""
        df["shot_type"] = ""

        # Process each rally segment
        for segment in segments:
            rally_df = df.loc[segment.start_frame : segment.end_frame]

            # Get racket hit frames in this rally (handle both bool and int columns)
            racket_hit_frames = rally_df[rally_df["is_racket_hit"] == True].index.tolist()

            if len(racket_hit_frames) < 2:
                continue

            # Get wall hit frames in this rally (handle both bool and int columns)
            wall_hit_df = rally_df[rally_df["is_wall_hit"] == True]

            # Process consecutive racket hits
            for i in range(len(racket_hit_frames) - 1):
                start_frame = racket_hit_frames[i]
                end_frame = racket_hit_frames[i + 1]

                # Find wall hit between these frames
                wall_hits_between = wall_hit_df[
                    (wall_hit_df.index > start_frame) & (wall_hit_df.index < end_frame)
                ]

                wall_hit_frame = None
                wall_hit_y = None
                if len(wall_hits_between) > 0:
                    # Take first wall hit
                    wall_hit_frame = wall_hits_between.index[0]
                    wall_hit_x = df.loc[wall_hit_frame, "wall_hit_x_meter"]

                # Extract features and classify
                features = self._extract_shot_features(
                    df, start_frame, end_frame, wall_hit_frame, wall_hit_x
                )

                direction = self._classify_direction(features)
                depth = self._classify_depth(features)
                shot_type = self._determine_shot_type(direction, depth)

                # Update DataFrame at the racket hit frame
                df.loc[start_frame, "shot_direction"] = (
                    direction.value if direction else ""
                )
                df.loc[start_frame, "shot_depth"] = depth.value if depth else ""
                df.loc[start_frame, "shot_type"] = shot_type.value if shot_type else ""

        # Compute stats
        stats = self.get_shot_stats(df)

        return ShotClassificationOutput(
            df=df,
            num_shots=stats["num_shots"],
            shot_counts=stats["shot_counts"],
        )

    def _extract_shot_features(
        self,
        df: pd.DataFrame,
        start_frame: int,
        end_frame: int,
        wall_hit_frame: Optional[int],
        wall_hit_x: Optional[float],
    ) -> Dict:
        """
        Extract features for shot classification.

        Args:
            df: DataFrame with positions
            start_frame: Frame of racket hit
            end_frame: Frame of next racket hit
            wall_hit_frame: Frame of wall hit (if any)
            wall_hit_x: X position of wall hit in meters
            calibration: Court calibration

        Returns:
            Dictionary of extracted features
        """
        features = {
            "has_wall_hit": False,
            "is_cross_court": False,
            "rebound_distance": None,
        }

        if wall_hit_frame is None:
            return features

        # Get hitting player ID
        hitting_player_id = int(df.loc[start_frame, "racket_hit_player_id"])

        # Get player positions at the time of hits
        if hitting_player_id == 1:
            hitting_x = df.loc[start_frame, "player_1_x_meter"]
            hitting_y = df.loc[start_frame, "player_1_y_meter"]
            receiving_x = df.loc[end_frame, "player_2_x_meter"]
            receiving_y = df.loc[end_frame, "player_2_y_meter"]
        else:
            hitting_x = df.loc[start_frame, "player_2_x_meter"]
            hitting_y = df.loc[start_frame, "player_2_y_meter"]
            receiving_x = df.loc[end_frame, "player_1_x_meter"]
            receiving_y = df.loc[end_frame, "player_1_y_meter"]

        # Skip if positions are NaN
        if any(pd.isna([hitting_x, hitting_y, receiving_x, receiving_y])):
            return features

        # Determine direction (cross-court vs straight)
        player_side = "left" if hitting_x < self.court_center_x else "right"
        wall_side = "left" if wall_hit_x < self.court_center_x else "right"
        is_cross_court = player_side != wall_side

        # Calculate rebound distance (wall to receiving player)
        wall_pos_array = np.array([wall_hit_x, 0.0])  # Front wall y=0
        receiving_pos_array = np.array([receiving_x, receiving_y])
        rebound_distance = np.linalg.norm(receiving_pos_array - wall_pos_array)

        features.update(
            {
                "has_wall_hit": True,
                "is_cross_court": is_cross_court,
                "rebound_distance": float(rebound_distance),
            }
        )

        return features

    def _classify_direction(self, features: Dict) -> Optional[ShotDirection]:
        """Classify shot direction based on court side crossing."""
        if not features["has_wall_hit"]:
            return ShotDirection.STRAIGHT

        if features["is_cross_court"]:
            return ShotDirection.CROSS_COURT
        else:
            return ShotDirection.STRAIGHT

    def _classify_depth(self, features: Dict) -> Optional[ShotDepth]:
        """Classify shot depth based on rebound distance."""
        if not features["has_wall_hit"]:
            return ShotDepth.LONG

        rebound_dist = features["rebound_distance"]
        drop_max = self.depth_thresholds.get("drop_max_rebound_distance_m", 3.0)

        if rebound_dist < drop_max:
            return ShotDepth.DROP
        else:
            return ShotDepth.LONG

    def _determine_shot_type(
        self, direction: Optional[ShotDirection], depth: Optional[ShotDepth]
    ) -> Optional[ShotType]:
        """Combine direction and depth into final shot type."""
        if direction is None or depth is None:
            return None

        mapping = {
            (ShotDirection.STRAIGHT, ShotDepth.LONG): ShotType.STRAIGHT_DRIVE,
            (ShotDirection.STRAIGHT, ShotDepth.DROP): ShotType.STRAIGHT_DROP,
            (ShotDirection.CROSS_COURT, ShotDepth.LONG): ShotType.CROSS_COURT_DRIVE,
            (ShotDirection.CROSS_COURT, ShotDepth.DROP): ShotType.CROSS_COURT_DROP,
        }

        return mapping.get((direction, depth), ShotType.STRAIGHT_DRIVE)

    def get_shot_stats(self, df: pd.DataFrame) -> Dict:
        """Get shot classification statistics."""
        # Count shots (racket hits with shot types assigned)
        shots_df = df[df["shot_type"] != ""]
        num_shots = len(shots_df)

        shot_counts = shots_df["shot_type"].value_counts().to_dict()

        return {
            "num_shots": num_shots,
            "shot_counts": shot_counts,
        }
