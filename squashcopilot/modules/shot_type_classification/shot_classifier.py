"""
Shot classification module for squash ball tracking.

Classifies shots based on player positions and wall hits in meter coordinates,
combining direction, depth, and spatial analysis to determine shot types.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from squashcopilot.common.utils import load_config

from squashcopilot.common import (
    ShotDirection,
    ShotDepth,
    ShotType,
    ShotClassificationInput,
    ShotResult,
    ShotClassificationResult,
    Point2D,
)


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
            fps: Video frame rate (not currently used, kept for compatibility)
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

    def classify(self, input_data: ShotClassificationInput) -> ShotClassificationResult:
        """
        Main classification method.

        Args:
            input_data: ShotClassificationInput with player positions and hit events

        Returns:
            ShotClassificationResult with structured shot classifications
        """
        # Convert WallHit/RacketHit to dicts for processing
        wall_hits_dicts = [hit.to_dict() for hit in input_data.wall_hits]
        racket_hits_dicts = [hit.to_dict() for hit in input_data.racket_hits]

        # Match consecutive racket hits (define shot boundaries)
        shot_segments = self._match_consecutive_racket_hits(racket_hits_dicts)

        shots = []
        for start_hit, end_hit in shot_segments:
            # Find wall hit within this shot's time window
            wall_hit = self._find_wall_hit_in_shot(
                wall_hits_dicts, start_hit["frame"], end_hit["frame"]
            )

            # Extract features (includes wall hit if detected)
            features = self._extract_shot_features(
                start_hit,
                end_hit,
                wall_hit,
                input_data.player1_positions_meter,
                input_data.player2_positions_meter,
            )

            # Classify direction and depth
            direction = self._classify_direction(features)
            depth = self._classify_depth(features)

            # Combine into shot type
            shot_type = self._determine_shot_type(direction, depth)

            # Adjust confidence based on whether wall hit was detected
            confidence = 1.0 if wall_hit is not None else 0.7

            # Create ShotResult
            # For wall_hit_pos, use pixel coordinates from the original wall_hit dict
            wall_hit_pos_pixel = None
            if wall_hit is not None:
                wall_hit_pos_pixel = Point2D(
                    x=wall_hit["position"]["x"],
                    y=wall_hit["position"]["y"]
                )

            shot_result = ShotResult(
                frame=start_hit["frame"],
                direction=direction,
                depth=depth,
                shot_type=shot_type,
                racket_hit_pos=Point2D(
                    x=start_hit["position"]["x"], y=start_hit["position"]["y"]
                ),
                next_racket_hit_pos=Point2D(
                    x=end_hit["position"]["x"], y=end_hit["position"]["y"]
                ),
                wall_hit_pos=wall_hit_pos_pixel,
                wall_hit_frame=features.get("wall_hit_frame"),
                rebound_distance=features["rebound_distance"],
                confidence=confidence,
            )
            shots.append(shot_result)

        return ShotClassificationResult(shots=shots)

    def _match_consecutive_racket_hits(
        self, racket_hits: List[Dict]
    ) -> List[Tuple[Dict, Dict]]:
        """
        Match each racket hit to the next consecutive racket hit.

        Args:
            racket_hits: List of racket hit dicts (sorted by frame)

        Returns:
            List of (start_racket_hit, end_racket_hit) pairs
        """
        matched_pairs = []

        # Pair consecutive racket hits
        for i in range(len(racket_hits) - 1):
            start_hit = racket_hits[i]
            end_hit = racket_hits[i + 1]
            matched_pairs.append((start_hit, end_hit))

        return matched_pairs

    def _find_wall_hit_in_shot(
        self,
        wall_hits: List[Dict],
        start_frame: int,
        end_frame: int,
    ) -> Optional[Dict]:
        """
        Find the primary (first) wall hit within a shot's time window.

        Args:
            wall_hits: List of wall hit dicts with {frame, x, y, prominence}
            start_frame: Start of shot (racket hit frame)
            end_frame: End of shot (next racket hit frame)

        Returns:
            First wall hit dict within the time window, or None if not found
        """
        candidate_wall_hits = []

        for wall_hit in wall_hits:
            wall_frame = wall_hit["frame"]

            # Wall hit must be after racket hit and before next racket hit
            if start_frame < wall_frame < end_frame:
                candidate_wall_hits.append(wall_hit)

        # Return the first (earliest) wall hit if any exist
        if candidate_wall_hits:
            return min(candidate_wall_hits, key=lambda w: w["frame"])

        return None

    def _extract_shot_features(
        self,
        start_hit: Dict,
        end_hit: Dict,
        wall_hit: Optional[Dict],
        player1_positions: List[Optional[Point2D]],
        player2_positions: List[Optional[Point2D]],
    ) -> Dict:
        """
        Extract vector-based features for a shot using player positions in meters.

        Constructs two vectors:
        1. Hitting Player Position → Wall (attack vector)
        2. Wall → Receiving Player Position (rebound vector)

        Args:
            start_hit: Starting racket hit dict (with player_id)
            end_hit: Ending racket hit dict (next player's hit)
            wall_hit: Optional wall hit dict (front wall with position_meter)
            player1_positions: List of player 1 positions in meters
            player2_positions: List of player 2 positions in meters

        Returns:
            Dictionary of extracted features including vectors and direction
        """
        # Initialize features
        features = {
            "has_wall_hit": False,
            "is_cross_court": False,
            "rebound_distance": None,
            "wall_hit_pos": None,
            "wall_hit_frame": None,
        }

        # Extract vector-based features if wall hit detected
        if wall_hit is not None:
            # Get player IDs
            hitting_player_id = start_hit.get("player_id", 1)

            # Get frame numbers
            start_frame = start_hit["frame"]
            end_frame = end_hit["frame"]

            # Get player positions at the time of racket hits
            if hitting_player_id == 1:
                hitting_player_pos = player1_positions[start_frame]
                receiving_player_pos = player2_positions[end_frame]
            else:
                hitting_player_pos = player2_positions[start_frame]
                receiving_player_pos = player1_positions[end_frame]

            # Check if we have valid positions
            if hitting_player_pos is None or receiving_player_pos is None:
                return features

            # Get wall hit position in meters (x, 0) as specified
            # Use position_meter if available, otherwise fallback to position
            if "position_meter" in wall_hit and wall_hit["position_meter"] is not None:
                wall_x_meter = wall_hit["position_meter"]["x"]
            else:
                # Fallback to pixel position if meter not available
                wall_x_meter = wall_hit["position"]["x"]

            # Construct positions as numpy arrays for rebound distance calculation
            wall_pos = np.array([wall_x_meter, 0.0])  # Wall at (x, 0)
            receiving_pos = np.array([receiving_player_pos.x, receiving_player_pos.y])

            # Vector: Wall → Receiving Player (rebound)
            vec_wall_to_player = receiving_pos - wall_pos

            # Determine direction based on court side crossing
            # Check if the ball went from one side of the court to the other
            player_x = hitting_player_pos.x
            wall_x = wall_x_meter

            # Determine if it's a cross-court shot
            # If player and wall hit are on opposite sides of the court center, it's cross-court
            player_side = "left" if player_x < self.court_center_x else "right"
            wall_side = "left" if wall_x < self.court_center_x else "right"
            is_cross_court = (player_side != wall_side)

            # Rebound distance (length of wall → receiving player vector) in meters
            rebound_distance = np.linalg.norm(vec_wall_to_player)

            features.update(
                {
                    "has_wall_hit": True,
                    "is_cross_court": is_cross_court,
                    "rebound_distance": float(rebound_distance),
                    "wall_hit_pos": (wall_x_meter, 0.0),
                    "wall_hit_frame": wall_hit["frame"],
                }
            )

        return features

    def _classify_direction(self, features: Dict) -> ShotDirection:
        """
        Classify shot direction based on court side crossing.

        Determines if the shot is straight or cross-court by checking if the ball
        crossed from one side of the court to the other:
        - Same side → Straight shot
        - Opposite side → Cross-court shot

        Args:
            features: Extracted shot features with is_cross_court flag

        Returns:
            ShotDirection enum value (STRAIGHT or CROSS_COURT only)
        """
        # Require wall hit for accurate classification
        if not features["has_wall_hit"]:
            return ShotDirection.STRAIGHT  # Default fallback

        is_cross_court = features.get("is_cross_court", False)

        if is_cross_court:
            return ShotDirection.CROSS_COURT
        else:
            return ShotDirection.STRAIGHT

    def _classify_depth(self, features: Dict) -> ShotDepth:
        """
        Classify shot depth based on rebound distance (wall→receiving player vector length).

        Uses the length of the wall→receiving player vector in meters:
        - Short rebound distance → Drop shot (player doesn't move far from wall)
        - Long rebound distance → Long shot/Drive (player is deeper in court)

        Args:
            features: Extracted shot features with rebound_distance in meters

        Returns:
            ShotDepth enum value
        """
        # Require wall hit for accurate classification
        if not features["has_wall_hit"]:
            return ShotDepth.LONG  # Default fallback

        rebound_dist = features["rebound_distance"]

        # Distance threshold in meters
        drop_max = self.depth_thresholds.get("drop_max_rebound_distance_m", 3.0)

        if rebound_dist < drop_max:
            return ShotDepth.DROP
        else:
            return ShotDepth.LONG

    def _determine_shot_type(
        self, direction: ShotDirection, depth: ShotDepth
    ) -> ShotType:
        """
        Combine direction and depth into final shot type.

        Args:
            direction: Classified direction (STRAIGHT or CROSS_COURT)
            depth: Classified depth (DROP or LONG)

        Returns:
            ShotType enum value
        """
        # Mapping from (direction, depth) to ShotType
        # Only STRAIGHT and CROSS_COURT directions are used
        mapping = {
            (ShotDirection.STRAIGHT, ShotDepth.LONG): ShotType.STRAIGHT_DRIVE,
            (ShotDirection.STRAIGHT, ShotDepth.DROP): ShotType.STRAIGHT_DROP,
            (ShotDirection.CROSS_COURT, ShotDepth.LONG): ShotType.CROSS_COURT_DRIVE,
            (ShotDirection.CROSS_COURT, ShotDepth.DROP): ShotType.CROSS_COURT_DROP,
        }

        return mapping.get((direction, depth), ShotType.STRAIGHT_DRIVE)
