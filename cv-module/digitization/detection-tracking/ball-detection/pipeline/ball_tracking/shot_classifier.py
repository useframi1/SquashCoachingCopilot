"""
Shot classification module for squash ball tracking.

Classifies shots based on trajectory analysis in pixel coordinates,
combining direction, depth, and speed to determine shot types.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from .shot_types import Shot, ShotType, ShotDirection, ShotDepth
from .utils import load_config


class ShotClassifier:
    """
    Classify shots based on pixel-based trajectory analysis.

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
        self.config = config if config else load_config()
        self.fps = fps

        # Load thresholds from config
        shot_config = self.config.get("shot_classification", {})
        self.dir_thresholds = shot_config.get("direction_thresholds", {})
        self.depth_thresholds = shot_config.get("depth_thresholds", {})

    def classify(
        self,
        ball_positions: List[Tuple[float, float]],
        wall_hits: List[Dict],
        racket_hits: List[Dict],
    ) -> List[Shot]:
        """
        Main classification method.

        Args:
            ball_positions: List of (x, y) positions in pixels for each frame
            wall_hits: List of wall hit dicts with {frame, x, y, prominence}
            racket_hits: List of racket hit dicts with {frame, x, y, slope}
            player_positions: Optional dict mapping player_id to list of (x, y) positions

        Returns:
            List of Shot objects with classifications
        """
        shots = []

        # Match consecutive racket hits (define shot boundaries)
        shot_segments = self._match_consecutive_racket_hits(racket_hits)

        for start_hit, end_hit in shot_segments:
            # Find wall hit within this shot's time window
            wall_hit = self._find_wall_hit_in_shot(
                wall_hits, start_hit["frame"], end_hit["frame"]
            )

            # Extract features (includes wall hit if detected)
            features = self._extract_shot_features(
                ball_positions, start_hit, end_hit, wall_hit
            )

            # Classify direction and depth
            direction = self._classify_direction(features)
            depth = self._classify_depth(features)

            # Combine into shot type
            shot_type = self._determine_shot_type(direction, depth, features)

            # Adjust confidence based on whether wall hit was detected
            confidence = 1.0 if wall_hit is not None else 0.7

            # Create Shot object
            shot = Shot(
                frame=start_hit["frame"],
                direction=direction,
                depth=depth,
                shot_type=shot_type,
                racket_hit_pos=(start_hit["x"], start_hit["y"]),
                next_racket_hit_pos=(end_hit["x"], end_hit["y"]),
                wall_hit_pos=features["wall_hit_pos"],
                wall_hit_frame=features.get("wall_hit_frame"),
                vector_angle_deg=features["vector_angle_deg"],
                rebound_distance=features["rebound_distance"],
                confidence=confidence,
            )

            shots.append(shot)

        return shots

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
        positions: List[Tuple[float, float]],
        start_hit: Dict,
        end_hit: Dict,
        wall_hit: Optional[Dict],
    ) -> Dict:
        """
        Extract vector-based features for a shot.

        Constructs two vectors:
        1. Racket → Wall (attack vector)
        2. Wall → Next Racket (rebound vector)

        Args:
            positions: Ball trajectory positions
            start_hit: Starting racket hit dict
            end_hit: Ending racket hit dict (next player's hit)
            wall_hit: Optional wall hit dict (front wall)

        Returns:
            Dictionary of extracted features including vectors and angle
        """
        start_pos = np.array([start_hit["x"], start_hit["y"]])
        end_pos = np.array([end_hit["x"], end_hit["y"]])

        # Initialize features
        features = {
            "has_wall_hit": False,
            "vector_angle_deg": None,
            "rebound_distance": None,
            "wall_hit_pos": None,
            "wall_hit_frame": None,
        }

        # Extract vector-based features if wall hit detected
        if wall_hit is not None:
            wall_pos = np.array([wall_hit["x"], wall_hit["y"]])

            # Vector 1: Racket → Wall (attack)
            vec_racket_to_wall = wall_pos - start_pos

            # Vector 2: Wall → Next Racket (rebound)
            vec_wall_to_next = end_pos - wall_pos

            # Calculate angle between the two vectors (in degrees)
            # Angle tells us the direction of the shot
            dot_product = np.dot(vec_racket_to_wall, vec_wall_to_next)
            mag1 = np.linalg.norm(vec_racket_to_wall)
            mag2 = np.linalg.norm(vec_wall_to_next)

            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                # Clamp to [-1, 1] to avoid numerical errors
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
            else:
                angle_deg = 0.0

            # Rebound distance (length of wall → next racket vector)
            rebound_distance = mag2

            features.update(
                {
                    "has_wall_hit": True,
                    "vector_angle_deg": float(angle_deg),
                    "rebound_distance": float(rebound_distance),
                    "wall_hit_pos": (wall_hit["x"], wall_hit["y"]),
                    "wall_hit_frame": wall_hit["frame"],
                }
            )

        return features

    def _classify_direction(self, features: Dict) -> ShotDirection:
        """
        Classify shot direction based on angle between attack and rebound vectors.

        Uses the angle between racket→wall and wall→next racket vectors:
        - Small angle → Straight shot (ball continues in same direction)
        - Large angle → Cross-court shot (ball changes direction significantly)
        - Medium angle → Down-the-line shot

        Args:
            features: Extracted shot features with vector_angle_deg

        Returns:
            ShotDirection enum value
        """
        # Require wall hit for accurate classification
        if not features["has_wall_hit"]:
            return ShotDirection.STRAIGHT  # Default fallback

        angle = features["vector_angle_deg"]

        # Angle thresholds
        straight_max = self.dir_thresholds.get("straight_max_angle_deg", 30)
        cross_min = self.dir_thresholds.get("cross_min_angle_deg", 120)

        if angle < straight_max:
            return ShotDirection.STRAIGHT
        elif angle > cross_min:
            return ShotDirection.CROSS_COURT
        else:
            return ShotDirection.DOWN_THE_LINE

    def _classify_depth(self, features: Dict) -> ShotDepth:
        """
        Classify shot depth based on rebound distance (wall→next racket vector length).

        Uses the length of the wall→next racket vector:
        - Short rebound distance → Drop shot (ball doesn't travel far after wall)
        - Long rebound distance → Long shot/Drive (ball travels deep into court)

        Args:
            features: Extracted shot features with rebound_distance

        Returns:
            ShotDepth enum value
        """
        # Require wall hit for accurate classification
        if not features["has_wall_hit"]:
            return ShotDepth.LONG  # Default fallback

        rebound_dist = features["rebound_distance"]

        # Distance threshold
        drop_max = self.depth_thresholds.get("drop_max_rebound_distance_px", 400)

        if rebound_dist < drop_max:
            return ShotDepth.DROP
        else:
            return ShotDepth.LONG

    def _determine_shot_type(
        self, direction: ShotDirection, depth: ShotDepth, features: Dict
    ) -> ShotType:
        """
        Combine direction and depth into final shot type.

        Args:
            direction: Classified direction
            depth: Classified depth
            features: Extracted features (for future refinement)

        Returns:
            ShotType enum value
        """
        # Mapping from (direction, depth) to ShotType
        mapping = {
            (ShotDirection.STRAIGHT, ShotDepth.LONG): ShotType.STRAIGHT_DRIVE,
            (ShotDirection.STRAIGHT, ShotDepth.DROP): ShotType.STRAIGHT_DROP,
            (ShotDirection.CROSS_COURT, ShotDepth.LONG): ShotType.CROSS_COURT_DRIVE,
            (ShotDirection.CROSS_COURT, ShotDepth.DROP): ShotType.CROSS_COURT_DROP,
            (ShotDirection.DOWN_THE_LINE, ShotDepth.LONG): ShotType.DOWN_LINE_DRIVE,
            (ShotDirection.DOWN_THE_LINE, ShotDepth.DROP): ShotType.DOWN_LINE_DROP,
        }

        return mapping.get((direction, depth), ShotType.STRAIGHT_DRIVE)

    def get_statistics(self, shots: List[Shot]) -> Dict:
        """
        Calculate statistics from classified shots.

        Args:
            shots: List of classified shots

        Returns:
            Dictionary with shot type distribution and other stats
        """
        if not shots:
            return {}

        stats = {
            "total_shots": len(shots),
            "by_type": {},
            "by_direction": {},
            "by_depth": {},
        }

        # Count by type
        for shot in shots:
            # By shot type
            type_name = shot.shot_type.value
            stats["by_type"][type_name] = stats["by_type"].get(type_name, 0) + 1

            # By direction
            dir_name = shot.direction.value
            stats["by_direction"][dir_name] = stats["by_direction"].get(dir_name, 0) + 1

            # By depth
            depth_name = shot.depth.value
            stats["by_depth"][depth_name] = stats["by_depth"].get(depth_name, 0) + 1

        # Wall hit statistics (if available)
        shots_with_wall = [s for s in shots if s.wall_hit_pos is not None]
        if shots_with_wall:
            stats["wall_hit_detection_rate"] = len(shots_with_wall) / len(shots)
            stats["average_vector_angle"] = np.mean(
                [
                    s.vector_angle_deg
                    for s in shots_with_wall
                    if s.vector_angle_deg is not None
                ]
            )
            stats["average_rebound_distance"] = np.mean(
                [
                    s.rebound_distance
                    for s in shots_with_wall
                    if s.rebound_distance is not None
                ]
            )

        return stats
