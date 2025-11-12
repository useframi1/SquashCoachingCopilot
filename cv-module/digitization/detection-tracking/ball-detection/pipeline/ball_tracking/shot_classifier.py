"""
Shot classification module for squash ball tracking.

Classifies shots based on trajectory analysis in pixel coordinates,
combining direction, depth, and speed to determine shot types.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from .shot_types import Shot, ShotType, ShotDirection, ShotDepth
from .shot_features import ShotFeatureExtractor
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
            fps: Video frame rate (used for time-based features)
        """
        self.config = config if config else load_config()
        self.fps = fps
        self.feature_extractor = ShotFeatureExtractor(self.config)

        # Load thresholds from config (all in pixels)
        shot_config = self.config.get("shot_classification", {})
        self.dir_thresholds = shot_config.get("direction_thresholds", {})
        self.depth_thresholds = shot_config.get("depth_thresholds", {})
        self.speed_thresholds = shot_config.get("speed_thresholds", {})
        self.player_config = shot_config.get("player_attribution", {})

    def classify(
        self,
        ball_positions: List[Tuple[float, float]],
        wall_hits: List[Dict],
        racket_hits: List[Dict],
        player_positions: Optional[Dict[int, List[Tuple[float, float]]]] = None,
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

        # Match racket hits to wall hits
        shot_segments = self._match_hits(racket_hits, wall_hits)

        for racket_hit, wall_hit in shot_segments:
            # Extract features (all pixel-based)
            features = self._extract_shot_features(
                ball_positions, racket_hit, wall_hit, player_positions
            )

            # Classify direction
            direction = self._classify_direction(features)

            # Classify depth
            depth = self._classify_depth(features)

            # Combine into shot type
            shot_type = self._determine_shot_type(direction, depth, features)

            # Create Shot object
            shot = Shot(
                frame=racket_hit["frame"],
                player_id=features.get("player_id"),
                direction=direction,
                depth=depth,
                shot_type=shot_type,
                racket_hit_pos=(racket_hit["x"], racket_hit["y"]),
                wall_hit_pos=(wall_hit["x"], wall_hit["y"]),
                lateral_displacement=features["lateral_displacement"],
                distance_to_wall=features["distance_to_wall"],
                velocity_px_per_frame=features["velocity_px_per_frame"],
                time_to_wall_frames=features["time_to_wall_frames"],
                confidence=1.0,  # Rule-based has fixed confidence
            )

            shots.append(shot)

        return shots

    def _match_hits(
        self, racket_hits: List[Dict], wall_hits: List[Dict]
    ) -> List[Tuple[Dict, Dict]]:
        """
        Match each racket hit to its corresponding wall hit.

        Args:
            racket_hits: List of racket hit dicts
            wall_hits: List of wall hit dicts

        Returns:
            List of (racket_hit, wall_hit) pairs
        """
        matched_pairs = []

        for racket_hit in racket_hits:
            racket_frame = racket_hit["frame"]

            # Find the next wall hit after this racket hit
            next_wall_hit = None
            min_frame_diff = float("inf")

            for wall_hit in wall_hits:
                wall_frame = wall_hit["frame"]

                # Wall hit must be after racket hit
                if wall_frame > racket_frame:
                    frame_diff = wall_frame - racket_frame

                    # Take the nearest wall hit
                    if frame_diff < min_frame_diff:
                        min_frame_diff = frame_diff
                        next_wall_hit = wall_hit

            # Only create pair if we found a matching wall hit
            if next_wall_hit is not None:
                matched_pairs.append((racket_hit, next_wall_hit))

        return matched_pairs

    def _extract_shot_features(
        self,
        positions: List[Tuple[float, float]],
        racket_hit: Dict,
        wall_hit: Dict,
        player_positions: Optional[Dict[int, List[Tuple[float, float]]]],
    ) -> Dict:
        """
        Extract all features for a shot in pixel space.

        Args:
            positions: Ball trajectory positions
            racket_hit: Racket hit dict
            wall_hit: Wall hit dict
            player_positions: Optional player position data

        Returns:
            Dictionary of extracted features
        """
        racket_frame = racket_hit["frame"]
        wall_frame = wall_hit["frame"]

        racket_pos = (racket_hit["x"], racket_hit["y"])
        wall_pos = (wall_hit["x"], wall_hit["y"])

        # Lateral displacement
        lateral_disp = self.feature_extractor.get_lateral_displacement(
            racket_pos, wall_pos
        )

        # Distance to wall
        distance = self.feature_extractor.get_distance(racket_pos, wall_pos)

        # Time to wall (in frames)
        time_frames = wall_frame - racket_frame

        # Velocity (average over the shot segment)
        velocity = self.feature_extractor.calculate_average_velocity_segment(
            positions, racket_frame, wall_frame
        )

        # Player attribution
        player_id = None
        if player_positions:
            max_dist = self.player_config.get("max_distance_px", 120)
            player_id = self.feature_extractor.get_player_at_hit(
                racket_pos, player_positions, racket_frame, max_distance_px=max_dist
            )

        return {
            "lateral_displacement": lateral_disp,
            "distance_to_wall": distance,
            "time_to_wall_frames": time_frames,
            "velocity_px_per_frame": velocity,
            "player_id": player_id,
        }

    def _classify_direction(self, features: Dict) -> ShotDirection:
        """
        Classify shot direction based on lateral displacement.

        Args:
            features: Extracted shot features

        Returns:
            ShotDirection enum value
        """
        dx = abs(features["lateral_displacement"])  # Absolute lateral movement

        straight_max = self.dir_thresholds.get("straight_max_lateral_px", 80)
        cross_min = self.dir_thresholds.get("cross_court_min_lateral_px", 300)

        if dx < straight_max:
            return ShotDirection.STRAIGHT
        elif dx > cross_min:
            return ShotDirection.CROSS_COURT
        else:
            return ShotDirection.DOWN_THE_LINE

    def _classify_depth(self, features: Dict) -> ShotDepth:
        """
        Classify shot depth based on time and velocity.

        Args:
            features: Extracted shot features

        Returns:
            ShotDepth enum value
        """
        time_frames = features["time_to_wall_frames"]
        velocity = features["velocity_px_per_frame"]

        drop_max_time = self.depth_thresholds.get("drop_max_time_frames", 15)
        drop_max_vel = self.depth_thresholds.get("drop_max_velocity_px_per_frame", 12.0)

        # Drop shots are characterized by:
        # - Short time to wall OR
        # - Slow velocity
        is_drop = time_frames < drop_max_time or velocity < drop_max_vel

        return ShotDepth.DROP if is_drop else ShotDepth.LONG

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
            "by_player": {},
            "average_velocity": 0.0,
            "average_time_to_wall": 0.0,
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

            # By player
            if shot.player_id:
                stats["by_player"][shot.player_id] = (
                    stats["by_player"].get(shot.player_id, 0) + 1
                )

        # Average metrics
        stats["average_velocity"] = np.mean(
            [s.velocity_px_per_frame for s in shots]
        )
        stats["average_time_to_wall"] = np.mean(
            [s.time_to_wall_frames for s in shots]
        )

        return stats
