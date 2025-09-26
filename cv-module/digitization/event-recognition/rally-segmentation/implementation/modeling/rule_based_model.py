"""
Rally State Segmentation Model
A rule-based model for segmenting squash rallies into start, active, and end states.
Enhanced with distance deltas, position-based logic, and lagged features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any

from utilities.general import load_config


class RallyStateSegmenter:
    """
    A rule-based model for segmenting squash rallies into start, active, and end states.
    Uses distance, position, and distance delta features with temporal constraints.
    """

    def __init__(self):
        """
        Initialize the segmenter with configuration.
        """
        self.config = load_config()["rally_segmenter"]
        self.current_state = "end"
        self.state_duration = 0
        self.state_history = []
        self.distance_history = []

        # New tracking for end state logic
        self.threshold_crossed_frame = (
            None  # Frame when distance first crossed end threshold
        )
        self.post_threshold_distances = []  # Distances after crossing threshold

    def _smooth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rolling average to smooth distance measurements.

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with smoothed features added
        """
        df = df.copy()
        df["mean_distance"] = (
            df["mean_distance"]
            .rolling(window=self.config["rolling_window"], center=True, min_periods=1)
            .mean()
        )
        return df

    def _is_behind_service_line(self, player_y: float) -> bool:
        """Check if a player is behind the service line."""
        service_line_y = self.config["service_line_y"]
        return player_y > service_line_y

    def _is_in_service_box(self, player_x: float, player_y: float, side: str) -> bool:
        """Check if a player is in their service box."""
        service_box_config = self.config["service_boxes"]
        if side not in service_box_config:
            return False

        box = service_box_config[side]
        return (
            box["x_min"] <= player_x <= box["x_max"]
            and box["y_min"] <= player_y <= box["y_max"]
        )

    def _are_on_opposite_sides(self, p1_x: float, p2_x: float) -> bool:
        """Check if players are on opposite sides of the court."""
        court_center_x = self.config["court_center_x"]
        return (p1_x < court_center_x) != (p2_x < court_center_x)

    def _check_start_position_constraints(self, metrics: Dict[str, any]) -> bool:
        """
        Check if player positions satisfy start state constraints.

        For squash start state:
        1. Both players are on opposite sides of the court
        2. Both players are behind the service line
        3. At least one player is in a service box
        """
        # Extract player positions
        positions = [
            metrics.get("median_player1_x"),
            metrics.get("median_player1_y"),
            metrics.get("median_player2_x"),
            metrics.get("median_player2_y"),
        ]

        if any(pos is None for pos in positions):
            return False

        p1_x, p1_y, p2_x, p2_y = positions

        print(f"Player Positions - P1: ({p1_x}, {p1_y}), P2: ({p2_x}, {p2_y})")

        # Check all constraints
        if not self._are_on_opposite_sides(p1_x, p2_x):
            return False

        if not (
            self._is_behind_service_line(p1_y) and self._is_behind_service_line(p2_y)
        ):
            return False

        # Check if at least one player is in service box
        court_center_x = self.config["court_center_x"]
        p1_side = "left" if p1_x < court_center_x else "right"
        p2_side = "left" if p2_x < court_center_x else "right"

        return self._is_in_service_box(p1_x, p1_y, p1_side) or self._is_in_service_box(
            p2_x, p2_y, p2_side
        )

    def _calculate_distance_features(self, current_distance: float) -> Dict[str, float]:
        """
        Calculate distance-based lagged features.

        Args:
            current_distance: Current frame distance

        Returns:
            Dictionary with distance delta features
        """
        # Add current distance to history
        self.distance_history.append(current_distance)

        # Keep only the required number of frames
        lookback_frames = self.config["distance_lookback_frames"]
        if len(self.distance_history) > lookback_frames + 1:
            self.distance_history = self.distance_history[-(lookback_frames + 1) :]

        # Calculate features if we have enough history
        if len(self.distance_history) < 2:
            return {
                "distance_mean_change": 0.0,
                "distance_abs_mean_change": 0.0,
                "distance_std_change": 0.0,
                "sufficient_history": False,
            }

        # Calculate differences between consecutive frames
        distances = np.array(self.distance_history)
        changes = np.diff(distances)

        return {
            "distance_mean_change": np.mean(changes),
            "distance_abs_mean_change": np.mean(np.abs(changes)),
            "distance_std_change": np.std(changes),
            "sufficient_history": len(self.distance_history) >= lookback_frames,
        }

    def _check_end_state_conditions(
        self, distance: float, distance_features: Dict[str, float]
    ) -> bool:
        """
        Check if conditions for end state are met, using the new logic where
        distance must continue to increase for n frames AFTER crossing threshold.

        Args:
            distance: Current distance between players
            distance_features: Dictionary with distance delta features

        Returns:
            bool: True if end state conditions are met
        """
        end_config = self.config["end_state_criteria"]
        distance_min = end_config["distance_min"]

        # Use the existing distance_lookback_frames parameter
        consecutive_frames_required = self.config["distance_lookback_frames"]

        # Check if we've crossed the threshold for the first time
        if self.threshold_crossed_frame is None and distance >= distance_min:
            self.threshold_crossed_frame = (
                len(self.distance_history) - 1
            )  # Current frame index
            self.post_threshold_distances = [distance]
            return False  # Don't switch to end immediately

        # If we've crossed the threshold, track post-threshold distances
        if self.threshold_crossed_frame is not None:
            # Only add to post-threshold if we're still above threshold
            if distance >= distance_min:
                self.post_threshold_distances.append(distance)

                # Check if we have enough frames to analyze
                if len(self.post_threshold_distances) >= consecutive_frames_required:
                    # Check if distance has been consistently increasing
                    recent_distances = self.post_threshold_distances[
                        -consecutive_frames_required:
                    ]

                    # Check if each frame has higher distance than the previous
                    is_consistently_increasing = all(
                        recent_distances[i] > recent_distances[i - 1]
                        for i in range(1, len(recent_distances))
                    )

                    if is_consistently_increasing:
                        return True
            else:
                # Distance dropped below threshold, reset tracking
                self.threshold_crossed_frame = None
                self.post_threshold_distances = []

        return False

    def _get_distance_based_state_fallback(self, distance: float) -> str:
        """Get the state based purely on distance thresholds (fallback method)."""
        # Use the new state criteria for fallback
        start_config = self.config["start_state_criteria"]
        active_config = self.config["active_state_criteria"]
        end_config = self.config["end_state_criteria"]

        if active_config["distance_min"] <= distance <= active_config["distance_max"]:
            return "active"
        elif start_config["distance_min"] <= distance <= start_config["distance_max"]:
            return "start"
        elif distance >= end_config["distance_min"]:
            return "end"
        else:
            return "start"  # Default fallback to start

    def _classify_with_features(
        self,
        distance: float,
        distance_features: Dict[str, float],
        metrics: Dict[str, any],
    ) -> str:
        """
        Classify state using distance, position, and distance delta features.

        Args:
            distance: Current distance between players
            distance_features: Dictionary with distance delta features
            metrics: All available metrics including positions

        Returns:
            Predicted state
        """
        # Extract distance delta features
        mean_change = distance_features["distance_mean_change"]
        abs_mean_change = distance_features["distance_abs_mean_change"]
        std_change = distance_features["distance_std_change"]

        # Get configuration thresholds
        start_config = self.config["start_state_criteria"]
        active_config = self.config["active_state_criteria"]
        end_config = self.config["end_state_criteria"]

        # Rule-based classification with state-specific feature priorities

        # START STATE: Position constraints + low variance + appropriate distance
        start_distance_ok = (
            start_config["distance_min"] <= distance <= start_config["distance_max"]
        )
        start_position_ok = self._check_start_position_constraints(metrics)
        start_stability_ok = abs_mean_change <= start_config["max_abs_mean_change"]
        start_low_variance = std_change <= start_config["max_std_change"]

        if start_position_ok and start_low_variance:
            return "start"

        # ACTIVE STATE: Medium distance + high variance (dynamic movement)
        active_distance_ok = (
            active_config["distance_min"] <= distance <= active_config["distance_max"]
        )
        active_high_variance = std_change >= active_config["min_std_change"]
        active_dynamic_movement = (
            abs_mean_change >= active_config["min_abs_mean_change"]
        )

        if active_distance_ok and (active_high_variance or active_dynamic_movement):
            return "active"

        # END STATE: Use new logic that requires consecutive increasing frames after threshold
        if self._check_end_state_conditions(distance, distance_features):
            return "end"

        if distance <= active_config["distance_max"]:
            return "active"
        else:
            return self.current_state if self.current_state else "start"

    def _is_valid_transition(self, from_state: str, to_state: str) -> bool:
        """Check if a state transition is valid."""
        valid_transitions = {
            "start": ["start", "active"],
            "active": ["active", "end"],
            "end": ["end", "start"],
        }
        return to_state in valid_transitions.get(from_state, [])

    def _meets_min_duration(self, state: str) -> bool:
        """Check if current state has met minimum duration requirement."""
        min_durations = {
            "start": self.config["min_start_duration"],
            "active": self.config["min_active_duration"],
            "end": self.config["min_end_duration"],
        }
        return self.state_duration >= min_durations.get(state, 1)

    def predict_next_state(self, metrics: Dict[str, any]) -> str:
        """
        Unified function to predict the next state based on distance, position, and delta features.

        Args:
            metrics: Dictionary containing distance and position metrics

        Returns:
            Predicted state: 'start', 'active', or 'end'
        """
        distance = metrics["mean_distance"]

        # Calculate distance-based features
        distance_features = self._calculate_distance_features(distance)

        # Step 1: Check if we have sufficient history for advanced features
        if not distance_features["sufficient_history"]:
            # Fallback to distance-only classification, defaulting to start
            suggested_state = self.current_state
        else:
            # Step 2: Use full feature-based classification
            suggested_state = self._classify_with_features(
                distance, distance_features, metrics
            )
        print(
            f"Suggested state: {suggested_state}, Current state: {self.current_state}, Distance: {distance}"
        )

        # Step 3: Apply hysteresis for stability
        # if self.current_state is not None:
        #     hysteresis_margin = self.config["hysteresis_margin"]

        #     # Apply hysteresis based on distance proximity to boundaries
        #     if self.current_state == suggested_state:
        #         # Already in suggested state, apply stronger hysteresis
        #         pass  # Keep suggested state
        #     elif (
        #         abs(distance - self._get_state_distance_center(self.current_state))
        #         <= hysteresis_margin
        #     ):
        #         # Close to current state's typical distance, prefer staying
        #         suggested_state = self.current_state

        # Step 4: Check transition validity and duration constraints
        if self.current_state is None:
            # First frame - accept any state
            next_state = suggested_state
        elif suggested_state == self.current_state:
            # Staying in same state - always valid
            next_state = suggested_state
        elif self._is_valid_transition(self.current_state, suggested_state):
            # Valid transition - check if we've met minimum duration
            if self._meets_min_duration(self.current_state):
                next_state = suggested_state
            else:
                # Haven't met minimum duration - stay in current state
                next_state = self.current_state
        else:
            # Invalid transition - stay in current state
            next_state = self.current_state

        # Step 5: Update internal state tracking
        if next_state == self.current_state:
            self.state_duration += 1
        else:
            self.state_duration = 1
            # Reset end state tracking when changing states
            if self.current_state != "end" or next_state != "end":
                self.threshold_crossed_frame = None
                self.post_threshold_distances = []

        self.current_state = next_state
        self.state_history.append(next_state)

        return next_state

    def _get_state_distance_center(self, state: str) -> float:
        """Get the typical center distance for a given state (for hysteresis)."""
        if state == "start":
            start_config = self.config["start_state_criteria"]
            return (start_config["distance_min"] + start_config["distance_max"]) / 2
        elif state == "active":
            active_config = self.config["active_state_criteria"]
            return (active_config["distance_min"] + active_config["distance_max"]) / 2
        elif state == "end":
            return (
                self.config["end_state_criteria"]["distance_min"] + 1.0
            )  # Assume typical end distance
        return 0.0

    def reset_state(self):
        """Reset the internal state tracking."""
        self.current_state = "end"
        self.state_duration = 0
        self.state_history = []
        self.distance_history = []
        # Reset end state tracking
        self.threshold_crossed_frame = None
        self.post_threshold_distances = []

    def predict(
        self,
        df: pd.DataFrame,
        apply_smoothing: bool = True,
    ) -> pd.DataFrame:
        """
        Predict rally states for the entire dataset.

        Args:
            df: DataFrame with 'mean_distance' and player position columns
            apply_smoothing: Whether to apply feature smoothing

        Returns:
            DataFrame with predictions added
        """
        df = df.copy()

        # Reset state tracking
        self.reset_state()

        # Step 1: Smooth features if requested
        if apply_smoothing:
            df = self._smooth_features(df)

        # Step 2: Predict states frame by frame
        predictions = []
        for _, row in df.iterrows():
            pred = self.predict_next_state(row.to_dict())
            if 0 <= _ <= 1000:
                print(row.to_dict())
            predictions.append(pred)

        df["predicted_state"] = predictions
        return df
