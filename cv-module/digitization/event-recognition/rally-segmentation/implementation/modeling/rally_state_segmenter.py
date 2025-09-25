"""
Rally State Segmentation Model
A rule-based model for segmenting squash rallies into start, active, and end states.
Enhanced with position-based logic for start state detection.
"""

import pandas as pd
from typing import List, Optional, Dict, Any

from utilities.general import load_config


class RallyStateSegmenter:
    """
    A rule-based model for segmenting squash rallies into start, active, and end states.
    Uses distance-based thresholds with temporal constraints and position-based logic.
    """

    def __init__(self):
        """
        Initialize the segmenter with configuration.

        Args:
            config: ModelConfig object with threshold and constraint parameters
        """
        self.config = load_config()["rally_segmenter"]
        self.state_history = []

    def _smooth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rolling average to smooth distance measurements.

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with smoothed features added
        """
        df = df.copy()

        # Apply rolling mean to distance
        df["mean_distance"] = (
            df["mean_distance"]
            .rolling(window=self.config["rolling_window"], center=True, min_periods=1)
            .mean()
        )

        return df

    def _is_behind_service_line(self, player_y: float) -> bool:
        """
        Check if a player is behind the service line.

        Args:
            player_y: Player's y-coordinate

        Returns:
            True if player is behind service line
        """
        # Assuming service line is at a specific y-coordinate
        # You'll need to adjust this based on your court coordinate system
        service_line_y = self.config["service_line_y"]
        return player_y > service_line_y

    def _is_in_service_box(self, player_x: float, player_y: float, side: str) -> bool:
        """
        Check if a player is in their service box.

        Args:
            player_x: Player's x-coordinate
            player_y: Player's y-coordinate
            side: 'left' or 'right' side of the court

        Returns:
            True if player is in service box
        """
        # Service box boundaries - adjust based on your coordinate system
        service_box_config = self.config["service_boxes"]

        if side not in service_box_config:
            return False

        box = service_box_config[side]
        return (
            box["x_min"] <= player_x <= box["x_max"]
            and box["y_min"] <= player_y <= box["y_max"]
        )

    def _are_on_opposite_sides(self, p1_x: float, p2_x: float) -> bool:
        """
        Check if players are on opposite sides of the court.

        Args:
            p1_x: Player 1's x-coordinate
            p2_x: Player 2's x-coordinate

        Returns:
            True if players are on opposite sides
        """
        court_center_x = self.config["court_center_x"]
        return (p1_x < court_center_x) != (p2_x < court_center_x)

    def _check_start_position_constraints(self, metrics: Dict[str, any]) -> bool:
        """
        Check if player positions satisfy start state constraints.

        For squash start state:
        1. Both players are on opposite sides of the court
        2. Both players are behind the service line
        3. At least one player is in a service box

        Args:
            metrics: Dictionary containing player position metrics

        Returns:
            True if position constraints are satisfied
        """
        # Extract player positions
        p1_x = metrics.get("median_player1_x")
        p1_y = metrics.get("median_player1_y")
        p2_x = metrics.get("median_player2_x")
        p2_y = metrics.get("median_player2_y")

        # Check if we have valid position data
        if any(pos is None for pos in [p1_x, p1_y, p2_x, p2_y]):
            return False

        # 1. Check if players are on opposite sides
        if not self._are_on_opposite_sides(p1_x, p2_x):
            return False

        # 2. Check if both players are behind service line
        if not (
            self._is_behind_service_line(p1_y) and self._is_behind_service_line(p2_y)
        ):
            return False

        # 3. Check if at least one player is in a service box
        # Determine which side each player is on
        # court_center_x = self.config["court_center_x"]
        # p1_side = "left" if p1_x < court_center_x else "right"
        # p2_side = "left" if p2_x < court_center_x else "right"

        # p1_in_box = self._is_in_service_box(p1_x, p1_y, p1_side)
        # p2_in_box = self._is_in_service_box(p2_x, p2_y, p2_side)

        # return p1_in_box or p2_in_box
        return True  # Relaxed for simplicity; implement as needed

    def _classify_frame_state(
        self, metrics: Dict[str, any], current_state: Optional[str] = None
    ) -> str:
        """
        Classify a single frame based on distance ranges and position constraints with hysteresis.

        Logic:
        - Active: low distance (players close, rally happening)
        - Start: medium-low distance + position constraints (serving position)
        - End: high distance (rally over, players separated)

        Args:
            metrics: Dictionary of all metrics
            current_state: Current state (for hysteresis logic)

        Returns:
            Predicted state: 'start', 'active', or 'end'
        """
        # Unpack ranges
        active_min, active_max = tuple(self.config["distance_active_range"])
        start_min, start_max = tuple(self.config["distance_start_range"])
        end_min, end_max = tuple(self.config["distance_end_range"])

        distance = metrics["mean_distance"]

        # Apply hysteresis to prevent flickering
        if current_state == "active":
            if distance < active_max + self.config["hysteresis_margin"]:
                return "active"
        elif current_state == "start":
            if (
                start_min - self.config["hysteresis_margin"]
                < distance
                < start_max + self.config["hysteresis_margin"]
            ):
                # For start state, also check position constraints
                if self._check_start_position_constraints(metrics):
                    return "start"
        elif current_state == "end":
            if distance > end_min - self.config["hysteresis_margin"]:
                return "end"

        # Base classification without hysteresis
        if active_min <= distance < active_max:
            return "active"
        elif start_min <= distance < start_max:
            # For start state, check both distance AND position constraints
            if self._check_start_position_constraints(metrics):
                return "start"
            else:
                # If distance suggests start but positions don't match,
                # it might be transitioning to/from active
                return "active" if current_state != "end" else current_state or "active"
        elif distance >= end_min:
            return "end"
        else:
            # Handle overlapping or gap regions
            if current_state:
                return current_state
            else:
                return "start"

    def _is_valid_transition(self, from_state: str, to_state: str) -> bool:
        """
        Check if a state transition is valid.

        Args:
            from_state: Current state
            to_state: Proposed next state

        Returns:
            True if transition is valid
        """
        valid_transitions = {
            "start": ["start", "active"],
            "active": ["active", "end"],
            "end": ["end", "start"],
        }

        return to_state in valid_transitions.get(from_state, [])

    def _check_min_duration(self, state: str, duration: int) -> bool:
        """
        Check if state has met minimum duration requirement.

        Args:
            state: State to check
            duration: Current duration in frames

        Returns:
            True if minimum duration is met
        """
        min_durations = {
            "start": self.config["min_start_duration"],
            "active": self.config["min_active_duration"],
            "end": self.config["min_end_duration"],
        }

        return duration >= min_durations.get(state, 1)

    def _enforce_transition_rules(self, states: List[str]) -> List[str]:
        """
        Enforce valid state transitions and minimum duration constraints.

        Valid transitions:
        - start -> active
        - active -> end
        - active -> active
        - end -> start (new rally)

        Args:
            states: List of raw state predictions

        Returns:
            List of corrected states
        """
        if len(states) == 0:
            return states

        corrected_states = [states[0]]
        state_duration = 1

        for i in range(1, len(states)):
            current = states[i]
            previous = corrected_states[-1]

            # Check if transition is valid
            valid_transition = self._is_valid_transition(previous, current)

            # Check minimum duration
            meets_min_duration = self._check_min_duration(previous, state_duration)

            if valid_transition and meets_min_duration:
                corrected_states.append(current)
                state_duration = 1
            else:
                # Stay in previous state
                corrected_states.append(previous)
                state_duration += 1

        return corrected_states

    def predict(
        self,
        df: pd.DataFrame,
        apply_smoothing: bool = True,
        apply_transitions: bool = True,
    ) -> pd.DataFrame:
        """
        Predict rally states for the entire dataset.

        Args:
            df: DataFrame with 'mean_distance' and player position columns
            apply_smoothing: Whether to apply feature smoothing
            apply_transitions: Whether to enforce transition rules

        Returns:
            DataFrame with predictions added
        """
        df = df.copy()

        # Step 1: Smooth features if requested
        if apply_smoothing:
            df = self._smooth_features(df)

        # Step 2: Make initial predictions
        predictions = []
        current_state = None

        for _, row in df.iterrows():
            pred = self._classify_frame_state(row.to_dict(), current_state)
            predictions.append(pred)
            current_state = pred

        df["predicted_state_raw"] = predictions

        # Step 3: Enforce transition rules
        if apply_transitions:
            corrected_predictions = self._enforce_transition_rules(predictions)
            df["predicted_state"] = corrected_predictions
        else:
            df["predicted_state"] = predictions

        return df
