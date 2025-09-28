"""
Rally State Segmentation Model
A rule-based model for segmenting squash rallies into start, active, and end states.
"""

import pandas as pd
from typing import Dict

from modeling.base_model import BaseModel
from config import CONFIG


class RuleBasedModel(BaseModel):
    """
    A rule-based model for segmenting squash rallies into start, active, and end states.

    Rules:
    - Start state: When starting position is met for at least state_window_frames
    - Active state: When players are in close range (below threshold)
    - End state: When distance stays above threshold for at least state_window_frames
    - Transition rules: start->active, active->end, end->start only
    """

    def __init__(self):
        """Initialize the rule-based model with configuration."""
        self.config = CONFIG["modeling"]["rule_based"]
        self.reset_state()

    def reset_state(self):
        """Reset the internal state tracking."""
        self.current_state = "end"  # Start with end state
        self.start_condition_frames = 0  # Consecutive frames meeting start conditions
        self.end_condition_frames = 0  # Consecutive frames meeting end conditions

    def _is_behind_service_line(self, player_y: float) -> bool:
        """Check if a player is behind the service line."""
        return player_y > self.config["service_line_y"]

    def _is_in_service_box(self, player_x: float, player_y: float, side: str) -> bool:
        """Check if a player is in their service box."""
        box = self.config["service_boxes"][side]
        return (
            box["x_min"] <= player_x <= box["x_max"]
            and box["y_min"] <= player_y <= box["y_max"]
        )

    def _are_on_opposite_sides(self, p1_x: float, p2_x: float) -> bool:
        """Check if players are on opposite sides of the court."""
        court_center_x = self.config["court_center_x"]
        return (p1_x < court_center_x) != (p2_x < court_center_x)

    def _check_start_position_constraints(self, metrics: Dict) -> bool:
        """
        Check if player positions satisfy start state constraints.

        Requirements:
        1. Both players are on opposite sides of the court
        2. Both players are behind the service line
        3. At least one player is in a service box
        """
        positions = [
            metrics.get("median_player1_x"),
            metrics.get("median_player1_y"),
            metrics.get("median_player2_x"),
            metrics.get("median_player2_y"),
        ]

        if any(pos is None for pos in positions):
            return False

        p1_x, p1_y, p2_x, p2_y = positions

        # Check opposite sides
        if not self._are_on_opposite_sides(p1_x, p2_x):
            return False

        # Check both behind service line
        if not (
            self._is_behind_service_line(p1_y) and self._is_behind_service_line(p2_y)
        ):
            return False

        # Check at least one in service box
        court_center_x = self.config["court_center_x"]
        p1_side = "left" if p1_x < court_center_x else "right"
        p2_side = "left" if p2_x < court_center_x else "right"

        return self._is_in_service_box(p1_x, p1_y, p1_side) or self._is_in_service_box(
            p2_x, p2_y, p2_side
        )

    def _get_distance_based_state(self, distance: float) -> str:
        """Get suggested state based purely on distance."""
        if distance <= self.config["active_state"]["distance_max"]:
            return "active"
        elif (
            self.config["start_state"]["distance_min"]
            <= distance
            <= self.config["start_state"]["distance_max"]
        ):
            return "start"
        elif distance >= self.config["end_state"]["distance_min"]:
            return "end"
        else:
            return "start"  # Default fallback

    def _is_valid_transition(self, from_state: str, to_state: str) -> bool:
        """Check if a state transition is valid according to transition rules."""
        valid_transitions = {
            "start": ["start", "active"],
            "active": ["active", "end"],
            "end": ["end", "start"],
        }
        return to_state in valid_transitions.get(from_state, [])

    def predict_next_state(self, metrics: Dict) -> str:
        """
        Predict the next state based on distance and position metrics.

        Args:
            metrics: Dictionary containing distance and position metrics

        Returns:
            Predicted state: 'start', 'active', or 'end'
        """
        distance = metrics["mean_distance"]

        # Check position constraints for start state
        start_position_ok = self._check_start_position_constraints(metrics)
        start_distance_ok = (
            self.config["start_state"]["distance_min"]
            <= distance
            <= self.config["start_state"]["distance_max"]
        )

        # Update start condition tracking
        if start_position_ok and start_distance_ok:
            self.start_condition_frames += 1
        else:
            self.start_condition_frames = 0

        # Check end condition (distance above threshold)
        if distance >= self.config["end_state"]["distance_min"]:
            self.end_condition_frames += 1
        else:
            self.end_condition_frames = 0

        # Determine suggested state based on rules
        suggested_state = None

        # Rule 1: Go to start state when starting position is met for required frames
        if self.start_condition_frames >= self.config["lookback_frames"]:
            suggested_state = "start"

        # Rule 2: Go to active state when players are in close range
        elif distance <= self.config["active_state"]["distance_max"]:
            suggested_state = "active"

        # Rule 3: Go to end state when distance stays above threshold for required frames
        elif self.end_condition_frames >= self.config["lookback_frames"]:
            suggested_state = "end"
        else:
            # Fallback to distance-based suggestion
            suggested_state = self._get_distance_based_state(distance)

        # Apply transition rules
        if suggested_state and self._is_valid_transition(
            self.current_state, suggested_state
        ):
            next_state = suggested_state
        else:
            next_state = self.current_state

        self.current_state = next_state
        return next_state

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict rally states for the entire dataset.

        Args:
            df: DataFrame with required columns (mean_distance, median_player1_x,
                median_player1_y, median_player2_x, median_player2_y)

        Returns:
            DataFrame with 'predicted_state' column added
        """
        df = df.copy()

        predictions = []
        for _, row in df.iterrows():
            pred = self.predict_next_state(row.to_dict())
            predictions.append(pred)

        df["predicted_state"] = predictions
        return df
