"""
Rally State Segmentation Model
A rule-based model for segmenting squash rallies into start, active, and end states.
"""

import pandas as pd
from typing import List, Optional

from utilities.general import load_config


class RallyStateSegmenter:
    """
    A rule-based model for segmenting squash rallies into start, active, and end states.
    Uses distance-based thresholds with temporal constraints.
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
        df["distance_smoothed"] = (
            df["mean_distance"]
            .rolling(window=self.config["rolling_window"], center=True, min_periods=1)
            .mean()
        )

        return df

    def _classify_frame_state(
        self, distance: float, current_state: Optional[str] = None
    ) -> str:
        """
        Classify a single frame based on distance ranges with hysteresis.

        Logic:
        - Active: low distance (players close, rally happening)
        - Start: medium-low distance (serving position)
        - End: high distance (rally over, players separated)

        Args:
            distance: Mean distance value for the frame
            current_state: Current state (for hysteresis logic)

        Returns:
            Predicted state: 'start', 'active', or 'end'
        """
        # Unpack ranges
        active_min, active_max = tuple(self.config["distance_active_range"])
        start_min, start_max = tuple(self.config["distance_start_range"])
        end_min, end_max = tuple(self.config["distance_end_range"])

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
                return "start"
        elif current_state == "end":
            if distance > end_min - self.config["hysteresis_margin"]:
                return "end"

        # Base classification without hysteresis
        if active_min <= distance < active_max:
            return "active"
        elif start_min <= distance < start_max:
            return "start"
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
            df: DataFrame with 'mean_distance' column
            apply_smoothing: Whether to apply feature smoothing
            apply_transitions: Whether to enforce transition rules

        Returns:
            DataFrame with predictions added
        """
        df = df.copy()

        # Step 1: Smooth features if requested
        if apply_smoothing:
            df = self._smooth_features(df)
            distance_col = "distance_smoothed"
        else:
            distance_col = "mean_distance"

        # Step 2: Make initial predictions
        predictions = []
        current_state = None

        for distance in df[distance_col]:
            pred = self._classify_frame_state(distance, current_state)
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
