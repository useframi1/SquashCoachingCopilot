import pandas as pd
from typing import List, Dict
from rally_state_detection.utilities.general import load_config
from rally_state_detection.utilities.metrics_aggregator import MetricsAggregator
from rally_state_detection.models.ml.ml_based_model import MLBasedModel
from rally_state_detection.models.rule.rule_based_model import RuleBasedModel


class RallyStateDetector:

    def __init__(self, config: dict = None):
        """
        Initialize rally state detector.

        Args:
            config: Configuration dictionary. If None, loads from config.json
        """
        self.config = config if config else load_config()

        # Initialize metrics aggregator with config for feature engineering
        self.metrics_aggregator = MetricsAggregator(
            window_size=self.config["window_size"],
            config=self.config
        )

        # Initialize the model based on config
        active_model_type = self.config["active_model"]

        if active_model_type == "ml_based":
            self.model = MLBasedModel(self.config)
        elif active_model_type == "rule_based":
            self.model = RuleBasedModel(self.config)
        else:
            raise ValueError(f"Unknown model type: {active_model_type}")

        # For frame-by-frame processing
        self.current_state = "end"  # Start with end state
        self.model.set_state(self.current_state)

    def reset(self):
        """Reset internal state for new video."""
        self.current_state = "end"

        # Reinitialize the model
        active_model_type = self.config["active_model"]
        if active_model_type == "ml_based":
            self.model = MLBasedModel(self.config)
        elif active_model_type == "rule_based":
            self.model = RuleBasedModel(self.config)

        self.model.set_state(self.current_state)
        self.metrics_aggregator.metrics_history.clear()

    def process_frames(self, metrics_list: List[Dict], aggregated: bool = False) -> pd.DataFrame:
        """
        Process a batch of frame metrics and return predictions.

        Args:
            metrics_list: List of dictionaries containing frame metrics.
                If aggregated=False (default):
                    - frame_number: Frame index
                    - player_distance: Distance between players
                    - player1_x, player1_y: Player 1 coordinates
                    - player2_x, player2_y: Player 2 coordinates
                If aggregated=True:
                    - frame_number: Frame index
                    - mean_distance: Mean distance (already aggregated)
                    - median_player1_x, median_player1_y: Player 1 median positions
                    - median_player2_x, median_player2_y: Player 2 median positions
            aggregated: Whether the metrics are already aggregated by window size.
                       If False, will aggregate frame-level metrics by window_size.
                       If True, assumes metrics are already aggregated.

        Returns:
            DataFrame with base metrics, engineered features, and predictions
        """
        # Use metrics aggregator to process and engineer features
        df_features = self.metrics_aggregator.process_and_engineer(
            metrics_list, aggregated=aggregated
        )

        # Make predictions using the model's batch prediction
        df_predictions = self.model.predict_batch(df_features)

        return df_predictions

    def postprocess(self, predictions: pd.Series) -> pd.Series:
        """
        Apply postprocessing to predictions to enforce minimum duration and state transitions.

        Args:
            predictions: Series of predicted states

        Returns:
            Series of postprocessed predictions
        """
        min_duration = self.config["postprocessing"]["min_duration"]

        # Valid state transitions (Finite State Machine)
        valid_transitions = {
            "start": ["start", "active"],
            "active": ["active", "end"],
            "end": ["end", "start"],
        }

        postprocessed = predictions.copy()

        # Always start with "start" state
        current_committed_state = "start"
        postprocessed.iloc[0] = current_committed_state

        i = 1
        while i < len(predictions):
            candidate_state = predictions.iloc[i]

            # Check if this is a valid transition
            if candidate_state not in valid_transitions[current_committed_state]:
                # Invalid transition - replace with current committed state
                postprocessed.iloc[i] = current_committed_state
                i += 1
                continue

            # If same state as current, just continue
            if candidate_state == current_committed_state:
                postprocessed.iloc[i] = current_committed_state
                i += 1
                continue

            # Valid new state - check if it persists long enough
            # Count consecutive occurrences of this candidate state
            count = 1
            j = i + 1
            while j < len(predictions) and predictions.iloc[j] == candidate_state:
                count += 1
                j += 1

            # Check if candidate state meets minimum duration requirement
            if count >= min_duration.get(candidate_state, 3):
                # Real transition - commit it
                for k in range(i, j):
                    postprocessed.iloc[k] = candidate_state
                current_committed_state = candidate_state
                i = j
            else:
                # Fluctuation - reject it and replace with current committed state
                for k in range(i, j):
                    postprocessed.iloc[k] = current_committed_state
                i = j

        return postprocessed
