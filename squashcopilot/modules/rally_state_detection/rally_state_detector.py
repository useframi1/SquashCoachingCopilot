import pandas as pd
from squashcopilot.common.utils import load_config
from .metrics_aggregator import MetricsAggregator
from .model.ml_based_model import MLBasedModel

from squashcopilot.common import (
    RallyState,
    RallyStateInput,
    RallyStateResult,
    RallyStateSequence,
)


class RallyStateDetector:

    def __init__(self, config: dict = None):
        """
        Initialize rally state detector.

        Args:
            config: Configuration dictionary. If None, loads from config.json
        """
        self.config = (
            config if config else load_config(config_name="rally_state_detection")
        )

        # Initialize metrics aggregator with config for feature engineering
        self.metrics_aggregator = MetricsAggregator(
            window_size=self.config["window_size"], config=self.config
        )

        self.model = MLBasedModel(self.config)

        # For frame-by-frame processing
        self.current_state = "end"  # Start with end state
        self.model.set_state(self.current_state)

    def reset(self):
        """Reset internal state for new video."""
        self.current_state = "end"

        # Reinitialize the model
        self.model = MLBasedModel(self.config)

        self.model.set_state(self.current_state)
        self.metrics_aggregator.metrics_history.clear()

    def process_frames(self, input_data: RallyStateInput) -> RallyStateSequence:
        """
        Process a batch of frame metrics and return predictions.

        Args:
            input_data: RallyStateInput with metrics and configuration

        Returns:
            RallyStateSequence with structured state predictions
        """
        # Convert PlayerMetrics to dict format for processing
        metrics_dicts = [m.to_dict() for m in input_data.metrics]

        # Use metrics aggregator to process and engineer features
        df_features = self.metrics_aggregator.process_and_engineer(
            metrics_dicts, aggregated=input_data.aggregated
        )

        # Make predictions using the model's batch prediction
        df_predictions = self.model.predict_batch(df_features)

        # Convert DataFrame to RallyStateSequence
        results = []
        for _, row in df_predictions.iterrows():
            result = RallyStateResult(
                frame_number=int(row["frame_number"]),
                state=RallyState.from_string(row["predicted_state"]),
                confidence=1.0,  # Can be enhanced if confidence is available
                features=None,  # Can include engineered features if needed
            )
            results.append(result)

        return RallyStateSequence(results=results, postprocessed=False)

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
