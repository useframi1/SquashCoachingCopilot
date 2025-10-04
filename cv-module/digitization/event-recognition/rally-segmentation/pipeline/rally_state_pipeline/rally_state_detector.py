from rally_state_pipeline.utilities.general import load_config
from rally_state_pipeline.models.predictor import StatePredictor
from rally_state_pipeline.utilities.metrics_aggregator import MetricsAggregator


class RallyStateDetector:

    def __init__(self, config: dict = None):
        """
        Initialize video inference pipeline.

        Args:
            video_path: Path to video file
            model_path: Path to trained model
        """
        self.config = config if config else load_config()
        self.predictor = StatePredictor(self.config)
        self.current_state = "end"  # Start with end state
        self.predictor.set_state(self.current_state)

        self.metrics_aggregator = MetricsAggregator(
            window_size=self.config["window_size"]
        )

    def reset(self):
        """Reset internal state for new video."""
        self.current_state = "end"
        self.predictor = StatePredictor(self.config)
        self.predictor.set_state(self.current_state)
        self.metrics_aggregator.metrics_history.clear()

    def process_frame(self, player_real_coords):
        if player_real_coords[1] is None or player_real_coords[2] is None:
            return self.current_state  # Skip if no player detected

        # Update metrics aggregator
        self.metrics_aggregator.update_metrics(player_real_coords)

        # Check if we have a full window for prediction
        if self.metrics_aggregator.has_full_window():
            # Get aggregated base metrics
            base_metrics = self.metrics_aggregator.get_aggregated_metrics()

            if base_metrics and base_metrics["mean_distance"] is not None:
                # Make prediction
                prediction = self.predictor.predict(base_metrics)
                self.current_state = prediction
                self.predictor.set_state(self.current_state)

        return self.current_state
