from utilities.general import load_config
from models.tracknet.tracknet_tracker import TrackNetTracker
from models.rf.rf_tracker import RFTracker


class BallTracker:
    """Wrapper class for ball tracking that selects between different tracker implementations.

    This class acts as a facade that delegates to either TrackNetTracker or RFTracker
    based on the configuration. It maintains the same interface so the evaluator
    doesn't need to change.
    """

    def __init__(self, config: dict = None):
        """Initialize the ball tracker wrapper.

        Args:
            config: Configuration dictionary. If None, loads from config.json
        """
        # Load configuration
        if config is None:
            config = load_config("config.json")

        self.config = config

        # Determine which tracker to use
        tracker_type = self.config.get("tracker", {}).get("type", "tracknet")

        # Initialize the appropriate tracker
        if tracker_type == "tracknet":
            self.tracker = TrackNetTracker(config=config)
        elif tracker_type == "rf":
            self.tracker = RFTracker(config=config)
        else:
            raise ValueError(
                f"Unknown tracker type: {tracker_type}. Must be 'tracknet' or 'rf'"
            )

        # Expose the device attribute for compatibility with evaluator
        self.device = getattr(self.tracker, "device", "N/A")

    def reset(self):
        """Reset the tracker state."""
        self.tracker.reset()

    def process_frame(self, frame):
        """Process a single frame and return ball coordinates.

        Args:
            frame: Input frame (BGR format, any resolution)

        Returns:
            tuple: (x, y) coordinates of the ball, or (None, None) if not detected.
        """
        return self.tracker.process_frame(frame)
