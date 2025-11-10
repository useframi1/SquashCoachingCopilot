import cv2
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

    def preprocess_frame(self, frame):
        """Preprocess the input frame if needed.

        Args:
            frame: Input frame (BGR format, any resolution)

        Returns:
            Preprocessed frame (BGR format).
        """
        frame = cv2.bitwise_not(frame)

        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE only for black ball
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)

        # Apply dilation in both cases
        l_dilated = cv2.dilate(
            l_channel,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

        # Merge and convert back to BGR for color-enhanced frame
        enhanced_lab = cv2.merge([l_dilated, a_channel, b_channel])
        enhanced_color = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Return the color-enhanced frame to the tracker
        return enhanced_color

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
        if self.config["tracker"].get("is_black_ball", False):
            frame = self.preprocess_frame(frame)

        return self.tracker.process_frame(frame)
