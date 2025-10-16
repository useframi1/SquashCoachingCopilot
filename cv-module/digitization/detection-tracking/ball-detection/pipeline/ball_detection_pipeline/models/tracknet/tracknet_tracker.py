import torch
import cv2
import numpy as np
from collections import deque
from ball_detection_pipeline.models.tracknet.model import BallTrackerNet
from ball_detection_pipeline.utils import postprocess


class TrackNetTracker:
    """Real-time ball tracker using TrackNet model.

    This class maintains a buffer of the last 3 frames and uses the TrackNet model
    to detect the ball position in each new frame.
    """

    def __init__(self, config: dict):
        """Initialize the ball tracker.

        Args:
            config_path: Path to configuration JSON file
        """
        # Load configuration
        self.config = config

        self.config = config["tracknet_model"]

        # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load model
        model_path = self.config["model_path"]
        self.model = BallTrackerNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Model input dimensions
        self.model_width = self.config.get("model_width", 640)
        self.model_height = self.config.get("model_height", 360)

        # Frame buffer (need 3 consecutive frames for inference)
        self.frame_buffer = deque(maxlen=3)

        # Track number of frames processed
        self.frame_count = 0

    def reset(self):
        """Reset the tracker state (clears frame buffer)."""
        self.frame_buffer.clear()
        self.frame_count = 0

    def _get_scaled_coordinates(self, x, y, original_width, original_height):
        """Scale coordinates from model space to original frame resolution.

        Args:
            x: X coordinate from process_frame
            y: Y coordinate from process_frame
            original_width: Width of the original frame
            original_height: Height of the original frame

        Returns:
            tuple: (x_scaled, y_scaled) in original frame coordinates, or (None, None)
        """
        if x is None or y is None:
            return None, None

        # Account for postprocess scale factor (2x)
        postprocess_scale = 2

        # Calculate scaling factors
        scale_x = (original_width / self.model_width) / postprocess_scale
        scale_y = (original_height / self.model_height) / postprocess_scale

        x_scaled = int(x * scale_x)
        y_scaled = int(y * scale_y)

        # Ensure coordinates are within bounds
        x_scaled = max(0, min(x_scaled, original_width - 1))
        y_scaled = max(0, min(y_scaled, original_height - 1))

        return x_scaled, y_scaled

    def process_frame(self, frame):
        """Process a single frame and return ball coordinates.

        Args:
            frame: Input frame (BGR format, any resolution)

        Returns:
            tuple: (x, y) coordinates of the ball, or (None, None) if not detected.
                   Coordinates are in the coordinate system of the model (640x360 scaled by 2).
        """
        # Resize frame to model dimensions
        frame_resized = cv2.resize(frame, (self.model_width, self.model_height))

        # Add frame to buffer
        self.frame_buffer.append(frame_resized)
        self.frame_count += 1

        # Need at least 3 frames for inference
        if len(self.frame_buffer) < 3:
            return None, None

        # Prepare input: concatenate 3 frames along channel dimension
        # Order: current, previous, pre-previous
        frames_list = list(self.frame_buffer)
        img = frames_list[2]  # current
        img_prev = frames_list[1]  # previous
        img_preprev = frames_list[0]  # pre-previous

        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        # Run inference
        with torch.no_grad():
            out = self.model(torch.from_numpy(inp).float().to(self.device))
            output = out.argmax(dim=1).detach().cpu().numpy()

        # Postprocess to get (x, y) coordinates
        x_pred, y_pred = postprocess(output)

        x, y = self._get_scaled_coordinates(
            x_pred, y_pred, frame.shape[1], frame.shape[0]
        )

        return x, y
