import torch
import cv2
import numpy as np
from collections import deque
from typing import List, Optional, Tuple
from .model import BallTrackerNet
from ..utils import postprocess
from squashcopilot.common.utils import get_package_dir


class TrackNetTracker:
    """Real-time ball tracker using TrackNet model.

    This class maintains a buffer of the last 3 frames and uses the TrackNet model
    to detect the ball position in each new frame.
    """

    def __init__(self, config: dict):
        """Initialize the ball tracker.

        Args:
            config: Configuration dictionary
        """
        # Load model configuration
        model_config = config.get("model", {})

        # Setup device
        device_config = model_config.get("device", "auto")
        if device_config == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_config
        self.device = torch.device(device)

        # Load model
        model_path = model_config["model_path"]
        # Get parent directory (ball_tracking module root)
        parent_dir = get_package_dir(__file__).replace("/model", "")
        model_path = parent_dir + "/" + model_path
        self.model = BallTrackerNet()
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=False)
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Model input dimensions
        self.model_width = model_config.get("model_width", 640)
        self.model_height = model_config.get("model_height", 360)

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

    def process_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 32,
    ) -> List[Tuple[Optional[int], Optional[int]]]:
        """
        Process a batch of frames and return ball coordinates for each.

        Builds 3-frame sliding windows from the input frames and processes
        them through TrackNet in batches for GPU efficiency.

        Args:
            frames: List of input frames (BGR format, any resolution).
                    Must be at least 3 frames for any detections.
            batch_size: Number of windows to process in parallel on GPU.

        Returns:
            List of (x, y) tuples for each input frame.
            First 2 frames return (None, None) due to temporal context requirement.
        """
        num_frames = len(frames)
        results: List[Tuple[Optional[int], Optional[int]]] = []

        if num_frames == 0:
            return results

        # Get original frame dimensions from first frame
        original_height, original_width = frames[0].shape[:2]

        # Resize all frames to model dimensions
        resized_frames = [
            cv2.resize(frame, (self.model_width, self.model_height))
            for frame in frames
        ]

        # First 2 frames have no temporal context - return None
        results.append((None, None))
        if num_frames >= 2:
            results.append((None, None))

        if num_frames < 3:
            return results

        # Build all 3-frame sliding windows
        # Window i contains frames [i, i+1, i+2] and produces output for frame i+2
        num_windows = num_frames - 2
        windows = []

        for i in range(num_windows):
            img_preprev = resized_frames[i]      # t-2
            img_prev = resized_frames[i + 1]     # t-1
            img_curr = resized_frames[i + 2]     # t (current)

            # Concatenate: current, previous, pre-previous (9 channels)
            window = np.concatenate((img_curr, img_prev, img_preprev), axis=2)
            window = window.astype(np.float32) / 255.0
            window = np.rollaxis(window, 2, 0)  # HWC -> CHW
            windows.append(window)

        # Process windows in batches
        all_outputs = []
        with torch.no_grad():
            for batch_start in range(0, num_windows, batch_size):
                batch_end = min(batch_start + batch_size, num_windows)
                batch_windows = windows[batch_start:batch_end]

                # Stack into batch tensor: (batch, 9, H, W)
                batch_tensor = torch.from_numpy(np.stack(batch_windows)).float()
                batch_tensor = batch_tensor.to(self.device)

                # Forward pass
                out = self.model(batch_tensor)
                output = out.argmax(dim=1).detach().cpu().numpy()

                all_outputs.append(output)

        # Concatenate all batch outputs
        all_outputs = np.concatenate(all_outputs, axis=0)

        # Postprocess each output to get coordinates
        for i in range(num_windows):
            # Extract single output (add batch dimension for postprocess)
            single_output = all_outputs[i:i+1]
            x_pred, y_pred = postprocess(single_output)

            x, y = self._get_scaled_coordinates(
                x_pred, y_pred, original_width, original_height
            )
            results.append((x, y))

        return results

    def get_carryover_frames(self) -> List[np.ndarray]:
        """
        Get frames needed for cross-batch continuity.

        When processing in batches, the last 2 frames of a batch need to be
        carried over to the next batch to maintain the 3-frame sliding window.

        Returns:
            List of last 2 frames from the buffer (may be empty or have 1-2 frames).
        """
        return list(self.frame_buffer)

    def set_carryover_frames(self, frames: List[np.ndarray]):
        """
        Set carryover frames from previous batch.

        Args:
            frames: List of frames to prepend (typically last 2 from previous batch).
        """
        self.frame_buffer.clear()
        for frame in frames:
            self.frame_buffer.append(frame)
