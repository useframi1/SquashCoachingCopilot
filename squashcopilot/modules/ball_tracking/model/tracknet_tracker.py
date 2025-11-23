import torch
import torch.nn.functional as F
import cv2
import numpy as np
from collections import deque
from contextlib import nullcontext
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import kornia
import kornia.color
import kornia.morphology
import kornia.enhance
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

        # Use FP16 (half precision) for faster inference on CUDA
        # Benchmarks show 1.66x speedup with 100% accuracy match
        self.use_fp16 = (
            model_config.get("use_fp16", True) and self.device.type == "cuda"
        )
        if self.use_fp16:
            self.model = self.model.half()

        # Model input dimensions
        self.model_width = model_config.get("model_width", 640)
        self.model_height = model_config.get("model_height", 360)

        # Frame buffer (need 3 consecutive frames for inference)
        self.frame_buffer = deque(maxlen=3)

        # Track number of frames processed
        self.frame_count = 0

        # Thread pool for parallel preprocessing (reused across batches)
        self._thread_pool = ThreadPoolExecutor(max_workers=8)

        # CUDA stream for parallel execution (can be assigned by pipeline)
        self._cuda_stream = None

        # Pre-allocated pinned memory buffer for faster CPU→GPU transfer
        # Will be lazily initialized on first batch
        self._pinned_buffer = None
        self._pinned_buffer_size = 0

        # Pre-allocated GPU tensor for input frames
        self._gpu_buffer = None
        self._gpu_buffer_size = 0

        # Pre-computed dilation kernel for black ball preprocessing (GPU)
        # Elliptical 3x3 kernel matching cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Kornia morphology expects 2D kernel (H, W)
        self._dilation_kernel = torch.tensor(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32, device=self.device
        )  # Shape: (3, 3)

    def reset(self):
        """Reset the tracker state (clears frame buffer)."""
        self.frame_buffer.clear()
        self.frame_count = 0

    def _preprocess_black_ball_gpu(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        """GPU-based preprocessing for black ball detection using Kornia.

        Replicates the CPU preprocessing pipeline:
        1. Bitwise NOT (invert colors)
        2. Convert BGR to LAB
        3. Apply CLAHE to L channel
        4. Apply dilation to L channel
        5. Convert LAB back to BGR

        Args:
            frames_tensor: Input tensor of shape (N, 3, H, W) in BGR format, values [0, 1]

        Returns:
            Preprocessed tensor of same shape in BGR format, values [0, 1]
        """
        # 1. Bitwise NOT (invert): 1.0 - tensor
        inverted = 1.0 - frames_tensor

        # 2. Convert BGR to RGB first (Kornia expects RGB)
        rgb = kornia.color.bgr_to_rgb(inverted)

        # 3. Convert RGB to LAB
        lab = kornia.color.rgb_to_lab(rgb)

        # 4. Extract L channel (index 0), normalize to [0, 1] for CLAHE
        # LAB L channel is in range [0, 100]
        l_channel = lab[:, 0:1, :, :] / 100.0

        # 5. Apply CLAHE to L channel
        # kornia.enhance.equalize_clahe expects input in [0, 1]
        l_clahe = kornia.enhance.equalize_clahe(
            l_channel, clip_limit=3.0, grid_size=(8, 8)
        )

        # 6. Apply dilation to L channel
        # Use kornia.morphology.dilation with pre-computed kernel
        l_dilated = kornia.morphology.dilation(l_clahe, self._dilation_kernel)

        # 7. Scale L back to [0, 100] and reconstruct LAB
        lab_enhanced = lab.clone()
        lab_enhanced[:, 0:1, :, :] = l_dilated * 100.0

        # 8. Convert LAB back to RGB
        rgb_enhanced = kornia.color.lab_to_rgb(lab_enhanced)

        # 9. Convert RGB back to BGR
        bgr_enhanced = kornia.color.rgb_to_bgr(rgb_enhanced)

        # Clamp to valid range
        return torch.clamp(bgr_enhanced, 0.0, 1.0)

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

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a single frame to model dimensions (for parallel processing)."""
        return cv2.resize(frame, (self.model_width, self.model_height))

    def _build_window(
        self, args: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """Build a single 3-frame window (for parallel processing).

        Args:
            args: Tuple of (img_curr, img_prev, img_preprev)

        Returns:
            Preprocessed window tensor in CHW format
        """
        img_curr, img_prev, img_preprev = args
        # Concatenate: current, previous, pre-previous (9 channels)
        window = np.concatenate((img_curr, img_prev, img_preprev), axis=2)
        window = window.astype(np.float32) / 255.0
        window = np.rollaxis(window, 2, 0)  # HWC -> CHW
        return window

    def _postprocess_single(
        self, args: Tuple[np.ndarray, int, int]
    ) -> Tuple[Optional[int], Optional[int]]:
        """Postprocess a single output heatmap (for parallel processing).

        Args:
            args: Tuple of (single_output, original_width, original_height)

        Returns:
            Tuple of (x, y) scaled coordinates
        """
        single_output, original_width, original_height = args
        x_pred, y_pred = postprocess(single_output)
        return self._get_scaled_coordinates(
            x_pred, y_pred, original_width, original_height
        )

    def process_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 32,
        is_black_ball: bool = False,
    ) -> List[Tuple[Optional[int], Optional[int]]]:
        """
        Process a batch of frames and return ball coordinates for each.

        Builds 3-frame sliding windows from the input frames and processes
        them through TrackNet in batches for GPU efficiency.

        Optimizations applied:
        - GPU-based frame resizing using Kornia
        - GPU-based window building (concatenation on GPU)
        - GPU-based black ball preprocessing using Kornia (if is_black_ball=True)

        Args:
            frames: List of input frames (BGR format, any resolution).
                    Must be at least 3 frames for any detections.
            batch_size: Number of windows to process in parallel on GPU.
            is_black_ball: If True, apply GPU-based preprocessing for black ball detection.

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

        # First 2 frames have no temporal context - return None
        results.append((None, None))
        if num_frames >= 2:
            results.append((None, None))

        if num_frames < 3:
            return results

        num_windows = num_frames - 2

        # Use assigned CUDA stream if available, otherwise use default stream
        stream_context = (
            torch.cuda.stream(self._cuda_stream)
            if self._cuda_stream is not None
            else (
                torch.cuda.stream(torch.cuda.default_stream(self.device))
                if self.device.type == "cuda"
                else nullcontext()
            )
        )

        with torch.no_grad(), stream_context:
            # Stack all frames into a single tensor
            # Convert BGR (numpy HWC) to NCHW format
            frames_np = np.stack(frames, axis=0)  # (N, H, W, 3)
            n, h, w, c = frames_np.shape
            required_size = n * h * w * c

            # Use pinned memory for faster CPU→GPU transfer (lazily allocate)
            if self.device.type == "cuda":
                # Allocate or resize pinned buffer if needed
                if self._pinned_buffer is None or self._pinned_buffer_size < required_size:
                    self._pinned_buffer = torch.empty(
                        (n, c, h, w), dtype=torch.float32, pin_memory=True
                    )
                    self._pinned_buffer_size = required_size

                # Copy to pinned memory with correct shape
                pinned_view = self._pinned_buffer[:n, :, :h, :w]
                # Transpose from NHWC to NCHW during copy
                np.copyto(
                    pinned_view.numpy(),
                    frames_np.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
                )

                # Async copy to GPU using non_blocking=True
                frames_tensor = pinned_view.to(self.device, non_blocking=True)
            else:
                # CPU fallback
                frames_tensor = (
                    torch.from_numpy(frames_np).float().permute(0, 3, 1, 2) / 255.0
                )

            # Convert to FP16 if enabled (before resize for memory efficiency)
            if self.use_fp16:
                frames_tensor = frames_tensor.half()

            # GPU-based resize using Kornia
            resized_tensor = kornia.geometry.transform.resize(
                frames_tensor,
                (self.model_height, self.model_width),
                interpolation="bilinear",
                antialias=True,
            )  # (N, 3, model_height, model_width)

            # Apply black ball preprocessing on GPU if needed
            if is_black_ball:
                # Need FP32 for color conversion operations
                if self.use_fp16:
                    resized_tensor = resized_tensor.float()
                resized_tensor = self._preprocess_black_ball_gpu(resized_tensor)
                if self.use_fp16:
                    resized_tensor = resized_tensor.half()

            # Build sliding windows on GPU
            # Window i needs frames [i, i+1, i+2] concatenated as [curr, prev, preprev]
            # Output for window i corresponds to frame i+2

            # Process all windows in batches
            all_outputs = []

            for batch_start in range(0, num_windows, batch_size):
                batch_end = min(batch_start + batch_size, num_windows)

                # Build windows for this batch on GPU
                # Each window: concatenate frame[i+2], frame[i+1], frame[i] along channel dim
                window_indices = list(range(batch_start, batch_end))

                # Gather frames for each position in the window
                curr_frames = resized_tensor[
                    torch.tensor([i + 2 for i in window_indices], device=self.device)
                ]
                prev_frames = resized_tensor[
                    torch.tensor([i + 1 for i in window_indices], device=self.device)
                ]
                preprev_frames = resized_tensor[
                    torch.tensor([i for i in window_indices], device=self.device)
                ]

                # Concatenate along channel dimension: (batch, 9, H, W)
                batch_windows = torch.cat(
                    [curr_frames, prev_frames, preprev_frames], dim=1
                )

                # Forward pass through model
                out = self.model(batch_windows)
                output = out.argmax(dim=1)  # (batch, H, W)

                all_outputs.append(output)

            # Concatenate all outputs: (num_windows, H, W)
            all_outputs = torch.cat(all_outputs, dim=0)

            # Move outputs to CPU for postprocessing with HoughCircles
            # (HoughCircles is CPU-only but more accurate for ball detection)
            all_outputs_np = all_outputs.cpu().numpy()

        # Parallel postprocessing using HoughCircles (CPU)
        # This is more accurate than GPU weighted centroid for ball detection
        postprocess_args = [
            (all_outputs_np[i : i + 1], original_width, original_height)
            for i in range(num_windows)
        ]

        postprocess_results = list(
            self._thread_pool.map(self._postprocess_single, postprocess_args)
        )
        results.extend(postprocess_results)

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
