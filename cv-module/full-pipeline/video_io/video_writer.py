"""Video writer for handling video output."""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Optional
from .video_reader import VideoMetadata


class VideoWriter:
    """
    Handles video writing.

    Responsibilities:
    - Write frames to output video file
    - Manage video writer lifecycle
    - Does NOT: Process or modify frames
    """

    def __init__(self, output_path: str, metadata: VideoMetadata, codec: str = "avc1"):
        """
        Initialize video writer.

        Args:
            output_path: Path to save output video
            metadata: Video metadata (fps, width, height)
            codec: Video codec (default: 'avc1' for H.264)
        """
        self.output_path = output_path
        self.metadata = metadata

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            metadata.fps,
            (metadata.width, metadata.height)
        )

        if not self.writer.isOpened():
            raise ValueError(f"Failed to create video writer: {output_path}")

    def write(self, frame: np.ndarray):
        """
        Write a frame to the output video.

        Args:
            frame: Frame to write
        """
        if self.writer:
            self.writer.write(frame)

    def release(self):
        """Release video writer resources."""
        if self.writer:
            self.writer.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
