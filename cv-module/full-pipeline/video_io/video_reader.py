"""Video reader for handling video I/O and metadata extraction."""

import cv2
import numpy as np
from typing import Optional, Generator, Tuple
from dataclasses import dataclass


@dataclass
class VideoMetadata:
    """Container for video metadata."""

    fps: float
    width: int
    height: int
    total_frames: int
    duration: float  # in seconds

    def __str__(self):
        return f"Video: {self.width}x{self.height}, {self.fps} fps, {self.total_frames} frames, {self.duration:.2f}s"


class VideoReader:
    """
    Handles video reading and metadata extraction.

    Responsibilities:
    - Open and read video files
    - Extract video metadata (fps, resolution, duration)
    - Provide frame iterator for processing
    - Does NOT: Process frames or handle output writing
    """

    def __init__(self, video_path: str):
        """
        Initialize video reader.

        Args:
            video_path: Path to input video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Extract metadata
        self.metadata = self._extract_metadata()

    def _extract_metadata(self) -> VideoMetadata:
        """Extract video metadata."""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        return VideoMetadata(
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            duration=duration,
        )

    def get_metadata(self) -> VideoMetadata:
        """
        Get video metadata.

        Returns:
            VideoMetadata object with video properties
        """
        return self.metadata

    def frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Generate frames from the video.

        Yields:
            Tuple of (frame_number, timestamp, frame)
        """
        frame_count = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = frame_count / self.metadata.fps
            yield frame_count, timestamp, frame
            frame_count += 1

    def release(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
