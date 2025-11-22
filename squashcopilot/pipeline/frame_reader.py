"""
Batch frame reader for efficient video processing.

Provides a generator-based frame reader that yields batches of frames
for optimized GPU inference in the pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
from dataclasses import dataclass
import threading
import queue


@dataclass
class FrameBatch:
    """Container for a batch of frames."""

    frame_numbers: List[int]
    timestamps: List[float]
    images: List[np.ndarray]

    def __len__(self) -> int:
        return len(self.frame_numbers)


class BatchFrameReader:
    """
    Efficient batch frame reader with optional prefetching.

    Reads video frames in configurable batch sizes and optionally
    prefetches batches in a background thread for I/O optimization.

    Usage:
        reader = BatchFrameReader("video.mp4", batch_size=32)
        for batch in reader:
            process(batch.frame_numbers, batch.timestamps, batch.images)
    """

    def __init__(
        self,
        video_path: str,
        batch_size: int = 32,
        max_frames: Optional[int] = None,
        fps: Optional[float] = None,
        prefetch: bool = True,
        prefetch_batches: int = 2,
    ):
        """
        Initialize the batch frame reader.

        Args:
            video_path: Path to the video file.
            batch_size: Number of frames per batch.
            max_frames: Maximum number of frames to read (None = entire video).
            fps: Video FPS for timestamp calculation (None = read from video).
            prefetch: Whether to prefetch batches in background thread.
            prefetch_batches: Number of batches to prefetch (if prefetch=True).
        """
        self.video_path = str(video_path)
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.prefetch_batches = prefetch_batches

        # Open video to get metadata
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        self.fps = fps if fps is not None else cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Apply frame limit
        if max_frames is not None:
            self.total_frames = min(max_frames, total_frames)
        else:
            self.total_frames = total_frames

        # Calculate number of batches
        self._num_batches = (self.total_frames + batch_size - 1) // batch_size

    def __len__(self) -> int:
        """Return the total number of batches."""
        return self._num_batches

    def __iter__(self) -> Iterator[FrameBatch]:
        """Iterate over frame batches."""
        if self.prefetch:
            yield from self._iter_with_prefetch()
        else:
            yield from self._iter_sequential()

    def _iter_sequential(self) -> Iterator[FrameBatch]:
        """Sequential frame reading without prefetching."""
        cap = cv2.VideoCapture(self.video_path)

        try:
            frame_number = 0
            while frame_number < self.total_frames:
                batch_frame_numbers = []
                batch_timestamps = []
                batch_images = []

                # Read up to batch_size frames
                for _ in range(self.batch_size):
                    if frame_number >= self.total_frames:
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break

                    batch_frame_numbers.append(frame_number)
                    batch_timestamps.append(frame_number / self.fps)
                    batch_images.append(frame)
                    frame_number += 1

                if batch_images:
                    yield FrameBatch(
                        frame_numbers=batch_frame_numbers,
                        timestamps=batch_timestamps,
                        images=batch_images,
                    )
        finally:
            cap.release()

    def _iter_with_prefetch(self) -> Iterator[FrameBatch]:
        """Frame reading with background prefetching."""
        batch_queue: queue.Queue[Optional[FrameBatch]] = queue.Queue(
            maxsize=self.prefetch_batches
        )
        stop_event = threading.Event()

        def producer():
            """Background thread that reads frames into queue."""
            cap = cv2.VideoCapture(self.video_path)
            try:
                frame_number = 0
                while frame_number < self.total_frames and not stop_event.is_set():
                    batch_frame_numbers = []
                    batch_timestamps = []
                    batch_images = []

                    for _ in range(self.batch_size):
                        if frame_number >= self.total_frames:
                            break

                        ret, frame = cap.read()
                        if not ret:
                            break

                        batch_frame_numbers.append(frame_number)
                        batch_timestamps.append(frame_number / self.fps)
                        batch_images.append(frame)
                        frame_number += 1

                    if batch_images and not stop_event.is_set():
                        batch = FrameBatch(
                            frame_numbers=batch_frame_numbers,
                            timestamps=batch_timestamps,
                            images=batch_images,
                        )
                        batch_queue.put(batch)

                # Signal end of stream
                batch_queue.put(None)
            finally:
                cap.release()

        # Start producer thread
        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        try:
            while True:
                batch = batch_queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            stop_event.set()
            producer_thread.join(timeout=1.0)

    def read_all(self) -> FrameBatch:
        """
        Read all frames at once into a single batch.

        Warning: This loads all frames into memory. Use with caution for large videos.

        Returns:
            FrameBatch containing all frames.
        """
        all_frame_numbers = []
        all_timestamps = []
        all_images = []

        for batch in self._iter_sequential():
            all_frame_numbers.extend(batch.frame_numbers)
            all_timestamps.extend(batch.timestamps)
            all_images.extend(batch.images)

        return FrameBatch(
            frame_numbers=all_frame_numbers,
            timestamps=all_timestamps,
            images=all_images,
        )
