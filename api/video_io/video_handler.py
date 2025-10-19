"""Video handler for managing video I/O operations."""

import cv2
import numpy as np
from typing import Generator, Tuple, Optional
from .video_reader import VideoReader, VideoMetadata
from .video_writer import VideoWriter


class VideoHandler:
    """
    Handles video reading and writing operations.

    Responsibilities:
    - Provide generator for reading video frames
    - Write processed frames to output video
    - Manage video metadata
    - Handle errors related to video I/O
    """

    def __init__(self, input_path: str, base_output_path: str, codec: str = "avc1"):
        """
        Initialize video handler.

        Args:
            input_path: Path to input video file
            output_path: Path to output video file
            codec: Video codec for output (default: 'avc1' for H.264)

        Raises:
            FileNotFoundError: If input video file doesn't exist
            ValueError: If video cannot be opened
        """
        self.input_path = input_path
        self.base_output_path = base_output_path
        self.codec = codec
        self.metadata: Optional[VideoMetadata] = None

        # Validate input video exists and can be opened
        self._validate_input()

    def _validate_input(self):
        """
        Validate that input video exists and can be opened.

        Raises:
            FileNotFoundError: If input video doesn't exist
            ValueError: If video cannot be opened
        """
        import os

        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input video not found: {self.input_path}")

        # Try opening to validate
        test_cap = cv2.VideoCapture(self.input_path)
        if not test_cap.isOpened():
            raise ValueError(f"Cannot open input video: {self.input_path}")

        # Extract and store metadata
        reader = VideoReader(self.input_path)
        self.metadata = reader.get_metadata()
        reader.release()

    def read_video(
        self, start_frame: int = None, end_frame: int = None
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Read video frames as a generator.

        Yields:
            Tuple of (frame_number, timestamp, frame)

        Raises:
            ValueError: If video cannot be opened
        """
        try:
            with VideoReader(self.input_path) as reader:
                for frame_number, timestamp, frame in reader.frames(
                    start_frame, end_frame
                ):
                    yield frame_number, timestamp, frame
        except Exception as e:
            raise ValueError(f"Error reading video {self.input_path}: {str(e)}")

    def get_metadata(self) -> VideoMetadata:
        """
        Get video metadata.

        Returns:
            VideoMetadata object with video properties

        Raises:
            ValueError: If metadata hasn't been extracted yet
        """
        if self.metadata is None:
            raise ValueError("Metadata not available. Video hasn't been validated.")
        return self.metadata

    def write_video(
        self,
        frames: Generator[np.ndarray, None, None],
        output_path: Optional[str] = None,
    ):
        """
        Write frames to output video file.

        Args:
            frames: Generator yielding frames to write

        Raises:
            ValueError: If video writer cannot be created
            IOError: If there are disk space or permission issues
        """
        if self.metadata is None:
            raise ValueError("Cannot write video: metadata not available")

        output_path = self.base_output_path + "/" + output_path + ".mp4"

        try:
            with VideoWriter(output_path, self.metadata, self.codec) as writer:
                frame_count = 0
                for frame in frames:
                    writer.write(frame)
                    frame_count += 1

                print(f"Successfully wrote {frame_count} frames to {output_path}")

        except PermissionError as e:
            raise IOError(f"Permission denied writing to {output_path}: {str(e)}")
        except OSError as e:
            raise IOError(f"Error writing video (disk space?): {str(e)}")
        except Exception as e:
            raise ValueError(f"Error writing video {output_path}: {str(e)}")
