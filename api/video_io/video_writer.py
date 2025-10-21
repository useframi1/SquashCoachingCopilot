"""Video writer for handling video output."""

import cv2
import numpy as np
import os
import subprocess
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
        self.use_ffmpeg_conversion = codec in ["mp4v", "MP4V"]

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # If using mp4v, write to temporary AVI file first
        if self.use_ffmpeg_conversion:
            self.temp_path = output_path.replace('.mp4', '_temp.avi')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Motion JPEG is reliable
            self.writer = cv2.VideoWriter(
                self.temp_path, fourcc, metadata.fps, (metadata.width, metadata.height)
            )
        else:
            self.temp_path = None
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.writer = cv2.VideoWriter(
                output_path, fourcc, metadata.fps, (metadata.width, metadata.height)
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
        """Release video writer resources and convert to H.264 if needed."""
        if self.writer:
            self.writer.release()

            # If we used temporary AVI, convert to H.264 MP4 using FFmpeg
            if self.use_ffmpeg_conversion and self.temp_path:
                try:
                    print(f"Converting {self.temp_path} to H.264 MP4...")
                    subprocess.run([
                        'ffmpeg',
                        '-i', self.temp_path,
                        '-c:v', 'libx264',  # H.264 codec
                        '-preset', 'medium',  # Balance between speed and compression
                        '-crf', '23',  # Quality (lower = better, 23 is default)
                        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                        '-movflags', '+faststart',  # Enable fast start for web streaming
                        '-y',  # Overwrite output file
                        self.output_path
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    # Remove temporary AVI file
                    os.remove(self.temp_path)
                    print(f"Successfully converted to {self.output_path}")
                except subprocess.CalledProcessError as e:
                    print(f"FFmpeg conversion failed: {e.stderr.decode()}")
                    raise ValueError(f"Failed to convert video to H.264: {e}")
                except Exception as e:
                    print(f"Error during conversion: {e}")
                    raise

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
