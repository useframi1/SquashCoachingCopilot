"""Video I/O module for reading and writing videos."""

from .video_reader import VideoReader, VideoMetadata
from .video_writer import VideoWriter

__all__ = ["VideoReader", "VideoMetadata", "VideoWriter"]
