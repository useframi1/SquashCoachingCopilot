"""Video I/O module for reading and writing videos."""

from .video_reader import VideoReader, VideoMetadata
from .video_handler import VideoHandler

__all__ = ["VideoReader", "VideoMetadata", "VideoHandler"]
