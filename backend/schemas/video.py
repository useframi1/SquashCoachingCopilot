"""Pydantic schemas for video-related API operations."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class VideoMetadataResponse(BaseModel):
    """Video metadata response."""

    id: str
    filename: str
    original_filename: str
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None
    has_annotated_video: bool = False
    uploaded_at: datetime
    processed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class VideoUploadResponse(BaseModel):
    """Response after successful video upload."""

    id: str
    filename: str
    message: str = "Video uploaded successfully"


class VideoListResponse(BaseModel):
    """List of videos response."""

    videos: list[VideoMetadataResponse]
    total: int
    page: int = 1
    page_size: int = 20


class VideoDeleteResponse(BaseModel):
    """Response after video deletion."""

    id: str
    message: str = "Video deleted successfully"
