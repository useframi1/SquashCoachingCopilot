"""Pydantic schemas for API request/response validation."""

from backend.schemas.video import (
    VideoUploadResponse,
    VideoMetadataResponse,
    VideoListResponse,
    VideoDeleteResponse,
)
from backend.schemas.job import (
    JobCreate,
    JobResponse,
    JobStatusResponse,
    JobListResponse,
)
from backend.schemas.analysis import (
    FrameDataResponse,
    FrameDataListResponse,
    RallySummary,
    RallyDetailResponse,
    MatchSummaryResponse,
    ShotAnalysisResponse,
    HeatmapDataResponse,
    PlayerStatsResponse,
)

__all__ = [
    # Video schemas
    "VideoUploadResponse",
    "VideoMetadataResponse",
    "VideoListResponse",
    "VideoDeleteResponse",
    # Job schemas
    "JobCreate",
    "JobResponse",
    "JobStatusResponse",
    "JobListResponse",
    # Analysis schemas
    "FrameDataResponse",
    "FrameDataListResponse",
    "RallySummary",
    "RallyDetailResponse",
    "MatchSummaryResponse",
    "ShotAnalysisResponse",
    "HeatmapDataResponse",
    "PlayerStatsResponse",
]
